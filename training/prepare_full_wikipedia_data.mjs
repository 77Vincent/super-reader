#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { createReadStream } from "node:fs";
import {
  mkdir,
  open,
  readFile,
  rename,
  stat,
  truncate,
  writeFile,
} from "node:fs/promises";
import { createInterface } from "node:readline";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  buildAdjacentSamples,
  wikipediaDocumentsFromXml,
} from "./prepare_smoke_data.mjs";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = dirname(SCRIPT_DIR);
const DEFAULT_SOURCE_DIR = join(SCRIPT_DIR, "data", "processed");
const DEFAULT_OUTPUT_DIR = join(SCRIPT_DIR, "data", "processed", "wiki-full-sharded-128");
const RAW_DIR = join(SCRIPT_DIR, "data", "raw");
const DUMP_PATH = join(RAW_DIR, "zhwiki-latest-pages-articles-multistream.xml.bz2");
const INDEX_PATH = join(RAW_DIR, "zhwiki-latest-pages-articles-multistream-index.txt.bz2");
const DOMAINS = ["news", "academic", "encyclopedia", "dialogue", "wikipedia"];
const LENGTH_BUCKET_MAXIMUMS = [8, 16, 32];
const BLOOM_BYTES = 256 * 1024 * 1024;
const BLOOM_BIT_MASK = 0x7fffffff;
const BLOOM_HASHES = 7;
const BUFFER_CHARACTERS = 1024 * 1024;

function parseArguments(argv) {
  const options = {
    sourceDir: DEFAULT_SOURCE_DIR,
    outputDir: DEFAULT_OUTPUT_DIR,
    shards: 128,
    checkpointBlocks: 100,
    maxBlocks: 0,
    maxSequenceLength: 2048,
    maxSamplesPerDocument: 128,
    positionBins: 10,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key.startsWith("--") || value === undefined) continue;
    const name = key.slice(2).replace(/-([a-z])/gu, (_, letter) => letter.toUpperCase());
    if (!Object.hasOwn(options, name)) continue;
    options[name] = ["sourceDir", "outputDir"].includes(name) ? resolve(value) : Number(value);
    index += 1;
  }
  if (!Number.isInteger(options.shards) || options.shards < 1) {
    throw new Error("--shards must be a positive integer");
  }
  if ((options.shards & (options.shards - 1)) !== 0) {
    throw new Error("--shards must be a power of two");
  }
  for (const name of [
    "checkpointBlocks",
    "maxBlocks",
    "maxSequenceLength",
    "maxSamplesPerDocument",
    "positionBins",
  ]) {
    if (!Number.isInteger(options[name]) || options[name] < (name === "maxBlocks" ? 0 : 1)) {
      throw new Error(`--${name} must be a valid integer`);
    }
  }
  return options;
}

async function forEachJsonLine(path, callback) {
  const input = createReadStream(path, { encoding: "utf8" });
  const lines = createInterface({ input, crlfDelay: Infinity });
  for await (const line of lines) {
    if (line) await callback(JSON.parse(line));
  }
}

function updateHashes(hashes, value) {
  let [first, second] = hashes;
  for (const character of value) {
    const code = character.codePointAt(0);
    first = Math.imul(first ^ code, 16777619) >>> 0;
    second = Math.imul(second ^ code, 2246822519) >>> 0;
  }
  return [first, second];
}

function boundaryHashes(tokens, targetIndex) {
  let hashes = [2166136261, 0x9e3779b9];
  for (let index = 0; index < tokens.length; index += 1) {
    hashes = updateHashes(hashes, tokens[index]);
    if (index === targetIndex) {
      hashes[0] = Math.imul(hashes[0] ^ 0xffffffff, 16777619) >>> 0;
      hashes[1] = Math.imul(hashes[1] ^ 0xffffffff, 2246822519) >>> 0;
    }
  }
  hashes[1] = (hashes[1] | 1) >>> 0;
  return hashes;
}

class BloomFilter {
  constructor() {
    this.bits = Buffer.alloc(BLOOM_BYTES);
  }

  add(hashes) {
    const [first, second] = hashes;
    let seen = true;
    for (let index = 0; index < BLOOM_HASHES; index += 1) {
      const bit = ((first + Math.imul(index, second)) >>> 0) & BLOOM_BIT_MASK;
      const byteIndex = bit >>> 3;
      const mask = 1 << (bit & 7);
      if ((this.bits[byteIndex] & mask) === 0) {
        seen = false;
        this.bits[byteIndex] |= mask;
      }
    }
    return seen;
  }
}

function lengthBucketIndex(length) {
  const index = LENGTH_BUCKET_MAXIMUMS.findIndex((maximum) => length <= maximum);
  return index < 0 ? LENGTH_BUCKET_MAXIMUMS.length : index;
}

function emptyStatistics(positionBins) {
  return {
    samples: 0,
    tokens: 0,
    random_baseline_sum: 0,
    center_correct: 0,
    duplicate_or_bloom_filtered: 0,
    overlength_filtered: 0,
    documents_seen: 0,
    documents_with_samples: 0,
    holdout_documents_filtered: 0,
    document_sample_cap_filtered: 0,
    domain_samples: Object.fromEntries(DOMAINS.map((domain) => [domain, 0])),
    position_histogram: Array(positionBins).fill(0),
    cells: Array.from(
      { length: LENGTH_BUCKET_MAXIMUMS.length + 1 },
      () => Array(positionBins).fill(0),
    ),
    maximum_sequence_length: 0,
  };
}

function readOffsets() {
  const text = execFileSync("bzip2", ["-dc", INDEX_PATH], {
    encoding: "utf8",
    maxBuffer: 512 * 1024 * 1024,
  });
  const offsets = new Set();
  for (const line of text.split(/\r?\n/u)) {
    const separator = line.indexOf(":");
    if (separator < 1) continue;
    const offset = Number(line.slice(0, separator));
    if (Number.isSafeInteger(offset) && offset >= 0) offsets.add(offset);
  }
  return Array.from(offsets).sort((left, right) => left - right);
}

function shardPartPath(outputDir, index) {
  return join(outputDir, `train-${String(index).padStart(3, "0")}.jsonl.part`);
}

class ShardWriter {
  constructor(outputDir, shardCount, initialSizes) {
    this.outputDir = outputDir;
    this.shardCount = shardCount;
    this.sizes = initialSizes || Array(shardCount).fill(0);
    this.buffers = Array(shardCount).fill("");
    this.handles = [];
  }

  async open() {
    for (let index = 0; index < this.shardCount; index += 1) {
      const path = shardPartPath(this.outputDir, index);
      await truncate(path, this.sizes[index]).catch(async (error) => {
        if (error.code !== "ENOENT") throw error;
        const handle = await open(path, "w");
        await handle.close();
      });
      this.handles.push(await open(path, "a"));
    }
  }

  async add(index, record) {
    this.buffers[index] += `${JSON.stringify(record)}\n`;
    if (this.buffers[index].length >= BUFFER_CHARACTERS) await this.flush(index);
  }

  async flush(index) {
    const buffer = this.buffers[index];
    if (!buffer) return;
    const { bytesWritten } = await this.handles[index].write(buffer);
    this.sizes[index] += bytesWritten;
    this.buffers[index] = "";
  }

  async flushAll() {
    for (let index = 0; index < this.shardCount; index += 1) await this.flush(index);
  }

  async close() {
    await this.flushAll();
    for (const handle of this.handles) await handle.close();
  }
}

async function writeJsonAtomic(path, value) {
  const temporaryPath = `${path}.part`;
  await writeFile(temporaryPath, `${JSON.stringify(value, null, 2)}\n`);
  await rename(temporaryPath, path);
}

function sourceIdentity(sourceDir, dump, index) {
  return {
    source_summary: join(sourceDir, "summary.json"),
    dump_bytes: dump.size,
    dump_mtime_ms: dump.mtimeMs,
    index_bytes: index.size,
    index_mtime_ms: index.mtimeMs,
  };
}

function compatibleState(state, options, identity) {
  return state
    && state.format_version === 1
    && state.options.shards === options.shards
    && state.options.max_sequence_length === options.maxSequenceLength
    && state.options.max_samples_per_document === options.maxSamplesPerDocument
    && state.options.position_bins === options.positionBins
    && JSON.stringify(state.source_identity) === JSON.stringify(identity);
}

async function addSample(sample, domainIndex, bloom, writer, statistics, options) {
  const hashes = boundaryHashes(sample.tokens, sample.target_index);
  if (bloom.add(hashes)) {
    statistics.duplicate_or_bloom_filtered += 1;
    return;
  }
  const length = sample.tokens.length;
  if (length > options.maxSequenceLength) {
    statistics.overlength_filtered += 1;
    return;
  }
  const bucket = lengthBucketIndex(length);
  const position = Math.min(
    options.positionBins - 1,
    Math.floor(((sample.target_index + 1) / length) * options.positionBins),
  );
  const shard = hashes[0] & (options.shards - 1);
  const text = sample.tokens.join("");
  await writer.add(shard, [text, sample.target_index, domainIndex, bucket, position]);
  statistics.samples += 1;
  statistics.tokens += length;
  statistics.random_baseline_sum += 1 / (length - 1);
  statistics.center_correct += sample.target_index === Math.floor((length - 1) / 2);
  statistics.domain_samples[DOMAINS[domainIndex]] += 1;
  statistics.position_histogram[position] += 1;
  statistics.cells[bucket][position] += 1;
  statistics.maximum_sequence_length = Math.max(statistics.maximum_sequence_length, length);
}

async function seedEvaluationBloom(sourceDir, bloom, holdoutDocuments) {
  let samples = 0;
  for (const split of ["test", "validation"]) {
    await forEachJsonLine(join(sourceDir, `${split}.jsonl`), (record) => {
      bloom.add(boundaryHashes(record.tokens, record.target_index));
      holdoutDocuments.add(record.document_id);
      samples += 1;
    });
  }
  return samples;
}

async function rebuildTrainingBloom(outputDir, shardCount, bloom) {
  let samples = 0;
  for (let index = 0; index < shardCount; index += 1) {
    await forEachJsonLine(shardPartPath(outputDir, index), (record) => {
      const tokens = Array.from(record[0]);
      bloom.add(boundaryHashes(tokens, record[1]));
      samples += 1;
    });
  }
  return samples;
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  await mkdir(options.outputDir, { recursive: true });
  const manifestPath = join(options.outputDir, "manifest.json");
  try {
    const existing = JSON.parse(await readFile(manifestPath, "utf8"));
    console.log(JSON.stringify(existing, null, 2));
    return;
  } catch {
    // Continue a partial build or start a new one.
  }

  const [dump, index] = await Promise.all([stat(DUMP_PATH), stat(INDEX_PATH)]);
  const identity = sourceIdentity(options.sourceDir, dump, index);
  const statePath = join(options.outputDir, "preparation-state.json");
  let state = null;
  try {
    state = JSON.parse(await readFile(statePath, "utf8"));
  } catch {
    // Start below.
  }
  if (state && !compatibleState(state, options, identity)) {
    throw new Error(`Incompatible preparation state already exists: ${statePath}`);
  }
  if (!state) {
    state = {
      format_version: 1,
      options: {
        shards: options.shards,
        max_sequence_length: options.maxSequenceLength,
        max_samples_per_document: options.maxSamplesPerDocument,
        position_bins: options.positionBins,
      },
      source_identity: identity,
      base_complete: false,
      next_block: 0,
      shard_sizes: Array(options.shards).fill(0),
      statistics: emptyStatistics(options.positionBins),
    };
  }

  const bloom = new BloomFilter();
  const holdoutDocuments = new Set();
  const evaluationSamples = await seedEvaluationBloom(
    options.sourceDir,
    bloom,
    holdoutDocuments,
  );
  const writer = new ShardWriter(options.outputDir, options.shards, state.shard_sizes);
  await writer.open();

  if (state.base_complete) {
    const restored = await rebuildTrainingBloom(options.outputDir, options.shards, bloom);
    if (restored !== state.statistics.samples) {
      throw new Error(`Expected ${state.statistics.samples} restored samples; found ${restored}`);
    }
  } else {
    await forEachJsonLine(join(options.sourceDir, "train.jsonl"), async (sample) => {
      if (sample.domain === "wikipedia") return;
      await addSample(
        sample,
        DOMAINS.indexOf(sample.domain),
        bloom,
        writer,
        state.statistics,
        options,
      );
    });
    await writer.flushAll();
    state.base_complete = true;
    state.shard_sizes = [...writer.sizes];
    await writeJsonAtomic(statePath, state);
  }

  const offsets = readOffsets();
  const lastBlock = options.maxBlocks > 0
    ? Math.min(offsets.length, options.maxBlocks)
    : offsets.length;
  const dumpHandle = await open(DUMP_PATH, "r");

  try {
    for (let blockIndex = state.next_block; blockIndex < lastBlock; blockIndex += 1) {
      const offset = offsets[blockIndex];
      const nextOffset = offsets[blockIndex + 1] ?? dump.size;
      const length = nextOffset - offset;
      const compressed = Buffer.allocUnsafe(length);
      const { bytesRead } = await dumpHandle.read(compressed, 0, length, offset);
      if (bytesRead !== length) throw new Error(`Short dump read at byte ${offset}`);
      const xml = execFileSync("bzip2", ["-dc"], {
        input: compressed,
        encoding: "utf8",
        maxBuffer: 256 * 1024 * 1024,
      });
      const documents = wikipediaDocumentsFromXml(xml, offset);
      for (const document of documents) {
        state.statistics.documents_seen += 1;
        if (holdoutDocuments.has(document.id)) {
          state.statistics.holdout_documents_filtered += 1;
          continue;
        }
        const candidates = buildAdjacentSamples(document, { tokenization: "character" });
        const samples = candidates.length <= options.maxSamplesPerDocument
          ? candidates
          : Array.from({ length: options.maxSamplesPerDocument }, (_, index) => (
            candidates[Math.floor(((index + 0.5) * candidates.length) / options.maxSamplesPerDocument)]
          ));
        state.statistics.document_sample_cap_filtered += candidates.length - samples.length;
        if (samples.length > 0) state.statistics.documents_with_samples += 1;
        for (const sample of samples) {
          await addSample(sample, 4, bloom, writer, state.statistics, options);
        }
      }

      state.next_block = blockIndex + 1;
      if (state.next_block % options.checkpointBlocks === 0 || state.next_block === lastBlock) {
        await writer.flushAll();
        state.shard_sizes = [...writer.sizes];
        await writeJsonAtomic(statePath, state);
        const rssGb = process.memoryUsage().rss / 1024 ** 3;
        console.log(
          `blocks=${state.next_block}/${lastBlock} samples=${state.statistics.samples} `
          + `rss_gb=${rssGb.toFixed(2)}`,
        );
      }
    }
  } finally {
    await dumpHandle.close();
    await writer.close();
  }

  if (state.next_block !== lastBlock || lastBlock !== offsets.length) {
    console.log(`Partial preparation stopped at block ${state.next_block}/${offsets.length}`);
    return;
  }

  const shards = [];
  for (let index = 0; index < options.shards; index += 1) {
    const partPath = shardPartPath(options.outputDir, index);
    const finalPath = partPath.slice(0, -".part".length);
    await rename(partPath, finalPath);
    shards.push({
      path: relative(PROJECT_DIR, finalPath),
      bytes: state.shard_sizes[index],
    });
  }
  const manifest = {
    format: "super-reader-sharded-training-v1",
    source: "full downloaded Chinese Wikipedia dump plus non-Wikipedia CLUE training data",
    source_identity: identity,
    evaluation_source_dir: relative(PROJECT_DIR, options.sourceDir),
    vocabulary_path: "training/artifacts/data-scale-wiki-250k-8conv-64ch-6ep/boundary-smoke-vocabulary.json",
    bloom_filter: {
      bytes: BLOOM_BYTES,
      hashes: BLOOM_HASHES,
      false_negatives: false,
      expected_false_positive_rate_at_50000000_items: 0.00000284,
    },
    length_bucket_maximums: LENGTH_BUCKET_MAXIMUMS,
    position_bins: options.positionBins,
    maximum_sequence_length: options.maxSequenceLength,
    evaluation_samples_seeded: evaluationSamples,
    blocks: offsets.length,
    shards,
    statistics: state.statistics,
  };
  await writeJsonAtomic(manifestPath, manifest);
  console.log(JSON.stringify(manifest, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
