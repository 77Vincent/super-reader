import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { createReadStream, createWriteStream } from "node:fs";
import {
  mkdir,
  open,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath, pathToFileURL } from "node:url";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const RAW_DIR = join(SCRIPT_DIR, "data", "raw");
const DEFAULT_PROCESSED_DIR = join(SCRIPT_DIR, "data", "processed");
const SEGMENTER = new Intl.Segmenter("zh-CN", { granularity: "word" });
const HAN_CHARACTER = /\p{Script=Han}/u;
const PROXY_PUNCTUATION = new Set([
  "，", ",", "。", ".", "！", "!", "？", "?", "；", ";", "：", ":", "、", "…",
]);
const LENGTH_BUCKET_MAXIMUMS = [8, 16, 32];

const SOURCES = [
  {
    domain: "news",
    filename: "tnews_public.zip",
    url: "https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip",
    sha256: "77c476e70cfe0b014a81b84c6e1db2142a8a2f52f4ae0a8216aa75e673933462",
    entries: ["train.json"],
  },
  {
    domain: "academic",
    filename: "csl_public.zip",
    url: "https://storage.googleapis.com/cluebenchmark/tasks/csl_public.zip",
    sha256: "795d1a2e475d59acad8236f6c5baba7a0b43d3e0508cb60f15ffbc76d5f437c4",
    entries: ["train.json"],
  },
  {
    domain: "encyclopedia",
    filename: "cmrc2018_public.zip",
    url: "https://storage.googleapis.com/cluebenchmark/tasks/cmrc2018_public.zip",
    sha256: "6c63dc27e728ec5231aeb7d2861b4c90b6c116390582e0c44416cf3edf030b16",
    entries: ["train.json"],
  },
  {
    domain: "dialogue",
    filename: "c3_public.zip",
    url: "https://storage.googleapis.com/cluebenchmark/tasks/c3_public.zip",
    sha256: "20dfa683c57c9795129e22d8dd0c299d6b2a29fbc84758ef2ebdb8b2904d7e12",
    entries: ["d-train.json", "m-train.json"],
  },
  {
    domain: "wikipedia",
    filename: "zhwiki-latest-pages-articles-multistream.xml.bz2",
    url: "https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles-multistream.xml.bz2",
    checksumUrl: "https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-md5sums.txt",
    checksumSuffix: "pages-articles-multistream.xml.bz2",
    index: {
      filename: "zhwiki-latest-pages-articles-multistream-index.txt.bz2",
      url: "https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles-multistream-index.txt.bz2",
      checksumUrl: "https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-md5sums.txt",
      checksumSuffix: "pages-articles-multistream-index.txt.bz2",
    },
  },
];

const CHECKSUM_MANIFESTS = new Map();

function hashText(text) {
  return createHash("sha256").update(text).digest("hex");
}

function normalizeDocument(text) {
  return String(text || "")
    .normalize("NFKC")
    .replace(/\s+/gu, " ")
    .trim();
}

function cleanFragment(text) {
  return normalizeDocument(text)
    .replace(/[\p{P}\p{S}]+/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
}

function tokenize(text, tokenization) {
  if (tokenization === "character") {
    return Array.from(text).filter((character) => HAN_CHARACTER.test(character));
  }

  return Array.from(SEGMENTER.segment(text))
    .filter((item) => item.isWordLike && HAN_CHARACTER.test(item.segment))
    .map((item) => Array.from(item.segment)
      .filter((character) => HAN_CHARACTER.test(character))
      .join(""))
    .filter(Boolean);
}

function contentCharacterLength(text) {
  return Array.from(text)
    .filter((character) => HAN_CHARACTER.test(character)).length;
}

export function splitIntoFragments(text) {
  const fragments = [];
  let buffer = "";

  for (const character of Array.from(normalizeDocument(text))) {
    if (!PROXY_PUNCTUATION.has(character)) {
      buffer += character;
      continue;
    }

    const cleaned = cleanFragment(buffer);
    if (cleaned) {
      fragments.push({ text: cleaned, punctuation: character });
    } else if (fragments.length > 0) {
      fragments[fragments.length - 1].punctuation += character;
    }
    buffer = "";
  }

  const tail = cleanFragment(buffer);
  if (tail) fragments.push({ text: tail, punctuation: "" });
  return fragments;
}

export function buildAdjacentSamples(document, options = {}) {
  const tokenization = options.tokenization || "character";
  const fragments = splitIntoFragments(document.text);
  const samples = [];

  for (let index = 0; index < fragments.length - 1; index += 1) {
    const leftText = fragments[index].text;
    const rightText = fragments[index + 1].text;
    const leftCharacterLength = contentCharacterLength(leftText);
    const rightCharacterLength = contentCharacterLength(rightText);

    const left = tokenize(leftText, tokenization);
    const right = tokenize(rightText, tokenization);
    if (left.length === 0 || right.length === 0) continue;

    const tokens = [...left, ...right];
    if (tokens.length < 2) continue;

    samples.push({
      id: `${document.id}:pair:${index}`,
      domain: document.domain,
      document_id: document.id,
      tokens,
      target_index: left.length - 1,
      punctuation: fragments[index].punctuation,
      left_character_length: leftCharacterLength,
      right_character_length: rightCharacterLength,
      relative_boundary_position: (
        leftCharacterLength / (leftCharacterLength + rightCharacterLength)
      ),
      tokenization,
    });
  }

  return samples;
}

function parseJsonLines(text) {
  return text
    .split(/\r?\n/u)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function readZipEntry(zipPath, entry) {
  return execFileSync("unzip", ["-p", zipPath, entry], {
    encoding: "utf8",
    maxBuffer: 128 * 1024 * 1024,
  });
}

function decodeXmlEntities(text) {
  return text.replace(/&(?:#(\d+)|#x([\da-f]+)|(amp|lt|gt|quot|apos));/giu, (
    entity,
    decimal,
    hexadecimal,
    named,
  ) => {
    if (decimal) return String.fromCodePoint(Number.parseInt(decimal, 10));
    if (hexadecimal) return String.fromCodePoint(Number.parseInt(hexadecimal, 16));
    return {
      amp: "&",
      lt: "<",
      gt: ">",
      quot: "\"",
      apos: "'",
    }[named.toLowerCase()];
  });
}

function removeBalancedMarkup(text, opening, closing) {
  let result = "";
  let depth = 0;

  for (let index = 0; index < text.length;) {
    if (text.startsWith(opening, index)) {
      depth += 1;
      index += opening.length;
      continue;
    }
    if (depth > 0 && text.startsWith(closing, index)) {
      depth -= 1;
      index += closing.length;
      continue;
    }
    if (depth === 0) result += text[index];
    index += 1;
  }

  return result;
}

export function cleanWikipediaMarkup(wikitext) {
  let text = String(wikitext || "")
    .replace(/<!--[\s\S]*?-->/gu, " ")
    .replace(/<ref\b[^>]*\/>/giu, " ")
    .replace(/<ref\b[^>]*>[\s\S]*?<\/ref\s*>/giu, " ")
    .replace(/<(math|code|timeline|gallery|mapframe|syntaxhighlight)\b[^>]*>[\s\S]*?<\/\1\s*>/giu, " ");

  text = removeBalancedMarkup(text, "{{", "}}");
  text = removeBalancedMarkup(text, "{|", "|}");
  text = text
    .replace(/\[\[(?:File|Image|文件|檔案|图像|圖像|Category|分类|分類):[^\]]*\]\]/giu, " ")
    .replace(/\[\[[^\]|]+\|([^\]]+)\]\]/gu, "$1")
    .replace(/\[\[([^\]]+)\]\]/gu, "$1")
    .replace(/\[(?:https?|ftp):\/\/\S+(?:\s+([^\]]+))?\]/giu, "$1")
    .replace(/(?:https?|ftp):\/\/\S+/giu, " ")
    .replace(/<[^>]+>/gu, " ")
    .replace(/'{2,5}/gu, "")
    .replace(/^\s*(?:={2,}|[*#:;]+|\|-?|!+).*$/gmu, " ")
    .replace(/__(?:TOC|NOTOC|FORCETOC|NOEDITSECTION|NEWSECTIONLINK|NONEWSECTIONLINK)__/giu, " ")
    .replace(/&nbsp;/giu, " ");

  return normalizeDocument(text);
}

export function wikipediaDocumentsFromXml(xml, streamOffset = 0) {
  const documents = [];
  const pages = String(xml).match(/<page>[\s\S]*?<\/page>/gu) || [];

  for (const page of pages) {
    const namespace = page.match(/<ns>(-?\d+)<\/ns>/u)?.[1];
    if (namespace !== "0" || /<redirect\b/iu.test(page)) continue;

    const pageId = page.match(/<id>(\d+)<\/id>/u)?.[1];
    const title = decodeXmlEntities(page.match(/<title>([\s\S]*?)<\/title>/u)?.[1] || "");
    const encodedText = page.match(/<text\b[^>]*>([\s\S]*?)<\/text>/u)?.[1] || "";
    const text = cleanWikipediaMarkup(decodeXmlEntities(encodedText));
    if (!pageId || !text || /^#(?:REDIRECT|重定向|重新導向)/iu.test(text)) continue;

    documents.push({
      id: `wikipedia:${pageId}`,
      domain: "wikipedia",
      title,
      text,
      stream_offset: streamOffset,
    });
  }

  return documents;
}

function readWikipediaStreamOffsets(indexPath) {
  const index = execFileSync("bzip2", ["-dc", indexPath], {
    encoding: "utf8",
    maxBuffer: 512 * 1024 * 1024,
  });
  const offsets = new Set();
  for (const line of index.split(/\r?\n/u)) {
    const separator = line.indexOf(":");
    if (separator < 1) continue;
    const offset = Number(line.slice(0, separator));
    if (Number.isSafeInteger(offset) && offset >= 0) offsets.add(offset);
  }
  return Array.from(offsets).sort((left, right) => left - right);
}

async function readWikipediaDocuments(paths, limit, seed) {
  if (!Number.isSafeInteger(limit) || limit < 1) return [];

  const offsets = readWikipediaStreamOffsets(paths.index.path);
  const dumpSize = (await stat(paths.primary.path)).size;
  const nextOffset = new Map(offsets.map((offset, index) => [
    offset,
    offsets[index + 1] ?? dumpSize,
  ]));
  const rankedOffsets = [...offsets].sort((left, right) => (
    hashText(`${seed}:wikipedia:${left}`)
      .localeCompare(hashText(`${seed}:wikipedia:${right}`))
  ));
  const dump = await open(paths.primary.path, "r");
  const documents = [];

  try {
    for (const offset of rankedOffsets) {
      const length = nextOffset.get(offset) - offset;
      const compressed = Buffer.allocUnsafe(length);
      const { bytesRead } = await dump.read(compressed, 0, length, offset);
      if (bytesRead !== length) {
        throw new Error(`Short read in Wikipedia dump at byte ${offset}`);
      }
      const xml = execFileSync("bzip2", ["-dc"], {
        input: compressed,
        encoding: "utf8",
        maxBuffer: 256 * 1024 * 1024,
      });
      documents.push(...wikipediaDocumentsFromXml(xml, offset));
      if (documents.length >= limit) break;
    }
  } finally {
    await dump.close();
  }

  if (documents.length < limit) {
    throw new Error(`Only ${documents.length} Wikipedia articles; need ${limit}`);
  }
  return documents.slice(0, limit);
}

async function documentsFromSource(source, downloaded, options) {
  if (source.domain === "wikipedia") {
    return readWikipediaDocuments(downloaded, options.wikipediaDocs, options.seed);
  }

  const zipPath = downloaded.primary.path;
  if (source.domain === "news") {
    return parseJsonLines(readZipEntry(zipPath, source.entries[0])).map((row, index) => ({
      id: `news:${index}`,
      domain: source.domain,
      text: row.sentence,
    }));
  }

  if (source.domain === "academic") {
    return parseJsonLines(readZipEntry(zipPath, source.entries[0])).map((row, index) => ({
      id: `academic:${row.id ?? index}`,
      domain: source.domain,
      text: row.abst,
    }));
  }

  if (source.domain === "encyclopedia") {
    const payload = JSON.parse(readZipEntry(zipPath, source.entries[0]));
    return payload.data.flatMap((article, articleIndex) => (
      article.paragraphs.map((paragraph, paragraphIndex) => ({
        id: `encyclopedia:${article.id ?? articleIndex}:${paragraphIndex}`,
        domain: source.domain,
        text: paragraph.context,
      }))
    ));
  }

  return source.entries.flatMap((entry) => {
    const payload = JSON.parse(readZipEntry(zipPath, entry));
    return payload.map((row, index) => ({
      id: `dialogue:${entry}:${index}`,
      domain: source.domain,
      text: Array.isArray(row[0]) ? row[0].join(" ") : row[0],
    }));
  });
}

async function checksumForSource(source) {
  if (source.sha256) return { algorithm: "sha256", value: source.sha256 };

  let manifest = CHECKSUM_MANIFESTS.get(source.checksumUrl);
  if (!manifest) {
    const response = await fetch(source.checksumUrl);
    if (!response.ok) {
      throw new Error(`Failed to download ${source.checksumUrl}: ${response.status}`);
    }
    manifest = await response.text();
    CHECKSUM_MANIFESTS.set(source.checksumUrl, manifest);
  }

  const match = manifest
    .split(/\r?\n/u)
    .map((line) => line.trim().match(/^([\da-f]+)\s+(.+)$/iu))
    .find((parts) => parts && parts[2].endsWith(`-${source.checksumSuffix}`));
  if (!match) {
    throw new Error(`No checksum for ${source.checksumSuffix} in ${source.checksumUrl}`);
  }
  return { algorithm: "md5", value: match[1].toLowerCase() };
}

function hashFile(path, algorithm) {
  return new Promise((resolveHash, rejectHash) => {
    const hash = createHash(algorithm);
    const stream = createReadStream(path);
    stream.on("error", rejectHash);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("end", () => resolveHash(hash.digest("hex")));
  });
}

async function ensureDownload(source) {
  await mkdir(RAW_DIR, { recursive: true });
  const path = join(RAW_DIR, source.filename);
  const temporaryPath = `${path}.part`;
  const checksum = await checksumForSource(source);
  let currentHash = null;

  try {
    currentHash = await hashFile(path, checksum.algorithm);
  } catch {
    // Download below.
  }

  if (currentHash !== checksum.value) {
    console.log(`Downloading ${source.url}`);
    const response = await fetch(source.url);
    if (!response.ok || !response.body) {
      throw new Error(`Failed to download ${source.url}: ${response.status}`);
    }
    try {
      await pipeline(Readable.fromWeb(response.body), createWriteStream(temporaryPath));
      currentHash = await hashFile(temporaryPath, checksum.algorithm);
    } catch (error) {
      await rm(temporaryPath, { force: true });
      throw error;
    }
    if (currentHash !== checksum.value) {
      await rm(temporaryPath, { force: true });
      throw new Error(`Checksum mismatch for ${source.filename}`);
    }
    await rename(temporaryPath, path);
  }

  return {
    path,
    filename: source.filename,
    url: source.url,
    checksum: `${checksum.algorithm}:${checksum.value}`,
  };
}

async function ensureSourceDownloads(source) {
  const primary = await ensureDownload(source);
  const index = source.index ? await ensureDownload(source.index) : null;
  return { primary, index };
}

function filterAndDeduplicate(documentsByDomain) {
  const seen = new Set();
  const result = new Map();
  let duplicateCount = 0;

  for (const [domain, documents] of documentsByDomain) {
    const accepted = [];
    for (const document of documents) {
      const text = normalizeDocument(document.text);
      if (splitIntoFragments(text).length < 2) continue;

      const fingerprint = hashText(text);
      if (seen.has(fingerprint)) {
        duplicateCount += 1;
        continue;
      }
      seen.add(fingerprint);
      accepted.push({ ...document, text, fingerprint });
    }
    result.set(domain, accepted);
  }

  return { documentsByDomain: result, duplicateCount };
}

function partitionDocuments(documents, docsPerDomain, seed) {
  const selectionCount = docsPerDomain > 0 ? docsPerDomain : documents.length;
  const selected = [...documents]
    .sort((left, right) => (
      hashText(`${seed}:${left.fingerprint}`)
        .localeCompare(hashText(`${seed}:${right.fingerprint}`))
    ))
    .slice(0, selectionCount);

  if (selected.length < selectionCount) {
    throw new Error(`Only ${selected.length} eligible documents; need ${selectionCount}`);
  }

  const trainEnd = Math.floor(selected.length * 0.8);
  const validationEnd = trainEnd + Math.floor(selected.length * 0.1);
  return {
    train: selected.slice(0, trainEnd),
    validation: selected.slice(trainEnd, validationEnd),
    test: selected.slice(validationEnd),
  };
}

function selectDeterministicSamples(samples, count, seed) {
  const selected = [...samples]
    .sort((left, right) => (
      hashText(`${seed}:${left.id}`).localeCompare(hashText(`${seed}:${right.id}`))
    ))
    .slice(0, count);

  if (selected.length < count) {
    throw new Error(`Only ${selected.length} pair samples; need ${count}`);
  }
  return selected;
}

function positionBin(sample, binCount) {
  return Math.min(
    binCount - 1,
    Math.floor(sample.relative_boundary_position * binCount),
  );
}

function lengthBucket(sample, maximums = LENGTH_BUCKET_MAXIMUMS) {
  const length = sample.left_character_length + sample.right_character_length;
  const maximum = maximums.find((candidate) => length <= candidate);
  return maximum === undefined ? `${maximums.at(-1) + 1}+` : `<=${maximum}`;
}

export function assignLengthPositionWeights(
  samples,
  positionBinCount = 10,
  lengthBucketMaximums = LENGTH_BUCKET_MAXIMUMS,
) {
  const cellCounts = new Map();
  const lengthBucketCounts = new Map();
  const annotated = samples.map((sample) => {
    const sampleLengthBucket = lengthBucket(sample, lengthBucketMaximums);
    const samplePositionBin = positionBin(sample, positionBinCount);
    const cellKey = `${sampleLengthBucket}:${samplePositionBin}`;
    cellCounts.set(cellKey, (cellCounts.get(cellKey) || 0) + 1);
    lengthBucketCounts.set(
      sampleLengthBucket,
      (lengthBucketCounts.get(sampleLengthBucket) || 0) + 1,
    );
    return {
      ...sample,
      length_bucket: sampleLengthBucket,
      position_bin: samplePositionBin,
      cell_key: cellKey,
    };
  });

  const occupiedPositionBins = new Map();
  for (const cellKey of cellCounts.keys()) {
    const sampleLengthBucket = cellKey.slice(0, cellKey.lastIndexOf(":"));
    occupiedPositionBins.set(
      sampleLengthBucket,
      (occupiedPositionBins.get(sampleLengthBucket) || 0) + 1,
    );
  }

  return annotated.map(({ cell_key: cellKey, ...sample }) => {
    const targetCellWeight = (
      lengthBucketCounts.get(sample.length_bucket)
      / occupiedPositionBins.get(sample.length_bucket)
    );
    return {
      ...sample,
      training_weight: targetCellWeight / cellCounts.get(cellKey),
    };
  });
}

export function selectPositionBalancedSamples(samples, count, seed, binCount = 10) {
  const bins = Array.from({ length: binCount }, () => []);
  for (const sample of samples) bins[positionBin(sample, binCount)].push(sample);
  for (const bin of bins) {
    bin.sort((left, right) => (
      hashText(`${seed}:${left.id}`).localeCompare(hashText(`${seed}:${right.id}`))
    ));
  }

  const selected = [];
  const cursors = Array(binCount).fill(0);
  const startBin = Number.parseInt(hashText(String(seed)).slice(0, 8), 16) % binCount;

  while (selected.length < count) {
    let madeProgress = false;
    for (let offset = 0; offset < binCount && selected.length < count; offset += 1) {
      const binIndex = (startBin + offset) % binCount;
      if (cursors[binIndex] >= bins[binIndex].length) continue;
      selected.push(bins[binIndex][cursors[binIndex]]);
      cursors[binIndex] += 1;
      madeProgress = true;
    }
    if (!madeProgress) break;
  }

  if (selected.length < count) {
    throw new Error(`Only ${selected.length} pair samples; need ${count}`);
  }
  return selected;
}

function sampleBoundarySignature(sample) {
  const left = sample.tokens.slice(0, sample.target_index + 1).join("");
  const right = sample.tokens.slice(sample.target_index + 1).join("");
  return `${left}\u0000${right}`;
}

export function deduplicateSampleBoundaries(samplesBySplit, seed) {
  const seen = new Set();
  const deduplicated = {};
  const removed = {};

  // Holdouts claim repeated boundaries first so evaluation examples can never
  // be learned from an identical training pair. Test precedes validation to
  // keep the final metric independent from validation-driven model choices.
  for (const split of ["test", "validation", "train"]) {
    const samples = [...(samplesBySplit[split] || [])].sort((left, right) => (
      hashText(`${seed}:sample-deduplication:${left.id}`)
        .localeCompare(hashText(`${seed}:sample-deduplication:${right.id}`))
    ));
    const kept = [];
    let removedCount = 0;

    for (const sample of samples) {
      const signature = sampleBoundarySignature(sample);
      if (seen.has(signature)) {
        removedCount += 1;
        continue;
      }
      seen.add(signature);
      kept.push(sample);
    }
    deduplicated[split] = kept;
    removed[split] = removedCount;
  }

  return {
    samplesBySplit: deduplicated,
    removed: {
      ...removed,
      total: Object.values(removed).reduce((total, count) => total + count, 0),
    },
  };
}

function positionHistogram(samples, binCount) {
  const histogram = Array(binCount).fill(0);
  for (const sample of samples) histogram[positionBin(sample, binCount)] += 1;
  return histogram;
}

function parseArguments(argv) {
  const options = {
    outputDir: DEFAULT_PROCESSED_DIR,
    tokenization: "character",
    docsPerDomain: 0,
    wikipediaDocs: 250000,
    trainPerDomain: 0,
    validationPerDomain: 0,
    testPerDomain: 0,
    positionBins: 10,
    seed: 2026090405,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key.startsWith("--") || value === undefined) continue;
    const optionName = key.slice(2).replace(/-([a-z])/gu, (_, letter) => letter.toUpperCase());
    if (Object.hasOwn(options, optionName)) {
      options[optionName] = ["outputDir", "tokenization"].includes(optionName)
        ? value
        : Number(value);
      index += 1;
    }
  }
  if (!["word", "character"].includes(options.tokenization)) {
    throw new Error(`Unknown tokenization mode: ${options.tokenization}`);
  }
  options.outputDir = resolve(options.outputDir);
  return options;
}

async function writeJsonLines(path, records) {
  const temporaryPath = `${path}.part`;
  const output = await open(temporaryPath, "w");
  const recordsPerChunk = 5000;
  let complete = false;

  try {
    for (let index = 0; index < records.length; index += recordsPerChunk) {
      const chunk = records
        .slice(index, index + recordsPerChunk)
        .map((record) => JSON.stringify(record))
        .join("\n");
      await output.write(`${chunk}\n`);
    }
    complete = true;
  } finally {
    await output.close();
    if (!complete) await rm(temporaryPath, { force: true });
  }
  await rename(temporaryPath, path);
}

function verifyNoDocumentLeakage(samplesBySplit) {
  const owners = new Map();
  for (const [split, samples] of Object.entries(samplesBySplit)) {
    for (const sample of samples) {
      const previous = owners.get(sample.document_id);
      if (previous && previous !== split) {
        throw new Error(`Document leakage: ${sample.document_id} in ${previous} and ${split}`);
      }
      owners.set(sample.document_id, split);
    }
  }
}

function verifyNoSampleLeakage(samplesBySplit) {
  const owners = new Map();
  for (const [split, samples] of Object.entries(samplesBySplit)) {
    for (const sample of samples) {
      const signature = sampleBoundarySignature(sample);
      const previous = owners.get(signature);
      if (previous) {
        throw new Error(`Sample boundary leakage: ${sample.id} duplicates ${previous.id}`);
      }
      owners.set(signature, { id: sample.id, split });
    }
  }
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const downloaded = await Promise.all(SOURCES.map(ensureSourceDownloads));
  const loaded = new Map();

  for (let index = 0; index < SOURCES.length; index += 1) {
    const source = SOURCES[index];
    loaded.set(
      source.domain,
      await documentsFromSource(source, downloaded[index], options),
    );
  }

  const deduplicated = filterAndDeduplicate(loaded);
  const candidatesBySplitAndDomain = {
    train: new Map(),
    validation: new Map(),
    test: new Map(),
  };
  const samplesBySplitAndDomain = {
    train: new Map(),
    validation: new Map(),
    test: new Map(),
  };
  const documentCounts = {};

  for (const source of SOURCES) {
    const domain = source.domain;
    const documents = deduplicated.documentsByDomain.get(domain);
    const partitions = partitionDocuments(documents, options.docsPerDomain, options.seed);
    documentCounts[domain] = {};

    for (const [split, partition] of Object.entries(partitions)) {
      documentCounts[domain][split] = partition.length;
      const candidates = partition.flatMap((document) => buildAdjacentSamples(document, {
        tokenization: options.tokenization,
      }));
      candidatesBySplitAndDomain[split].set(domain, candidates);
    }
  }

  const sampleDeduplication = deduplicateSampleBoundaries(
    Object.fromEntries(Object.entries(candidatesBySplitAndDomain).map(([split, byDomain]) => [
      split,
      Array.from(byDomain.values()).flat(),
    ])),
    options.seed,
  );
  for (const [split, samples] of Object.entries(sampleDeduplication.samplesBySplit)) {
    const byDomain = new Map(SOURCES.map((source) => [source.domain, []]));
    for (const sample of samples) byDomain.get(sample.domain).push(sample);
    candidatesBySplitAndDomain[split] = byDomain;
  }

  const availableSampleCounts = {};
  const selectedSamplesPerDomain = {};
  for (const [split, byDomain] of Object.entries(candidatesBySplitAndDomain)) {
    availableSampleCounts[split] = Object.fromEntries(
      Array.from(byDomain, ([domain, candidates]) => [domain, candidates.length]),
    );
    const requested = options[`${split}PerDomain`];
    selectedSamplesPerDomain[split] = {};

    for (const [domain, candidates] of byDomain) {
      const selectionSeed = `${options.seed}:${split}:${domain}`;
      const selected = requested > 0
        ? split === "train"
          ? selectPositionBalancedSamples(
            candidates,
            requested,
            selectionSeed,
            options.positionBins,
          )
          : selectDeterministicSamples(candidates, requested, selectionSeed)
        : candidates;
      selectedSamplesPerDomain[split][domain] = selected.length;
      samplesBySplitAndDomain[split].set(
        domain,
        selected,
      );
    }
  }

  const samplesBySplit = {};
  const sampleCounts = {};
  for (const [split, byDomain] of Object.entries(samplesBySplitAndDomain)) {
    const combined = Array.from(byDomain.values()).flat();
    samplesBySplit[split] = (split === "train"
      ? assignLengthPositionWeights(combined, options.positionBins)
      : combined)
      .sort((left, right) => (
        hashText(`${options.seed}:${split}:${left.id}`)
          .localeCompare(hashText(`${options.seed}:${split}:${right.id}`))
      ));
    sampleCounts[split] = Object.fromEntries(
      Array.from(byDomain, ([domain, samples]) => [domain, samples.length]),
    );
  }

  verifyNoDocumentLeakage(samplesBySplit);
  verifyNoSampleLeakage(samplesBySplit);
  await mkdir(options.outputDir, { recursive: true });
  await Promise.all(Object.entries(samplesBySplit).map(([split, samples]) => (
    writeJsonLines(join(options.outputDir, `${split}.jsonl`), samples)
  )));

  const summary = {
    seed: options.seed,
    tokenization: options.tokenization,
    candidate_positions: options.tokenization === "word"
      ? "between adjacent Intl.Segmenter word tokens"
      : "between every adjacent Han character",
    non_han_policy: "excluded before model input",
    max_side_characters: null,
    sample_selection: "all eligible samples unless an explicit per-domain cap is provided",
    domain_balancing: false,
    training_balance: {
      strategy: "loss weights balance relative boundary positions within each total-length bucket",
      length_bucket_maximums: LENGTH_BUCKET_MAXIMUMS,
      position_bins: options.positionBins,
    },
    sources: SOURCES.map((source, index) => ({
      domain: source.domain,
      downloads: [downloaded[index].primary, downloaded[index].index]
        .filter(Boolean)
        .map(({ filename, url, checksum }) => ({ filename, url, checksum })),
    })),
    duplicate_documents_removed: deduplicated.duplicateCount,
    duplicate_sample_boundaries_removed: sampleDeduplication.removed,
    documents: documentCounts,
    available_samples: availableSampleCounts,
    selected_samples_per_domain: selectedSamplesPerDomain,
    samples: sampleCounts,
    relative_position_histograms: Object.fromEntries(
      Object.entries(samplesBySplit).map(([split, samples]) => [
        split,
        positionHistogram(samples, options.positionBins),
      ]),
    ),
    leakage_check: "passed (document IDs and sample boundaries)",
  };
  await writeFile(
    join(options.outputDir, "summary.json"),
    `${JSON.stringify(summary, null, 2)}\n`,
  );
  console.log(JSON.stringify(summary, null, 2));
}

const isMain = process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url;
if (isMain) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
