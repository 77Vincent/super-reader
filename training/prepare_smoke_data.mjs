import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const RAW_DIR = join(SCRIPT_DIR, "data", "raw");
const PROCESSED_DIR = join(SCRIPT_DIR, "data", "processed");
const SEGMENTER = new Intl.Segmenter("zh-CN", { granularity: "word" });
const CONTENT_CHARACTER = /[\p{L}\p{N}]/u;
const PROXY_PUNCTUATION = new Set([
  "，", ",", "。", ".", "！", "!", "？", "?", "；", ";", "、",
]);

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
];

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

function tokenize(text) {
  return Array.from(SEGMENTER.segment(text))
    .filter((item) => item.isWordLike)
    .map((item) => item.segment);
}

function contentCharacterLength(text) {
  return Array.from(text)
    .filter((character) => CONTENT_CHARACTER.test(character)).length;
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

export function buildAdjacentSamples(document) {
  const fragments = splitIntoFragments(document.text);
  const samples = [];

  for (let index = 0; index < fragments.length - 1; index += 1) {
    const leftText = fragments[index].text;
    const rightText = fragments[index + 1].text;
    const leftCharacterLength = contentCharacterLength(leftText);
    const rightCharacterLength = contentCharacterLength(rightText);

    const left = tokenize(leftText);
    const right = tokenize(rightText);
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

function documentsFromSource(source, zipPath) {
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

async function ensureDownload(source) {
  await mkdir(RAW_DIR, { recursive: true });
  const path = join(RAW_DIR, source.filename);
  let current = null;

  try {
    current = await readFile(path);
  } catch {
    // Download below.
  }

  if (!current || hashText(current) !== source.sha256) {
    const response = await fetch(source.url);
    if (!response.ok) {
      throw new Error(`Failed to download ${source.url}: ${response.status}`);
    }
    current = Buffer.from(await response.arrayBuffer());
    if (hashText(current) !== source.sha256) {
      throw new Error(`Checksum mismatch for ${source.filename}`);
    }
    await writeFile(path, current);
  }

  return path;
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
  const selected = [...documents]
    .sort((left, right) => (
      hashText(`${seed}:${left.fingerprint}`)
        .localeCompare(hashText(`${seed}:${right.fingerprint}`))
    ))
    .slice(0, docsPerDomain);

  if (selected.length < docsPerDomain) {
    throw new Error(`Only ${selected.length} eligible documents; need ${docsPerDomain}`);
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

function positionHistogram(samples, binCount) {
  const histogram = Array(binCount).fill(0);
  for (const sample of samples) histogram[positionBin(sample, binCount)] += 1;
  return histogram;
}

function parseArguments(argv) {
  const options = {
    docsPerDomain: 400,
    trainPerDomain: 128,
    validationPerDomain: 32,
    testPerDomain: 32,
    positionBins: 10,
    seed: 2026090405,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key.startsWith("--") || value === undefined) continue;
    const optionName = key.slice(2).replace(/-([a-z])/gu, (_, letter) => letter.toUpperCase());
    if (Object.hasOwn(options, optionName)) {
      options[optionName] = Number(value);
      index += 1;
    }
  }
  return options;
}

async function writeJsonLines(path, records) {
  await writeFile(path, `${records.map((record) => JSON.stringify(record)).join("\n")}\n`);
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

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const downloaded = await Promise.all(SOURCES.map(ensureDownload));
  const loaded = new Map();

  SOURCES.forEach((source, index) => {
    loaded.set(source.domain, documentsFromSource(source, downloaded[index]));
  });

  const deduplicated = filterAndDeduplicate(loaded);
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
      const candidates = partition.flatMap(buildAdjacentSamples);
      const requested = options[`${split}PerDomain`];
      const selectionSeed = `${options.seed}:${split}:${domain}`;
      const selected = split === "train"
        ? selectPositionBalancedSamples(
          candidates,
          requested,
          selectionSeed,
          options.positionBins,
        )
        : selectDeterministicSamples(candidates, requested, selectionSeed);
      samplesBySplitAndDomain[split].set(
        domain,
        selected,
      );
    }
  }

  const samplesBySplit = {};
  const sampleCounts = {};
  for (const [split, byDomain] of Object.entries(samplesBySplitAndDomain)) {
    samplesBySplit[split] = Array.from(byDomain.values())
      .flat()
      .sort((left, right) => (
        hashText(`${options.seed}:${split}:${left.id}`)
          .localeCompare(hashText(`${options.seed}:${split}:${right.id}`))
      ));
    sampleCounts[split] = Object.fromEntries(
      Array.from(byDomain, ([domain, samples]) => [domain, samples.length]),
    );
  }

  verifyNoDocumentLeakage(samplesBySplit);
  await mkdir(PROCESSED_DIR, { recursive: true });
  await Promise.all(Object.entries(samplesBySplit).map(([split, samples]) => (
    writeJsonLines(join(PROCESSED_DIR, `${split}.jsonl`), samples)
  )));

  const summary = {
    seed: options.seed,
    max_side_characters: null,
    training_position_bins: options.positionBins,
    sources: SOURCES.map(({ domain, url, sha256 }) => ({ domain, url, sha256 })),
    duplicate_documents_removed: deduplicated.duplicateCount,
    documents: documentCounts,
    samples: sampleCounts,
    relative_position_histograms: Object.fromEntries(
      Object.entries(samplesBySplit).map(([split, samples]) => [
        split,
        positionHistogram(samples, options.positionBins),
      ]),
    ),
    leakage_check: "passed",
  };
  await writeFile(
    join(PROCESSED_DIR, "summary.json"),
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
