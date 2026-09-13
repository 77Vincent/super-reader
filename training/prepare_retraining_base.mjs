#!/usr/bin/env node
// Regenerate training and evaluation labels while retaining document ownership.
// The full-Wikipedia preparer adds Wikipedia training examples afterward.
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdir, open, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath, pathToFileURL } from "node:url";
import { SOURCES, buildAdjacentSamples, documentsFromSource } from "./prepare_smoke_data.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const raw = join(root, "training/data/raw");
export const LOCAL_CLUE_ENTRIES = {
  news: ["train.json", "dev.json", "test.json", "test1.0.json"],
  academic: ["train.json", "dev.json", "test.json"],
  encyclopedia: ["train.json", "dev.json", "test.json", "trial.json"],
  dialogue: ["d-train.json", "m-train.json", "d-dev.json", "m-dev.json", "test1.0.json", "test1.1.json"],
};

export function documentFingerprint(text) {
  const han = Array.from(String(text || "").normalize("NFKC"))
    .filter((character) => /\p{Script=Han}/u.test(character)).join("");
  return createHash("sha256").update(han).digest("hex");
}

export async function localClueDocuments(source, downloaded) {
  const result = [];
  for (const entry of LOCAL_CLUE_ENTRIES[source.domain]) {
    const documents = await documentsFromSource({ ...source, entries: [entry] }, downloaded, {});
    for (const document of documents) {
      result.push(source.entries.includes(entry) ? document : {
        ...document, id: `${source.domain}:local:${entry}:${document.id}`,
      });
    }
  }
  return result;
}

async function hashFile(path, algorithm = "sha256") {
  const hash = createHash(algorithm);
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

export function samplesForReferenceDocument(document, owners) {
  const split = owners.get(document.id);
  if (!split) return null;
  return { split, samples: buildAdjacentSamples(document, { tokenization: "character" }) };
}

async function main() {
  const options = { "reference-data-dir": "training/data/processed", "output-dir": "", "wikipedia-docs": "250000", "all-local-clue": "0" };
  for (let index = 2; index < process.argv.length; index += 2) {
    const key = process.argv[index].replace(/^--/, "");
    if (!Object.hasOwn(options, key) || process.argv[index + 1] === undefined) {
      throw new Error(`Unknown or incomplete option: ${process.argv[index]}`);
    }
    options[key] = process.argv[index + 1];
  }
  if (!options["output-dir"]) throw new Error("--output-dir is required and must be a new directory");
  const reference = resolve(root, options["reference-data-dir"]);
  const output = resolve(root, options["output-dir"]);
  const summaryBytes = await readFile(join(reference, "summary.json"));
  const previous = JSON.parse(summaryBytes);
  if (previous.tokenization !== "character") throw new Error("Expected character-tokenized reference data");
  const wikipediaDocs = Number(options["wikipedia-docs"]);
  if (!Number.isSafeInteger(wikipediaDocs) || wikipediaDocs < 1) throw new Error("Invalid --wikipedia-docs");
  if (!["0", "1"].includes(options["all-local-clue"])) throw new Error("--all-local-clue must be 0 or 1");
  const allLocalClue = options["all-local-clue"] === "1";

  const owners = new Map();
  const referenceHashes = {};
  for (const split of ["train", "validation", "test"]) {
    const hash = createHash("sha256");
    const stream = createReadStream(join(reference, `${split}.jsonl`));
    stream.on("data", (chunk) => hash.update(chunk));
    for await (const line of createInterface({ input: stream, crlfDelay: Infinity })) {
      if (!line.trim()) continue;
      const record = JSON.parse(line);
      const owner = owners.get(record.document_id);
      if (owner && owner !== split) throw new Error(`Reference document leakage: ${record.document_id}`);
      owners.set(record.document_id, split);
    }
    referenceHashes[split] = hash.digest("hex");
  }
  const requiredIds = new Set([...owners].filter(([id, split]) => split !== "train" || !id.startsWith("wikipedia:")).map(([id]) => id));
  console.log(`Retaining document ownership; regenerating ${requiredIds.size} source documents`);

  // Refuse reuse of an output directory to avoid mixing label definitions.
  await mkdir(output);
  const handles = {};
  for (const split of ["train", "validation", "test"]) {
    handles[split] = await open(join(output, `${split}.raw.jsonl`), "wx");
  }
  const counts = { train: {}, validation: {}, test: {} };
  const foundIds = new Set();
  const sources = [];
  const loadedSources = [];
  const holdoutHashes = new Set();
  const trainingHashes = new Set();
  const corpusCounts = {};
  try {
    for (const source of SOURCES) {
      const recordedSource = previous.sources.find(({ domain }) => domain === source.domain);
      const downloaded = {};
      for (const [key, descriptor] of [["primary", source], ["index", source.index]]) {
        if (!descriptor) continue;
        const recorded = recordedSource?.downloads.find(({ filename }) => filename === descriptor.filename);
        if (!recorded) throw new Error(`Missing recorded source: ${descriptor.filename}`);
        const path = join(raw, descriptor.filename);
        const [algorithm, expected] = recorded.checksum.split(":");
        if (await hashFile(path, algorithm) !== expected) throw new Error(`Cached source checksum mismatch: ${path}`);
        downloaded[key] = { ...recorded, path };
      }
      sources.push(recordedSource);
      for (const split of Object.keys(counts)) counts[split][source.domain] = 0;
      const documents = allLocalClue && source.domain !== "wikipedia"
        ? await localClueDocuments(source, downloaded)
        : await documentsFromSource(source, downloaded, { wikipediaDocs, seed: previous.seed });
      loadedSources.push({ source, documents });
      corpusCounts[source.domain] = { documents_read: documents.length, protected_or_duplicate_training_documents: 0 };
      for (const document of documents) {
        if (["validation", "test"].includes(owners.get(document.id))) {
          holdoutHashes.add(documentFingerprint(document.text));
        }
      }
    }
    for (const { source, documents } of loadedSources) {
      for (const document of documents) {
        const required = requiredIds.has(document.id);
        if (!required && !(allLocalClue && source.domain !== "wikipedia")) continue;
        if (required) foundIds.add(document.id);
        const split = owners.get(document.id) || "train";
        if (allLocalClue && split === "train") {
          const fingerprint = documentFingerprint(document.text);
          if (holdoutHashes.has(fingerprint) || trainingHashes.has(fingerprint)) {
            corpusCounts[source.domain].protected_or_duplicate_training_documents += 1;
            continue;
          }
          trainingHashes.add(fingerprint);
        }
        const samples = buildAdjacentSamples(document, { tokenization: "character" });
        if (samples.some(({ punctuation }) => punctuation.includes("、"))) throw new Error("Enumeration target generated");
        if (samples.length) await handles[split].write(samples.map((sample) => JSON.stringify(sample)).join("\n") + "\n");
        counts[split][source.domain] += samples.length;
      }
      console.log(`${source.domain}: ${JSON.stringify(Object.fromEntries(Object.entries(counts).map(([split, byDomain]) => [split, byDomain[source.domain]])))}`);
    }
  } finally {
    for (const handle of Object.values(handles)) await handle.close();
  }
  if (foundIds.size !== requiredIds.size) {
    const missing = [...requiredIds].filter((id) => !foundIds.has(id));
    throw new Error(`Some reference documents were not recovered: ${missing.slice(0, 10).join(", ")}`);
  }
  // Keep every holdout document excluded even if its revised labels are empty.
  await writeFile(join(output, "holdout-documents.json"), JSON.stringify(
    [...owners].filter(([, split]) => split !== "train").map(([id]) => id),
  ) + "\n");
  await writeFile(join(output, "holdout-document-hashes.json"), JSON.stringify([...holdoutHashes]) + "\n");
  const seen = new Set();
  const deduplicated = {};
  for (const split of ["test", "validation", "train"]) {
    let kept = 0;
    const handle = await open(join(output, `${split}.jsonl.part`), "wx");
    const input = createReadStream(join(output, `${split}.raw.jsonl`));
    try {
      let buffer = "";
      for await (const line of createInterface({ input, crlfDelay: Infinity })) {
        const record = JSON.parse(line);
        const signature = createHash("sha256").update(JSON.stringify([record.tokens.join(""), record.target_index])).digest("hex");
        if (seen.has(signature)) continue;
        seen.add(signature);
        buffer += line + "\n";
        kept += 1;
        if (buffer.length >= 1024 * 1024) { await handle.write(buffer); buffer = ""; }
      }
      if (buffer) await handle.write(buffer);
    } finally { await handle.close(); }
    await rename(join(output, `${split}.jsonl.part`), join(output, `${split}.jsonl`));
    await rm(join(output, `${split}.raw.jsonl`));
    deduplicated[split] = { count: kept, sha256: await hashFile(join(output, `${split}.jsonl`)) };
    console.log(`${split}: ${kept} unique revised examples`);
  }
  const summary = {
    seed: previous.seed,
    tokenization: "character",
    purpose: "Revised proxies in every split, with original document ownership preserved; full preparation adds Wikipedia training pairs",
    excluded_proxy_punctuation: ["、"],
    all_local_clue_entries: allLocalClue ? LOCAL_CLUE_ENTRIES : null,
    corpus_counts: corpusCounts,
    reference_directory: reference,
    reference_summary_sha256: createHash("sha256").update(summaryBytes).digest("hex"),
    reference_split_sha256: referenceHashes,
    sources,
    regenerated_documents: foundIds.size,
    pairs_before_deduplication: counts,
    splits: deduplicated,
    holdout_documents: "holdout-documents.json",
    holdout_document_hashes: "holdout-document-hashes.json",
    deduplication: "Boundary pairs are unique across splits, with test then validation priority. Full preparation also excludes all holdout document IDs.",
  };
  await writeFile(join(output, "summary.json"), JSON.stringify(summary, null, 2) + "\n");
  console.log(JSON.stringify(summary, null, 2));
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  main().catch((error) => { console.error(error); process.exitCode = 1; });
}
