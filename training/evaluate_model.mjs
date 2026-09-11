#!/usr/bin/env node
// Evaluate the shipped JavaScript model without training or changing runtime rules.
import { createHash } from "node:crypto";
import { createReadStream, readFileSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createInterface } from "node:readline";
import { arch, cpus, platform } from "node:os";
import backend from "../src/backend/inference.js";
import chunker from "../src/backend/chunker.js";
import modelData from "../src/boundary-model-data.js";

const project = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const sha256 = (value) => createHash("sha256").update(value).digest("hex");

/** Separate seeded reservoirs preserve each domain's natural length distribution. */
export function createDomainSampler(limit, seed) {
  if (!Number.isSafeInteger(limit) || limit <= 0) throw new Error("per-domain must be a positive integer");
  const domains = new Map();
  return {
    add(record) {
      let bucket = domains.get(record.domain);
      if (!bucket) {
        const state = createHash("sha256").update(`${seed}:${record.domain}`).digest().readUInt32LE(0);
        bucket = { count: 0, state: state || 1, records: [] };
        domains.set(record.domain, bucket);
      }
      bucket.count += 1;
      if (bucket.records.length < limit) {
        bucket.records.push(record);
        return;
      }
      let state = bucket.state;
      state ^= state << 13;
      state ^= state >>> 17;
      state ^= state << 5;
      bucket.state = state >>> 0;
      const index = Math.floor(bucket.state / 4294967296 * bucket.count);
      if (index < limit) bucket.records[index] = record;
    },
    result() {
      const entries = [...domains].sort(([a], [b]) => a.localeCompare(b, "en"));
      return {
        counts: Object.fromEntries(entries.map(([domain, bucket]) => [domain, bucket.count])),
        records: entries.flatMap(([, bucket]) => [...bucket.records].sort((a, b) => (
          a.id < b.id ? -1 : a.id > b.id ? 1 : 0
        ))),
      };
    },
  };
}

/** Scores describe one labeled gap; softmax confidence is audited, not assumed calibrated. */
export function scorePrediction(scores, target) {
  if (!scores.length || !scores.every(Number.isFinite) ||
      !Number.isInteger(target) || target < 0 || target >= scores.length) {
    throw new Error("Invalid logits or target gap");
  }
  const ranked = scores.map((score, index) => ({ score, index }))
    .sort((a, b) => b.score - a.score || a.index - b.index);
  const predicted = ranked[0].index;
  const rank = ranked.findIndex(({ index }) => index === target) + 1;
  const confidence = 1 / scores.reduce((sum, score) => sum + Math.exp(score - scores[predicted]), 0);
  return {
    predicted, rank, confidence,
    correct: predicted === target,
    top3: rank <= 3,
    absoluteGapError: Math.abs(predicted - target),
    negativeLogLikelihood: (scores[predicted] - scores[target]) + Math.log(1 / confidence),
  };
}

function emptyMetrics() {
  return { count: 0, correct: 0, top3: 0, reciprocalRank: 0, gapError: 0, loss: 0,
    balancedCorrect: 0, balanceChanges: 0, confidence: 0, unknownTokens: 0, tokens: 0 };
}

function accumulate(metrics, prediction, balanced, target, tokens) {
  metrics.count += 1;
  metrics.correct += prediction.correct;
  metrics.top3 += prediction.top3;
  metrics.reciprocalRank += 1 / prediction.rank;
  metrics.gapError += prediction.absoluteGapError;
  metrics.loss += prediction.negativeLogLikelihood;
  metrics.balancedCorrect += balanced === target;
  metrics.balanceChanges += balanced !== prediction.predicted;
  metrics.confidence += prediction.confidence;
  metrics.tokens += tokens.length;
  metrics.unknownTokens += tokens.filter((token) => !Object.hasOwn(modelData.vocabulary, token)).length;
}

function summarize(metrics) {
  const n = metrics.count;
  return {
    count: n, accuracy: metrics.correct / n, top3Accuracy: metrics.top3 / n,
    meanReciprocalRank: metrics.reciprocalRank / n, meanAbsoluteGapError: metrics.gapError / n,
    crossEntropy: metrics.loss / n, balanceOnlyAccuracy: metrics.balancedCorrect / n,
    balanceChangedChoiceRate: metrics.balanceChanges / n, meanConfidence: metrics.confidence / n,
    unknownTokenRate: metrics.unknownTokens / metrics.tokens,
  };
}

const lengthBucket = (n) => n <= 8 ? "2-8" : n <= 16 ? "9-16" : n <= 32 ? "17-32" : n <= 256 ? "33-256" : "257+";
const confidenceBucket = (p) => p < 0.5 ? "0-50%" : p < 0.9 ? "50-90%" : p < 0.99 ? "90-99%" : "99-100%";

function optionsFrom(args) {
  const options = {
    input: "training/data/processed/validation.jsonl", "per-domain": "500", seed: "20260911",
    cases: "training/evaluation-cases.json", output: "training/artifacts/model-baseline.json",
  };
  for (let i = 0; i < args.length; i += 2) {
    const key = args[i].replace(/^--/u, "");
    if (!args[i].startsWith("--") || !Object.hasOwn(options, key) || args[i + 1] === undefined) {
      throw new Error(`Unknown or incomplete option: ${args[i]}`);
    }
    options[key] = args[i + 1];
  }
  return options;
}

async function main() {
  const options = optionsFrom(process.argv.slice(2));
  const sampler = createDomainSampler(Number(options["per-domain"]), options.seed);
  const stream = createReadStream(resolve(project, options.input));
  const inputHash = createHash("sha256");
  stream.on("data", (data) => inputHash.update(data));
  for await (const line of createInterface({ input: stream, crlfDelay: Infinity })) {
    if (line.trim()) sampler.add(JSON.parse(line));
  }
  const { counts, records } = sampler.result();
  if (!records.length) throw new Error("No evaluation records found");
  console.log(`Selected ${records.length} samples: ${JSON.stringify(counts)} available by domain`);

  const overall = emptyMetrics();
  const groups = { domain: {}, length: {}, confidence: {} };
  const predictions = [];
  const timings = [];
  // Exclude lazy model decoding and initial JIT warm-up from per-example timings.
  for (let i = 0; i < 5; i++) backend.scoreTokens(Array.from("我一直在思考明天早上的早餐吃什么"));
  for (const record of records) {
    if (record.tokenization !== "character" || record.tokens.some((token) => !/^\p{Script=Han}$/u.test(token))) {
      throw new Error(`Expected individual Han-character tokens: ${record.id}`);
    }
    const started = performance.now();
    const scores = backend.scoreTokens(record.tokens);
    timings.push(performance.now() - started);
    const prediction = scorePrediction(scores, record.target_index);
    const balanced = chunker.selectBestBoundary(scores);
    accumulate(overall, prediction, balanced, record.target_index, record.tokens);
    for (const [kind, key] of [
      ["domain", record.domain], ["length", lengthBucket(record.tokens.length)],
      ["confidence", confidenceBucket(prediction.confidence)],
    ]) {
      groups[kind][key] ??= emptyMetrics();
      accumulate(groups[kind][key], prediction, balanced, record.target_index, record.tokens);
    }
    predictions.push({
      id: record.id, documentId: record.document_id, domain: record.domain,
      text: record.tokens.join(""), tokenCount: record.tokens.length,
      goldGap: record.target_index, predictedGap: prediction.predicted, balancedGap: balanced,
      goldRank: prediction.rank, confidence: prediction.confidence, punctuation: record.punctuation,
    });
    if (predictions.length % 500 === 0) console.log(`Scored ${predictions.length}/${records.length}`);
  }
  const casesBytes = readFileSync(resolve(project, options.cases));
  const cases = JSON.parse(casesBytes).map((item) => {
    const started = performance.now();
    const offsets = chunker.process(item.texts);
    const elapsedMs = performance.now() - started;
    return {
      ...item, offsets, elapsedMs,
      rendered: item.texts.map((text, index) => {
        let previous = 0;
        return [...offsets[index], text.length].map((end) => {
          const part = text.slice(previous, end); previous = end; return part;
        }).join("｜");
      }),
    };
  });
  timings.sort((a, b) => a - b);
  const perDomain = Object.fromEntries(Object.entries(groups.domain).map(([key, value]) => [key, summarize(value)]));
  const output = {
    createdAt: new Date().toISOString(),
    model: { ...backend.getModelInfo(),
      parameters: Object.values(modelData.tensors).reduce((sum, tensor) => sum + tensor.length, 0),
      sourceSha256: sha256(readFileSync(resolve(project, "src/boundary-model-data.js"))),
    },
    evaluation: {
      input: options.input, inputSha256: inputHash.digest("hex"), availableCounts: counts,
      seed: options.seed, perDomainLimit: Number(options["per-domain"]),
      sampledIdsSha256: sha256(predictions.map(({ id }) => id).join("\n")),
      target: "Recover the single removed punctuation boundary; not human-labeled reading chunks.",
      sampling: "Seeded reservoir per domain; aggregate is domain-balanced, not corpus-weighted.",
      balanceOnly: "Same full-input logits plus p*(1-p); no length limit or word protection. Not full pipeline accuracy.",
      tiePolicy: "Descending score, then ascending gap index for both prediction and rank.",
    },
    environment: { node: process.version, icu: process.versions.icu, platform: platform(), arch: arch(), cpu: cpus()[0]?.model },
    evaluatorSha256: sha256(readFileSync(fileURLToPath(import.meta.url))),
    inferenceSha256: sha256(readFileSync(resolve(project, "src/backend/inference.js"))),
    overall: summarize(overall), perDomain,
    macroDomainAccuracy: Object.values(perDomain).reduce((sum, metrics) => sum + metrics.accuracy, 0) / Object.keys(perDomain).length,
    perLength: Object.fromEntries(Object.entries(groups.length).map(([key, value]) => [key, summarize(value)])),
    perConfidence: Object.fromEntries(Object.entries(groups.confidence).map(([key, value]) => [key, summarize(value)])),
    latency: {
      note: "Warm Node.js scoreTokens only, full input, sequential; excludes DOM, messaging and chunking. Not Chrome end-to-end latency.",
      medianMs: timings[Math.floor(timings.length * 0.5)], p95Ms: timings[Math.min(timings.length - 1, Math.ceil(timings.length * 0.95) - 1)],
    },
    casesSha256: sha256(casesBytes), chunkerSha256: sha256(readFileSync(resolve(project, "src/backend/chunker.js"))),
    casesNote: "Unlabeled inspection examples; texts are separate DOM-node inputs. Outputs are observations, not gold labels.",
    cases, predictions,
  };
  const outputPath = resolve(project, options.output);
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, JSON.stringify(output, null, 2) + "\n");
  console.log(JSON.stringify({ output: outputPath, overall: output.overall, perDomain, latency: output.latency }, null, 2));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => { console.error(error); process.exitCode = 1; });
}
