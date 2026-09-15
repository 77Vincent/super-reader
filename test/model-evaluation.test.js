const test = require("node:test");
const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const { createHash } = require("node:crypto");
const { copyFileSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } = require("node:fs");
const { tmpdir } = require("node:os");
const { dirname, join } = require("node:path");
const evaluator = import("../training/evaluate_model.mjs");

function evaluationFixture(t, records, summary) {
  const directory = realpathSync(mkdtempSync(join(tmpdir(), "reader-evaluation-standard-")));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const input = join(directory, "validation.jsonl");
  const bytes = records.map((record) => JSON.stringify(record)).join("\n") + "\n";
  writeFileSync(input, bytes);
  if (summary !== undefined) writeFileSync(join(directory, "summary.json"), JSON.stringify(summary));
  return { input, sha256: createHash("sha256").update(bytes).digest("hex"), directory };
}

const contextSummary = require("../training/text-policy.json");

test("evaluation uses regenerated list fragments and records the Unicode context standard", async (t) => {
  const { loadEvaluationSamples } = await evaluator;
  const { buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  const records = buildAdjacentSamples({
    id: "news:gold", domain: "news", text: "苹果、香蕉，准备做果汁。",
  });
  const fixture = evaluationFixture(t, records, contextSummary);
  writeFileSync(join(fixture.directory, "summary.json"), JSON.stringify({
    ...contextSummary, splits: { validation: { sha256: fixture.sha256 } },
  }));
  const result = await loadEvaluationSamples(fixture.input, 1);
  assert.equal(result.standard, "unicode-context-v1");
  assert.deepEqual(result.excludedProxyPunctuation, ["、", "：", ":"]);
  assert.equal(result.inputSha256, fixture.sha256);
  assert.match(result.summarySha256, /^[a-f0-9]{64}$/u);
  assert.equal(result.records.length, 1);
  assert.equal(result.records[0].tokens.join(""), "苹果、香蕉准备做果汁");
  assert.equal(result.records[0].target_index, 4);
});

test("evaluation rejects missing or legacy metadata even when rows have no enumeration labels", async (t) => {
  const { loadEvaluationSamples } = await evaluator;
  const records = [{ id: "legacy-filtered", domain: "news", punctuation: "，" }];
  for (const summary of [undefined, { tokenization: "character" },
    { tokenization: "character", excluded_proxy_punctuation: [] },
    { tokenization: "word", excluded_proxy_punctuation: ["、"] }]) {
    const fixture = evaluationFixture(t, records, summary);
    await assert.rejects(loadEvaluationSamples(fixture.input),
      summary === undefined ? /Missing evaluation metadata/u : /Data requires unicode-context-v1/u);
  }
});

test("evaluation validates proxy labels on rows outside the sampled reservoir", async (t) => {
  const { createDomainSampler, loadEvaluationSamples } = await evaluator;
  const records = Array.from({ length: 20 }, (_, index) => ({
    id: `row-${index}`, domain: "news", punctuation: "，",
  }));
  const sampler = createDomainSampler(1, "standard");
  records.forEach((record) => sampler.add(record));
  const selectedId = sampler.result().records[0].id;
  const excludedIndex = records.findIndex(({ id }) => id !== selectedId);
  for (const punctuation of ["、", "，、", ":", "：", "，:", undefined, ""]) {
    const modified = records.map((record, index) => index === excludedIndex ? { ...record, punctuation } : record);
    const fixture = evaluationFixture(t, modified, contextSummary);
    await assert.rejects(loadEvaluationSamples(fixture.input, 1, "standard"),
      punctuation ? /Invalid proxy label remains/u : /Missing punctuation proxy label/u);
  }
});

test("evaluation rejects a split that no longer matches its recorded checksum", async (t) => {
  const { loadEvaluationSamples } = await evaluator;
  const fixture = evaluationFixture(t, [{ id: "changed", domain: "news", punctuation: "，" }], {
    ...contextSummary, splits: { validation: { sha256: "0".repeat(64) } },
  });
  await assert.rejects(loadEvaluationSamples(fixture.input), /Evaluation split checksum mismatch/u);
});

test("default evaluation selects the corrected holdout even when a legacy generic split exists", (t) => {
  const records = [{
    id: "news:tiny", domain: "news", document_id: "news:tiny", tokenization: "character",
    tokens: ["甲", "乙"], target_index: 0, punctuation: "，",
  }];
  const { directory } = evaluationFixture(t, records, contextSummary);
  const dataDirectory = join(directory, "training/data/processed/unicode-context-192ch-12conv-20260913-eval");
  mkdirSync(dataDirectory, { recursive: true });
  for (const name of ["validation.jsonl", "summary.json"]) {
    copyFileSync(join(directory, name), join(dataDirectory, name));
  }
  writeFileSync(join(dirname(dataDirectory), "validation.jsonl"), JSON.stringify({
    ...records[0], id: "news:legacy", punctuation: "、",
  }) + "\n");
  writeFileSync(join(dirname(dataDirectory), "summary.json"), JSON.stringify({ tokenization: "character" }));
  for (const name of ["training/evaluate_model.mjs", "training/text_policy.mjs", "training/text-policy.json", "src/backend/inference.js",
    "src/backend/chunker.js", "src/boundary-model-data.js"]) {
    const destination = join(directory, name);
    mkdirSync(dirname(destination), { recursive: true });
    copyFileSync(join(__dirname, "..", name), destination);
  }
  const candidate = join(directory, "training/artifacts/unicode-context-192ch-12conv-20260913/candidate/boundary-model-data.js");
  mkdirSync(dirname(candidate), { recursive: true });
  writeFileSync(candidate, readFileSync(join(directory, "src/boundary-model-data.js"), "utf8")
    .replace('"tokenization":"character"', '"inputRepresentation":"unicode-context-v1","tokenization":"character"'));
  writeFileSync(join(directory, "training/evaluation-cases.json"), "[]");
  execFileSync(process.execPath, [join(directory, "training/evaluate_model.mjs")], {
    cwd: directory, stdio: "pipe", timeout: 30000,
  });
  const report = JSON.parse(readFileSync(join(directory, "training/artifacts/model-baseline.json"), "utf8"));
  assert.equal(report.evaluation.standard, "unicode-context-v1");
  assert.deepEqual(report.evaluation.excludedProxyPunctuation, ["、", "：", ":"]);
  assert.equal(report.evaluation.input, "training/data/processed/unicode-context-192ch-12conv-20260913-eval/validation.jsonl");
  assert.equal(report.overall.count, 1);
  assert.equal(report.overall.accuracy, 1);
  assert.equal(Object.hasOwn(report.overall, "balanceOnlyAccuracy"), false);
  assert.equal(Object.hasOwn(report.overall, "balanceChangedChoiceRate"), false);
  assert.equal(report.predictions[0].id, "news:tiny");
  assert.equal(report.predictions[0].goldGap, 0);
  assert.equal(report.predictions[0].predictedGap, 0);
  assert.equal(Object.hasOwn(report.predictions[0], "balancedGap"), false);
});

test("domain sampling is reproducible, bounded and independent of other domains", async () => {
  const { createDomainSampler } = await evaluator;
  const sample = (withOtherDomain) => {
    const sampler = createDomainSampler(5, "baseline");
    for (let i = 0; i < 100; i++) {
      sampler.add({ id: `news-${i}`, domain: "news" });
      if (withOtherDomain) sampler.add({ id: `wiki-${i}`, domain: "wiki" });
    }
    sampler.add({ id: "rare-1", domain: "rare" });
    return sampler.result();
  };
  const result = sample(true);
  assert.deepEqual(result, sample(true));
  assert.deepEqual(result.counts, { news: 100, rare: 1, wiki: 100 });
  assert.equal(result.records.length, 11);
  assert.equal(new Set(result.records.map(({ id }) => id)).size, 11);
  assert.deepEqual(result.records.filter(({ domain }) => domain === "news"),
    sample(false).records.filter(({ domain }) => domain === "news"));
  assert.ok(result.records.some(({ id }) => id.startsWith("news-") && Number(id.slice(5)) >= 5));
  assert.throws(() => createDomainSampler(0, "baseline"));
});

test("boundary metrics use stable softmax and a consistent rank for tied logits", async () => {
  const { scorePrediction } = await evaluator;
  const prediction = scorePrediction([-1000, -1000, -1000, -1000], 3);
  assert.equal(prediction.predicted, 0);
  assert.equal(prediction.rank, 4);
  assert.equal(prediction.top3, false);
  assert.equal(prediction.correct, false);
  assert.equal(prediction.confidence, 0.25);
  assert.equal(prediction.absoluteGapError, 3);
  assert.equal(prediction.negativeLogLikelihood, Math.log(4));
  const certain = scorePrediction([-1000, 1000], 1);
  assert.equal(certain.correct, true);
  assert.equal(certain.rank, 1);
  assert.equal(certain.confidence, 1);
  assert.equal(certain.negativeLogLikelihood, 0);
  assert.throws(() => scorePrediction([NaN], 0));
  assert.throws(() => scorePrediction([1, 2], 2));
});
