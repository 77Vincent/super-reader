const test = require("node:test");
const assert = require("node:assert/strict");
const evaluator = import("../training/evaluate_model.mjs");

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
