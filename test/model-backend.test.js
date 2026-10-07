const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const backend = require("../src/backend/inference.js");
const reference = require("./model-backend-reference.json");

const projectRoot = join(__dirname, "..");

test("browser backend loads the exported best checkpoint", () => {
  const info = backend.getModelInfo();

  assert.equal(info.bestEpoch, 1);
  assert.equal(info.testAccuracy, 0.9068468088924256);
  assert.equal(info.tokenization, "character");
  assert.equal(info.inputRepresentation, "unicode-context-v1");
  assert.equal(info.candidatePositions, "between every adjacent Unicode code point");
  assert.equal(info.vocabularySize, 8192);
  assert.equal(info.channels, 192);
  assert.equal(info.residualBlocks, 8);
  assert.equal(info.convolutionLayers, 16);
  assert.equal(info.checkpointSha256, reference.checkpointSha256);
});

test("browser inference matches independent PyTorch CPU and Metal references", () => {
  for (const example of reference.cases) {
    const scores = backend.scoreTokens(Array.from(example.text));
    assert.equal(scores.length, example.scores.length);
    scores.forEach((score, index) => {
      // Compare to CPU float64 using the same stored float32 weights. This
      // avoids Metal's reduction-order cancellation near zero; tolerances
      // remain unchanged. Check actual Metal probabilities and top-1 below.
      const tolerance = 2e-5 + 4e-6 * Math.abs(example.scores[index]);
      assert.ok(Math.abs(score - example.scores[index]) < tolerance, `${example.text}, gap ${index}`);
    });
    assert.equal(scores.indexOf(Math.max(...scores)), example.bestGap);
    assert.equal(example.bestGap, example.mpsBestGap);
    assert.equal(example.mpsScores.length, scores.length);
    const softmax = (values) => {
      const exponentials = values.map((value) => Math.exp(value - Math.max(...values)));
      const sum = exponentials.reduce((a, b) => a + b, 0);
      return exponentials.map((value) => value / sum);
    };
    const probabilities = softmax(scores);
    softmax(example.scores).forEach((probability, index) => {
      assert.ok(Math.abs(probabilities[index] - probability) < 1e-5, `${example.text}, probability ${index}`);
    });
    softmax(example.mpsScores).forEach((probability, index) => {
      assert.ok(Math.abs(probabilities[index] - probability) < 1e-5, `${example.text}, Metal probability ${index}`);
    });
  }
});

test("runtime inference contains no handwritten Chinese lexical rules", () => {
  const implementation = ["src/backend/chunker.js", "src/backend/inference.js"]
    .map((path) => readFileSync(join(projectRoot, path), "utf8"))
    .join("\n");

  assert.doesNotMatch(implementation, /\p{Script=Han}/u);
  assert.doesNotMatch(
    implementation,
    /lexicon|dictionary|wordOverrides|buildStableUnits|shouldMerge/iu,
  );
});
