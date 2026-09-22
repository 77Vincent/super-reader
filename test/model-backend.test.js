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
  assert.equal(info.testAccuracy, 0.8889453524441283);
  assert.equal(info.tokenization, "character");
  assert.equal(info.inputRepresentation, "unicode-context-v1");
  assert.equal(info.candidatePositions, "between every adjacent Unicode code point");
  assert.equal(info.vocabularySize, 8192);
  assert.equal(info.channels, 192);
  assert.equal(info.residualBlocks, 8);
  assert.equal(info.convolutionLayers, 16);
  assert.equal(info.checkpointSha256, reference.checkpointSha256);
});

test("browser inference matches PyTorch reference logits", () => {
  for (const example of reference.cases) {
    const scores = backend.scoreTokens(Array.from(example.text));
    assert.equal(scores.length, example.scores.length);
    scores.forEach((score, index) => {
      // Metal and JS accumulate float32 operations differently; relative error
      // Relative error stays below 4e-6 for this checkpoint, including large
      // mixed-number logits; probability and best-gap checks remain separate.
      const tolerance = 2e-5 + 4e-6 * Math.abs(example.scores[index]);
      assert.ok(Math.abs(score - example.scores[index]) < tolerance, `${example.text}, gap ${index}`);
    });
    assert.equal(scores.indexOf(Math.max(...scores)), example.bestGap);
    const softmax = (values) => {
      const exponentials = values.map((value) => Math.exp(value - Math.max(...values)));
      const sum = exponentials.reduce((a, b) => a + b, 0);
      return exponentials.map((value) => value / sum);
    };
    const probabilities = softmax(scores);
    softmax(example.scores).forEach((probability, index) => {
      assert.ok(Math.abs(probabilities[index] - probability) < 1e-5, `${example.text}, probability ${index}`);
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
