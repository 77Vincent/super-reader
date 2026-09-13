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
  assert.equal(info.testAccuracy, null, "The interim epoch-1 export has not been evaluated on test data");
  assert.equal(info.tokenization, "character");
  assert.equal(info.candidatePositions, "between every adjacent Han character");
  assert.equal(info.vocabularySize, 4096);
  assert.equal(info.channels, 192);
  assert.equal(info.residualBlocks, 6);
  assert.equal(info.convolutionLayers, 12);
  assert.equal(info.checkpointSha256, reference.checkpointSha256);
});

test("browser inference matches PyTorch reference logits", () => {
  for (const example of reference.cases) {
    const scores = backend.scoreTokens(Array.from(example.text));
    assert.equal(scores.length, example.scores.length);
    scores.forEach((score, index) => {
      assert.ok(Math.abs(score - example.scores[index]) < 2e-5, `${example.text}, gap ${index}`);
    });
    assert.equal(scores.indexOf(Math.max(...scores)), example.bestGap);
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
