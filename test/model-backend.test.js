const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const backend = require("../src/model-backend.js");

const projectRoot = join(__dirname, "..");

test("browser backend loads the exported best checkpoint", () => {
  const info = backend.getModelInfo();

  assert.equal(info.bestEpoch, 3);
  assert.equal(info.testAccuracy, 0.802323742375865);
  assert.equal(info.tokenization, "character");
  assert.equal(info.candidatePositions, "between every adjacent Han character");
  assert.equal(info.vocabularySize, 4096);
  assert.equal(info.channels, 128);
  assert.equal(info.residualBlocks, 4);
  assert.equal(info.convolutionLayers, 8);
  assert.match(info.checkpointSha256, /^[a-f0-9]{64}$/u);
});

test("browser inference matches PyTorch reference logits", () => {
  const tokens = Array.from("我一直在思考明天早上的早餐吃什么");
  const expected = [
    -36.6672058,
    -40.0465393,
    -37.313324,
    -37.4124794,
    -35.6227837,
    -33.1555634,
    -33.9455643,
    -36.1302299,
    -37.6914749,
    -37.2810211,
    -35.4790611,
    -36.3977318,
    -32.6919632,
    -35.0108185,
    -37.7038155,
  ];
  const scores = backend.scoreTokens(tokens);

  assert.equal(scores.length, expected.length);
  scores.forEach((score, index) => {
    assert.ok(Math.abs(score - expected[index]) < 2e-5);
  });
  assert.equal(scores.indexOf(Math.max(...scores)), 12);
});

test("runtime inference contains no handwritten Chinese lexical rules", () => {
  const implementation = ["src/chunker.js", "src/model-backend.js"]
    .map((path) => readFileSync(join(projectRoot, path), "utf8"))
    .join("\n");

  assert.doesNotMatch(implementation, /\p{Script=Han}/u);
  assert.doesNotMatch(
    implementation,
    /lexicon|dictionary|wordOverrides|buildStableUnits|shouldMerge/iu,
  );
});
