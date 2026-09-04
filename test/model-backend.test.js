const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const backend = require("../src/model-backend.js");

const projectRoot = join(__dirname, "..");

test("browser backend loads the exported best checkpoint", () => {
  const info = backend.getModelInfo();

  assert.equal(info.bestEpoch, 3);
  assert.equal(info.testAccuracy, 0.55875);
  assert.equal(info.vocabularySize, 4096);
  assert.match(info.checkpointSha256, /^[a-f0-9]{64}$/u);
});

test("browser inference matches tinygrad reference logits", () => {
  const tokens = ["我", "一直", "在", "思考", "明天", "早上", "的", "早餐", "吃", "什么"];
  const expected = [
    -2.4880805,
    -2.3944204,
    -6.6517124,
    2.5348675,
    -0.20889784,
    -6.0269656,
    -4.211154,
    0.6422689,
    -3.3894043,
  ];
  const scores = backend.scoreTokens(tokens);

  assert.equal(scores.length, expected.length);
  scores.forEach((score, index) => {
    assert.ok(Math.abs(score - expected[index]) < 2e-5);
  });
  assert.equal(scores.indexOf(Math.max(...scores)), 3);
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
