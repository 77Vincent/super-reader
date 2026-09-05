const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const backend = require("../src/model-backend.js");

const projectRoot = join(__dirname, "..");

test("browser backend loads the exported best checkpoint", () => {
  const info = backend.getModelInfo();

  assert.equal(info.bestEpoch, 2);
  assert.equal(info.testAccuracy, 0.6041125029543843);
  assert.equal(info.tokenization, "character");
  assert.equal(info.candidatePositions, "between every adjacent Han character");
  assert.equal(info.vocabularySize, 4096);
  assert.equal(info.channels, 48);
  assert.equal(info.residualBlocks, 3);
  assert.equal(info.convolutionLayers, 6);
  assert.match(info.checkpointSha256, /^[a-f0-9]{64}$/u);
});

test("browser inference matches tinygrad reference logits", () => {
  const tokens = Array.from("我一直在思考明天早上的早餐吃什么");
  const expected = [
    -4.9833841,
    -3.9611714,
    -1.3251973,
    -3.8059237,
    -0.141679,
    0.7641446,
    0.2002495,
    0.0686523,
    -4.1940913,
    -2.7289252,
    -0.3707583,
    -3.0019574,
    1.742052,
    -2.3135548,
    -5.555111,
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
