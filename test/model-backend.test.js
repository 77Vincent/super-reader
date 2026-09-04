const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const backend = require("../src/model-backend.js");

const projectRoot = join(__dirname, "..");

test("browser backend loads the exported best checkpoint", () => {
  const info = backend.getModelInfo();

  assert.equal(info.bestEpoch, 2);
  assert.equal(info.testAccuracy, 0.530125);
  assert.equal(info.tokenization, "character");
  assert.equal(info.candidatePositions, "between every adjacent Han character");
  assert.equal(info.vocabularySize, 4096);
  assert.match(info.checkpointSha256, /^[a-f0-9]{64}$/u);
});

test("browser inference matches tinygrad reference logits", () => {
  const tokens = Array.from("我一直在思考明天早上的早餐吃什么");
  const expected = [
    -1.0187765,
    -1.9855889,
    -1.5573953,
    -1.6147989,
    -0.5052901,
    2.0354855,
    1.3623402,
    1.2437651,
    -1.8649274,
    -2.0338097,
    -0.6719301,
    -2.7048666,
    1.5034223,
    -1.034253,
    -2.5506806,
  ];
  const scores = backend.scoreTokens(tokens);

  assert.equal(scores.length, expected.length);
  scores.forEach((score, index) => {
    assert.ok(Math.abs(score - expected[index]) < 2e-5);
  });
  assert.equal(scores.indexOf(Math.max(...scores)), 5);
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
