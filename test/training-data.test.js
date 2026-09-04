const test = require("node:test");
const assert = require("node:assert/strict");

test("training pairs use adjacent punctuation fragments with one target gap", async () => {
  const { buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  const samples = buildAdjacentSamples({
    id: "fixture:1",
    domain: "fixture",
    text: "今天下雨，我们留在家里；明天再出去。",
  });

  assert.equal(samples.length, 2);
  assert.equal(samples[0].tokens.join(""), "今天下雨我们留在家里");
  assert.equal(samples[0].target_index, 1);
  assert.equal(samples[0].tokens[samples[0].target_index], "下雨");
  assert.equal(samples[0].tokens[samples[0].target_index + 1], "我们");
  assert.equal(samples[1].document_id, samples[0].document_id);
});

test("training input removes all punctuation while keeping the source signal", async () => {
  const { buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  const [sample] = buildAdjacentSamples({
    id: "fixture:2",
    domain: "fixture",
    text: "模型只看文字，“标点”不会泄露。",
  });

  assert.equal(sample.punctuation, ",");
  assert.doesNotMatch(sample.tokens.join(""), /[，。“”]/u);
});

test("each side allows 32 content characters and rejects 33 without cropping", async () => {
  const { buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  const accepted = buildAdjacentSamples({
    id: "fixture:limit:32",
    domain: "fixture",
    text: `${"甲".repeat(32)}，好。`,
  });
  const rejected = buildAdjacentSamples({
    id: "fixture:limit:33",
    domain: "fixture",
    text: `${"甲".repeat(33)}，好。`,
  });

  assert.equal(accepted.length, 1);
  assert.equal(accepted[0].left_character_length, 32);
  assert.equal(accepted[0].right_character_length, 1);
  assert.equal(accepted[0].tokens.join(""), `${"甲".repeat(32)}好`);
  assert.equal(rejected.length, 0);
});

test("one-character fragments remain valid training sides", async () => {
  const { buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  const [sample] = buildAdjacentSamples({
    id: "fixture:minimum",
    domain: "fixture",
    text: "好，走。",
  });

  assert.deepEqual(sample.tokens, ["好", "走"]);
  assert.equal(sample.target_index, 0);
  assert.equal(sample.left_character_length, 1);
  assert.equal(sample.right_character_length, 1);
});
