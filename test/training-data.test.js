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
  assert.equal(samples[0].target_index, 3);
  assert.equal(samples[0].tokens[samples[0].target_index], "雨");
  assert.equal(samples[0].tokens[samples[0].target_index + 1], "我");
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

test("training sides have no character-length ceiling", async () => {
  const { buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  const [sample] = buildAdjacentSamples({
    id: "fixture:no-limit",
    domain: "fixture",
    text: `${"甲".repeat(64)}，好。`,
  });

  assert.equal(sample.left_character_length, 64);
  assert.equal(sample.right_character_length, 1);
  assert.equal(sample.tokens.join(""), `${"甲".repeat(64)}好`);
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

test("training selection balances relative boundary-position buckets", async () => {
  const { selectPositionBalancedSamples } = await import(
    "../training/prepare_smoke_data.mjs"
  );
  const candidates = Array.from({ length: 10 }, (_, bin) => (
    Array.from({ length: 4 }, (_, index) => ({
      id: `fixture:bin:${bin}:${index}`,
      relative_boundary_position: (bin + 0.5) / 10,
    }))
  )).flat();
  const selected = selectPositionBalancedSamples(candidates, 30, "fixture", 10);
  const histogram = Array(10).fill(0);
  for (const sample of selected) {
    const bin = Math.min(9, Math.floor(sample.relative_boundary_position * 10));
    histogram[bin] += 1;
  }

  assert.deepEqual(histogram, Array(10).fill(3));
});

test("comparison modes keep the same text and gold boundary but change candidates", async () => {
  const { buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  const document = {
    id: "fixture:comparison",
    domain: "fixture",
    text: "阅读不是更快地扫过文字，模型开始工作。",
  };
  const [word] = buildAdjacentSamples(document, { tokenization: "word" });
  const [character] = buildAdjacentSamples(document, { tokenization: "character" });

  assert.equal(word.id, character.id);
  assert.equal(word.tokens.join(""), character.tokens.join(""));
  assert.equal(
    word.tokens.slice(0, word.target_index + 1).join(""),
    character.tokens.slice(0, character.target_index + 1).join(""),
  );
  assert.ok(word.tokens.some((token) => token.length > 1));
  assert.ok(character.tokens.every((token) => Array.from(token).length === 1));
  assert.ok(character.tokens.length > word.tokens.length);
});

test("both comparison modes exclude non-Chinese tokens", async () => {
  const { buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  const document = {
    id: "fixture:han-only",
    domain: "fixture",
    text: "Synthetic 中文 20，AI 模型。",
  };

  for (const tokenization of ["word", "character"]) {
    const [sample] = buildAdjacentSamples(document, { tokenization });
    assert.equal(sample.tokens.join(""), "中文模型");
    assert.ok(sample.tokens.every((token) => /^\p{Script=Han}+$/u.test(token)));
  }
});
