const test = require("node:test");
const assert = require("node:assert/strict");
const {
  buildVisualChunks,
  chunkText,
  chunkTextByClause,
  splitUnderlineRuns,
  splitClauses,
  visualLength,
} = require("../src/chunker.js");

function createFixedSegmenter(singletons) {
  return {
    segment() {
      return [
        { segment: "系统", isWordLike: true },
        { segment: "能够", isWordLike: true },
        ...singletons.map((segment) => ({ segment, isWordLike: true })),
        { segment: "稳定", isWordLike: true },
        { segment: "运行", isWordLike: true },
        { segment: "。", isWordLike: false },
      ];
    },
  };
}

test("combines Chinese words into readable chunks without splitting words", () => {
  const chunks = chunkText("我一直在思考明天早上的早餐吃什么", { targetLength: 7 });

  assert.deepEqual(chunks, ["我一直在思考", "明天早上的早餐", "吃什么"]);
});

test("keeps strong punctuation with the preceding phrase", () => {
  const text = "今天下雨了。我们明天再出发！";
  const chunks = chunkText(text, { targetLength: 7 });

  assert.equal(chunks.join(""), text);
  assert.ok(chunks.includes("今天下雨了。"));
  assert.equal(chunks.at(-1), "我们明天再出发！");
});

test("splits at punctuation before making visual chunks", () => {
  const text = "在左侧输入一段中文，看看它如何被重新组织。";
  const clauses = chunkTextByClause(text, { targetLength: 7 });

  assert.deepEqual(clauses, [
    ["在左侧输入一段中文，"],
    ["看看它如何被", "重新组织。"],
  ]);
  assert.equal(clauses.flat().join(""), text);
});

test("does not leave a single word stranded before punctuation", () => {
  const chunks = chunkText("在左侧输入一段中文，", { targetLength: 7 });

  assert.deepEqual(chunks, ["在左侧输入一段中文，"]);
  assert.ok(!chunks.some((chunk) => chunk === "中文，"));
});

test("keeps closing quotes with the punctuation-delimited clause", () => {
  assert.deepEqual(splitClauses("他说：“明天见。”然后离开。"), [
    "他说：",
    "“明天见。”",
    "然后离开。",
  ]);
});

test("treats paired double quotes as structural boundaries", () => {
  const text = "必须在“连续单字合并”之前执行。";

  assert.deepEqual(splitClauses(text), [
    "必须在",
    "“连续单字合并”",
    "之前执行。",
  ]);
  assert.deepEqual(chunkTextByClause(text, { targetLength: 2 }), [
    ["必须在"],
    ["“连续", "单字", "合并”"],
    ["之前", "执行。"],
  ]);
});

test("treats straight double quotes as structural boundaries", () => {
  assert.deepEqual(splitClauses('在"quoted words"之后。'), [
    "在",
    '"quoted words"',
    "之后。",
  ]);
});

test("treats enumeration commas as clause boundaries", () => {
  assert.deepEqual(splitClauses("清晰、效率与视觉愉悦。"), [
    "清晰、",
    "效率与视觉愉悦。",
  ]);
});

test("punctuation ends one visual block before alternation continues", () => {
  const text = "我们需要对齐模型的输入结构。如果可以的话，请告诉我。";
  const chunks = buildVisualChunks(text, { targetLength: 7 });
  const sentenceEndIndex = chunks.findIndex((chunk) => chunk.text.endsWith("。"));

  assert.ok(sentenceEndIndex >= 0);
  assert.ok(sentenceEndIndex < chunks.length - 1);
  assert.equal(chunks.some((chunk) => chunk.text.includes("。如果")), false);
  assert.notEqual(
    chunks[sentenceEndIndex].underlined,
    chunks[sentenceEndIndex + 1].underlined,
  );
});

test("merges short singleton runs between stable word boundaries", () => {
  const text = "阅读不是更快的速度，而是获得更好的体验。";
  const clauses = chunkTextByClause(text, { targetLength: 5 });
  const boundaries = clauses.flat().join("|");

  assert.equal(clauses.flat().join(""), text);
  assert.doesNotMatch(boundaries, /更\|快|快\|的/u);
  assert.ok(boundaries.includes("更快的"));
});

test("does not rely on a vocabulary list for singleton-run merging", () => {
  const segmenter = createFixedSegmenter(["甲", "乙", "丙"]);
  const boundaries = chunkText("系统能够甲乙丙稳定运行。", {
    targetLength: 3,
    segmenter,
  }).join("|");

  assert.doesNotMatch(boundaries, /甲\|乙|乙\|丙/u);
  assert.ok(boundaries.includes("甲乙丙"));
});

test("keeps four-or-more singleton tokens available for normal chunking", () => {
  const segmenter = createFixedSegmenter(["甲", "乙", "丙", "丁"]);
  const boundaries = chunkText("系统能够甲乙丙丁稳定运行。", {
    targetLength: 3,
    segmenter,
  }).join("|");

  assert.match(boundaries, /[甲乙丙丁]\|[甲乙丙丁]/u);
});

test("supports a two-character minimum target without merging valid words", () => {
  const text = "春天来了我们出发。";
  const segmenter = {
    segment() {
      return [
        { segment: "春天", isWordLike: true },
        { segment: "来了", isWordLike: true },
        { segment: "我们", isWordLike: true },
        { segment: "出发", isWordLike: true },
        { segment: "。", isWordLike: false },
      ];
    },
  };

  assert.deepEqual(chunkText(text, { targetLength: 2, segmenter }), [
    "春天",
    "来了",
    "我们",
    "出发。",
  ]);
  assert.deepEqual(
    chunkText(text, { targetLength: 1, segmenter }),
    chunkText(text, { targetLength: 2, segmenter }),
  );
});

test("never loses whitespace or mixed-language content", () => {
  const text = "这是 Super Reader 的 2.0 版本，很好用。";
  const chunks = chunkText(text, { targetLength: 6 });

  assert.equal(chunks.join(""), text);
});

test("visual length counts letters and numbers but ignores punctuation", () => {
  assert.equal(visualLength("你好，Reader 2.0！"), 10);
});

test("underline runs exclude punctuation, symbols, and whitespace", () => {
  assert.deepEqual(splitUnderlineRuns("你好，Reader 2.0！"), [
    { text: "你好", underlinable: true },
    { text: "，", underlinable: false },
    { text: "Reader", underlinable: true },
    { text: " ", underlinable: false },
    { text: "2", underlinable: true },
    { text: ".", underlinable: false },
    { text: "0", underlinable: true },
    { text: "！", underlinable: false },
  ]);
});
