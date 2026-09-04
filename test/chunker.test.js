const test = require("node:test");
const assert = require("node:assert/strict");
const {
  boundaryFallsInsideWord,
  buildVisualChunks,
  chunkText,
  chunkTextByClause,
  selectBestBoundary,
  splitClauses,
  tokenizeHanCharacters,
  visualLength,
} = require("../src/chunker.js");

test("model chunks preserve the complete source text", () => {
  const text = "我一直在思考明天早上的早餐吃什么";
  assert.equal(chunkText(text).join(""), text);
});

test("punctuation creates hard clause boundaries before model recursion", () => {
  const text = "我们需要对齐模型的输入结构。如果可以的话，请告诉我。";
  const expectedClauses = [
    "我们需要对齐模型的输入结构。",
    "如果可以的话，",
    "请告诉我。",
  ];
  const modelChunksByClause = chunkTextByClause(text);

  assert.deepEqual(splitClauses(text), expectedClauses);
  assert.deepEqual(
    modelChunksByClause.map((chunks) => chunks.join("")),
    expectedClauses,
  );
  assert.equal(modelChunksByClause.flat().join(""), text);
});

test("a model chunk never crosses a punctuation boundary", () => {
  const chunks = chunkText("输入结构。如果可以的话，请告诉我。");

  assert.equal(chunks.some((chunk) => chunk.includes("。如果")), false);
  assert.equal(chunks.some((chunk) => chunk.includes("，请")), false);
});

test("keeps closing quotes with the punctuation-delimited clause", () => {
  assert.deepEqual(splitClauses("他说：“明天见。”然后离开。"), [
    "他说：",
    "“明天见。”",
    "然后离开。",
  ]);
});

test("treats paired double quotes as hard structural boundaries", () => {
  const text = "必须在“连续单字合并”之前执行。";
  const expectedClauses = ["必须在", "“连续单字合并”", "之前执行。"];
  const modelChunksByClause = chunkTextByClause(text);

  assert.deepEqual(splitClauses(text), expectedClauses);
  assert.deepEqual(
    modelChunksByClause.map((chunks) => chunks.join("")),
    expectedClauses,
  );
});

test("treats straight double quotes as hard structural boundaries", () => {
  assert.deepEqual(splitClauses('在"quoted words"之后。'), [
    "在",
    '"quoted words"',
    "之后。",
  ]);
});

test("treats punctuation and newlines as hard boundaries", () => {
  assert.deepEqual(splitClauses("清晰、效率，稳定；自然：继续\n结束。"), [
    "清晰、",
    "效率，",
    "稳定；",
    "自然：",
    "继续\n",
    "结束。",
  ]);
});

test("always selects the model's highest-scoring boundary without a threshold", () => {
  assert.equal(selectBestBoundary([]), null);
  assert.equal(selectBestBoundary([0, 0]), 0);
  assert.equal(selectBestBoundary([0, 2, 0]), 1);
  assert.equal(selectBestBoundary([0.01, 0]), 0);
});

test("selects the highest-scoring allowed boundary", () => {
  assert.equal(selectBestBoundary([1, 3, 2], (index) => index !== 1), 2);
  assert.equal(selectBestBoundary([1, 3, 2], () => false), null);
});

test("only asks the model to split clauses longer than seven Han characters", () => {
  assert.deepEqual(chunkText("甲乙丙丁戊己庚", { segmenter: null }), [
    "甲乙丙丁戊己庚",
  ]);
  assert.deepEqual(chunkText("春天来了我们出发", { segmenter: null }), [
    "春天来了",
    "我们出发",
  ]);
});

test("punctuation-delimited clauses of seven characters or fewer stay intact", () => {
  assert.deepEqual(chunkText("甲乙丙丁，戊己庚辛。", { segmenter: null }), [
    "甲乙丙丁，",
    "戊己庚辛。",
  ]);
});

test("continues recursively when the remainder after a split still exceeds seven characters", () => {
  const chunks = chunkText("而是帮助大脑更快地识别信息结构。");

  assert.deepEqual(chunks, ["而是", "帮助大脑更快地", "识别信息结构。"]);
  assert.ok(chunks.every((chunk) => visualLength(chunk) <= 7));
});

test("detects a predicted boundary strictly inside a Segmenter word", () => {
  const segmenter = {
    segment() {
      return [{ segment: "春天", index: 0, isWordLike: true }];
    },
  };

  assert.equal(boundaryFallsInsideWord("春天", 1, segmenter), true);
  assert.equal(boundaryFallsInsideWord("春天", 2, segmenter), false);
});

test("keeps a clause intact when every model boundary is inside a word", () => {
  const text = "春天来了我们出发";
  const wholeWordSegmenter = {
    segment() {
      return [{ segment: text, index: 0, isWordLike: true }];
    },
  };

  assert.deepEqual(chunkText(text, { segmenter: wholeWordSegmenter }), [text]);
});

test("falls back to a valid model boundary when the winner is inside a word", () => {
  const text = "同一个无标点子句内的短语块用细竖线分隔；";
  const chunks = chunkText(text);

  assert.ok(chunks.length > 1);
  assert.equal(chunks.join(""), text);
  assert.equal(chunks.some((chunk) => chunk.endsWith("分")), false);
});

test("Segmenter is only a guard and the model can run without it", () => {
  assert.deepEqual(chunkText("春天来了我们出发", { segmenter: null }), [
    "春天来了",
    "我们出发",
  ]);
});

test("visual chunks preserve punctuation-delimited boundaries", () => {
  const text = "我们需要对齐模型的输入结构。如果可以的话，请告诉我。";
  const chunks = buildVisualChunks(text);
  const sentenceEndIndex = chunks.findIndex((chunk) => chunk.text.endsWith("。"));

  assert.ok(sentenceEndIndex >= 0);
  assert.ok(sentenceEndIndex < chunks.length - 1);
  assert.equal(chunks[sentenceEndIndex].processed, true);
  assert.equal(chunks[sentenceEndIndex + 1].processed, true);
  assert.equal(chunks[sentenceEndIndex + 1].separated, false);
  assert.equal(chunks.some((chunk) => "underlined" in chunk), false);
});

test("vertical separators appear only at model boundaries inside one clause", () => {
  assert.deepEqual(buildVisualChunks("春天来了我们出发。"), [
    { text: "春天来了", processed: true, separated: false },
    { text: "我们出发。", processed: true, separated: true },
  ]);
  assert.deepEqual(buildVisualChunks("春天来了，我们出发。"), [
    { text: "春天来了，", processed: true, separated: false },
    { text: "我们出发。", processed: true, separated: false },
  ]);
});

test("character tokenization keeps original UTF-16 source offsets", () => {
  assert.deepEqual(tokenizeHanCharacters("A中😀文"), [
    { segment: "中", index: 1 },
    { segment: "文", index: 4 },
  ]);
});

test("never loses whitespace or mixed-language content", () => {
  const text = "这是 Super Reader 的 2.0 版本，很好用。";
  assert.equal(chunkText(text).join(""), text);
});

test("visual length counts only Chinese characters", () => {
  assert.equal(visualLength("你好，Reader 2.0！"), 2);
});

test("non-Chinese chunks remain unprocessed", () => {
  const text = "中文。Synthetic English.汉字。";
  const chunks = buildVisualChunks(text);
  const chineseChunks = chunks.filter((chunk) => chunk.processed);
  const englishChunk = chunks.find((chunk) => chunk.text.includes("Synthetic"));

  assert.equal(chineseChunks.length, 2);
  assert.equal(englishChunk.processed, false);
  assert.equal(chunks.some((chunk) => "underlined" in chunk), false);
  assert.equal(chunks.map((chunk) => chunk.text).join(""), text);
});
