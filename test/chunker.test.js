const test = require("node:test");
const assert = require("node:assert/strict");
const {
  boundaryFallsInsideWord,
  buildVisualChunks,
  chunkText,
  chunkTextByClause,
  selectConfidentBoundary,
  splitUnderlineRuns,
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

test("accepts only a uniquely high-confidence softmax winner", () => {
  assert.equal(selectConfidentBoundary([0, 0]), null);

  const prediction = selectConfidentBoundary([0, 2, 0]);
  assert.equal(prediction.index, 1);
  assert.ok(prediction.confidence > 0.6);

  assert.equal(selectConfidentBoundary([Math.log(1.5), 0]), null);
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

test("detects a predicted boundary strictly inside a Segmenter word", () => {
  const segmenter = {
    segment() {
      return [{ segment: "春天", index: 0, isWordLike: true }];
    },
  };

  assert.equal(boundaryFallsInsideWord("春天", 1, segmenter), true);
  assert.equal(boundaryFallsInsideWord("春天", 2, segmenter), false);
});

test("abandons a model split when its winning boundary is inside a word", () => {
  const text = "春天来了我们出发";
  const wholeWordSegmenter = {
    segment() {
      return [{ segment: text, index: 0, isWordLike: true }];
    },
  };

  assert.deepEqual(chunkText(text, { segmenter: wholeWordSegmenter }), [text]);
});

test("Segmenter is only a guard and the model can run without it", () => {
  assert.deepEqual(chunkText("春天来了我们出发", { segmenter: null }), [
    "春天来了",
    "我们出发",
  ]);
});

test("visual alternation continues across punctuation-delimited clauses", () => {
  const text = "我们需要对齐模型的输入结构。如果可以的话，请告诉我。";
  const chunks = buildVisualChunks(text);
  const sentenceEndIndex = chunks.findIndex((chunk) => chunk.text.endsWith("。"));

  assert.ok(sentenceEndIndex >= 0);
  assert.ok(sentenceEndIndex < chunks.length - 1);
  assert.notEqual(
    chunks[sentenceEndIndex].underlined,
    chunks[sentenceEndIndex + 1].underlined,
  );
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

test("underline runs exclude all non-Chinese content", () => {
  assert.deepEqual(splitUnderlineRuns("你好，Reader 2.0！世界"), [
    { text: "你好", underlinable: true },
    { text: "，Reader 2.0！", underlinable: false },
    { text: "世界", underlinable: true },
  ]);
});

test("non-Chinese chunks neither receive underlines nor advance alternation", () => {
  const text = "中文。Synthetic English.汉字。";
  const chunks = buildVisualChunks(text);
  const chineseChunks = chunks.filter((chunk) => chunk.processed);
  const englishChunk = chunks.find((chunk) => chunk.text.includes("Synthetic"));

  assert.deepEqual(chineseChunks.map((chunk) => chunk.underlined), [true, false]);
  assert.equal(englishChunk.processed, false);
  assert.equal(englishChunk.underlined, false);
  assert.equal(chunks.map((chunk) => chunk.text).join(""), text);
});
