const test = require("node:test");
const assert = require("node:assert/strict");
const {
  boundaryFallsInsideWord,
  buildVisualChunks,
  chunkText,
  chunkTextByClause,
  selectBestBoundary,
  selectCenteredModelBoundary,
  splitClauses,
  tokenizeHanCharacters,
  visualLength,
} = require("../src/chunker.js");

test("model chunks preserve the complete source text", () => {
  const text = "我一直在思考明天早上的早餐吃什么";
  assert.equal(chunkText(text).join(""), text);
});

test("punctuation creates hard clause boundaries before model planning", () => {
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

test("keeps paired Chinese quotes inside the surrounding punctuation clause", () => {
  const text = "必须在“连续单字合并”之前执行。";
  const expectedClauses = [text];
  const modelChunksByClause = chunkTextByClause(text);

  assert.deepEqual(splitClauses(text), expectedClauses);
  assert.deepEqual(
    modelChunksByClause.map((chunks) => chunks.join("")),
    expectedClauses,
  );
});

test("keeps straight double quotes inside the surrounding punctuation clause", () => {
  const text = '在"quoted words"之后。';

  assert.deepEqual(splitClauses(text), [text]);
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

test("treats parentheses and slashes as hard clause boundaries", () => {
  const text = "语料库（或包含项目）用于训练/微调。";

  assert.deepEqual(splitClauses(text), [
    "语料库（",
    "或包含项目）",
    "用于训练/",
    "微调。",
  ]);
  assert.equal(buildVisualChunks(text).some((chunk) => chunk.separated), false);
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

test("uses balance only among boundaries accepted by the model", () => {
  const centerIsPlausible = [10, 0, 0, 7.1, 0, 0, 0];
  const centerIsRejected = [10, 0, 0, 6.9, 0, 0, 0];
  const strongerNeighbor = [0, 0, 0, 7, 8, 0, 0];

  assert.equal(selectCenteredModelBoundary(centerIsPlausible), 3);
  assert.equal(selectCenteredModelBoundary(centerIsRejected), 0);
  assert.equal(selectCenteredModelBoundary(strongerNeighbor), 4);
});

test("only asks the model to split clauses longer than ten Han characters", () => {
  assert.deepEqual(chunkText("甲乙丙丁戊己庚辛壬癸", { segmenter: null }), [
    "甲乙丙丁戊己庚辛壬癸",
  ]);
  assert.ok(
    chunkText("甲乙丙丁戊己庚辛壬癸子", { segmenter: null }).length > 1,
  );
});

test("punctuation-delimited clauses of ten characters or fewer stay intact", () => {
  assert.deepEqual(chunkText("甲乙丙丁戊，己庚辛壬癸。", { segmenter: null }), [
    "甲乙丙丁戊，",
    "己庚辛壬癸。",
  ]);
});

test("recursively splits either result when it remains longer than ten characters", () => {
  const chunks = chunkText("同一个无标点子句内的短语块用细竖线分隔；");

  assert.deepEqual(chunks, ["同一个无标点子句", "内的短语块", "用细竖线分隔；"]);
  assert.ok(chunks.every((chunk) => visualLength(chunk) <= 10));
});

test("keeps model-approved terms together instead of forcing equal lengths", () => {
  const chunks = chunkText("而是帮助大脑更快地识别信息结构。");

  assert.deepEqual(chunks, ["而是帮助大脑", "更快地识别信息结构。"]);
});

test("the demo fallback clauses stay intact at the ten-character threshold", () => {
  const chunks = chunkText("在左侧输入一段中文，看看它如何被重新组织。");

  assert.deepEqual(chunks, [
    "在左侧输入一段中文，",
    "看看它如何被重新组织。",
  ]);
  assert.deepEqual(chunks.map(visualLength), [9, 10]);
});

test("selects the most centered model-approved boundary for a technical phrase", () => {
  assert.deepEqual(chunkText("因此它被称为一种数值积分方法"), [
    "因此它被称为",
    "一种数值积分方法",
  ]);
  assert.deepEqual(chunkText('"无数个小矩形累加"'), ['"无数个小矩形累加"']);
});

test("keeps the technical terms intact in the reported Euler-method example", () => {
  const text = '欧拉法是在积分无法直接计算时，用"无数个小矩形累加"来近似积分，因此它被称为一种数值积分方法（numerical integration method）。';

  assert.deepEqual(splitClauses(text), [
    "欧拉法是在积分无法直接计算时，",
    '用"无数个小矩形累加"来近似积分，',
    "因此它被称为一种数值积分方法（",
    "numerical integration method）。",
  ]);
  assert.deepEqual(chunkText(text), [
    "欧拉法是在积分",
    "无法直接计算时，",
    '用"无数个小矩形',
    '累加"来近似积分，',
    "因此它被称为",
    "一种数值积分方法（",
    "numerical integration method）。",
  ]);
});

test("does not trade a one-character balance gain for broken compound words", () => {
  const text = "婴儿安全座椅更多推荐放在副驾驶后侧（驾驶侧的对侧），虽然在纯粹的碰撞安全性上驾驶员后侧略占优势，但综合日常便利性与上下车安全，副驾驶后侧是现实生活中更普及、更实用的选择。";
  const rendered = buildVisualChunks(text)
    .map((chunk) => `${chunk.separated ? "｜" : ""}${chunk.text}`)
    .join("");

  assert.doesNotMatch(rendered, /便利｜性/u);
  assert.doesNotMatch(rendered, /现实｜生活/u);
  assert.match(rendered, /便利性｜与/u);
  assert.match(rendered, /后侧｜是现实生活/u);
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
  const text = "春天来了我们出发继续前进";
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
  assert.deepEqual(chunkText("而是帮助大脑更快地识别信息结构。", { segmenter: null }), [
    "而是帮助大脑",
    "更快地识别信息结构。",
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
  assert.deepEqual(buildVisualChunks("而是帮助大脑更快地识别信息结构。"), [
    { text: "而是帮助大脑", processed: true, separated: false },
    { text: "更快地识别信息结构。", processed: true, separated: true },
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
