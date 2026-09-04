const test = require("node:test");
const assert = require("node:assert/strict");
const {
  buildVisualChunks,
  chunkText,
  chunkTextByClause,
  createBoundaryTree,
  splitUnderlineRuns,
  splitClauses,
  tokenizeHanCharacters,
  visualLength,
} = require("../src/chunker.js");

test("combines Chinese characters into target-sized model chunks", () => {
  const chunks = chunkText("我一直在思考明天早上的早餐吃什么", { targetLength: 7 });

  assert.equal(chunks.join(""), "我一直在思考明天早上的早餐吃什么");
  assert.ok(chunks.every((chunk) => visualLength(chunk) <= 7));
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

  assert.equal(clauses.length, 2);
  assert.ok(clauses[0].at(-1).endsWith("，"));
  assert.ok(clauses[1].at(-1).endsWith("。"));
  assert.equal(clauses.flat().join(""), text);
});

test("never joins text across a punctuation boundary", () => {
  const chunks = chunkText("在左侧输入一段中文，", { targetLength: 7 });

  assert.equal(chunks.join(""), "在左侧输入一段中文，");
  assert.ok(chunks.at(-1).endsWith("，"));
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
  const clauses = chunkTextByClause(text, { targetLength: 2 });
  assert.equal(clauses.length, 3);
  assert.equal(clauses.flat().join(""), text);
  assert.equal(clauses[1].join(""), "“连续单字合并”");
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

test("builds a full recursive tree from ranked character gaps", () => {
  const tokens = ["甲", "乙", "丙", "丁"].map((segment, index) => ({
    segment,
    index,
    isWordLike: true,
  }));
  const tree = createBoundaryTree(tokens, [0.2, 0.9, 0.4]);

  assert.equal(tree.boundaryAfter, 1);
  assert.equal(tree.left.text, "甲乙");
  assert.equal(tree.right.text, "丙丁");
  assert.equal(tree.left.left.text, "甲");
  assert.equal(tree.left.right.text, "乙");
});

test("can emit any adjacent Chinese-character boundary", () => {
  const text = "我一直在思考明天早上的早餐吃什么";
  const validBoundaries = new Set(Array.from({ length: text.length - 1 }, (_, index) => index + 1));
  const chunks = chunkText(text, { targetLength: 2 });
  let offset = 0;

  chunks.slice(0, -1).forEach((chunk) => {
    offset += chunk.length;
    assert.ok(validBoundaries.has(offset), `unexpected boundary at ${offset}`);
  });
  assert.equal(chunks.join(""), text);
  assert.ok(chunks.every((chunk) => visualLength(chunk) <= 2));
});

test("does not restrict candidates to browser word boundaries", () => {
  const text = "超长而不可拆分的浏览器词";
  const segmenter = {
    segment() {
      return [{ segment: text, index: 0, isWordLike: true }];
    },
  };

  const chunks = chunkText(text, { targetLength: 2, segmenter });
  assert.equal(chunks.join(""), text);
  assert.ok(chunks.length > 1);
  assert.ok(chunks.every((chunk) => visualLength(chunk) <= 2));
});

test("does not depend on browser word segmentation", () => {
  const text = "无法确认词语边界时保留整段文字";
  const chunks = chunkText(text, { targetLength: 2, segmenter: null });

  assert.equal(chunks.join(""), text);
  assert.ok(chunks.length > 1);
  assert.ok(chunks.every((chunk) => visualLength(chunk) <= 2));
});

test("supports a two-character minimum target", () => {
  const text = "春天来了我们出发。";
  const chunks = chunkText(text, { targetLength: 2 });

  assert.equal(chunks.join(""), text);
  assert.ok(chunks.every((chunk) => visualLength(chunk) <= 2));
  assert.deepEqual(chunkText(text, { targetLength: 1 }), chunks);
});

test("character tokenization keeps original UTF-16 source offsets", () => {
  assert.deepEqual(tokenizeHanCharacters("A中😀文"), [
    { segment: "中", index: 1 },
    { segment: "文", index: 4 },
  ]);
});

test("never loses whitespace or mixed-language content", () => {
  const text = "这是 Super Reader 的 2.0 版本，很好用。";
  const chunks = chunkText(text, { targetLength: 6 });

  assert.equal(chunks.join(""), text);
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
  const chunks = buildVisualChunks("中文。Synthetic English.汉字。", { targetLength: 7 });
  const chineseChunks = chunks.filter((chunk) => chunk.processed);
  const englishChunk = chunks.find((chunk) => chunk.text.includes("Synthetic"));

  assert.deepEqual(chineseChunks.map((chunk) => chunk.underlined), [true, false]);
  assert.equal(englishChunk.processed, false);
  assert.equal(englishChunk.underlined, false);
  assert.equal(chunks.map((chunk) => chunk.text).join(""), "中文。Synthetic English.汉字。");
});
