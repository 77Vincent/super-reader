const test = require("node:test");
const assert = require("node:assert/strict");
const {
  buildVisualChunks,
  chunkText,
  chunkTextByClause,
  createBoundaryTree,
  splitUnderlineRuns,
  splitClauses,
  visualLength,
} = require("../src/chunker.js");

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

test("builds a full recursive tree from ranked word gaps", () => {
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

test("only emits boundaries exposed by the browser word segmenter", () => {
  const text = "我一直在思考明天早上的早餐吃什么";
  const segmenter = new Intl.Segmenter("zh-CN", { granularity: "word" });
  const validBoundaries = new Set(
    Array.from(segmenter.segment(text), (item) => item.index).slice(1),
  );
  const chunks = chunkText(text, { targetLength: 2, segmenter });
  let offset = 0;

  chunks.slice(0, -1).forEach((chunk) => {
    offset += chunk.length;
    assert.ok(validBoundaries.has(offset), `unexpected boundary at ${offset}`);
  });
  assert.equal(chunks.join(""), text);
});

test("abandons a hard cut when the remaining span is one segmenter word", () => {
  const text = "超长而不可拆分的浏览器词";
  const segmenter = {
    segment() {
      return [{ segment: text, index: 0, isWordLike: true }];
    },
  };

  assert.deepEqual(chunkText(text, { targetLength: 2, segmenter }), [text]);
});

test("abandons chunking when browser word segmentation is unavailable", () => {
  const text = "无法确认词语边界时保留整段文字";

  assert.deepEqual(chunkText(text, { targetLength: 2, segmenter: null }), [text]);
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
