const test = require("node:test");
const assert = require("node:assert/strict");
const {
  boundaryFallsInsideQuantityPhrase,
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

test("treats parentheses, slashes, and title marks as hard clause boundaries", () => {
  const text = "语料库（或包含项目）用于训练/微调《家庭晚餐》。";

  assert.deepEqual(splitClauses(text), [
    "语料库（",
    "或包含项目）",
    "用于训练/",
    "微调《",
    "家庭晚餐》。",
  ]);
  assert.equal(buildVisualChunks(text).some((chunk) => chunk.separated), false);
});

test("keeps a slash inside a numeric fraction but pre-splits an ordinary slash", () => {
  assert.deepEqual(splitClauses("训练/微调1/4英寸，"), [
    "训练/",
    "微调1/4英寸，",
  ]);
  assert.deepEqual(splitClauses("比例为1 ／ 4，继续。"), [
    "比例为1 ／ 4，",
    "继续。",
  ]);
});

test("title marks pre-split a title without drawing a divider beside it", () => {
  const text = "美國重口食人族電影《家庭晚餐》，";

  assert.deepEqual(splitClauses(text), [
    "美國重口食人族電影《",
    "家庭晚餐》，",
  ]);
  assert.doesNotMatch(
    buildVisualChunks(text)
      .map((chunk) => `${chunk.separated ? "｜" : ""}${chunk.text}`)
      .join(""),
    /《｜|｜《|》｜|｜》/u,
  );
});

test("uses boundary balance as a weak prior below model confidence", () => {
  assert.equal(selectBestBoundary([]), null);
  assert.equal(selectBestBoundary([0, 0]), 0);
  assert.equal(selectBestBoundary([0, 2, 0]), 1);
  assert.equal(selectBestBoundary([0.01, 0]), 0);
  assert.equal(selectBestBoundary([0, 0, 0]), 1);
  assert.equal(selectBestBoundary([1.5, 0, 0]), 0);
  assert.equal(
    selectBestBoundary([-10, -10, -10, -10, -10, 0, -10, 0.311, -10, -10]),
    7,
  );
});

test("rejects candidate boundaries inside a segmented word", () => {
  const segmenter = {
    segment() {
      return [{ segment: "初中", index: 1, isWordLike: true }];
    },
  };

  assert.equal(boundaryFallsInsideWord("但初中", 2, segmenter), true);
  assert.equal(boundaryFallsInsideWord("但初中", 1, segmenter), false);
  assert.equal(boundaryFallsInsideWord("但初中", 3, segmenter), false);
});

test("segments each punctuation-delimited clause only once during recursive planning", () => {
  const baseSegmenter = new Intl.Segmenter("zh-CN", { granularity: "word" });
  let calls = 0;
  const segmenter = {
    segment(text) {
      calls += 1;
      return baseSegmenter.segment(text);
    },
  };

  chunkText("同一个无标点子句内的短语块用细竖线分隔", { segmenter });

  assert.equal(calls, 1);
});

test("protects a numeric classifier and its following noun as one phrase", () => {
  const text = "分为5个等级";
  const segmenter = {
    segment() {
      return [
        { segment: "分为", index: 0, isWordLike: true },
        { segment: "5", index: 2, isWordLike: true },
        { segment: "个", index: 3, isWordLike: true },
        { segment: "等级", index: 4, isWordLike: true },
      ];
    },
  };

  assert.equal(boundaryFallsInsideQuantityPhrase(text, 2, segmenter), false);
  assert.equal(boundaryFallsInsideQuantityPhrase(text, 3, segmenter), true);
  assert.equal(boundaryFallsInsideQuantityPhrase(text, 4, segmenter), true);
  assert.equal(boundaryFallsInsideQuantityPhrase(text, 6, segmenter), false);
});

test("keeps a reported numeric quantity phrase free of dividers", () => {
  const rendered = buildVisualChunks(
    "网站里的海量视频被划分为了5个等级，",
  ).map((chunk) => `${chunk.separated ? "｜" : ""}${chunk.text}`).join("");

  assert.match(rendered, /5个等级/u);
  assert.doesNotMatch(rendered, /5｜个|个｜等级/u);
});

test("counts numeric expressions toward the threshold without splitting a fraction from its unit", () => {
  const samples = [
    "前端的宽度是9 1/4英寸，",
    "后端则拉宽到10 1/2英寸，",
  ];

  assert.equal(visualLength(samples[0]), 9);
  assert.equal(visualLength(samples[1]), 9);
  samples.forEach((text) => {
    const chunks = buildVisualChunks(text);
    const rendered = chunks
      .map((chunk) => `${chunk.separated ? "｜" : ""}${chunk.text}`)
      .join("");

    assert.equal(chunks.map((chunk) => chunk.text).join(""), text);
    assert.equal(chunks.some((chunk) => chunk.separated), true);
    assert.doesNotMatch(rendered, /[\/／]｜?\p{Number}*｜英寸|[\/／]\p{Number}+｜英寸/u);
    assert.ok(chunks.every((chunk) => visualLength(chunk.text) <= 8));
  });
});

test("only asks the model to split clauses longer than eight visual units", () => {
  assert.deepEqual(chunkText("甲乙丙丁戊己庚辛"), [
    "甲乙丙丁戊己庚辛",
  ]);
  assert.ok(chunkText("甲乙丙丁戊己庚辛壬").length > 1);
});

test("punctuation-delimited clauses of eight characters or fewer stay intact", () => {
  assert.deepEqual(chunkText("甲乙丙丁，戊己庚辛。"), [
    "甲乙丙丁，",
    "戊己庚辛。",
  ]);
});

test("recursively selects the highest combined score until every chunk is at most eight characters", () => {
  const chunks = chunkText("同一个无标点子句内的短语块用细竖线分隔；");

  assert.deepEqual(chunks, ["同一个无标点", "子句内的短语块", "用细竖线分隔；"]);
  assert.ok(chunks.every((chunk) => visualLength(chunk) <= 8));
});

test("uses model confidence, weak balance, and word protection", () => {
  const chunks = chunkText("而是帮助大脑更快地识别信息结构。");

  assert.deepEqual(chunks, ["而是帮助大脑", "更快地", "识别信息结构。"]);
  assert.ok(chunks.every((chunk) => visualLength(chunk) <= 8));
});

test("the demo fallback recursively respects the eight-character threshold", () => {
  const chunks = chunkText("在左侧输入一段中文，看看它如何被重新组织。");

  assert.deepEqual(chunks, [
    "在左侧",
    "输入一段中文，",
    "看看它",
    "如何被重新组织。",
  ]);
  assert.ok(chunks.every((chunk) => visualLength(chunk) <= 8));
});

test("uses combined scoring and word protection for a technical phrase", () => {
  assert.deepEqual(chunkText("因此它被称为一种数值积分方法"), [
    "因此",
    "它被称为",
    "一种数值积分方法",
  ]);
  assert.deepEqual(chunkText('"无数个小矩形累加"'), ['"无数个小矩形累加"']);
});

test("renders the reported Euler-method example with combined scoring", () => {
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
    "因此",
    "它被称为",
    "一种数值积分方法（",
    "numerical integration method）。",
  ]);
});

test("combined scoring keeps 方向盘 together in the reported sentence", () => {
  assert.deepEqual(chunkText("驾驶员会出于本能进行向左打方向盘等避险动作，"), [
    "驾驶员会",
    "出于本能进行",
    "向左打",
    "方向盘等避险动作，",
  ]);
});

test("word protection keeps 初中 together in the reported title", () => {
  const rendered = buildVisualChunks(
    "这个现象你肯定见过，但初中教材上的解释几乎都是错的",
  ).map((chunk) => `${chunk.separated ? "｜" : ""}${chunk.text}`).join("");

  assert.doesNotMatch(rendered, /初｜中/u);
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
    { text: "更快地", processed: true, separated: true },
    { text: "识别信息结构。", processed: true, separated: true },
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

test("visual length counts Chinese characters and numeric expressions", () => {
  assert.equal(visualLength("你好，Reader 2.0！"), 3);
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
