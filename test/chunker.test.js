const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const vm = require("node:vm");
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
} = require("../src/backend/chunker.js");

function withModel(scoreTokens) {
  const context = vm.createContext({ SuperReaderModelBackend: { scoreTokens }, Intl });
  vm.runInContext(readFileSync(join(__dirname, "../src/backend/chunker.js"), "utf8"), context);
  return context.SuperReaderChunker;
}

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

test("treats parentheses and title marks as hard clause boundaries", () => {
  const text = "语料库（或包含项目）用于训练/微调《家庭晚餐》。";

  assert.deepEqual(splitClauses(text), [
    "语料库（",
    "或包含项目）",
    "用于训练/微调《",
    "家庭晚餐》。",
  ]);
  assert.equal(buildVisualChunks(text).some((chunk) => chunk.separated), false);
});

test("ordinary and numeric slashes stay inside clauses in both widths", () => {
  for (const slash of ["/", "／"]) {
    const clause = `训练${slash}微调1${slash}4英寸，`;
    assert.deepEqual(splitClauses(clause), [clause]);
    assert.deepEqual(chunkTextByClause(clause).map((chunks) => chunks.join("")), [clause]);
    assert.deepEqual(splitClauses(`比例为1 ${slash} 4，继续。`), [
      `比例为1 ${slash} 4，`,
      "继续。",
    ]);
  }
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

test("balance favors the center for close scores but a stronger model choice can win", () => {
  assert.equal(selectBestBoundary([]), null);
  assert.equal(selectBestBoundary([0, 0]), 0);
  assert.equal(selectBestBoundary([0, 2, 0]), 1);
  assert.equal(selectBestBoundary([0.01, 0]), 0);
  assert.equal(selectBestBoundary([0, 0, 0]), 1);
  assert.equal(selectBestBoundary([0.01, 0, 0]), 1);
  assert.equal(selectBestBoundary([0, 0, 0.01]), 1);
  assert.equal(selectBestBoundary([1.5, 0, 0]), 0);
  assert.equal(
    selectBestBoundary([-10, -10, -10, -10, -10, 0, -10, 0.311, -10, -10]),
    7,
  );
});

test("boundary ranking matches softmax share multiplied by p times one minus p", () => {
  const examples = [
    [-41.05, -45.59, -39.40, -38.58, -37.75, -33.99, -35.99, -36.98, -40.03, -38.42, -36.51, -38.66, -32.85, -36.24, -40.46],
    [0.2, -1, 0, 0.05, 0, -2, 0.5, -1, 0.2],
    [2, 0, 0, 0, 0, 0, 0, 0, 0],
  ];
  for (const logits of examples) {
    const exponentials = logits.map((score) => Math.exp(score - Math.max(...logits)));
    const sum = exponentials.reduce((total, value) => total + value, 0);
    const weighted = exponentials.map((value, index) => {
      const p = (index + 1) / (logits.length + 1);
      return value / sum * p * (1 - p);
    });
    const expected = weighted.indexOf(Math.max(...weighted));
    for (const shift of [-1000, 0, 1000]) {
      assert.equal(selectBestBoundary(logits.map((score) => score + shift)), expected);
    }
  }
});

test("small center drift survives a small model advantage while extreme edges are penalized", () => {
  const scores = Array(99).fill(-100);
  scores[49] = 0; // 50/50
  scores[44] = 0.02; // 45/55: only a 1% reduction relative to the center.
  assert.equal(selectBestBoundary(scores), 44);
  scores[44] = -100;
  scores[0] = 2; // 1/99: even this larger model score loses after weighting.
  assert.equal(selectBestBoundary(scores), 49);
  scores[0] = 4;
  assert.equal(selectBestBoundary(scores), 0);
});

test("boundary selection excludes protected gaps and invalid scores", () => {
  assert.equal(selectBestBoundary([10, -2, -1], (index) => index !== 0), 2);
  assert.equal(selectBestBoundary([0, 0, 0], (index) => index !== 0), 1);
  assert.equal(selectBestBoundary([NaN, Infinity, -Infinity, -3, -1]), 4);
  assert.equal(selectBestBoundary([NaN, Infinity, -Infinity]), null);
  assert.equal(selectBestBoundary([10, 20], () => false), null);
});

test("boundary selection searches only inside the requested fragment and returns its original index", () => {
  const scores = [100, 2, 3, 2, 100];
  assert.equal(selectBestBoundary(scores, undefined, 1, 4), 2);
  assert.equal(selectBestBoundary(scores, (index) => index !== 2, 1, 4), 1);
  assert.equal(selectBestBoundary(scores, undefined, 2, 2), null);
  assert.equal(selectBestBoundary(Array(19).fill(0), undefined, 10, 19), 14,
    "balance must use the child fragment's center, not the original text's center");
});

test("balance uses visual units including leading numbers, fractions and supplementary Han", () => {
  const chunker = withModel((tokens) => Array(tokens.length - 1).fill(0));
  for (const [text, expected] of [
    ["1甲乙丙丁戊己庚辛", ["1甲乙丙", "丁戊己庚辛"]],
    ["甲乙丙丁戊己庚辛 1 2", ["甲乙丙丁戊", "己庚辛 1 2"]],
    ["1/4 English 𠀀甲乙丙丁戊己庚", ["1/4 English 𠀀甲乙", "丙丁戊己庚"]],
  ]) {
    const chunks = Array.from(chunker.chunkText(text, { segmenter: null }));
    assert.deepEqual(chunks, expected);
    assert.equal(chunks.join(""), text);
  }
});

test("one model evaluation supplies every split in a clause, with scores local to that call", () => {
  const text = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥天地";
  const inputs = [];
  const chunker = withModel((tokens) => {
    inputs.push(Array.from(tokens));
    assert.equal(tokens.join(""), text, "the model must never receive a recursive fragment");
    const scores = Array(tokens.length - 1).fill(-100);
    scores[7] = 5;
    scores[15] = 10;
    return scores;
  });
  const chunks = chunker.chunkText(text, { segmenter: null });
  assert.deepEqual(Array.from(chunks), [text.slice(0, 8), text.slice(8, 16), text.slice(16)]);
  assert.equal(inputs.length, 1);
  // A new operation gets fresh scores; they are not retained across requests.
  chunker.chunkText(text, { segmenter: null });
  assert.equal(inputs.length, 2);
});

test("short clauses skip inference and separate long clauses receive separate score arrays", () => {
  const inputs = [];
  const chunker = withModel((tokens) => {
    inputs.push(tokens.join(""));
    return Array(tokens.length - 1).fill(0);
  });
  chunker.chunkText("甲乙丙丁戊己庚辛，中文。English", { segmenter: null });
  assert.deepEqual(inputs, []);
  const first = "甲乙丙丁戊己庚辛壬";
  const second = "天地玄黄宇宙洪荒日";
  const text = `${first}，中文。${second}`;
  assert.equal(chunker.chunkText(text, { segmenter: null }).join(""), text);
  assert.deepEqual(inputs, [first, second]);
});

test("fixed model windows score every gap once, without forcing cuts at window edges", () => {
  for (const length of [255, 256, 257, 511, 512, 513, 1000]) {
    const tokens = Array.from({ length }, (_, index) => String.fromCodePoint(0x20000 + index));
    const inputs = [];
    const gaps = [];
    const chunker = withModel((windowTokens) => {
      inputs.push(Array.from(windowTokens));
      return windowTokens.slice(0, -1).map((token) => {
        const index = token.codePointAt(0) - 0x20000;
        gaps.push(index);
        return (index + 1) % 8 === 0 ? 10 : -1;
      });
    });
    const chunks = chunker.chunkText(tokens.join(""), { segmenter: null });
    assert.equal(inputs.length, Math.ceil((length - 1) / 255));
    assert.ok(inputs.every((window) => window.length >= 2 && window.length <= 256));
    assert.deepEqual(inputs[0].concat(...inputs.slice(1).map((window) => window.slice(1))), tokens);
    assert.deepEqual(gaps, Array.from({ length: length - 1 }, (_, index) => index));
    assert.equal(chunks.join(""), tokens.join(""));
    assert.equal(chunks.length, Math.ceil(length / 8));
    assert.ok(chunks.slice(0, -1).every((chunk) => Array.from(chunk).length === 8));
  }
});

test("long clauses with strongly favored edge scores split without overflowing the call stack", () => {
  const text = "甲".repeat(9000);
  let calls = 0;
  const chunker = withModel((tokens) => {
    const start = calls++ * 255;
    return tokens.slice(1).map((_, index) => -10 * (start + index));
  });
  const chunks = chunker.chunkText(text, { segmenter: null });
  assert.equal(chunks.join(""), text);
  assert.equal(chunks.length, text.length - 7);
  assert.equal(chunks.at(-1), "甲".repeat(8));
  assert.equal(calls, Math.ceil((text.length - 1) / 255));
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

test("recursively selects the highest allowed weighted score until the length threshold is met", () => {
  const chunks = chunkText("同一个无标点子句内的短语块用细竖线分隔；");

  assert.deepEqual(chunks, ["同一个无标点", "子句内的短语块", "用细竖线分隔；"]);
  assert.ok(chunks.every((chunk) => visualLength(chunk) <= 8));
});

test("uses model scores and word protection for the reading example", () => {
  const chunks = chunkText("而是帮助大脑更快地识别信息结构。");

  assert.deepEqual(chunks, ["而是帮助大脑", "更快地识别信息", "结构。"]);
  assert.ok(chunks.every((chunk) => visualLength(chunk) <= 8));
});

test("recursive splitting respects the eight-character threshold across clauses", () => {
  const chunks = chunkText("在左侧输入一段中文，看看它如何被重新组织。");

  assert.deepEqual(chunks, [
    "在左侧",
    "输入一段中文，",
    "看看它",
    "如何被重新组织。",
  ]);
  assert.ok(chunks.every((chunk) => visualLength(chunk) <= 8));
});

test("uses model scores and word protection for a technical phrase", () => {
  assert.deepEqual(chunkText("因此它被称为一种数值积分方法"), [
    "因此",
    "它被称为",
    "一种数值积分方法",
  ]);
  assert.deepEqual(chunkText('"无数个小矩形累加"'), ['"无数个小矩形累加"']);
});

test("renders the reported Euler-method example with model scores", () => {
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

test("model scoring keeps 方向盘 together at the current length threshold", () => {
  assert.deepEqual(chunkText("驾驶员会出于本能进行向左打方向盘等避险动作，"), [
    "驾驶员会出于本能",
    "进行",
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
    { text: "更快地识别信息", processed: true, separated: true },
    { text: "结构。", processed: true, separated: true },
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
