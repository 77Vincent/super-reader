const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const vm = require("node:vm");
const backend = require("../src/backend/inference.js");
const realChunker = require("../src/backend/chunker.js");
const source = readFileSync(join(__dirname, "../src/backend/chunker.js"), "utf8");
const text = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥天地";

function withScores(scoreTokens) {
  const context = vm.createContext({ Intl, SuperReaderModelBackend: {
    scoreTokens, getModelInfo: () => ({ inputRepresentation: "unicode-context-v1" }),
  } });
  vm.runInContext(source, context);
  return context.SuperReaderChunker;
}

test("confidence is stable unweighted softmax and invalid logits fail closed", () => {
  const { gapProbabilities } = realChunker;
  assert.deepEqual(gapProbabilities([]), []);
  assert.deepEqual(gapProbabilities([-1000, -1000]), [.5, .5]);
  for (const shift of [-1000, 0, 1000]) {
    const p = gapProbabilities([2 + shift, shift]);
    assert.ok(Math.abs(p[0] - 1 / (1 + Math.exp(-2))) < 1e-12);
    assert.ok(Math.abs(p[0] + p[1] - 1) < 1e-12);
  }
  for (const invalid of [NaN, Infinity, -Infinity]) {
    assert.deepEqual(gapProbabilities([10, invalid]), [0, 0]);
  }
});

test("default 75% gate leaves uncertain long clauses intact and zero disables abstention", () => {
  let calls = 0;
  const chunker = withScores((tokens) => { calls++; return tokens.slice(1).map(() => 0); });
  assert.equal(chunker.MIN_SPLIT_CONFIDENCE, .75);
  assert.deepEqual(Array.from(chunker.chunkText(text, { segmenter: null })), [text]);
  assert.equal(calls, 1, "long input still receives a score");
  const baseline = Array.from(chunker.chunkText(text, { segmenter: null, minConfidence: 0 }));
  assert.ok(baseline.length > 1);
  assert.ok(baseline.every((part) => chunker.visualLength(part) <= 12));
  assert.equal(baseline.join(""), text);
  chunker.chunkText(text.slice(0, 12));
  assert.equal(calls, 2, "short clauses skip inference regardless of confidence");
});

test("threshold is strict and configurable and does not use a per-gap sigmoid", () => {
  const chunker = withScores((tokens) => tokens.slice(1).map((_, i) => i === 7 || i === 15 ? 1000 : -1000));
  assert.deepEqual(Array.from(chunker.chunkText(text, { segmenter: null, minConfidence: .5 })), [text]);
  assert.equal(chunker.chunkText(text, { segmenter: null, minConfidence: .49 }).length, 3);
  for (const invalid of [-.1, 1.1, NaN, Infinity, "90"]) {
    assert.throws(() => chunker.chunkText(text, { minConfidence: invalid }), /between 0 and 1/u);
  }
});

test("default recursion renormalizes cached logits without running the model again", () => {
  let calls = 0;
  const chunker = withScores((tokens) => {
    calls++;
    assert.equal(tokens.join(""), text);
    return tokens.slice(1).map((_, i) => i === 15 ? 10 : i === 7 ? 5 : -1000);
  });
  const chunks = Array.from(chunker.chunkText(text, { segmenter: null }));
  assert.deepEqual(chunks, [text.slice(0, 8), text.slice(8, 16), text.slice(16)]);
  assert.equal(calls, 1);
  assert.deepEqual(Array.from(chunker.chunkText(text, { segmenter: null, scoringStrategy: "fixed" })),
    [text.slice(0, 16), text.slice(16)]);
  assert.equal(calls, 2, "fixed-probability comparison also scores the original clause once");
});

test("protected high-confidence gaps do not promote an uncertain alternative", () => {
  const chunker = withScores((tokens) => tokens.slice(1).map((_, i) => i === 7 ? 10 : i === 15 ? 5 : -1000));
  const segmenter = { segment: () => [{ segment: text.slice(6, 10), index: 6, isWordLike: true }] };
  assert.deepEqual(Array.from(chunker.chunkText(text, { segmenter })), [text]);
});

test("cached recursion keeps the original word protection inside a child", () => {
  let calls = 0, segmentations = 0;
  const chunker = withScores((tokens) => {
    calls++;
    return tokens.slice(1).map((_, i) => i === 5 ? 10 : i === 11 ? 5 : -1000);
  });
  const segmenter = { segment: () => {
    segmentations++;
    return [{ segment: text.slice(10, 14), index: 10, isWordLike: true }];
  } };
  assert.deepEqual(Array.from(chunker.chunkText(text, { segmenter })), [text.slice(0, 6), text.slice(6)]);
  assert.equal(calls, 1);
  assert.equal(segmentations, 1);
});

test("a confident edge gap remains eligible regardless of its position", () => {
  const chunker = withScores((tokens) => tokens.slice(1).map((_, i) => i === 0 ? Math.log(10) : i === 11 ? 0 : -1000));
  // Raw confidence is 10/11 at the edge and is used directly for acceptance.
  assert.deepEqual(Array.from(chunker.chunkText(text, { segmenter: null })),
    [text.slice(0, 1), text.slice(1, 12), text.slice(12)]);
});

test("confidence cannot create a fragment with no visual content", () => {
  for (const leading of [true, false]) {
    const input = leading ? "🌈" + text : text + "🌈";
    const chunker = withScores((tokens) => tokens.slice(1).map((_, i) => (
      i === (leading ? 0 : tokens.length - 2) ? 100 : 0
    )));
    for (const scoringStrategy of ["recursive-softmax", "fixed", "recursive-model"]) {
      assert.deepEqual(Array.from(chunker.chunkText(input, { segmenter: null, scoringStrategy })), [input]);
    }
  }
});

test("model cuts require adjacent Han in the source, before normalization or whitespace cleanup", () => {
  const snippets = [
    "｜8:30", "8｜:30", "8:｜30", "8:30｜", "3｜.14", "3.｜14", "1/｜4",
    "｜AI", "AI｜", "｜（切勿模仿）", "（｜切勿模仿）", "（切勿模仿｜）", "（切勿模仿）｜",
    "｜ 切勿模仿", " ｜切勿模仿", "｜㍿", "㍿｜", "｜ﬃ", "ﬃ｜", "🌈｜", "𠮷\u{E0100}｜",
  ];
  for (const snippet of snippets) {
    const marked = text.slice(0, 8) + snippet + text.slice(8);
    const offset = marked.indexOf("｜");
    const input = marked.replace("｜", "");
    const target = realChunker.tokenizeContext(input).findIndex((token) => token.index === offset) - 1;
    assert.ok(target >= 0, snippet);
    const chunker = withScores((tokens) => tokens.slice(1).map((_, i) => i === target ? 10 : i === 2 ? 5 : -1000));
    // The forbidden top probability is retained; the weaker Han gap is not promoted.
    assert.deepEqual(Array.from(chunker.chunkText(input, { segmenter: null })), [input], snippet);
    const chunks = Array.from(chunker.chunkText(input, { segmenter: null, minConfidence: 0 }));
    assert.equal(chunks.join(""), input);
    let end = 0;
    for (const part of chunks.slice(0, -1)) {
      end += part.length;
      assert.notEqual(end, offset, snippet);
    }
  }
});

test("supplementary Han on both sides remains a valid model cut", () => {
  const left = text.slice(0, 8) + "𠮷";
  const right = "𠀀" + text.slice(8);
  const chunker = withScores((tokens) => tokens.slice(1).map((_, i) => i === 8 ? 100 : 0));
  assert.deepEqual(Array.from(chunker.chunkText(left + right, { segmenter: null })), [left, right]);
});

test("quantity protection retains Han-to-Han boundaries not covered by character eligibility", () => {
  for (const quantity of ["5个｜等级", "〇｜等级", "〡｜等级"]) {
    const marked = text.slice(0, 8) + quantity + text.slice(8);
    const input = marked.replace("｜", "");
    const target = realChunker.tokenizeContext(input).findIndex((token) => token.index === marked.indexOf("｜")) - 1;
    const chunker = withScores((tokens) => tokens.slice(1).map((_, i) => i === target ? 100 : 0));
    assert.deepEqual(Array.from(chunker.chunkText(input)), [input], quantity);
  }
});

test("reported warning parentheses no longer receive an adjacent model divider", () => {
  for (const aside of ["（切勿模仿）", "(切勿模仿)"]) {
    const input = "在没有人被人特别留意的情况下，偷偷少去公司一天，几乎不会被他人发觉" + aside + "。";
    assert.deepEqual(realChunker.process([input]), [[]]);
    assert.equal(realChunker.chunkText(input).join(""), input);
  }
});

test("long inputs normalize across all windows and never force the one-gap tail", () => {
  for (const length of [256, 257, 511, 512, 1000]) {
    const input = "甲".repeat(length);
    let calls = 0;
    const chunker = withScores((tokens) => { calls++; return tokens.slice(1).map(() => 0); });
    assert.deepEqual(Array.from(chunker.chunkText(input, { segmenter: null })), [input]);
    assert.equal(calls, Math.ceil((length - 1) / 255));
  }
});

test("punctuation separates independent confidence distributions and preserves Unicode offsets", () => {
  const item = "𠮷🌈" + text;
  let calls = 0;
  const chunker = withScores((tokens) => { calls++; return tokens.slice(1).map((_, i) => i === 9 ? 100 : 0); });
  const input = `${item}、${item}，短句。`;
  const chunks = Array.from(chunker.chunkText(input, { segmenter: null }));
  assert.equal(chunks.join(""), input);
  assert.equal(calls, 2);
  const cuts = Array.from(chunker.process([input])[0]);
  assert.equal(cuts.length, 2);
  for (const offset of cuts) assert.doesNotMatch(input[offset], /[\uDC00-\uDFFF]/u);
});

test("bundled model uses the 75% default and honors a stricter explicit threshold", () => {
  const high = "我一直在思考明天早上的早餐吃什么";
  const low = "而是帮助大脑更快地识别信息结构。";
  assert.deepEqual(realChunker.process([high, low]), [[6], [6]]);
  assert.deepEqual(realChunker.process([low], { minConfidence: .9 }), [[]]);
  assert.deepEqual(realChunker.process([low], { minConfidence: 0 }), [[6]]);
  const tokens = realChunker.tokenizeContext(high);
  const probabilities = realChunker.gapProbabilities(backend.scoreTokens(tokens.map((t) => t.segment)));
  assert.ok(probabilities[5] > .9);
});

test("recursive option runs fresh inference and can accept a different second boundary", () => {
  const inputs = [];
  const chunker = withScores((tokens) => {
    inputs.push(tokens.join(""));
    return tokens.slice(1).map((_, i) => i === (tokens.length === 24 ? 15 : 3) ? 100 : 0);
  });
  assert.deepEqual(Array.from(chunker.chunkText(text, { segmenter: null, scoringStrategy: "recursive-model" })),
    [text.slice(0, 4), text.slice(4, 16), text.slice(16)]);
  assert.deepEqual(inputs, [text, text.slice(0, 16)]);
  for (const scoringStrategy of ["unknown", true, 1]) {
    assert.throws(() => chunker.chunkText(text, { scoringStrategy }), /scoringStrategy/u);
  }
});

test("recursive scoring abstains on an uncertain child even when it exceeds the length threshold", () => {
  const inputs = [];
  const chunker = withScores((tokens) => {
    inputs.push(tokens.join(""));
    return tokens.slice(1).map((_, i) => tokens.length === 24 && i === 15 ? 100 : 0);
  });
  assert.deepEqual(Array.from(chunker.chunkText(text, { segmenter: null, scoringStrategy: "recursive-model" })),
    [text.slice(0, 16), text.slice(16)]);
  assert.deepEqual(inputs, [text, text.slice(0, 16)]);
});

test("recursive scoring preserves original word protections and shifts right-child indices", () => {
  const inputs = [];
  let segmentations = 0;
  const segmenter = { segment: () => {
    segmentations++;
    return [{ segment: text.slice(10, 14), index: 10, isWordLike: true }];
  } };
  const chunker = withScores((tokens) => {
    inputs.push(tokens.join(""));
    return tokens.slice(1).map((_, i) => i === 5 ? 100 : 0);
  });
  assert.deepEqual(Array.from(chunker.chunkText(text, { segmenter, scoringStrategy: "recursive-model" })),
    [text.slice(0, 6), text.slice(6)]);
  assert.deepEqual(inputs, [text, text.slice(6)]);
  assert.equal(segmentations, 1, "model context is the only experimental variable");
});

test("recursive scoring maintains window limits and original supplementary Unicode offsets", () => {
  const input = Array.from({ length: 600 }, (_, i) => String.fromCodePoint(0x20000 + i)).join("");
  const windows = [];
  const chunker = withScores((tokens) => {
    windows.push(Array.from(tokens));
    return tokens.slice(0, -1).map((token) => {
      const index = token.codePointAt(0) - 0x20000;
      return index === 299 ? 100 : index === 99 ? 80 : index === 499 ? 60 : 0;
    });
  });
  const chunks = Array.from(chunker.chunkText(input, { segmenter: null }));
  assert.deepEqual(chunks.map((part) => Array.from(part).length), [100, 200, 200, 100]);
  assert.equal(chunks.join(""), input);
  assert.equal(windows.length, 3, "only the three original windows run CNN inference");
  assert.ok(windows.every((tokens) => tokens.length >= 2 && tokens.length <= 256));
  assert.deepEqual(Array.from(chunker.process([input])[0]), [200, 600, 1000]);
});
