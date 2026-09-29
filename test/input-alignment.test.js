const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { execFileSync } = require("node:child_process");
const vm = require("node:vm");
const chunker = require("../src/backend/chunker.js");
const backend = require("../src/backend/inference.js");
const policy = require("../training/text-policy.json");

const source = readFileSync(require.resolve("../src/backend/chunker.js"), "utf8");
function withModel(scoreTokens, convolutionLayers = 16) {
  const context = vm.createContext({ Intl, SuperReaderModelBackend: {
    scoreTokens, getModelInfo: () => ({ inputRepresentation: "unicode-context-v1", convolutionLayers }),
  } });
  vm.runInContext(source, context);
  return context.SuperReaderChunker;
}
const tokenText = (text) => chunker.tokenizeContext(text).map((token) => token.segment).join("");

test("runtime fragments match Python and JavaScript training preparation before sample filtering", async () => {
  const { trainingFragmentLines } = await import("../training/text_policy.mjs");
  const fixtures = [
    "女：那可挺麻烦的，吃点儿治疗过敏的药吧。",
    "英文,逗号和句点.以及分号;问号?叹号!全部保留，只有中文代理分段。",
    "苹果、香蕉､梨﹑橘子︑桃子，放入（A）袋。",
    "保留﹐﹔‼⁇︒｡，然后继续。",
    "  全角ＡＢＣ及８：３０\t  还有cafe\u0301和ﬃ𠮷🌈  ，  继续阅读。",
    "他说：“明天见。”然后离开。",
    "先停顿……然后继续...最后结束。",
    "坐标为(-8545，-27679)，继续计算。",
    "读数为−1.5，+.25，－３．５，＋４，继续记录。",
    "数值为-1e-3，+2E+4，继续计算。",
    "数值为 1\t，\u00a0- 2 ， +3 ，继续。",
    "输入𠮷１２，－３，完成。🧪数值为𝟙，-𝟚，继续。",
    "金额1，000，000元，明天继续。",
    "前段，\n后段。\r\n最后一行没有结束标点",
    ...policy.proxy_punctuation.map((p) => `引句。完整左段${p}完整右段。`),
  ];
  const prepared = fixtures.map((text) => Array.from(trainingFragmentLines(text))
    .flat().map((fragment) => fragment.text).filter(Boolean));
  const python = JSON.parse(execFileSync("python3", ["-c", String.raw`
import json,sys
sys.path.insert(0, 'training')
from text_policy import training_fragment_lines
print(json.dumps([[f['text'] for line in training_fragment_lines(t) for f in line if f['text']]
                  for t in json.load(sys.stdin)], ensure_ascii=False))
`], { input: JSON.stringify(fixtures), encoding: "utf8" }));
  assert.deepEqual(prepared, python);
  for (let i = 0; i < fixtures.length; i++) {
    const clauses = chunker.splitClauses(fixtures[i]);
    assert.equal(clauses.join(""), fixtures[i]);
    assert.deepEqual(clauses.map(tokenText).filter(Boolean), prepared[i], fixtures[i]);
  }
});

test("hiding a training proxy gives the same model tokens as the corresponding raw AB input", async () => {
  const { trainingPairs } = await import("../training/text_policy.mjs");
  const fixtures = [
    ["我们需要了解ASCII,英文逗号的保留方式", "这样才能让推理输入和训练保持一致"],
    ["我们已经读过苹果、香蕉和梨这些内容", "现在继续介绍后面的完整内容"],
    ["我们记录读数为−1.5，+.25然后继续观察", "这些数据可以用于后续分析"],
    ["我们研究﹐﹔‼⁇︒｡这些兼容字形的处理方法", "然后继续检查原文是否完整"],
    ["我们已经了解这个问题的主要情况", "”然后继续介绍这个话题的其他内容"],
    ["这里记录ＡＢＣ及cafe\u0301和ﬃ还有𠮷这些字形", "接下来还会逐个检查这些字符"],
  ];
  for (const [left, right] of fixtures) {
    const pairs = Array.from(trainingPairs(`引句。${left}，${right}。`));
    assert.equal(pairs.length, 1, left);
    const trainingInput = pairs[0].left + pairs[0].right;
    const seen = [];
    const runtime = withModel((tokens) => {
      seen.push(tokens.join(""));
      return tokens.slice(1).map(() => 0);
    });
    const rawAB = left + right;
    assert.equal(runtime.chunkText(rawAB).join(""), rawAB);
    assert.deepEqual(seen, [trainingInput], left);
  }
});

test("all training line boundaries stop inference before whitespace folding", () => {
  const first = "第一行包含足够多的汉字需要送给模型处理";
  const second = "第二行也包含足够多的汉字需要单独处理";
  const breaks = ["\r\n", "\n", "\r", "\v", "\f", "\u001c", "\u001d", "\u001e", "\u0085", "\u2028", "\u2029"];
  for (const newline of breaks) {
    assert.match(newline, new RegExp(`^(?:${policy.line_boundaries})$`, "u"));
    const seen = [];
    const runtime = withModel((tokens) => {
      seen.push(tokens.join(""));
      return tokens.slice(1).map(() => 0);
    });
    const text = `${first}${newline}${second}`;
    assert.equal(runtime.chunkText(text).join(""), text);
    assert.deepEqual(seen, [first, second]);
    assert.deepEqual(Array.from(runtime.splitClauses(`数值1，${newline}-2然后继续。`)),
      [`数值1，${newline}`, "-2然后继续。"]);
  }
});

test("source offsets survive normalization expansion, quotes and compatibility punctuation", () => {
  const text = "𠮷ﬃcafe\u0301｡﹐﹔‼⁇我们需要保留这些字符继续阅读这些内容。";
  const tokens = chunker.tokenizeContext(text);
  assert.equal(tokens.map((token) => token.segment).join(""), text.slice(0, -1).normalize("NFKC"));
  const cut = text.indexOf("继续");
  const runtime = withModel((input) => input.slice(1).map((token, i) => (
    token === "继" && input[i] === "符" ? 100 : 0
  )));
  assert.deepEqual(Array.from(runtime.process([text])[0]), [cut]);
  assert.equal(runtime.chunkText(text).join(""), text);
  assert.equal(tokens.find((token) => token.segment === "继").index, cut);
});

test("bounded windows retain full receptive context and own every gap exactly once", () => {
  // An asymmetric finite-context scorer detects lost, duplicated and misaligned
  // gaps. Boundary logits depend on all 17 tokens on each side, like this CNN.
  const radius = 16;
  function score(tokens) {
    return tokens.slice(1).map((_, gap) => {
      let value = 0;
      for (let i = gap - radius; i <= gap + 1 + radius; i++) {
        if (i >= 0 && i < tokens.length) value += tokens[i] * (i - gap + radius + 1);
      }
      return value;
    });
  }
  for (const length of [0, 1, 2, 255, 256, 257, 446, 447, 511, 512, 513, 1000]) {
    const tokens = Array.from({ length }, (_, i) => i + 1);
    const windows = [];
    const runtime = withModel((window) => { windows.push(window.length); return score(window); }, radius);
    assert.deepEqual(Array.from(runtime.scoreTokenWindows(tokens)), score(tokens), `length ${length}`);
    assert.ok(windows.every((size) => size <= 256));
    if (length <= 256) assert.equal(windows.length, length < 2 ? 0 : 1);
  }
});

test("windowed bundled CNN logits match unwindowed inference at every gap", () => {
  const sentence = Array.from("中文阅读需要了解前后文以及人物之间的关系才能判断这个位置是否适合切分保留API和８：３０也很重要");
  const tokens = Array.from({ length: 513 }, (_, i) => sentence[i % sentence.length]);
  const full = backend.scoreTokens(tokens);
  const windowed = chunker.scoreTokenWindows(tokens);
  assert.equal(windowed.length, full.length);
  for (let i = 0; i < full.length; i++) {
    assert.ok(Math.abs(windowed[i] - full[i]) <= 1e-6, `gap ${i}: ${windowed[i]} vs ${full[i]}`);
  }
  assert.deepEqual(chunker.gapProbabilities(windowed), chunker.gapProbabilities(full));
});
