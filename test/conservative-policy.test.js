const test = require("node:test");
const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");

const newlines = ["\n", "\r\n", "\r", "\v", "\f", "\u001c", "\u001d", "\u001e", "\u0085", "\u2028", "\u2029"];
const controls = ["\u200b", "\u200c", "\u200d", "\u200e", "\u200f", "\u2060", "\ufeff"];

test("rejection symbols poison whole fragments, including fullwidth forms, without reconnecting", async () => {
  const { trainingPairs, trainingFragmentLines } = await import("../training/text_policy.mjs");
  for (const symbol of Array.from("{}<>&=｛｝＜＞＆＝")) {
    const text = `引句。正常甲，污染${symbol}内容，正常乙，正常丙。`;
    const rows = [...trainingPairs(text)].map(({ left, right }) => [left, right]);
    assert.deepEqual(rows, [["正常乙", "正常丙"]], symbol);
    const bad = [...trainingFragmentLines(text)][0].find(({ text }) => text.startsWith("污染"));
    assert(bad.reasons.includes("structural_symbols"), symbol);
  }
  for (const bad of ["国家{#blank#}1{#/blank#}", "<*次PBS重悬", "&n 实验方法原理", "“十三步”>神秘的描述", "关系式:y ="]) {
    assert.deepEqual([...trainingPairs(`引句。正常甲，${bad}，正常乙，正常丙。`)]
      .map(({ left, right }) => [left, right]), [["正常乙", "正常丙"]], bad);
  }
});

test("one invisible control rejects a fragment before normalization, even if the fragment becomes empty", async () => {
  const { trainingPairs } = await import("../training/text_policy.mjs");
  for (const control of controls) {
    for (const bad of [`${control}有内容`, `有${control}内容`, control]) {
      const text = `引句。正常甲，${bad}，正常乙，正常丙。`;
      assert.deepEqual([...trainingPairs(text)].map(({ left, right }) => [left, right]), [["正常乙", "正常丙"]], JSON.stringify(bad));
    }
  }
  const rows = [...trainingPairs("引句。笔记 本里有  多余空格，照常保留😀。")] ;
  assert.deepEqual(rows.map(({ left, right }) => [left, right]), [["笔记 本里有 多余空格", "照常保留😀"]]);
});

test("physical line edges discard first fragments and unclosed tails, not their surviving neighbors", async () => {
  const { trainingPairs, trainingFragmentLines } = await import("../training/text_policy.mjs");
  const fixtures = [
    ["甲，乙。", []],
    ["甲，乙，丙。", [["乙", "丙"]]],
    ["甲，乙，丙", []],
    ["甲，乙，丙，丁", [["乙", "丙"]]],
    ["甲，乙，丙，丁!", [["乙", "丙"]]], // ASCII ! does not close a fragment.
    ["甲，乙，丙，丁……", [["乙", "丙"], ["丙", "丁"]]],
    ["，甲，乙。", []], // Leading empty proxy does not make the first real fragment safe.
  ];
  for (const [text, expected] of fixtures) {
    assert.deepEqual([...trainingPairs(text)].map(({ left, right }) => [left, right]), expected, text);
  }
  for (const br of newlines) {
    const text = `牛肉面${br}店。她本人很少炫耀，大家继续交流。`;
    assert.deepEqual([...trainingPairs(text)].map(({ left, right }) => [left, right]), [["她本人很少炫耀", "大家继续交流"]], JSON.stringify(br));
    assert.deepEqual([...trainingPairs(`引句。两市共有21只，股价最低的是${br}某只股票。`)], []);
  }
  const fragments = [...trainingFragmentLines("首段，正常甲，正常乙，未完尾段")][0];
  assert.deepEqual(fragments.map(({ reasons }) => reasons), [["line_first_fragment"], [], [], ["line_unterminated_tail"]]);
});

test("Japanese with Han and semantic topic changes are not newly blacklisted", async () => {
  const { trainingPairs } = await import("../training/text_policy.mjs");
  const text = "引句。お気軽にご連絡ください。一緒に春節をお祝いしましょう。";
  assert.deepEqual([...trainingPairs(text)].map(({ left, right }) => [left, right]),
    [["お気軽にご連絡ください", "一緒に春節をお祝いしましょう"]]);
  assert.equal([...trainingPairs("引句。股票代表股权。圆柱误差很小。")].length, 1);
});

test("new gates agree across Python, JavaScript, synthetic, web and base generation", async () => {
  const { trainingPairs } = await import("../training/text_policy.mjs");
  const { buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  const texts = [
    "甲，乙。", "甲，乙，丙。", "甲，乙，丙，丁", "引句。好文，公式y =，后文，继续。",
    "引句。前文，{#blank#}，后文，继续。", "引句。お気軽にご連絡ください。一緒に春節をお祝いしましょう。",
    ...controls.flatMap((control) => [`引句。正常甲，${control}坏内容，正常乙，正常丙。`, `引句。正常甲，${control}，正常乙，正常丙。`]),
    ...newlines.map((br) => `原文首段，保留甲，保留乙${br}下一行首段，完好甲，完好乙。`),
    "引句。真实__init__调用，普通 空格，正常内容。", "引句。甲{{删去\n内容}}乙，正常丙，正常丁。",
  ];
  const js = texts.map((text) => [...trainingPairs(text)].map(({ left, right, punctuation }) => [left+right, Array.from(left).length-1, punctuation]));
  const base = texts.map((text) => buildAdjacentSamples({ id: "conservative", domain: "fixture", text })
    .map((row) => [row.tokens.join(""), row.target_index, row.punctuation]));
  assert.deepEqual(base, js);
  const python = JSON.parse(execFileSync("python3", ["-c", String.raw`
import sys,json,types
sys.path.insert(0,'training')
sys.modules['pyarrow']=types.ModuleType('pyarrow');sys.modules['pyarrow.parquet']=types.ModuleType('pyarrow.parquet')
from text_policy import training_pairs
from prepare_web_data import labeled_samples
from prepare_synthetic_data import adjacent_samples
results=[]
for text in json.load(sys.stdin):
    rows=list(training_pairs(text))
    assert rows==list(labeled_samples(text))
    assert [(t,k) for t,k,_ in rows]==adjacent_samples(text)
    results.append(rows)
print(json.dumps(results,ensure_ascii=False))
`], { input: JSON.stringify(texts), encoding: "utf8" }));
  assert.deepEqual(python, js);
});
