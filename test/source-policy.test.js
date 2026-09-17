const test = require("node:test");
const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");

test("symbol windows use both nearest symbols and total code-point distance", async () => {
  const { symbolEnclosures, trainingPairs, DATA_POLICY } = await import("../training/text_policy.mjs");
  assert.equal(DATA_POLICY.symbol_window.max_distance, 15);
  const text = "集合 {空，a，aa，aaa，a…a(n个a)}";
  const first = Array.from(text).indexOf("，");
  assert.equal(symbolEnclosures(text, 14).has(first), false);
  assert.equal(symbolEnclosures(text, 15).get(first), 15);
  assert.deepEqual([...symbolEnclosures("甲，乙）", 64)], []);
  assert.deepEqual([...symbolEnclosures("（甲，乙", 64)], []);
  assert.deepEqual([...symbolEnclosures("甲，乙；丙。", 64)], []);
  // Symbols need not be matching brackets; emoji count as symbols too.
  assert.deepEqual([...symbolEnclosures("《甲，乙%", 4)], [[2, 4]]);
  assert.deepEqual([...symbolEnclosures("🌈𠮷，乙）", 4)], [[2, 4]]);
  assert.deepEqual([...symbolEnclosures("（甲\u0000，乙）", 64)], []);
  const pairs = (t, n) => [...trainingPairs(t, null, { symbolWindow: n })];
  assert.equal(pairs("（甲，乙）", 4).length, 0);
  assert.equal(pairs("（甲 ， 乙）", 4).length, 1); // Spaces count before normalization.
  assert.equal(pairs("（甲，乙\n丙）", 64).length, 1); // Cannot look across lines.
  assert.equal(pairs("甲，乙。丙，丁", 64).length, 3);
});

test("a rejected target remains a barrier without discarding intact neighboring targets", async () => {
  const { trainingPairs, trainingFragmentLines } = await import("../training/text_policy.mjs");
  const text = "前句，甲{，乙}，后句。";
  const rows = [...trainingPairs(text)];
  assert.deepEqual(rows.map(({ left, right }) => [left, right]), [["前句", "甲{"], ["乙}", "后句"]]);
  assert.equal([...trainingFragmentLines(text)][0][1].boundary_reasons[0], "symbol_window");
  assert(rows.every(({ left, right }) => !(left + right).includes("甲{乙}")));
  // An enclosed member of a punctuation run blocks the entire target.
  assert.deepEqual([...trainingPairs("（甲……乙）")], []);
});

test("discarded source spans poison touching pairs but preserve every physical newline", async () => {
  const { prepareSourceText, trainingPairs, DATA_POLICY } = await import("../training/text_policy.mjs");
  const newlinePattern = new RegExp(DATA_POLICY.line_boundaries, "gu");
  const breaks = ["\n", "\r\n", "\r", "\v", "\f", "\u001c", "\u001d", "\u001e", "\u0085", "\u2028", "\u2029"];
  const wrappers = [
    ["```", "```"], ["~~~", "~~~"], ["{{", "}}"], ["{|", "|}"], ["-{", "}-"],
    ["<ref>", "</ref>"], ["<!--", "-->"], ["<math>", "</math>"], ["<script>", "</script>"],
  ];
  for (const br of breaks) for (const [open, close] of wrappers) {
    const source = `甲，${open}坏部分${br}还有坏部分${close}乙。正常丙，正常丁。`;
    assert.deepEqual(prepareSourceText(source).match(newlinePattern), [br], source);
    assert.deepEqual([...trainingPairs(source)].map(({ left, right }) => [left, right]), [["正常丙", "正常丁"]], source);
  }
  for (const span of ["{{值}}", "{{外{{内}}尾}}", "<ref name='a'/>", "**强调**", "__强调__", "`code`", "<b>强调</b>"]) {
    const source = `前句，甲${span}乙，后句。完好甲，完好乙。`;
    assert.deepEqual([...trainingPairs(source)].map(({ left, right }) => [left, right]), [["后句", "完好甲"], ["完好甲", "完好乙"]], span);
  }
  for (const span of ["{{未闭合", "```未闭合", "<!--未闭合", "<ref>未闭合"]) {
    assert.equal([...trainingPairs(`甲，${span}乙，丙。`)].length, 0);
  }
});

test("source residue rejects pairs without reconnecting fragments or repairing inputs", async () => {
  const { trainingPairs } = await import("../training/text_policy.mjs");
  for (const bad of [String.raw`字面\r转义`, "�损坏", "网址https://example.com", "[链接](path)", "残留**标记", "残留&unknown;"]) {
    const rows = [...trainingPairs(`前句，${bad}，后句，继续。`)];
    assert.deepEqual(rows.map(({ left, right }) => [left, right]), [["后句", "继续"]], bad);
  }
  const rows = [...trainingPairs("调用__init__初始化，继续处理。")];
  assert.equal(rows.length, 1);
  assert(rows[0].left.includes("__init__"));
});

test("Wikipedia extraction cannot create targets from removed templates or multiline links", async () => {
  const { cleanWikipediaMarkup } = await import("../training/prepare_smoke_data.mjs");
  const { trainingPairs } = await import("../training/text_policy.mjs");
  for (const text of [
    "地點位於{{Coord|53|S|48|W}}，在福克蘭群島附近。",
    "邮政编码为{{邮编}}，INSEE市镇编码为{{编号}}。",
    "甲，[[目标\n|乙]]。", "甲，[[目标|乙\n丙]]。", "甲，[[文件:图\n片.jpg]]乙。",
    "甲，'''粗体\n内容'''乙。", "甲，<ref>引用\n内容</ref>乙。",
  ]) {
    const cleaned = cleanWikipediaMarkup(text);
    assert.deepEqual([...trainingPairs(cleaned)], [], text);
    assert.equal((cleaned.match(/\n/g) || []).length, (text.match(/\n/g) || []).length, text);
  }
  assert.equal(cleanWikipediaMarkup("[[目标|正常文字]]"), "正常文字");
});

test("Python and JavaScript agree on source barriers, Unicode windows and all source routes", async () => {
  const { trainingPairs, prepareSourceText, symbolEnclosures } = await import("../training/text_policy.mjs");
  const texts = [
    "前句，甲{，乙}，后句。", "集合 {空，a，aa，aaa，a…a(n个a)}", "甲，<ref name='a'/>乙，丙。",
    "调用__init__初始化，继续。", "甲，__加粗__乙，丙。", "甲，{{外{{内}}尾}}乙。",
    "甲，```代码\r\n另行```乙。丙，丁。", "甲，`代码`乙。", "🧪数值为𝟙，-𝟚，继续。",
    "甲，<SCRIPT>代码\n代码</SCRIPT>乙。", String.raw`甲，\r残留，乙。`, "甲，<!--不闭合",
    ...["（", "[", "🌈", "`", "@", "、", ":", "…"].flatMap((a) =>
      ["）", "!", "＿", "%", "🧪"].map((b) => `前句，${a}𠮷甲，乙${b}，后句。`)),
  ];
  const python = JSON.parse(execFileSync("python3", ["-c", String.raw`
import json,sys,types
sys.path.insert(0,'training')
sys.modules['pyarrow']=types.ModuleType('pyarrow');sys.modules['pyarrow.parquet']=types.ModuleType('pyarrow.parquet')
from text_policy import training_pairs,prepare_source_text,symbol_enclosures
from prepare_web_data import labeled_samples
from prepare_synthetic_data import adjacent_samples
rows=[]
for text in json.load(sys.stdin):
    pairs=list(training_pairs(text))
    assert pairs==list(labeled_samples(text))
    assert [(t,k) for t,k,_ in pairs]==adjacent_samples(text)
    rows.append([prepare_source_text(text),list(symbol_enclosures(text).items()),pairs])
print(json.dumps(rows,ensure_ascii=False))
`], {input: JSON.stringify(texts), encoding: "utf8"}));
  const js = texts.map((text) => [prepareSourceText(text), [...symbolEnclosures(text)],
    [...trainingPairs(text)].map(({left,right,punctuation}) => [left+right,Array.from(left).length-1,punctuation])]);
  assert.deepEqual(js, python);
});
