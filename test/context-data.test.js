const test = require("node:test");
const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");

const cases = [
  ["女：那可挺麻烦的，吃点儿治疗过敏的药吧。", "女:那可挺麻烦的", "吃点儿治疗过敏的药吧"],
  ["上午8:30至下午4:30；假日关门。", "上午8:30至下午4:30", "假日关门"],
  ["价格是1,000.50元，型号是AI-20。", "价格是1,000.50元", "型号是AI-20"],
  ["苹果、香蕉，放入（A）袋。", "苹果、香蕉", "放入(A)袋"],
  ["先看“🌈𠮷”，再读café。", "先看“🌈𠮷”", "再读café"],
  ["版本为３．１４，使用ﬃ字形。", "版本为3.14", "使用ffi字形"],
  ["3.高压系统设备安装完成80%，准备检查。", "3.高压系统设备安装完成80%", "准备检查"],
  ["F. Billinghurst负责设计，团队实施。", "F. Billinghurst负责设计", "团队实施"],
  ["请看docs.example.com，随后阅读说明。", "请看docs.example.com", "随后阅读说明"],
  ["头文件为unistd.h，编号是２．３。", "头文件为unistd.h", "编号是2.3"],
  ["序号是３．检查完毕。随后提交报告。", "序号是3.检查完毕", "随后提交报告"],
  ["先停顿……然后继续...最后结束。", "先停顿", "然后继续...最后结束"],
  ["坐标为(a,b)，继续计算。", "坐标为(a,b)", "继续计算"],
  ["调用f(x,y);然后返回，稍后重试。", "调用f(x,y);然后返回", "稍后重试"],
  ["共计１，０００人，准备出发。", "共计1,000人", "准备出发"],
  ["保留﹐﹔‼⁇︒｡，然后继续。", "保留,;!!??。。", "然后继续"],
];

test("context samples hide only the target proxy and use code-point gap indices", async () => {
  const { buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  for (const [text, left, right] of cases) {
    const rows = buildAdjacentSamples({ id: "context", domain: "fixture", text });
    assert.equal(rows.length, 1, text);
    const row = rows[0];
    assert.equal(row.tokens.join(""), left + right);
    assert.equal(row.target_index, Array.from(left).length - 1);
    assert.equal(row.input_representation, "unicode-context-v1");
  }
});

test("JavaScript and Python preparation agree on punctuation, numbers and Unicode", async () => {
  const { buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  const texts = [...cases.map(([text]) => text), "模型只看文字，“标点”仍保留。", "甲！？”；乙。", "English text,中文内容。"];
  const python = JSON.parse(execFileSync("python3", ["-c", String.raw`
import sys,json,types
sys.path.insert(0,'training')
sys.modules['pyarrow']=types.ModuleType('pyarrow')
sys.modules['pyarrow.parquet']=types.ModuleType('pyarrow.parquet')
from prepare_synthetic_data import adjacent_samples
from prepare_web_data import labeled_samples
texts=json.load(sys.stdin)
print(json.dumps({"synthetic":[adjacent_samples(t) for t in texts],
                  "web":[list(labeled_samples(t)) for t in texts]},ensure_ascii=False))
`], { input: JSON.stringify(texts), encoding: "utf8" }));
  const js = texts.map((text) => buildAdjacentSamples({id:"parity", domain:"fixture", text})
    .map((row) => [row.tokens.join(""), row.target_index]));
  assert.deepEqual(js, python.synthetic);
  const labeledJs = texts.map((text) => buildAdjacentSamples({ id: "parity", domain: "fixture", text })
    .map((row) => [row.tokens.join(""), row.target_index, row.punctuation]));
  assert.deepEqual(labeledJs, python.web);
});

test("non-proxy punctuation creates no target in any preparation path", async () => {
  const { buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  const texts = ["3.高压系统设备安装完成80%", "F. Billinghurst", "docs.example.com", "３．检查完毕", "甲...乙",
    ...[",", ";", "!", "?", "﹐", "﹔", "‼", "⁇", "︒", "｡", "、", ":"].map((mark) => `甲${mark}乙`)];
  for (const text of texts) assert.deepEqual(buildAdjacentSamples({ id: "dots", domain: "fixture", text }), []);
  execFileSync("python3", ["-c", String.raw`
import sys,json,types
sys.path.insert(0,'training')
sys.modules['pyarrow']=types.ModuleType('pyarrow')
sys.modules['pyarrow.parquet']=types.ModuleType('pyarrow.parquet')
from prepare_synthetic_data import adjacent_samples
from prepare_web_data import labeled_samples
for text in json.load(sys.stdin):
    assert adjacent_samples(text) == [], text
    assert list(labeled_samples(text)) == [], text
`], { input: JSON.stringify(texts) });
});

test("every raw Chinese proxy remains a label before normalization", async () => {
  const { buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  const { DATA_POLICY } = await import("../training/text_policy.mjs");
  const texts = DATA_POLICY.proxy_punctuation.map((mark) => `甲${mark}乙`);
  const js = texts.map((text) => buildAdjacentSamples({ id: "raw", domain: "fixture", text })
    .map((row) => [row.tokens.join(""), row.target_index, row.punctuation]));
  assert.deepEqual(js, DATA_POLICY.proxy_punctuation.map((mark) => [["甲乙", 0, mark]]));
  const python = JSON.parse(execFileSync("python3", ["-c", String.raw`
import sys,json,types
sys.path.insert(0,'training')
sys.modules['pyarrow']=types.ModuleType('pyarrow')
sys.modules['pyarrow.parquet']=types.ModuleType('pyarrow.parquet')
from prepare_web_data import labeled_samples
print(json.dumps([list(labeled_samples(t)) for t in json.load(sys.stdin)],ensure_ascii=False))
`], { input: JSON.stringify(texts), encoding: "utf8" }));
  assert.deepEqual(python, js);
});

test("signed numeric separators remain context and only two non-Han neighbors reject a target", async () => {
  const { buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  const { trainingPairs } = await import("../training/text_policy.mjs");
  const pair = (left, right, punctuation = "，") => [left + right, Array.from(left).length - 1, punctuation];
  const fixtures = [
    ["我得到(-8545，-27679)", []],
    ["坐标为(-8545，-27679)，继续计算。", [pair("坐标为(-8545,-27679)", "继续计算")]],
    ["读数为−1.5，+.25，－３．５，＋４，继续记录。", [pair("读数为−1.5,+.25,-3.5,+4", "继续记录")]],
    ["数值为-1e-3，+2E+4，继续计算。", [pair("数值为-1e-3,+2E+4", "继续计算")]],
    ["数值为 1 ， - 2 ， +3 ，继续。", [pair("数值为 1 , - 2 , +3", "继续")]],
    ["输入𠮷１２，－３，完成。", [pair("输入𠮷12,-3", "完成")]],
    ["🧪数值为𝟙，-𝟚，继续。", [pair("🧪数值为1,-2", "继续")]],
    ["甲A，B乙，丙丁。", [pair("B乙", "丙丁")]],
    ["甲，ABC；DEF，乙。", [pair("甲", "ABC"), pair("DEF", "乙")]],
    ["使用API ， HTTP处理。", []],
    ["得分42；64结束。", []],
    ["“甲”，（乙）。", []],
    ["“甲”，乙。", [pair('“甲”', "乙")]],
    ["甲，（乙）。", [pair("甲", "(乙)")]],
    ["时间8:30，明天继续。", [pair("时间8:30", "明天继续")]],
    ["今天，8:30出发。", [pair("今天", "8:30出发")]],
    ["𠮷，API。", [pair("𠮷", "API")]],
    ["〇，API。", [pair("〇", "API")]],
    ["甲，-暂停。", [pair("甲", "-暂停")]],
    ["ASCII,-2，下一句。", [pair("ASCII,-2", "下一句")]],
  ];
  const js = fixtures.map(([text]) => buildAdjacentSamples({ id: "boundary-context", domain: "fixture", text })
    .map((row) => [row.tokens.join(""), row.target_index, row.punctuation]));
  fixtures.forEach(([text, expected], i) => assert.deepEqual(js[i], expected, text));
  const counts = {};
  const retained = Array.from(trainingPairs("甲A，B乙，丙丁。", counts));
  assert.equal(retained.length, 1);
  assert.deepEqual(counts, { pairs: 1, both_neighbors_non_han: 1 });
  const python = JSON.parse(execFileSync("python3", ["-c", String.raw`
import sys,json,types
sys.path.insert(0,'training')
sys.modules['pyarrow']=types.ModuleType('pyarrow')
sys.modules['pyarrow.parquet']=types.ModuleType('pyarrow.parquet')
from prepare_web_data import labeled_samples
from prepare_synthetic_data import adjacent_samples, split_into_fragments
from text_policy import training_pairs
assert split_into_fragments('我得到(-8545，-27679)')==['我得到(-8545,-27679)']
result=[]
for text in json.load(sys.stdin):
    rows=list(labeled_samples(text))
    assert [(t,k) for t,k,_ in rows]==adjacent_samples(text)
    result.append(rows)
counts={}
assert list(training_pairs('甲A，B乙，丙丁。',counts))==[('B乙丙丁',1,'，')]
assert counts=={'pairs':1,'both_neighbors_non_han':1}
print(json.dumps(result,ensure_ascii=False))
`], { input: JSON.stringify(fixtures.map(([text]) => text)), encoding: "utf8" }));
  assert.deepEqual(python, js);
});

test("Wikipedia extraction preserves raw punctuation until sample generation", async () => {
  const { wikipediaDocumentsFromXml, buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  const [document] = wikipediaDocumentsFromXml('<mediawiki><page><title>示例</title><ns>0</ns><id>1</id><revision><text>甲,乙，丙﹐丁。</text></revision></page></mediawiki>');
  assert.equal(document.text, "甲,乙，丙﹐丁。");
  const [row] = buildAdjacentSamples(document);
  assert.equal(row.tokens.join(""), "甲,乙丙,丁");
  assert.equal(row.target_index, 2);
  assert.equal(row.punctuation, "，");
});

test("both policy validators compare the whitelist and normalization semantics", async () => {
  const { DATA_POLICY, requireDataPolicy } = await import("../training/text_policy.mjs");
  const reordered = { ...DATA_POLICY, proxy_punctuation: [...DATA_POLICY.proxy_punctuation].reverse() };
  requireDataPolicy(reordered);
  const invalid = [
    { ...DATA_POLICY, proxy_punctuation: ["，"] },
    { ...DATA_POLICY, proxy_punctuation: [...DATA_POLICY.proxy_punctuation, ","] },
    { ...DATA_POLICY, normalization: "NFKC before proxy detection" },
    { ...DATA_POLICY, numeric_punctuation: "none" },
    { ...DATA_POLICY, boundary_context: undefined },
    { ...DATA_POLICY, boundary_context: "Require both neighbors to be Han" },
    { ...DATA_POLICY, line_boundaries: "none" },
    { ...DATA_POLICY, sample_filter: { version: "obsolete" } },
  ];
  invalid.forEach((summary) => assert.throws(() => requireDataPolicy(summary), /Data requires/u));
  execFileSync("python3", ["-c", String.raw`
import sys,json
sys.path.insert(0,'training')
from text_policy import require_data_policy
valid,invalid=json.load(sys.stdin)
require_data_policy(valid)
for summary in invalid:
    try:
        require_data_policy(summary)
    except ValueError:
        continue
    raise AssertionError('incompatible policy was accepted')
`], { input: JSON.stringify([reordered, invalid]) });
});

test("surface rules reject affected pairs without bridging or crossing actual lines", async () => {
  const { buildAdjacentSamples, wikipediaDocumentsFromXml } = await import("../training/prepare_smoke_data.mjs");
  const fixtures = [
    ["实用性强。\nEND", []],
    ["正常甲，正常乙。\r\n正常丙，正常丁。", [["正常甲正常乙", 2, "，"], ["正常丙正常丁", 2, "，"]]],
    ["甲，$P$乙，丙，丁。", [["丙丁", 0, "，"]]],
    ["正常正文。...更多", []],
    ["更多的人喜欢阅读，大家继续学习。", [["更多的人喜欢阅读大家继续学习", 7, "，"]]],
    ["博·杰克逊（，）是运动员，是明星。正常段，继续读。", [["是明星正常段", 2, "。"], ["正常段继续读", 2, "，"]]],
    ["材料强度为___MPa，准备检查。", []],
    ["材料强度为 ___ MPa，准备检查。", []],
    ["调用__init__完成初始化，准备检查。", [["调用__init__完成初始化准备检查", 14, "，"]]],
    ["## 今天开始，明天继续。正常段，继续读。", [["明天继续正常段", 3, "。"], ["正常段继续读", 2, "，"]]],
    ["笔记 本里有  多余空格，照常保留。", [["笔记 本里有 多余空格照常保留", 10, "，"]]],
    ["股票代表股权。圆柱误差很小。", [["股票代表股权圆柱误差很小", 5, "。"]]],
  ];
  const js = fixtures.map(([text]) => buildAdjacentSamples({ id: "surface", domain: "fixture", text })
    .map((r) => [r.tokens.join(""), r.target_index, r.punctuation]));
  fixtures.forEach(([text, expected], i) => assert.deepEqual(js[i], expected, text));
  const python = JSON.parse(execFileSync("python3", ["-c", String.raw`
import sys,json,types
sys.path.insert(0,'training')
sys.modules['pyarrow']=types.ModuleType('pyarrow')
sys.modules['pyarrow.parquet']=types.ModuleType('pyarrow.parquet')
from prepare_web_data import labeled_samples
from prepare_synthetic_data import adjacent_samples
rows=[]
for text in json.load(sys.stdin):
    result=list(labeled_samples(text))
    assert [(t,k) for t,k,_ in result]==adjacent_samples(text)
    rows.append(result)
print(json.dumps(rows,ensure_ascii=False))
`], {input: JSON.stringify(fixtures.map(([text]) => text)),encoding:"utf8"}));
  assert.deepEqual(js, python);
  const [doc] = wikipediaDocumentsFromXml('<mediawiki><page><title>示例</title><ns>0</ns><id>1</id><revision><text>正文结束。\nEND\n正常甲，正常乙。</text></revision></page></mediawiki>');
  assert.equal(buildAdjacentSamples(doc).length, 1);
});

test("vocabulary extension preserves existing IDs and assigns context distinct IDs", () => {
  execFileSync("python3", ["-c", String.raw`
import sys
sys.path.insert(0,'training')
from build_context_vocabulary import extend_vocabulary
old={'<pad>':0,'<unk>':1,'甲':2,'乙':3}
new=extend_vocabulary(old,{'新':100,'甲':1000},256)
assert all(new[t]==i for t,i in old.items())
assert len({new[c] for c in '0123456789:、ABC '})==len('0123456789:、ABC ')
assert '新' in new
assert sorted(new.values())==list(range(len(new)))
`]);
});
