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

test("period-only text creates no target in any preparation path", async () => {
  const { buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  const texts = ["3.高压系统设备安装完成80%", "F. Billinghurst", "docs.example.com", "３．检查完毕", "甲...乙"];
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
