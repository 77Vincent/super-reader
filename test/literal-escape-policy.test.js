const test = require("node:test");
const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");

// Explicit agreed code points, independent of the production regular expression.
const range = (first, last) => Array.from({ length: last - first + 1 }, (_, i) => first + i);
const spaces = [0x0020, 0x00a0, 0x1680, ...range(0x2000, 0x200a), 0x202f, 0x205f, 0x3000];
const codePoints = [...new Set([
  ...spaces, ...range(0x200b, 0x200f), 0x2060, 0xfeff,
  0x0085, 0x2028, 0x2029, ...range(0, 0x001f), 0x007f,
])].sort((a, b) => a - b);
const unicodeEscapes = [...new Set(codePoints.flatMap((cp) => {
  const hex = cp.toString(16).padStart(4, "0");
  return [`u${hex}`, `u${hex.toUpperCase()}`];
}))];
const bodies = [...unicodeEscapes, "r", "n", "t", "b", "f", "v", "0"];
const markedDocument = (bad) => `引句。前甲，前乙，${bad}，后甲𠮷，后乙。`;
const expectedPairs = [["前甲前乙", 1, "，"], ["后甲𠮷后乙", 2, "，"]];
const tuples = (pairs) => [...pairs].map(({ left, right, punctuation }) =>
  [left + right, Array.from(left).length - 1, punctuation]);

test("Unicode residue rule covers exactly the agreed BMP code points and both hex cases", async () => {
  const { DATA_POLICY } = await import("../training/text_policy.mjs");
  const pattern = new RegExp(DATA_POLICY.sample_filter.fragment_patterns.literal_unicode_residue, "u");
  for (const uppercase of [false, true]) {
    const matches = [];
    for (let cp = 0; cp <= 0xffff; cp += 1) {
      const hex = cp.toString(16).padStart(4, "0");
      if (pattern.test(`\\u${uppercase ? hex.toUpperCase() : hex}`)) matches.push(cp);
    }
    assert.deepEqual(matches, codePoints);
  }
  assert.equal(codePoints.length, 60);
  assert(pattern.test(String.raw`\uFeFf`));
  // No universal unescaping: other code points and syntaxes are outside this rule.
  for (const text of [String.raw`\u4E00`, String.raw`\u3001`, String.raw`\u200`,
    String.raw`\u200G`, String.raw`\U00003000`, String.raw`\x20`, String.raw`\u{3000}`, "u3000"]) {
    assert.equal(pattern.test(text), false, text);
  }
});

test("literal escapes poison whole fragments without deletion, bridging, or index shifts", async () => {
  const { trainingPairs, trainingFragmentLines, prepareSourceText } = await import("../training/text_policy.mjs");
  for (const body of bodies) for (const slashes of [1, 2, 3]) {
    const escape = "\\".repeat(slashes) + body;
    const reason = body.startsWith("u") ? "literal_unicode_residue"
      : ["r", "n"].includes(body) ? "literal_line_escape" : "literal_control_escape";
    for (const bad of [escape, escape + "坏段", "坏" + escape + "段", "坏段" + escape]) {
      const text = markedDocument(bad);
      const counts = {};
      assert.equal(prepareSourceText(text), text, "Source must not be decoded or rewritten");
      assert.deepEqual(tuples(trainingPairs(text, counts)), expectedPairs, JSON.stringify(bad));
      const lines = [...trainingFragmentLines(text)];
      assert.equal(lines.length, 1, "Literal newlines must not become physical line breaks");
      assert.equal(lines[0][3].text, bad);
      assert(lines[0][3].reasons.includes(reason), JSON.stringify(bad));
      assert.equal(counts[reason], 2, "Both original touching pairs must be counted");
    }
  }
  const lines = [...trainingFragmentLines(markedDocument(String.raw`\u3000\u200B坏段`))];
  assert(lines[0][3].reasons.includes("literal_unicode_residue"));
  assert.deepEqual(tuples(trainingPairs(markedDocument(String.raw`\u3000\u200B坏段`))), expectedPairs);
});

test("actual spaces and unlisted Unicode escapes keep their text and code-point targets", async () => {
  const { trainingPairs } = await import("../training/text_policy.mjs");
  for (const cp of spaces) {
    const space = String.fromCodePoint(cp);
    const text = `引句。${space}甲${space}${space}𠮷${space}，乙丙。`;
    assert.deepEqual(tuples(trainingPairs(text)), [["甲 𠮷乙丙", 2, "，"]], cp.toString(16));
  }
  assert.deepEqual(tuples(trainingPairs(String.raw`引句。字符\u4E00，后面继续。`)),
    [[String.raw`字符\u4E00后面继续`, 7, "，"]]);
  const encodedSpace = JSON.parse(String.raw`{"text":"引句。\u3000甲乙，丙丁。"}`).text;
  const literalEscape = JSON.parse(String.raw`{"text":"引句。\\u3000甲乙，丙丁。"}`).text;
  assert.deepEqual(tuples(trainingPairs(encodedSpace)), [["甲乙丙丁", 1, "，"]]);
  assert.deepEqual(tuples(trainingPairs(literalEscape)), []);
});

test("all preparation routes agree on literal residues, counts, surviving IDs and old-policy rejection", async () => {
  const { trainingPairs, trainingFragmentLines, DATA_POLICY, requireDataPolicy } = await import("../training/text_policy.mjs");
  const { buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  const texts = bodies.flatMap((body) => [1, 2, 3].flatMap((slashes) => {
    const escape = "\\".repeat(slashes) + body;
    return [escape, escape + "坏段", "坏" + escape + "段", "坏段" + escape].map(markedDocument);
  }));
  texts.push(...spaces.map((cp) => `引句。甲${String.fromCodePoint(cp)}𠮷，乙丙。`),
    String.raw`引句。字符\u4E00，后面继续。`, markedDocument(String.raw`\u3000\u200B坏段`),
    markedDocument("＼ｕ３０００坏段"));
  const invalidPolicies = [
    { ...DATA_POLICY, standard: "unicode-context-v7" },
    { ...DATA_POLICY, sample_filter: { ...DATA_POLICY.sample_filter, version: "surface-noise-v4" } },
    ...["literal_unicode_residue", "literal_control_escape"].map((name) => ({
      ...DATA_POLICY, sample_filter: { ...DATA_POLICY.sample_filter,
        fragment_patterns: Object.fromEntries(Object.entries(DATA_POLICY.sample_filter.fragment_patterns)
          .filter(([key]) => key !== name)) },
    })),
  ];
  assert.equal(DATA_POLICY.standard, "unicode-context-v8");
  assert.equal(DATA_POLICY.sample_filter.version, "surface-noise-v5");
  assert.equal(DATA_POLICY.input_representation, "unicode-context-v1");
  invalidPolicies.forEach((policy) => assert.throws(() => requireDataPolicy(policy), /Data requires/u));
  const js = texts.map((text) => {
    const counts = {};
    const rows = tuples(trainingPairs(text, counts));
    const base = buildAdjacentSamples({ id: "escape", domain: "fixture", text });
    assert.deepEqual(base.map((r) => [r.tokens.join(""), r.target_index, r.punctuation]), rows);
    if (text.startsWith("引句。前甲")) {
      assert.deepEqual(base.map((r) => r.id), ["escape:line:0:pair:1", "escape:line:0:pair:4"]);
    }
    return [rows, counts, [...trainingFragmentLines(text)].flat().map((f) => [f.text, f.reasons])];
  });
  const python = JSON.parse(execFileSync("python3", ["-c", String.raw`
import sys,json,types,re
sys.path.insert(0,'training')
sys.modules['pyarrow']=types.ModuleType('pyarrow')
sys.modules['pyarrow.parquet']=types.ModuleType('pyarrow.parquet')
from text_policy import training_pairs,training_fragment_lines,require_data_policy,DATA_POLICY
from prepare_web_data import labeled_samples
from prepare_synthetic_data import adjacent_samples
texts,invalid,codepoints=json.load(sys.stdin)
require_data_policy(DATA_POLICY)
for policy in invalid:
    try: require_data_policy(policy)
    except ValueError: pass
    else: raise AssertionError('incompatible policy accepted')
pattern=re.compile(DATA_POLICY['sample_filter']['fragment_patterns']['literal_unicode_residue'])
for fmt in ['04x','04X']:
    assert [cp for cp in range(0x10000) if pattern.search('\\u'+format(cp,fmt))]==codepoints
results=[]
for text in texts:
    counts={}
    rows=list(training_pairs(text,counts))
    web_counts={}
    assert rows==list(labeled_samples(text,web_counts))
    assert counts==web_counts
    assert [(t,k) for t,k,_ in rows]==adjacent_samples(text)
    results.append([rows,counts,[[f['text'],f['reasons']] for line in training_fragment_lines(text) for f in line]])
print(json.dumps(results,ensure_ascii=False))
`], { input: JSON.stringify([texts, invalidPolicies, codePoints]), encoding: "utf8", maxBuffer: 8 * 1024 * 1024 }));
  assert.deepEqual(python, js);
});
