#!/usr/bin/env node
// Fixed weights, text, tokens, protections and threshold; only child inference varies.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createContext, runInContext } from "node:vm";
import { cpus, platform, arch } from "node:os";

const require = createRequire(import.meta.url);
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const backend = require("../src/backend/inference.js");
const chunker = require("../src/backend/chunker.js");
const source = readFileSync(resolve(root, "src/backend/chunker.js"), "utf8");
const casesBytes = readFileSync(resolve(root, "training/evaluation-cases.json"));
const demo = readFileSync(resolve(root, "demo.html"), "utf8");
const sha = (value) => createHash("sha256").update(value).digest("hex");
const cases = JSON.parse(casesBytes);
cases.push(
  { id: "divider", texts: ["同一个无标点子句内的短语块用细竖线分隔；"] },
  { id: "driving", texts: ["驾驶员会出于本能进行向左打方向盘等避险动作，"] },
  { id: "euler", texts: ['欧拉法是在积分无法直接计算时，用"无数个小矩形累加"来近似积分，因此它被称为一种数值积分方法（numerical integration method）。'] },
  { id: "time", texts: ["女：我们计划明天上午8:30出发前往目的地，请记住。"] },
);
const primaryCases = cases.length;
cases.push(
  { id: "reported-database", texts: ["币安数据库里的一条记录（IOU/欠条）"] },
  { id: "reported-child", texts: ["能够更加轻松地找到句子中的重点"] },
);
// This demo has simple paragraph/strong markup; keep its text nodes separate.
for (const [index, match] of [...demo.matchAll(/<p>([\s\S]*?)<\/p>/gu)].entries()) {
  cases.push({ id: `current-demo-${index + 1}`, texts: match[1].split(/<[^>]*>/u).filter((t) => /\p{Script=Han}/u.test(t)) });
}
const methods = {
  once: { minConfidence: .9, scoringStrategy: "fixed" },
  recursive: { minConfidence: .9, scoringStrategy: "recursive-model" },
};
const needle = "      const boundaryAfter = selected === null ? null : selected + indexOffset;";
assert.equal(source.split(needle).length, 2, "Trace hook must match exactly one decision point");
// Read-only instrumentation is used only for explanations, never for timings.
const tracedSource = source.replace(needle, needle + `
      globalThis.recordDecision({ text, tokens, start, end, fresh, indexOffset, selected,
        boundaryAfter, confidence, originalConfidence: initial.confidence });`);
const render = (texts, offsets) => texts.map((text, i) => {
  let previous = 0;
  return [...offsets[i], text.length].map((end) => {
    const result = text.slice(previous, end); previous = end; return result;
  }).join("｜");
});

function diagnose(texts, options) {
  const decisions = [];
  let calls = 0, inputTokens = 0;
  const context = createContext({ Intl, SuperReaderModelBackend: {
    getModelInfo: backend.getModelInfo,
    scoreTokens(tokens) { calls++; inputTokens += tokens.length; return backend.scoreTokens(tokens); },
  }, recordDecision(d) {
    const offset = (index) => index === d.tokens.length ? d.text.length : index === 0 ? 0 : d.tokens[index].index;
    const fragment = d.text.slice(offset(d.start), offset(d.end));
    const probability = d.selected === null ? null : d.confidence[d.selected];
    if (d.selected !== null) assert.ok(probability > .9);
    decisions.push({ originalClause: d.text, fragment, start: offset(d.start), end: offset(d.end), fresh: d.fresh,
      maximumProbability: Math.max(...d.confidence.slice(d.start - d.indexOffset, d.end - 1 - d.indexOffset)),
      probability, originalProbability: d.selected === null ? null : d.originalConfidence[d.boundaryAfter],
      cut: d.boundaryAfter === null ? null : offset(d.boundaryAfter + 1),
      renderedDecision: d.boundaryAfter === null ? fragment :
        d.text.slice(offset(d.start), offset(d.boundaryAfter + 1)) + "｜" + d.text.slice(offset(d.boundaryAfter + 1), offset(d.end)),
    });
  } });
  runInContext(tracedSource, context);
  const offsets = Array.from(context.SuperReaderChunker.process(texts, options), (cuts) => Array.from(cuts));
  assert.deepEqual(offsets, chunker.process(texts, options));
  texts.forEach((text, i) => {
    assert.equal(chunker.chunkText(text, options).join(""), text);
    assert.deepEqual(offsets[i], [...new Set(offsets[i])].sort((a, b) => a - b));
    for (const cut of offsets[i]) {
      assert.ok(cut > 0 && cut < text.length);
      assert.doesNotMatch(text[cut], /[\uDC00-\uDFFF]/u);
    }
  });
  return { offsets, rendered: render(texts, offsets), calls, inputTokens, cuts: offsets.flat().length, decisions };
}

const rows = cases.map((example) => {
  const once = diagnose(example.texts, methods.once), recursive = diagnose(example.texts, methods.recursive);
  const roots = (result) => result.decisions.filter((d) => d.start === 0 && d.end === d.originalClause.length);
  assert.deepEqual(roots(once), roots(recursive), "The first decision must be identical");
  once.offsets.forEach((cuts, i) => cuts.forEach((cut) => assert.ok(recursive.offsets[i].includes(cut))));
  return { ...example, once, recursive, changed: JSON.stringify(once.offsets) !== JSON.stringify(recursive.offsets) };
});

const warmups = 2, rounds = 9;
for (let round = 0; round < warmups; round++) {
  for (const options of Object.values(methods)) for (const row of rows) chunker.process(row.texts, options);
}
const times = Object.fromEntries(Object.keys(methods).map((method) => [method, rows.map(() => [])]));
for (let round = 0; round < rounds; round++) {
  for (const index of (round % 2 ? rows.map((_, i) => i).reverse() : rows.map((_, i) => i))) {
    for (const method of (round % 2 ? ["recursive", "once"] : ["once", "recursive"])) {
      const started = performance.now();
      const result = chunker.process(rows[index].texts, methods[method]);
      times[method][index].push(performance.now() - started);
      assert.deepEqual(result, rows[index][method].offsets);
    }
  }
}
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
function summarize(selectedRows, count) {
  const stats = Object.fromEntries(Object.keys(methods).map((method) => [method, {
    cuts: selectedRows.reduce((n, row) => n + row[method].cuts, 0),
    calls: selectedRows.reduce((n, row) => n + row[method].calls, 0),
    inputTokens: selectedRows.reduce((n, row) => n + row[method].inputTokens, 0),
    medianMs: median(Array.from({ length: rounds }, (_, r) => times[method].slice(0, count).reduce((n, row) => n + row[r], 0))),
  }]));
  return { cases: selectedRows.length, textNodes: selectedRows.reduce((n, row) => n + row.texts.length, 0),
    changedCases: selectedRows.filter((row) => row.changed).map((row) => row.id), ...stats,
    relativeTimeIncrease: stats.recursive.medianMs / stats.once.medianMs - 1 };
}
assert.equal(readFileSync(resolve(root, "src/backend/chunker.js"), "utf8"), source);
const report = { createdAt: new Date().toISOString(), passed: true, model: backend.getModelInfo(),
  bundleSha256: sha(readFileSync(resolve(root, "src/boundary-model-data.js"))), chunkerSha256: sha(source),
  sourceCasesSha256: sha(casesBytes), demoSha256: sha(demo), completeInputSha256: sha(JSON.stringify(cases)),
  settings: { threshold: .9, warmups, rounds, order: "alternating AB/BA and forward/reverse cases", protections: "fixed from original clause" },
  environment: { node: process.version, platform: platform(), arch: arch(), cpu: cpus()[0].model, trainingRunning: true },
  scope: "Warm Node backend processing; no DOM, Chrome messaging, human segmentation labels or new full-validation inference.",
  primary: summarize(rows.slice(0, primaryCases), primaryCases), all: summarize(rows, rows.length), rows, times };
const output = resolve(root, "training/artifacts/recursive-confidence-16conv-epoch1-20260915/report.json");
mkdirSync(dirname(output), { recursive: true });
writeFileSync(output, JSON.stringify(report, null, 2) + "\n");
console.log(JSON.stringify({ passed: report.passed, primary: report.primary, all: report.all, output }, null, 2));
