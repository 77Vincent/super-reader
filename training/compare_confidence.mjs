#!/usr/bin/env node
// Compare the same bundled model and backend rules; only abstention varies.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(import.meta.url);
const chunker = require("../src/backend/chunker.js");
const backend = require("../src/backend/inference.js");
const cases = JSON.parse(readFileSync(resolve(root, "training/evaluation-cases.json")));
cases.push(
  { id: "divider", texts: ["同一个无标点子句内的短语块用细竖线分隔；"] },
  { id: "driving", texts: ["驾驶员会出于本能进行向左打方向盘等避险动作，"] },
  { id: "euler", texts: ['欧拉法是在积分无法直接计算时，用"无数个小矩形累加"来近似积分，因此它被称为一种数值积分方法（numerical integration method）。'] },
  { id: "time", texts: ["女：我们计划明天上午8:30出发前往目的地，请记住。"] },
);

const rows = cases.map((example) => ({ ...example, results: [0, .8, .9, .95, .99].map((minConfidence) => {
  const options = { minConfidence, scoringStrategy: "fixed" };
  const offsets = chunker.process(example.texts, options);
  const rendered = example.texts.map((text, index) => {
    assert.deepEqual(offsets[index], [...new Set(offsets[index])].sort((a, b) => a - b));
    let previous = 0;
    return [...offsets[index], text.length].map((end) => {
      const part = text.slice(previous, end);
      previous = end;
      return part;
    }).join("｜");
  });
  for (const text of example.texts) {
    assert.equal(chunker.chunkText(text, options).join(""), text);
    if (!minConfidence) continue;
    for (const clause of chunker.splitClauses(text)) {
      const cuts = chunker.process([clause], options)[0];
      assert.ok(cuts.length <= 1, "thresholds above 50% permit at most one cut per clause");
      if (!cuts.length) continue;
      const tokens = chunker.tokenizeContext(clause);
      const scores = [];
      for (let start = 0; start < tokens.length - 1; start += 255) {
        scores.push(...backend.scoreTokens(tokens.slice(start, start + 256).map((token) => token.segment)));
      }
      const probabilities = chunker.gapProbabilities(scores);
      for (const offset of cuts) {
        const tokenIndex = tokens.findIndex((token) => token.index === offset);
        assert.ok(tokenIndex > 0 && probabilities[tokenIndex - 1] > minConfidence);
      }
    }
  }
  return { minConfidence, cuts: offsets.reduce((sum, cuts) => sum + cuts.length, 0), offsets, rendered };
}) }));
const totals = rows[0].results.map(({ minConfidence }, index) => ({
  minConfidence, cuts: rows.reduce((sum, row) => sum + row.results[index].cuts, 0),
}));
const sha256 = (path) => createHash("sha256").update(readFileSync(resolve(root, path))).digest("hex");
const report = { model: backend.getModelInfo(), bundleSha256: sha256("src/boundary-model-data.js"),
  chunkerSha256: sha256("src/backend/chunker.js"), sourceCasesSha256: sha256("training/evaluation-cases.json"),
  scope: "Fixed original-clause probabilities for the historical threshold comparison; not the default recursive softmax. Identical texts, weights, preprocessing, raw-logit ranking and protections; only minConfidence changes. No human quality labels.",
  passed: true, totals, rows };
const output = resolve(root, "training/artifacts/confidence-gate-16conv-epoch1-20260915/examples.json");
mkdirSync(dirname(output), { recursive: true });
writeFileSync(output, JSON.stringify(report, null, 2) + "\n");
console.log(JSON.stringify({ passed: report.passed, totals, output }, null, 2));
