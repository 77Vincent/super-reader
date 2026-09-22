// Run the unchanged browser inference code in isolated contexts. This measures
// Node/V8 CPU inference, not page layout, Chrome end-to-end latency or GPU time.
import { readFileSync, writeFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import { performance } from 'node:perf_hooks';

const run = process.argv[2];
if (!run) throw new Error('Usage: node benchmark_width.mjs RUN_DIRECTORY [--depth]');
const depthComparison = process.argv.includes('--depth');
const arms = depthComparison ? [16, 20] : [192, 256];
const folderFor = arm => join(run, `${depthComparison ? 'layers' : 'ch'}${arm}/final`);
const implementation = readFileSync(join(run, 'source/src/backend/inference.js'), 'utf8');
const backends = new Map();
const softmax = scores => {
  const maximum = Math.max(...scores);
  const e = Array.from(scores, score => Math.exp(score - maximum));
  const total = e.reduce((a, b) => a + b, 0);
  return e.map(value => value / total);
};
const allCases = new Map();
for (const width of arms) {
  const folder = folderFor(width);
  const context = vm.createContext({ Buffer });
  vm.runInContext(readFileSync(join(folder, 'boundary-model-data.js'), 'utf8'), context);
  vm.runInContext(implementation, context);
  const api = context.SuperReaderModelBackend;
  assert.equal(api.getModelInfo().channels, depthComparison ? 192 : width);
  if (depthComparison) assert.equal(api.getModelInfo().residualBlocks, width / 2);
  backends.set(width, api);
  const cases = JSON.parse(readFileSync(join(folder, 'reference.json'), 'utf8')).cases;
  allCases.set(width, cases);
  for (const example of cases) {
    const scores = api.scoreTokens(Array.from(example.text));
    assert.equal(scores.length, example.scores.length);
    assert.equal(scores.indexOf(Math.max(...scores)), example.bestGap);
    const expected = softmax(example.scores);
    softmax(scores).forEach((probability, i) => assert.ok(Math.abs(probability - expected[i]) < 3e-5));
    scores.forEach((score, i) => assert.ok(Math.abs(score - example.scores[i]) < 2e-4 + 2e-5 * Math.abs(example.scores[i])));
  }
}
assert.deepEqual(allCases.get(arms[0]).map(c => c.text), allCases.get(arms[1]).map(c => c.text));
const cases = allCases.get(arms[0]).map(c => ({ text: c.text, tokens: Array.from(c.text) }));
for (let warmup = 0; warmup < 3; warmup++) {
  for (const api of backends.values()) for (const example of cases) api.scoreTokens(example.tokens);
}
const times = Object.fromEntries(arms.map(arm => [arm, cases.map(() => [])]));
for (let repetition = 0; repetition < 8; repetition++) {
  for (const width of repetition % 2 ? [...arms].reverse() : arms) {
    cases.forEach((example, i) => {
      const start = performance.now();
      backends.get(width).scoreTokens(example.tokens);
      times[width][i].push(performance.now() - start);
    });
  }
}
const median = values => {
  const sorted = values.toSorted((a, b) => a - b);
  return (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.floor(sorted.length / 2)]) / 2;
};
const result = {
  runtime: process.version, platform: `${process.platform}/${process.arch}`,
  method: 'unchanged JS backend in isolated V8 contexts, 3 warmups, 8 alternating-order repetitions; excludes parsing, rendering and model load',
  metal_reference_checks_passed: true,
  arms: Object.fromEntries(arms.map(width => [width, {
    model_bytes: statSync(join(folderFor(width), 'boundary-model-data.js')).size,
    suite_median_ms: median(Array.from({ length: 8 }, (_, r) => times[width].reduce((sum, row) => sum + row[r], 0))),
    cases: cases.map((example, i) => ({ text: example.text, tokens: example.tokens.length, median_ms: median(times[width][i]) })),
  }])),
};
writeFileSync(join(run, 'browser-benchmark.json'), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result, null, 2));
