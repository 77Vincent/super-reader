// Read-only replay of the frozen teacher review against the production bundle.
// Run from the repository root: node training/diagnose_good_teacher_errors.mjs
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import vm from 'node:vm';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(import.meta.url);
const read = (p) => fs.readFileSync(path.join(root, p), 'utf8');
const sha = (data) => crypto.createHash('sha256').update(data).digest('hex');
const sourcePath = 'training/training-teacher-review-20260924.json';
const reviewPath = 'training/good-teacher-errors-20260924.json';
const source = JSON.parse(read(sourcePath));
const reviewed = new Map(JSON.parse(read(reviewPath)).cases.map(r => [r.id, r]));
const modelData = require('../src/boundary-model-data.js');
const model = require('../src/backend/inference.js');
const chunker = require('../src/backend/chunker.js');
const out = path.join(root, 'training/artifacts/good-teacher-diagnosis-v7-20260924');
fs.mkdirSync(out, { recursive: true });
let trace = [];
const sandbox = { SuperReaderModelBackend: model, Intl, recordDecision: r => trace.push(r) };
vm.createContext(sandbox);
const point = 'const boundaryAfter = selected === null ? null : selected + indexOffset;';
const original = read('src/backend/chunker.js');
assert.equal(original.split(point).length, 2);
vm.runInContext(original.replace(point, `${point}
globalThis.recordDecision({text, start, end, selected:boundaryAfter,
  candidateProbabilities: confidence, indexOffset,
  selectedSourceOffset: boundaryAfter === null ? null : tokens[boundaryAfter+1].index,
  tokens, protectedOffsets:Array.from(protectedBoundaryOffsets)});`), sandbox);
const instrumented = sandbox.SuperReaderChunker;
const render = (text, offsets) => {
  const set = new Set(offsets);
  return Array.from(text).reduce((r, c) => ({text:r.text+(set.has(r.offset)?'｜':'')+c, offset:r.offset+c.length}), {text:'',offset:0}).text;
};
const radius = model.getModelInfo().convolutionLayers;
const coverage = (g, n) => ({left:g+1, right:n-g-1,
  visible:Math.min(g+1,radius+1)+Math.min(n-g-1,radius+1),
  full:g+1<=radius+1 && n-g-1<=radius+1});
const rows = [];
const started = Date.now();
for (const r of source.cases) {
  const fd = fs.openSync(path.join(root, r.path), 'r');
  let line;
  try {
    const buffer = Buffer.alloc(Buffer.byteLength(r.text)*2+4096);
    const bytes = fs.readSync(fd, buffer, 0, buffer.length, r.offset);
    const newline = buffer.subarray(0,bytes).indexOf(10);
    assert(newline >= 0);
    line = buffer.subarray(0,newline+1);
  } finally { fs.closeSync(fd); }
  assert.equal(sha(line), r.row_sha256);
  const [text, gold] = JSON.parse(line);
  assert.equal(text, r.text); assert.equal(gold, r.target_index);
  const chars = Array.from(text);
  const scores = model.scoreTokens(chars);
  const probs = chunker.gapProbabilities(scores);
  const ranked = scores.map((_,g)=>g).sort((a,b)=>scores[b]-scores[a] || a-b);
  const predicted = ranked[0];
  const goldOffset = chars.slice(0,gold+1).join('').length;
  const predictedOffset = chars.slice(0,predicted+1).join('').length;
  const row = {id:r.id, domain:r.domain, stratum:r.stratum, text, teacherLabel:r.teacher_label,
    category:reviewed.get(r.id)?.category ?? null, gold, predicted, correct:gold===predicted,
    storedPrediction:r.prediction,
    n:chars.length, scores, confidence:probs[predicted], goldProbability:probs[gold],
    confidenceDelta:Math.abs(probs[predicted]-r.prediction.confidence),
    goldRank:ranked.indexOf(gold)+1, logitMargin:scores[predicted]-scores[gold],
    unknownCharacters:chars.filter(c=>!Object.hasOwn(modelData.vocabulary,c)),
    goldCoverage:coverage(gold,chars.length), predictedCoverage:coverage(predicted,chars.length),
    top5:ranked.slice(0,5).map(g=>({gap:g,probability:probs[g],
      split:render(text,[chars.slice(0,g+1).join('').length])})),
    goldSplit:render(text,[goldOffset]), predictedSplit:render(text,[predictedOffset]),
    lossWeight:r.loss_weight};
  if (reviewed.has(r.id)) {
    trace=[];
    const cuts = chunker.process([text])[0];
    const traced = Array.from(instrumented.process([text])[0]);
    assert.deepEqual(cuts,traced,`instrumentation parity: ${r.id}`);
    const clauses=chunker.splitClauses(text);
    row.backend={cuts,split:render(text,cuts),goldRecovered:cuts.includes(goldOffset),
      rawBadCutPresent:cuts.includes(predictedOffset),clauses,
      rawCandidateInsideWord:chunker.boundaryFallsInsideWord(text,predictedOffset,chunker.createSegmenter()),
      goldInsideWord:chunker.boundaryFallsInsideWord(text,goldOffset,chunker.createSegmenter()),
      rawCandidateHanNeighbors:/\p{Script=Han}/u.test(chars[predicted])&&/\p{Script=Han}/u.test(chars[predicted+1]),
      goldHanNeighbors:/\p{Script=Han}/u.test(chars[gold])&&/\p{Script=Han}/u.test(chars[gold+1]),
      visualLength:chunker.visualLength(text), trace};
  }
  rows.push(row);
  if(rows.length%50===0) console.log(`Scored and verified ${rows.length}/500 in ${((Date.now()-started)/1000).toFixed(1)}s`);
}
const result={scope:'Frozen 500 training-row reviews, including all 45 acceptable-target misses; no new sampling or training',
  model:model.getModelInfo(),node:process.version,icu:process.versions.icu,
  durationSeconds:(Date.now()-started)/1000,
  inputs:Object.fromEntries([sourcePath,reviewPath,'src/boundary-model-data.js','src/backend/inference.js','src/backend/chunker.js'].map(p=>[p,sha(read(p))])),
  verification:{trainingRowsVerified:rows.length,storedTop1Matches:rows.filter(r=>r.predicted===r.storedPrediction.predicted).length,backendInstrumentationParity:45,
    maxConfidenceDelta:Math.max(...rows.map(r=>r.confidenceDelta))},rows};
fs.writeFileSync(path.join(out,'replay.json'),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({verification:result.verification,seconds:result.durationSeconds},null,2));
