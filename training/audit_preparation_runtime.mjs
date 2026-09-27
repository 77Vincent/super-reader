// Read-only input-contract audit of the existing 24,000-row probability sample.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
const require=createRequire(import.meta.url);
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const read=p=>fs.readFileSync(path.join(root,p),'utf8');
const c=require('../src/backend/chunker.js');
const model=require('../src/boundary-model-data.js');
const samplePath='training/artifacts/training-teacher-audit-v7-20260924/sample.jsonl';
const records=read(samplePath).trim().split('\n').map(JSON.parse);
const profile=JSON.parse(read('training/artifacts/training-teacher-audit-v7-20260924/profile.json'));
const predictions=new Map(JSON.parse(read('training/artifacts/training-teacher-audit-v7-20260924/predictions.json')).rows.map(r=>[r.id,r]));
const segmenter=c.createSegmenter();
const groups={}; const examples={}; const rows=[];
for(const r of records){
  const text=r.text, chars=Array.from(text), g=r.target_index;
  const targetOffset=chars.slice(0,g+1).join('').length;
  const clauses=c.splitClauses(text);
  const tokens=c.tokenizeContext(text).map(t=>t.segment);
  const unknown=chars.filter(t=>!Object.hasOwn(model.vocabulary,t));
  const flags={
    raw_input_changed_by_backend_tokenizer:tokens.join('')!==text,
    pre_split_into_multiple_clauses:clauses.length>1,
    target_rejected_by_han_neighbors:!(/\p{Script=Han}/u.test(chars[g])&&/\p{Script=Han}/u.test(chars[g+1])),
    target_inside_intl_word:c.boundaryFallsInsideWord(text,targetOffset,segmenter),
    whole_input_within_12_visual_units:c.visualLength(text)<=12,
    contains_unknown:unknown.length>0,
    above_256_tokens:chars.length>256,
    contains_ascii_proxy:/[,.;!?]/u.test(text),
    contains_enumeration_comma:text.includes('、'),
  };
  const group=groups[r.stratum]??={n:0,tokens:0,unknownTokens:0,counts:{},errors:{}};
  group.n++;group.tokens+=chars.length;group.unknownTokens+=unknown.length;
  for(const [k,v] of Object.entries(flags))if(v){
    group.counts[k]=(group.counts[k]??0)+1;
    group.errors[k]=(group.errors[k]??0)+Number(!predictions.get(r.id).correct);
    if((examples[k]??=[]).length<8)examples[k].push({id:r.id,stratum:r.stratum,text,
      teacher:text.slice(0,targetOffset)+'｜'+text.slice(targetOffset),
      backendTokens:tokens.join(''),clauses,unknown});
  }
  rows.push({id:r.id,stratum:r.stratum,flags});
}
const weighted={};
for(const [stratum,g] of Object.entries(groups)){
  const p=profile.find(p=>p.stratum===stratum);
  if(p.sample_n!==g.n)throw new Error('Cohort mismatch');
  for(const [key,n] of Object.entries(g.counts))weighted[key]=(weighted[key]??0)+p.row_share*n/g.n;
}
const summary={scope:'Existing stratified 24,000 training rows; rates standardized by full training source populations, not fresh sample',
  sampleSha256:crypto.createHash('sha256').update(read(samplePath)).digest('hex'),
  inputs:Object.fromEntries(['src/backend/chunker.js','src/boundary-model-data.js'].map(p=>[p,crypto.createHash('sha256').update(read(p)).digest('hex')])),
  node:process.version,icu:process.versions.icu,groups,populationStandardizedRates:weighted,examples,rows};
const out=path.join(root,'training/artifacts/data-preparation-audit-v7-20260925');fs.mkdirSync(out,{recursive:true});
fs.writeFileSync(path.join(out,'runtime-contract.json'),JSON.stringify(summary,null,2)+'\n');
console.log(JSON.stringify(weighted,null,2));
