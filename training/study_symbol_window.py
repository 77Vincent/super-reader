#!/usr/bin/env python3
"""Bounded local policy study. No training, downloads, or full-corpus preparation."""
import argparse, bz2, hashlib, heapq, inspect, json, random, re, subprocess, sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'training/.deps'))
import text_policy as policy
from prepare_synthetic_data import quality_document, clean_document
SEED=2026091720
BOUNDS=[4,8,12,16,24,32,48,64]
WINDOWS=[0,4,6,8,10,12,15,16,20,24,32,48,64]
POOLS={'web':'full-web-yield-20260916/documents.jsonl', 'l3':'l3-study-20260916/documents.jsonl',
       'wiki':'all-local-quick-yield-20260917/wiki-documents.jsonl','clue':'chinese-line-quality-20260917/clue-documents.jsonl'}

class Reservoir:
    def __init__(self,count,seed):self.count=count;self.rng=random.Random(seed);self.rows=[];self.n=0
    def add(self,row):
        self.n+=1
        if len(self.rows)<self.count:self.rows.append(row)
        else:
            i=self.rng.randrange(self.n)
            if i<self.count:self.rows[i]=row

def fresh_wiki(out):
    target=out/'wiki-current-extraction.jsonl'
    signature=hashlib.sha256(b''.join((ROOT/'training'/n).read_bytes() for n in
        ('prepare_smoke_data.mjs','text_policy.mjs','text-policy.json'))).hexdigest()
    stamp=out/'wiki-extraction-signature.txt'
    if target.exists() and stamp.exists() and stamp.read_text()==signature:return target
    pool=[json.loads(s) for s in (ROOT/'training/artifacts'/POOLS['wiki']).read_text().splitlines()]
    by_offset={}
    for i,d in enumerate(pool):by_offset.setdefault(d['stream_offset'],{})[d['id'].split(':')[-1]]=(i,d)
    rawrows=[]
    with (ROOT/'training/data/raw/zhwiki-latest-pages-articles-multistream.xml.bz2').open('rb') as f:
        for offset,wanted in by_offset.items():
            f.seek(offset);decoder=bz2.BZ2Decompressor();parts=[]
            while not decoder.eof:parts.append(decoder.decompress(f.read(65536)))
            xml=b''.join(parts).decode('utf8')
            for rawpage in re.findall(r'<page>.*?</page>',xml,re.S):
                page=ET.fromstring(rawpage);key=page.findtext('id')
                if key in wanted:
                    index,old=wanted[key];rawrows.append((index,{**old,'text':page.findtext('revision/text') or ''}))
    assert len(rawrows)==len(pool)
    rawpath=out/'wiki-raw-documents.jsonl'
    rawpath.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for _,d in sorted(rawrows)))
    code="""import readline from 'node:readline';import fs from 'node:fs';import {cleanWikipediaMarkup} from './training/prepare_smoke_data.mjs';
for await (const line of readline.createInterface({input:fs.createReadStream(process.argv[1]),crlfDelay:Infinity})){const row=JSON.parse(line);row.text=cleanWikipediaMarkup(row.text);console.log(JSON.stringify(row));}"""
    with target.open('w') as f:subprocess.run(['node','--input-type=module','-e',code,str(rawpath)],cwd=ROOT,stdout=f,check=True)
    stamp.write_text(signature)
    return target

def trace_generator():
    source=inspect.getsource(policy.training_fragment_lines)
    token="'boundary_reasons': boundary_reasons or []}"
    assert source.count(token)==1
    source=source.replace(token,"'boundary_reasons': boundary_reasons or [], 'end': end, 'source_line': line}")
    namespace=dict(vars(policy));exec(source,namespace)
    return namespace['training_fragment_lines']

def reviewed_documents():
    excluded={k:set() for k in POOLS}
    for k in POOLS:
        path=ROOT/f'training/artifacts/chinese-line-quality-20260917/{k}-samples.jsonl'
        if path.exists():excluded[k].update(json.loads(s)['document_number'] for s in path.read_text().splitlines())
    path=ROOT/'training/artifacts/v4-policy-validation-20260917/reviews.json'
    if path.exists():
        for r in json.loads(path.read_text()):excluded[r['corpus']].add(r['docno'])
    return excluded

def heldout(corpus,docno):return int.from_bytes(hashlib.sha256(f'{SEED}:{corpus}:{docno}'.encode()).digest()[:4],'little')%5==0

def study(out):
    trace=trace_generator();excluded=reviewed_documents();profiles={};tuning=[]
    for corpus,source in POOLS.items():
        path=fresh_wiki(out) if corpus=='wiki' else ROOT/'training/artifacts'/source
        draws={b:Reservoir(12 if corpus=='web' else 4,SEED+b+len(corpus)) for b in BOUNDS}
        counts=Counter();hist=Counter()
        for docno,s in enumerate(path.open()):
            d=json.loads(s);raw=d.get('content',d.get('text','')) or '';counts['documents']+=1
            if corpus in ('web','l3') and not quality_document(clean_document(raw)):continue
            for fragments in trace(raw,symbol_window=0):
                if not fragments:continue
                line=fragments[0]['source_line'];enclosed=policy.symbol_enclosures(line,64)
                for left,right in zip(fragments,fragments[1:]):
                    a,b=left['text'],right['text']
                    if left['reasons'] or right['reasons'] or not policy.valid_boundary_context(a,b):continue
                    if not any(c.isalnum() for c in a) or not any(c.isalnum() for c in b):continue
                    counts['eligible_proxies_n0']+=1
                    width=enclosed.get(left['end'])
                    if width is None:continue
                    hist[width]+=1
                    if docno in excluded[corpus] or heldout(corpus,docno):continue
                    band=next(x for x in BOUNDS if width<=x)
                    k=left['end'];start=max(0,k-100);end=min(len(line),k+101)
                    draws[band].add({'corpus':corpus,'docno':docno,'width':width,'band':band,'left':a,'right':b,'display':a+'｜'+b,'proxy':left['punctuation'],'source_line':line,'source_excerpt':line[start:end],'meta':{k:v for k,v in d.items() if k not in ('content','text')}})
        for band,draw in draws.items():
            for i,row in enumerate(draw.rows,1):row['id']=f'{corpus}-{band:02}-{i:02}';tuning.append(row)
        profiles[corpus]={'source':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'counts':dict(counts),'width_histogram':dict(sorted(hist.items())),'direct_proxy_rejections':{n:sum(v for k,v in hist.items() if k<=n) for n in WINDOWS}}
        print(corpus,counts,flush=True)
    (out/'tuning-samples.json').write_text(json.dumps(tuning,ensure_ascii=False,indent=2)+'\n')
    (out/'search.json').write_text(json.dumps({'seed':SEED,'windows':WINDOWS,'profiles':profiles,'policy':policy.DATA_POLICY,'method':'First enumerate eligible original proxies with the new source/barrier/surface rules and n=0, then measure their nearest-symbol width. Each rejected proxy remains a delimiter and only its own target is skipped. Consecutive punctuation runs are blocked if any member is enclosed; verify actual production yield at the chosen n. Independent review reserves 20% of documents by stable hash, excludes all documents in the prior 420 and 600 reviews. Tuning samples are stratified by corpus and distance band, not population-proportional.'},ensure_ascii=False,indent=2)+'\n')

def validate(out):
    excluded=reviewed_documents();profiles={};rows=[]
    for corpus,source in POOLS.items():
        path=fresh_wiki(out) if corpus=='wiki' else ROOT/'training/artifacts'/source
        counts=Counter();draw=Reservoir({'web':120,'l3':40,'wiki':20,'clue':20}[corpus],SEED+500+len(corpus))
        for docno,s in enumerate(path.open()):
            d=json.loads(s);raw=d.get('content',d.get('text','')) or ''
            if corpus in ('web','l3') and not quality_document(clean_document(raw)):continue
            pairs=list(policy.training_pairs(raw));counts['pairs']+=len(pairs)
            if docno in excluded[corpus] or not heldout(corpus,docno):continue
            for text,target,punctuation in pairs:
                assert policy.BARRIER not in text and not policy.LINE_BOUNDARIES.search(text)
                draw.add({'corpus':corpus,'docno':docno,'text':text,'target':target,'proxy':punctuation,'display':text[:target+1]+'｜'+text[target+1:],'meta':{k:v for k,v in d.items() if k not in ('content','text')}})
        for i,r in enumerate(draw.rows,1):r['id']=f'{corpus}-validation-{i:03}';rows.append(r)
        profiles[corpus]=dict(counts);print(corpus,counts,flush=True)
    (out/'validation-samples.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    (out/'final-yield.json').write_text(json.dumps({'policy':policy.DATA_POLICY,'profiles':profiles},ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'training/artifacts/v5-window-study-20260917');p.add_argument('--validate',action='store_true');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    (validate if args.validate else study)(args.output)
