"""Read-only sampling of original proxy glyphs; never infer width after NFKC.

Audits raw candidate pairs, before global training deduplication. Reuses the
historical v1 parser to validate the independently source-mapped boundary counts.
"""
from __future__ import annotations

import argparse
import bz2
import collections
import hashlib
import importlib.util
import json
import random
import re
import subprocess
import sys
import time
import unicodedata
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'training/artifacts/proxy-glyph-20260916'
RUN = ROOT/'training/artifacts/web-mix-40m-192ch-16conv-20260916'
SEED = 2026091603
sys.path.insert(0,str(ROOT/'training/.deps'))
import pyarrow.parquet as pq
from audit_corpus import consumed_documents
from prepare_synthetic_data import adjacent_samples, clean_document
from prepare_web_data import labeled_samples as current_samples
from text_policy import DATA_POLICY


def save(name, value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def historical_modules():
    directory=RUN/'source/training'
    plan=json.loads((RUN/'run.json').read_text())
    assert hashlib.sha256((directory/'text-policy.json').read_bytes()).hexdigest()==plan['source_hashes']['training/text-policy.json']
    originals={name:sys.modules.get(name) for name in ['text_policy','prepare_synthetic_data','prepare_web_data']}
    old_path=list(sys.path)
    modules={}
    try:
        for name in originals:
            path=directory/(name+'.py')
            assert hashlib.sha256(path.read_bytes()).hexdigest()==plan['source_hashes']['training/'+name+'.py']
            spec=importlib.util.spec_from_file_location(name,path)
            module=importlib.util.module_from_spec(spec)
            sys.modules[name]=module
            spec.loader.exec_module(module)
            modules[name]=module
    finally:
        sys.path[:]=old_path
        for name,module in originals.items():
            if module is None:sys.modules.pop(name,None)
            else:sys.modules[name]=module
    return modules


OLD=historical_modules()
OLD_PROXY=OLD['text_policy'].PROXY_PUNCTUATION
CHINESE=set('，。；！？…')
ASCII=set(',. ;!?'.replace(' ',''))


@lru_cache(maxsize=65536)
def nfkc(c):
    return unicodedata.normalize('NFKC',c)


def replace_mapped(text, origins, pattern):
    parts=[]; mapped=[]; start=0
    for match in pattern.finditer(text):
        parts.extend([text[start:match.start()],' '])
        mapped.extend(origins[start:match.start()]);mapped.append(None)
        start=match.end()
    parts.append(text[start:]);mapped.extend(origins[start:])
    return ''.join(parts),mapped


def normalize_mapped(raw, wiki=False):
    # Non-punctuation runs normalize together, preserving combining sequences.
    parts=[];origins=[];start=0
    for index,c in enumerate(raw):
        n=nfkc(c)
        if not any(p in OLD_PROXY for p in n):continue
        prefix=unicodedata.normalize('NFKC',raw[start:index])
        assert not any(p in OLD_PROXY for p in prefix)
        parts.extend([prefix,n]);origins.extend([None]*len(prefix)+[index]*len(n))
        start=index+1
    tail=unicodedata.normalize('NFKC',raw[start:])
    assert not any(p in OLD_PROXY for p in tail)
    parts.append(tail);origins.extend([None]*len(tail))
    text=''.join(parts)
    assert text==unicodedata.normalize('NFKC',raw)
    if not wiki:
        for pattern in [OLD['prepare_synthetic_data'].MARKDOWN_FENCE_PATTERN,OLD['prepare_synthetic_data'].URL_PATTERN]:
            text,origins=replace_mapped(text,origins,pattern)
    text,origins=replace_mapped(text,origins,re.compile(r'\s+'))
    left=len(text)-len(text.lstrip());right=len(text.rstrip())
    text,origins=text[left:right],origins[left:right]
    if not wiki:assert text==OLD['prepare_synthetic_data'].normalize_document(raw)
    return text,origins


def legacy_pairs(raw, wiki=False):
    text,origins=normalize_mapped(raw,wiki)
    fragments=[];labels=[];buffer=[]
    for i,c in enumerate(text):
        numeric=c in '.,' and 0<i<len(text)-1 and text[i-1].isdecimal() and text[i+1].isdecimal()
        if c not in OLD_PROXY or numeric:
            buffer.append(c);continue
        fragment=''.join(buffer).strip();buffer.clear()
        assert origins[i] is not None
        if fragment:
            fragments.append(fragment);labels.append([origins[i]])
        elif labels:labels[-1].append(origins[i])
    tail=''.join(buffer).strip()
    if tail:fragments.append(tail)
    result=[]
    for i,(left,right) in enumerate(zip(fragments,fragments[1:])):
        if any(c.isalnum() for c in left) and any(c.isalnum() for c in right):
            source=''.join(raw[p] for p in dict.fromkeys(labels[i]))
            result.append((left+right,len(left)-1,source))
    return result


def raw_events(raw):
    counts=collections.Counter();i=0
    chinese_family={'，':'comma','。':'period','；':'semicolon','！':'exclamation','？':'question','…':'ellipsis'}
    ascii_family={',':'comma','.':'period',';':'semicolon','!':'exclamation','?':'question'}
    while i<len(raw):
        c=raw[i]
        if c in '.…':
            end=i+1
            while end<len(raw) and raw[end]==c:end+=1
            if c=='…' or end-i>=2:
                counts[('ellipsis','chinese' if c=='…' else 'ascii')]+=1
                i=end;continue
        if c in chinese_family:counts[(chinese_family[c],'chinese')]+=1
        elif c in ascii_family:counts[(ascii_family[c],'ascii')]+=1
        else:
            normal=nfkc(c)
            if normal and all(p in OLD_PROXY for p in normal):
                families={chinese_family.get(p,ascii_family.get(p)) for p in normal}
                family=next(iter(families)) if len(families)==1 else 'mixed'
                if family:counts[(family,'variant')]+=1
        i+=1
    return counts


def sample_documents():
    OUT.mkdir(parents=True,exist_ok=True)
    inventory={'seed':SEED,'policy':DATA_POLICY,'scope':'Original documents from the current corpus sources; candidate pairs before training deduplication'}
    registries=OLD['prepare_web_data'].holdout_registries(ROOT/'training/data/processed/web-mix-40m-192ch-16conv-20260916')
    blocked={key for p in registries for key in json.loads(p.read_text())}
    rng=random.Random(SEED)
    with (OUT/'documents.jsonl').open('w') as destination:
        def emit(corpus,raw,weight,**location):
            key=OLD['prepare_synthetic_data'].document_signature(raw)
            if key in blocked:return False
            if corpus!='wikipedia':
                if not OLD['prepare_synthetic_data'].quality_document(OLD['prepare_synthetic_data'].normalize_document(raw)):return False
            if corpus=='web' and OLD['prepare_web_data'].document_split(key,20260915)!='train':return False
            row={'corpus':corpus,'text':raw,'weight':weight,'document_sha256':hashlib.sha256(raw.encode()).hexdigest(),**location}
            destination.write(json.dumps(row,ensure_ascii=False)+'\n')
            return True
        # Uniform random multistream blocks; all extracted main-namespace articles.
        index=ROOT/'training/data/raw/zhwiki-latest-pages-articles-multistream-index.txt.bz2'
        offsets=[];last=None
        with bz2.open(index,'rt') as source:
            for line in source:
                offset=int(line.split(':',1)[0])
                if offset!=last:offsets.append(offset);last=offset
        chosen=sorted(rng.sample(range(len(offsets)),128))
        dump=ROOT/'training/data/raw/zhwiki-latest-pages-articles-multistream.xml.bz2'
        blocks=[{'index':i,'offset':offsets[i],'bytes':(offsets[i+1] if i+1<len(offsets) else dump.stat().st_size)-offsets[i]} for i in chosen]
        save('wiki-blocks.json',blocks)
        js="""import fs from 'node:fs';
import {execFileSync} from 'node:child_process';
import {wikipediaDocumentsFromXml} from './training/prepare_smoke_data.mjs';
const blocks=JSON.parse(fs.readFileSync(process.argv[1],'utf8'));
const input=fs.openSync(process.argv[2],'r');
const output=fs.openSync(process.argv[3],'w');
for(const b of blocks){const compressed=Buffer.alloc(b.bytes);fs.readSync(input,compressed,0,b.bytes,b.offset);
const xml=execFileSync('bzip2',['-dc'],{input:compressed,encoding:'utf8',maxBuffer:256*1024*1024});
for(const d of wikipediaDocumentsFromXml(xml,b.offset))fs.writeSync(output,JSON.stringify({...d,block:b.index})+'\\n');}
fs.closeSync(input);fs.closeSync(output);"""
        subprocess.run(['node','--input-type=module','-e',js,str(OUT/'wiki-blocks.json'),str(dump),str(OUT/'wiki-documents.jsonl')],cwd=ROOT,check=True)
        kept=0;read=0
        for line in (OUT/'wiki-documents.jsonl').open():
            row=json.loads(line);read+=1
            kept+=emit('wikipedia',row['text'],len(offsets)/len(chosen),file=dump.name,block=row['block'],document_id=row['id'])
        inventory['wikipedia']={'blocks_total':len(offsets),'blocks_sampled':len(chosen),'documents_read':read,'eligible_documents':kept}
        print(json.dumps({'stage':'sampled','corpus':'wikipedia',**inventory['wikipedia']}),flush=True)
        # Uniform random rows over the complete locally consumed synthetic corpus.
        path=ROOT/'training/data/raw/ultra-fineweb-l3-zh-multistyle/00000.parquet'
        parquet=pq.ParquetFile(path);total=parquet.metadata.num_rows
        selected=set(rng.sample(range(total),5000));row_number=0;kept=0
        for group in range(parquet.num_row_groups):
            for batch in parquet.iter_batches(batch_size=512,row_groups=[group],columns=['content'],use_threads=False):
                wanted=[i for i in range(batch.num_rows) if row_number+i in selected]
                for i in wanted:
                    kept+=emit('synthetic',batch.column(0)[i].as_py() or '',total/5000,file=path.name,row=row_number+i,row_group=group)
                row_number+=batch.num_rows
        assert row_number==total
        inventory['synthetic']={'documents_total':total,'documents_sampled':5000,'eligible_documents':kept}
        print(json.dumps({'stage':'sampled','corpus':'synthetic',**inventory['synthetic']}),flush=True)
        # Stratified random rows inside every file's consumed raw-document prefix.
        runs=['web-mix-20m-192ch-16conv-20260915','web-mix-40m-192ch-16conv-20260916']
        limits=[max(a,b) for a,b in zip(*(consumed_documents(run) for run in runs))]
        kept=0;sampled=0
        for file_index,limit in enumerate(limits):
            path=ROOT/'training/data/raw/ultra-fineweb-zh'/f'ultrafineweb-zh-part-{file_index+1:03d}-of-256.parquet'
            parquet=pq.ParquetFile(path)
            groups=list(range(parquet.num_row_groups));random.Random(20260915+file_index).shuffle(groups)
            selected=set(rng.sample(range(limit),20));maximum=max(selected);position=0
            for group in groups:
                group_row=0
                for batch in parquet.iter_batches(batch_size=128,row_groups=[group],columns=['content','source'],use_threads=False):
                    wanted=[i for i in range(batch.num_rows) if position+i in selected]
                    for i in wanted:
                        sampled+=1
                        kept+=emit('web',batch.column(0)[i].as_py() or '',limit/20,file=path.name,file_index=file_index,row_group=group,row=group_row+i,source=batch.column(1)[i].as_py())
                    group_row+=batch.num_rows;position+=batch.num_rows
                    if position>maximum:break
                if position>maximum:break
            if file_index%32==31:print(json.dumps({'stage':'sampled_web','files':file_index+1,'eligible_documents':kept}),flush=True)
        assert sampled==5120
        inventory['web']={'consumed_documents':sum(limits),'files':256,'rows_per_file':20,'documents_sampled':sampled,'eligible_documents':kept,'limits':limits}
    save('sampling.json',inventory)


def analyze():
    summaries={};metrics=[];examples=[];by_source={};glyphs=collections.defaultdict(collections.Counter)
    rng=random.Random(SEED+1)
    with (OUT/'documents.jsonl').open() as handle:
        for row_index,line in enumerate(handle):
            row=json.loads(line);raw=row['text'];corpus=row['corpus'];weight=row['weight']
            if corpus not in summaries:
                summaries[corpus]={'documents':0,'old_targets':0,'new_targets':0,'non_chinese_targets':0,
                                  'ascii_only_targets':0,'chinese_only_targets':0,'mixed_targets':0,'variant_only_targets':0,
                                  'weighted_old_targets':0.,'weighted_new_targets':0.,'weighted_non_chinese_targets':0.,
                                  'raw_counts':collections.Counter(),'weighted_raw_counts':collections.Counter()}
            s=summaries[corpus];s['documents']+=1
            pairs=legacy_pairs(raw,corpus=='wikipedia')
            if corpus!='wikipedia':
                actual=list(OLD['prepare_web_data'].labeled_samples(raw))
                assert [(text,target) for text,target,_ in pairs]==[(text,target) for text,target,_ in actual]
            new=list(current_samples(raw)) if corpus!='wikipedia' else None
            if new is None:
                # Wikipedia has already received its own markup cleanup. The
                # pure splitter avoids applying synthetic URL/fence rules again.
                original_current=raw
                pieces=[];buffer=[]
                for i,c in enumerate(original_current):
                    number=c=='，' and 0<i<len(raw)-1 and raw[i-1].isdecimal() and raw[i+1].isdecimal()
                    if c not in CHINESE or number:buffer.append(c);continue
                    fragment=unicodedata.normalize('NFKC',''.join(buffer)).strip();buffer.clear()
                    if fragment:pieces.append(fragment)
                fragment=unicodedata.normalize('NFKC',''.join(buffer)).strip()
                if fragment:pieces.append(fragment)
                new=[1 for a,b in zip(pieces,pieces[1:]) if any(c.isalnum() for c in a) and any(c.isalnum() for c in b)]
            non=0
            for text,target,source in pairs:
                cn=any(c in CHINESE for c in source)
                asc=all(c in ASCII for c in source)
                if not cn:non+=1
                category='chinese_only_targets' if all(c in CHINESE for c in source) else 'ascii_only_targets' if asc else 'mixed_targets' if cn else 'variant_only_targets'
                s[category]+=1
                glyphs[corpus][source]+=1
                # Bounded ordinary-language examples, not a quality prevalence sample.
                if source==',' and target>8 and len(text)-target>8 and len(examples)<18 and rng.random()<.0005:
                    examples.append({'corpus':corpus,'glyph':source,'text':text[max(0,target-22):target+1]+'｜'+text[target+1:target+24]})
            cleaned=raw if corpus=='wikipedia' else clean_document(raw)
            events=raw_events(cleaned)
            for (family,style),count in events.items():
                key=family+':'+style
                s['raw_counts'][key]+=count;s['weighted_raw_counts'][key]+=weight*count
            s['old_targets']+=len(pairs);s['new_targets']+=len(new);s['non_chinese_targets']+=non
            s['weighted_old_targets']+=weight*len(pairs);s['weighted_new_targets']+=weight*len(new);s['weighted_non_chinese_targets']+=weight*non
            metrics.append({k:row[k] for k in ['corpus','weight','file','document_sha256']}|{
                'cluster':row.get('block',row.get('file_index',row.get('row'))),'old_targets':len(pairs),'new_targets':len(new),'non_chinese_targets':non,
                'characters':len(raw)})
            if corpus=='web':
                source=row['source'];ss=by_source.setdefault(source,{'documents':0,'old_targets':0,'non_chinese_targets':0,'new_targets':0})
                ss['documents']+=1;ss['old_targets']+=len(pairs);ss['non_chinese_targets']+=non;ss['new_targets']+=len(new)
            if row_index%2000==1999:print(json.dumps({'stage':'analyzing','documents':row_index+1}),flush=True)
    for s in summaries.values():
        s['non_chinese_share']=s['weighted_non_chinese_targets']/s['weighted_old_targets']
        s['candidate_count_change']=s['weighted_new_targets']/s['weighted_old_targets']-1
    for s in by_source.values():s['non_chinese_share']=s['non_chinese_targets']/s['old_targets'] if s['old_targets'] else None
    populations={'wikipedia':41886890,'synthetic':31904356,'web':40000000}
    covered=sum(populations.values())
    approximate=sum(populations[k]/covered*summaries[k]['non_chinese_share'] for k in populations)
    findings={'seed':SEED,'corpora':summaries,'web_sources':by_source,'ordinary_comma_examples':examples,
              'glyph_counts':glyphs,'covered_training_samples':covered,'training_population':114121940,
              'rough_training_mix_non_chinese_share':approximate,
              'rough_training_mix_note':'Source-weighted approximation from raw candidate pairs, not a direct estimate from deduplicated training rows. Omits small CLUE sources (0.29%).',
              'denominator':'Adjacent-fragment targets emitted by old v1 rules, on sampled original documents after corpus cleanup and eligibility checks, before global sample deduplication.',
              'raw_counts_note':'Original glyphs before NFKC, after corpus cleanup; repeated … or runs of 2+ ASCII periods count as one ellipsis occurrence. Includes numeric punctuation.',
              'new_count_note':'Same sampled documents through current six-character raw-source whitelist, before corpus-wide deduplication.',
              'validation':'All synthetic/web old input+gap pairs exactly match frozen historical parser; new parser used directly. Wikipedia uses unchanged XML extraction plus equivalent boundary logic.'}
    save('findings.json',findings)
    with (OUT/'document-metrics.jsonl').open('w') as handle:
        for row in metrics:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps({k:{kk:vv for kk,vv in s.items() if 'raw_counts' not in kk} for k,s in summaries.items()},ensure_ascii=False,indent=2),flush=True)
    print('rough_mix',approximate,flush=True)


def verify_training_web():
    """Independent check using previously drawn uniform training-row samples."""
    by_file=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(list)))
    path=ROOT/'training/artifacts/corpus-quality-20260916/web-provenance.jsonl'
    for line in path.open():
        row=json.loads(line)
        if int(row['id'].rsplit('-',1)[1])<=250:
            by_file[row['raw_file']][row['row_group']][row['row_in_group']].append(row)
    results=[]
    for filename,groups in sorted(by_file.items()):
        parquet=pq.ParquetFile(ROOT/'training/data/raw/ultra-fineweb-zh'/filename)
        for group,wanted in groups.items():
            offset=0;maximum=max(wanted)
            for batch in parquet.iter_batches(batch_size=256,row_groups=[group],columns=['content'],use_threads=False):
                for i in range(batch.num_rows):
                    if offset+i not in wanted:continue
                    raw=batch.column(0)[i].as_py() or ''
                    pairs=legacy_pairs(raw)
                    for target in wanted[offset+i]:
                        forms=sorted({glyph for text,gap,glyph in pairs if (text,gap)==(target['text'],target['target_index'])})
                        assert forms,target['id']
                        non_chinese=[not any(c in CHINESE for c in glyph) for glyph in forms]
                        results.append({'id':target['id'],'stratum':target['id'].rsplit('-',1)[0],
                                        'original_forms':forms,'non_chinese':non_chinese[0] if len(set(non_chinese))==1 else None})
                offset+=batch.num_rows
                if offset>maximum:break
    assert len(results)==500
    summary={}
    for name in ['web_first','web_added']:
        rr=[r for r in results if r['stratum']==name]
        assert len(rr)==250
        summary[name]={'n':len(rr),'non_chinese':sum(r['non_chinese'] is True for r in rr),
                       'ambiguous':sum(r['non_chinese'] is None for r in rr)}
    save('training-web-check.json',{'strata':summary,'rows':results,
          'provenance_caveat':'Exact original-text matches from earlier audit, not guaranteed unique originating documents.'})
    print(json.dumps({'stage':'training_row_check',**summary}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['sample','analyze','verify-training-web','all']);args=parser.parse_args()
    started=time.time()
    if args.action in ['sample','all']:sample_documents()
    if args.action in ['analyze','all']:analyze()
    if args.action in ['verify-training-web','all']:verify_training_web()
    print(json.dumps({'stage':'complete','seconds':time.time()-started}),flush=True)
