#!/usr/bin/env python3
"""Index intact local web documents suitable for a subsequent prose experiment.

No corpus, labels, holdouts, checkpoint or training process is modified.
Per-file atomic outputs permit resume, with source and rule identity checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'training/.deps'))
import pyarrow as pa
import pyarrow.parquet as pq
from audit_corpus import consumed_documents
from prepare_web_data import document_split, holdout_registries
from web_document_quality import POLICY, REASON_BITS, NON_HAN, inspect_document

DEFAULT_OUTPUT = ROOT/'training/artifacts/web-prose-screen-v2-20260916'
DEFAULT_EVALUATION = ROOT/'training/data/processed/web-mix-40m-192ch-16conv-20260916'
SCHEMA = pa.schema([
    ('file_index',pa.int32()),('row_group',pa.int32()),('row_in_group',pa.int32()),
    ('source',pa.string()),('score',pa.float64()),('document_id',pa.string()),
    ('raw_sha256',pa.string()),('characters',pa.int32()),('han',pa.int32()),
    ('sentence_units',pa.int32()),('reason_mask',pa.int32()),
    ('partition',pa.string()),('inherited_holdout',pa.bool_()),('candidate',pa.bool_()),
])
BLOCKED = set()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path);part=path.with_suffix(path.suffix+'.tmp')
    part.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    part.replace(path)


def worker_setup(registries):
    global BLOCKED
    BLOCKED={key for path in registries for key in json.loads(Path(path).read_text())}


def screen_file(job):
    file_index,path,limit,output,scope,rule_hash = job
    path=Path(path);output=Path(output)
    assert digest(ROOT/'training/web_document_quality.py')==rule_hash
    destination=output/f'index-{file_index:03d}.parquet'
    stats_path=output/f'stats-{file_index:03d}.json'
    if stats_path.exists():
        stats=json.loads(stats_path.read_text())
        if (stats['rule_sha256']!=rule_hash or stats['scope']!=scope
                or stats['source_bytes']!=path.stat().st_size
                or stats['source_mtime_ns']!=path.stat().st_mtime_ns
                or stats['index_sha256']!=digest(destination)):
            raise ValueError(f'Existing screening output does not match source or policy: {path}')
        return stats
    part=destination.with_suffix('.parquet.tmp')
    parquet=pq.ParquetFile(path)
    groups=list(range(parquet.num_row_groups))
    if scope=='consumed': random.Random(20260915+file_index).shuffle(groups)
    maximum=parquet.metadata.num_rows if limit is None else limit
    stats={'file_index':file_index,'file':path.name,'source_bytes':path.stat().st_size,
           'source_mtime_ns':path.stat().st_mtime_ns,'scope':scope,'rule_sha256':rule_hash,
           'documents':0,'accepted':0,'candidates':0,'holdout_accepted':0,
           'by_source':{},'reasons':Counter(),'score_profiles':{},'review_examples':[]}
    pools={'candidate':[],'quarantine':[]}; seen=Counter()
    rng=random.Random(2026091609+file_index)
    started=time.time()
    with pq.ParquetWriter(part,SCHEMA,compression='zstd') as writer:
        for group in groups:
            offset=0
            for batch in parquet.iter_batches(batch_size=256,row_groups=[group],columns=['content','source','score'],use_threads=False):
                rows=[]
                for i,item in enumerate(batch.to_pylist()):
                    if stats['documents']>=maximum: break
                    raw=item['content'] or '';source=item['source'] or 'unknown'
                    quality=inspect_document(raw,item['score'])
                    normalized=unicodedata.normalize('NFKC',raw)
                    key=hashlib.sha256(NON_HAN.sub('',normalized).encode()).hexdigest()
                    partition=document_split(key,20260915)
                    inherited=key in BLOCKED
                    candidate=quality['accepted'] and partition=='train' and not inherited
                    row={'file_index':file_index,'row_group':group,'row_in_group':offset+i,
                         'source':source,'score':quality['score'],'document_id':key,
                         'raw_sha256':hashlib.sha256(raw.encode()).hexdigest(),
                         'characters':len(raw),'han':quality['han'],'sentence_units':quality['sentence_units'],
                         'reason_mask':quality['reason_mask'],'partition':partition,
                         'inherited_holdout':inherited,'candidate':candidate}
                    rows.append(row)
                    stats['documents']+=1;stats['accepted']+=quality['accepted'];stats['candidates']+=candidate
                    stats['holdout_accepted']+=quality['accepted'] and (partition!='train' or inherited)
                    stats['reasons'].update(quality['reasons'])
                    ss=stats['by_source'].setdefault(source,Counter())
                    ss['documents']+=1;ss['accepted']+=quality['accepted'];ss['candidates']+=candidate
                    for threshold in [.5,.7,.8,.9,.95]:
                        if quality['score'] is not None and quality['score']>=threshold:
                            rr=stats['score_profiles'].setdefault(str(threshold),Counter())
                            rr['documents']+=1;rr['accepted']+=quality['accepted'];rr['candidates']+=candidate
                    category='candidate' if candidate else 'quarantine' if not quality['accepted'] else None
                    if category:
                        seen[category]+=1;pool=pools[category];k=2 if category=='candidate' else 1
                        slot=rng.randrange(seen[category])
                        if len(pool)<k: slot=len(pool)
                        if slot<k:
                            example=dict(row,category=category,reasons=quality['reasons'],text=raw)
                            if len(pool)<k:pool.append(example)
                            else:pool[slot]=example
                if rows:writer.write_table(pa.Table.from_pylist(rows,schema=SCHEMA))
                offset+=batch.num_rows
                if stats['documents']>=maximum:break
            if stats['documents']>=maximum:break
    assert stats['documents']==maximum
    part.replace(destination)
    stats['index_sha256']=digest(destination)
    stats['seconds']=time.time()-started
    stats['review_examples']=[dict(r,sampling_pool_documents=seen[category]) for category,pool in pools.items() for r in pool]
    write_json(stats_path,stats)
    return stats


def merge(output,statistics,identity):
    totals=Counter();reasons=Counter();sources={};profiles={};seen=set()
    selected=output/'train-candidates.parquet';part=selected.with_suffix('.parquet.tmp')
    unique=0;duplicates=0;examples=[];index_manifest=[]
    with pq.ParquetWriter(part,SCHEMA,compression='zstd') as writer:
        for stats in sorted(statistics,key=lambda r:r['file_index']):
            for key in ['documents','accepted','candidates','holdout_accepted']:totals[key]+=stats[key]
            reasons.update(stats['reasons'])
            for name,counts in stats['by_source'].items():sources.setdefault(name,Counter()).update(counts)
            for name,counts in stats['score_profiles'].items():profiles.setdefault(name,Counter()).update(counts)
            index=output/f"index-{stats['file_index']:03d}.parquet"
            index_manifest.append({'file':index.name,'sha256':stats['index_sha256'],'documents':stats['documents']})
            for batch in pq.ParquetFile(index).iter_batches(batch_size=8192):
                kept=[]
                for row in batch.to_pylist():
                    if not row['candidate']:continue
                    if row['document_id'] in seen:duplicates+=1;continue
                    assert row['partition']=='train' and not row['inherited_holdout'] and row['reason_mask']==0
                    seen.add(row['document_id']);kept.append(row);unique+=1
                if kept:writer.write_table(pa.Table.from_pylist(kept,schema=SCHEMA))
            examples.extend(stats['review_examples'])
    part.replace(selected)
    with (output/'review-documents.jsonl').open('w') as handle:
        for row in examples:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    result={'stage':'complete','identity':identity,'policy':POLICY,'reason_bits':REASON_BITS,
            'totals':totals,'quarantined':totals['documents']-totals['accepted'],
            'unique_train_candidates':unique,'duplicate_candidate_documents':duplicates,
            'by_source':sources,'score_profiles':profiles,'indices':index_manifest,
            'candidate_index':{'path':str(selected.relative_to(ROOT)) if selected.is_relative_to(ROOT) else str(selected),
                               'sha256':digest(selected),'documents':unique},
            'limitations':['Admission is a conservative structural heuristic, not a semantic quality guarantee.',
                           'Candidates preserve whole raw documents; no text editing or boundary labels were generated.',
                           'Inherited holdout documents and hash-reserved validation/test documents are excluded from the train candidate index.',
                           'Han-projected document identities are deduplicated in this candidate index. Pair-level deduplication against old training/evaluation is still required at preparation.',
                           'Consumed scope means the raw documents visited by the previous web preparation, not the entire downloaded corpus.'],
            'created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    write_json(output/'manifest.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=['consumed','all'],default='consumed')
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--raw-dir',type=Path,default=ROOT/'training/data/raw/ultra-fineweb-zh')
    parser.add_argument('--evaluation',type=Path,default=DEFAULT_EVALUATION)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args();output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    if args.workers<1:parser.error('--workers must be positive')
    with (output/'.screen.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        downloaded=json.loads((args.raw_dir/'download-manifest.json').read_text())
        verified=json.loads((args.raw_dir/'verified-files.json').read_text())
        registries=holdout_registries(args.evaluation.resolve())
        assert registries, 'Existing holdout registries required'
        identity={'scope':args.scope,'raw_directory':str(args.raw_dir.resolve()),
                  'source_revision':downloaded['revision'],'source_manifest_sha256':digest(args.raw_dir/'download-manifest.json'),
                  'policy_sha256':digest(ROOT/'training/web_document_quality.py'),
                  'runner_sha256':digest(__file__),'partition_seed':20260915,
                  'dependency_sha256':{name:digest(ROOT/'training'/name) for name in ['audit_corpus.py','prepare_web_data.py','prepare_synthetic_data.py']},
                  'holdout_registries':{str(p):digest(p) for p in registries}}
        if (output/'run.json').exists():
            if json.loads((output/'run.json').read_text())!=identity:raise ValueError('Scope, inputs or screening code changed; use a new output directory')
        else:
            write_json(output/'run.json',identity)
            for name in ['screen_local_web.py','web_document_quality.py','audit_corpus.py','prepare_web_data.py','prepare_synthetic_data.py']:
                shutil.copyfile(ROOT/'training'/name,output/('snapshot-'+name))
        if args.scope=='consumed':
            runs=['web-mix-20m-192ch-16conv-20260915','web-mix-40m-192ch-16conv-20260916']
            limits=[max(a,b) for a,b in zip(*(consumed_documents(r) for r in runs))]
        else:limits=[None]*len(downloaded['files'])
        jobs=[]
        for i,info in enumerate(downloaded['files']):
            path=args.raw_dir/info['filename']
            known=verified['files'][info['filename']]
            if path.stat().st_size!=info['bytes'] or known['sha256']!=info['sha256']:raise ValueError(f'Unverified source {path}')
            jobs.append((i,str(path),limits[i],str(output),args.scope,identity['policy_sha256']))
        assert len(jobs)==256
        statistics=[];started=time.time()
        write_json(output/'status.json',{'stage':'screening','scope':args.scope,'pid':os.getpid(),'completed_files':0})
        try:
            with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_setup,initargs=([str(p) for p in registries],)) as executor:
                for future in as_completed([executor.submit(screen_file,job) for job in jobs]):
                    statistics.append(future.result())
                    state={'stage':'screening','scope':args.scope,'pid':os.getpid(),'completed_files':len(statistics),
                           'documents':sum(s['documents'] for s in statistics),'candidates':sum(s['candidates'] for s in statistics),
                           'seconds':round(time.time()-started,1)}
                    write_json(output/'status.json',state)
                    if len(statistics)%16==0:print(json.dumps(state),flush=True)
            write_json(output/'status.json',{'stage':'merging','scope':args.scope,'completed_files':len(statistics)})
            result=merge(output,statistics,identity)
            state={'stage':'complete','scope':args.scope,**result['totals'],
                   'unique_train_candidates':result['unique_train_candidates'],'seconds':round(time.time()-started,1)}
            write_json(output/'status.json',state);print(json.dumps(state),flush=True)
        except BaseException as error:
            write_json(output/'status.json',{'stage':'interrupted','scope':args.scope,'error':str(error),'completed_files':len(statistics)})
            raise


if __name__=='__main__':main()
