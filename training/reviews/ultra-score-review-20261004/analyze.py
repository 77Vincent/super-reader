"""Read-only score census and frozen stratified review. Never starts training."""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'training/.deps'), str(ROOT/'training')]
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REVIEW = Path(__file__).resolve().parent
OUT = ROOT/'training/artifacts/ultra-score-review-20261004'
RAW = ROOT/'training/data/raw/ultra-fineweb-zh'
SEED = 2026100451
EDGES = [5000, 8000, 9000, 9500, 9800, 9900, 9990, 10000, 10001]
LABELS = ['[0.5,0.8)', '[0.8,0.9)', '[0.9,0.95)', '[0.95,0.98)',
          '[0.98,0.99)', '[0.99,0.999)', '[0.999,1)', '1.0']


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def scores(table):
    assert table['score'].null_count == table['source'].null_count == 0
    values = pc.cast(table['score'], pa.float64()).to_numpy()
    assert np.isfinite(values).all() and ((values >= .5) & (values <= 1)).all()
    ticks = np.rint(values*10000).astype(np.int32)
    assert np.max(np.abs(values-ticks/10000)) < 1e-8
    return ticks


def profile():
    manifest = json.loads((RAW/'download-manifest.json').read_text())
    files = sorted(manifest['files'], key=lambda x:x['filename'])
    assert len(files) == 256
    counts = collections.defaultdict(lambda:np.zeros(10001, dtype=np.int64))
    inventory, groups = [], []
    began = time.monotonic()
    for index, entry in enumerate(files):
        path = RAW/entry['filename']; stat = path.stat()
        assert stat.st_size == entry['bytes']
        parquet = pq.ParquetFile(path)
        table = parquet.read(columns=['score','source'], use_threads=False)
        ticks = scores(table)
        encoded = table['source'].combine_chunks().dictionary_encode()
        source_codes = encoded.indices.to_numpy()
        for code, name in enumerate(encoded.dictionary.to_pylist()):
            counts[name] += np.bincount(ticks[source_codes == code], minlength=10001)
        inventory.append(dict(**entry, rows=parquet.metadata.num_rows, row_groups=parquet.num_row_groups,
                              mtime_ns=stat.st_mtime_ns))
        for group in range(parquet.num_row_groups):
            meta = parquet.metadata.row_group(group)
            groups.append(dict(file=index, group=group, rows=meta.num_rows,
                               content_compressed_bytes=meta.column(0).total_compressed_size))
        assert len(ticks) == parquet.metadata.num_rows and path.stat().st_mtime_ns == stat.st_mtime_ns
        if (index+1)%32 == 0:
            print(json.dumps(dict(files=index+1, rows=sum(int(v.sum()) for v in counts.values()),
                                  seconds=round(time.monotonic()-began,1))),flush=True)
    total_hist = sum(counts.values())
    n = int(total_hist.sum()); cum = np.cumsum(total_hist)
    thresholds = [5000,8000,9000,9500,9800,9900,9950,9990,9999,10000]
    result = dict(repository=manifest['repository'], revision=manifest['revision'],
                  rows=n, files=inventory, groups=groups,
                  raw_bytes=sum(x['bytes'] for x in files),
                  score_type='string, parsed as float; all values on 0.0001 grid',
                  null_or_invalid_scores=0,
                  histograms={k:v.tolist() for k,v in counts.items()},
                  quantiles={str(q):int(np.searchsorted(cum,q*n))/10000 for q in [.1,.25,.5,.75,.9,.95,.99,.999]},
                  thresholds=[dict(threshold=t/10000, documents=int(total_hist[t:].sum()),
                                   fraction=float(total_hist[t:].sum()/n),
                                   sources={k:int(v[t:].sum()) for k,v in counts.items()}) for t in thresholds],
                  seconds=time.monotonic()-began,
                  verification='File sizes compared with existing pinned download manifest; prior upstream hashes recorded, not reread all content for hashing.')
    save(REVIEW/'profile.json',result)
    print(json.dumps({k:result[k] for k in ['rows','raw_bytes','quantiles','thresholds','seconds']},ensure_ascii=False),flush=True)


def sample():
    """SRS of row groups, then SRS docs in score strata; keep context for review."""
    from prepare_web_data import document_split
    from prepare_synthetic_data import clean_document, quality_document, document_signature
    from text_policy import training_pairs, DATA_POLICY
    profile = json.loads((REVIEW/'profile.json').read_text())
    # Reproducible sample of 32 entire row groups; actual frame is recorded.
    group_indices = sorted(random.Random(SEED).sample(range(len(profile['groups'])),32))
    rng = np.random.default_rng(SEED)
    pools = [[] for _ in LABELS]
    candidate_counts = [0]*len(LABELS)
    frame_sources = [collections.Counter() for _ in LABELS]
    per_group_counts=[]
    # Top-random-priority sampling yields uniform documents within each score
    # stratum of this sampled frame, without reading all 324 GB of text.
    for group_index in group_indices:
        group = profile['groups'][group_index]
        file = profile['files'][group['file']]
        table = pq.ParquetFile(RAW/file['filename']).read_row_group(group['group'],columns=['score','source'])
        ticks = scores(table); priorities = rng.random(len(ticks))
        src = table['source'].to_pylist()
        group_counts=[]
        for bucket,(lo,hi) in enumerate(zip(EDGES,EDGES[1:])):
            mask = np.flatnonzero((ticks>=lo)&(ticks<hi))
            candidate_counts[bucket]+=len(mask);group_counts.append(len(mask))
            frame_sources[bucket].update(src[i] for i in mask)
            chosen = mask[np.argpartition(priorities[mask],min(599,len(mask)-1))[:600]] if len(mask) else []
            pool = pools[bucket]
            pool.extend(dict(priority=float(priorities[i]), file=group['file'], group=group['group'],
                             row=int(i), tick=int(ticks[i]), source=src[i], bucket=bucket) for i in chosen)
            pools[bucket] = sorted(pool,key=lambda x:x['priority'])[:600]
        per_group_counts.append(dict(**group, strata=group_counts))
    assert all(len(p)==600 for p in pools)
    selected = [r for pool in pools for r in pool]
    by_group=collections.defaultdict(list)
    for r in selected: by_group[r['file'],r['group']].append(r)
    output=[]
    for index,((file_index,group_index),rows) in enumerate(sorted(by_group.items())):
        file=profile['files'][file_index]
        path=RAW/file['filename']
        assert path.stat().st_size==file['bytes'] and path.stat().st_mtime_ns==file['mtime_ns']
        table=pq.ParquetFile(path).read_row_group(group_index, columns=['content'])
        docs=table.take(pa.array([r['row'] for r in rows]))['content'].to_pylist()
        for row,content in zip(rows,docs):
            signature=document_signature(content or '')
            split=document_split(signature,20260915)
            # Keep the existing random hash holdouts entirely outside semantic
            # review and yield calculations. Filtering rates are logged.
            record=dict(**row, raw_file=file['filename'], document_id=signature,
                        content_sha256=hashlib.sha256((content or '').encode()).hexdigest(),
                        split=split, quality_document=quality_document(clean_document(content or '')),
                        content=content, characters=len(content or ''))
            rejection_counts={}
            pairs=[]
            if split=='train' and record['quality_document']:
                seen=set()
                for pair_index,(text,target,punctuation) in enumerate(training_pairs(content,rejection_counts)):
                    cut=target+1
                    if min(cut,len(text)-cut)<2:continue
                    if (text,cut) in seen: continue
                    seen.add((text,cut))
                    pairs.append(dict(text=text,cut=cut,punctuation=punctuation,pair_index=pair_index,
                                      length=len(text),distance=min(cut,len(text)-cut)))
            record.update(pairs=pairs,pair_count=len(pairs),rejections=rejection_counts)
            output.append(record)
        print(json.dumps(dict(content_groups=index+1,documents=len(output))),flush=True)
    output.sort(key=lambda r:(r['bucket'],r['priority']))
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'documents.jsonl').open('w') as f:
        for r in output:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    # One randomly chosen eligible pair per distinct reviewed document. Score
    # and source hidden in the separate blind-review view.
    blind=[]; pair_rng=random.Random(SEED+1); seen_docs=set()
    for bucket in range(len(LABELS)):
        candidates=[r for r in output if r['bucket']==bucket and r['split']=='train' and r['pairs']]
        pair_rng.shuffle(candidates)
        selected_bucket=[]
        for r in candidates:
            if r['document_id'] in seen_docs:continue
            seen_docs.add(r['document_id'])
            p=pair_rng.choice(r['pairs'])
            selected_bucket.append(dict(**{k:r[k] for k in ['bucket','tick','source','raw_file','group','row','document_id','content_sha256']},**p))
            if len(selected_bucket)==80:break
        assert len(selected_bucket)==80
        blind.extend(selected_bucket)
    pair_rng.shuffle(blind)
    for i,r in enumerate(blind,1):r['review_id']=f'S{i:04d}'
    save(REVIEW/'sample.json',blind)
    save(REVIEW/'protocol.json',dict(seed=SEED,score_edges=EDGES,strata=LABELS,documents_per_stratum=600,
        groups=per_group_counts,frame_rows=sum(g['rows'] for g in per_group_counts),
        frame_score_counts=candidate_counts,frame_sources=[dict(c) for c in frame_sources],
        policy=DATA_POLICY,minimum_side_characters=2,semantic_pairs_per_stratum=80,
        semantic_unit='One uniformly random retained pair per sampled distinct eligible training document; document-weighted, not training-row-weighted.',
        selection='32 row groups sampled uniformly without replacement from all local row groups; uniform random-priority sample of 600 documents per score stratum within sampled frame; then 80 distinct eligible train documents per stratum.',
        holdout='Existing prepare_web_data document hash partition seed 20260915; no validation/test texts included in semantic review or extracted-pair yield.',
        limits=['Two-stage clustered document sample; score census is exact but semantic rates are sample estimates.',
                'Deduplication inside each document only for yield; no historical Bloom or cross-corpus dedup, so yield is not final production sample count.',
                'Current text policy v8 and min side 2, no model training.',
                'Source is provenance only, not semantic domain; score remains hidden during pair review.']))
    print(json.dumps(dict(documents=len(output),blind_pairs=len(blind),frame_rows=sum(candidate_counts)),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['profile','sample'])
    args=parser.parse_args()
    globals()[args.stage]()
