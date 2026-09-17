#!/usr/bin/env python3
"""Extract original prose spans from a completed local web document screening.

Each output row is independent. Never concatenate rows when preparing labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'training/.deps'))
import pyarrow as pa
import pyarrow.parquet as pq
from web_document_quality import SPAN_POLICY, select_prose_spans

SCHEMA = pa.schema([
    ('file_index', pa.int32()), ('row_group', pa.int32()), ('row_in_group', pa.int32()),
    ('document_id', pa.string()), ('raw_sha256', pa.string()),
    ('start', pa.int32()), ('end', pa.int32()), ('span_sha256', pa.string()),
    ('source', pa.string()), ('score', pa.float64()),
    ('han', pa.int32()), ('long_clauses', pa.int32()), ('content', pa.string()),
])


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, data):
    path = Path(path)
    part = path.with_suffix(path.suffix + '.tmp')
    part.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    part.replace(path)


def extract_file(job):
    file_index, source, candidates, output = job
    output = Path(output)
    groups = defaultdict(dict)
    for row in candidates:
        if not row['candidate'] or row['partition'] != 'train' or row['inherited_holdout']:
            raise ValueError('Only screened training candidates may be extracted')
        position = row['row_in_group']
        if position in groups[row['row_group']]:
            raise ValueError('Duplicate source document position')
        groups[row['row_group']][position] = row
    stats = {'documents': 0, 'documents_with_spans': 0, 'spans': 0, 'rejected_lines_by_reason': Counter()}
    destination = output / f'spans-{file_index:03d}.parquet'
    part = destination.with_suffix('.parquet.tmp')
    parquet = pq.ParquetFile(source)
    with pq.ParquetWriter(part, SCHEMA, compression='zstd') as writer:
        for group, wanted in groups.items():
            offset = 0
            maximum = max(wanted)
            for batch in parquet.iter_batches(batch_size=256, row_groups=[group], columns=['content'], use_threads=False):
                rows = []
                for i, item in enumerate(batch.to_pylist()):
                    row = wanted.get(offset + i)
                    if row is None:
                        continue
                    raw = item['content'] or ''
                    if hashlib.sha256(raw.encode()).hexdigest() != row['raw_sha256']:
                        raise ValueError('Source text changed since document screening')
                    spans, rejected = select_prose_spans(raw, row['score'])
                    stats['documents'] += 1
                    stats['documents_with_spans'] += bool(spans)
                    stats['spans'] += len(spans)
                    stats['rejected_lines_by_reason'].update(rejected)
                    for span in spans:
                        content = raw[span['start']:span['end']]
                        if '\n' in content or '\r' in content:
                            raise ValueError('Span unexpectedly crosses source lines')
                        rows.append({**{key: row[key] for key in ['file_index', 'row_group', 'row_in_group',
                                                                 'document_id', 'raw_sha256', 'source', 'score']},
                                     **span, 'span_sha256': hashlib.sha256(content.encode()).hexdigest(),
                                     'content': content})
                if rows:
                    writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
                offset += batch.num_rows
                if offset > maximum:
                    break
    if stats['documents'] != len(candidates):
        raise ValueError('Source document positions missing')
    part.replace(destination)
    stats.update(file_index=file_index, sha256=digest(destination))
    write_json(output / f'stats-{file_index:03d}.json', stats)
    return stats


def merge(output, stats):
    output = Path(output)
    seen = set(); documents = set(); sources = {}; tiers = {}; totals = Counter(); rejected = Counter()
    destination = output / 'train-prose.parquet'
    part = destination.with_suffix('.parquet.tmp')
    with pq.ParquetWriter(part, SCHEMA, compression='zstd') as writer:
        for entry in sorted(stats, key=lambda row: row['file_index']):
            for name in ['documents', 'documents_with_spans', 'spans']:
                totals[name] += entry[name]
            rejected.update(entry['rejected_lines_by_reason'])
            path = output / f"spans-{entry['file_index']:03d}.parquet"
            if digest(path) != entry['sha256']:
                raise ValueError('Intermediate span file changed')
            for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
                kept = []
                for row in batch.to_pylist():
                    if row['span_sha256'] in seen:
                        totals['duplicate_spans'] += 1
                        continue
                    seen.add(row['span_sha256']); documents.add(row['document_id']); kept.append(row)
                    totals['selected_han'] += row['han']
                    totals['selected_characters'] += len(row['content'])
                    sources.setdefault(row['source'], Counter()).update(spans=1, han=row['han'])
                    for threshold in [.5, .7, .8, .9, .95]:
                        if row['score'] >= threshold:
                            tiers.setdefault(str(threshold), Counter()).update(spans=1, han=row['han'])
                if kept:
                    writer.write_table(pa.Table.from_pylist(kept, schema=SCHEMA))
    part.replace(destination)
    return {'totals': dict(totals, unique_spans=len(seen), unique_documents=len(documents)),
            'by_source': sources, 'score_profiles': tiers, 'rejected_lines_by_reason': rejected,
            'corpus': {'path': str(destination), 'sha256': digest(destination)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    screen = args.screen_dir.resolve(); output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.extract.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((screen / 'manifest.json').read_text())
        if manifest['stage'] != 'complete':
            raise ValueError('Document screening is not complete')
        index = ROOT / manifest['candidate_index']['path']
        if digest(index) != manifest['candidate_index']['sha256']:
            raise ValueError('Candidate index changed')
        identity = {'screen_manifest': str(screen / 'manifest.json'),
                    'screen_manifest_sha256': digest(screen / 'manifest.json'),
                    'candidate_index_sha256': digest(index),
                    'rule_sha256': digest(ROOT / 'training/web_document_quality.py'),
                    'runner_sha256': digest(__file__)}
        if (output / 'run.json').exists() and json.loads((output / 'run.json').read_text()) != identity:
            raise ValueError('Inputs or rules changed; use a new output directory')
        write_json(output / 'run.json', identity)
        for name in ['web_document_quality.py', 'select_web_prose_spans.py']:
            shutil.copyfile(ROOT / 'training' / name, output / ('snapshot-' + name))
        grouped = defaultdict(list)
        for batch in pq.ParquetFile(index).iter_batches():
            for row in batch.to_pylist():
                grouped[row['file_index']].append(row)
        raw = Path(manifest['identity']['raw_directory'])
        downloaded = json.loads((raw / 'download-manifest.json').read_text())
        if digest(raw / 'download-manifest.json') != manifest['identity']['source_manifest_sha256']:
            raise ValueError('Source manifest changed')
        jobs = [(i, str(raw / downloaded['files'][i]['filename']), rows, str(output)) for i, rows in grouped.items()]
        started = time.time(); stats = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for future in as_completed([executor.submit(extract_file, job) for job in jobs]):
                stats.append(future.result())
                state = {'stage': 'extracting', 'files': len(stats), 'seconds': round(time.time() - started, 1)}
                write_json(output / 'status.json', state)
                if len(stats) % 32 == 0:
                    print(json.dumps(state), flush=True)
        write_json(output / 'status.json', {'stage': 'merging', 'files': len(stats)})
        result = merge(output, stats)
        result.update(stage='complete', identity=identity, policy=SPAN_POLICY,
                      limitations=['Structural selection is not a semantic or factual quality guarantee.',
                                   'Every corpus row is an unchanged source span. Do not concatenate rows.',
                                   'Exact span deduplication only. Pair-level deduplication against old train/evaluation remains required.',
                                   'The upstream document screen excludes inherited and hash-reserved holdout documents.'],
                      seconds=round(time.time() - started, 1))
        write_json(output / 'manifest.json', result)
        write_json(output / 'status.json', {'stage': 'complete', **result['totals'], 'seconds': result['seconds']})
        print(json.dumps(result['totals']), flush=True)


if __name__ == '__main__':
    main()
