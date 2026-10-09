#!/usr/bin/env python3
"""Exhaust unscanned web records into a separate, resumable compressed corpus."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(os.environ.get('SUPER_READER_PROJECT_ROOT', Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / 'training/.deps'))
import pyarrow.parquet as pq
from prepare_web_data import (Bloom, DATA_POLICY, PROVENANCE_FORMAT, add_statistics,
    clean_document, document_signature, document_split, fresh_statistics,
    holdout_registries, labeled_samples, projected_key, quality_document, read, sha, write)

DEFAULT_BASE = ROOT / 'training/data/processed/padding-fixed-web-350m-v7-20260925'
DEFAULT_POLICY = ROOT / 'training/artifacts/padding-fixed-web-350m-v7-20260925/source/training'
FROZEN_POLICY_FILES = ('prepare_synthetic_data.py', 'text_policy.py', 'text-policy.json', 'unicode-symbols.json')
CURRENT_FILES = ('prepare_remaining_web_data.py', 'prepare_web_data.py', 'inspect_training_sample.py')
FORMAT = 'remaining-web-preparation-v1'


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def resolve(path):
    return (ROOT / path).resolve()


def make_plan(args):
    out = args.output_dir.resolve()
    if (out / 'plan.json').exists():
        raise FileExistsError('Use --resume for the existing preparation')
    base_path = args.base_manifest.resolve()
    base = read(base_path)
    configuration = base_path.parent / 'configuration.json'
    if 'web_source' in base and read(configuration)['seed'] != args.seed:
        raise ValueError('Keep the original document partition seed')
    policy_dir = args.policy_source.resolve()
    policy = read(policy_dir / 'text-policy.json')
    if any(base.get(k) != v for k, v in policy.items()):
        raise ValueError('Frozen preparation policy differs from the existing corpus')
    usage_path = args.usage_report.resolve()
    usage = read(usage_path)
    for name, digest in usage['inputs'].items():
        if sha(resolve(name)) != digest:
            raise ValueError(f'Cursor evidence changed: {name}')
    raw = args.raw_dir.resolve()
    downloaded = read(raw / 'download-manifest.json')
    verified = read(raw / 'verified-files.json')
    if downloaded['revision'] != verified['revision']:
        raise ValueError('Raw revision mismatch')
    if base.get('web_source', {}).get('revision', downloaded['revision']) != downloaded['revision']:
        raise ValueError('Base corpus uses another raw revision')
    if len(downloaded['files']) != len(usage['files']):
        raise ValueError('Cursor file inventory mismatch')
    files = []
    for i, (entry, cursor) in enumerate(zip(downloaded['files'], usage['files'])):
        path = raw / entry['filename']
        if entry['filename'] != cursor['file'] or path.stat().st_size != entry['bytes']:
            raise ValueError(f'Raw file changed: {path}')
        if verified['files'].get(entry['filename']) != {k: entry[k] for k in ('bytes', 'sha256')}:
            raise ValueError(f'Missing download verification: {path}')
        metadata = pq.ParquetFile(path).metadata
        order = list(range(metadata.num_row_groups))
        random.Random(args.seed + i).shuffle(order)
        start = cursor['last_processed_group']
        ordinal, used = start['group_ordinal'], start['processed_rows_in_group']
        if not 0 <= ordinal < len(order) or order[ordinal] != start['physical_row_group']:
            raise ValueError('Cursor traversal order mismatch')
        if not 0 <= used <= metadata.row_group(order[ordinal]).num_rows:
            raise ValueError('Invalid row cursor')
        scanned = sum(metadata.row_group(g).num_rows for g in order[:ordinal]) + used
        if scanned != cursor['scanned_documents'] or metadata.num_rows != cursor['total_documents']:
            raise ValueError('Cursor cannot reproduce logged document count')
        files.append({**entry, 'mtime_ns': path.stat().st_mtime_ns,
                      'order': order, 'start': {'group': ordinal, 'row': used},
                      'remaining_documents': metadata.num_rows - scanned})
    evaluation = args.evaluation.resolve()
    summary = read(evaluation / 'summary.json')
    if any(summary.get(k) != v for k, v in policy.items()):
        raise ValueError('Evaluation preparation policy differs')
    holdouts = {}
    for split in ('validation', 'test'):
        p = evaluation / f'{split}.jsonl'
        digest = sha(p)
        if digest != summary['splits'][split]['sha256']:
            raise ValueError(f'Frozen {split} checksum mismatch')
        holdouts[split] = {'path': str(p), 'sha256': digest, 'bytes': p.stat().st_size}
    registries = holdout_registries(evaluation)
    if not registries:
        raise ValueError('Missing holdout document registry')
    snapshot = out / 'source'
    snapshot.mkdir()
    for name in FROZEN_POLICY_FILES:
        shutil.copy2(policy_dir / name, snapshot / name)
    for name in CURRENT_FILES:
        shutil.copy2(ROOT / 'training' / name, snapshot / name)
    vocabulary = resolve(base['vocabulary_path'])
    plan = dict(format=FORMAT, created_at=timestamp(), output_dir=str(out),
        base_manifest=str(base_path), base_manifest_sha256=sha(base_path),
        usage_report=str(usage_path), usage_report_sha256=sha(usage_path),
        raw_directory=str(raw), repository=downloaded['repository'], revision=downloaded['revision'],
        download_manifest_sha256=sha(raw / 'download-manifest.json'), files=files,
        remaining_documents=sum(f['remaining_documents'] for f in files), seed=args.seed,
        documents_per_shard=args.documents_per_shard, seen_bloom_bytes=args.seen_bloom_bytes,
        holdout_bloom_bytes=args.holdout_bloom_bytes, reserve_bytes=int(args.reserve_gib * 2**30),
        compression='gzip level 3; independent compact JSONL shards', policy=policy,
        vocabulary={'path': str(vocabulary), 'sha256': sha(vocabulary)},
        evaluation=str(evaluation), evaluation_summary_sha256=sha(evaluation / 'summary.json'),
        holdouts=holdouts, holdout_registries={str(p): sha(p) for p in registries},
        source_hashes={p.name: sha(p) for p in snapshot.iterdir()},
        old_training_shards_included=False, historical_training_pair_deduplication=False,
        new_pair_deduplication='NFKC Han-projected input Bloom + exact current-chunk set',
        boundary_document='Skip the entire previously visited last document, including unused pairs in it',
        training_subset={'minimum_side_characters': 2, 'maximum_sequence_length': 256,
                         'applied_during_preparation': False}, automatic_training=False)
    write(out / 'plan.json', plan)
    write(out / 'status.json', {'stage': 'prepared', 'updated_at': timestamp(),
                               'remaining_documents': plan['remaining_documents']})
    return plan


def verify_plan(out, plan):
    if plan['format'] != FORMAT:
        raise ValueError('Unsupported plan')
    for name, digest in plan['source_hashes'].items():
        if sha(out / 'source' / name) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    for key, hash_key in [('base_manifest', 'base_manifest_sha256'), ('usage_report', 'usage_report_sha256')]:
        if sha(Path(plan[key])) != plan[hash_key]:
            raise ValueError(f'Frozen input changed: {key}')
    if sha(Path(plan['evaluation']) / 'summary.json') != plan['evaluation_summary_sha256']:
        raise ValueError('Evaluation summary changed')
    if sha(Path(plan['vocabulary']['path'])) != plan['vocabulary']['sha256']:
        raise ValueError('Vocabulary changed')
    if sha(Path(plan['raw_directory']) / 'download-manifest.json') != plan['download_manifest_sha256']:
        raise ValueError('Raw inventory changed')
    for entry in plan['files']:
        stat = (Path(plan['raw_directory']) / entry['filename']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (entry['bytes'], entry['mtime_ns']):
            raise ValueError(f'Raw source changed: {entry["filename"]}')
    for item in plan['holdouts'].values():
        if sha(Path(item['path'])) != item['sha256']:
            raise ValueError('Evaluation changed')
    for name, digest in plan['holdout_registries'].items():
        if sha(Path(name)) != digest:
            raise ValueError('Holdout document registry changed')


def save_bloom(path, bloom, committed_chunks, plan_hash):
    """Atomic full snapshot; only contains keys from already published chunks."""
    temporary = path.with_suffix('.tmp')
    header = {'bytes': len(bloom.bits), 'chunks': committed_chunks, 'plan_sha256': plan_hash}
    digest = hashlib.sha256()
    with temporary.open('wb') as handle:
        handle.write((json.dumps(header) + '\n').encode())
        view = memoryview(bloom.bits)
        for offset in range(0, len(view), 8 * 1024**2):
            block = view[offset:offset + 8 * 1024**2]
            handle.write(block)
            digest.update(block)
        handle.write(digest.digest())
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_bloom(path, bloom, plan_hash):
    if not path.exists():
        return 0
    with path.open('rb') as handle:
        header = json.loads(handle.readline())
        if header['bytes'] != len(bloom.bits) or header['plan_sha256'] != plan_hash:
            raise ValueError('Bloom snapshot configuration mismatch')
        digest = hashlib.sha256()
        view = memoryview(bloom.bits)
        for offset in range(0, len(view), 8 * 1024**2):
            block = view[offset:offset + 8 * 1024**2]
            if handle.readinto(block) != len(block):
                raise ValueError('Truncated Bloom snapshot')
            digest.update(block)
        if handle.read(32) != digest.digest() or handle.read(1):
            raise ValueError('Corrupt Bloom snapshot')
    return header['chunks']


def zero_counts():
    return dict(documents=0, quality_filtered=0, holdout_documents_filtered=0,
                holdout_partition_filtered=0, holdout_pairs_filtered=0,
                duplicate_or_bloom_filtered=0, samples=0, contributing_documents=0)


class Chunk:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir()
        self.handles = {}
        self.files = {}
        for name in ('data', 'pairs', 'documents'):
            raw = (directory / f'{name}.jsonl.gz').open('wb')
            self.files[name] = raw
            self.handles[name] = gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=3, mtime=0)
        self.keys = set()
        self.statistics = fresh_statistics()
        self.counts = zero_counts()
        self.surface_rejected = {}
        self.per_source = {}

    def emit(self, name, value):
        self.handles[name].write((json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n').encode())

    def close(self):
        for name, handle in self.handles.items():
            raw = self.files[name]
            if raw.closed:
                continue
            handle.close()
            raw.flush()
            os.fsync(raw.fileno())
            raw.close()


def publish_chunk(out, number, chunk, cursor, plan_hash):
    chunk.close()
    metadata = dict(index=number, next_cursor=cursor, plan_sha256=plan_hash,
        counts=chunk.counts, statistics=chunk.statistics, surface_rejected=chunk.surface_rejected,
        per_source=chunk.per_source, files={})
    for name in ('data', 'pairs', 'documents'):
        path = chunk.directory / f'{name}.jsonl.gz'
        metadata['files'][name] = {'name': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)}
    write(chunk.directory / 'metadata.json', metadata)
    with (chunk.directory / 'metadata.json').open('rb') as handle:
        os.fsync(handle.fileno())
    chunk.directory.rename(out / f'chunk-{number:06d}')
    directory_fd = os.open(out, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return metadata


def remaining_documents(plan, cursor):
    for file_index in range(cursor['file'], len(plan['files'])):
        info = plan['files'][file_index]
        start = cursor if file_index == cursor['file'] else info['start']
        parquet = pq.ParquetFile(Path(plan['raw_directory']) / info['filename'])
        for group_index in range(start['group'], len(info['order'])):
            group = info['order'][group_index]
            skip = start['row'] if group_index == start['group'] else 0
            row = 0
            for batch in parquet.iter_batches(batch_size=256, row_groups=[group], columns=['content', 'source'], use_threads=False):
                for record in batch.to_pylist():
                    current = row
                    row += 1
                    if current < skip:
                        continue
                    yield file_index, group, current, record, {'file': file_index, 'group': group_index, 'row': row}
        yield file_index, None, None, None, {'file': file_index + 1,
            **(plan['files'][file_index + 1]['start'] if file_index + 1 < len(plan['files']) else {'group': 0, 'row': 0})}


def finish(out, plan, chunks):
    stats = fresh_statistics()
    counts = zero_counts()
    shards = []
    for item in chunks:
        for k in counts:
            counts[k] += item['counts'][k]
        s = item['statistics']
        for key in ('samples', 'tokens', 'random_baseline_sum', 'center_correct'):
            stats[key] += s[key]
        stats['maximum_sequence_length'] = max(stats['maximum_sequence_length'], s['maximum_sequence_length'])
        stats['cells'] = [[a+b for a,b in zip(x,y)] for x,y in zip(stats['cells'],s['cells'])]
        stats['position_histogram'] = [a+b for a,b in zip(stats['position_histogram'],s['position_histogram'])]
        if not s['samples']:
            continue
        folder = out / f'chunk-{item["index"]:06d}'
        def entry(name):
            f = item['files'][name]
            return {'path': str(folder / f['name']), 'bytes': f['bytes'], 'sha256': f['sha256'], 'compression': 'gzip'}
        shards.append({**entry('data'), 'rows': s['samples'], 'provenance': {
            **entry('pairs'), 'rows': s['samples'], 'format': PROVENANCE_FORMAT,
            'documents_path': str(folder / item['files']['documents']['name']),
            'documents_sha256': item['files']['documents']['sha256'],
            'raw_directory': plan['raw_directory']}})
    if counts['documents'] != plan['remaining_documents'] or not stats['samples']:
        raise ValueError('Preparation did not exhaust the remaining records')
    stats['domain_samples'] = {'web': stats['samples']}
    # Inherited holdouts stay byte-identical; do not regenerate or enlarge them.
    for split, entry in plan['holdouts'].items():
        target = out / f'{split}.jsonl'
        if not target.exists():
            shutil.copyfile(entry['path'], target.with_suffix('.tmp'))
            target.with_suffix('.tmp').replace(target)
        if sha(target) != entry['sha256']:
            raise ValueError('Published holdout differs')
    vocabulary = out / 'vocabulary.json'
    shutil.copyfile(plan['vocabulary']['path'], vocabulary)
    if sha(vocabulary) != plan['vocabulary']['sha256']:
        raise ValueError('Vocabulary changed')
    protection = {'document_split': 'Same 96/2/2 document hash partition and inherited holdout registries',
        'pair_deduplication': 'Projected input against all frozen evaluation rows and new rows; Bloom false positives may discard extra samples',
        'historical_training_pair_deduplication': False, 'old_raw_records': 'Skipped using reconstructed per-file traversal cursors',
        'near_duplicate_audit': 'Not performed'}
    summary = read(Path(plan['evaluation']) / 'summary.json')
    write(out / 'summary.json', {**plan['policy'], 'splits': summary['splits'],
                                'source_directory': plan['evaluation'], 'protection': protection})
    write(out / 'manifest.json', {**plan['policy'], 'format': 'super-reader-sharded-training-v1',
        'source': 'Previously unscanned Ultra-FineWeb Chinese records only',
        'domains': ['web'], 'evaluation_source_dir': str(out), 'vocabulary_path': str(vocabulary),
        'length_bucket_maximums': [8,16,32], 'position_bins': 10, 'maximum_sequence_length': 0,
        'shards': shards, 'statistics': stats, 'counts': counts, 'protection': protection,
        'web_source': {'repository': plan['repository'], 'revision': plan['revision'],
                       'total_training_samples': stats['samples']},
        'preparation_plan_sha256': sha(out / 'plan.json'), 'frozen_source': str(out / 'source'),
        'training_subset': plan['training_subset']})
    return counts


def worker(out, session_documents=0):
    plan = read(out / 'plan.json')
    verify_plan(out, plan)
    if DATA_POLICY != plan['policy']:
        raise ValueError('Run the frozen worker to preserve the original preparation policy')
    plan_hash = sha(out / 'plan.json')
    started = time.monotonic()
    chunks = []
    totals = zero_counts()
    for path in sorted(out.glob('chunk-*/metadata.json')):
        item = read(path)
        if item['index'] != len(chunks) or item['plan_sha256'] != plan_hash:
            raise ValueError('Invalid committed chunk sequence')
        for entry in item['files'].values():
            p = path.parent / entry['name']
            if p.stat().st_size != entry['bytes'] or sha(p) != entry['sha256']:
                raise ValueError(f'Committed output changed: {p}')
        chunks.append(item)
        for k in totals:
            totals[k] += item['counts'][k]
    initial_documents = totals['documents']
    def status(stage, **kwargs):
        value = dict(stage=stage, updated_at=timestamp(), worker_pid=os.getpid(),
            total_remaining_documents=plan['remaining_documents'], completed_chunks=len(chunks),
            **totals, session_seconds=time.monotonic()-started, **kwargs)
        write(out / 'status.json', value)
        print(json.dumps(value), flush=True)
    if (out / 'manifest.json').exists():
        status('complete')
        return
    reserve = plan['reserve_bytes']
    if shutil.disk_usage(out).free < reserve + 2 * plan['seen_bloom_bytes']:
        status('stopped', reason='disk_reserve')
        return
    stop = {'requested': False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(requested=True))
    status('loading_deduplication')
    seen = Bloom(plan['seen_bloom_bytes'])
    restored = load_bloom(out / 'seen.bloom', seen, plan_hash)
    if not 0 <= restored <= len(chunks):
        raise ValueError('Bloom snapshot exceeds committed data')
    for item in chunks[restored:]:
        path = out / f'chunk-{item["index"]:06d}' / item['files']['data']['name']
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            for line in handle:
                seen.add(projected_key(json.loads(line)[0]))
    holdouts = Bloom(plan['holdout_bloom_bytes'])
    if (out / 'holdouts.bloom').exists():
        load_bloom(out / 'holdouts.bloom', holdouts, plan_hash)
    else:
        for split, entry in plan['holdouts'].items():
            status('indexing_holdout', split=split)
            with Path(entry['path']).open() as handle:
                for line in handle:
                    holdouts.add(projected_key(''.join(json.loads(line)['tokens'])))
        save_bloom(out / 'holdouts.bloom', holdouts, 0, plan_hash)
    blocked = {key for p in plan['holdout_registries'] for key in read(p)}
    cursor = chunks[-1]['next_cursor'] if chunks else {'file': 0, **plan['files'][0]['start']}
    staging = out / '.chunk-writing'
    if staging.exists():
        shutil.rmtree(staging)  # Only this run's explicitly uncommitted transaction.
    chunk = None
    def commit(next_cursor):
        nonlocal chunk
        if chunk is None:
            return
        item = publish_chunk(out, len(chunks), chunk, next_cursor, plan_hash)
        chunks.append(item)
        for key in chunk.keys:
            seen.add(key)
        for key in totals:
            totals[key] += item['counts'][key]
        chunk = None
        status('preparing', cursor=next_cursor,
               documents_per_second=(totals['documents']-initial_documents)/max(.001,time.monotonic()-started))
    try:
        for file_index, group, row, record, next_cursor in remaining_documents(plan, cursor):
            if record is None:
                commit(next_cursor)
                # Save at file boundaries, avoiding a full replay after interruptions.
                status('saving_deduplication', cursor=next_cursor)
                save_bloom(out / 'seen.bloom', seen, len(chunks), plan_hash)
                continue
            if chunk is None:
                chunk = Chunk(staging)
            chunk.counts['documents'] += 1
            content = record['content'] or ''
            source = str(record['source'] or 'unknown')
            document = document_signature(content)
            if document in blocked:
                chunk.counts['holdout_documents_filtered'] += 1
            elif document_split(document, plan['seed']) != 'train':
                chunk.counts['holdout_partition_filtered'] += 1
            elif not quality_document(clean_document(content)):
                chunk.counts['quality_filtered'] += 1
            else:
                document_ref = None
                for pair_index, (text, target, punctuation) in enumerate(labeled_samples(content, chunk.surface_rejected)):
                    key = projected_key(text)
                    if holdouts.contains(key):
                        chunk.counts['holdout_pairs_filtered'] += 1
                        continue
                    if key in chunk.keys or seen.contains(key):
                        chunk.counts['duplicate_or_bloom_filtered'] += 1
                        continue
                    chunk.keys.add(key)
                    if document_ref is None:
                        document_ref = chunk.handles['documents'].tell()
                        chunk.emit('documents', {'byte_offset': document_ref, 'document_id': document,
                            'raw_file': plan['files'][file_index]['filename'], 'row_group': group,
                            'row_index': row, 'source': source,
                            'content_sha256': hashlib.sha256(content.encode()).hexdigest()})
                        chunk.counts['contributing_documents'] += 1
                    bucket, position = add_statistics(chunk.statistics, text, target)
                    chunk.emit('data', [text, target, 0, bucket, position])
                    chunk.emit('pairs', [document_ref, pair_index, punctuation])
                    chunk.counts['samples'] += 1
                    chunk.per_source[source] = chunk.per_source.get(source, 0) + 1
            session_limit = session_documents and totals['documents'] + chunk.counts['documents'] - initial_documents >= session_documents
            if chunk.counts['documents'] >= plan['documents_per_shard'] or stop['requested'] or session_limit:
                commit(next_cursor)
                low_space = shutil.disk_usage(out).free < reserve + plan['seen_bloom_bytes']
                if stop['requested'] or session_limit or low_space:
                    save_bloom(out / 'seen.bloom', seen, len(chunks), plan_hash)
                    status('stopped', reason='signal' if stop['requested'] else 'disk_reserve' if low_space else 'session_limit')
                    return
        commit({'file': len(plan['files']), 'group': 0, 'row': 0})
        finish(out, plan, chunks)
        status('complete')
    except Exception as error:
        status('failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if chunk is not None:
            chunk.close()  # Uncommitted bytes are discarded on the next resume.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--base-manifest', type=Path, default=DEFAULT_BASE/'manifest.json')
    p.add_argument('--evaluation', type=Path, default=DEFAULT_BASE)
    p.add_argument('--usage-report', type=Path, default=ROOT/'training/artifacts/web-source-usage-20261009/usage-and-cursors.json')
    p.add_argument('--raw-dir', type=Path, default=ROOT/'training/data/raw/ultra-fineweb-zh')
    p.add_argument('--policy-source', type=Path, default=DEFAULT_POLICY)
    p.add_argument('--seed', type=int, default=20260915)
    p.add_argument('--documents-per-shard', type=int, default=5000)
    p.add_argument('--seen-bloom-bytes', type=int, default=4 * 1024**3)
    p.add_argument('--holdout-bloom-bytes', type=int, default=64 * 1024**2)
    p.add_argument('--reserve-gib', type=float, default=32)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--session-documents', type=int, default=0, help='Optional finite pilot; 0 exhausts all remaining records')
    args = p.parse_args()
    if args.documents_per_shard < 1 or not math.isfinite(args.reserve_gib) or args.reserve_gib < 0 or args.session_documents < 0:
        p.error('Invalid size or document limit')
    for size in (args.seen_bloom_bytes, args.holdout_bloom_bytes):
        if size < 1024 or size & (size-1):
            p.error('Bloom sizes must be powers of two >= 1024')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        with (out / '.worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker(out, args.session_documents)
        return
    with (out / '.coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(out / 'plan.json') if args.resume else make_plan(args)
        verify_plan(out, plan)
        if args.prepare_only:
            print(json.dumps({'stage':'prepared','output':str(out),'remaining_documents':plan['remaining_documents']}))
            return
        env = dict(os.environ, SUPER_READER_PROJECT_ROOT=str(ROOT), PYTHONPATH=str(ROOT/'training/.deps'), PYTHONUNBUFFERED='1')
        awake = subprocess.Popen(['/usr/bin/caffeinate','-is','-w',str(os.getpid())], stdin=subprocess.DEVNULL) if sys.platform=='darwin' else None
        child = None
        def stop(signum, _frame):
            if child is not None and child.poll() is None:
                child.send_signal(signum)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop)
        try:
            with (out/'preparation.log').open('ab') as log:
                child = subprocess.Popen([sys.executable,str(out/'source/prepare_remaining_web_data.py'),
                    '--worker','--output-dir',str(out),'--session-documents',str(args.session_documents)],
                    cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                print(json.dumps({'coordinator_pid':os.getpid(),'worker_pid':child.pid,'output':str(out)}),flush=True)
                code = child.wait()
                if code:
                    previous = read(out / 'status.json')
                    write(out / 'status.json', {**previous, 'stage': 'failed', 'updated_at': timestamp(),
                        'exit_code': code, 'error': previous.get('error', 'See preparation.log')})
                    raise RuntimeError(f'Preparation exited {code}; see {out}/preparation.log')
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait()
            if awake is not None:
                awake.terminate()
                awake.wait()


if __name__ == '__main__':
    main()
