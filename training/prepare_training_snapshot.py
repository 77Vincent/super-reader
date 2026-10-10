#!/usr/bin/env python3
"""Freeze a randomized selection of already committed preparation chunks.

Reference immutable shards in place; never publish into the active preparation
directory. The effective sample budget is estimated from complete random chunks
and the exact retained count is recorded by the trainer as it reads each shard.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

from sample_subset import keep_target, write_json


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def snapshot(args):
    source, destination = args.source_dir.resolve(), args.output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f'Snapshot output must be new: {destination}')
    plan = read(source / 'plan.json')
    status = read(source / 'status.json')
    plan_hash = sha(source / 'plan.json')
    chunks = []
    for index in range(status['completed_chunks']):
        directory = source / f'chunk-{index:06d}'
        metadata = read(directory / 'metadata.json')
        if metadata['index'] != index or metadata['plan_sha256'] != plan_hash:
            raise ValueError(f'Chunk identity mismatch: {directory}')
        for item in metadata['files'].values():
            if (directory / item['name']).stat().st_size != item['bytes']:
                raise ValueError(f'Chunk size mismatch: {directory}')
        chunks.append(metadata)
    if sum(item['counts']['samples'] for item in chunks) != status['samples']:
        raise ValueError('Status sample count does not match committed chunks')
    eligible_chunks = [item for item in chunks if item['counts']['samples']]
    sample = random.Random(args.seed + 1).sample(eligible_chunks, min(args.estimate_chunks, len(eligible_chunks)))
    estimate_read = estimate_kept = 0
    for item in sample:
        path = source / f'chunk-{item["index"]:06d}' / item['files']['data']['name']
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            for line in handle:
                text, target, *_ = json.loads(line)
                estimate_read += 1
                estimate_kept += keep_target(len(text), target, args.min_side_characters, args.max_sequence_length)
    if not estimate_kept:
        raise ValueError('No retained rows in the sizing sample')
    retention = estimate_kept / estimate_read
    target_raw = math.ceil(args.target_samples / retention)
    random.Random(args.seed).shuffle(eligible_chunks)
    selected, raw_samples = [], 0
    for item in eligible_chunks:
        selected.append(item)
        raw_samples += item['counts']['samples']
        if raw_samples >= target_raw:
            break
    if raw_samples < target_raw:
        raise ValueError('Not enough committed data for the estimated sample budget')
    print(json.dumps({'stage': 'selection', 'committed_chunks': len(chunks),
        'selected_chunks': len(selected), 'raw_samples': raw_samples,
        'estimated_effective_samples': round(raw_samples * retention),
        'sizing_sample_rows': estimate_read, 'sizing_sample_retained': estimate_kept}), flush=True)

    stats = {'samples': 0, 'tokens': 0, 'random_baseline_sum': 0., 'center_correct': 0,
             'maximum_sequence_length': 0, 'cells': [[0] * 10 for _ in range(4)],
             'position_histogram': [0] * 10}
    shards = []
    for ordinal, item in enumerate(selected):
        directory = source / f'chunk-{item["index"]:06d}'
        entry = item['files']['data']; path = directory / entry['name']
        if sha(path) != entry['sha256']:
            raise ValueError(f'Corrupt selected shard: {path}')
        shards.append({'path': str(path), 'bytes': entry['bytes'], 'sha256': entry['sha256'],
            'compression': 'gzip', 'rows': item['counts']['samples'],
            'preparation_chunk': item['index'], 'metadata_path': str(directory / 'metadata.json'),
            'metadata_sha256': sha(directory / 'metadata.json')})
        original = item['statistics']
        for key in ('samples', 'tokens', 'random_baseline_sum', 'center_correct'):
            stats[key] += original[key]
        stats['maximum_sequence_length'] = max(stats['maximum_sequence_length'], original['maximum_sequence_length'])
        stats['cells'] = [[a + b for a, b in zip(x, y)] for x, y in zip(stats['cells'], original['cells'])]
        stats['position_histogram'] = [a + b for a, b in zip(stats['position_histogram'], original['position_histogram'])]
        if (ordinal + 1) % 1000 == 0:
            print(json.dumps({'stage': 'verified-shards', 'verified': ordinal + 1, 'total': len(selected)}), flush=True)

    evaluation = Path(plan['evaluation'])
    if sha(evaluation / 'summary.json') != plan['evaluation_summary_sha256']:
        raise ValueError('Frozen evaluation summary changed')
    for split, entry in plan['holdouts'].items():
        if sha(Path(entry['path'])) != entry['sha256']:
            raise ValueError(f'Frozen {split} changed')
    if sha(Path(plan['vocabulary']['path'])) != plan['vocabulary']['sha256']:
        raise ValueError('Vocabulary changed')

    destination.mkdir(parents=True)
    for split, entry in plan['holdouts'].items():
        (destination / f'{split}.jsonl').symlink_to(Path(entry['path']))
    shutil.copyfile(evaluation / 'summary.json', destination / 'summary.json')
    shutil.copyfile(plan['vocabulary']['path'], destination / 'vocabulary.json')
    stats['corpus_source_samples'] = {'openbmb/Ultra-FineWeb (zh)': stats['samples']}
    selection = {'created_at': datetime.now(timezone.utc).isoformat(),
        'source_dir': str(source), 'source_plan_sha256': plan_hash, 'source_status': status,
        'seed': args.seed, 'sampling': 'uniform shuffle of all committed nonempty chunks; whole chunks retained',
        'target_effective_samples': args.target_samples, 'raw_samples': raw_samples,
        'estimated_effective_samples': round(raw_samples * retention),
        'effective_count_is_estimate': True, 'exact_count': 'trainer source-statistics.json as shards are read',
        'sizing_sample': {'chunks': [item['index'] for item in sample], 'rows': estimate_read,
                         'retained': estimate_kept, 'retention': retention},
        'training_subset': {'minimum_side_characters': args.min_side_characters,
                            'maximum_sequence_length': args.max_sequence_length},
        'historical_training_pair_deduplication': plan['historical_training_pair_deduplication'],
        'holdouts': plan['holdouts'], 'all_selected_data_sha256_verified': True}
    write_json(destination / 'selection.json', selection)
    write_json(destination / 'manifest.json', {**plan['policy'],
        'format': 'super-reader-sharded-training-v1', 'source': 'Frozen subset of committed new web chunks',
        'corpus_sources': ['web'], 'evaluation_source_dir': str(destination),
        'vocabulary_path': str(destination / 'vocabulary.json'), 'length_bucket_maximums': [8, 16, 32],
        'position_bins': 10, 'maximum_sequence_length': 0, 'shards': shards, 'statistics': stats,
        'selection': selection, 'preparation_plan_sha256': plan_hash})
    print(json.dumps({'stage': 'snapshot-complete', 'manifest': str(destination / 'manifest.json'),
        'selected_chunks': len(shards), 'raw_samples': raw_samples,
        'estimated_effective_samples': selection['estimated_effective_samples']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--target-samples', type=int, default=400_000_000)
    parser.add_argument('--estimate-chunks', type=int, default=24)
    parser.add_argument('--seed', type=int, default=2026101003)
    parser.add_argument('--min-side-characters', type=int, default=2)
    parser.add_argument('--max-sequence-length', type=int, default=256)
    args = parser.parse_args()
    if min(args.target_samples, args.estimate_chunks, args.min_side_characters) < 1 or args.max_sequence_length < 0:
        parser.error('Sample counts and minimum side must be positive; maximum length cannot be negative')
    snapshot(args)


if __name__ == '__main__':
    main()
