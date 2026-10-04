#!/usr/bin/env python3
"""Paired position-weight continuation smoke using the actual sharded trainer.

Fixed source-stratified training subset, uniform validation reservoir, no test
selection or model promotion. Frozen sources and checkpoints support resuming.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(os.environ.get('SUPER_READER_PROJECT_ROOT', Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / 'training/.deps'))
from text_policy import DATA_POLICY, require_data_policy
from sample_subset import record_source

BASE = ROOT / 'training/artifacts/padding-fixed-web-350m-v7-20260925/epoch-1-backend'
DATA = ROOT / 'training/data/processed/padding-fixed-web-350m-v7-20260925'
SOURCES = ('run_position_ablation.py', 'train_sharded.py','sample_subset.py', 'train_smoke.py',
           'text_policy.py', 'text-policy.json', 'unicode-symbols.json')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def allocate(counts, total):
    population = sum(counts.values())
    quotas = {key: total * count // population for key, count in counts.items()}
    order = sorted(counts, key=lambda key: (-(total * counts[key] % population), key))
    for key in order[:total - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def reservoir(pool, row, seen, limit, rng):
    if len(pool) < limit:
        pool.append(row)
    else:
        index = rng.randrange(seen)
        if index < limit:
            pool[index] = row


def prepare(run, samples, validation_count, seed):
    data = run / 'data'
    data.mkdir(exist_ok=True)
    source = run / 'source/training'
    source.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        shutil.copy2(ROOT / 'training' / name, source / name)
    manifest = read(DATA / 'manifest.json')
    require_data_policy(manifest)
    write(run / 'weighting-manifest.json', manifest)
    shutil.copy2(BASE / 'training-state.pt', run / 'initialization.pt')
    vocabulary_path = Path(manifest['vocabulary_path'])
    if not vocabulary_path.is_absolute():
        vocabulary_path = ROOT / vocabulary_path
    shutil.copy2(vocabulary_path, data / 'vocabulary.json')
    groups = defaultdict(list)
    for shard in manifest['shards']:
        path = Path(shard['path'])
        path = path if path.is_absolute() else ROOT / path
        groups[str(path.parent)].append(path)
    counts = {}
    for directory in groups:
        own = read(Path(directory) / 'manifest.json')
        parents = {str((Path(s['path']) if Path(s['path']).is_absolute() else ROOT / s['path']).parent)
                   for s in own['shards']} - {directory}
        counts[directory] = own['statistics']['samples'] - sum(counts[p] for p in parents)
    if sum(counts.values()) != manifest['statistics']['samples']:
        raise ValueError('Source population mismatch')
    # Sample before viewing either arm's results. Document IDs are retained for
    # paired document-bootstrap intervals; this remains a reused validation set.
    pool, eligible = [], 0
    rng = random.Random(seed + 1)
    with (DATA / 'validation.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            if len(row['tokens']) > 256:
                continue
            eligible += 1
            reservoir(pool, row, eligible, validation_count, rng)
    if len(pool) != validation_count:
        raise ValueError('Insufficient validation samples')
    (data / 'validation.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in pool))
    forbidden = {hashlib.sha256(''.join(r['tokens']).encode()).digest() for r in pool}
    # The trainer records the unchanged test file's size for identity, but
    # --defer-test prevents reading/scoring it. No sampled test is created.
    if not (data / 'test.jsonl').exists():
        (data / 'test.jsonl').symlink_to(DATA / 'test.jsonl')
    summary = read(DATA / 'summary.json')
    summary['splits']['validation'] = {'count': len(pool), 'sha256': sha(data / 'validation.jsonl'),
        'per_source': dict(Counter(record_source(r) for r in pool)), 'eligible_population': eligible}
    write(data / 'summary.json', summary)
    del pool
    training, sources = [], []
    allocation = allocate(counts, samples)
    for index, (directory, target) in enumerate(allocation.items()):
        rng = random.Random(seed + 100 + index)
        paths = list(groups[directory])
        rng.shuffle(paths)
        minimum = min(len(paths), max(2, math.ceil(target * 1.5 / (counts[directory] / len(paths)))))
        pool, eligible, rejected, scanned = [], 0, 0, []
        for path in paths:
            with path.open() as handle:
                for line_number, line in enumerate(handle):
                    row = json.loads(line)
                    if len(row[0]) > 256 or hashlib.sha256(row[0].encode()).digest() in forbidden:
                        rejected += 1
                        continue
                    eligible += 1
                    reservoir(pool, (row, [str(path), line_number]), eligible, target, rng)
            scanned.append({'path': str(path), 'sha256': sha(path)})
            if len(scanned) >= minimum and len(pool) == target:
                break
        if len(pool) != target:
            raise ValueError('Insufficient source rows: ' + directory)
        training.extend(pool)
        sources.append({'directory': directory, 'population': counts[directory], 'sampled': target,
                        'eligible_in_scanned_shards': eligible, 'rejected': rejected, 'shards': scanned})
        print(json.dumps({'stage': 'sampled-source', 'directory': directory, 'samples': target}), flush=True)
    keys = [hashlib.sha256(row[0].encode()).digest() for row, _ in training]
    if len(set(keys)) != samples:
        raise ValueError('Duplicate input in smoke sample; choose a fresh sample before comparing')
    random.Random(seed).shuffle(training)
    stats = {'samples': samples, 'tokens': 0, 'cells': [[0] * 10 for _ in range(4)],
             'maximum_sequence_length': 0, 'domain_samples': Counter(), 'random_baseline_sum': 0., 'center_correct': 0}
    shards = []
    for number, start in enumerate(range(0, samples, 50000)):
        path = data / f'train-{number:03d}.jsonl'
        rows = training[start:start + 50000]
        path.write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r, _ in rows))
        provenance = path.with_suffix('.source.jsonl')
        provenance.write_text(''.join(json.dumps(origin) + '\n' for _, origin in rows))
        shards.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)})
        for row, _ in rows:
            text, target, domain, bucket, position = row
            stats['tokens'] += len(text)
            stats['cells'][bucket][position] += 1
            stats['domain_samples'][manifest['domains'][domain]] += 1
            stats['maximum_sequence_length'] = max(stats['maximum_sequence_length'], len(text))
            stats['random_baseline_sum'] += 1 / (len(text) - 1)
            stats['center_correct'] += target == (len(text) - 1) // 2
    smoke = {**DATA_POLICY, 'format': manifest['format'], 'domains': manifest['domains'],
             'length_bucket_maximums': manifest['length_bucket_maximums'], 'position_bins': 10,
             'vocabulary_path': str(data / 'vocabulary.json'), 'statistics': stats, 'shards': shards}
    write(data / 'manifest.json', smoke)
    plan = {'samples': samples, 'validation_count': validation_count, 'seed': seed,
            'maximum_sample_length': 256, 'base_manifest_sha256': sha(DATA / 'manifest.json'),
            'weighting_manifest_sha256': sha(run / 'weighting-manifest.json'),
            'initialization_sha256': sha(run / 'initialization.pt'), 'sources': sources,
            'source_hashes': {name: sha(source / name) for name in SOURCES},
            'data_hashes': {name: sha(data / name) for name in ('manifest.json', 'validation.jsonl', 'summary.json', 'vocabulary.json')},
            'source_weighting': 'none', 'selection_metric': 'overall_validation_accuracy',
            'training': {'epochs': 1, 'learning_rate': .00003, 'batch_size': 512, 'max_tokens_per_batch': 8192},
            'comparison': 'Only position weighting changes; normalize each arm by its actual training-subset mean weight; same shuffled shards/batches and fresh AdamW.',
            'limitations': 'One seeded continuation, <=256 characters, proportional source-block quotas and random whole-shard clusters; validation is reused. No test or promotion.'}
    write(run / 'plan.json', plan)
    return plan


def command(run, arm, seed):
    result = [sys.executable, str(run / 'source/training/train_sharded.py'),
        '--manifest', str(run / 'data/manifest.json'), '--data-dir', str(run / 'data'),
        '--artifact-dir', str(run / arm), '--weighting-manifest', str(run / 'weighting-manifest.json'),
        '--position-weighting', arm, '--epochs', '1', '--channels', '192', '--residual-blocks', '8',
        '--learning-rate', '.00003', '--gradient-clip', '1',
        '--batch-size', '512', '--max-tokens-per-batch', '8192', '--checkpoint-shards', '1',
        '--seed', str(seed), '--defer-test']
    # Existing experiments run their frozen trainer with the recorded settings.
    # New experiments use the current trainer, which has no source weighting.
    if (run / 'plan.json').exists():
        legacy_power = read(run / 'plan.json')['training'].get('domain_weight_power')
        if legacy_power is not None:
            result += ['--domain-weight-power', str(legacy_power)]
    if (run / arm / 'training-state.pt').exists():
        result += ['--resume']
    else:
        result += ['--initialize-from', str(run / 'initialization.pt')]
    return result


def compare(run, plan):
    import numpy as np
    import torch
    # Compare using the exact frozen architecture and evaluation implementation.
    sys.path.insert(0, str(run / 'source/training'))
    from train_smoke import BoundaryChooser, configure_cpu, configure_mps, iterate_batches, batch_to_device
    from train_sharded import read_evaluation_records
    configure_cpu(1, 1)
    device = configure_mps()
    vocab = read(run / 'data/vocabulary.json')
    records = read_evaluation_records(run / 'data/validation.jsonl', vocab)
    index = {r['id']: i for i, r in enumerate(records)}
    if len(index) != len(records):
        raise ValueError('Validation IDs must uniquely identify paired predictions')
    predictions, reports = {}, {}
    for arm in ('inverse-cell', 'none'):
        reports[arm] = read(run / arm / 'validation-only.json')
        state = torch.load(run / arm / 'training-state.pt', map_location='cpu', weights_only=True)
        model = BoundaryChooser(len(vocab), 192, 8).to(device).eval()
        model.load_state_dict(state['model_state'])  # equal-exposure endpoint, even if epoch 0 wins
        predictions[arm] = np.zeros(len(records), dtype=np.int32)
        with torch.inference_mode():
            for batch in iterate_batches(records, 512, 8192, shuffle=False, seed=0):
                batch = batch_to_device(batch, device)
                predicted = model(batch['token_ids'], batch['token_mask'], batch['gap_mask']).argmax(1).cpu().numpy()
                for r, p in zip(batch['records'], predicted):
                    predictions[arm][index[r['id']]] = p
        del model, state
        torch.mps.empty_cache()
    targets = np.array([r['target_index'] for r in records])
    correct = {arm: p == targets for arm, p in predictions.items()}
    for arm in correct:
        if abs(float(correct[arm].mean()) - reports[arm]['endpoint_validation']['accuracy']) > 1e-10:
            raise ValueError('Paired predictions do not reproduce endpoint accuracy')
    delta = correct['none'].astype(float) - correct['inverse-cell'].astype(float)
    document_ids = [r.get('document_id', r['id']) for r in records]
    _, clusters = np.unique(document_ids, return_inverse=True)
    cluster_sum = np.bincount(clusters, weights=delta)
    cluster_count = np.bincount(clusters)
    rng = np.random.default_rng(plan['seed'])
    draws = []
    for _ in range(1000):
        chosen = rng.integers(0, len(cluster_sum), len(cluster_sum))
        draws.append(float(cluster_sum[chosen].sum() / cluster_count[chosen].sum()))
    strata = {}
    for dimension in ('length', 'position', 'corpus_source'):
        labels = []
        for r in records:
            n = len(r['tokens'])
            labels.append(record_source(r) if dimension == 'corpus_source' else
                          str(min(9, int((r['target_index'] + 1) / n * 10))) if dimension == 'position' else
                          str(next((end for end in (8, 16, 32, 64, 128, 256) if n <= end))))
        strata[dimension] = {}
        labels = np.array(labels)
        for label in np.unique(labels):
            mask = labels == label
            strata[dimension][str(label)] = {'count': int(mask.sum()), **{arm: float(values[mask].mean()) for arm, values in correct.items()}}
    result = {'scope': plan['limitations'], 'arms': reports,
        'difference_none_minus_weighted_pp': float(delta.mean() * 100),
        'paired_document_bootstrap_95_percent_interval_pp': [float(x * 100) for x in np.percentile(draws, [2.5, 97.5])],
        'none_only_correct': int(((delta) > 0).sum()), 'weighted_only_correct': int((delta < 0).sum()),
        'strata': strata, 'test_evaluated': False, 'promoted': False}
    np.savez_compressed(run / 'paired-predictions.npz', targets=targets, **predictions)
    write(run / 'comparison.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'training/artifacts/position-ablation-1m-20261004')
    parser.add_argument('--samples', type=int, default=1000000)
    parser.add_argument('--validation-count', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=2026100409)
    args = parser.parse_args()
    if min(args.samples, args.validation_count) < 1:
        raise ValueError('Sample counts must be positive')
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    with (run / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(run / 'plan.json') if (run / 'plan.json').exists() else prepare(run, args.samples, args.validation_count, args.seed)
        if any(plan[k] != getattr(args, k) for k in ('samples', 'validation_count', 'seed')):
            raise ValueError('Use the original sample counts and seed on resume')
        for name, expected in plan['source_hashes'].items():
            if sha(run / 'source/training' / name) != expected:
                raise ValueError('Frozen source changed: ' + name)
        for name, expected in plan['data_hashes'].items():
            if sha(run / 'data' / name) != expected:
                raise ValueError('Frozen data changed: ' + name)
        if sha(run / 'initialization.pt') != plan['initialization_sha256'] or sha(run / 'weighting-manifest.json') != plan['weighting_manifest_sha256']:
            raise ValueError('Initialization/weight reference changed')
        for shard in read(run / 'data/manifest.json')['shards']:
            if sha(shard['path']) != shard['sha256']:
                raise ValueError('Training shard changed')
        env = {**os.environ, 'PYTHONPATH': str(ROOT / 'training/.deps'), 'SUPER_READER_PROJECT_ROOT': str(ROOT), 'DEBUG': '0'}
        for arm in ('inverse-cell', 'none'):
            (run / arm).mkdir(exist_ok=True)
            while not (run / arm / 'validation-only.json').exists():
                write(run / 'status.json', {'stage': 'training', 'arm': arm, 'updated_at': time.time()})
                with (run / arm / 'training.log').open('a') as log:
                    result = subprocess.run(command(run, arm, args.seed), env=env, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode == 75:
                    continue
                if result.returncode or not (run / arm / 'validation-only.json').exists():
                    raise RuntimeError(f'{arm} stopped; inspect training.log, rerun to resume')
        write(run / 'status.json', {'stage': 'paired-validation', 'updated_at': time.time()})
        result = compare(run, plan)
        write(run / 'status.json', {'stage': 'complete', 'difference_pp': result['difference_none_minus_weighted_pp'], 'updated_at': time.time()})
        print(json.dumps({'run': str(run), 'difference_pp': result['difference_none_minus_weighted_pp']}), flush=True)


if __name__ == '__main__':
    main()
