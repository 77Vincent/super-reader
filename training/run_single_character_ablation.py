#!/usr/bin/env python3
"""Filter single-character boundary sides during a paired continuation smoke.

Reuse the original smoke's initialization, frozen trainer, weight reference, and
complete validation set. Compare filtered endpoints to the saved unfiltered ones.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
from itertools import zip_longest
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(os.environ.get('SUPER_READER_PROJECT_ROOT', Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / 'training/.deps'))
from run_position_ablation import SOURCES, command, compare, read, sha, write

ARMS = ('inverse-cell', 'none')
DEFAULT_CONTROL = ROOT / 'training/artifacts/position-ablation-1m-20261004'
DEFAULT_RUN = ROOT / 'training/artifacts/single-character-ablation-1m-20261004'


def single_character_side(text, target):
    if not 0 <= target < len(text) - 1:
        raise ValueError('Invalid target gap')
    return target == 0 or target == len(text) - 2


def filter_shard(source, destination, domains, statistics):
    """Retain original row bytes/order and aligned source pointers; no relabeling."""
    removed = 0
    with source.open() as rows, source.with_suffix('.source.jsonl').open() as origins, \
            destination.open('w') as output, destination.with_suffix('.source.jsonl').open('w') as provenance:
        for line, origin in zip_longest(rows, origins):
            if line is None or origin is None:
                raise ValueError('Source sidecar row count mismatch')
            text, target, domain, bucket, position = json.loads(line)
            if single_character_side(text, target):
                removed += 1
                continue
            output.write(line)
            provenance.write(origin)
            statistics['samples'] += 1
            statistics['tokens'] += len(text)
            statistics['cells'][bucket][position] += 1
            statistics['domain_samples'][domains[domain]] += 1
            statistics['maximum_sequence_length'] = max(statistics['maximum_sequence_length'], len(text))
            statistics['random_baseline_sum'] += 1 / (len(text) - 1)
            statistics['center_correct'] += target == (len(text) - 1) // 2
    return removed


def prepare(run, control):
    base = read(control / 'plan.json')
    expected_training = {'epochs': 1, 'learning_rate': .00003, 'domain_weight_power': .65,
                         'batch_size': 512, 'max_tokens_per_batch': 8192}
    if base['training'] != expected_training:
        raise ValueError('Control hyperparameters differ from the supported paired smoke')
    data, source_dir = run / 'data', run / 'source/training'
    data.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        source = control / 'source/training' / name
        if sha(source) != base['source_hashes'][name]:
            raise ValueError('Control frozen source changed: ' + name)
        if name == 'run_position_ablation.py' and sha(ROOT / 'training' / name) != sha(source):
            raise ValueError('Use the original comparator/command helper for this experiment')
        shutil.copy2(source, source_dir / name)
    shutil.copy2(Path(__file__), source_dir / Path(__file__).name)
    for name, expected in [('initialization.pt', base['initialization_sha256']),
                           ('weighting-manifest.json', base['weighting_manifest_sha256'])]:
        if sha(control / name) != expected:
            raise ValueError('Control initialization or weights changed')
        shutil.copy2(control / name, run / name)
    for name, expected in base['data_hashes'].items():
        if sha(control / 'data' / name) != expected:
            raise ValueError('Control data changed: ' + name)
    for name in ('vocabulary.json', 'validation.jsonl', 'summary.json'):
        shutil.copy2(control / 'data' / name, data / name)
    # Required only for the frozen trainer's identity stat; --defer-test prevents scoring.
    if not (data / 'test.jsonl').exists():
        (data / 'test.jsonl').symlink_to((control / 'data/test.jsonl').resolve())
    manifest = read(control / 'data/manifest.json')
    statistics = {'samples': 0, 'tokens': 0, 'cells': [[0] * 10 for _ in range(4)],
                  'domain_samples': Counter(), 'maximum_sequence_length': 0,
                  'random_baseline_sum': 0., 'center_correct': 0}
    shards, removals, source_hashes = [], [], {}
    for shard in manifest['shards']:
        source = Path(shard['path'])
        source = source if source.is_absolute() else ROOT / source
        if sha(source) != shard['sha256']:
            raise ValueError('Control training shard changed')
        origin = source.with_suffix('.source.jsonl')
        source_hashes[str(source)] = shard['sha256']
        source_hashes[str(origin)] = sha(origin)
        destination = data / source.name
        removed = filter_shard(source, destination, manifest['domains'], statistics)
        shards.append({'path': str(destination), 'bytes': destination.stat().st_size, 'sha256': sha(destination),
                       'source_sidecar_sha256': sha(destination.with_suffix('.source.jsonl'))})
        removals.append({'source': str(source), 'removed': removed})
    removed = sum(r['removed'] for r in removals)
    if statistics['samples'] + removed != base['samples'] or not statistics['samples']:
        raise ValueError('Filtered population mismatch')
    manifest.update(statistics=statistics, shards=shards, vocabulary_path=str(data / 'vocabulary.json'),
                    training_subset_filter={'minimum_side_characters': 2, 'validation_filtered': False})
    write(data / 'manifest.json', manifest)
    # Freeze the already evaluated controls so later edits cannot change the comparison.
    (run / 'control').mkdir(exist_ok=True)
    for name in ('paired-predictions.npz', 'comparison.json'):
        shutil.copy2(control / name, run / 'control' / name)
    plan = {'control_run': str(control), 'control_plan_sha256': sha(control / 'plan.json'),
            'control_source_hashes': source_hashes,
            'control_samples': base['samples'], 'samples': statistics['samples'], 'removed': removed,
            'removed_by_shard': removals, 'validation_count': base['validation_count'],
            'maximum_sample_length': base['maximum_sample_length'], 'seed': base['seed'],
            'training': expected_training,
            'initialization_sha256': base['initialization_sha256'],
            'weighting_manifest_sha256': base['weighting_manifest_sha256'],
            'source_hashes': {name: sha(source_dir / name) for name in (*SOURCES, Path(__file__).name)},
            'data_hashes': {name: sha(data / name) for name in base['data_hashes']},
            'control_hashes': {name: sha(run / 'control' / name) for name in ('paired-predictions.npz', 'comparison.json')},
            'comparison': 'Within each position-weight setting, remove only rows with either side exactly one normalized code point. Freeze domain/position tables; recompute actual subset mean normalization. Keep all candidate gaps.',
            'limitations': 'One seeded short continuation from a model previously trained on single-character sides; not training from scratch. Training rows are removed without replacement. Shard/row order and batching algorithm/seed are retained, but removing rows changes shuffled batches and potentially update count. Same reused validation, including single-character sides; no test or promotion.'}
    write(run / 'plan.json', plan)
    return plan


def verify(run, plan):
    for directory, key in [('source/training', 'source_hashes'), ('data', 'data_hashes'), ('control', 'control_hashes')]:
        for name, expected in plan[key].items():
            if sha(run / directory / name) != expected:
                raise ValueError('Frozen file changed: ' + directory + '/' + name)
    for name, key in [('initialization.pt', 'initialization_sha256'), ('weighting-manifest.json', 'weighting_manifest_sha256')]:
        if sha(run / name) != plan[key]:
            raise ValueError('Frozen initialization/weight reference changed')
    for shard in read(run / 'data/manifest.json')['shards']:
        if sha(shard['path']) != shard['sha256'] or sha(Path(shard['path']).with_suffix('.source.jsonl')) != shard['source_sidecar_sha256']:
            raise ValueError('Frozen filtered shard or sidecar changed')


def group_result(before, after, document_ids, seed):
    import numpy as np
    if len(before) == 0:
        return {'count': 0}
    delta = after.astype(float) - before.astype(float)
    _, clusters = np.unique(document_ids, return_inverse=True)
    sums, counts = np.bincount(clusters, weights=delta), np.bincount(clusters)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(1000):
        chosen = rng.integers(0, len(sums), len(sums))
        draws.append(sums[chosen].sum() / counts[chosen].sum())
    return {'count': len(before), 'unfiltered_correct': int(before.sum()), 'filtered_correct': int(after.sum()),
            'unfiltered_accuracy': float(before.mean()), 'filtered_accuracy': float(after.mean()),
            'difference_pp': float(delta.mean() * 100),
            'newly_correct': int((delta > 0).sum()), 'newly_wrong': int((delta < 0).sum()),
            'paired_document_bootstrap_95_percent_interval_pp': [float(x * 100) for x in np.percentile(draws, [2.5, 97.5])]}


def compare_to_control(run, plan):
    import numpy as np
    records = [json.loads(line) for line in (run / 'data/validation.jsonl').open()]
    targets = np.array([r['target_index'] for r in records])
    lengths = np.array([len(r['tokens']) for r in records])
    documents = np.array([r.get('document_id', r['id']) for r in records])
    single = (targets == 0) | (targets == lengths - 2)
    masks = {'all': np.ones(len(records), dtype=bool), 'single_character_side': single, 'both_sides_at_least_two': ~single}
    base, filtered = read(run / 'control/comparison.json'), read(run / 'comparison.json')
    with np.load(run / 'control/paired-predictions.npz', allow_pickle=False) as a, \
            np.load(run / 'paired-predictions.npz', allow_pickle=False) as b:
        if not np.array_equal(a['targets'], targets) or not np.array_equal(b['targets'], targets):
            raise ValueError('Prediction alignment mismatch')
        arms = {}
        for arm in ARMS:
            old, new = a[arm] == targets, b[arm] == targets
            old_report, new_report = base['arms'][arm], filtered['arms'][arm]
            if old_report['configuration'] != new_report['configuration']:
                raise ValueError('Control configuration mismatch: ' + arm)
            if old_report['initialization']['validation_before_training'] != new_report['initialization']['validation_before_training']:
                raise ValueError('Initial model validation differs')
            for key in ('position_weights', 'domain_weights'):
                if old_report['training_weighting'][key] != new_report['training_weighting'][key]:
                    raise ValueError('Weight reference changed: ' + key)
            for correct, report in ((old, old_report), (new, new_report)):
                if abs(float(correct.mean()) - report['endpoint_validation']['accuracy']) > 1e-12:
                    raise ValueError('Endpoint accuracy mismatch')
            groups = {name: group_result(old[mask], new[mask], documents[mask], plan['seed']) for name, mask in masks.items()}
            for field in ('count', 'unfiltered_correct', 'filtered_correct', 'newly_correct', 'newly_wrong'):
                if groups['all'][field] != sum(groups[name][field] for name in ('single_character_side', 'both_sides_at_least_two')):
                    raise ValueError('Group totals do not reconcile')
            strata = {}
            for dimension, labels in [('position', np.minimum(9, (targets + 1) * 10 // lengths)),
                                      ('length', np.array([next(end for end in (8, 16, 32, 64, 128, 256) if n <= end) for n in lengths]))]:
                strata[dimension] = {str(label): {'count': int((labels == label).sum()),
                    'unfiltered_accuracy': float(old[labels == label].mean()),
                    'filtered_accuracy': float(new[labels == label].mean())} for label in np.unique(labels)}
            arms[arm] = {'groups': groups, 'strata': strata,
                         'endpoint_validation_unfiltered': old_report['endpoint_validation'],
                         'endpoint_validation_filtered': new_report['endpoint_validation'],
                         'filtered_best_epoch': new_report['best_epoch'],
                         'combined_mean_before': old_report['training_weighting']['combined_weight_sample_mean_before_normalization'],
                         'combined_mean_after': new_report['training_weighting']['combined_weight_sample_mean_before_normalization']}
    result = {'scope': plan['limitations'], 'plan': plan, 'arms': arms,
              'validation_unchanged': True, 'test_evaluated': False, 'promoted': False,
              'checkpoint_sha256': {arm: sha(run / arm / 'training-state.pt') for arm in ARMS},
              'filtered_predictions_sha256': sha(run / 'paired-predictions.npz')}
    write(run / 'filter-comparison.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-run', type=Path, default=DEFAULT_CONTROL)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run, control = args.run_dir.resolve(), args.control_run.resolve()
    if run == control:
        raise ValueError('Use a separate output directory')
    run.mkdir(parents=True, exist_ok=True)
    with (run / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(run / 'plan.json') if (run / 'plan.json').exists() else prepare(run, control)
        if plan['control_run'] != str(control):
            raise ValueError('Control run differs from the original plan')
        verify(run, plan)
        print(json.dumps({'stage': 'prepared', 'samples': plan['samples'], 'removed': plan['removed'],
                          'validation': plan['validation_count']}), flush=True)
        env = {**os.environ, 'PYTHONPATH': str(ROOT / 'training/.deps'), 'SUPER_READER_PROJECT_ROOT': str(ROOT), 'DEBUG': '0'}
        for arm in ARMS:
            (run / arm).mkdir(exist_ok=True)
            while not (run / arm / 'validation-only.json').exists():
                write(run / 'status.json', {'stage': 'training', 'arm': arm, 'updated_at': time.time()})
                with (run / arm / 'training.log').open('a') as log:
                    completed = subprocess.run(command(run, arm, plan['seed']), env=env, stdout=log, stderr=subprocess.STDOUT)
                if completed.returncode == 75:
                    continue
                if completed.returncode or not (run / arm / 'validation-only.json').exists():
                    raise RuntimeError(arm + ' stopped; inspect training.log and rerun to resume')
            print(json.dumps({'stage': 'trained', 'arm': arm}), flush=True)
        write(run / 'status.json', {'stage': 'paired-validation', 'updated_at': time.time()})
        if not (run / 'comparison.json').exists():
            compare(run, plan)
        result = compare_to_control(run, plan)
        write(run / 'status.json', {'stage': 'complete', 'updated_at': time.time()})
        print(json.dumps({arm: report['groups'] for arm, report in result['arms'].items()}), flush=True)


if __name__ == '__main__':
    main()
