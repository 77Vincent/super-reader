#!/usr/bin/env python3
"""Compare text-blind baselines with the existing paired smoke endpoints.

Fits the length prior on training rows only; reuses saved validation predictions.
Does not train, score test data, or select/promote model weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'training/.deps'))
import numpy as np

from run_position_ablation import read, sha, write
from train_smoke import build_length_prior, center_baseline, length_prior_baseline

ARMS = ('inverse-cell', 'none')


def check_predictions(lengths, targets, predictions):
    if len(lengths) == 0 or targets.shape != lengths.shape or np.any(lengths < 2):
        raise ValueError('Invalid or empty validation population')
    for name, values in {'targets': targets, **predictions}.items():
        if (values.shape != lengths.shape or not np.issubdtype(values.dtype, np.integer)
                or np.any(values < 0) or np.any(values >= lengths - 1)):
            raise ValueError('Invalid gap predictions: ' + name)


def summarize(lengths, targets, predictions, domains):
    check_predictions(lengths, targets, predictions)
    correct = {name: values == targets for name, values in predictions.items()}
    positions = np.minimum(9, (targets + 1) * 10 // lengths)
    length_bins = np.array([next(end for end in (8, 16, 32, 64, 128, 256) if n <= end)
                            for n in lengths])

    def group(mask):
        count = int(mask.sum())
        return {'count': count, 'correct': {k: int(v[mask].sum()) for k, v in correct.items()},
                'accuracy': {k: float(v[mask].mean()) if count else None for k, v in correct.items()}}

    strata = {}
    for dimension, labels in [('position', positions), ('length', length_bins), ('domain', domains)]:
        strata[dimension] = {str(label): group(labels == label) for label in np.unique(labels)}
        assert sum(row['count'] for row in strata[dimension].values()) == len(targets)
        for name in predictions:
            assert sum(row['correct'][name] for row in strata[dimension].values()) == int(correct[name].sum())

    gained = correct['none'] & ~correct['inverse-cell']
    lost = correct['inverse-cell'] & ~correct['none']

    def changes(mask):
        g, l = int((gained & mask).sum()), int((lost & mask).sum())
        count = int(mask.sum())
        return {'count': count, 'none_only_correct': g, 'weighted_only_correct': l,
                'net_correct': g - l,
                'within_group_difference_pp': (g - l) / count * 100 if count else None,
                'contribution_to_total_difference_pp': (g - l) / len(targets) * 100}

    decomposition = {}
    for baseline in ('center', 'length-prior'):
        decomposition[baseline] = {
            'baseline_correct': changes(correct[baseline]),
            'baseline_wrong': changes(~correct[baseline]),
        }
        assert sum(g['net_correct'] for g in decomposition[baseline].values()) == int(gained.sum() - lost.sum())

    # Integer distance from the true geometric midpoint avoids float tie errors.
    distance_change = (np.abs(2 * (predictions['none'] + 1) - lengths)
                       - np.abs(2 * (predictions['inverse-cell'] + 1) - lengths))
    direction = {name: changes(mask) for name, mask in (
        ('closer_to_center', distance_change < 0),
        ('farther_from_center', distance_change > 0),
        ('equal_distance', distance_change == 0))}
    profiles = {}
    for name, values in {'gold': targets, **predictions}.items():
        bins = np.minimum(9, (values + 1) * 10 // lengths)
        profiles[name] = {'position_bin_counts': np.bincount(bins, minlength=10).tolist(),
                         'mean_relative_distance_to_center': float(np.abs((values + 1) / lengths - .5).mean())}

    return {'overall': group(np.ones(len(targets), dtype=bool)),
            'random_gap_expected_accuracy': float((1 / (lengths - 1)).mean()),
            'strata': strata,
            'equal_gold_position_bin_accuracy': {
                name: float(np.mean([g['accuracy'][name] for g in strata['position'].values()]))
                for name in predictions},
            'gain_decomposition': decomposition, 'prediction_movement': direction,
            'prediction_profiles': profiles}


def evaluate(run):
    import json

    run = Path(run).resolve()
    plan, manifest = read(run / 'plan.json'), read(run / 'data/manifest.json')
    for name in ('manifest.json', 'validation.jsonl'):
        if sha(run / 'data' / name) != plan['data_hashes'][name]:
            raise ValueError('Frozen data changed: ' + name)
    length_counts = Counter()

    def training_rows():
        for shard in manifest['shards']:
            path = Path(shard['path'])
            path = path if path.is_absolute() else ROOT / path
            if sha(path) != shard['sha256']:
                raise ValueError('Training shard changed: ' + str(path))
            with path.open() as handle:
                for line in handle:
                    text, target, *_ = json.loads(line)
                    if not 0 <= target < len(text) - 1:
                        raise ValueError('Invalid training target')
                    length_counts[len(text)] += 1
                    # No token identities or domain/position weights reach the prior.
                    yield {'tokens': text, 'target_index': target}

    prior = build_length_prior(training_rows())
    if sum(length_counts.values()) != plan['samples']:
        raise ValueError('Training sample count mismatch')
    with (run / 'data/validation.jsonl').open() as handle:
        records = [json.loads(line) for line in handle]
    if len(records) != plan['validation_count'] or len({r['id'] for r in records}) != len(records):
        raise ValueError('Validation count or unique IDs mismatch')
    lengths = np.array([len(r['tokens']) for r in records])
    targets = np.array([r['target_index'] for r in records])
    domains = np.array([r['domain'] for r in records])
    predictions = {'center': (lengths - 1) // 2,
                   'length-prior': np.array([prior.get(int(n), (int(n) - 1) // 2) for n in lengths])}
    with np.load(run / 'paired-predictions.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['targets'], targets):
            raise ValueError('Saved prediction order does not match validation targets')
        predictions.update({arm: saved[arm].copy() for arm in ARMS})
    result = summarize(lengths, targets, predictions, domains)
    comparison = read(run / 'comparison.json')
    for arm in ARMS:
        expected = comparison['arms'][arm]['endpoint_validation']['accuracy']
        if abs(result['overall']['accuracy'][arm] - expected) > 1e-12:
            raise ValueError('Endpoint accuracy mismatch: ' + arm)
        for label, group in comparison['strata']['position'].items():
            actual = result['strata']['position'][label]
            if actual['count'] != group['count'] or abs(actual['accuracy'][arm] - group[arm]) > 1e-12:
                raise ValueError('Position breakdown mismatch')
    # Reconcile vectorized counts with the existing scalar baseline evaluators.
    for name, expected in [('center', center_baseline(records)),
                           ('length-prior', length_prior_baseline(records, prior))]:
        if result['overall']['accuracy'][name] != expected:
            raise ValueError('Scalar baseline mismatch: ' + name)
    source_files = ['plan.json', 'data/manifest.json', 'data/validation.jsonl',
                    'paired-predictions.npz', 'comparison.json', 'source/training/run_position_ablation.py']
    return {
        'scope': plan['limitations'],
        'method': {
            'training_samples': sum(length_counts.values()), 'validation_samples': len(records),
            'center': 'Predict zero-based gap (length - 1) // 2; for odd character counts choose the right of the two nearest gaps.',
            'length_prior': 'Most frequent exact gap for each exact character length, using raw training counts only. Ties use existing build_length_prior: nearest (length - 1) / 2, then smaller index. Unseen lengths fall back to center.',
            'length_prior_unseen_validation_rows': sum(int(n) not in prior for n in lengths),
            'metric': 'Unweighted exact gap accuracy; one valid label per example.',
            'position_bins': 'floor(10 * (zero-based gap + 1) / character length), capped at 9; gold-defined strata.',
            'equal_gold_position_bin_accuracy': 'Arithmetic mean of accuracy over occupied gold-position bins; an alternative diagnostic weighting, not a human quality measure.',
            'causal_limit': 'Baseline comparisons and gain partitions cannot identify how much of the learned models\' decisions causally comes from position versus text. They do not test wording understanding.',
            'prediction_alignment': 'Saved arrays use frozen comparator validation-file order; targets, total accuracy and position strata reconciled.',
        },
        'sources': {'run_dir': str(run),
                    'sha256': {name: sha(run / name) for name in source_files},
                    'analysis_source_sha256': sha(Path(__file__)),
                    'baseline_helper_source_sha256': sha(ROOT / 'training/train_smoke.py')},
        'length_prior': {str(n): {'training_count': length_counts[n], 'prediction': p}
                         for n, p in sorted(prior.items())},
        **result, 'test_evaluated': False, 'retrained': False, 'promoted': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'training/artifacts/position-ablation-1m-20261004')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = evaluate(args.run_dir)
    output = args.output or args.run_dir / 'position-baselines.json'
    write(output, result)
    print({'output': str(output), 'accuracy': result['overall']['accuracy'],
           'position_macro_accuracy': result['equal_gold_position_bin_accuracy'],
           'gain_decomposition': result['gain_decomposition']})


if __name__ == '__main__':
    main()
