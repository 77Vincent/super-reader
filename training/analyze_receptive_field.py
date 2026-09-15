#!/usr/bin/env python3
"""Full-corpus receptive-field coverage and full-validation error stratification."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / 'training/artifacts/unicode-context-192ch-12conv-20260913'
MANIFEST = ROOT / 'training/data/processed/unicode-context-192ch-12conv-20260913-combined/manifest.json'
EVAL = ROOT / 'training/data/processed/unicode-context-192ch-12conv-20260913-eval'
OUTPUT = ROOT / 'training/artifacts/receptive-field-coverage-20260914'
DEPTHS = (12, 16, 20, 24, 32, 48, 64)


def read(path):
    return json.loads(Path(path).read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def full_gap_count(n, radius):
    return max(0, min(radius, n - 1) - max(1, n - radius) + 1)


def visible_gap_sum(n, radius):
    k = min(n - 1, radius)
    return 2 * (k * (k + 1) // 2 + (n - 1 - k) * radius)


def self_check():
    # Independently propagate token dependency sets through kernel-3 layers.
    for layers in (2, 4, 12):
        radius = layers + 1
        for n in range(2, 36):
            dependencies = [{i} for i in range(n)]
            for _ in range(layers):
                dependencies = [set().union(*dependencies[max(0, i - 1):i + 2]) for i in range(n)]
            gaps = [dependencies[i] | dependencies[i + 1] for i in range(n - 1)]
            assert sum(len(g) == n for g in gaps) == full_gap_count(n, radius)
            assert sum(map(len, gaps)) == visible_gap_sum(n, radius)
            for b, gap in enumerate(gaps, 1):
                assert len(gap) == min(b, radius) + min(n - b, radius)
                assert (len(gap) == n) == (max(b, n - b) <= radius)
    assert min(1, 13) + min(25, 13) == 14
    assert full_gap_count(26, 13) == 1 and full_gap_count(27, 13) == 0


def empty_histograms():
    return {name: {} for name in ('length', 'far', 'near')}


def add(histogram, key, weight, inverse_length):
    values = histogram.get(key)
    if values is None:
        values = histogram[key] = [0, 0.0, 0.0, 0.0]
    values[0] += 1
    values[1] += weight
    values[2] += inverse_length
    values[3] += weight * inverse_length


def scan_shard(job):
    item, position_weights, domain_weights, domain_count = job
    path = ROOT / item['path']
    raw = path.read_bytes()
    assert len(raw) == item['bytes'], str(path)
    domains = [empty_histograms() for _ in range(domain_count)]
    cells = [[0] * 10 for _ in range(4)]
    count = tokens = 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        text, target, domain, bucket, position = json.loads(line)
        n = len(text)
        assert isinstance(text, str) and 0 <= target < n - 1 and 0 <= domain < domain_count
        b = target + 1
        assert bucket == (0 if n <= 8 else 1 if n <= 16 else 2 if n <= 32 else 3)
        assert position == min(9, int(b / n * 10))
        cells[bucket][position] += 1
        weight = position_weights[bucket][position] * domain_weights[domain]
        inverse = 1 / n
        hist = domains[domain]
        add(hist['length'], n, weight, inverse)
        add(hist['far'], max(b, n - b), weight, inverse)
        add(hist['near'], min(b, n - b), weight, inverse)
        count += 1
        tokens += n
    return {'path': item['path'], 'sha256': sha(raw), 'samples': count, 'tokens': tokens,
            'domains': domains, 'cells': cells}


def merge(target, source):
    for name, histogram in source.items():
        for key, values in histogram.items():
            row = target[name].setdefault(key, [0, 0.0, 0.0, 0.0])
            for index, value in enumerate(values):
                row[index] += value


def summarize(hist):
    lengths, far, near = (hist[name] for name in ('length', 'far', 'near'))
    count = sum(v[0] for v in lengths.values())
    weight = sum(v[1] for v in lengths.values())
    tokens = sum(n * v[0] for n, v in lengths.items())
    gaps = tokens - count
    def quantile(p):
        required, cumulative = math.ceil(count * p), 0
        for n, v in sorted(lengths.items()):
            cumulative += v[0]
            if cumulative >= required:
                return n
    depths = []
    for layers in DEPTHS:
        radius = layers + 1
        short = sum(v[0] for n, v in lengths.items() if n <= 2 * radius)
        covered = sum(v[0] for n, v in far.items() if n <= radius)
        weighted = sum(v[1] for n, v in far.items() if n <= radius)
        visible = sum(min(n, radius) * v[0] for h in (far, near) for n, v in h.items())
        fraction = sum(min(n, radius) * v[2] for h in (far, near) for n, v in h.items())
        weighted_fraction = sum(min(n, radius) * v[3] for h in (far, near) for n, v in h.items())
        depths.append({
            'convolution_layers': layers, 'tokens_per_side': radius, 'maximum_gap_context': 2 * radius,
            'parameters_at_192_channels_8192_vocabulary': 8192 * 192 + (layers // 2) * 221953 + 147841,
            'length_within_maximum_context_count': short, 'length_within_maximum_context_rate': short / count,
            'gold_gap_full_context_count': covered, 'gold_gap_full_context_rate': covered / count,
            'training_weighted_gold_gap_full_context_rate': weighted / weight,
            'mean_gold_gap_visible_tokens': visible / count,
            'mean_gold_gap_visible_fraction': fraction / count,
            'training_weighted_mean_gold_gap_visible_fraction': weighted_fraction / weight,
            'all_candidate_gaps_full_context_rate': sum(full_gap_count(n, radius) * v[0] for n, v in lengths.items()) / gaps,
            'mean_visible_tokens_over_all_candidate_gaps': sum(visible_gap_sum(n, radius) * v[0] for n, v in lengths.items()) / gaps,
            'samples_with_every_gap_fully_covered_rate': sum(v[0] for n, v in lengths.items() if n <= radius + 1) / count,
        })
    return {'samples': count, 'tokens': tokens, 'candidate_gaps': gaps, 'weight_sum': weight,
            'mean_length': tokens / count, 'maximum_length': max(lengths),
            'length_quantiles': {str(p): quantile(p) for p in (.5, .75, .9, .95, .99, .999)},
            'depths': depths}


def train_coverage(workers):
    started = time.perf_counter()
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    metrics = read(RUN / 'candidate/smoke-metrics.json')
    assert manifest['input_representation'] == 'unicode-context-v1'
    cells = manifest['statistics']['cells']
    position_weights = [[sum(row) / sum(c > 0 for c in row) / c if c else 0 for c in row] for row in cells]
    weights = [metrics['training_weighting']['domain_weights'][d] for d in manifest['domains']]
    merged = [empty_histograms() for _ in weights]
    observed_cells = [[0] * 10 for _ in range(4)]
    sources = []
    count = 0
    jobs = ((item, position_weights, weights, len(weights)) for item in manifest['shards'])
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(scan_shard, jobs), 1):
            for target, source in zip(merged, result.pop('domains')):
                merge(target, source)
            for target, source in zip(observed_cells, result.pop('cells')):
                for cell, value in enumerate(source):
                    target[cell] += value
            sources.append(result)
            count += result['samples']
            if index % 16 == 0 or index == len(manifest['shards']):
                status = {'stage': 'scanning-training', 'shards': index, 'total_shards': len(manifest['shards']),
                          'samples': count, 'seconds': time.perf_counter() - started}
                write(OUTPUT / 'training-status.json', status)
                print(json.dumps(status), flush=True)
    overall = empty_histograms()
    for hist in merged:
        merge(overall, hist)
    report = {'manifest_sha256': sha(manifest_bytes), 'scope': 'all_training_shards_once', 'shards': sources,
              'overall': summarize(overall),
              'per_domain': {name: summarize(hist) for name, hist in zip(manifest['domains'], merged)}}
    assert report['overall']['samples'] == manifest['statistics']['samples']
    assert report['overall']['tokens'] == manifest['statistics']['tokens']
    assert report['overall']['maximum_length'] == manifest['statistics']['maximum_sequence_length']
    assert observed_cells == cells
    assert {d: v['samples'] for d, v in report['per_domain'].items()} == manifest['statistics']['domain_samples']
    observed_mean = report['overall']['weight_sum'] / count
    assert abs(observed_mean - metrics['training_weighting']['combined_weight_sample_mean_before_normalization']) < 1e-8
    assert sha(MANIFEST.read_bytes()) == report['manifest_sha256']
    report['audit'] = {'passed': True, 'counts_tokens_domains_cells_and_weight_normalizer_match_manifest': True,
                       'formula_checks': 'exact match with propagated convolution dependency sets'}
    report['elapsed_seconds'] = time.perf_counter() - started
    write(OUTPUT / 'training-coverage.json', report)
    write(OUTPUT / 'training-histograms.json', {'columns': ['count', 'weight', 'inverse_length', 'weighted_inverse_length'],
                                               'domains': dict(zip(manifest['domains'], merged))})
    write(OUTPUT / 'training-status.json', {'stage': 'complete', 'samples': count, 'seconds': report['elapsed_seconds']})
    print(json.dumps({'stage': 'training-complete', 'summary': report['overall']}), flush=True)


def validation_coverage(threads):
    started = time.perf_counter()
    sys.path.insert(0, str(ROOT / 'training/.deps'))
    import torch
    from train_smoke import BoundaryChooser, configure_cpu, iterate_batches
    from text_policy import require_data_policy, valid_proxy_label
    configure_cpu(threads, 1)
    with (RUN / 'candidate/training-state.pt').open('rb') as handle:
        checkpoint = torch.load(handle, map_location='cpu', weights_only=True)
    assert checkpoint['best_epoch'] == 2
    vocabulary = checkpoint['vocabulary']
    model = BoundaryChooser(len(vocabulary), 192, 6)
    model.load_state_dict(checkpoint['best_state'])
    model.eval()
    del checkpoint
    summary = read(EVAL / 'summary.json')
    require_data_policy(summary)
    records = []
    input_hash = hashlib.sha256()
    with (EVAL / 'validation.jsonl').open('rb') as handle:
        for line in handle:
            input_hash.update(line)
            if not line.strip():
                continue
            r = json.loads(line)
            assert valid_proxy_label(r['punctuation'])
            tokens = r['tokens']
            assert all(len(token) == 1 for token in tokens)
            assert 0 <= r['target_index'] < len(tokens) - 1
            records.append({'token_ids': [vocabulary.get(t, vocabulary['<unk>']) for t in tokens],
                            'target_index': r['target_index'], 'domain': r['domain'], 'training_weight': 1.0})
    assert input_hash.hexdigest() == summary['splits']['validation']['sha256']
    assert len(records) == 727578
    metrics = read(RUN / 'candidate/smoke-metrics.json')
    groups = {key: defaultdict(lambda: [0, 0, 0]) for key in ('length', 'needed_layers', 'domain', 'domain_coverage', 'coverage')}
    total = correct = 0
    last_notice = 0
    def accumulate(kind, key, success, error):
        row = groups[kind][key]
        row[0] += 1
        row[1] += success
        row[2] += error
    with torch.inference_mode():
        for batch in iterate_batches(records, 512, 8192, shuffle=False, seed=0):
            logits = model(batch['token_ids'], batch['token_mask'], batch['gap_mask'])
            predictions = logits.argmax(dim=1).tolist()
            for r, predicted in zip(batch['records'], predictions):
                n = len(r['token_ids'])
                b = r['target_index'] + 1
                far = max(b, n - b)
                success = int(predicted == r['target_index'])
                error = abs(predicted - r['target_index'])
                covered = 'full' if far <= 13 else 'partial'
                length_group = next((str(edge) for edge in (14, 26, 34, 42, 50, 66, 128, 256) if n <= edge), '257+')
                needed = next((str(depth) for depth in DEPTHS if far <= depth + 1), 'over64')
                for kind, key in [('length', length_group), ('needed_layers', needed), ('domain', r['domain']),
                                  ('domain_coverage', r['domain'] + ':' + covered), ('coverage', covered)]:
                    accumulate(kind, key, success, error)
                total += 1
                correct += success
            elapsed = time.perf_counter() - started
            if elapsed - last_notice >= 30:
                status = {'stage': 'validating', 'samples': total, 'total': len(records), 'seconds': elapsed}
                write(OUTPUT / 'validation-status.json', status)
                print(json.dumps(status), flush=True)
                last_notice = elapsed
    result = {kind: {key: {'samples': v[0], 'correct': v[1], 'accuracy': v[1] / v[0],
                           'mean_absolute_gap_error': v[2] / v[0], 'errors': v[0] - v[1]}
                     for key, v in values.items()} for kind, values in groups.items()}
    assert total == len(records)
    assert abs(correct / total - metrics['validation']['accuracy']) < 1e-12
    for domain, value in result['domain'].items():
        assert abs(value['accuracy'] - metrics['validation']['per_domain_accuracy'][domain]) < 1e-12
    report = {'scope': 'full_validation', 'best_epoch': 2, 'input_sha256': input_hash.hexdigest(),
              'samples': total, 'correct': correct, 'accuracy': correct / total, 'groups': result,
              'audit': {'passed': True, 'exact_match_with_completed_training_validation_accuracy': True},
              'elapsed_seconds': time.perf_counter() - started}
    write(OUTPUT / 'validation-coverage.json', report)
    write(OUTPUT / 'validation-status.json', {'stage': 'complete', 'samples': total, 'seconds': report['elapsed_seconds']})
    print(json.dumps({'stage': 'validation-complete', 'accuracy': report['accuracy'], 'coverage': result['coverage']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['training', 'validation', 'self-check'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    os.nice(10)
    self_check()
    if args.mode == 'training':
        train_coverage(args.workers)
    elif args.mode == 'validation':
        validation_coverage(args.threads)
    else:
        print('Receptive-field formulas verified against explicit convolution dependencies.')
