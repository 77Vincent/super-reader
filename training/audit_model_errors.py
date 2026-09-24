#!/usr/bin/env python3
"""Measure full-split boundary errors on Metal and reservoir-sample disagreements."""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ['DEBUG'] = '0'
sys.path.insert(0, str(ROOT / 'training/.deps'))
import numpy as np
import torch
from train_smoke import BoundaryChooser, configure_cpu, configure_mps, bucket_ceiling, iterate_batches, batch_to_device
from text_policy import require_data_policy, valid_proxy_label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'test'), default='test')
    parser.add_argument('--seed', type=int, default=20260924)
    parser.add_argument('--review-count', type=int, default=120)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_cpu(1, 1)
    device = configure_mps()
    metrics = json.loads((args.artifact_dir / 'smoke-metrics.json').read_text())
    summary = json.loads((args.data_dir / 'summary.json').read_text())
    require_data_policy(metrics)
    require_data_policy(summary)
    state = torch.load(args.artifact_dir / 'training-state.pt', map_location='cpu', weights_only=True)
    vocabulary = state['vocabulary']
    architecture = metrics['architecture']
    model = BoundaryChooser(len(vocabulary), architecture['channels'], architecture['residual_blocks']).to(device)
    model.load_state_dict(state['best_state'], strict=True)
    model.eval()
    del state
    unknown = vocabulary['<unk>']
    radius = architecture['residual_blocks'] * architecture['convolutions_per_block'] + 1
    stats = defaultdict(lambda: defaultdict(Counter))
    reservoirs, seen, randomizers = {}, Counter(), {}
    total = Counter()
    started = time.monotonic()
    next_notice = 250000

    def sample(key, record, limit):
        if key not in reservoirs:
            reservoirs[key] = []
            randomizers[key] = random.Random(f'{args.seed}:{key}')
        seen[key] += 1
        rows = reservoirs[key]
        if len(rows) < limit:
            rows.append(record)
        else:
            index = randomizers[key].randrange(seen[key])
            if index < limit:
                rows[index] = record

    def score(records):
        nonlocal next_notice
        for batch in iterate_batches(records, 512, 8192, shuffle=False, seed=0):
            batch = batch_to_device(batch, device)
            logits = model(batch['token_ids'], batch['token_mask'], batch['gap_mask']).cpu().numpy()
            predictions = logits.argmax(axis=1)
            for row, (record, predicted) in enumerate(zip(batch['records'], predictions)):
                tokens = record['tokens']
                length, gold, predicted = len(tokens), record['target_index'], int(predicted)
                scores = logits[row, :length - 1].astype(np.float64)
                probabilities = np.exp(scores - scores.max())
                probabilities /= probabilities.sum()
                confidence = float(probabilities[predicted])
                rank = int(1 + (scores > scores[gold]).sum())
                wrong = predicted != gold
                distance = abs(predicted - gold)
                total['count'] += 1
                total['errors'] += wrong
                total['gold_top3'] += rank <= 3
                total['error_gold_top3'] += wrong and rank <= 3
                total['absolute_gap_error_sum'] += distance
                length_bin = next((f'<= {n}' for n in (8, 16, 32, 64, 128) if length <= n), '> 128')
                confidence_bin = next((f'< {n}' for n in (.5, .75, .9, .99) if confidence < n), '>= 0.99')
                distance_bin = next((f'<= {n}' for n in (0, 1, 2, 5, 10) if distance <= n), '> 10')
                groups = {
                    'domain': record['domain'], 'punctuation': record['punctuation'],
                    'length': length_bin, 'confidence': confidence_bin, 'gap_error': distance_bin,
                    'contains_unknown': str(unknown in record['token_ids']),
                    'gold_full_receptive_coverage': str(gold + 1 <= radius and length - gold - 1 <= radius),
                    'shorter_gold_side': str(min(gold + 1, length - gold - 1)) if min(gold + 1, length - gold - 1) <= 3 else '> 3',
                }
                for axis, key in groups.items():
                    stats[axis][key]['count'] += 1
                    stats[axis][key]['errors'] += wrong
                if wrong:
                    review = {
                        'id': record['id'], 'domain': record['domain'],
                        'text': ''.join(tokens), 'punctuation': record['punctuation'],
                        'gold': gold, 'predicted': predicted,
                        'gold_split': ''.join(tokens[:gold + 1]) + '｜' + ''.join(tokens[gold + 1:]),
                        'predicted_split': ''.join(tokens[:predicted + 1]) + '｜' + ''.join(tokens[predicted + 1:]),
                        'confidence': confidence, 'gold_probability': float(probabilities[gold]),
                        'gold_rank': rank, 'gap_error': distance,
                    }
                    sample('uniform_errors', review, args.review_count)
                    sample('domain:' + record['domain'], review, 8)
                    if confidence >= .9:
                        sample('high_confidence_errors', review, 20)
        if total['count'] >= next_notice:
            progress = {'scored': total['count'], 'errors': total['errors'], 'elapsed_seconds': round(time.monotonic() - started, 1)}
            print(json.dumps(progress), flush=True)
            (args.output_dir / 'progress.json').write_text(json.dumps(progress))
            next_notice += 250000

    # Preserve the official evaluator's within-bucket order and batch composition
    # while retaining only one pending batch per length bucket in host memory.
    buffers = defaultdict(list)
    checksum = hashlib.sha256()
    path = args.data_dir / f'{args.split}.jsonl'
    with torch.inference_mode(), path.open('rb') as handle:
        for line in handle:
            checksum.update(line)
            if not line.strip():
                continue
            record = json.loads(line)
            tokens = record['tokens']
            assert record['tokenization'] == 'character'
            assert all(isinstance(t, str) and len(t) == 1 for t in tokens)
            assert 0 <= record['target_index'] < len(tokens) - 1
            assert valid_proxy_label(record['punctuation'])
            record['token_ids'] = [vocabulary.get(t, unknown) for t in tokens]
            record['training_weight'] = 1.0
            ceiling = bucket_ceiling(len(tokens))
            buffers[ceiling].append(record)
            if len(buffers[ceiling]) >= max(1, min(512, 8192 // ceiling)):
                score(buffers.pop(ceiling))
        for ceiling in sorted(buffers):
            score(buffers[ceiling])
    assert checksum.hexdigest() == summary['splits'][args.split]['sha256']
    assert total['count'] == metrics['data_sizes'][args.split]
    expected_errors = round((1 - metrics[args.split]['accuracy']) * total['count'])
    # Different ordering of length buckets can alter a few near-tied FP32 gaps.
    assert abs(total['errors'] - expected_errors) <= 5, (total['errors'], expected_errors)
    report = {
        'scope': f'full_{args.split}', 'artifact_dir': str(args.artifact_dir),
        'checkpoint_sha256': hashlib.sha256((args.artifact_dir / 'boundary-smoke.safetensors').read_bytes()).hexdigest(),
        'input_sha256': checksum.hexdigest(), 'seed': args.seed, 'device': str(device),
        'total': total, 'expected_errors': expected_errors, 'buckets': stats,
        'elapsed_seconds': time.monotonic() - started,
        'review_sampling': 'Independent seeded reservoirs, uniform among errors; domain and high-confidence samples are supplemental, not weighted into the uniform review.',
        'limitations': 'Raw first-gap prediction against one punctuation-derived target; no backend rules or recursive cuts. Semantic judgments require review, not these automatic buckets.',
    }
    (args.output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (args.output_dir / 'review.json').write_text(json.dumps(reservoirs, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'complete': True, 'count': total['count'], 'errors': total['errors'], 'expected_errors': expected_errors}), flush=True)


if __name__ == '__main__':
    main()
