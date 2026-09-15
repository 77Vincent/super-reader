#!/usr/bin/env python3
"""Audit abstention thresholds on the complete validation split of a frozen release."""
import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'training/.deps'))
import numpy as np
import torch
from text_policy import require_data_policy, valid_proxy_label
from train_smoke import BoundaryChooser, configure_cpu, iterate_batches


def digest(path):
    with path.open('rb') as handle:
        checksum = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(block)
        return checksum.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def metrics(confidence, correct, thresholds):
    count = len(confidence)
    rows = []
    for threshold in thresholds:
        accepted = confidence > threshold
        cuts = int(accepted.sum())
        true_positives = int((accepted & correct).sum())
        rows.append({
            'threshold': threshold, 'samples': count, 'cuts': cuts,
            'correct_cuts': true_positives, 'wrong_cuts': cuts - true_positives,
            'precision': true_positives / cuts if cuts else None,
            'recall': true_positives / count if count else None,
            'coverage': cuts / count if count else None,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, default=ROOT / 'training/artifacts/unicode-context-192ch-16conv-20260914/epoch-1-backend')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'training/data/processed/unicode-context-192ch-12conv-20260913-eval')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'training/artifacts/confidence-gate-16conv-epoch1-20260915')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    configure_cpu(args.threads, 1)
    started = time.monotonic()
    release = args.artifact_dir
    metadata = json.loads((release / 'smoke-metrics.json').read_text())
    require_data_policy(metadata)
    expected_accuracy = metadata['validation']['accuracy']
    expected_macro_accuracy = metadata['validation']['macro_accuracy']
    expected_count = metadata['validation']['count']
    bundle_sha = digest(ROOT / 'src/boundary-model-data.js')
    assert bundle_sha == digest(release / 'boundary-model-data.js'), 'Frozen release differs from bundled model'
    verified = json.loads((release / 'backend-verification.json').read_text())
    assert verified['passed'] and verified['bundleSha256'] == bundle_sha
    assert verified['model']['checkpointSha256'] == digest(release / 'boundary-smoke.safetensors')
    state = torch.load(release / 'selected-state.pt', map_location='cpu', weights_only=True)
    assert state['best_epoch'] == metadata['best_epoch']
    vocabulary = state['vocabulary']
    architecture = metadata['architecture']
    model = BoundaryChooser(len(vocabulary), architecture['channels'], architecture['residual_blocks'])
    model.load_state_dict(state['best_state'])
    model.eval()
    del state

    # Validate provenance and every row before any metric is accepted.
    summary = json.loads((args.data_dir / 'summary.json').read_text())
    require_data_policy(summary)
    input_path = args.data_dir / 'validation.jsonl'
    input_sha = digest(input_path)
    assert input_sha == summary['splits']['validation']['sha256']
    records = []
    domains = {}
    with input_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            assert record['tokenization'] == 'character' and valid_proxy_label(record['punctuation'])
            tokens = record['tokens']
            assert all(isinstance(t, str) and len(t) == 1 for t in tokens)
            assert 0 <= record['target_index'] < len(tokens) - 1
            domain = record['domain']
            domains.setdefault(domain, len(domains))
            records.append({'token_ids': [vocabulary.get(t, vocabulary['<unk>']) for t in tokens],
                            'target_index': record['target_index'], 'domain': domain, 'training_weight': 1.0})
    assert len(records) == expected_count
    print(json.dumps({'stage': 'validation', 'samples': len(records), 'threads': args.threads}), flush=True)

    columns = defaultdict(list)
    seen = 0
    next_notice = 25000
    with torch.inference_mode():
        for batch in iterate_batches(records, 512, 8192, shuffle=False, seed=0):
            logits = model(batch['token_ids'], batch['token_mask'], batch['gap_mask'])
            # Double precision matches JS's stable softmax of float32 logits.
            confidence, predicted = torch.softmax(logits.double(), dim=1).max(dim=1)
            columns['confidence'].append(confidence.numpy())
            columns['correct'].append((predicted == batch['targets']).numpy())
            columns['domain'].append(np.array([domains[r['domain']] for r in batch['records']], dtype=np.int16))
            columns['tokens'].append(np.array([len(r['token_ids']) for r in batch['records']], dtype=np.int32))
            seen += len(batch['records'])
            if seen >= next_notice:
                status = {'stage': 'validation', 'scored': seen, 'total': len(records),
                          'elapsed_seconds': time.monotonic() - started}
                print(json.dumps(status), flush=True)
                write(args.output_dir / 'status.json', status)
                next_notice += 25000
    arrays = {key: np.concatenate(values) for key, values in columns.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / 'predictions.npz', **arrays)
    confidence, correct = arrays['confidence'], arrays['correct']
    assert len(confidence) == len(records)
    assert abs(float(correct.mean()) - expected_accuracy) < 1e-12
    thresholds = [0, .5, .7, .8, .85, .9, .95, .98, .99]
    per_domain = {}
    domain_accuracies = []
    for domain, index in domains.items():
        mask = arrays['domain'] == index
        domain_accuracies.append(float(correct[mask].mean()))
        per_domain[domain] = metrics(confidence[mask], correct[mask], thresholds)
    assert abs(sum(domain_accuracies) / len(domain_accuracies) - expected_macro_accuracy) < 1e-12
    report = {
        'scope': 'full_validation', 'model': verified['model'], 'bundle_sha256': bundle_sha,
        'selected_state_sha256': digest(release / 'selected-state.pt'), 'input_sha256': input_sha,
        'confidence': 'Unweighted softmax over every original sample gap; accept strictly greater than threshold.',
        'definitions': {'precision': 'correct accepted predictions / accepted predictions',
                        'recall': 'correct accepted predictions / all labeled gaps (one per sample)',
                        'coverage': 'accepted predictions / all samples'},
        'limitations': 'Single hidden-punctuation target per input. No no-cut labels or human full-segmentation targets. '
                      'Does not apply backend length gating, word protection, balance or 256-token windowing; '
                      'these are model confidence metrics, not end-to-end runtime precision.',
        'overall': metrics(confidence, correct, thresholds), 'per_domain': per_domain,
        'over_256_tokens': int((arrays['tokens'] > 256).sum()),
        'domains': domains, 'elapsed_seconds': time.monotonic() - started,
        'audit': {'passed': True, 'exact_full_validation_accuracy_and_macro_match_release': True},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir / 'report.json', report)
    write(args.output_dir / 'status.json', {'stage': 'complete', 'scored': seen, 'elapsed_seconds': report['elapsed_seconds']})
    print(json.dumps({'stage': 'complete', 'overall': report['overall']}), flush=True)


if __name__ == '__main__':
    main()
