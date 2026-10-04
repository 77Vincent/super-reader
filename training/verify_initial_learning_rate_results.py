#!/usr/bin/env python3
"""Independently reproduce paired validation predictions for the extended LR arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def paired_document_interval(first, second, document_ids, seed, draws=2000):
    import numpy as np
    first, second = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    if len(first) != len(second) or len(first) != len(document_ids) or not len(first):
        raise ValueError('Paired predictions and documents must align and be nonempty')
    _, groups = np.unique(document_ids, return_inverse=True)
    differences = first.astype(float) - second.astype(float)
    sums = np.bincount(groups, weights=differences)
    counts = np.bincount(groups)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(draws):
        selected = rng.integers(0, len(sums), len(sums))
        boot.append(sums[selected].sum() / counts[selected].sum())
    return {'samples': len(first), 'documents': len(sums),
        'first_only_correct': int((first & ~second).sum()),
        'second_only_correct': int((~first & second).sum()),
        'both_correct': int((first & second).sum()), 'both_wrong': int((~first & ~second).sum()),
        'difference_pp': float(differences.mean() * 100),
        'paired_document_bootstrap_95_percent_interval_pp': [float(x) * 100 for x in np.percentile(boot, [2.5, 97.5])],
        'bootstrap_draws': draws, 'seed': seed,
        'scope': 'Validation-document sampling uncertainty, conditional on these trained checkpoints; not training-seed uncertainty.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    result = json.loads((run / 'comparison.json').read_text())
    plan = result['plan']
    if result['test_evaluated'] or result['promoted']:
        raise ValueError('Expected a validation-only experiment')
    for name, expected in plan['source_hashes'].items():
        if sha(run / 'source' / name) != expected:
            raise ValueError('Frozen source changed: ' + name)
    for name, expected in plan['data_hashes'].items():
        if sha(run / 'data' / name) != expected:
            raise ValueError('Frozen data changed: ' + name)
    sys.path.insert(0, str(run / 'source'))
    import numpy as np
    import torch
    from train_smoke import BoundaryChooser, configure_cpu, configure_mps, iterate_batches, batch_to_device
    from train_sharded import read_evaluation_records
    configure_cpu(1, 1)
    device = configure_mps()
    vocabulary = json.loads((run / 'data/vocabulary.json').read_text())
    records = read_evaluation_records(run / 'data/validation.jsonl', vocabulary, 2)
    indices = {row['id']: i for i, row in enumerate(records)}
    if len(indices) != len(records) or len(records) != plan['validation_samples']:
        raise ValueError('Validation population mismatch')
    targets = np.array([row['target_index'] for row in records], dtype=np.int32)
    predictions, verified = {}, {}
    for arm, report in result['extended'].items():
        checkpoint = run / arm / 'training-state.pt'
        if sha(checkpoint) != report['verified_checkpoint']['checkpoint_sha256']:
            raise ValueError('Endpoint checkpoint changed: ' + arm)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model = BoundaryChooser(len(vocabulary), plan['training']['channels'], plan['training']['residual_blocks'])
        model.load_state_dict(state['model_state'])
        model.to(device).eval()
        values = np.full(len(records), -1, dtype=np.int32)
        with torch.inference_mode():
            for batch in iterate_batches(records, 512, 8192, shuffle=False, seed=0):
                batch = batch_to_device(batch, device)
                guessed = model(batch['token_ids'], batch['token_mask'], batch['gap_mask']).argmax(1).cpu().numpy()
                for row, value in zip(batch['records'], guessed):
                    values[indices[row['id']]] = value
        if np.any(values < 0):
            raise ValueError('Unscored validation rows')
        accuracy = float((values == targets).mean())
        if abs(accuracy - report['endpoint_validation']['accuracy']) > 1e-12:
            raise ValueError('Independent endpoint accuracy differs: ' + arm)
        predictions[arm] = values
        verified[arm] = {'samples': len(records), 'correct': int((values == targets).sum()),
                         'accuracy': accuracy, 'checkpoint_sha256': sha(checkpoint)}
        del model, state
        torch.mps.empty_cache()
        print(json.dumps({'stage': 'predictions-verified', 'arm': arm, **verified[arm]}), flush=True)
    first = result['winner']
    second = next(arm for arm in predictions if arm != first)
    documents = np.array([row.get('document_id', row['id']) for row in records])
    interval = paired_document_interval(predictions[first] == targets, predictions[second] == targets,
                                        documents, plan['seed'] + 3)
    np.savez_compressed(run / 'paired-validation-predictions.npz', targets=targets, **predictions)
    audit = {'assessment': 'Share with caveats', 'verified': verified, 'first': first, 'second': second,
        'paired_difference': interval, 'rows_without_document_id': sum('document_id' not in row for row in records),
        'validation_sha256': sha(run / 'data/validation.jsonl'),
        'predictions_sha256': sha(run / 'paired-validation-predictions.npz'),
        'test_evaluated': False, 'limitations': plan['limitations']}
    (run / 'validation-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    report_path = run / 'report.md'
    text = report_path.read_text().split('\n## 配对复核\n')[0]
    lower, upper = interval['paired_document_bootstrap_95_percent_interval_pp']
    text += ('\n## 配对复核\n\n'
        f"独立重算两个延长训练端点的 {len(records):,} 条预测，准确率与训练报告完全一致。\n\n"
        f"{plan['rates'][first]:g} 相对 {plan['rates'][second]:g} 的准确率差为 "
        f"{interval['difference_pp']:.4f} 个百分点；按文档聚类重采样的 95% 区间为 "
        f"[{lower:.4f}, {upper:.4f}] 个百分点。\n\n"
        '该区间只反映这两个固定模型的验证文档抽样不确定性，不涵盖训练随机种子变化。\n'
        '完整计数及复核记录见 validation-audit.json，逐样本预测见 paired-validation-predictions.npz。\n')
    report_path.write_text(text)
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == '__main__':
    main()
