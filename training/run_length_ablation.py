#!/usr/bin/env python3
"""Compare a whole-sample length cap with the saved random-initialized LR control."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys

from sample_subset import keep_target, limit_batch_lengths, write_json

ROOT = Path(os.environ.get('SUPER_READER_PROJECT_ROOT', Path(__file__).resolve().parent.parent))
DEFAULT_CONTROL = ROOT / 'training/artifacts/initial-learning-rate-1m-20261004'
SOURCES = ('train_sharded.py', 'train_smoke.py', 'sample_subset.py', 'text_policy.py',
           'unicode-symbols.json', 'run_length_ablation.py', 'verify_initial_learning_rate_results.py')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def prepare(run, control, maximum):
    from train_smoke import make_batch_indices
    base = read(control / 'plan.json')
    result = read(control / 'comparison.json')
    report = result['extended']['lr-3e-4']
    audit = read(control / 'validation-audit.json')
    if report['configuration']['learning_rate'] != .0003 or len(report['history']) != 3:
        raise ValueError('Expected the three-epoch 0.0003 control')
    if report['test_evaluated'] or report['configuration']['source_weighting'] != 'none':
        raise ValueError('Control used test data or source weights')
    data, source = control / 'data', run / 'source'
    source.mkdir(exist_ok=True)
    for name, digest in base['data_hashes'].items():
        if sha(data / name) != digest:
            raise ValueError('Control input changed: ' + name)
    for name, digest in base['source_hashes'].items():
        if sha(control / 'source' / name) != digest:
            raise ValueError('Control source changed: ' + name)
    # Architecture, optimizer loop primitives and batching must match the saved control.
    if sha(ROOT / 'training/train_smoke.py') != base['source_hashes']['train_smoke.py']:
        raise ValueError('Model or batching changed since control training')
    for name in SOURCES:
        shutil.copy2(ROOT / 'training' / name, source / name)
    shutil.copy2(control / 'source/text-policy.json', source / 'text-policy.json')
    initialization = control / 'initialization.pt'
    checkpoint = control / 'lr-3e-4/training-state.pt'
    if sha(initialization) != base['initialization_sha256']:
        raise ValueError('Control random initialization changed')
    if sha(checkpoint) != report['verified_checkpoint']['checkpoint_sha256']:
        raise ValueError('Control checkpoint changed')
    if sha(control / 'paired-validation-predictions.npz') != audit['predictions_sha256']:
        raise ValueError('Verified control predictions changed')

    manifest = read(data / 'manifest.json')
    counts, schedule, dropped = [], [], []
    all_records = []
    for item in manifest['shards']:
        path = Path(item['path'])
        records, count = [], {'read': 0, 'retained': 0, 'excluded': 0, 'tokens_retained': 0}
        with path.open() as f:
            for line_number, line in enumerate(f):
                text, target = json.loads(line)[:2]
                if not keep_target(len(text), target, 2):
                    raise ValueError('Unexpected singleton in the fixed control subset')
                records.append({'token_ids': [0] * len(text)})
                count['read'] += 1
                if len(text) <= maximum:
                    count['retained'] += 1
                    count['tokens_retained'] += len(text)
                else:
                    count['excluded'] += 1
                    dropped.append({'sample_file': str(path), 'line_zero_based': line_number,
                                    'characters': len(text), 'target_index': target})
        all_records.append(records)
        counts.append(count)
    for epoch in range(1, 4):
        control_steps = capped_steps = seen = 0
        digest = hashlib.sha256()
        order = list(range(len(all_records)))
        random.Random(base['seed'] + epoch).shuffle(order)
        for shard in order:
            records = all_records[shard]
            batches = make_batch_indices(records, 512, 8192, shuffle=True,
                                         seed=base['seed'] + epoch * 1000 + shard)
            filtered = limit_batch_lengths(records, batches, maximum)
            expected = [[i for i in indices if len(records[i]['token_ids']) <= maximum]
                        for indices in batches]
            if filtered != [indices for indices in expected if indices]:
                raise ValueError('Retained batch order changed')
            control_steps += len(batches)
            capped_steps += len(filtered)
            seen += sum(map(len, filtered))
            digest.update(json.dumps([shard, filtered], separators=(',', ':')).encode())
        schedule.append({'epoch': epoch, 'control_updates': control_steps, 'capped_updates': capped_steps,
                         'retained_examples': seen, 'retained_batch_stream_sha256': digest.hexdigest()})
    if sum(x['control_updates'] for x in schedule) != report['verified_checkpoint']['optimizer_steps']:
        raise ValueError('Reconstructed control update budget differs')
    validation_lengths = []
    with (data / 'validation.jsonl').open() as f:
        for line in f:
            validation_lengths.append(len(json.loads(line)['tokens']))
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'control': str(control),
        'data': str(data), 'maximum_sequence_length': maximum, 'minimum_side_characters': 2,
        'epochs': 3, 'learning_rate': .0003, 'seed': base['seed'],
        'initialization': str(initialization), 'initialization_sha256': sha(initialization),
        'control_checkpoint': str(checkpoint), 'control_checkpoint_sha256': sha(checkpoint),
        'control_predictions_sha256': audit['predictions_sha256'],
        'control_report': report, 'data_hashes': base['data_hashes'],
        'source_hashes': {p.name: sha(p) for p in source.iterdir() if p.is_file()},
        'training_counts': {k: sum(x[k] for x in counts) for k in counts[0]}, 'per_shard_counts': counts,
        'validation_counts': {'read': len(validation_lengths),
            'retained': sum(n <= maximum for n in validation_lengths),
            'excluded': sum(n > maximum for n in validation_lengths)},
        'schedule': schedule, 'primary_population': 'same validation rows with length <= maximum for both models',
        'batch_order': 'Build uncapped batches, then remove overlength records and empty batches; no reshuffle',
        'budget': 'Three passes through retained data; omitted long batches reduce optimizer updates, no replacement rows',
        'test_evaluated': False, 'automatic_promotion': False,
        'limitations': 'One initialization seed, reused validation subset and repeated sampled training data; '
                       'bootstrap describes validation-document uncertainty, not training-seed variability.'}
    write_json(run / 'excluded-training-rows.json', dropped)
    write_json(run / 'plan.json', plan)
    return plan


def verify(run, plan):
    for name, digest in plan['source_hashes'].items():
        if sha(run / 'source' / name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    for name, digest in plan['data_hashes'].items():
        if sha(Path(plan['data']) / name) != digest:
            raise ValueError('Frozen input changed: ' + name)
    for path, key in (('initialization', 'initialization_sha256'),
                      ('control_checkpoint', 'control_checkpoint_sha256')):
        if sha(plan[path]) != plan[key]:
            raise ValueError('Reference changed: ' + path)
    if sha(Path(plan['control']) / 'paired-validation-predictions.npz') != plan['control_predictions_sha256']:
        raise ValueError('Control prediction cache changed')


def command(run, plan):
    cmd = [sys.executable, str(run / 'source/train_sharded.py'),
        '--manifest', str(Path(plan['data']) / 'manifest.json'), '--data-dir', plan['data'],
        '--artifact-dir', str(run / 'capped'), '--epochs', '3', '--learning-rate', '.0003',
        '--channels', '192', '--residual-blocks', '8', '--batch-size', '512',
        '--max-tokens-per-batch', '8192', '--gradient-clip', '1', '--min-side-characters', '2',
        '--max-sequence-length', str(plan['maximum_sequence_length']), '--position-weighting', 'none',
        '--seed', str(plan['seed']), '--checkpoint-shards', '1', '--defer-test']
    if (run / 'capped/training-state.pt').exists():
        cmd.append('--resume')
    else:
        cmd += ['--initialize-from', plan['initialization']]
    return cmd


def compare(run, plan, stopped):
    import numpy as np
    import torch
    from train_smoke import BoundaryChooser, configure_cpu, configure_mps, iterate_batches, batch_to_device
    from train_sharded import read_evaluation_records
    from verify_initial_learning_rate_results import paired_document_interval
    configure_cpu(1, 1)
    device = configure_mps()
    report = read(run / 'capped/validation-only.json')
    if report['test_evaluated'] or len(report['history']) != plan['epochs']:
        raise ValueError('Incomplete validation-only endpoint')
    expected = dict(plan['control_report']['configuration'], maximum_sequence_length=plan['maximum_sequence_length'])
    if report['configuration'] != expected:
        raise ValueError('Training configuration differs from the intended cap-only change')
    if report['data_sizes'] != {'train': plan['training_counts']['retained'],
                               'validation': plan['validation_counts']['retained']}:
        raise ValueError('Retained population differs')
    state = torch.load(run / 'capped/training-state.pt', map_location='cpu', weights_only=True)
    steps = {int(x['step'].item()) for x in state['optimizer_state']['state'].values()}
    if steps != {sum(x['capped_updates'] for x in plan['schedule'])}:
        raise ValueError('Filtered optimizer budget differs')
    if state['progress']['epoch'] != 4 or not all(torch.isfinite(v).all() for v in state['model_state'].values()):
        raise ValueError('Invalid endpoint checkpoint')
    vocab = read(Path(plan['data']) / 'vocabulary.json')
    records = read_evaluation_records(Path(plan['data']) / 'validation.jsonl', vocab, 2)
    indices = {r['id']: i for i, r in enumerate(records)}
    if len(indices) != len(records):
        raise ValueError('Validation IDs are not unique')
    targets = np.array([r['target_index'] for r in records], dtype=np.int32)
    lengths = np.array([len(r['tokens']) for r in records], dtype=np.int32)
    saved = np.load(Path(plan['control']) / 'paired-validation-predictions.npz')
    if not np.array_equal(saved['targets'], targets):
        raise ValueError('Validation order differs from verified control predictions')
    control_predictions = saved['lr-3e-4']
    if float((control_predictions == targets).mean()) != plan['control_report']['endpoint_validation']['accuracy']:
        raise ValueError('Control accuracy does not reproduce')
    model = BoundaryChooser(len(vocab), 192, 8)
    model.load_state_dict(state['model_state'])
    model.to(device).eval()
    predictions = np.full(len(records), -1, dtype=np.int32)
    with torch.inference_mode():
        for batch in iterate_batches(records, 512, 8192, shuffle=False, seed=0):
            if stopped():
                raise InterruptedError('Stopped during independent prediction replay')
            batch = batch_to_device(batch, device)
            guesses = model(batch['token_ids'], batch['token_mask'], batch['gap_mask']).argmax(1).cpu().numpy()
            for row, guess in zip(batch['records'], guesses):
                predictions[indices[row['id']]] = guess
    if (predictions < 0).any():
        raise ValueError('Unscored validation samples')
    short = lengths <= plan['maximum_sequence_length']
    if float((predictions[short] == targets[short]).mean()) != report['endpoint_validation']['accuracy']:
        raise ValueError('Independent capped endpoint accuracy differs from trainer report')
    metrics = {}
    for name, mask in (('common_within_limit', short), ('all_validation', np.ones(len(records), dtype=bool)),
                       ('excluded_over_limit', ~short)):
        n = int(mask.sum())
        a, b = int((control_predictions[mask] == targets[mask]).sum()), int((predictions[mask] == targets[mask]).sum())
        metrics[name] = {'samples': n, 'control_correct': a, 'capped_correct': b,
                         'control_accuracy': a / n if n else None, 'capped_accuracy': b / n if n else None,
                         'difference_pp': (b-a) / n * 100 if n else None}
    documents = np.array([r['document_id'] for r in records])
    interval = paired_document_interval(predictions[short] == targets[short],
        control_predictions[short] == targets[short], documents[short], plan['seed'] + 20)
    np.savez_compressed(run / 'paired-validation-predictions.npz', targets=targets, lengths=lengths,
                        control=control_predictions, capped=predictions)
    result = {'assessment': 'Share with caveats', 'plan': plan, 'metrics': metrics,
        'primary_paired_difference': interval, 'capped_report': report, 'capped_optimizer_updates': steps.pop(),
        'checkpoint_sha256': sha(run / 'capped/training-state.pt'),
        'predictions_sha256': sha(run / 'paired-validation-predictions.npz'),
        'test_evaluated': False, 'promoted': False, 'limitations': plan['limitations']}
    write_json(run / 'comparison.json', result)
    lo, hi = interval['paired_document_bootstrap_95_percent_interval_pp']
    lines = ['# A+B 总长度上限对比', '',
        f"从相同随机初始参数开始，学习率 0.0003，训练三轮；无位置或来源权重，排除单字侧目标。上限 {plan['maximum_sequence_length']} 字符。",
        f"对照每轮 {plan['training_counts']['read']:,} 条；过滤组每轮 {plan['training_counts']['retained']:,} 条，整条排除 {plan['training_counts']['excluded']:,} 条。原始文件不修改。",
        '对照复用已验证的学习率实验检查点。过滤组保留其余样本原有批次及顺序，跳过超长样本和空批次，不补充替代样本。',
        f"三轮参数更新：对照 {plan['control_report']['verified_checkpoint']['optimizer_steps']:,} 次，过滤组 {result['capped_optimizer_updates']:,} 次。",
        '主比较在完全相同的上限内验证样本上进行；完整验证集及超长组仅作诊断，测试集未读取或评分。', '',
        '| 验证范围 | 样本数 | 不限长训练 | 限长训练 | 差距（百分点） |', '| --- | ---: | ---: | ---: | ---: |']
    for key, label in (('common_within_limit', '共同的上限内样本（主比较）'),
                       ('all_validation', '全部原验证样本'), ('excluded_over_limit', '排除的超长样本')):
        m = metrics[key]
        if m['samples']:
            lines.append(f"| {label} | {m['samples']:,} | {100*m['control_accuracy']:.4f}% | {100*m['capped_accuracy']:.4f}% | {m['difference_pp']:+.4f} |")
    lines += ['', f"主比较净多答对 {interval['first_only_correct']-interval['second_only_correct']:+,} 条；按原文档聚类重采样 2,000 次，95% 差距区间 [{lo:+.4f}, {hi:+.4f}] 个百分点。",
        '这是单个随机种子的短程比较，区间不涵盖训练随机性，也不能证明完整语料上的最终效果。准确率衡量标点代理标签恢复，不等于人工阅读质量。',
        '改变验证集范围本身造成的分数变化不计作训练收益。长度过滤也不能保证剩余样本没有噪声。',
        '全量训练仍暂停，默认上限仍为 0（不限制），未自动应用或发布模型。', '',
        '复现与核验：plan.json、comparison.json、excluded-training-rows.json、paired-validation-predictions.npz、capped/validation-only.json。', '']
    (run / 'report.md').write_text('\n'.join(lines))
    return metrics['common_within_limit']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--control-dir', type=Path, default=DEFAULT_CONTROL)
    parser.add_argument('--max-sequence-length', type=int, default=256)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.max_sequence_length < 4:
        parser.error('Maximum length must allow two characters on each side')
    from run_smoke import ensure_dependencies
    ensure_dependencies()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    stopped, child = False, None
    def stop(signum, _frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)
    def notify(stage, **details):
        write_json(run / 'status.json', {'stage': stage, 'coordinator_pid': os.getpid(),
            'updated_at': datetime.now(timezone.utc).isoformat(), **details})
        print(json.dumps({'stage': stage, **details}), flush=True)
        if stopped:
            raise InterruptedError('Stop requested')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    with (run / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        awake = subprocess.Popen(['/usr/bin/caffeinate', '-is', '-w', str(os.getpid())], stdin=subprocess.DEVNULL)
        try:
            plan = read(run / 'plan.json') if (run / 'plan.json').exists() else prepare(run, args.control_dir.resolve(), args.max_sequence_length)
            if plan['maximum_sequence_length'] != args.max_sequence_length or Path(plan['control']) != args.control_dir.resolve():
                raise ValueError('Saved comparison plan differs; use its original arguments or a fresh run directory')
            verify(run, plan)
            notify('prepared', training=plan['training_counts'], validation=plan['validation_counts'])
            if args.prepare_only:
                return
            (run / 'capped').mkdir(exist_ok=True)
            report = run / 'capped/validation-only.json'
            environment = dict(os.environ, DEBUG='0', PYTHONUNBUFFERED='1',
                PYTHONPATH=str(ROOT / 'training/.deps'), SUPER_READER_PROJECT_ROOT=str(ROOT),
                PYTORCH_ENABLE_MPS_FALLBACK='0', PYTORCH_MPS_FAST_MATH='0', PYTORCH_MPS_PREFER_METAL='0')
            while not report.exists() or len(read(report)['history']) < plan['epochs']:
                checkpoint = run / 'capped/training-state.pt'
                before = checkpoint.stat().st_mtime_ns if checkpoint.exists() else None
                with (run / 'capped/training.log').open('ab') as log:
                    child = subprocess.Popen(command(run, plan), cwd=ROOT, env=environment,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    notify('training', child_pid=child.pid, maximum_sequence_length=plan['maximum_sequence_length'])
                    code = child.wait()
                if stopped:
                    raise InterruptedError('Worker stopped safely')
                if code == 75:
                    if not checkpoint.exists() or checkpoint.stat().st_mtime_ns == before:
                        raise RuntimeError('Memory refresh failed to advance checkpoint')
                    continue
                if code or not report.exists():
                    raise RuntimeError(f'Training failed with code {code}; inspect capped/training.log')
            verify(run, plan)
            notify('verifying-paired-predictions')
            metrics = compare(run, plan, lambda: stopped)
            notify('complete', primary=metrics, report=str(run / 'report.md'), test_evaluated=False, promoted=False)
        except InterruptedError as error:
            write_json(run / 'status.json', {'stage': 'stopped', 'reason': str(error), 'coordinator_pid': os.getpid()})
        except BaseException as error:
            write_json(run / 'status.json', {'stage': 'failed', 'error': str(error), 'coordinator_pid': os.getpid()})
            raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait()
            awake.terminate()
            awake.wait()


if __name__ == '__main__':
    main()
