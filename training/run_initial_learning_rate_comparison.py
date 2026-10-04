#!/usr/bin/env python3
"""Compare initial learning rates from common random weights, without test selection."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys

from sample_subset import keep_target, source_name, write_json

ROOT = Path(os.environ.get('SUPER_READER_PROJECT_ROOT', Path(__file__).resolve().parent.parent))
DEFAULT_DATA = ROOT / 'training/data/processed/padding-fixed-web-350m-v7-20260925'
RATES = {'lr-1e-4': .0001, 'lr-3e-4': .0003, 'lr-1e-3': .001, 'lr-2e-3': .002}
SOURCES = ('run_initial_learning_rate_comparison.py', 'train_sharded.py', 'train_smoke.py',
           'sample_subset.py', 'text_policy.py', 'unicode-symbols.json')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def inventory_shards(shards, notify):
    """Count every row without decoding the full corpus; hash the same byte stream."""
    inventory = []
    for i, item in enumerate(shards):
        path = absolute(item['path'])
        if path.stat().st_size != item['bytes']:
            raise ValueError(f'Shard size differs: {path}')
        digest, count, last = hashlib.sha256(), 0, b''
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024**2), b''):
                digest.update(block)
                count += block.count(b'\n')
                last = block[-1:]
        count += bool(last and last != b'\n')
        checksum = digest.hexdigest()
        if item.get('sha256', checksum) != checksum:
            raise ValueError(f'Shard hash differs: {path}')
        inventory.append({'path': str(path), 'rows': count, 'bytes': path.stat().st_size,
                          'sha256': checksum})
        if (i + 1) % 128 == 0 or i + 1 == len(shards):
            notify('counting-training-rows', shards=i + 1, total_shards=len(shards))
    return inventory


def sample_rows(inventory, requested, seed, *, evaluation=False, forbidden=(), notify=None):
    """Uniform row draws over all files, then condition on the same target filter.

    Random draw priority is retained while disk reads happen in sorted row order.
    Selecting the first eligible draws yields a uniform sample without replacement,
    without source quotas, whole-shard clustering or a maximum-length exclusion.
    """
    population = sum(item['rows'] for item in inventory)
    draws = min(population, math.ceil(requested * 1.1) + 100)
    chosen = sorted((index, priority) for priority, index in
                    enumerate(random.Random(seed).sample(range(population), draws)))
    pool, rejected, cursor, start = [], Counter(), 0, 0
    forbidden = set(forbidden)
    for file_number, item in enumerate(inventory):
        end = start + item['rows']
        with Path(item['path']).open('rb') as handle:
            previous = -1
            while cursor < len(chosen) and chosen[cursor][0] < end:
                index, priority = chosen[cursor]
                local = index - start
                line = next(islice(handle, local - previous - 1, None), None)
                if line is None:
                    raise ValueError(f'Row count differs: {item["path"]}')
                previous = local
                row = json.loads(line)
                text, target = (''.join(row['tokens']), row['target_index']) if evaluation else row[:2]
                if not keep_target(len(text), target, 2):
                    rejected['single_character_side'] += 1
                elif hashlib.sha256(text.encode()).digest() in forbidden:
                    rejected['validation_overlap'] += 1
                else:
                    pool.append((priority, row, [item['path'], local]))
                cursor += 1
        start = end
        if notify and (file_number + 1) % 256 == 0:
            notify('sampling-training-rows', shards=file_number + 1, sampled=len(pool))
    if len(pool) < requested:
        raise ValueError('Not enough eligible draws; increase oversampling before any trial')
    pool.sort(key=lambda item: item[0])
    return [(row, origin) for _, row, origin in pool[:requested]], {
        'population_rows': population, 'random_draws': draws, 'rejected_in_draws': dict(rejected),
        'retained_rows': requested, 'seed': seed, 'maximum_length_filter': None}


def prepare(run, args, notify):
    from text_policy import DATA_POLICY
    data = run / 'data'
    data.mkdir(exist_ok=True)
    snapshot = run / 'source'
    snapshot.mkdir(exist_ok=True)
    manifest_path = args.data_dir / 'manifest.json'
    manifest = read(manifest_path)
    policy = {key: manifest[key] for key in DATA_POLICY}
    for key in DATA_POLICY:
        if key not in ('standard', 'sample_filter') and policy[key] != DATA_POLICY[key]:
            raise ValueError('Prepared input/label semantics differ: ' + key)
    for name in SOURCES:
        shutil.copy2(ROOT / 'training' / name, snapshot / name)
    write_json(snapshot / 'text-policy.json', policy)
    vocabulary = read(absolute(manifest['vocabulary_path']))
    write_json(data / 'vocabulary.json', vocabulary)

    original_summary = read(args.data_dir / 'summary.json')
    validation_path = args.data_dir / 'validation.jsonl'
    if sha(validation_path) != original_summary['splits']['validation']['sha256']:
        raise ValueError('Validation file changed')
    validation, validation_sampling = sample_rows([
        {'path': str(validation_path), 'rows': original_summary['splits']['validation']['count']}],
        args.validation_samples, args.seed + 1, evaluation=True)
    validation_hashes = {hashlib.sha256(''.join(row['tokens']).encode()).digest() for row, _ in validation}
    if len({row['id'] for row, _ in validation}) != len(validation):
        raise ValueError('Validation IDs are not unique')
    with (data / 'validation.jsonl').open('w') as handle, (data / 'validation.source.jsonl').open('w') as origins:
        for row, origin in validation:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
            origins.write(json.dumps(origin) + '\n')
    del validation
    # --defer-test is mandatory. This empty placeholder satisfies only identity stat().
    (data / 'test.jsonl').write_text('')
    write_json(data / 'summary.json', {**policy, 'splits': {
        'validation': {'count': args.validation_samples, 'sha256': sha(data / 'validation.jsonl')},
        'test': {'count': 0, 'purpose': 'empty placeholder; test is never read or scored'}}})

    inventory_path = run / 'corpus-inventory.json'
    if inventory_path.exists():
        cached = read(inventory_path)
        if cached['manifest_sha256'] != sha(manifest_path):
            raise ValueError('Cached inventory belongs to another manifest')
        inventory = cached['shards']
        if any(Path(item['path']).stat().st_size != item['bytes'] for item in inventory):
            raise ValueError('Cached inventory size changed')
    else:
        inventory = inventory_shards(manifest['shards'], notify)
        write_json(inventory_path, {'manifest_sha256': sha(manifest_path), 'shards': inventory})
    if sum(item['rows'] for item in inventory) != manifest['statistics']['samples']:
        raise ValueError('Full row count differs from manifest population')
    training, sampling = sample_rows(inventory, args.samples, args.seed + 2,
                                    forbidden=validation_hashes, notify=notify)
    keys = {hashlib.sha256(row[0].encode()).digest() for row, _ in training}
    if len(keys) != len(training):
        raise ValueError('Duplicate sampled training inputs require investigation')
    stats = {'samples': len(training), 'tokens': 0, 'cells': [[0] * len(row)
             for row in manifest['statistics']['cells']], 'maximum_sequence_length': 0,
             'source_samples': Counter(), 'random_baseline_sum': 0., 'center_correct': 0}
    shards, hashes = [], {}
    for number, offset in enumerate(range(0, len(training), 50000)):
        path = data / f'train-{number:03d}.jsonl'
        origin_path = path.with_suffix('.source.jsonl')
        with path.open('w') as handle, origin_path.open('w') as origins:
            for row, origin in training[offset:offset + 50000]:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
                origins.write(json.dumps(origin) + '\n')
                text, target, source, bucket, position = row
                stats['tokens'] += len(text)
                stats['cells'][bucket][position] += 1
                stats['source_samples'][source_name(manifest['domains'][source])] += 1
                stats['maximum_sequence_length'] = max(stats['maximum_sequence_length'], len(text))
                stats['random_baseline_sum'] += 1 / (len(text) - 1)
                stats['center_correct'] += target == (len(text) - 1) // 2
        shards.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)})
        hashes[path.name], hashes[origin_path.name] = sha(path), sha(origin_path)
    del training, keys
    write_json(data / 'manifest.json', {**policy, 'format': manifest['format'],
        'domains': manifest['domains'], 'length_bucket_maximums': manifest['length_bucket_maximums'],
        'position_bins': manifest['position_bins'], 'vocabulary_path': str(data / 'vocabulary.json'),
        'statistics': stats, 'shards': shards})
    for name in ('manifest.json', 'vocabulary.json', 'summary.json', 'validation.jsonl',
                 'validation.source.jsonl', 'test.jsonl'):
        hashes[name] = sha(data / name)

    from run_smoke import ensure_dependencies
    ensure_dependencies()
    import torch
    from train_smoke import BoundaryChooser, seed_everything, save_training_state
    seed_everything(args.seed)
    model = BoundaryChooser(len(vocabulary), 192, 8)
    save_training_state(run / 'initialization.pt', {'model_state': model.state_dict(), 'best_state': None,
        'vocabulary': vocabulary, 'best_epoch': 0, 'initialization_kind': 'random; never trained',
        'seed': args.seed})
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'seed': args.seed,
        'samples': args.samples, 'validation_samples': args.validation_samples,
        'initialization': 'common random weights; fresh AdamW in every arm',
        'initialization_sha256': sha(run / 'initialization.pt'), 'rates': RATES,
        'stage_one_epochs': 1, 'extended_epochs': args.extended_epochs, 'extend_top': 2,
        'training': {'channels': 192, 'residual_blocks': 8, 'batch_size': 512,
                     'max_tokens_per_batch': 8192, 'gradient_clip': 1, 'min_side_characters': 2,
                     'position_weighting': 'none'},
        'source_weighting': 'none', 'selection_metric': 'overall_validation_accuracy',
        'tie_break': 'smaller learning rate',
        'source_hashes': {p.name: sha(p) for p in snapshot.iterdir() if p.is_file()},
        'data_hashes': hashes, 'original_manifest': str(manifest_path),
        'original_manifest_sha256': sha(manifest_path), 'prepared_data_standard': policy['standard'],
        'training_sampling': sampling, 'validation_sampling': validation_sampling,
        'test_evaluated': False, 'automatic_promotion': False,
        'limitations': 'One initialization seed and a reused validation subset. Stage two repeats the same training subset; '
                      'short-run ranking is not proof of full-corpus or decayed-schedule performance.'}
    write_json(run / 'plan.json', plan)
    return plan


def training_command(run, plan, arm, epochs):
    command = [sys.executable, str(run / 'source/train_sharded.py'),
        '--manifest', str(run / 'data/manifest.json'), '--data-dir', str(run / 'data'),
        '--artifact-dir', str(run / arm), '--epochs', str(epochs),
        '--learning-rate', str(plan['rates'][arm]), '--seed', str(plan['seed']),
        '--checkpoint-shards', '1', '--defer-test']
    for name, value in plan['training'].items():
        command += ['--' + name.replace('_', '-'), str(value)]
    if (run / arm / 'training-state.pt').exists():
        command.append('--resume')
    else:
        command += ['--initialize-from', str(run / 'initialization.pt')]
    return command


def rank_arms(reports):
    for arm, report in reports.items():
        if report['test_evaluated'] or not math.isfinite(report['endpoint_validation']['accuracy']):
            raise ValueError('Invalid validation-only result: ' + arm)
    # Tie-break predeclared: smaller learning rate; no test score or loss tie-break.
    return sorted(reports, key=lambda arm: (-reports[arm]['endpoint_validation']['accuracy'], RATES[arm]))


def verify(run, plan):
    for directory, field in (('source', 'source_hashes'), ('data', 'data_hashes')):
        for name, expected in plan[field].items():
            if sha(run / directory / name) != expected:
                raise ValueError('Frozen file changed: ' + name)
    if sha(run / 'initialization.pt') != plan['initialization_sha256']:
        raise ValueError('Common random initialization changed')


def verify_trial_reports(reports):
    """Check equal budgets and actual starting scores, not just intended commands."""
    reference_config, reference_initial, steps = None, None, set()
    for report in reports.values():
        if report['test_evaluated'] or report['training_weighting']['source_weighting'] != 'none':
            raise ValueError('Trial changed test use or source weighting')
        configuration = dict(report['configuration'])
        configuration.pop('learning_rate')
        initial = report['initialization']['validation_before_training']
        if reference_config is None:
            reference_config, reference_initial = configuration, initial
        elif configuration != reference_config or any(
                not math.isclose(initial[key], reference_initial[key], rel_tol=0, abs_tol=1e-6)
                for key in ('accuracy', 'loss', 'mean_reciprocal_rank')):
            raise ValueError('Trial configuration or initial validation differs')
        steps.add(report['verified_checkpoint']['optimizer_steps'])
    if len(steps) != 1:
        raise ValueError('Optimizer update counts differ')


def checkpoint_summary(path, plan, epochs):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['progress']['epoch'] != epochs + 1 or len(state['history']) != epochs:
        raise ValueError('Incomplete equal-budget endpoint')
    if state['source_inventory'] and sum(s['samples'] for s in state['source_inventory'].values()) != plan['samples']:
        raise ValueError('Training sample count differs')
    if not all(torch.isfinite(v).all().item() for v in state['model_state'].values()):
        raise ValueError('Non-finite model weights')
    steps = {int(value['step'].item()) for value in state['optimizer_state']['state'].values()}
    if len(steps) != 1:
        raise ValueError('Optimizer parameter step counts differ')
    return {'optimizer_steps': steps.pop(), 'weights_finite': True,
            'checkpoint_sha256': sha(path), 'best_epoch': state['best_epoch']}


def write_report(run, plan, stage_one, extended):
    ranking = rank_arms(extended)
    result = {'plan': plan, 'stage_one': stage_one, 'extended': extended,
              'winner': ranking[0], 'confidence': 'Share with caveats',
              'criterion': 'Highest endpoint validation accuracy at equal extended exposure',
              'test_evaluated': False, 'promoted': False}
    write_json(run / 'comparison.json', result)
    lines = ['# 从随机初始化比较初始学习率', '',
        f"固定训练样本 {plan['samples']:,} 条、验证样本 {plan['validation_samples']:,} 条；原数据 {plan['prepared_data_standard']}。",
        '训练与验证均排除任一侧为单字的目标；无句长上限，无位置或来源权重。',
        '全库按行等概率抽样，各组共用随机初始权重、全新 AdamW、样本顺序和批次。测试集未使用。', '',
        '| 学习率 | 1 轮验证准确率 | 延长后验证准确率 | 延长后验证损失 |',
        '| --- | ---: | ---: | ---: |']
    for arm, rate in plan['rates'].items():
        first = stage_one[arm]['endpoint_validation']['accuracy'] * 100
        final = extended.get(arm, {}).get('endpoint_validation')
        lines.append(f"| {rate:g} | {first:.4f}% | " +
                     (f"{final['accuracy'] * 100:.4f}% | {final['loss']:.6f} |" if final else '未延长 | — |'))
    lines += ['', f"延长至同一子集 {plan['extended_epochs']} 轮后，候选中表现最好的是 **{plan['rates'][ranking[0]]:g}**。",
        '这是单个初始化种子和复用验证子集上的短程比较，不能保证完整语料或学习率衰减后的排名相同。',
        '只延长首轮前两名，因此没有排除其他学习率在更长训练中反超的可能。',
        '比较指标衡量标点代理断点的恢复，不等于人工阅读质量。原完整训练保持暂停，模型未发布。', '',
        '复核依据：plan.json、corpus-inventory.json、stage-one.json、comparison.json、各组 checkpoint-summary-*.json。',
        '训练样本和验证样本均有原分片及从 0 开始的行号；保留代码、输入和检查点哈希。', '']
    (run / 'report.md').write_text('\n'.join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--samples', type=int, default=1000000)
    parser.add_argument('--validation-samples', type=int, default=100000)
    parser.add_argument('--extended-epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2026100411)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if min(args.samples, args.validation_samples) < 1 or args.extended_epochs < 2:
        parser.error('Positive sample sizes and at least two extended epochs required')
    run, args.data_dir = args.run_dir.resolve(), args.data_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    stopped, child = False, None

    def stop(signum, _frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    def notify(stage, **details):
        write_json(run / 'status.json', {'stage': stage, 'updated_at': datetime.now(timezone.utc).isoformat(),
                                        'coordinator_pid': os.getpid(), **details})
        print(json.dumps({'stage': stage, **details}), flush=True)
        if stopped:
            raise InterruptedError('Stop requested')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    with (run / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        awake = subprocess.Popen(['/usr/bin/caffeinate', '-is', '-w', str(os.getpid())], stdin=subprocess.DEVNULL)
        try:
            plan = read(run / 'plan.json') if (run / 'plan.json').exists() else prepare(run, args, notify)
            verify(run, plan)
            from run_smoke import ensure_dependencies
            ensure_dependencies()
            environment = dict(os.environ, DEBUG='0', PYTHONUNBUFFERED='1',
                PYTHONPATH=str(ROOT / 'training/.deps'), SUPER_READER_PROJECT_ROOT=str(ROOT),
                PYTORCH_ENABLE_MPS_FALLBACK='0', PYTORCH_MPS_FAST_MATH='0', PYTORCH_MPS_PREFER_METAL='0')
            notify('prepared', samples=plan['samples'], validation=plan['validation_samples'])
            if args.prepare_only:
                return

            def train(arm, epochs):
                nonlocal child
                output = run / arm
                output.mkdir(exist_ok=True)
                report_path = output / 'validation-only.json'
                while not report_path.exists() or len(read(report_path)['history']) < epochs:
                    command = training_command(run, plan, arm, epochs)
                    checkpoint = output / 'training-state.pt'
                    stamp = checkpoint.stat().st_mtime_ns if checkpoint.exists() else None
                    with (output / 'training.log').open('ab') as log:
                        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                                 stdout=log, stderr=subprocess.STDOUT)
                        notify('training', arm=arm, epochs=epochs, child_pid=child.pid)
                        code = child.wait()
                    if stopped:
                        raise InterruptedError('Worker stopped safely')
                    if code == 75:
                        if not checkpoint.exists() or checkpoint.stat().st_mtime_ns == stamp:
                            raise RuntimeError('Memory refresh did not advance checkpoint')
                        continue
                    if code or not report_path.exists() or len(read(report_path)['history']) != epochs:
                        raise RuntimeError(f'{arm} exited {code}; inspect training.log')
                report = read(report_path)
                if report['data_sizes'] != {'train': plan['samples'], 'validation': plan['validation_samples']}:
                    raise ValueError('Trial sample counts differ')
                check = checkpoint_summary(output / 'training-state.pt', plan, epochs)
                write_json(output / f'checkpoint-summary-{epochs}.json', check)
                return {**report, 'verified_checkpoint': check}

            if (run / 'stage-one.json').exists():
                stage_one = read(run / 'stage-one.json')['arms']
            else:
                stage_one = {arm: train(arm, 1) for arm in plan['rates']}
                verify_trial_reports(stage_one)
                write_json(run / 'stage-one.json', {'arms': stage_one, 'ranking': rank_arms(stage_one)})
            chosen = rank_arms(stage_one)[:plan['extend_top']]
            notify('extending-top-two', arms=chosen)
            extended = {arm: train(arm, plan['extended_epochs']) for arm in chosen}
            verify_trial_reports(extended)
            result = write_report(run, plan, stage_one, extended)
            notify('complete', winner=result['winner'], learning_rate=plan['rates'][result['winner']],
                   report=str(run / 'report.md'), test_evaluated=False, promoted=False)
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
