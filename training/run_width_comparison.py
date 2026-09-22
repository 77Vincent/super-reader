#!/usr/bin/env python3
"""Run a resumable, fixed-data 192/256-channel comparison on Apple Metal."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys

from run_web_continuation import read, sha, write, run_with_recovery

ROOT = Path(os.environ.get('SUPER_READER_PROJECT_ROOT', Path(__file__).resolve().parent.parent))
BASE = ROOT / 'training/data/processed/chinese-line-web-200m-16conv-v7-20260920/manifest.json'
EVALUATION = ROOT / 'training/data/processed/chinese-line-web-100m-16conv-v7-20260917-eval'
BEST = ROOT / 'training/artifacts/chinese-line-web-200m-16conv-v7-20260920/epoch-1-backend'
RUN = ROOT / 'training/artifacts/width-192-vs-256-10m-v7-20260922'
SEED = 2026092201
SOURCES = ['training/' + name for name in (
    'run_width_comparison.py', 'widen_boundary.py', 'finalize_width_arm.py', 'benchmark_width.mjs',
    'run_web_continuation.py', 'run_sharded.py', 'run_smoke.py', 'train_sharded.py',
    'train_smoke.py', 'text_policy.py', 'text-policy.json', 'unicode-symbols.json', 'export_browser_model.py')]
SOURCES += ['src/backend/inference.js', 'test/model-backend-reference.json']


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def select_shards(base, target, seed):
    """Sample hash-distributed shards within each source generation, not a prefix.

    Whole shards keep original samples, order and labels untouched. The result
    is approximately the requested count, with rounding recorded per source.
    """
    groups = defaultdict(list)
    for shard in base['shards']:
        groups[str(project_path(shard['path']).parent)].append(shard)
    fraction = target / base['statistics']['samples']
    if not 0 < fraction <= 1:
        raise ValueError('Target must be positive and no larger than the base corpus')
    chosen, selections = [], []
    rng = random.Random(seed)
    for parent, shards in sorted(groups.items()):
        number = max(1, min(len(shards), round(len(shards) * fraction)))
        sample = sorted(rng.sample(shards, number), key=lambda item: item['path'])
        chosen.extend(sample)
        selections.append({'source_directory': parent, 'available_shards': len(shards), 'selected_shards': number})
    return chosen, selections


def prepare_pilot(run, target):
    output = run / 'pilot-manifest.json'
    if output.exists():
        manifest = read(output)
        for item in manifest['shards']:
            if sha(Path(item['path'])) != item['sha256']:
                raise ValueError('Selected training shard changed: ' + item['path'])
        return manifest
    base = read(BASE)
    chosen, selections = select_shards(base, target, SEED)
    stats = {'samples': 0, 'tokens': 0, 'random_baseline_sum': 0., 'center_correct': 0,
             'maximum_sequence_length': 0, 'domain_samples': {domain: 0 for domain in base['domains']},
             'position_histogram': [0] * base['position_bins'],
             'cells': [[0] * base['position_bins'] for _ in base['statistics']['cells']]}
    shards, counts = [], Counter()
    for index, item in enumerate(chosen):
        path = project_path(item['path'])
        if path.stat().st_size != item['bytes']:
            raise ValueError(f'Shard size changed: {path}')
        digest = sha(path)
        if item.get('sha256', digest) != digest:
            raise ValueError(f'Shard hash changed: {path}')
        samples = 0
        with path.open() as handle:
            for line in handle:
                text, target_index, domain, bucket, position = json.loads(line)
                length = len(text)
                if not 0 <= target_index < length - 1:
                    raise ValueError(f'Invalid target in {path}')
                samples += 1
                stats['samples'] += 1
                stats['tokens'] += length
                stats['random_baseline_sum'] += 1 / (length - 1)
                stats['center_correct'] += target_index == (length - 1) // 2
                stats['maximum_sequence_length'] = max(stats['maximum_sequence_length'], length)
                stats['domain_samples'][base['domains'][domain]] += 1
                stats['position_histogram'][position] += 1
                stats['cells'][bucket][position] += 1
        counts[str(path.parent)] += samples
        shards.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest, 'samples': samples})
        print(f'pilot shards={index + 1}/{len(chosen)} samples={stats["samples"]}', flush=True)
    if any(count <= 0 for count in stats['domain_samples'].values()):
        raise ValueError('Pilot omitted a training domain')
    for item in selections:
        item['actual_samples'] = counts[item['source_directory']]
    from text_policy import DATA_POLICY, require_data_policy
    require_data_policy(base)
    manifest = {**DATA_POLICY, 'format': base['format'], 'source': 'fixed source-stratified width comparison',
        'base_manifest': str(BASE), 'base_manifest_sha256': sha(BASE), 'domains': base['domains'],
        'evaluation_source_dir': str(EVALUATION), 'vocabulary_path': str(run / 'vocabulary.json'),
        'length_bucket_maximums': base['length_bucket_maximums'], 'position_bins': base['position_bins'],
        'maximum_sequence_length': base['maximum_sequence_length'], 'statistics': stats, 'shards': shards,
        'sampling': {'seed': SEED, 'requested_samples': target, 'method': 'seeded selection of existing hash shards within each source directory',
                     'sources': selections, 'same_samples_and_order_for_both_arms': True},
        'protection': base['protection']}
    write(output, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=RUN)
    parser.add_argument('--target-samples', type=int, default=10_000_000)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot = run / 'source'
    plan_path = run / 'run.json'
    if plan_path.exists() and not args.resume:
        raise FileExistsError('Experiment exists; resume with --resume')
    if not plan_path.exists():
        if read(BEST / 'smoke-metrics.json')['architecture'] != {
            'channels': 192, 'residual_blocks': 8, 'convolutions_per_block': 2, 'kernel_size': 3, 'dilation': 1
        }:
            raise ValueError('This experiment expects the frozen 192-channel, 16-convolution model')
        for name in SOURCES:
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        (snapshot / 'training/.deps').symlink_to(ROOT / 'training/.deps', target_is_directory=True)
        shutil.copy2(BEST / 'training-state.pt', run / 'initialization-192.pt')
        shutil.copy2(BEST / 'boundary-smoke-vocabulary.json', run / 'vocabulary.json')
        # Do not reuse an unrecorded exploratory preflight from this directory.
        for name in ('initialization.pt', 'verification.json'):
            (run / 'preflight' / name).unlink(missing_ok=True)
        splits = {}
        for split in ('validation', 'test'):
            path = EVALUATION / f'{split}.jsonl'
            splits[split] = {'count': read(EVALUATION / 'summary.json')['splits'][split]['count'],
                             'bytes': path.stat().st_size, 'sha256': sha(path)}
        write(plan_path, {'created_at': datetime.now(timezone.utc).isoformat(), 'target_samples': args.target_samples,
            'seed': SEED, 'arms': [192, 256], 'epochs_per_arm': 1, 'convolution_layers': 16,
            'initialization_source': str(BEST), 'initialization_sha256': sha(run / 'initialization-192.pt'),
            'vocabulary_sha256': sha(run / 'vocabulary.json'), 'base_manifest_sha256': sha(BASE),
            'evaluation_summary_sha256': sha(EVALUATION / 'summary.json'), 'evaluation': splits,
            'source_hashes': {name: sha(snapshot / name) for name in SOURCES},
            'backend_sha256_at_start': sha(ROOT / 'src/boundary-model-data.js'),
            'comparison': 'same samples, order, batch sizes, updates, fresh AdamW and full holdouts; inherited function preserved during widening',
            'scope': 'continued training from the current best model; not a from-scratch or equal-compute architecture comparison',
            'training': {'learning_rate': .0003, 'batch_size': 512, 'max_tokens_per_batch': 8192,
                         'domain_weight_power': .65, 'selection_macro_weight': .5, 'gradient_clip': 1.0}})
    plan = read(plan_path)
    if args.target_samples != plan['target_samples']:
        raise ValueError('Resume with the original sample target')
    for path, expected in [(run / 'initialization-192.pt', plan['initialization_sha256']),
                           (run / 'vocabulary.json', plan['vocabulary_sha256']), (BASE, plan['base_manifest_sha256']),
                           (EVALUATION / 'summary.json', plan['evaluation_summary_sha256'])]:
        if sha(path) != expected:
            raise ValueError(f'Frozen input changed: {path}')
    for split, item in plan['evaluation'].items():
        if sha(EVALUATION / f'{split}.jsonl') != item['sha256']:
            raise ValueError(f'Frozen holdout changed: {split}')
    for name, expected in plan['source_hashes'].items():
        if sha(snapshot / name) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    # On restart use the immutable coordinator too, not a later working-tree edit.
    if Path(__file__).resolve() != snapshot / 'training/run_width_comparison.py':
        # Release this process's lock before the frozen coordinator acquires it.
        fcntl.flock(lock, fcntl.LOCK_UN)
        os.environ['SUPER_READER_PROJECT_ROOT'] = str(ROOT)
        os.execv(sys.executable, [sys.executable, str(snapshot / 'training/run_width_comparison.py'),
            '--run-dir', str(run), '--target-samples', str(args.target_samples), '--resume'])
    child, stopped = None, False
    def stop(signum, _frame):
        nonlocal stopped
        if stopped:
            return
        stopped = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    def status(stage, **details):
        value = {'stage': stage, 'updated_at': datetime.now(timezone.utc).isoformat(),
                 'coordinator_pid': os.getpid(), **details}
        write(run / 'status.json', value)
        print(json.dumps(value), flush=True)
    environment = dict(os.environ, DEBUG='0', PYTHONUNBUFFERED='1', PYTHONPATH=str(ROOT / 'training/.deps'),
        SUPER_READER_PROJECT_ROOT=str(ROOT), PYTORCH_ENABLE_MPS_FALLBACK='0',
        PYTORCH_MPS_FAST_MATH='0', PYTORCH_MPS_PREFER_METAL='0')
    def execute(stage, command, checkpoint=None):
        def run_once(current_command):
            nonlocal child
            with (run / f'{stage}.log').open('ab') as log:
                child = subprocess.Popen(current_command, cwd=snapshot, env=environment, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
                status(stage, child_pid=child.pid, command=current_command, log=str(run / f'{stage}.log'))
                return child.wait()
        run_with_recovery('training' if checkpoint else stage, command, run_once=run_once,
            checkpoint=checkpoint or run / 'unused', should_stop=lambda: stopped,
            on_refresh=lambda: status('refreshing-worker', arm=stage, reason='saved next batch under MPS memory pressure'))
    awake = subprocess.Popen(['/usr/bin/caffeinate', '-is', '-w', str(os.getpid())], stdin=subprocess.DEVNULL)
    try:
        status('preparing-pilot')
        pilot = prepare_pilot(run, args.target_samples)
        if stopped:
            raise InterruptedError('Stopped after pilot preparation')
        if not (run / 'preflight/initialization.pt').exists() or not (run / 'preflight/verification.json').exists():
            execute('preflight', [sys.executable, str(snapshot / 'training/widen_boundary.py'),
                '--checkpoint', str(run / 'initialization-192.pt'), '--output-dir', str(run / 'preflight'),
                '--reference', str(snapshot / 'test/model-backend-reference.json')])
        if not read(run / 'preflight/verification.json')['passed']:
            raise ValueError('Widening preflight failed')
        for channels in plan['arms']:
            arm = run / f'ch{channels}'
            checkpoint = arm / 'training-state.pt'
            if not (arm / 'smoke-metrics.json').exists():
                command = [sys.executable, str(snapshot / 'training/run_sharded.py'),
                    '--manifest', str(run / 'pilot-manifest.json'), '--data-dir', str(EVALUATION),
                    '--artifact-dir', str(arm), '--epochs', '1', '--channels', str(channels), '--residual-blocks', '8',
                    '--seed', str(SEED), '--checkpoint-shards', '1', '--threads', '1', '--interop-threads', '1', '--device', 'mps']
                for key, value in plan['training'].items():
                    command += ['--' + key.replace('_', '-'), str(value)]
                if checkpoint.exists():
                    command += ['--resume']
                else:
                    initialization = run / 'initialization-192.pt' if channels == 192 else run / 'preflight/initialization.pt'
                    command += ['--initialize-from', str(initialization)]
                execute(f'train-{channels}', command, checkpoint)
            if not (arm / 'smoke-metrics.json').exists():
                raise InterruptedError(f'Arm {channels} stopped before final evaluation; resume its saved batch')
            metrics = read(arm / 'smoke-metrics.json')
            expected = {'train': pilot['statistics']['samples'], **{key: value['count'] for key, value in plan['evaluation'].items()}}
            if metrics['data_sizes'] != expected or len(metrics['history']) != 1:
                raise ValueError('Incomplete arm or mismatched data sizes')
            if not (arm / 'final/complete.json').exists():
                execute(f'finalize-{channels}', [sys.executable, str(snapshot / 'training/finalize_width_arm.py'),
                    '--artifact-dir', str(arm), '--evaluation', str(EVALUATION),
                    '--reference', str(snapshot / 'test/model-backend-reference.json')])
        execute('browser-benchmark', ['node', str(snapshot / 'training/benchmark_width.mjs'), str(run)])
        arms = {str(ch): read(run / f'ch{ch}/final/smoke-metrics.json') for ch in plan['arms']}
        initial = [arms[str(ch)]['initialization']['validation_before_training']['accuracy'] for ch in plan['arms']]
        if abs(initial[0] - initial[1]) > .0001:
            raise ValueError('Initial full-validation accuracy differs by more than 0.01 percentage point')
        result = {'scope': plan['scope'], 'training_samples_per_arm': pilot['statistics']['samples'],
            'full_holdouts': plan['evaluation'], 'initial_validation_accuracy': dict(zip(map(str, plan['arms']), initial)),
            'arms': {key: {'parameters': m['parameter_count'], 'validation': m['validation'], 'test': m['test'],
                          'history': m['history'], 'selected_best_epoch': read(run / f'ch{key}/smoke-metrics.json')['best_epoch']}
                     for key, m in arms.items()},
            'delta_256_minus_192': {split + '_accuracy_percentage_points': 100 * (arms['256'][split]['accuracy'] - arms['192'][split]['accuracy'])
                                   for split in ('validation', 'test')},
            'browser_inference': read(run / 'browser-benchmark.json'),
            'backend_unchanged': sha(ROOT / 'src/boundary-model-data.js') == plan['backend_sha256_at_start']}
        write(run / 'comparison.json', result)
        status('complete', report=str(run / 'comparison.json'), **result['delta_256_minus_192'])
    except InterruptedError as error:
        status('stopped', reason=str(error))
    except BaseException as error:
        status('failed', error=str(error))
        raise
    finally:
        awake.terminate()


if __name__ == '__main__':
    main()
