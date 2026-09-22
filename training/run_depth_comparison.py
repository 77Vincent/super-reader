#!/usr/bin/env python3
"""Train a 20-layer arm against the completed, frozen 16-layer low-LR pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from run_learning_rate_comparison import training_command as rate_training_command
from run_web_continuation import read, write, sha, run_with_recovery

ROOT = Path(os.environ.get('SUPER_READER_PROJECT_ROOT', Path(__file__).resolve().parent.parent))
BASE = ROOT / 'training/artifacts/learning-rate-192ch-10m-v7-20260922'
RUN = ROOT / 'training/artifacts/depth-16-vs-20-10m-lr3e-5-v7-20260922'
SOURCES = ['training/' + name for name in (
    'run_depth_comparison.py', 'analyze_depth_comparison.py', 'run_learning_rate_comparison.py',
    'run_web_continuation.py', 'run_sharded.py', 'run_smoke.py', 'train_sharded.py',
    'train_smoke.py', 'text_policy.py', 'text-policy.json', 'unicode-symbols.json',
    'widen_boundary.py', 'export_browser_model.py', 'benchmark_width.mjs')]
SOURCES += ['src/backend/inference.js', 'test/model-backend-reference.json']


def training_command(snapshot, run, plan, *, finalize=False):
    command = rate_training_command(snapshot, run, plan, 'layers20', finalize=finalize)
    command[command.index('--residual-blocks') + 1] = '10'
    return command


def select_depth(control, candidate):
    import math
    scores = [m['endpoint_selection_score'] for m in (control, candidate)]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError('Non-finite validation score')
    return {'winner': 'layers20' if scores[1] > scores[0] else 'layers16',
            'selection_score': max(scores),
            'criterion': '0.5 overall + 0.5 macro-domain validation accuracy; ties keep 16 layers'}


def prepare(run, base):
    baseline = read(base / 'run.json')
    control = read(base / 'lr-3e-5/validation-only.json')
    if read(base / 'status.json')['stage'] != 'complete':
        raise ValueError('Reference pilot is not complete')
    expected = {'channels': 192, 'residual_blocks': 8, 'learning_rate': 0.00003,
                'validation_limit': 0, 'test_limit': 0, 'max_shards': 0}
    if any(control['configuration'][k] != v for k, v in expected.items()):
        raise ValueError('Expected the full-holdout, 192-channel, 16-layer low-LR control')
    if len(control['history']) != 1 or control['best_epoch'] != 1:
        raise ValueError('Expected a completed one-epoch control selected at epoch 1')
    for name, digest in baseline['source_hashes'].items():
        if sha(base / 'source' / name) != digest:
            raise ValueError('Reference source changed: ' + name)
    for name in SOURCES:
        # Use the actual frozen control trainer, even if workspace code changes later.
        origin = base / 'source' / name if name in baseline['source_hashes'] else ROOT / name
        destination = run / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, destination)
    (run / 'source/training/.deps').symlink_to(ROOT / 'training/.deps', target_is_directory=True)
    for name in ('initialization.pt', 'vocabulary.json', 'pilot-manifest.json'):
        if sha(base / name) != baseline['input_hashes'][str(base / name)]:
            raise ValueError('Reference input changed: ' + name)
        shutil.copy2(base / name, run / name)
    control_dir = run / 'layers16'
    control_dir.mkdir()
    for name in ('training-state.pt', 'validation-only.json', 'smoke-metrics.json'):
        shutil.copy2(base / 'lr-3e-5' / name, control_dir / name)
    inputs = dict(baseline['input_hashes'])
    for path in [run / name for name in ('initialization.pt', 'vocabulary.json', 'pilot-manifest.json')]:
        inputs[str(path)] = sha(path)
    for path in control_dir.iterdir():
        inputs[str(path)] = sha(path)
    training = {**baseline['training'], 'learning_rate': 0.00003}
    if any(control['configuration'][key] != value for key, value in training.items()):
        raise ValueError('Reference training options differ from recorded plan')
    write(run / 'run.json', {
        'created_at': datetime.now(timezone.utc).isoformat(), 'reference_run': str(base),
        'rates': {'layers20': 0.00003}, 'training': training, 'seed': baseline['seed'],
        'epochs_per_arm': 1, 'training_samples_per_arm': baseline['training_samples_per_arm'],
        'evaluation_dir': baseline['evaluation_dir'], 'evaluation': baseline['evaluation'],
        'initialization_source': str(base / 'initialization.pt'),
        'control_reused': True, 'control_endpoint_used_for_initialization': False,
        'architecture': {'channels': 192, 'control_blocks': 8, 'candidate_blocks': 10},
        'source_hashes': {name: sha(run / 'source' / name) for name in SOURCES},
        'input_hashes': inputs, 'backend_sha256_at_start': sha(ROOT / 'src/boundary-model-data.js'),
        'test_protocol': 'Select on complete validation. Test only a winning 20-layer candidate; otherwise reuse the measured 16-layer test.',
        'strata': 'All validation: token lengths 2-17/18-34/35-42/43-64/65-128/129+; gold-gap context fully covered by 16 / newly by 20 / neither.',
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=RUN)
    parser.add_argument('--reference-run', type=Path, default=BASE)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / 'run.json').exists() and not args.resume:
        raise FileExistsError('Experiment exists; use --resume')
    if not (run / 'run.json').exists():
        prepare(run, args.reference_run.resolve())
    plan = read(run / 'run.json')
    snapshot = run / 'source'
    for name, expected in plan['source_hashes'].items():
        if sha(snapshot / name) != expected:
            raise ValueError('Frozen source changed: ' + name)
    if Path(__file__).resolve() != snapshot / 'training/run_depth_comparison.py':
        fcntl.flock(lock, fcntl.LOCK_UN)
        os.environ['SUPER_READER_PROJECT_ROOT'] = str(ROOT)
        os.execv(sys.executable, [sys.executable, str(snapshot / 'training/run_depth_comparison.py'),
                                 '--run-dir', str(run), '--resume'])
    child, stopped = None, False

    def stop(signum, _frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)

    def status(stage, **details):
        value = {'stage': stage, 'updated_at': datetime.now(timezone.utc).isoformat(),
                 'coordinator_pid': os.getpid(), **details}
        write(run / 'status.json', value)
        print(value, flush=True)

    environment = dict(os.environ, DEBUG='0', PYTHONUNBUFFERED='1', PYTHONPATH=str(ROOT / 'training/.deps'),
        SUPER_READER_PROJECT_ROOT=str(ROOT), PYTORCH_ENABLE_MPS_FALLBACK='0',
        PYTORCH_MPS_FAST_MATH='0', PYTORCH_MPS_PREFER_METAL='0')

    def execute(stage, command, *, training=False):
        def run_once(current):
            nonlocal child
            with (run / f'{stage}.log').open('ab') as log:
                child = subprocess.Popen(current, cwd=snapshot, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                status(stage, child_pid=child.pid, log=str(run / f'{stage}.log'), command=current)
                return child.wait()
        run_with_recovery('training' if training else stage, command, run_once=run_once,
            checkpoint=run / 'layers20/training-state.pt', should_stop=lambda: stopped,
            on_refresh=lambda: status('refreshing-worker', reason='Saved next batch before MPS worker refresh'))

    helper = [sys.executable, str(snapshot / 'training/analyze_depth_comparison.py'), '--run-dir', str(run)]
    awake = subprocess.Popen(['/usr/bin/caffeinate', '-is', '-w', str(os.getpid())], stdin=subprocess.DEVNULL)
    try:
        status('verifying-frozen-inputs')
        for name, expected in plan['input_hashes'].items():
            if stopped:
                raise InterruptedError('Stopped during input verification')
            if sha(Path(name)) != expected:
                raise ValueError('Frozen input changed: ' + name)
        if not (run / 'preflight/verification.json').exists():
            execute('preflight', helper + ['--preflight'])
        if not (run / 'preflight/browser-benchmark.json').exists():
            execute('preflight-browser', ['node', str(snapshot / 'training/benchmark_width.mjs'),
                                        str(run / 'preflight'), '--depth'])
        if not (run / 'layers20/validation-only.json').exists():
            execute('train-layers20', training_command(snapshot, run, plan), training=True)
        control = read(run / 'layers16/validation-only.json')
        candidate = read(run / 'layers20/validation-only.json')
        expected_config = {**control['configuration'], 'residual_blocks': 10}
        if candidate['configuration'] != expected_config or candidate['data_identity'] != control['data_identity']:
            raise ValueError('Comparison changed variables other than depth')
        if candidate['data_sizes'] != control['data_sizes'] or len(candidate['history']) != 1:
            raise ValueError('Candidate did not complete the same full epoch/validation')
        for key in ('accuracy', 'loss'):
            difference = abs(candidate['initialization']['validation_before_training'][key]
                             - control['initialization']['validation_before_training'][key])
            if difference > 1e-7:
                raise ValueError('Starting full-validation predictions differ: ' + key)
        selection = select_depth(control, candidate)
        if (run / 'selection.json').exists() and read(run / 'selection.json') != selection:
            raise ValueError('Previously frozen selection changed')
        write(run / 'selection.json', selection)
        winner = selection['winner']
        if winner == 'layers20' and not (run / 'layers20/smoke-metrics.json').exists():
            execute('test-layers20', training_command(snapshot, run, plan, finalize=True), training=True)
        selected = read(run / winner / 'smoke-metrics.json')
        if selected['best_epoch'] != 1 or selected['data_sizes']['test'] != plan['evaluation']['test']['count']:
            raise ValueError('Selected test must cover the full split at epoch 1')
        if not (run / 'validation-strata.json').exists():
            execute('validation-strata', helper)
        if not (run / 'browser-benchmark.json').exists():
            execute('browser-benchmark', ['node', str(snapshot / 'training/benchmark_width.mjs'), str(run), '--depth'])
        write(run / 'comparison.json', {
            'training_samples_per_arm': plan['training_samples_per_arm'], 'full_holdouts': plan['evaluation'],
            'endpoints': {'layers16': control, 'layers20': candidate}, 'control_reused': True,
            'selection': selection, 'selected_test': selected['test'], 'selected_test_reused': winner == 'layers16',
            'validation_strata': read(run / 'validation-strata.json'),
            'browser_benchmark': read(run / 'browser-benchmark.json'),
            'backend_unchanged': sha(ROOT / 'src/boundary-model-data.js') == plan['backend_sha256_at_start'],
        })
        status('complete', winner=winner, report=str(run / 'comparison.json'), test_accuracy=selected['test']['accuracy'])
    except InterruptedError as error:
        status('stopped', reason=str(error))
    except BaseException as error:
        status('failed', error=str(error))
        raise
    finally:
        awake.terminate()
        awake.wait()


if __name__ == '__main__':
    main()
