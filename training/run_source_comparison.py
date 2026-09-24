#!/usr/bin/env python3
"""Resume a matched current-mixture / Wikipedia+synthetic Metal pilot."""
import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from prepare_source_comparison import ARMS, prepare
from run_learning_rate_comparison import training_command as base_command, select_candidate
from run_web_continuation import read, write, sha, run_with_recovery

ROOT = Path(os.environ.get('SUPER_READER_PROJECT_ROOT', Path(__file__).resolve().parent.parent))
REFERENCE = ROOT / 'training/artifacts/learning-rate-192ch-full-246m-v7-20260923'
BEST = REFERENCE / 'epoch-1-backend'
RUN = ROOT / 'training/artifacts/source-mix-vs-wiki-synthetic-10m-v7-20260924'
SEED = 2026092407
SOURCES = ['training/' + name for name in (
    'run_source_comparison.py', 'prepare_source_comparison.py', 'run_width_comparison.py',
    'run_learning_rate_comparison.py', 'run_web_continuation.py', 'run_sharded.py',
    'run_smoke.py', 'train_sharded.py', 'train_smoke.py', 'text_policy.py',
    'text-policy.json', 'unicode-symbols.json')]


def training_command(snapshot, run, plan, arm, *, finalize=False):
    return base_command(snapshot, run, {**plan, 'manifest_name': arm + '-manifest.json',
        'rates': dict.fromkeys(ARMS, plan['training']['learning_rate'])}, arm, finalize=finalize)


def freeze(run, target, shard_count):
    reference = read(REFERENCE / 'run.json')
    metrics = read(BEST / 'smoke-metrics.json')
    if metrics['architecture'] != {'channels': 192, 'residual_blocks': 8,
        'convolutions_per_block': 2, 'kernel_size': 3, 'dilation': 1}:
        raise ValueError('Expected the current 16-convolution 192-channel release')
    if target <= 0 or shard_count <= 0 or target < shard_count:
        raise ValueError('Invalid sample/shard count')
    if shutil.disk_usage(run).free < 15 * 1024**3:
        raise RuntimeError('Need 15 GiB free for matched data and resumable states')
    for name in SOURCES:
        destination = run / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    link = run / 'source/training/.deps'
    if not link.exists():
        link.symlink_to(ROOT / 'training/.deps', target_is_directory=True)
    for original, name in [(BEST / 'training-state.pt', 'initialization.pt'),
                           (BEST / 'boundary-smoke-vocabulary.json', 'vocabulary.json'),
                           (BEST / 'smoke-metrics.json', 'initial-metrics.json'),
                           (REFERENCE / 'manifest.json', 'base-manifest.json')]:
        shutil.copy2(original, run / name)
    evaluation = Path(reference['evaluation_dir'])
    inputs = {str(run / name): sha(run / name) for name in
              ('initialization.pt', 'vocabulary.json', 'base-manifest.json', 'initial-metrics.json')}
    inputs[str(evaluation / 'summary.json')] = sha(evaluation / 'summary.json')
    for split, details in reference['evaluation'].items():
        inputs[str(evaluation / (split + '.jsonl'))] = details['sha256']
    write(run / 'run.json', {
        'created_at': datetime.now(timezone.utc).isoformat(), 'seed': SEED,
        'training_samples_per_arm': target, 'shards_per_arm': shard_count,
        'arms': list(ARMS), 'epochs_per_arm': 1, 'checkpoint_shards': 1,
        'initialization_source': str(BEST), 'initialization_sha256': inputs[str(run / 'initialization.pt')],
        'evaluation_dir': str(evaluation), 'evaluation': reference['evaluation'],
        'training': {'learning_rate': .00003, 'batch_size': 512, 'max_tokens_per_batch': 8192,
                     'domain_weight_power': .65, 'selection_macro_weight': .5, 'gradient_clip': 1.0},
        'source_hashes': {name: sha(run / 'source' / name) for name in SOURCES}, 'input_hashes': inputs,
        'backend_sha256_at_start': sha(ROOT / 'src/boundary-model-data.js'),
        'scope': 'continued training source-composition ablation; both arms inherit prior mixed-source knowledge',
        'controls': 'same initial weights, fresh AdamW, lr, epochs, vocabulary, cleaning policy, exact length and position-bin counts, shard batch counts, full holdouts',
        'limitations': ['Original proxy glyphs and document IDs are absent from compact training rows; proxy types cannot be matched.',
                       'Source frequencies change, so the unchanged inverse-frequency weighting formula yields different domain weights.',
                       'Source/topic/style changes cannot be separated from teacher quality in this comparison.',
                       'Both candidate sets come from existing training data; this is not training from scratch.',
                       'The existing validation and test have been consulted in previous experiments.'],
        'test_protocol': 'Freeze validation comparison and selection first, then evaluate both arms best states on the same full test for reporting; no reselection using test.',
        'deployment': 'No automatic backend replacement; review domain results and real reading cuts first.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=RUN)
    parser.add_argument('--target-samples', type=int, default=10_000_000)
    parser.add_argument('--shards', type=int, default=128)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / 'run.json').exists() and not args.resume:
        raise FileExistsError('Experiment exists; use --resume')
    if not (run / 'run.json').exists():
        freeze(run, args.target_samples, args.shards)
    plan = read(run / 'run.json')
    if (args.target_samples, args.shards) != (plan['training_samples_per_arm'], plan['shards_per_arm']):
        raise ValueError('Resume with the original sample and shard counts')
    snapshot = run / 'source'
    for name, expected in plan['source_hashes'].items():
        if sha(snapshot / name) != expected:
            raise ValueError('Frozen source changed: ' + name)
    if Path(__file__).resolve() != snapshot / 'training/run_source_comparison.py':
        fcntl.flock(lock, fcntl.LOCK_UN)
        os.environ['SUPER_READER_PROJECT_ROOT'] = str(ROOT)
        os.execv(sys.executable, [sys.executable, str(snapshot / 'training/run_source_comparison.py'),
            '--run-dir', str(run), '--target-samples', str(args.target_samples), '--shards', str(args.shards), '--resume'])
    child, stopped = None, False

    def stop(signum, _frame):
        nonlocal stopped
        if stopped:
            return
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

    def execute(stage, command, checkpoint):
        def run_once(current):
            nonlocal child
            with (run / (stage + '.log')).open('ab') as log:
                child = subprocess.Popen(current, cwd=snapshot, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                status(stage, child_pid=child.pid, log=str(run / (stage + '.log')), command=current)
                return child.wait()
        run_with_recovery('training', command, run_once=run_once, checkpoint=checkpoint,
            should_stop=lambda: stopped, on_refresh=lambda: status('refreshing-worker', arm=stage))

    awake = subprocess.Popen(['/usr/bin/caffeinate', '-is', '-w', str(os.getpid())], stdin=subprocess.DEVNULL)
    try:
        status('verifying-frozen-inputs')
        for path, expected in plan['input_hashes'].items():
            if stopped:
                raise InterruptedError('Stopped during input verification')
            if sha(Path(path)) != expected:
                raise ValueError('Frozen input changed: ' + path)
        if (run / 'comparison.json').exists():
            status('complete', report=str(run / 'comparison.json'))
            return
        prepare(run, read(run / 'base-manifest.json'), args.target_samples, plan['seed'],
                args.shards, lambda: stopped, status)
        initial = read(run / 'initial-metrics.json')
        v = initial['validation']
        initial_score = .5 * (v['accuracy'] + sum(v['per_domain_accuracy'].values()) / len(v['per_domain_accuracy']))
        endpoints = {}
        for arm in ARMS:
            report = run / arm / 'validation-only.json'
            if not report.exists():
                execute('train-' + arm, training_command(snapshot, run, plan, arm), run / arm / 'training-state.pt')
            if not report.exists():
                raise InterruptedError('Arm paused before endpoint validation completed: ' + arm)
            metrics = read(report)
            if metrics['data_sizes'] != {'train': args.target_samples, 'validation': plan['evaluation']['validation']['count']} or len(metrics['history']) != 1:
                raise ValueError('Incomplete arm or mismatched validation size')
            if abs(metrics['initialization']['selection_score_before_training'] - initial_score) > 1e-7:
                raise ValueError('Initial predictions differ from the released model')
            endpoints[arm] = {'validation': metrics['endpoint_validation'],
                              'selection_score': metrics['endpoint_selection_score'],
                              'selected_best_epoch': metrics['best_epoch'], 'history': metrics['history']}
            write(run / 'validation-progress.json', {'endpoints': endpoints, 'test_deferred': True})
        selection = select_candidate(initial_score, endpoints)
        if (run / 'selection.json').exists() and read(run / 'selection.json') != selection:
            raise ValueError('Frozen validation selection changed')
        write(run / 'selection.json', selection)
        tests = {}
        for arm in ARMS:
            # An arm retaining epoch 0 has exactly the inherited test result.
            if endpoints[arm]['selected_best_epoch'] == 0:
                tests[arm] = {'best_epoch': 0, 'test': initial['test'], 'reused_initial_test': True}
                continue
            report = run / arm / 'smoke-metrics.json'
            if not report.exists():
                execute('test-' + arm, training_command(snapshot, run, plan, arm, finalize=True), run / arm / 'training-state.pt')
            if not report.exists():
                raise InterruptedError('Paused during test finalization')
            metrics = read(report)
            if metrics['data_sizes']['test'] != plan['evaluation']['test']['count']:
                raise ValueError('Full test size differs')
            tests[arm] = {'best_epoch': metrics['best_epoch'], 'test': metrics['test'],
                          'training_weighting': metrics['training_weighting'], 'reused_initial_test': False}
        result = {'initial_validation': v, 'initial_test': initial['test'], 'selection': selection,
                  'endpoints': endpoints, 'best_state_tests': tests, 'data': read(run / 'data-verification.json'),
                  'limitations': plan['limitations'], 'backend_unchanged': sha(ROOT / 'src/boundary-model-data.js') == plan['backend_sha256_at_start']}
        write(run / 'comparison.json', result)
        status('complete', winner=selection['winner'], report=str(run / 'comparison.json'))
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
