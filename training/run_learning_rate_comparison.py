#!/usr/bin/env python3
"""Compare two lower learning rates against the frozen 192-channel pilot."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import fcntl
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from run_web_continuation import read, write, sha, run_with_recovery

ROOT = Path(os.environ.get('SUPER_READER_PROJECT_ROOT', Path(__file__).resolve().parent.parent))
BASE = ROOT / 'training/artifacts/width-192-vs-256-10m-v7-20260922'
RUN = ROOT / 'training/artifacts/learning-rate-192ch-10m-v7-20260922'
RATES = {'lr-1e-4': 0.0001, 'lr-3e-5': 0.00003}
SOURCES = ['training/' + name for name in (
    'run_learning_rate_comparison.py', 'run_web_continuation.py', 'run_sharded.py',
    'run_smoke.py', 'train_sharded.py', 'train_smoke.py', 'text_policy.py',
    'text-policy.json', 'unicode-symbols.json')]


def optimization_signature(source):
    """Ignore only the new reporting flag and post-training finalization."""
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'parse_arguments':
            node.body = [item for item in node.body if not (
                isinstance(item, ast.Expr) and isinstance(item.value, ast.Call)
                and isinstance(item.value.func, ast.Attribute) and item.value.func.attr == 'add_argument'
                and item.value.args and isinstance(item.value.args[0], ast.Constant)
                and item.value.args[0].value == '--defer-test')]
        if isinstance(node, ast.FunctionDef) and node.name == 'main':
            end = next(i for i, item in enumerate(node.body) if isinstance(item, ast.If)
                       and ast.unparse(item.test) == 'best_state is None')
            node.body = node.body[:end + 1]
    return ast.dump(module, include_attributes=False)


def training_command(snapshot, run, plan, arm, *, finalize=False):
    command = [sys.executable, str(snapshot / 'training/run_sharded.py'),
        '--manifest', str(run / 'pilot-manifest.json'), '--data-dir', plan['evaluation_dir'],
        '--artifact-dir', str(run / arm), '--epochs', '1', '--channels', '192',
        '--residual-blocks', '8', '--seed', str(plan['seed']), '--checkpoint-shards', '1',
        '--threads', '1', '--interop-threads', '1', '--device', 'mps']
    for key, value in {**plan['training'], 'learning_rate': plan['rates'][arm]}.items():
        command += ['--' + key.replace('_', '-'), str(value)]
    if not finalize:
        command.append('--defer-test')
    if (run / arm / 'training-state.pt').exists():
        command.append('--resume')
    else:
        if finalize:
            raise FileNotFoundError('Selected candidate has no checkpoint')
        command += ['--initialize-from', str(run / 'initialization.pt')]
    return command


def select_candidate(initial_score, endpoints):
    """Validation only; ties retain inherited weights or the earlier arm."""
    if not math.isfinite(initial_score):
        raise ValueError('Non-finite initial validation score')
    winner, score = 'initial', initial_score
    for arm, metrics in endpoints.items():
        candidate = metrics['selection_score']
        if not math.isfinite(candidate):
            raise ValueError('Non-finite endpoint validation score')
        if candidate > score:
            winner, score = arm, candidate
    return {'winner': winner, 'selection_score': score, 'criterion': '0.5 overall + 0.5 macro-domain validation accuracy'}


def prepare(run, base):
    baseline = read(base / 'run.json')
    if read(base / 'status.json')['stage'] != 'complete':
        raise ValueError('The reference pilot must be complete')
    if baseline['training']['learning_rate'] != 0.0003 or baseline['epochs_per_arm'] != 1:
        raise ValueError('Expected the completed one-epoch 0.0003 control')
    for name, digest in baseline['source_hashes'].items():
        if sha(base / 'source' / name) != digest:
            raise ValueError('Reference source changed: ' + name)
    for name in SOURCES:
        if name == 'training/train_sharded.py':
            if optimization_signature((ROOT / name).read_text()) != optimization_signature((base / 'source' / name).read_text()):
                raise ValueError('Optimization code differs from the control')
        elif name in baseline['source_hashes'] and sha(ROOT / name) != baseline['source_hashes'][name]:
            raise ValueError('Training dependency differs from the control: ' + name)
        destination = run / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    dependency_link = run / 'source/training/.deps'
    if not dependency_link.exists():
        dependency_link.symlink_to(ROOT / 'training/.deps', target_is_directory=True)
    for original, name, expected in [
        (base / 'initialization-192.pt', 'initialization.pt', baseline['initialization_sha256']),
        (base / 'vocabulary.json', 'vocabulary.json', baseline['vocabulary_sha256'])]:
        if sha(original) != expected:
            raise ValueError('Reference input changed: ' + str(original))
        shutil.copy2(original, run / name)
    # Keep the exact manifest bytes, shard order and vocabulary reference.
    shutil.copy2(base / 'pilot-manifest.json', run / 'pilot-manifest.json')
    pilot = read(run / 'pilot-manifest.json')
    selected = read(base / 'ch192/smoke-metrics.json')
    endpoint = read(base / 'ch192/final/smoke-metrics.json')
    if endpoint['architecture'] != {
        'channels': 192, 'residual_blocks': 8, 'convolutions_per_block': 2, 'kernel_size': 3, 'dilation': 1}:
        raise ValueError('Expected the 192-channel, 16-convolution control')
    if endpoint['data_sizes']['train'] != pilot['statistics']['samples'] or len(endpoint['history']) != 1:
        raise ValueError('Reference endpoint is incomplete')
    write(run / 'control.json', {
        'initial_validation': selected['initialization']['validation_before_training'],
        'initial_selection_score': selected['initialization']['selection_score_before_training'],
        'initial_test': selected['test'] if selected['best_epoch'] == 0 else None,
        'endpoint_validation': endpoint['validation'], 'endpoint_test': endpoint['test'],
        'history': endpoint['history'], 'parameter_count': endpoint['parameter_count']})
    files = [run / name for name in ('initialization.pt', 'vocabulary.json', 'pilot-manifest.json', 'control.json')]
    files += [Path(pilot['vocabulary_path'])]
    evaluation = Path(pilot['evaluation_source_dir'])
    if sha(evaluation / 'summary.json') != baseline['evaluation_summary_sha256']:
        raise ValueError('Evaluation metadata differs from the control')
    files.append(evaluation / 'summary.json')
    expected_files = {str(path): sha(path) for path in files}
    for split, details in baseline['evaluation'].items():
        expected_files[str(evaluation / f'{split}.jsonl')] = details['sha256']
    for shard in pilot['shards']:
        expected_files[shard['path']] = shard['sha256']
    write(run / 'run.json', {
        'created_at': datetime.now(timezone.utc).isoformat(), 'reference_run': str(base),
        'rates': RATES, 'control_learning_rate': baseline['training']['learning_rate'],
        'seed': baseline['seed'], 'epochs_per_arm': 1,
        'training_samples_per_arm': pilot['statistics']['samples'],
        'training': baseline['training'], 'evaluation_dir': str(evaluation), 'evaluation': baseline['evaluation'],
        'initialization_source': baseline['initialization_source'],
        'source_hashes': {name: sha(run / 'source' / name) for name in SOURCES},
        'input_hashes': expected_files, 'optimization_matches_control': True,
        'backend_sha256_at_start': sha(ROOT / 'src/boundary-model-data.js'),
        'test_protocol': 'Select by full validation first; evaluate only the selected new candidate on full test. Reuse already measured control/initial test results.'})


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
    if Path(__file__).resolve() != snapshot / 'training/run_learning_rate_comparison.py':
        fcntl.flock(lock, fcntl.LOCK_UN)
        os.environ['SUPER_READER_PROJECT_ROOT'] = str(ROOT)
        os.execv(sys.executable, [sys.executable, str(snapshot / 'training/run_learning_rate_comparison.py'),
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

    def execute(stage, command, checkpoint):
        def run_once(current_command):
            nonlocal child
            with (run / f'{stage}.log').open('ab') as log:
                child = subprocess.Popen(current_command, cwd=snapshot, env=environment,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                status(stage, child_pid=child.pid, log=str(run / f'{stage}.log'), command=current_command)
                return child.wait()
        run_with_recovery('training', command, run_once=run_once, checkpoint=checkpoint,
            should_stop=lambda: stopped,
            on_refresh=lambda: status('refreshing-worker', arm=stage, reason='MPS memory pressure; saved next batch'))

    awake = subprocess.Popen(['/usr/bin/caffeinate', '-is', '-w', str(os.getpid())], stdin=subprocess.DEVNULL)
    try:
        status('verifying-frozen-inputs')
        for name, expected in plan['input_hashes'].items():
            if stopped:
                raise InterruptedError('Stopped during input verification')
            if sha(Path(name)) != expected:
                raise ValueError('Frozen input changed: ' + name)
        control = read(run / 'control.json')
        endpoints = {'lr-3e-4-control': {
            'learning_rate': plan['control_learning_rate'], 'validation': control['endpoint_validation'],
            **control['history'][0], 'reused_reference_result': True}}
        for arm, rate in plan['rates'].items():
            report = run / arm / 'validation-only.json'
            if not report.exists():
                execute('train-' + arm, training_command(snapshot, run, plan, arm), run / arm / 'training-state.pt')
            if not report.exists():
                raise InterruptedError('Training or validation stopped; resume ' + arm)
            metrics = read(report)
            if len(metrics['history']) != 1 or metrics['data_sizes'] != {
                'train': plan['training_samples_per_arm'], 'validation': plan['evaluation']['validation']['count']}:
                raise ValueError('Incomplete arm: ' + arm)
            if abs(metrics['initialization']['selection_score_before_training'] - control['initial_selection_score']) > 1e-7:
                raise ValueError('Inherited validation predictions differ from the control')
            endpoints[arm] = {'learning_rate': rate, 'validation': metrics['endpoint_validation'],
                **metrics['history'][0], 'selection_score': metrics['endpoint_selection_score'],
                'selected_best_epoch': metrics['best_epoch']}
            write(run / 'validation-progress.json', {'endpoints': endpoints, 'test_deferred': True})
        selection = select_candidate(control['initial_selection_score'], endpoints)
        selection_path = run / 'selection.json'
        if selection_path.exists() and read(selection_path) != selection:
            raise ValueError('Previously frozen validation selection changed')
        write(selection_path, selection)
        winner = selection['winner']
        if stopped:
            raise InterruptedError('Stopped before selected test evaluation')
        if winner in plan['rates']:
            selected_path = run / winner / 'smoke-metrics.json'
            if not selected_path.exists():
                execute('test-' + winner, training_command(snapshot, run, plan, winner, finalize=True),
                    run / winner / 'training-state.pt')
            metrics = read(selected_path)
            if metrics['best_epoch'] != 1 or metrics['data_sizes']['test'] != plan['evaluation']['test']['count']:
                raise ValueError('Selected checkpoint or full test size differs')
            test = metrics['test']
            reused_test = False
        else:
            test = control['initial_test'] if winner == 'initial' else control['endpoint_test']
            if test is None:
                raise ValueError('Reference has no test result for the selected initial weights')
            reused_test = True
        result = {'training_samples_per_arm': plan['training_samples_per_arm'],
            'full_holdouts': plan['evaluation'], 'initial_validation': control['initial_validation'],
            'initial_selection_score': control['initial_selection_score'], 'endpoints': endpoints,
            'selection': selection, 'selected_test': test, 'selected_test_reused': reused_test,
            'backend_unchanged': sha(ROOT / 'src/boundary-model-data.js') == plan['backend_sha256_at_start']}
        write(run / 'comparison.json', result)
        status('complete', winner=winner, report=str(run / 'comparison.json'), test_accuracy=test['accuracy'])
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
