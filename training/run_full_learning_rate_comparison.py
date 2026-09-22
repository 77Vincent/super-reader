#!/usr/bin/env python3
"""Replay the bundled model's full training epoch, changing only learning rate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shutil
import sys

from run_learning_rate_comparison import optimization_signature
from run_web_continuation import read, write, sha

ROOT = Path(os.environ.get('SUPER_READER_PROJECT_ROOT', Path(__file__).resolve().parent.parent))
BASE = ROOT / 'training/artifacts/chinese-line-web-200m-16conv-v7-20260920'
RUN = ROOT / 'training/artifacts/learning-rate-192ch-full-246m-v7-20260923'
SOURCES = ['training/' + name for name in (
    'run_full_learning_rate_comparison.py', 'run_learning_rate_comparison.py', 'run_web_continuation.py',
    'run_sharded.py', 'run_smoke.py', 'train_sharded.py', 'train_smoke.py', 'text_policy.py',
    'text-policy.json', 'unicode-symbols.json', 'export_browser_model.py')]
OPTIMIZATION_KEYS = ('learning_rate', 'batch_size', 'max_tokens_per_batch',
                     'domain_weight_power', 'selection_macro_weight', 'gradient_clip')


def validate_control(configuration, metrics, manifest):
    expected = {'channels': 192, 'residual_blocks': 8, 'learning_rate': 0.0003,
                'max_shards': 0, 'validation_limit': 0, 'test_limit': 0}
    if any(configuration.get(key) != value for key, value in expected.items()):
        raise ValueError('Expected the uncapped 16-layer, 192-channel, 0.0003 control')
    if metrics['seed'] != configuration['seed']:
        raise ValueError('Reference seed differs between checkpoint and metrics')
    if metrics['best_epoch'] != 1 or len(metrics['history']) != 1 or metrics['history'][0]['epoch'] != 1:
        raise ValueError('Expected a completed one-epoch reference selected at epoch 1')
    if metrics['data_sizes']['train'] != manifest['statistics']['samples']:
        raise ValueError('Reference training size differs from the full manifest')
    if metrics['data_sizes']['train'] != 246_266_210:
        raise ValueError('Expected the 200-million-web plus 46,266,210-other control')
    if metrics['history'][0]['validation_accuracy'] != metrics['validation']['accuracy']:
        raise ValueError('Reference report does not describe its equal-update endpoint')


def prepare(run, base):
    from run_smoke import ensure_dependencies
    ensure_dependencies()
    import torch
    from train_sharded import expanded_initialization
    from train_smoke import BoundaryChooser

    baseline = read(base / 'run.json')
    release = base / 'epoch-1-backend'
    verification = read(release / 'release-verification.json')
    if baseline['epochs'] != 1 or read(base / 'status.json')['stage'] != 'complete':
        raise ValueError('Full-data reference must have completed exactly one epoch')
    bundle_digest = sha(ROOT / 'src/boundary-model-data.js')
    if bundle_digest != verification['files']['boundary-model-data.js']['sha256']:
        raise ValueError('Current backend no longer matches the intended reference release')
    control_checkpoint = release / 'training-state.pt'
    if sha(control_checkpoint) != verification['files']['training-state.pt']['sha256']:
        raise ValueError('Reference release checkpoint changed')
    state = torch.load(control_checkpoint, map_location='cpu', weights_only=True)
    if (state['progress']['epoch'], state['progress']['shard'], state['progress']['next_batch']) != (2, 0, 0):
        raise ValueError('Reference checkpoint is not a completed one-epoch endpoint')
    if not all(torch.equal(t, state['best_state'][name]) for name, t in state['model_state'].items()):
        raise ValueError('Reference release is not the final equal-update weights')
    configuration = state['configuration']
    identity = state['data_identity']
    metrics = read(release / 'smoke-metrics.json')
    data = Path(baseline['data'])
    manifest_path = data / 'manifest.json'
    manifest = read(manifest_path)
    validate_control(configuration, metrics, manifest)
    if identity != verification['data_identity'] or sha(manifest_path) != identity['manifest_sha256']:
        raise ValueError('Reference manifest identity changed')
    if sha(data / 'summary.json') != identity['summary_sha256']:
        raise ValueError('Reference evaluation metadata changed')
    if state['history'] != metrics['history']:
        raise ValueError('Reference metrics and checkpoint history differ')

    for name, digest in baseline['source_hashes'].items():
        if sha(base / 'source' / name) != digest:
            raise ValueError('Original training snapshot changed: ' + name)
    # Only the endpoint-reporting flag/finalization may differ. The optimizer,
    # sampling, batch order, transfer, loss and checkpoint loop must be identical.
    if optimization_signature((ROOT / 'training/train_sharded.py').read_text()) != optimization_signature(
            (base / 'source/training/train_sharded.py').read_text()):
        raise ValueError('Trainer optimization differs from the original full-data run')
    for name in SOURCES:
        origin = (base / 'source' / name if name in baseline['source_hashes']
                  and name != 'training/train_sharded.py' else ROOT / name)
        destination = run / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, destination)
    link = run / 'source/training/.deps'
    if not link.exists():
        link.symlink_to(ROOT / 'training/.deps', target_is_directory=True)

    # This is the OLD epoch-2 initialization of the full-data reference, not
    # its trained endpoint and not either of the newer 10-million-pair pilots.
    initialization = base / 'initialization.pt'
    if sha(initialization) != baseline['initialization_sha256']:
        raise ValueError('Original full-data initialization changed')
    shutil.copy2(initialization, run / 'initialization.pt')
    shutil.copy2(manifest_path, run / 'manifest.json')
    vocabulary_path = Path(manifest['vocabulary_path'])
    if not vocabulary_path.is_absolute():
        vocabulary_path = ROOT / vocabulary_path
    vocabulary = read(vocabulary_path)
    shutil.copy2(vocabulary_path, run / 'vocabulary.json')
    initial = torch.load(initialization, map_location='cpu', weights_only=True)
    if initial['vocabulary'] != vocabulary or state['vocabulary'] != vocabulary:
        raise ValueError('Both starting and reference vocabularies must match the full dataset')
    model = BoundaryChooser(len(vocabulary), 192, 8)
    transfer = expanded_initialization(model, run / 'initialization.pt', vocabulary)
    if not all(torch.equal(value, initial['best_state'][name]) for name, value in model.state_dict().items()):
        raise ValueError('Transfer changed the initial weights')
    write(run / 'initialization-verification.json', {
        'passed': True, 'all_parameters_equal_original_best_state': True,
        'source_best_epoch': initial['best_epoch'], 'parameters': sum(p.numel() for p in model.parameters()),
        'optimizer': 'fresh AdamW, as in the original full-data control', 'transfer': transfer,
    })
    write(run / 'control.json', {
        'initial_validation': metrics['initialization']['validation_before_training'],
        'initial_selection_score': metrics['initialization']['selection_score_before_training'],
        'initial_test': None, 'endpoint_validation': metrics['validation'], 'endpoint_test': metrics['test'],
        'history': metrics['history'], 'parameter_count': metrics['parameter_count'],
        'release': str(release), 'configuration': configuration, 'data_identity': identity,
    })
    files = [run / name for name in ('initialization.pt', 'manifest.json', 'vocabulary.json',
                                     'control.json', 'initialization-verification.json')]
    files += [control_checkpoint, release / 'smoke-metrics.json', vocabulary_path, data / 'summary.json']
    inputs = {str(path): sha(path) for path in files}
    evaluation = {}
    for split in ('validation', 'test'):
        path = data / f'{split}.jsonl'
        if path.stat().st_size != identity[f'{split}_bytes']:
            raise ValueError('Reference holdout size changed: ' + split)
        digest = sha(path)
        # The data expansion kept these holdouts byte-identical to the preceding run.
        if digest != sha(Path(baseline['old_evaluation']) / f'{split}.jsonl'):
            raise ValueError('Expanded-run holdout differs from the fixed prior split')
        inputs[str(path)] = digest
        evaluation[split] = {'count': metrics['data_sizes'][split], 'bytes': path.stat().st_size, 'sha256': digest}
    supplied_hashes = 0
    for index, shard in enumerate(manifest['shards']):
        path = Path(shard['path'])
        if not path.is_absolute():
            path = ROOT / path
        if path.stat().st_size != shard['bytes']:
            raise ValueError('Shard byte count differs from reference manifest: ' + str(path))
        digest = sha(path)
        if 'sha256' in shard:
            supplied_hashes += 1
            if digest != shard['sha256']:
                raise ValueError('Shard content differs from reference manifest: ' + str(path))
        inputs[str(path)] = digest
        if (index + 1) % 256 == 0:
            print(f'Hashed {index + 1}/{len(manifest["shards"])} full-data shards', flush=True)
    write(run / 'run.json', {
        'created_at': datetime.now(timezone.utc).isoformat(), 'mode': 'full-data-learning-rate-comparison',
        'reference_run': str(base), 'reference_release': str(release), 'control_reused': True,
        'rates': {'lr-3e-5': 0.00003}, 'control_learning_rate': configuration['learning_rate'],
        'seed': configuration['seed'], 'epochs_per_arm': 1, 'manifest_name': 'manifest.json',
        'checkpoint_shards': 4, 'training_samples_per_arm': manifest['statistics']['samples'],
        'training': {key: configuration[key] for key in OPTIMIZATION_KEYS},
        'control_configuration': configuration, 'control_data_identity': identity,
        'evaluation_dir': str(data), 'evaluation': evaluation,
        'initialization_source': str(initialization), 'initialization_sha256': baseline['initialization_sha256'],
        'source_hashes': {name: sha(run / 'source' / name) for name in SOURCES},
        'input_hashes': inputs, 'optimization_matches_control': True,
        'shard_verification': {'total': len(manifest['shards']), 'historical_sha256_checked': supplied_hashes,
                               'all_current_contents_hashed': True, 'all_historical_byte_counts_checked': True},
        'backend_sha256_at_start': bundle_digest,
        'test_protocol': 'Select on full validation, then test the selected new candidate on the full fixed test split; otherwise reuse the measured control test. Never automatically promote.',
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
    if plan.get('mode') != 'full-data-learning-rate-comparison' or plan['reference_run'] != str(args.reference_run.resolve()):
        raise ValueError('Resume must use the original full-data comparison')
    fcntl.flock(lock, fcntl.LOCK_UN)
    os.environ['SUPER_READER_PROJECT_ROOT'] = str(ROOT)
    os.execv(sys.executable, [sys.executable, str(run / 'source/training/run_learning_rate_comparison.py'),
                             '--run-dir', str(run), '--resume'])


if __name__ == '__main__':
    main()
