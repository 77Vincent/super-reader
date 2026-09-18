#!/usr/bin/env python3
"""Rebuild all cached corpora under the current policy and train a separate candidate."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = ROOT / 'training/artifacts/unicode-context-192ch-12conv-20260913'
PREVIOUS = ROOT / 'training/artifacts/all-local-192ch-12conv-20260912/candidate'
PREVIOUS_MANIFEST = ROOT / 'training/data/processed/all-local-192ch-12conv-20260912-combined/manifest.json'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    part = path.with_name(path.name + '.part')
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    part.replace(path)


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def run_pipeline(run, resume):
    os.environ['DEBUG'] = '0'
    run.mkdir(parents=True, exist_ok=True)
    processed = ROOT / 'training/data/processed'
    base, wiki, combined, evaluation = [processed / (run.name + suffix) for suffix in ('-base', '-wiki', '-combined', '-eval')]
    candidate = run / 'candidate'
    plan = read(run / 'run.json')
    completed_path = run / 'completed-stages.json'
    completed = read(completed_path) if completed_path.exists() else {}

    def status(stage, **details):
        value = {'stage': stage, 'updated_at': datetime.now(timezone.utc).isoformat(),
                 'run_directory': str(run), 'coordinator_pid': os.getpid(), **details}
        write(run / 'status.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def execute(stage, command, partial_output=None):
        for name, expected in plan['source_hashes'].items():
            if sha(ROOT / name) != expected:
                raise RuntimeError(f'Source changed after launch: {name}')
        if stage in completed:
            return
        if partial_output and partial_output.exists():
            # Only this run's unfinished, non-resumable stage output is archived.
            backup = partial_output.with_name(partial_output.name + '.interrupted-' + datetime.now().strftime('%Y%m%dT%H%M%S'))
            partial_output.rename(backup)
        log = run / (stage + '.log')
        with log.open('a') as handle:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=handle, stderr=subprocess.STDOUT)
            status(stage, child_pid=child.pid, log=str(log))
            code = child.wait()
        if code:
            raise RuntimeError(f'{stage} exited {code}; see {log}')
        completed[stage] = {'command': command, 'finished_at': datetime.now(timezone.utc).isoformat()}
        write(completed_path, completed)

    try:
        if sha(run / 'initialization.pt') != plan['initialization_sha256']:
            raise RuntimeError('Initialization checkpoint changed')
        execute('preflight', [sys.executable, 'training/preflight_context.py', '--run-dir', str(run)], run / 'preflight')
        assert read(run / 'preflight.json')['passed']
        execute('prepare-base', ['node', '--max-old-space-size=8192', 'training/prepare_retraining_base.mjs',
                                '--output-dir', str(base), '--all-local-clue', '1'], base)
        # The inherited checkpoint has seen Han-only inputs. Match their projected
        # text/gaps too, before establishing this new evaluation population.
        legacy_manifests = [str(PREVIOUS_MANIFEST)] + [item['path'] for item in
            read(processed / 'no-enumeration-aligned-eval-20260912/summary.json')['training_manifests']]
        command = [sys.executable, 'training/filter_retraining_holdouts.py', '--data-dir', str(base),
                   '--output-dir', str(evaluation), '--han-projection']
        for path in legacy_manifests:
            command += ['--training-manifest', path]
        execute('protect-inherited-holdouts', command, evaluation)
        execute('prepare-wikipedia', ['node', '--max-old-space-size=4096', 'training/prepare_full_wikipedia_data.mjs',
                                     '--source-dir', str(base), '--output-dir', str(wiki), '--shards', '256',
                                     '--max-samples-per-document', '0', '--max-sequence-length', '0'])
        execute('prepare-synthetic', [sys.executable, 'training/run_prepare_synthetic.py',
                                     '--base-manifest', str(wiki / 'manifest.json'), '--output-dir', str(combined),
                                     '--source-manifest', str(PREVIOUS_MANIFEST), '--target-samples', '0',
                                     '--synthetic-shards', '128', '--max-samples-per-document', '0', '--max-sequence-length', '0'])
        manifest = read(combined / 'manifest.json')
        synthetic = read(combined / 'preparation-state.json')
        wiki_state = read(wiki / 'preparation-state.json')
        assert manifest['source_exhausted']
        assert synthetic['statistics']['documents_seen'] == sum(x['rows'] for x in manifest['synthetic_source']['downloads'])
        assert wiki_state['next_block'] == read(wiki / 'manifest.json')['blocks']
        assert manifest['statistics']['overlength_filtered'] == 0
        assert manifest['statistics']['document_sample_cap_filtered'] == 0
        write(run / 'corpus-coverage.json', {
            'training_samples': manifest['statistics']['samples'],
            'domain_samples': manifest['statistics']['domain_samples'],
            'training_tokens': manifest['statistics']['tokens'],
            'maximum_sequence_length': manifest['statistics']['maximum_sequence_length'],
            'wikipedia_blocks_processed': wiki_state['next_block'],
            'synthetic_rows_processed': synthetic['statistics']['documents_seen'],
            'clue': read(base / 'summary.json')['corpus_counts']})
        audit = run / 'training-overlap-audit'
        execute('audit-new-training', [sys.executable, 'training/filter_retraining_holdouts.py',
                                      '--data-dir', str(evaluation), '--output-dir', str(audit),
                                      '--training-manifest', str(combined / 'manifest.json')], audit)
        for split in ('validation', 'test'):
            assert sha(audit / (split + '.jsonl')) == sha(evaluation / (split + '.jsonl')), f'Training leaked into {split}'
        execute('build-vocabulary', [sys.executable, 'training/build_context_vocabulary.py',
                                    '--manifest', str(combined / 'manifest.json'),
                                    '--source-vocabulary', str(run / 'source-vocabulary.json'),
                                    '--output', str(combined / 'vocabulary.json'), '--maximum-size', '8192'])
        command = [sys.executable, 'training/run_sharded.py', '--manifest', str(combined / 'manifest.json'),
                   '--data-dir', str(evaluation), '--artifact-dir', str(candidate), '--epochs', '2',
                   '--channels', '192', '--residual-blocks', '6', '--learning-rate', '0.0003',
                   '--domain-weight-power', '0.65', '--selection-macro-weight', '0.5', '--gradient-clip', '1.0',
                   '--batch-size', '512', '--max-tokens-per-batch', '8192', '--threads', '1', '--interop-threads', '1', '--device', 'mps']
        state = candidate / 'training-state.pt'
        if state.exists():
            if not resume:
                raise FileExistsError('Training checkpoint already exists; use --resume')
            command += ['--resume']
            completed.pop('training', None)
        else:
            command += ['--initialize-from', str(run / 'initialization.pt')]
        execute('training', command)
        sys.path.insert(0, str(ROOT / 'training/.deps'))
        import torch
        checkpoint = torch.load(state, map_location='cpu', weights_only=True)
        if checkpoint['progress']['epoch'] < 3:
            completed.pop('training', None)
            write(completed_path, completed)
            status('training_stopped', progress=checkpoint['progress'])
            return
        execute('export', [sys.executable, 'training/export_browser_model.py', '--artifact-dir', str(candidate),
                           '--output', str(candidate / 'boundary-model-data.js')])
        execute('browser-evaluation', ['node', 'training/evaluate_model.mjs', '--input', str(evaluation / 'validation.jsonl'),
                                       '--model', str(candidate / 'boundary-model-data.js'),
                                       '--output', str(run / 'browser-evaluation.json')])
        metrics = read(candidate / 'smoke-metrics.json')
        status('complete', selected_epoch=metrics['best_epoch'], validation_accuracy=metrics['validation']['accuracy'],
               test_accuracy=metrics['test']['accuracy'], model=str(candidate / 'boundary-model-data.js'))
    except BaseException as error:
        status('failed', error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if not (run / 'run.json').exists():
        metrics = read(PREVIOUS / 'smoke-metrics.json')
        if metrics['best_epoch'] != 2:
            raise ValueError('Expected the completed best epoch-2 checkpoint; inspect the previous run')
        shutil.copy2(PREVIOUS / 'training-state.pt', run / 'initialization.pt')
        shutil.copy2(PREVIOUS / 'boundary-smoke-vocabulary.json', run / 'source-vocabulary.json')
        sources = [p for pattern in ('*.py', '*.mjs', '*.json') for p in (ROOT / 'training').glob(pattern)]
        sources += list((ROOT / 'src/backend').glob('*.js'))
        write(run / 'run.json', {'created_at':datetime.now(timezone.utc).isoformat(),
                                 'policy':read(ROOT/'training/text-policy.json'), 'epochs':2,
                                 'channels':192,'residual_blocks':6,'maximum_vocabulary_size':8192,
                                 'initialization':str(PREVIOUS/'training-state.pt'),
                                 'initialization_sha256':sha(run/'initialization.pt'),
                                 'previous_metrics':metrics,
                                 'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sources}})
    run_pipeline(args.run_dir.resolve(), args.resume)


if __name__ == '__main__':
    main()
