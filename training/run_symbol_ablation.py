#!/usr/bin/env python3
"""Paired Metal continuation smoke: context versus Han/ASCII letters/decimal digits.

Uses already cleaned A/B pairs. Proxy selection and source barriers do not change.
All new files are isolated in run-dir; no production model is exported or promoted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import unicodedata

from text_policy import DATA_POLICY, HAN, require_data_policy, valid_proxy_label

ROOT = Path(os.environ.get('SUPER_READER_PROJECT_ROOT', Path(__file__).resolve().parent.parent))
BASE = ROOT / 'training/artifacts/padding-fixed-web-350m-v7-20260925'
DATA = ROOT / 'training/data/processed/padding-fixed-web-350m-v7-20260925'
RUN = ROOT / 'training/artifacts/symbol-ablation-1m-20260929'
ARMS = ('context', 'text-only')
SEED = 2026092907
SOURCES = ('run_symbol_ablation.py', 'train_smoke.py', 'text_policy.py', 'text-policy.json', 'unicode-symbols.json')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + '.part')
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    part.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def retained(c):
    return bool(HAN.fullmatch(c)) or 'a' <= c <= 'z' or 'A' <= c <= 'Z' or c.isdecimal()


def project(text, target):
    """Target is a code-point gap, never an offset into encoded bytes/UTF-16."""
    if not 0 <= target < len(text) - 1:
        raise ValueError('Invalid source target')
    kept = [i for i, c in enumerate(text) if retained(c)]
    new_target = sum(i <= target for i in kept) - 1
    if not 0 <= new_target < len(kept) - 1:
        return None
    return ''.join(text[i] for i in kept), new_target, kept


def common_gaps(kept):
    # Only gaps whose two original neighbors both survive; no guessing where a
    # collapsed gap inside a deleted quote/punctuation run belongs in the source.
    return [(left, j) for j, (left, right) in enumerate(zip(kept, kept[1:])) if right == left + 1]


def key(text):
    return hashlib.sha256(text.encode()).digest()


def quotas(counts, total):
    population = sum(counts.values())
    result = {k: total * n // population for k, n in counts.items()}
    order = sorted(counts, key=lambda k: (-(total * counts[k] % population), k))
    for k in order[:total - sum(result.values())]:
        result[k] += 1
    return result


def reservoir_add(pool, row, seen, limit, rng):
    if len(pool) < limit:
        pool.append(row)
    else:
        index = rng.randrange(seen)
        if index < limit:
            pool[index] = row


def prepare(run, count, holdout_count, notify):
    manifest = read(DATA / 'manifest.json')
    require_data_policy(manifest)
    base_metrics = read(BASE / 'epoch-1-backend/smoke-metrics.json')
    architecture = base_metrics['architecture']
    if architecture['channels'] != 192 or architecture['residual_blocks'] != 8:
        raise ValueError('Unexpected initialization architecture')
    if shutil.disk_usage(run).free < 3 * 1024**3:
        raise RuntimeError('Need 3 GiB free for smoke data/checkpoints')
    for name in SOURCES:
        destination = run / 'source/training' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'training' / name, destination)
    dependency_link = run / 'source/training/.deps'
    if not dependency_link.exists():
        dependency_link.symlink_to(ROOT / 'training/.deps', target_is_directory=True)
    shutil.copy2(BASE / 'epoch-1-backend/training-state.pt', run / 'initialization.pt')
    write(run / 'base-manifest.json', manifest)
    initial_hash = sha(run / 'initialization.pt')
    groups = defaultdict(list)
    for shard in manifest['shards']:
        path = Path(shard['path'])
        path = path if path.is_absolute() else ROOT / path
        groups[str(path.parent)].append(path)
    group_counts = {}
    # Cumulative continuation manifests contain all prior sources. Subtract
    # their known populations to get the contribution of each own directory.
    for directory in groups:
        own_manifest = read(Path(directory) / 'manifest.json')
        parents = set()
        for shard in own_manifest['shards']:
            path = Path(shard['path'])
            path = path if path.is_absolute() else ROOT / path
            if str(path.parent) != directory:
                parents.add(str(path.parent))
        group_counts[directory] = own_manifest['statistics']['samples'] - sum(group_counts[p] for p in parents)
    if sum(group_counts.values()) != manifest['statistics']['samples']:
        raise ValueError('Source population mismatch')
    inputs, splits, forbidden = {}, {}, set()
    maximum_length = 256
    for split in ('validation', 'test'):
        pool, seen, rejected = [], 0, Counter()
        rng = random.Random(SEED + (1 if split == 'validation' else 2))
        path = DATA / (split + '.jsonl')
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for line_number, line in enumerate(handle):
                digest.update(line)
                r = json.loads(line)
                if not valid_proxy_label(r.get('punctuation')):
                    raise ValueError('Invalid source proxy in ' + str(path))
                text, target = ''.join(r['tokens']), r['target_index']
                projected = project(text, target)
                if len(text) > maximum_length or projected is None:
                    rejected['overlength_or_empty_side'] += 1
                    continue
                if key(projected[0]) in forbidden:
                    rejected['projected_input_in_validation'] += 1
                    continue
                seen += 1
                row = [text, target, r['domain'], 1., r['id'], r.get('document_id', r['id'])]
                reservoir_add(pool, row, seen, holdout_count, rng)
                if (line_number + 1) % 500000 == 0:
                    notify('preparing-' + split, rows_scanned=line_number + 1)
        if len(pool) != holdout_count:
            raise ValueError('Insufficient evaluation rows')
        inputs[str(path)] = digest.hexdigest()
        forbidden.update(key(project(r[0], r[1])[0]) for r in pool)
        output = run / 'data' / (split + '.jsonl')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in pool))
        splits[split] = {'count': len(pool), 'eligible_population': seen, 'rejected': dict(rejected),
                         'domains': dict(Counter(r[2] for r in pool))}
        notify('prepared-' + split, count=len(pool))

    domains = manifest['domains']
    counts = manifest['statistics']['domain_samples']
    raw_weights = [counts[d] ** -.65 for d in domains]
    mean = sum(counts[d] * w for d, w in zip(domains, raw_weights)) / sum(counts.values())
    domain_weights = [w / mean for w in raw_weights]
    positions = []
    for cell in manifest['statistics']['cells']:
        total, occupied = sum(cell), sum(n > 0 for n in cell)
        positions.append([total / occupied / n if n else 0. for n in cell])
    allocation = quotas(group_counts, count)
    shard_count = max(1, math.ceil(count / 50000))
    paths = [run / 'data' / f'train-{i:03d}.jsonl' for i in range(shard_count)]
    training = {'count': 0, 'weight_sum': 0., 'changed': 0, 'removed_characters': 0,
                'original_characters': 0, 'domain_counts': Counter(), 'sources': []}
    handles = [path.open('w') for path in paths]
    try:
        for source_index, (directory, quota) in enumerate(allocation.items()):
            rng = random.Random(SEED + 100 + source_index)
            pool, seen, rejected = [], 0, Counter()
            candidates = list(groups[directory])
            rng.shuffle(candidates)
            average = group_counts[directory] / len(candidates)
            minimum_shards = min(len(candidates), max(2, math.ceil(quota * 1.4 / average)))
            scanned = []
            for path in candidates:
                digest = hashlib.sha256()
                with path.open('rb') as handle:
                    for line_number, line in enumerate(handle):
                        digest.update(line)
                        text, target, domain, bucket, position = json.loads(line)
                        projected = project(text, target)
                        if len(text) > maximum_length or projected is None:
                            rejected['overlength_or_empty_side'] += 1
                            continue
                        if key(projected[0]) in forbidden:
                            rejected['projected_input_in_holdout'] += 1
                            continue
                        weight = positions[bucket][position] * domain_weights[domain]
                        row = [text, target, domains[domain], weight, f'{path}:{line_number}', '']
                        seen += 1
                        reservoir_add(pool, row, seen, quota, rng)
                inputs[str(path)] = digest.hexdigest()
                scanned.append(str(path))
                notify('sampling-training', source=source_index + 1, scanned_shards=len(scanned), selected=len(pool))
                if len(scanned) >= minimum_shards and len(pool) == quota:
                    break
            if len(pool) != quota:
                raise ValueError('Insufficient training rows in ' + directory)
            rng.shuffle(pool)
            for row in pool:
                text_only = project(row[0], row[1])[0]
                handles[training['count'] % shard_count].write(json.dumps(row, ensure_ascii=False) + '\n')
                training['count'] += 1
                training['weight_sum'] += row[3]
                training['changed'] += text_only != row[0]
                training['removed_characters'] += len(row[0]) - len(text_only)
                training['original_characters'] += len(row[0])
                training['domain_counts'][row[2]] += 1
            training['sources'].append({'directory': directory, 'population': group_counts[directory],
                'selected': quota, 'eligible_pool': seen, 'rejected': dict(rejected), 'shards': scanned})
    finally:
        for handle in handles:
            handle.close()
    splits['train'] = training
    data_hashes = {str(p.relative_to(run)): sha(p) for p in sorted((run / 'data').glob('*.jsonl'))}
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'seed': SEED,
        'arms': list(ARMS), 'epochs_per_arm': 1, 'architecture': architecture,
        'batch_size': 512, 'max_tokens_per_batch': 8192, 'learning_rate': 3e-5,
        'weight_decay': 1e-4, 'gradient_clip': 1., 'maximum_source_length': maximum_length,
        'batch_order': 'same source-length buckets, seed, sample order and batch IDs in both arms',
        'weight_mean': training['weight_sum'] / training['count'],
        'weighting': 'same frozen original length/position and domain^(-0.65) weights in both arms',
        'text_only': 'NFKC already applied by v7; retain Han, ASCII A-Z/a-z and Unicode decimal digits only; remove whitespace too',
        'sampling': 'proportional source quotas; seeded hash-shard pools then uniform reservoir within each pool',
        'splits': splits, 'data_hashes': data_hashes, 'input_hashes': inputs,
        'initialization_sha256': initial_hash, 'policy': DATA_POLICY,
        'source_hashes': {name: sha(run / 'source/training' / name) for name in SOURCES},
        'backend_sha256': sha(ROOT / 'src/boundary-model-data.js'),
        'evaluation_protocol': 'fixed one-epoch endpoints, not best-of-test; both validation endpoints before either test; common-original-gap metric plus native metrics',
        'limitations': ['single-seed short continuation from a context-trained model, not training from scratch',
            'holdouts reused historically; smoke subsets are not full holdouts',
            'new continuation excludes projected inputs in the sampled holdouts; inherited training was not re-audited under projection']}
    write(run / 'run.json', plan)
    return plan


def load_records(path, vocabulary, arm):
    result = []
    with Path(path).open() as handle:
        for index, line in enumerate(handle):
            text, target, domain, weight, sample_id, document_id = json.loads(line)
            projected, new_target, kept = project(text, target)
            current = text if arm == 'context' else projected
            result.append({'text': text, 'tokens': list(current), 'token_ids': [vocabulary.get(c, 1) for c in current],
                'target_index': target if arm == 'context' else new_target, 'source_target': target,
                'training_weight': weight, 'domain': domain, 'kept': kept,
                'changed': current != text if arm == 'text-only' else projected != text,
                'id': sample_id, 'document_id': document_id, 'row_index': index,
                'original_length': len(text)})
    return result


def matched_batches(records, seed, shuffle):
    from train_smoke import make_batch_indices
    lengths = [{'token_ids': range(r['original_length'])} for r in records]
    return make_batch_indices(lengths, 512, 8192, shuffle=shuffle, seed=seed)


def batches(records, indices, start=0):
    from train_smoke import iterate_batches, bucket_ceiling
    from torch.nn import functional as F
    for batch in iterate_batches(records, 512, 8192, shuffle=False, seed=0,
                                  batch_indices=indices, start_batch=start):
        # Same padded shape for paired batches, avoiding MPS graph-cache growth
        # from different shortened lengths. Intermediate padding is masked.
        width = bucket_ceiling(max(r['original_length'] for r in batch['records']))
        padding = width - batch['token_ids'].shape[1]
        if padding:
            for name in ('token_ids', 'token_mask', 'gap_mask'):
                batch[name] = F.pad(batch[name], (0, padding), value=0)
        yield batch


def evaluate_arm(model, records, arm, output, stop):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from train_smoke import batch_to_device, TrainingStopRequested
    model.eval()
    groups = defaultdict(lambda: {'count': 0, 'correct': 0, 'common_count': 0, 'common_correct': 0,
        'candidates': 0, 'random_sum': 0., 'common_candidates': 0, 'common_random_sum': 0.})
    native = np.zeros(len(records), dtype=np.int8)
    common = np.full(len(records), -1, dtype=np.int8)
    predictions = np.zeros(len(records), dtype=np.int32)
    common_predictions = np.full(len(records), -1, dtype=np.int32)
    total_loss = 0.
    with torch.inference_mode():
        for batch in batches(records, matched_batches(records, 0, False)):
            if stop():
                raise TrainingStopRequested
            batch = batch_to_device(batch, next(model.parameters()).device)
            scores = model(batch['token_ids'], batch['token_mask'], batch['gap_mask'])
            total_loss += F.cross_entropy(scores, batch['targets'], reduction='sum').item()
            scores = scores.cpu().numpy()
            for row, record in enumerate(batch['records']):
                index = record['row_index']
                prediction = int(scores[row].argmax())
                native[index] = prediction == record['target_index']
                predictions[index] = prediction
                gaps = common_gaps(record['kept'])
                eligible = any(g == record['source_target'] for g, _ in gaps)
                if eligible:
                    choice = max(gaps, key=lambda p: scores[row, p[0] if arm == 'context' else p[1]])
                    common[index] = choice[0] == record['source_target']
                    common_predictions[index] = choice[0]
                labels = ['all', 'changed' if record['changed'] else 'unchanged', 'domain:' + record['domain']]
                removed = [c for c in record['text'] if not retained(c)]
                for name, match in [('punctuation', lambda c: unicodedata.category(c).startswith('P')),
                                    ('whitespace', str.isspace),
                                    ('other', lambda c: not c.isspace() and not unicodedata.category(c).startswith('P'))]:
                    if any(match(c) for c in removed):
                        labels.append('removed:' + name)
                for label in labels:
                    g = groups[label]
                    g['count'] += 1
                    g['correct'] += int(native[index])
                    candidates = len(record['token_ids']) - 1
                    g['candidates'] += candidates
                    g['random_sum'] += 1 / candidates
                    if eligible:
                        g['common_count'] += 1
                        g['common_correct'] += int(common[index])
                        g['common_candidates'] += len(gaps)
                        g['common_random_sum'] += 1 / len(gaps)
            del batch, scores
    for g in groups.values():
        g['accuracy'] = g['correct'] / g['count']
        g['mean_candidates'] = g.pop('candidates') / g['count']
        g['random_accuracy'] = g.pop('random_sum') / g['count']
        n = g['common_count']
        g['common_accuracy'] = g['common_correct'] / n if n else None
        g['common_mean_candidates'] = g.pop('common_candidates') / n if n else None
        g['common_random_accuracy'] = g.pop('common_random_sum') / n if n else None
    metrics = {'loss': total_loss / len(records), 'groups': dict(groups), 'arm': arm}
    np.savez_compressed(str(output) + '.npz', native=native, common=common, predictions=predictions,
                        common_predictions=common_predictions)
    write(output.with_suffix('.json'), metrics)
    return metrics


def worker(run, arm, stage):
    # Torch imports happen only in short-lived Metal workers.
    import torch
    from torch.nn import functional as F
    from train_smoke import (BoundaryChooser, configure_cpu, configure_mps, seed_everything,
        batch_to_device, save_training_state, mps_memory_restart_needed, TrainingStopRequested)
    plan = read(run / 'run.json')
    configure_cpu(1, 1)
    device = configure_mps(.4)
    seed_everything(plan['seed'])
    initialization = torch.load(run / 'initialization.pt', map_location='cpu', weights_only=True)
    vocabulary = initialization['vocabulary']
    model = BoundaryChooser(len(vocabulary), 192, 8).to(device)
    model.load_state_dict(initialization['best_state'])
    del initialization
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'],
                                  weight_decay=plan['weight_decay'], foreach=True)
    output = run / arm
    output.mkdir(exist_ok=True)
    state_path = output / 'training-state.pt'
    if stage == 'test' and not state_path.exists():
        raise FileNotFoundError('Test requires the trained endpoint checkpoint')
    identity = sha(run / 'run.json')
    progress = {'shard': 0, 'batch': 0, 'seen': 0, 'updates': 0, 'training_seconds': 0., 'loss_sum': 0.}
    if state_path.exists():
        state = torch.load(state_path, map_location='cpu', weights_only=True)
        if state['identity'] != identity or state['arm'] != arm:
            raise ValueError('Resume configuration mismatch')
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        progress = state['progress']
        del state
    stopped = [False]
    def request_stop(*_):
        stopped[0] = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    def persist():
        save_training_state(state_path, {'format_version': 1, 'identity': identity, 'arm': arm,
            'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
            'progress': progress, 'vocabulary': vocabulary})
        write(output / 'progress.json', {**progress, 'pid': os.getpid(), 'stage': stage})
    try:
        if stage == 'test':
            if not (output / 'endpoint-validation.json').exists():
                raise ValueError('Test requires a completed endpoint')
            records = load_records(run / 'data/test.jsonl', vocabulary, arm)
            evaluate_arm(model, records, arm, output / 'test', lambda: stopped[0])
            return 0
        initial_path = output / 'initial-validation.json'
        if not initial_path.exists():
            if progress['seen']:
                raise ValueError('Missing initial validation on a resumed training run')
            records = load_records(run / 'data/validation.jsonl', vocabulary, arm)
            evaluate_arm(model, records, arm, output / 'initial-validation', lambda: stopped[0])
            del records
            torch.mps.empty_cache()
        shards = sorted((run / 'data').glob('train-*.jsonl'))
        random.Random(plan['seed']).shuffle(shards)
        for shard_index in range(progress['shard'], len(shards)):
            records = load_records(shards[shard_index], vocabulary, arm)
            indices = matched_batches(records, plan['seed'] + shard_index, True)
            model.train()
            for batch_index, batch in enumerate(batches(records, indices, progress['batch']), progress['batch']):
                if stopped[0]:
                    persist()
                    return 130
                started = time.perf_counter()
                batch = batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch['token_ids'], batch['token_mask'], batch['gap_mask'])
                loss = (F.cross_entropy(logits, batch['targets'], reduction='none') * batch['sample_weights']).mean() / plan['weight_mean']
                if not torch.isfinite(loss).item():
                    raise RuntimeError('Non-finite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), plan['gradient_clip'])
                optimizer.step()
                value = loss.item()
                progress['training_seconds'] += time.perf_counter() - started
                progress['loss_sum'] += value * len(batch['records'])
                progress['seen'] += len(batch['records'])
                progress['updates'] += 1
                progress['batch'] = batch_index + 1
                del logits, loss, batch
                if progress['updates'] % 100 == 0:
                    write(output / 'progress.json', {**progress, 'pid': os.getpid(), 'stage': stage})
                    print(arm, json.dumps(progress), flush=True)
                    if mps_memory_restart_needed(.4):
                        persist()
                        return 75
            progress['shard'], progress['batch'] = shard_index + 1, 0
            persist()
            del records, indices
            torch.mps.empty_cache()
        if progress['seen'] != plan['splits']['train']['count']:
            raise ValueError('Incomplete or repeated training exposure')
        records = load_records(run / 'data/validation.jsonl', vocabulary, arm)
        evaluate_arm(model, records, arm, output / 'endpoint-validation', lambda: stopped[0])
        return 0
    except TrainingStopRequested:
        persist()
        return 130


def compare(run):
    import numpy as np
    report = {'plan': read(run / 'run.json'), 'arms': {}, 'paired_test': {}}
    for arm in ARMS:
        report['arms'][arm] = {name: read(run / arm / (name + '.json')) for name in
            ('initial-validation', 'endpoint-validation', 'test', 'progress')}
    a = np.load(run / 'context/test.npz')
    b = np.load(run / 'text-only/test.npz')
    for metric in ('native', 'common'):
        mask = (a[metric] >= 0) & (b[metric] >= 0)
        delta = b[metric][mask].astype(float) - a[metric][mask]
        report['paired_test'][metric] = {'count': len(delta), 'text_only_minus_context': float(delta.mean()),
            'text_only_correct_context_wrong': int((delta == 1).sum()),
            'context_correct_text_only_wrong': int((delta == -1).sum())}
    if sha(ROOT / 'src/boundary-model-data.js') != report['plan']['backend_sha256']:
        raise ValueError('Production model changed during experiment')
    write(run / 'comparison.json', report)
    lines = ['# Symbol removal smoke', '', 'Fixed one-epoch endpoints; percentages are top-1 hidden-boundary recovery.', '',
        '| Split / population | Context | Text only | Difference (pp) |', '|---|---:|---:|---:|']
    for stage in ('initial-validation', 'endpoint-validation', 'test'):
        for subset in ('all', 'changed', 'unchanged'):
            for metric in ('accuracy', 'common_accuracy'):
                x = report['arms']['context'][stage]['groups'][subset][metric]
                y = report['arms']['text-only'][stage]['groups'][subset][metric]
                if x is not None and y is not None:
                    lines.append(f'| {stage} / {subset} / {metric} | {x*100:.4f}% | {y*100:.4f}% | {(y-x)*100:+.4f} |')
    lines += ['', 'Common accuracy uses identical original gaps with two retained neighboring characters.',
        'Native accuracy has different candidate counts. Test results do not choose endpoints.', '',
        *['- ' + item for item in report['plan']['limitations']]]
    (run / 'comparison.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=RUN)
    parser.add_argument('--train-samples', type=int, default=1000000)
    parser.add_argument('--holdout-samples', type=int, default=100000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', choices=ARMS)
    parser.add_argument('--stage', choices=('train', 'test'), default='train')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return worker(run, args.worker, args.stage)
    lock = (run / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child, stopped = None, False
    def notify(stage, **details):
        if stopped and stage not in ('stopped', 'failed'):
            raise InterruptedError('Paused during preparation')
        value = {'stage': stage, 'updated_at': datetime.now(timezone.utc).isoformat(),
                 'coordinator_pid': os.getpid(), **details}
        write(run / 'status.json', value)
        print(json.dumps(value), flush=True)
    def stop(signum, _):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    awake = subprocess.Popen(['/usr/bin/caffeinate', '-is', '-w', str(os.getpid())], stdin=subprocess.DEVNULL)
    try:
        if (run / 'run.json').exists():
            if not args.resume:
                raise FileExistsError('Use --resume for an existing experiment')
            plan = read(run / 'run.json')
        else:
            request = run / 'request.json'
            if request.exists():
                if not args.resume:
                    raise FileExistsError('Use --resume for interrupted preparation')
                requested = read(request)
                args.train_samples, args.holdout_samples = requested['train'], requested['holdout']
            else:
                write(request, {'train': args.train_samples, 'holdout': args.holdout_samples})
            if args.train_samples < 1 or args.holdout_samples < 1:
                raise ValueError('Sample sizes must be positive')
            plan = prepare(run, args.train_samples, args.holdout_samples, notify)
        for name, expected in plan['source_hashes'].items():
            if sha(run / 'source/training' / name) != expected:
                raise ValueError('Frozen source changed: ' + name)
        for name, expected in plan['data_hashes'].items():
            if sha(run / name) != expected:
                raise ValueError('Frozen data changed: ' + name)
        if sha(run / 'initialization.pt') != plan['initialization_sha256']:
            raise ValueError('Initialization changed')
        environment = dict(os.environ, PYTHONPATH=str(ROOT / 'training/.deps'), DEBUG='0',
            SUPER_READER_PROJECT_ROOT=str(ROOT), PYTHONUNBUFFERED='1',
            PYTORCH_ENABLE_MPS_FALLBACK='0', PYTORCH_MPS_FAST_MATH='0', PYTORCH_MPS_PREFER_METAL='0')
        for stage in ('train', 'test'):
            for arm in ARMS:
                result = 'endpoint-validation.json' if stage == 'train' else 'test.json'
                while not (run / arm / result).exists():
                    if stopped:
                        raise InterruptedError('Paused between workers')
                    command = [sys.executable, str(run / 'source/training/run_symbol_ablation.py'),
                               '--run-dir', str(run), '--worker', arm, '--stage', stage]
                    with (run / (arm + '-' + stage + '.log')).open('ab') as log:
                        child = subprocess.Popen(command, cwd=ROOT, env=environment,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                        notify(stage + '-' + arm, child_pid=child.pid)
                        code = child.wait()
                    if stopped or code == 130:
                        raise InterruptedError('Paused at a safe checkpoint')
                    if code not in (0, 75):
                        raise RuntimeError(f'{arm}/{stage} failed with {code}')
                    if code == 0 and not (run / arm / result).exists():
                        raise RuntimeError('Worker completed without its result')
        sys.path.insert(0, str(ROOT / 'training/.deps'))
        compare(run)
        notify('complete', report=str(run / 'comparison.json'))
    except InterruptedError as error:
        notify('stopped', reason=str(error))
        return 130
    except Exception as error:
        notify('failed', error=str(error))
        raise
    finally:
        awake.terminate()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
