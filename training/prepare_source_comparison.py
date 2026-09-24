#!/usr/bin/env python3
"""Select matched existing v7 training pairs; never rebuild or repair labels."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

from run_web_continuation import read, write, sha
from run_width_comparison import select_shards, project_path
from text_policy import require_data_policy

CLEAN_DOMAINS = ('wikipedia', 'synthetic_multistyle')
ARMS = ('current-mix', 'wiki-synthetic')


def stratum(row):
    text, target, _, bucket, position = row
    if not 0 <= target < len(text) - 1:
        raise ValueError('Invalid inherited label')
    return f'{len(text)},{bucket},{position}'


def matched_quotas(a, b, target):
    """Capped proportional allocation on common support, with exact total."""
    capacities = {k: min(n, b.get(k, 0)) for k, n in a.items() if b.get(k, 0)}
    if target <= 0 or sum(capacities.values()) < target:
        raise ValueError('Insufficient common length/position support')
    remaining, active, quotas = target, dict(capacities), {}
    while remaining and active:
        total = sum(a[k] for k in active)
        saturated = [k for k, cap in active.items() if remaining * a[k] >= cap * total]
        if saturated:
            for k in saturated:
                quotas[k] = active.pop(k)
                remaining -= quotas[k]
            continue
        for k in active:
            quotas[k] = remaining * a[k] // total
        extra = remaining - sum(quotas[k] for k in active)
        order = sorted(active, key=lambda k: (-(remaining * a[k] % total), k))
        for k in order[:extra]:
            quotas[k] += 1
        remaining = 0
    assert sum(quotas.values()) == target
    return {k: n for k, n in quotas.items() if n}


def scan(path, allowed, domains, stop):
    counts, source_counts = Counter(), Counter()
    digest = hashlib.sha256()
    rows = 0
    with path.open('rb') as handle:
        for line in handle:
            digest.update(line)
            row = json.loads(line)
            rows += 1
            if rows % 10000 == 0 and stop():
                raise InterruptedError('Stopped while counting candidate shards')
            domain = domains[row[2]]
            if domain in allowed:
                counts[stratum(row)] += 1
                source_counts[domain] += 1
    return {'sha256': digest.hexdigest(), 'counts': dict(counts), 'domains': dict(source_counts),
            'path': str(path), 'bytes': path.stat().st_size, 'rows': rows}


def count_pool(run, arm, pool, domains, allowed, stop, progress):
    cache = run / 'counts' / arm
    cache.mkdir(parents=True, exist_ok=True)
    total, sources, inputs = Counter(), Counter(), []
    for index, item in enumerate(pool):
        if stop():
            raise InterruptedError('Stopped between candidate shards')
        path = project_path(item['path'])
        cached = cache / (hashlib.sha256(str(path).encode()).hexdigest() + '.json')
        if path.stat().st_size != item['bytes']:
            raise ValueError('Source shard size changed: ' + str(path))
        if cached.exists():
            result = read(cached)
            if result['sha256'] != sha(path):
                raise ValueError('Counted source changed: ' + str(path))
        else:
            result = scan(path, allowed, domains, stop)
            write(cached, result)
        if item.get('sha256', result['sha256']) != result['sha256']:
            raise ValueError('Frozen source hash differs: ' + str(path))
        total.update(result['counts'])
        sources.update(result['domains'])
        inputs.append({k: result[k] for k in ('path', 'bytes', 'sha256')})
        if (index + 1) % 8 == 0 or index + 1 == len(pool):
            progress('counting-' + arm, shards_done=index + 1, shards_total=len(pool), samples=sum(total.values()))
    return dict(total), dict(sources), inputs


def sample_pool(run, arm, inputs, base, counts, quotas, domains, seed, shard_count, stop, progress):
    """Uniform within stratum, without replacement; paired per-shard histograms."""
    destination = run / (arm + '-manifest.json')
    if destination.exists():
        manifest = read(destination)
        for item in manifest['shards']:
            if sha(Path(item['path'])) != item['sha256']:
                raise ValueError('Prepared shard changed: ' + item['path'])
        return manifest
    output = run / 'data' / arm
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / f'train-{i:03d}.jsonl' for i in range(shard_count)]
    handles = [path.with_suffix('.part').open('wb') for path in paths]
    remaining, needed, selected = Counter(counts), Counter(quotas), Counter()
    offsets = {k: int.from_bytes(hashlib.sha256(k.encode()).digest()[:4], 'little') % shard_count for k in quotas}
    mapping = {base['domains'].index(domain): index for index, domain in enumerate(domains)}
    rng = random.Random(seed)
    stats = {'samples': 0, 'tokens': 0, 'random_baseline_sum': 0., 'center_correct': 0,
             'maximum_sequence_length': 0, 'domain_samples': dict.fromkeys(domains, 0),
             'position_histogram': [0] * base['position_bins'],
             'cells': [[0] * base['position_bins'] for _ in base['statistics']['cells']]}
    length_histograms = [Counter() for _ in paths]
    source_position = Counter()
    try:
        for index, item in enumerate(inputs):
            if stop():
                raise InterruptedError('Stopped during matched sampling; resume restarts this arm only')
            digest = hashlib.sha256()
            with Path(item['path']).open('rb') as handle:
                for line_number, line in enumerate(handle):
                    digest.update(line)
                    row = json.loads(line)
                    if line_number % 10000 == 0 and stop():
                        raise InterruptedError('Stopped during matched sampling')
                    if row[2] not in mapping:
                        continue
                    key = stratum(row)
                    take = needed[key] > 0 and rng.randrange(remaining[key]) < needed[key]
                    remaining[key] -= 1
                    if not take:
                        continue
                    shard = (offsets[key] + selected[key]) % shard_count
                    needed[key] -= 1
                    selected[key] += 1
                    text, target, old_domain, bucket, position = row
                    domain = mapping[old_domain]
                    row[2] = domain
                    handles[shard].write((json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n').encode())
                    length = len(text)
                    length_histograms[shard][length] += 1
                    stats['samples'] += 1
                    stats['tokens'] += length
                    stats['random_baseline_sum'] += 1 / (length - 1)
                    stats['center_correct'] += target == (length - 1) // 2
                    stats['maximum_sequence_length'] = max(stats['maximum_sequence_length'], length)
                    stats['domain_samples'][domains[domain]] += 1
                    stats['position_histogram'][position] += 1
                    stats['cells'][bucket][position] += 1
                    source_position[f'{domain},{bucket},{position}'] += 1
            if digest.hexdigest() != item['sha256']:
                raise ValueError('Input changed between counting and sampling')
            if (index + 1) % 8 == 0 or index + 1 == len(inputs):
                progress('sampling-' + arm, shards_done=index + 1, shards_total=len(inputs), samples=stats['samples'])
        if any(needed.values()) or dict(selected) != quotas:
            raise ValueError('Matched sampling did not meet every quota')
    finally:
        for handle in handles:
            handle.close()
    if any(n == 0 for n in stats['domain_samples'].values()):
        raise ValueError('Sampling lost a requested source domain')
    shards = []
    for path, histogram in zip(paths, length_histograms):
        path.with_suffix('.part').replace(path)
        buckets = Counter()
        for length, count in histogram.items():
            buckets[1 << (length - 1).bit_length()] += count
        batches = sum(math.ceil(count / max(1, min(512, 8192 // ceiling))) for ceiling, count in buckets.items())
        shards.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path),
                       'samples': sum(histogram.values()),
                       'length_histogram': {str(k): v for k, v in histogram.items()}, 'batches': batches})
    manifest = {k: v for k, v in base.items() if k in (
        'standard', 'input_representation', 'tokenization', 'proxy_punctuation', 'normalization',
        'numeric_punctuation', 'boundary_context', 'line_boundaries', 'sample_filter', 'symbol_window',
        'source_filter', 'format', 'length_bucket_maximums', 'position_bins', 'maximum_sequence_length', 'protection')}
    manifest.update(source='matched source composition pilot: ' + arm, domains=list(domains),
        vocabulary_path=str(run / 'vocabulary.json'), evaluation_source_dir=base['evaluation_source_dir'],
        statistics=stats, shards=shards, source_position_counts=dict(source_position),
        sampling={'seed': seed, 'method': 'uniform without replacement in exact length / position-bin cells',
                  'quotas_sha256': sha(run / 'quotas.json'), 'input_shards': inputs,
                  'proxy_type_matching': False, 'labels_and_text_unchanged': True})
    write(destination, manifest)
    return manifest


def weighting_summary(manifest, power=.65):
    """Report the unchanged trainer's data-dependent effective loss mixture."""
    counts = manifest['statistics']['domain_samples']
    raw = [counts[domain] ** -power for domain in manifest['domains']]
    mean = sum(n * w for n, w in zip(counts.values(), raw)) / sum(counts.values())
    domain_weights = [w / mean for w in raw]
    position_weights = []
    for row in manifest['statistics']['cells']:
        occupied = sum(n > 0 for n in row)
        position_weights.append([sum(row) / occupied / n if n else 0 for n in row])
    contributions = Counter()
    for key, n in manifest['source_position_counts'].items():
        domain, bucket, position = map(int, key.split(','))
        contributions[manifest['domains'][domain]] += n * domain_weights[domain] * position_weights[bucket][position]
    total = sum(contributions.values())
    return {'domain_weights': dict(zip(manifest['domains'], domain_weights)),
            'combined_weight_mean': total / sum(counts.values()),
            'effective_loss_share': {k: v / total for k, v in contributions.items()}}


def prepare(run, base, target, seed, shard_count, stop, progress):
    require_data_policy(base)
    a_pool, groups = select_shards(base, min(base['statistics']['samples'], math.ceil(target * 1.2)), seed)
    # The first source generation contains Wikipedia and the four small old
    # domains; filter compact domain IDs rather than assuming all rows are wiki.
    parents = {str(project_path(item['path']).parent) for item in base['shards']
               if '-wiki/' in item['path'] or '-local/' in item['path']}
    if len(parents) != 2:
        raise ValueError('Unexpected wiki/synthetic shard layout')
    b_pool = [item for item in base['shards'] if str(project_path(item['path']).parent) in parents]
    pool_stats = {}
    for arm, pool, domains in [(ARMS[0], a_pool, base['domains']), (ARMS[1], b_pool, CLEAN_DOMAINS)]:
        pool_stats[arm] = count_pool(run, arm, pool, base['domains'], domains, stop, progress)
    quotas = matched_quotas(pool_stats[ARMS[0]][0], pool_stats[ARMS[1]][0], target)
    quota_path = run / 'quotas.json'
    if quota_path.exists() and read(quota_path) != quotas:
        raise ValueError('Previously frozen quotas changed')
    write(quota_path, quotas)
    manifests = {}
    for arm, domains in [(ARMS[0], base['domains']), (ARMS[1], CLEAN_DOMAINS)]:
        counts, _, inputs = pool_stats[arm]
        manifests[arm] = sample_pool(run, arm, inputs, base, counts, quotas, domains,
                                    seed, shard_count, stop, progress)
    a, b = (manifests[arm] for arm in ARMS)
    for key in ('samples', 'tokens', 'position_histogram', 'cells', 'maximum_sequence_length'):
        if a['statistics'][key] != b['statistics'][key]:
            raise ValueError('Paired global statistics differ: ' + key)
    for left, right in zip(a['shards'], b['shards']):
        for key in ('samples', 'batches', 'length_histogram'):
            if left[key] != right[key]:
                raise ValueError('Paired shard batching differs: ' + key)
    report = {'samples_per_arm': target, 'shards_per_arm': shard_count,
              'batches_per_arm': sum(item['batches'] for item in a['shards']),
              'matched_exact_length_and_position_bin': True, 'matched_proxy_type': False,
              'source_sampling': groups, 'candidate_source_counts': {arm: pool_stats[arm][1] for arm in ARMS},
              'selected_source_counts': {arm: manifests[arm]['statistics']['domain_samples'] for arm in ARMS},
              'weighting': {arm: weighting_summary(manifests[arm]) for arm in ARMS},
              'common_support_candidates': sum(min(n, pool_stats[ARMS[1]][0].get(k, 0)) for k, n in pool_stats[ARMS[0]][0].items()),
              'tokens_per_arm': a['statistics']['tokens'], 'passed': True}
    write(run / 'data-verification.json', report)
    return manifests
