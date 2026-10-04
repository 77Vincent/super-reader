"""Read-time filtering and descriptive corpus provenance; never training weights."""
from __future__ import annotations

import json
from pathlib import Path

# These are legacy dataset IDs, not inferred semantic categories. The mapping is
# grounded in prepare_smoke_data.mjs, prepare_synthetic_data.py and prepare_web_data.py.
CORPUS_SOURCES = {
    'news': 'CLUE TNEWS',
    'academic': 'CLUE CSL',
    'encyclopedia': 'CLUE CMRC2018',
    'dialogue': 'CLUE C3',
    'wikipedia': 'Chinese Wikipedia',
    'synthetic_multistyle': 'openbmb/Ultra-FineWeb-L3 (zh Multi-Style-Synthetic)',
    'web': 'openbmb/Ultra-FineWeb (zh)',
}
UNKNOWN_SOURCE = 'Unknown corpus source'


def source_name(value) -> str:
    if isinstance(value, str) and value in CORPUS_SOURCES.values():
        return value
    return CORPUS_SOURCES.get(value, UNKNOWN_SOURCE) if isinstance(value, str) else UNKNOWN_SOURCE


def record_source(record) -> str:
    return source_name(record.get('corpus_source', record.get('domain')))


def keep_target(length: int, target: int, minimum: int = 1, maximum: int = 0) -> bool:
    if minimum < 1:
        raise ValueError('Minimum side characters must be positive')
    if maximum < 0:
        raise ValueError('Maximum sequence length cannot be negative')
    if not isinstance(target, int) or not 0 <= target < length - 1:
        raise ValueError(f'Invalid target {target} for {length} characters')
    cut = target + 1
    return min(cut, length - cut) >= minimum and (maximum == 0 or length <= maximum)


def limit_batch_lengths(records, batches, maximum=0):
    """Drop whole overlength records without reshuffling the surviving batches."""
    if maximum < 0:
        raise ValueError('Maximum sequence length cannot be negative')
    if maximum == 0:
        return batches
    result = []
    for indices in batches:
        retained = [index for index in indices if len(records[index]['token_ids']) <= maximum]
        if retained:
            result.append(retained)
    return result


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def empty_statistics():
    return dict(source_samples=0, excluded_samples=0, samples=0, tokens=0,
                maximum_sequence_length=0, random_baseline_sum=0., center_correct=0,
                per_source={})


def count_sample(stats, text_length, target, source, kept):
    stats['source_samples'] += 1
    row = stats['per_source'].setdefault(source, dict(read=0, retained=0, excluded=0))
    row['read'] += 1
    if not kept:
        stats['excluded_samples'] += 1
        row['excluded'] += 1
        return
    stats['samples'] += 1
    stats['tokens'] += text_length
    stats['maximum_sequence_length'] = max(stats['maximum_sequence_length'], text_length)
    stats['random_baseline_sum'] += 1 / (text_length - 1)
    stats['center_correct'] += target == (text_length - 1) // 2
    row['retained'] += 1


def summarize_inventory(inventory, shard_count):
    """One entry per loaded file: rereads/resumes/epochs cannot inflate counts."""
    total = empty_statistics()
    for stats in inventory.values():
        for key in ('source_samples', 'excluded_samples', 'samples', 'tokens', 'random_baseline_sum', 'center_correct'):
            total[key] += stats[key]
        total['maximum_sequence_length'] = max(total['maximum_sequence_length'], stats['maximum_sequence_length'])
        for source, values in stats['per_source'].items():
            row = total['per_source'].setdefault(source, dict(read=0, retained=0, excluded=0))
            for key in row:
                row[key] += values[key]
    total['loaded_shards'] = len(inventory)
    total['total_shards'] = shard_count
    total['complete'] = len(inventory) == shard_count
    total['scope'] = 'unique loaded shard contents, not optimizer examples or epoch totals'
    return total
