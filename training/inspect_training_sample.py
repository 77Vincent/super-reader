#!/usr/bin/env python3
"""Resolve a zero-based compact training row back to its original source document."""
from __future__ import annotations

import argparse
import gzip
import hashlib
from itertools import islice
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'training/.deps'))
import pyarrow.parquet as pq
from prepare_web_data import PROVENANCE_FORMAT, labeled_samples


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_row(path, index):
    if index < 0:
        raise ValueError('Row indices are zero-based and cannot be negative')
    path = absolute(path)
    with (gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz' else path.open()) as handle:
        line = next(islice(handle, index, index + 1), None)
    if line is None:
        raise IndexError(f'Row {index} does not exist in {path}')
    return json.loads(line)


def inspect_sample(manifest_path, shard_index, row_index, raw_dir=None):
    manifest = json.loads(Path(manifest_path).read_text())
    if not 0 <= shard_index < len(manifest['shards']):
        raise IndexError('Shard index is out of range')
    shard = manifest['shards'][shard_index]
    provenance = shard.get('provenance')
    if not provenance:
        raise ValueError('This inherited shard has no provenance; backfill from its original source is required')
    if provenance.get('format') != PROVENANCE_FORMAT:
        raise ValueError('Unsupported provenance format')
    text, target, domain, bucket, position = read_row(shard['path'], row_index)
    document_ref, pair_index, punctuation = read_row(provenance['path'], row_index)
    if document_ref < 0:
        raise ValueError('Document byte offset cannot be negative')
    documents_path = absolute(provenance['documents_path'])
    with (gzip.open(documents_path, 'rb') if documents_path.suffix == '.gz' else documents_path.open('rb')) as handle:
        handle.seek(document_ref)
        origin = json.loads(handle.readline())
    if origin['byte_offset'] != document_ref:
        raise ValueError('Document index is not aligned')
    source_path = absolute(raw_dir or provenance['raw_directory']) / origin['raw_file']
    group = pq.ParquetFile(source_path).read_row_group(origin['row_group'], columns=['content', 'source'])
    content = group.column('content')[origin['row_index']].as_py() or ''
    source = str(group.column('source')[origin['row_index']].as_py() or 'unknown')
    if hashlib.sha256(content.encode()).hexdigest() != origin['content_sha256'] or source != origin['source']:
        raise ValueError('Raw document differs from the recorded source')
    pair = next(islice(labeled_samples(content), pair_index, pair_index + 1), None)
    if pair != (text, target, punctuation):
        raise ValueError('Pair cannot be reproduced; verify provenance alignment and use the original frozen preparation policy')
    return {'sample': {'text': text, 'target_index': target, 'punctuation': punctuation,
                       'domain': manifest['domains'][domain], 'length_bucket': bucket, 'position_bin': position},
            'origin': {**origin, 'raw_path': str(source_path), 'eligible_pair_index': pair_index},
            'raw_content': content, 'verified': True,
            'location_precision': 'Raw document and eligible pair ordinal; no pre-cleaning character-offset map'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--shard-index', type=int, required=True)
    parser.add_argument('--row-index', type=int, required=True)
    parser.add_argument('--raw-dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_sample(args.manifest, args.shard_index, args.row_index, args.raw_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
