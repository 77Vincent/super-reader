"""Exhaustive suffix traversal, frozen evaluation, gzip provenance and crash recovery."""
import argparse
import contextlib
import gzip
import io
import json
from pathlib import Path
import random
import signal
import tempfile
import unittest
from unittest.mock import patch

import pyarrow.parquet as pq

import prepare_remaining_web_data as preparation
from prepare_web_data import (clean_document, document_signature, document_split,
    labeled_samples, projected_key, quality_document, read, sha, write)
import test_web_preparation


class RemainingPreparationTests(unittest.TestCase):
    def fixture(self, root):
        old, _ = test_web_preparation.WebPreparationTests().fixture(root)
        files = read(old.raw_dir / 'download-manifest.json')['files']
        cursors, suffix_files = [], []
        for index, entry in enumerate(files):
            metadata = pq.ParquetFile(old.raw_dir / entry['filename']).metadata
            order = list(range(metadata.num_row_groups))
            random.Random(old.seed + index).shuffle(order)
            # Different nonzero cursors ensure resumption cannot restart files at zero.
            ordinal, used = index, 2 + index
            suffix_files.append({**entry, 'order': order, 'start': {'group': ordinal, 'row': used}})
            cursors.append({'file': entry['filename'], 'total_documents': metadata.num_rows,
                'scanned_documents': sum(metadata.row_group(g).num_rows for g in order[:ordinal]) + used,
                'last_processed_group': {'group_ordinal': ordinal, 'physical_row_group': order[ordinal],
                                         'processed_rows_in_group': used}})
        usage = root / 'usage.json'
        write(usage, {'inputs': {}, 'files': cursors})
        # Real overlaps: a remaining training document in the frozen registry,
        # and a different document whose first pair appears in validation.
        suffix = {'files': suffix_files, 'raw_directory': str(old.raw_dir)}
        candidates = [r[3]['content'] for r in preparation.remaining_documents(suffix, {'file': 0, **suffix_files[0]['start']})
                      if r[3] and document_split(document_signature(r[3]['content']), old.seed) == 'train']
        registry = old.evaluation / 'holdout-document-hashes.json'
        write(registry, read(registry) + [document_signature(candidates[0])])
        pair, target, punctuation = next(labeled_samples(candidates[1]))
        validation = old.evaluation / 'validation.jsonl'
        with validation.open('a') as handle:
            handle.write(json.dumps({'id': 'overlap', 'domain': 'web', 'tokens': list(pair),
                'target_index': target, 'punctuation': punctuation}) + '\n')
        summary = read(old.evaluation / 'summary.json')
        summary['splits']['validation'] = {'count': 2, 'per_domain': {'news': 1, 'web': 1}, 'sha256': sha(validation)}
        write(old.evaluation / 'summary.json', summary)
        args = argparse.Namespace(**{**vars(old), 'usage_report': usage,
            'policy_source': Path(__file__).resolve().parent, 'documents_per_shard': 7,
            'seen_bloom_bytes': 1024**2, 'holdout_bloom_bytes': 1024**2, 'reserve_gib': 0})
        args.output_dir.mkdir()
        return args

    def run_worker(self, out, limit=0):
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                preparation.worker(out, limit)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)

    def rows(self, out):
        manifest = read(out / 'manifest.json')
        return [json.loads(line) for shard in manifest['shards']
                for line in gzip.open(shard['path'], 'rt', encoding='utf-8')]

    def test_exhausts_suffix_preserves_targets_and_excludes_holdouts(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.fixture(Path(folder))
            plan = preparation.make_plan(args)
            remaining = [r for r in preparation.remaining_documents(plan, {'file': 0, **plan['files'][0]['start']}) if r[3] is not None]
            self.assertEqual(len(remaining), 58)
            self.assertEqual(len({(r[0], r[1], r[2]) for r in remaining}), 58)
            blocked = {key for p in plan['holdout_registries'] for key in read(p)}
            held = {projected_key(''.join(json.loads(line)['tokens'])) for p in plan['holdouts'].values()
                    for line in Path(p['path']).read_text().splitlines()}
            expected, keys = [], set()
            for _, _, _, record, _ in remaining:
                text = record['content']
                sig = document_signature(text)
                if sig in blocked or document_split(sig, plan['seed']) != 'train' or not quality_document(clean_document(text)):
                    continue
                for text, target, _ in labeled_samples(text):
                    key = projected_key(text)
                    if key in keys or key in held:
                        continue
                    keys.add(key)
                    expected.append((text, target))
            self.run_worker(args.output_dir)
            manifest = read(args.output_dir / 'manifest.json')
            self.assertEqual([(r[0], r[1]) for r in self.rows(args.output_dir)], expected)
            self.assertEqual(manifest['counts']['documents'], 58)
            self.assertGreater(manifest['counts']['holdout_documents_filtered'], 0)
            self.assertGreater(manifest['counts']['holdout_pairs_filtered'], 0)
            self.assertEqual(manifest['statistics']['samples'], len(expected))
            self.assertEqual(manifest['domains'], ['web'])
            self.assertNotIn('base_manifest', manifest)
            self.assertTrue(all(Path(s['path']).is_relative_to(args.output_dir) for s in manifest['shards']))
            for split in ('validation', 'test'):
                self.assertEqual((args.output_dir / f'{split}.jsonl').read_bytes(), (args.evaluation / f'{split}.jsonl').read_bytes())
            from inspect_training_sample import inspect_sample
            for i, shard in enumerate(manifest['shards']):
                for row in range(shard['rows']):
                    result = inspect_sample(args.output_dir / 'manifest.json', i, row)
                    self.assertTrue(result['verified'])
                    origin = result['origin']
                    self.assertIn((int(Path(origin['raw_file']).stem), origin['row_group'], origin['row_index']),
                                  {(r[0], r[1], r[2]) for r in remaining})
            self.run_worker(args.output_dir)
            self.assertEqual(read(args.output_dir / 'status.json')['stage'], 'complete')

    def test_crash_after_publish_and_pilot_resume_match_uninterrupted_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = self.fixture(root)
            preparation.make_plan(args)
            self.run_worker(args.output_dir)
            expected = self.rows(args.output_dir)
            for kind in ('crash', 'pilot'):
                out = root / kind
                out.mkdir()
                preparation.make_plan(argparse.Namespace(**{**vars(args), 'output_dir': out}))
                if kind == 'crash':
                    real_publish = preparation.publish_chunk
                    def crash(*a, **kw):
                        real_publish(*a, **kw)
                        raise InterruptedError('after atomic publish, before Bloom update')
                    with patch.object(preparation, 'publish_chunk', side_effect=crash):
                        with self.assertRaises(InterruptedError):
                            self.run_worker(out)
                    # An unrelated incomplete transaction is never imported on recovery.
                    (out / '.chunk-writing').mkdir()
                    (out / '.chunk-writing/data.jsonl.gz').write_bytes(b'partial')
                else:
                    self.run_worker(out, 7)
                    self.assertEqual(read(out / 'status.json')['reason'], 'session_limit')
                    self.assertFalse((out / 'manifest.json').exists())
                self.run_worker(out)
                self.assertEqual(self.rows(out), expected)
                self.assertEqual(read(out / 'manifest.json')['counts']['documents'], 58)

    def test_rejects_wrong_cursor_changed_policy_and_corrupt_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.fixture(Path(folder))
            usage = read(args.usage_report)
            usage['files'][0]['scanned_documents'] += 1
            write(args.usage_report, usage)
            with self.assertRaisesRegex(ValueError, 'logged document count'):
                preparation.make_plan(args)
            usage['files'][0]['scanned_documents'] -= 1
            write(args.usage_report, usage)
            plan = preparation.make_plan(args)
            self.run_worker(args.output_dir, 7)
            snapshot = args.output_dir / 'seen.bloom'
            value = bytearray(snapshot.read_bytes()); value[-1] ^= 1; snapshot.write_bytes(value)
            with self.assertRaisesRegex(ValueError, 'Corrupt Bloom snapshot'):
                self.run_worker(args.output_dir)
            (args.output_dir / 'source/text-policy.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Frozen source changed'):
                preparation.verify_plan(args.output_dir, plan)


if __name__ == '__main__':
    unittest.main()
