"""Subset membership, effective weighting, immutable data, and fresh/resume runs."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from sample_subset import keep_target, mean_weight, scan_shard, subset_statistics
from train_sharded import domain_weight_table, read_evaluation_records, read_training_shard, weight_table
from text_policy import DATA_POLICY
from run_fresh_training import prepare

ROOT = Path(__file__).resolve().parent.parent


class SampleSubsetTests(unittest.TestCase):
    def fixture(self, directory):
        rows = [['甲乙', 0, 0, 0, 0], ['甲乙丙丁', 0, 0, 0, 0],
                ['甲乙丙丁', 2, 1, 0, 1], ['甲乙丙丁', 1, 0, 0, 1],
                ['𠀀乙丙丁戊己', 1, 1, 0, 0]]
        shard = directory / 'train.jsonl'
        shard.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        manifest = {**DATA_POLICY, 'format': 'super-reader-sharded-training-v1',
                    'domains': ['news', 'web'], 'statistics': {'cells': [[0, 0]]},
                    'vocabulary_path': str(directory / 'vocabulary.json'),
                    'shards': [{'path': str(shard)}]}
        manifest['statistics'], _ = scan_shard(shard, manifest, 1)
        vocabulary = {'<pad>': 0, '<unk>': 1}
        for char in sorted(set('0123456789:、ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ' + ''.join(r[0] for r in rows))):
            vocabulary[char] = len(vocabulary)
        (directory / 'vocabulary.json').write_text(json.dumps(vocabulary))
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        (directory / 'summary.json').write_text(json.dumps(DATA_POLICY))
        for split in ('validation', 'test'):
            records = [{'id': f'{split}-{i}', 'tokens': list(r[0]), 'target_index': r[1],
                        'punctuation': '，', 'domain': manifest['domains'][r[2]]} for i, r in enumerate(rows)]
            (directory / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
        return rows, shard, manifest, vocabulary

    def test_exact_unicode_boundaries_and_invalid_targets(self):
        self.assertFalse(keep_target(len('𠀀乙丙丁'), 0, 2))
        self.assertFalse(keep_target(4, 2, 2))
        self.assertTrue(keep_target(4, 1, 2))
        self.assertTrue(keep_target(2, 0, 1))
        for length, target, minimum in [(4, -1, 2), (4, 3, 2), (4, 1, 0)]:
            with self.assertRaises(ValueError):
                keep_target(length, target, minimum)

    def test_all_splits_filter_same_targets_without_modifying_files_or_gaps(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows, shard, manifest, vocab = self.fixture(root)
            before = {p: p.read_bytes() for p in root.iterdir()}
            weights = [[1., 1.]]
            self.assertEqual(len(read_training_shard(shard, vocab, weights, manifest['domains'], [1., 1.])), 5)
            train = read_training_shard(shard, vocab, weights, manifest['domains'], [1., 1.], 2)
            for split in ('validation', 'test'):
                counts = {'source_samples': 0, 'excluded_samples': 0}
                records = read_evaluation_records(root / f'{split}.jsonl', vocab, 2, counts)
                self.assertEqual([(r['token_ids'], r['target_index']) for r in records],
                                 [(r['token_ids'], r['target_index']) for r in train])
                self.assertEqual(counts, {'source_samples': 5, 'excluded_samples': 3})
            from train_smoke import iterate_batches
            batch = next(iterate_batches(train[:1], 8, 128, shuffle=False, seed=0))
            self.assertEqual(batch['gap_mask'].sum().item(), len(rows[3][0]) - 1)
            self.assertEqual(before, {p: p.read_bytes() for p in root.iterdir()})

    def test_statistics_weights_and_cache_invalidation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, shard, manifest, _ = self.fixture(root)
            cache = root / 'cache'
            stats = subset_statistics([shard], manifest, 2, cache)
            self.assertEqual((stats['source_samples'], stats['samples'], stats['excluded_samples'], stats['tokens']), (5, 2, 3, 10))
            self.assertEqual(stats['center_correct'], 1)
            self.assertAlmostEqual(stats['random_baseline_sum'], 1/3 + 1/5)
            self.assertEqual(stats['cells'], [[1, 1]])
            self.assertAlmostEqual(mean_weight(stats, [[2., 4.]], [3., 5.]), 11.)
            effective = {**manifest, 'statistics': stats}
            self.assertEqual(domain_weight_table(effective, manifest['domains'], .65), [1., 1.])
            self.assertEqual(weight_table(effective), [[1., 1.]])
            with patch('sample_subset.scan_shard', side_effect=AssertionError('should use cache')):
                self.assertEqual(subset_statistics([shard], manifest, 2, cache), stats)
            with shard.open('a') as handle:
                handle.write(json.dumps(['甲乙丙丁', 1, 0, 0, 1]) + '\n')
            self.assertEqual(subset_statistics([shard], manifest, 2, cache)['samples'], 3)
            self.assertEqual(subset_statistics([shard], manifest, 1, cache)['samples'], 6)

    def test_fresh_plan_pins_actual_policy_and_never_inherits_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, _, manifest, _ = self.fixture(root)
            manifest = copy.deepcopy(manifest)
            manifest['standard'] = 'unicode-context-v7'
            manifest['sample_filter']['version'] = 'surface-noise-v4'
            (root / 'manifest.json').write_text(json.dumps(manifest))
            run = root / 'run'; run.mkdir()
            args = argparse.Namespace(manifest=root / 'manifest.json', epochs=3, learning_rate=.002, seed=123)
            plan = prepare(run, args)
            self.assertNotIn('--initialize-from', plan['command'])
            self.assertNotIn('--resume', plan['command'])
            self.assertEqual(json.loads((run / 'source/text-policy.json').read_text())['standard'], 'unicode-context-v7')
            self.assertFalse(plan['new_preparation_filters_applied'])
            manifest['normalization'] = 'different'
            (root / 'manifest.json').write_text(json.dumps(manifest))
            other = root / 'other'; other.mkdir()
            with self.assertRaisesRegex(ValueError, 'semantics'):
                prepare(other, args)

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS required')
    def test_fresh_training_evaluation_and_resume_enforce_subset_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            artifact = root / 'candidate'
            command = [sys.executable, str(ROOT / 'training/run_sharded.py'),
                       '--manifest', str(root / 'manifest.json'), '--data-dir', str(root),
                       '--artifact-dir', str(artifact), '--epochs', '1', '--channels', '8',
                       '--residual-blocks', '1', '--position-weighting', 'none',
                       '--min-side-characters', '2', '--domain-weight-power', '.65', '--gradient-clip', '1']
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90,
                                    env=dict(os.environ, DEBUG='0'))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            metrics = json.loads((artifact / 'smoke-metrics.json').read_text())
            self.assertEqual(metrics['data_sizes'], {'train': 2, 'validation': 2, 'test': 2})
            self.assertIsNone(metrics['initialization'])
            self.assertEqual(metrics['sample_subset']['train_excluded'], 3)
            checkpoint = torch.load(artifact / 'training-state.pt', map_location='cpu', weights_only=True)
            self.assertEqual(checkpoint['configuration']['minimum_side_characters'], 2)
            self.assertEqual(checkpoint['progress']['epoch'], 2)
            command[command.index('--min-side-characters') + 1] = '1'
            result = subprocess.run([*command, '--resume'], cwd=ROOT, capture_output=True, text=True, timeout=90,
                                    env=dict(os.environ, DEBUG='0'))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('configuration differs', result.stderr)


if __name__ == '__main__':
    unittest.main()
