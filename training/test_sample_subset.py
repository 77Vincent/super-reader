"""Read-time filtering, passive source statistics, immutable data and fresh/resume runs."""
import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from sample_subset import keep_target, empty_statistics, summarize_inventory, source_name, limit_batch_lengths
from train_sharded import combined_weight_mean, read_evaluation_records, read_training_shard
from text_policy import DATA_POLICY
from run_fresh_training import prepare

ROOT = Path(__file__).resolve().parent.parent


class SampleSubsetTests(unittest.TestCase):
    def test_compressed_shards_preserve_records_filters_and_weight_mean(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, plain, manifest, vocabulary = self.fixture(root)
            packed = root / 'train.jsonl.gz'
            with gzip.open(packed, 'wb') as handle:
                handle.write(plain.read_bytes())
            weights = [[0.5, 2.0]]
            records, counters = [], []
            for path in (plain, packed):
                counts = empty_statistics()
                records.append(read_training_shard(path, vocabulary, weights, manifest['domains'], 2, counts, 4))
                counters.append(counts)
            self.assertEqual(records[0], records[1])
            self.assertEqual(counters[0], counters[1])
            self.assertEqual(combined_weight_mean([plain], weights, 2, 4),
                             combined_weight_mean([packed], weights, 2, 4))

    def fixture(self, directory):
        rows = [['甲乙', 0, 0, 0, 0], ['甲乙丙丁', 0, 0, 0, 0],
                ['甲乙丙丁', 2, 1, 0, 1], ['甲乙丙丁', 1, 0, 0, 1],
                ['𠀀乙丙丁戊己', 1, 1, 0, 0]]
        shard = directory / 'train.jsonl'
        shard.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        manifest = {**DATA_POLICY, 'format': 'super-reader-sharded-training-v1',
                    'domains': ['news', 'web'], 'statistics': {'cells': [[3, 2]]},
                    'vocabulary_path': str(directory / 'vocabulary.json'),
                    'shards': [{'path': str(shard)}]}
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

    def test_length_cap_counts_code_points_and_includes_exact_limit(self):
        text = '𠀀' + '甲' * 252 + '1。 '
        self.assertEqual(len(text), 256)
        self.assertTrue(keep_target(len(text), 1, 2, 256))
        self.assertFalse(keep_target(len(text + '乙'), 1, 2, 256))
        self.assertTrue(keep_target(20000, 1, 2, 0))
        with self.assertRaises(ValueError):
            keep_target(4, 1, 2, -1)

    def test_all_splits_apply_whole_sample_cap_and_batch_order_is_preserved(self):
        from train_smoke import make_batch_indices, iterate_batches
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows, shard, manifest, vocab = self.fixture(root)
            before = {p: p.read_bytes() for p in root.iterdir()}
            counts = empty_statistics()
            train = read_training_shard(shard, vocab, [[1., 1.]], manifest['domains'], 2, counts, 4)
            self.assertEqual([r['token_ids'] for r in train], [[vocab[c] for c in rows[3][0]]])
            self.assertEqual((counts['samples'], counts['excluded_samples'], counts['tokens']), (1, 4, 4))
            for split in ('validation', 'test'):
                other = empty_statistics()
                records = read_evaluation_records(root / f'{split}.jsonl', vocab, 2, other, 4)
                self.assertEqual(other, counts)
                self.assertEqual(records[0]['target_index'], rows[3][1])
            plan_counts = empty_statistics()
            planned = read_training_shard(shard, vocab, [[1., 1.]], manifest['domains'],
                                          2, plan_counts, 4, include_overlength_for_batching=True)
            self.assertEqual(plan_counts, counts)
            self.assertEqual(len(planned), 2)  # Overlength record used only for order planning.
            for seed in range(5):
                batches = make_batch_indices(planned, 8, 128, shuffle=True, seed=seed)
                kept = limit_batch_lengths(planned, batches, 4)
                self.assertEqual(kept, [[0]])
                batch = next(iterate_batches(planned, 8, 128, shuffle=False, seed=0, batch_indices=kept))
                self.assertEqual(tuple(batch['token_ids'].shape), (1, 4))
                self.assertEqual(int(batch['gap_mask'].sum()), 3)
                self.assertIs(limit_batch_lengths(planned, batches, 0), batches)
            self.assertEqual(before, {p: p.read_bytes() for p in root.iterdir()})

    def test_length_filter_keeps_batch_and_record_order_without_empty_updates(self):
        records = [{'token_ids': [0] * n} for n in (3, 9, 4, 8, 5)]
        batches = [[3], [4, 2], [1], [0]]
        self.assertEqual(limit_batch_lengths(records, batches, 4), [[2], [0]])
        self.assertEqual(limit_batch_lengths(records, batches, 2), [])

    def test_all_splits_filter_same_targets_without_modifying_files_or_gaps(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows, shard, manifest, vocab = self.fixture(root)
            before = {p: p.read_bytes() for p in root.iterdir()}
            weights = [[1., 1.]]
            self.assertEqual(len(read_training_shard(shard, vocab, weights, manifest['domains'])), 5)
            train_counts = empty_statistics()
            train = read_training_shard(shard, vocab, weights, manifest['domains'], 2, train_counts)
            for split in ('validation', 'test'):
                counts = empty_statistics()
                records = read_evaluation_records(root / f'{split}.jsonl', vocab, 2, counts)
                self.assertEqual([(r['token_ids'], r['target_index']) for r in records],
                                 [(r['token_ids'], r['target_index']) for r in train])
                self.assertEqual(counts, train_counts)
                self.assertEqual((counts['source_samples'], counts['samples'], counts['excluded_samples']), (5, 2, 3))
                self.assertTrue(all('domain' not in r for r in records))
            from train_smoke import iterate_batches
            batch = next(iterate_batches(train[:1], 8, 128, shuffle=False, seed=0))
            self.assertEqual(batch['gap_mask'].sum().item(), len(rows[3][0]) - 1)
            self.assertEqual(before, {p: p.read_bytes() for p in root.iterdir()})

    def test_statistics_count_unique_loaded_shards_not_epochs_or_resumes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, shard, manifest, vocabulary = self.fixture(root)
            inventory = {}
            self.assertFalse(summarize_inventory(inventory, 1)['complete'])
            for _ in range(3):
                counts = empty_statistics()
                read_training_shard(shard, vocabulary, [[1., 1.]], manifest['domains'], 2, counts)
                inventory[str(shard)] = counts
                inventory = json.loads(json.dumps(inventory))  # serialization / reload
            stats = summarize_inventory(inventory, 1)
            self.assertTrue(stats['complete'])
            self.assertEqual((stats['source_samples'], stats['samples'], stats['excluded_samples'], stats['tokens']), (5, 2, 3, 10))
            self.assertEqual(stats['center_correct'], 1)
            self.assertAlmostEqual(stats['random_baseline_sum'], 1/3 + 1/5)
            self.assertEqual(stats['per_source']['CLUE TNEWS'], {'read': 3, 'retained': 1, 'excluded': 2})
            self.assertEqual(stats['per_source'][source_name('web')], {'read': 2, 'retained': 1, 'excluded': 1})
            self.assertNotIn('domain_samples', stats)

    def test_uniform_weights_do_not_prescan_even_with_subset_filter(self):
        with patch.object(Path, 'open', side_effect=AssertionError('must not scan')):
            self.assertEqual(combined_weight_mean([Path('unread.jsonl')], [[1., 1.]], 2), 1.)

    def test_unknown_source_is_explicit_not_a_guessed_topic(self):
        self.assertEqual(source_name('academic'), 'CLUE CSL')
        self.assertEqual(source_name('encyclopedia'), 'CLUE CMRC2018')
        self.assertEqual(source_name('Chinese Wikipedia'), 'Chinese Wikipedia')
        for value in ('medical', '', None):
            self.assertEqual(source_name(value), 'Unknown corpus source')

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
            self.assertNotIn('--domain-weight-power', plan['command'])
            self.assertNotIn('--selection-macro-weight', plan['command'])
            self.assertEqual(plan['source_weighting'], 'none')
            self.assertEqual(plan['selection_metric'], 'overall_validation_accuracy')
            self.assertEqual(json.loads((run / 'source/text-policy.json').read_text())['standard'], 'unicode-context-v7')
            self.assertFalse(plan['new_preparation_filters_applied'])
            self.assertFalse(plan['defer_test'])
            self.assertNotIn('--defer-test', plan['command'])
            manifest['normalization'] = 'different'
            (root / 'manifest.json').write_text(json.dumps(manifest))
            other = root / 'other'; other.mkdir()
            with self.assertRaisesRegex(ValueError, 'semantics'):
                prepare(other, args)

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS required')
    def test_fresh_coordinator_completes_validation_without_reading_test_set(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            # A successful run proves that even the test records were not read.
            (root / 'test.jsonl').write_text('not valid JSON; test set must stay unread\n')
            run = root / 'run'
            command = [sys.executable, str(ROOT / 'training/run_fresh_training.py'),
                       '--manifest', str(root / 'manifest.json'), '--run-dir', str(run),
                       '--epochs', '1', '--learning-rate', '0.0003',
                       '--max-sequence-length', '4', '--defer-test', '--prepare-only']
            environment = dict(os.environ, DEBUG='0')
            prepared = subprocess.run(command, cwd=ROOT, env=environment,
                                      capture_output=True, text=True, timeout=90)
            self.assertEqual(prepared.returncode, 0, prepared.stdout + prepared.stderr)
            plan = json.loads((run / 'plan.json').read_text())
            self.assertTrue(plan['defer_test'])
            self.assertIn('--defer-test', plan['command'])
            self.assertEqual(plan['subset']['maximum_sequence_length'], 4)
            result = subprocess.run([sys.executable, str(ROOT / 'training/run_fresh_training.py'),
                                     '--run-dir', str(run), '--resume'], cwd=ROOT, env=environment,
                                    capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            status = json.loads((run / 'status.json').read_text())
            self.assertEqual(status['stage'], 'complete')
            self.assertFalse(status['test_evaluated'])
            self.assertFalse(status['promoted'])
            metrics = json.loads(Path(status['metrics']).read_text())
            self.assertEqual(metrics['data_sizes'], {'train': 1, 'validation': 1})
            self.assertEqual(len(metrics['history']), 1)
            self.assertFalse(metrics['test_evaluated'])
            self.assertIsNone(metrics['initialization'])
            self.assertFalse((run / 'candidate/smoke-metrics.json').exists())

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS required')
    def test_length_cap_training_counts_resume_and_rejects_changed_cap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            artifact = root / 'candidate'
            command = [sys.executable, str(ROOT / 'training/run_sharded.py'),
                       '--manifest', str(root / 'manifest.json'), '--data-dir', str(root),
                       '--artifact-dir', str(artifact), '--epochs', '1', '--channels', '8',
                       '--residual-blocks', '1', '--position-weighting', 'none',
                       '--min-side-characters', '2', '--max-sequence-length', '4']
            def run(cmd):
                return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=90,
                                      env=dict(os.environ, DEBUG='0'))
            first = run(command)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            metrics = json.loads((artifact / 'smoke-metrics.json').read_text())
            self.assertEqual(metrics['data_sizes'], {'train': 1, 'validation': 1, 'test': 1})
            self.assertEqual(metrics['sample_subset']['maximum_sequence_length'], 4)
            self.assertEqual(metrics['sample_subset']['train_excluded'], 4)
            command[command.index('--epochs') + 1] = '2'
            second = run([*command, '--resume'])
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            state = torch.load(artifact / 'training-state.pt', map_location='cpu', weights_only=True)
            self.assertEqual({int(v['step']) for v in state['optimizer_state']['state'].values()}, {2})
            command[command.index('--max-sequence-length') + 1] = '0'
            rejected = run([*command, '--resume'])
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('configuration differs', rejected.stderr)

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS required')
    def test_fresh_training_evaluation_and_resume_enforce_subset_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            artifact = root / 'candidate'
            command = [sys.executable, str(ROOT / 'training/run_sharded.py'),
                       '--manifest', str(root / 'manifest.json'), '--data-dir', str(root),
                       '--artifact-dir', str(artifact), '--epochs', '2', '--channels', '8',
                       '--residual-blocks', '1', '--position-weighting', 'none',
                       '--min-side-characters', '2', '--gradient-clip', '1']
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90,
                                    env=dict(os.environ, DEBUG='0'))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            metrics = json.loads((artifact / 'smoke-metrics.json').read_text())
            self.assertEqual(metrics['data_sizes'], {'train': 2, 'validation': 2, 'test': 2})
            self.assertIsNone(metrics['initialization'])
            self.assertEqual(metrics['sample_subset']['train_excluded'], 3)
            checkpoint = torch.load(artifact / 'training-state.pt', map_location='cpu', weights_only=True)
            self.assertEqual(checkpoint['configuration']['minimum_side_characters'], 2)
            self.assertEqual(checkpoint['progress']['epoch'], 3)
            self.assertEqual(checkpoint['configuration']['source_weighting'], 'none')
            stats = json.loads((artifact / 'source-statistics.json').read_text())
            self.assertEqual(stats['train']['samples'], 2)
            self.assertEqual(stats['train']['source_samples'], 5)
            self.assertEqual(metrics['checkpoint_selection']['best_score'],
                             max(row['validation_accuracy'] for row in metrics['history']))
            result = subprocess.run([*command, '--resume'], cwd=ROOT, capture_output=True, text=True, timeout=90,
                                    env=dict(os.environ, DEBUG='0'))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads((artifact / 'source-statistics.json').read_text()), stats)
            command[command.index('--min-side-characters') + 1] = '1'
            result = subprocess.run([*command, '--resume'], cwd=ROOT, capture_output=True, text=True, timeout=90,
                                    env=dict(os.environ, DEBUG='0'))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('configuration differs', result.stderr)


if __name__ == '__main__':
    unittest.main()
