"""Sampling, equal-budget fresh starts, and resumable validation-only LR trials."""
import json
import hashlib
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch

from run_initial_learning_rate_comparison import (
    RATES, inventory_shards, sample_rows, training_command, rank_arms, checkpoint_summary,
    verify_trial_reports, prepare)
from train_smoke import BoundaryChooser, save_training_state, seed_everything
from train_sharded import expanded_initialization
import test_sample_subset as subset_fixtures
from text_policy import DATA_POLICY

ROOT = Path(__file__).resolve().parent.parent


class InitialLearningRateComparisonTests(unittest.TestCase):
    def test_prepare_retains_overflow_bucket_and_actual_v7_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); data = root / 'original'; data.mkdir()
            out = root / 'run'; out.mkdir()
            rows = [[text, 1, 0, 1, 1] for text in ('中国人民甲', '中国人民乙', '中国人民丙', '中国人民丁')]
            shard = data / 'train.jsonl'
            shard.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            vocabulary = {'<pad>': 0, '<unk>': 1, '中': 2}
            (data / 'vocabulary.json').write_text(json.dumps(vocabulary))
            policy = {**DATA_POLICY, 'standard': 'unicode-context-v7'}
            manifest = {**policy, 'format': 'super-reader-sharded-training-v1',
                'domains': ['web'], 'length_bucket_maximums': [4], 'position_bins': 2,
                'vocabulary_path': str(data / 'vocabulary.json'),
                'statistics': {'samples': 4, 'cells': [[0, 0], [0, 4]]},
                'shards': [{'path': str(shard), 'bytes': shard.stat().st_size}]}
            (data / 'manifest.json').write_text(json.dumps(manifest))
            validation = data / 'validation.jsonl'
            validation.write_text(json.dumps({'id': 'v1', 'tokens': list('天地玄黄'),
                'target_index': 1, 'punctuation': '，', 'domain': 'web'}) + '\n')
            (data / 'summary.json').write_text(json.dumps({**policy, 'splits': {'validation': {
                'count': 1, 'sha256': hashlib.sha256(validation.read_bytes()).hexdigest()}}}))
            args = SimpleNamespace(data_dir=data, samples=2, validation_samples=1, seed=3, extended_epochs=3)
            plan = prepare(out, args, lambda *a, **kw: None)
            sampled = json.loads((out / 'data/manifest.json').read_text())
            self.assertEqual(sampled['statistics']['cells'], [[0, 0], [0, 2]])
            self.assertEqual(plan['prepared_data_standard'], 'unicode-context-v7')
            self.assertEqual((out / 'data/test.jsonl').read_bytes(), b'')
            self.assertFalse(plan['test_evaluated'])
            init = torch.load(out / 'initialization.pt', map_location='cpu', weights_only=True)
            self.assertEqual(init['initialization_kind'], 'random; never trained')
            self.assertNotIn('optimizer_state', init)

    def test_global_draw_order_survives_disk_order_and_unequal_shard_sizes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [[f'甲乙{i}丙丁', 1, 0, 0, 0] for i in range(80)]
            rows[3] = ['甲' * 1024, 1, 0, 0, 0]  # No hidden 256-character cap.
            entries = []
            for number, part in enumerate((rows[:3], rows[3:73], rows[73:])):
                path = root / f'{number}.jsonl'
                path.write_text('\n'.join(json.dumps(row) for row in part))
                entries.append({'path': str(path), 'bytes': path.stat().st_size})
            inventory = inventory_shards(entries, lambda *a, **kw: None)
            self.assertEqual([r['rows'] for r in inventory], [3, 70, 7])
            for seed in range(8):
                actual, meta = sample_rows(inventory, 40, seed)
                indices = random.Random(seed).sample(range(80), 80)[:40]
                self.assertEqual([row for row, _ in actual], [rows[i] for i in indices])
                self.assertIsNone(meta['maximum_length_filter'])
                for row, (path, line) in actual:
                    self.assertEqual(json.loads(Path(path).read_text().splitlines()[line]), row)

    def test_same_side_filter_for_training_and_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [['甲乙丙丁', 0, 0, 0, 0], ['甲乙丙丁', 1, 0, 0, 0],
                    ['𠀀乙丙丁戊己', 1, 0, 0, 0], ['甲乙丙丁', 2, 0, 0, 0]]
            for evaluation in (False, True):
                data = [{'tokens': list(r[0]), 'target_index': r[1]} for r in rows] if evaluation else rows
                path = root / str(evaluation)
                path.write_text(''.join(json.dumps(r) + '\n' for r in data))
                selected, meta = sample_rows([{'path': str(path), 'rows': 4}], 2, 99, evaluation=evaluation)
                self.assertEqual(meta['rejected_in_draws'], {'single_character_side': 2})
                self.assertEqual(len(selected), 2)

    def test_commands_only_change_learning_rate_and_output_and_resume_same_arm(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            plan = {'seed': 1, 'rates': RATES, 'training': {'position_weighting': 'none', 'min_side_characters': 2}}
            commands = [training_command(run, plan, arm, 1) for arm in RATES]
            for cmd in commands:
                self.assertIn('--defer-test', cmd)
                self.assertIn('--initialize-from', cmd)
                for flag in ('--artifact-dir', '--learning-rate'):
                    cmd[cmd.index(flag) + 1] = 'varying'
                self.assertEqual(cmd, commands[0])
            (run / 'lr-1e-4').mkdir()
            (run / 'lr-1e-4/training-state.pt').touch()
            extended = training_command(run, plan, 'lr-1e-4', 3)
            self.assertIn('--resume', extended)
            self.assertNotIn('--initialize-from', extended)
            self.assertEqual(extended[extended.index('--epochs') + 1], '3')

    def test_selection_uses_equal_endpoint_validation_and_rejects_test(self):
        reports = {arm: {'endpoint_validation': {'accuracy': .8}, 'test_evaluated': False}
                   for arm in reversed(RATES)}
        self.assertEqual(rank_arms(reports), list(RATES))
        reports['lr-1e-3']['endpoint_validation']['accuracy'] = .81
        self.assertEqual(rank_arms(reports)[0], 'lr-1e-3')
        reports['lr-1e-3']['test_evaluated'] = True
        with self.assertRaises(ValueError):
            rank_arms(reports)

    def test_actual_starting_metrics_and_update_budgets_must_match(self):
        def report(lr):
            return {'test_evaluated': False, 'training_weighting': {'source_weighting': 'none'},
                'configuration': {'learning_rate': lr, 'minimum_side_characters': 2},
                'initialization': {'validation_before_training': {
                    'accuracy': .1, 'loss': 3., 'mean_reciprocal_rank': .2}},
                'verified_checkpoint': {'optimizer_steps': 100}}
        a, b = report(.0001), report(.002)
        verify_trial_reports({'a': a, 'b': b})
        b['verified_checkpoint']['optimizer_steps'] = 101
        with self.assertRaisesRegex(ValueError, 'counts differ'):
            verify_trial_reports({'a': a, 'b': b})
        b = report(.002)
        b['initialization']['validation_before_training']['accuracy'] = .11
        with self.assertRaisesRegex(ValueError, 'initial validation differs'):
            verify_trial_reports({'a': a, 'b': b})

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS required')
    def test_identical_random_transfer_and_extended_trial_preserves_update_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            data = run / 'data'; data.mkdir()
            subset_fixtures.SampleSubsetTests().fixture(data)
            vocab = json.loads((data / 'vocabulary.json').read_text())
            seed_everything(123)
            initial = BoundaryChooser(len(vocab), 8, 1)
            save_training_state(run / 'initialization.pt', {'model_state': initial.state_dict(),
                'best_state': None, 'best_epoch': 0, 'vocabulary': vocab})
            restored = BoundaryChooser(len(vocab), 8, 1)
            expanded_initialization(restored, run / 'initialization.pt', vocab)
            for name, tensor in initial.state_dict().items():
                torch.testing.assert_close(tensor, restored.state_dict()[name], rtol=0, atol=0)
            (run / 'source').mkdir()
            for name in ('train_sharded.py', 'train_smoke.py', 'text_policy.py', 'text-policy.json',
                         'unicode-symbols.json', 'sample_subset.py'):
                shutil.copy2(ROOT / 'training' / name, run / 'source' / name)
            plan = {'seed': 123, 'samples': 2, 'rates': RATES,
                    'training': {'channels': 8, 'residual_blocks': 1, 'position_weighting': 'none',
                                 'min_side_characters': 2, 'batch_size': 8, 'max_tokens_per_batch': 128}}
            checks = []
            for arm in ('lr-1e-4', 'lr-2e-3'):
                steps_per_epoch = None
                for epochs in (1, 3):
                    completed = subprocess.run(training_command(run, plan, arm, epochs), cwd=ROOT,
                        env=dict(os.environ, DEBUG='0', PYTHONPATH=str(ROOT / 'training/.deps'),
                                 SUPER_READER_PROJECT_ROOT=str(ROOT)), capture_output=True, text=True, timeout=90)
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    report = json.loads((run / arm / 'validation-only.json').read_text())
                    self.assertFalse(report['test_evaluated'])
                    self.assertFalse((run / arm / 'smoke-metrics.json').exists())
                    checked = checkpoint_summary(run / arm / 'training-state.pt', plan, epochs)
                    if epochs == 1:
                        steps_per_epoch = checked['optimizer_steps']
                        self.assertGreater(steps_per_epoch, 0)
                    self.assertEqual(checked['optimizer_steps'], steps_per_epoch * epochs)
                    if epochs == 3:
                        checks.append(checked['optimizer_steps'])
            self.assertEqual(checks[0], checks[1])


if __name__ == '__main__':
    unittest.main()
