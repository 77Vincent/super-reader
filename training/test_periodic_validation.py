"""Exercise real optimizer updates across validation, interruption and new data."""
import copy
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

import train_sharded as trainer
from text_policy import DATA_POLICY
from train_smoke import BoundaryChooser, TrainingStopRequested


class PeriodicValidationTests(unittest.TestCase):
    def fixture(self, root):
        root.mkdir(exist_ok=True)
        vocabulary = {'<pad>': 0, '<unk>': 1}
        for char in '0123456789:、ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz 甲乙丙丁戊己':
            if char not in vocabulary:
                vocabulary[char] = len(vocabulary)
        (root / 'vocabulary.json').write_text(json.dumps(vocabulary))
        shards = []
        for index in range(4):
            rows = [['甲乙丙丁戊己', target, 0, 0, 0] for target in (1, 2, 3)]
            path = root / f'shard-{index}.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            shards.append({'path': str(path)})
        manifest = {**DATA_POLICY, 'format': 'super-reader-sharded-training-v1',
                    'corpus_sources': ['web'], 'statistics': {'cells': [[12]]},
                    'vocabulary_path': str(root / 'vocabulary.json'), 'shards': shards}
        (root / 'manifest.json').write_text(json.dumps(manifest))
        (root / 'summary.json').write_text(json.dumps(DATA_POLICY))
        rows = [{'tokens': list('甲乙丙丁戊己'), 'target_index': i, 'punctuation': '，'} for i in (1, 2)]
        (root / 'validation.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        (root / 'test.jsonl').write_text('never read the test set')

    def run_training(self, root, artifact, extra=(), evaluator=None):
        argv = ['train_sharded.py', '--manifest', str(root / 'manifest.json'),
                '--data-dir', str(root), '--artifact-dir', str(artifact), '--epochs', '1',
                '--channels', '8', '--residual-blocks', '1', '--min-side-characters', '2',
                '--max-sequence-length', '256', '--batch-size', '3', '--max-tokens-per-batch', '64',
                '--learning-rate', '0.0003', '--checkpoint-shards', '2', '--defer-test', *extra]
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            with ExitStack() as stack:
                stack.enter_context(redirect_stdout(io.StringIO()))
                stack.enter_context(patch.object(sys, 'argv', argv))
                stack.enter_context(patch.object(trainer, 'configure_mps', return_value=torch.device('cpu')))
                stack.enter_context(patch.object(trainer, 'configure_cpu', side_effect=lambda *a: torch.set_num_threads(1)))
                stack.enter_context(patch.object(trainer, 'mps_memory_restart_needed', return_value=False))
                stack.enter_context(patch.object(torch.mps, 'empty_cache'))
                stack.enter_context(patch.object(torch.mps, 'driver_allocated_memory', return_value=0))
                if evaluator:
                    stack.enter_context(patch.object(trainer, 'evaluate', side_effect=evaluator))
                trainer.main()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        return torch.load(artifact / 'training-state.pt', map_location='cpu', weights_only=True)

    def assert_same_training(self, first, second):
        for key, value in first['model_state'].items():
            torch.testing.assert_close(value, second['model_state'][key], rtol=0, atol=0)
        for index, state in first['optimizer_state']['state'].items():
            for key, value in state.items():
                torch.testing.assert_close(value, second['optimizer_state']['state'][index][key], rtol=0, atol=0)

    def test_validation_does_not_change_updates_and_archives_resume_at_next_shard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.fixture(root)
            plain = self.run_training(root, root / 'plain')
            periodic = self.run_training(root, root / 'periodic', ['--validation-every-samples', '4'])
            self.assert_same_training(plain, periodic)
            self.assertEqual([r['samples'] for r in periodic['validation_history']], [6, 9, 12])
            for row in periodic['validation_history']:
                saved = torch.load(row['checkpoint'], map_location='cpu', weights_only=True)
                self.assertEqual(saved['progress']['next_batch'], 0)
                self.assertEqual(saved['last_validation']['samples'], row['samples'])
            self.assertEqual(periodic['best_validation'], max(r['validation']['accuracy'] for r in periodic['validation_history']))
            resumed = self.run_training(root, root / 'periodic', ['--validation-every-samples', '4', '--resume'])
            self.assert_same_training(periodic, resumed)
            self.assertEqual(resumed['validation_history'], periodic['validation_history'])

    def test_interrupted_validation_retries_without_repeating_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.fixture(root)
            expected = self.run_training(root, root / 'expected', ['--validation-every-samples', '4'])
            def interrupt(model, *args, **kwargs):
                model.eval()
                raise TrainingStopRequested
            stopped = self.run_training(root, root / 'stopped', ['--validation-every-samples', '4'], interrupt)
            self.assertEqual(stopped['progress']['seen_examples'], 6)
            self.assertEqual(stopped['progress']['shard'], 2)
            self.assertEqual(stopped['validation_history'], [])
            resumed = self.run_training(root, root / 'stopped', ['--validation-every-samples', '4', '--resume'])
            self.assert_same_training(expected, resumed)
            self.assertEqual([r['samples'] for r in resumed['validation_history']], [6, 9, 12])

    def test_validation_restores_mode_even_when_interrupted(self):
        model = BoundaryChooser(4, 8, 1)
        for training in (True, False):
            model.train(training)
            def interrupt(model, *args, **kwargs):
                model.eval()
                raise TrainingStopRequested
            with patch.object(trainer, 'evaluate', side_effect=interrupt):
                with self.assertRaises(TrainingStopRequested):
                    trainer.evaluate_preserving_mode(model, [], 3, 64)
            self.assertEqual(model.training, training)

    def test_mid_epoch_best_is_kept_even_if_endpoint_is_worse(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.fixture(root)
            original = trainer.evaluate
            scores = iter([.99, .5, .4, .4])
            def evaluate(model, *args, **kwargs):
                return {**original(model, *args, **kwargs), 'accuracy': next(scores)}
            state = self.run_training(root, root / 'candidate', ['--validation-every-samples', '4'], evaluate)
            self.assertEqual(state['best_position'], {'epoch': 1, 'samples': 6})
            saved = torch.load(state['validation_history'][0]['checkpoint'], weights_only=True)
            for key, value in state['best_state'].items():
                torch.testing.assert_close(value, saved['model_state'][key], rtol=0, atol=0)

    def test_new_manifest_continuation_preserves_optimizer_and_requires_epoch_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old = root / 'old'; new = root / 'new'
            self.fixture(old); self.fixture(new)
            parent = self.run_training(old, root / 'parent')
            checkpoint = root / 'parent/training-state.pt'
            child = self.run_training(new, root / 'child',
                ['--continue-from', str(checkpoint), '--epochs', '2', '--validation-every-samples', '4'])
            self.assertEqual({int(v['step']) for v in parent['optimizer_state']['state'].values()}, {4})
            self.assertEqual({int(v['step']) for v in child['optimizer_state']['state'].values()}, {8})
            self.assertEqual(child['history'][0]['epoch'], 2)
            self.assertEqual(child['progress']['epoch'], 3)
            self.assertNotEqual(parent['data_identity']['manifest_sha256'], child['data_identity']['manifest_sha256'])
            config = child['configuration']; identity = child['data_identity']
            trainer.validate_continuation(parent, config, parent['vocabulary'], identity)
            with self.assertRaisesRegex(ValueError, 'learning_rate'):
                trainer.validate_continuation(parent, {**config, 'learning_rate': .002}, parent['vocabulary'], identity)
            with self.assertRaisesRegex(ValueError, 'vocabulary'):
                trainer.validate_continuation(parent, config, {}, identity)
            broken = copy.deepcopy(parent); broken['progress']['shard'] = 1
            with self.assertRaisesRegex(ValueError, 'completed epoch'):
                trainer.validate_continuation(broken, config, parent['vocabulary'], identity)

    def test_schedule_uses_fixed_multiples_and_starts_again_each_epoch(self):
        self.assertFalse(trainer.validation_due(50, 3, 49, {}))
        self.assertTrue(trainer.validation_due(50, 3, 52, {}))
        self.assertFalse(trainer.validation_due(50, 3, 99, {'epoch': 3, 'samples': 52}))
        self.assertTrue(trainer.validation_due(50, 3, 101, {'epoch': 3, 'samples': 52}))
        self.assertTrue(trainer.validation_due(50, 4, 51, {'epoch': 3, 'samples': 400}))
        self.assertFalse(trainer.validation_due(0, 4, 51, {}))


if __name__ == '__main__':
    unittest.main()
