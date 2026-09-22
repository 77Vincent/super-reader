"""The full-data replay must not silently become an extra continuation epoch."""
import copy
from pathlib import Path
import tempfile
import unittest

from run_full_learning_rate_comparison import validate_control, OPTIMIZATION_KEYS
from run_learning_rate_comparison import training_command


class FullLearningRateComparisonTests(unittest.TestCase):
    def setUp(self):
        self.configuration = {
            'channels': 192, 'residual_blocks': 8, 'learning_rate': .0003,
            'max_shards': 0, 'validation_limit': 0, 'test_limit': 0,
            'seed': 2026090405, 'batch_size': 512, 'max_tokens_per_batch': 8192,
            'domain_weight_power': .65, 'selection_macro_weight': .5, 'gradient_clip': 1.,
        }
        self.metrics = {'seed': 2026090405, 'best_epoch': 1,
                        'history': [{'epoch': 1, 'validation_accuracy': .88}],
                        'validation': {'accuracy': .88}, 'data_sizes': {'train': 246266210}}
        self.manifest = {'statistics': {'samples': 246266210}}

    def test_rejects_partial_data_wrong_seed_or_wrong_reference_epoch(self):
        validate_control(self.configuration, self.metrics, self.manifest)
        for key, value in [('channels', 256), ('residual_blocks', 10), ('max_shards', 99),
                           ('validation_limit', 2500), ('seed', 2026092201)]:
            config = {**self.configuration, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_control(config, self.metrics, self.manifest)
        for changes in [{'best_epoch': 2}, {'data_sizes': {'train': 10015214}},
                        {'history': [{'epoch': 1, 'validation_accuracy': .87}]}]:
            with self.assertRaises(ValueError):
                validate_control(self.configuration, {**self.metrics, **changes}, self.manifest)

    def test_full_run_preserves_original_seed_checkpoint_frequency_and_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            plan = {'manifest_name': 'manifest.json', 'checkpoint_shards': 4,
                    'evaluation_dir': '/original/full-data', 'seed': self.configuration['seed'],
                    'training': {key: self.configuration[key] for key in OPTIMIZATION_KEYS},
                    'rates': {'lr-3e-4-control': .0003, 'lr-3e-5': .00003}}
            original = training_command(run / 'source', run, plan, 'lr-3e-4-control')
            candidate = training_command(run / 'source', run, plan, 'lr-3e-5')
            normalized = copy.copy(candidate)
            for flag in ('--learning-rate', '--artifact-dir'):
                normalized[normalized.index(flag) + 1] = original[original.index(flag) + 1]
            self.assertEqual(normalized, original)
            for flag, value in [('--manifest', str(run / 'manifest.json')), ('--checkpoint-shards', '4'),
                                ('--seed', '2026090405'), ('--epochs', '1'),
                                ('--initialize-from', str(run / 'initialization.pt'))]:
                self.assertEqual(candidate[candidate.index(flag) + 1], value)
            self.assertNotIn('--resume', candidate)
            self.assertNotIn('--max-shards', candidate)
            (run / 'lr-3e-5').mkdir()
            (run / 'lr-3e-5/training-state.pt').touch()
            resumed = training_command(run / 'source', run, plan, 'lr-3e-5')
            self.assertIn('--resume', resumed)
            self.assertNotIn('--initialize-from', resumed)
            self.assertEqual(training_command(run / 'source', run, plan, 'lr-3e-5', finalize=True),
                             [arg for arg in resumed if arg != '--defer-test'])


if __name__ == '__main__':
    unittest.main()
