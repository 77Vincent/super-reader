"""Learning-rate trials must preserve optimization and select without test scores."""
import copy
from pathlib import Path
import tempfile
import unittest

from run_learning_rate_comparison import optimization_signature, select_candidate, training_command


class LearningRateComparisonTests(unittest.TestCase):
    def test_selection_uses_validation_and_keeps_initial_on_ties(self):
        endpoints = {'high-test': {'selection_score': .88, 'test_accuracy': 1.},
                     'best-validation': {'selection_score': .90, 'test_accuracy': .1}}
        self.assertEqual(select_candidate(.89, endpoints)['winner'], 'best-validation')
        self.assertEqual(select_candidate(.90, endpoints)['winner'], 'initial')
        endpoints['tie'] = {'selection_score': .90}
        self.assertEqual(select_candidate(.89, endpoints)['winner'], 'best-validation')
        with self.assertRaises(ValueError):
            select_candidate(.89, {'invalid': {'selection_score': float('nan')}})

    def test_commands_change_only_learning_rate_and_output_and_resume_selected_test(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            plan = {'evaluation_dir': '/fixed/eval', 'seed': 123,
                    'training': {'learning_rate': .0003, 'batch_size': 512, 'max_tokens_per_batch': 8192},
                    'rates': {'a': .0001, 'b': .00003}}
            a = training_command(run / 'source', run, plan, 'a')
            b = training_command(run / 'source', run, plan, 'b')
            normalized = copy.copy(b)
            for flag in ['--learning-rate', '--artifact-dir']:
                normalized[normalized.index(flag) + 1] = a[a.index(flag) + 1]
            self.assertEqual(a, normalized)
            self.assertIn('--defer-test', a)
            self.assertIn('--initialize-from', a)
            with self.assertRaises(FileNotFoundError):
                training_command(run / 'source', run, plan, 'a', finalize=True)
            (run / 'a').mkdir()
            (run / 'a/training-state.pt').touch()
            resumed = training_command(run / 'source', run, plan, 'a')
            self.assertIn('--resume', resumed)
            self.assertNotIn('--initialize-from', resumed)
            final = training_command(run / 'source', run, plan, 'a', finalize=True)
            self.assertEqual(final, [x for x in resumed if x != '--defer-test'])

    def test_optimization_signature_detects_training_changes_but_allows_reporting(self):
        original = '''
def parse_arguments():
    parser.add_argument('--learning-rate', default=.0003)
    return parser.parse_args()
def main():
    optimizer.step()
    if best_state is None:
        raise RuntimeError('no model')
    restore_state(model, best_state)
'''
        reporting = original.replace("    return parser.parse_args()",
            "    parser.add_argument('--defer-test', action='store_true')\n    return parser.parse_args()")
        reporting = reporting.replace('    restore_state(model, best_state)',
            '    if args.defer_test:\n        save_validation()\n        return\n    restore_state(model, best_state)')
        self.assertEqual(optimization_signature(original), optimization_signature(reporting))
        self.assertNotEqual(optimization_signature(original), optimization_signature(reporting.replace(
            'optimizer.step()', 'optimizer.zero_grad()')))


if __name__ == '__main__':
    unittest.main()
