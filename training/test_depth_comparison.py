"""Guard control reuse, held-out selection, and edge-sensitive context accounting."""
from pathlib import Path
import tempfile
import unittest

from analyze_depth_comparison import coverage_bucket, length_bucket
from run_depth_comparison import select_depth, training_command
from run_learning_rate_comparison import training_command as control_command


class DepthComparisonTests(unittest.TestCase):
    def test_only_depth_and_output_change_from_control_command(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            plan = {'evaluation_dir': '/fixed/evaluation', 'seed': 2026092201,
                    'training': {'learning_rate': .00003, 'batch_size': 512,
                                 'max_tokens_per_batch': 8192, 'gradient_clip': 1},
                    'rates': {'layers16': .00003, 'layers20': .00003}}
            control = control_command(run / 'source', run, plan, 'layers16')
            candidate = training_command(run / 'source', run, plan)
            normalized = candidate.copy()
            for flag in ('--artifact-dir', '--residual-blocks'):
                normalized[normalized.index(flag) + 1] = control[control.index(flag) + 1]
            self.assertEqual(normalized, control)
            self.assertEqual(candidate[candidate.index('--initialize-from') + 1], str(run / 'initialization.pt'))
            self.assertIn('--defer-test', candidate)
            self.assertEqual(candidate[candidate.index('--residual-blocks') + 1], '10')
            (run / 'layers20').mkdir()
            (run / 'layers20/training-state.pt').touch()
            resume = training_command(run / 'source', run, plan)
            self.assertIn('--resume', resume)
            self.assertNotIn('--initialize-from', resume)
            self.assertEqual(training_command(run / 'source', run, plan, finalize=True),
                             [x for x in resume if x != '--defer-test'])

    def test_coverage_counts_each_side_instead_of_total_length(self):
        self.assertEqual(coverage_bucket(34, 16), 'covered-by-16')
        self.assertEqual(coverage_bucket(34, 0), 'not-fully-covered-by-20')
        self.assertEqual(coverage_bucket(34, 12), 'newly-covered-by-20')
        self.assertEqual(coverage_bucket(42, 20), 'newly-covered-by-20')
        self.assertEqual(coverage_bucket(43, 20), 'not-fully-covered-by-20')
        self.assertEqual(coverage_bucket(2, 0), 'covered-by-16')
        with self.assertRaises(ValueError):
            coverage_bucket(10, 9)
        self.assertEqual([length_bucket(n) for n in (17, 18, 34, 35, 42, 43, 128, 129)],
                         ['2-17', '18-34', '18-34', '35-42', '35-42', '43-64', '65-128', '129+'])

    def test_selection_uses_only_validation_and_keeps_control_on_tie(self):
        control = {'endpoint_selection_score': .89, 'test_accuracy': .1}
        candidate = {'endpoint_selection_score': .889, 'test_accuracy': 1.0}
        self.assertEqual(select_depth(control, candidate)['winner'], 'layers16')
        candidate['endpoint_selection_score'] = .89
        self.assertEqual(select_depth(control, candidate)['winner'], 'layers16')
        candidate['endpoint_selection_score'] = .90
        self.assertEqual(select_depth(control, candidate)['winner'], 'layers20')
        candidate['endpoint_selection_score'] = float('nan')
        with self.assertRaises(ValueError):
            select_depth(control, candidate)


if __name__ == '__main__':
    unittest.main()
