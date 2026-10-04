"""The paired smoke must hold all options except position weighting fixed."""
from pathlib import Path
import json
import tempfile
import unittest
from run_position_ablation import allocate, command


class PositionAblationTests(unittest.TestCase):
    def test_source_quotas_preserve_total(self):
        self.assertEqual(allocate({'a': 90, 'b': 10}, 10), {'a': 9, 'b': 1})
        counts = {'a': 21365413, 'b': 24711553, 'c': 100000000, 'd': 150000000}
        self.assertEqual(sum(allocate(counts, 1000000).values()), 1000000)

    def test_only_position_weighting_and_output_directory_differ(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            a, b = command(run, 'inverse-cell', 99), command(run, 'none', 99)
            for flag in ('--artifact-dir', '--position-weighting'):
                a[a.index(flag) + 1] = b[b.index(flag) + 1]
            self.assertEqual(a, b)
            self.assertIn('--defer-test', a)
            self.assertIn('--weighting-manifest', a)
            self.assertIn('--initialize-from', a)
            self.assertNotIn('--domain-weight-power', a)
            (run / 'none').mkdir()
            (run / 'none/training-state.pt').touch()
            resumed = command(run, 'none', 99)
            self.assertIn('--resume', resumed)
            self.assertNotIn('--initialize-from', resumed)

    def test_old_frozen_experiment_keeps_its_recorded_weighting(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'plan.json').write_text(json.dumps({'training': {'domain_weight_power': .65}}))
            old = command(run, 'none', 99)
            self.assertEqual(old[old.index('--domain-weight-power') + 1], '0.65')


if __name__ == '__main__':
    unittest.main()
