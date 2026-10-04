"""Guard baseline isolation, gap conventions, and gain decomposition denominators."""
import unittest

import numpy as np

from evaluate_position_baselines import check_predictions, summarize
from train_smoke import build_length_prior, center_baseline, length_prior_baseline


class PositionBaselineTests(unittest.TestCase):
    def test_length_prior_uses_training_only_and_falls_back_for_unseen_lengths(self):
        train = [{'tokens': '甲乙丙丁', 'target_index': t} for t in (0, 0, 1)]
        prior = build_length_prior(iter(train))
        self.assertEqual(prior, {4: 0})
        validation = [{'tokens': '甲乙丙丁', 'target_index': 2},
                      {'tokens': '甲乙丙丁戊', 'target_index': 2}]
        self.assertEqual(length_prior_baseline(validation, prior), .5)
        validation[0]['target_index'] = 0
        self.assertEqual(length_prior_baseline(validation, prior), 1.)
        self.assertEqual(prior, {4: 0})

    def test_center_gap_convention_and_prior_ties(self):
        rows = [{'tokens': '甲乙丙丁', 'target_index': 1},
                {'tokens': '甲乙丙丁戊', 'target_index': 2},
                {'tokens': '甲乙', 'target_index': 0}]
        self.assertEqual(center_baseline(rows), 1.)
        self.assertEqual(build_length_prior([
            {'tokens': '甲乙丙丁戊己', 'target_index': t} for t in (2, 3)
        ]), {6: 2})

    def test_gain_partitions_and_macro_have_explicit_denominators(self):
        lengths = np.array([10, 10, 10, 10])
        targets = np.array([4, 4, 4, 0])
        predictions = {'center': np.array([4, 4, 4, 4]),
                       'length-prior': np.array([4, 4, 4, 4]),
                       'inverse-cell': np.array([3, 4, 3, 0]),
                       'none': np.array([4, 4, 4, 4])}
        result = summarize(lengths, targets, predictions, np.array(['test'] * 4))
        self.assertEqual(result['overall']['accuracy']['none'], .75)
        self.assertEqual(result['overall']['accuracy']['inverse-cell'], .5)
        self.assertEqual(result['equal_gold_position_bin_accuracy']['none'], .5)
        self.assertAlmostEqual(result['equal_gold_position_bin_accuracy']['inverse-cell'], 2/3)
        self.assertAlmostEqual(result['random_gap_expected_accuracy'], 1/9)
        groups = result['gain_decomposition']['center']
        self.assertEqual(groups['baseline_correct']['net_correct'], 2)
        self.assertEqual(groups['baseline_wrong']['net_correct'], -1)
        self.assertEqual(sum(g['contribution_to_total_difference_pp'] for g in groups.values()), 25.)
        self.assertEqual(result['prediction_movement']['closer_to_center']['none_only_correct'], 2)
        self.assertEqual(result['prediction_movement']['closer_to_center']['weighted_only_correct'], 1)

    def test_invalid_gap_or_misaligned_predictions_are_rejected(self):
        for prediction in (np.array([0]), np.array([0, 3]), np.array([0., 1.])):
            with self.assertRaises(ValueError):
                check_predictions(np.array([4, 4]), np.array([0, 1]), {'bad': prediction})


if __name__ == '__main__':
    unittest.main()
