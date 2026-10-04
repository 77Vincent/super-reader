import unittest

from verify_initial_learning_rate_results import paired_document_interval


class PairedValidationTests(unittest.TestCase):
    def test_document_resampling_keeps_rows_together_and_counts_reconcile(self):
        result = paired_document_interval([True, True, False, False], [False, True, True, False],
                                          ['a', 'a', 'b', 'b'], 42, draws=1000)
        self.assertEqual(result['documents'], 2)
        self.assertEqual(result['difference_pp'], 0.)
        for key in ('first_only_correct', 'second_only_correct', 'both_correct', 'both_wrong'):
            self.assertEqual(result[key], 1)
        self.assertEqual(result['paired_document_bootstrap_95_percent_interval_pp'], [-50., 50.])

    def test_identical_predictions_have_zero_difference_and_bad_alignment_is_rejected(self):
        result = paired_document_interval([True, False], [True, False], ['a', 'b'], 1, draws=100)
        self.assertEqual(result['paired_document_bootstrap_95_percent_interval_pp'], [0., 0.])
        with self.assertRaises(ValueError):
            paired_document_interval([True], [False, True], ['a'], 1)


if __name__ == '__main__':
    unittest.main()
