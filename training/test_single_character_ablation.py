"""The experimental filter removes rows, without modifying labels or evaluation."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from run_single_character_ablation import filter_shard, group_result, single_character_side


class SingleCharacterAblationTests(unittest.TestCase):
    def test_only_exactly_one_code_point_sides_are_excluded(self):
        self.assertTrue(single_character_side('甲乙', 0))
        self.assertTrue(single_character_side('𠀀甲乙丙', 0))
        self.assertTrue(single_character_side('甲乙丙丁', 2))
        self.assertFalse(single_character_side('因此继续', 1))
        self.assertFalse(single_character_side('甲乙丙丁戊', 2))
        for target in (-1, 3):
            with self.assertRaises(ValueError):
                single_character_side('甲乙丙丁', target)

    def test_rows_and_provenance_remain_aligned_and_unmodified(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'input.jsonl', Path(directory) / 'output.jsonl'
            rows = [['但继续前进', 0, 0, 0, 2], ['因此继续', 1, 0, 0, 5],
                    ['德宗闻之怒', 3, 0, 0, 8], ['另有两个片段', 1, 0, 0, 3]]
            lines = [json.dumps(r, ensure_ascii=False) + '\n' for r in rows]
            origins = [json.dumps(['source', i]) + '\n' for i in range(4)]
            source.write_text(''.join(lines))
            source.with_suffix('.source.jsonl').write_text(''.join(origins))
            statistics = {'samples': 0, 'tokens': 0, 'cells': [[0] * 10 for _ in range(4)],
                          'domain_samples': Counter(), 'maximum_sequence_length': 0,
                          'random_baseline_sum': 0., 'center_correct': 0}
            self.assertEqual(filter_shard(source, output, ['test'], statistics), 2)
            self.assertEqual(output.read_text(), lines[1] + lines[3])
            self.assertEqual(output.with_suffix('.source.jsonl').read_text(), origins[1] + origins[3])
            self.assertEqual(source.read_text(), ''.join(lines))
            self.assertEqual(statistics['samples'], 2)
            self.assertEqual(statistics['tokens'], 10)
            source.with_suffix('.source.jsonl').write_text(''.join(origins[:-1]))
            with self.assertRaises(ValueError):
                filter_shard(source, output, ['test'], statistics)

    def test_paired_gain_and_loss_counts_and_zero_change_interval(self):
        before = np.array([True, False, True, False])
        after = np.array([True, True, False, True])
        docs = np.array(['a', 'a', 'b', 'b'])
        result = group_result(before, after, docs, 1)
        self.assertEqual(result['count'], 4)
        self.assertEqual(result['newly_correct'], 2)
        self.assertEqual(result['newly_wrong'], 1)
        self.assertEqual(result['difference_pp'], 25.)
        same = group_result(before, before, docs, 1)
        self.assertEqual(same['paired_document_bootstrap_95_percent_interval_pp'], [0., 0.])


if __name__ == '__main__':
    unittest.main()
