import json
from pathlib import Path
import random
import tempfile
import unittest

from run_symbol_ablation import retained, project, common_gaps, quotas, reservoir_add, load_records, matched_batches, batches


class RepresentationTests(unittest.TestCase):
    def test_projection_preserves_sides_and_recomputes_target(self):
        text = '他说:“明天见”然后离开'
        target = text.index('见')
        value, boundary, kept = project(text, target)
        self.assertEqual(value, '他说明天见然后离开')
        self.assertEqual(value[:boundary + 1], '他说明天见')
        self.assertEqual(value[boundary + 1:], '然后离开')
        self.assertNotIn(target, [i for i, _ in common_gaps(kept)])
        self.assertEqual([text[i] for i in kept], list(value))

    def test_plain_inputs_and_unicode_codepoint_targets(self):
        text = '𠮷甲ABC123乙'
        self.assertEqual(project(text, 2), (text, 2, list(range(len(text)))))
        self.assertTrue(retained('〇'))
        self.assertTrue(retained('３'))
        self.assertFalse(retained('é'))
        self.assertFalse(retained('あ'))
        self.assertFalse(retained(' '))
        self.assertEqual(project('金额-1.5元随后再说', 6)[0], '金额15元随后再说')

    def test_empty_sides_are_rejected_and_invalid_targets_raise(self):
        self.assertIsNone(project('”中文', 0))
        self.assertIsNone(project('中文”', 1))
        for target in (-1, 2):
            with self.assertRaises(ValueError):
                project('中文乙', target)

    def test_common_candidates_are_identical_original_gaps(self):
        text = '甲乙”丙丁A B'
        pure, _, kept = project(text, 0)
        for original_gap, pure_gap in common_gaps(kept):
            self.assertEqual(text[original_gap:original_gap + 2], pure[pure_gap:pure_gap + 2])
        self.assertEqual(common_gaps(kept), [(0, 0), (3, 2), (4, 3)])

    def test_source_quotas_preserve_total_and_proportions(self):
        result = quotas({'a': 10, 'b': 20, 'c': 70}, 101)
        self.assertEqual(result, {'a': 10, 'b': 20, 'c': 71})

    def test_reservoir_is_deterministic_without_replacement(self):
        def sample():
            pool, rng = [], random.Random(42)
            for i in range(1000):
                reservoir_add(pool, i, i + 1, 20, rng)
            return pool
        self.assertEqual(sample(), sample())
        self.assertEqual(len(set(sample())), 20)
        self.assertGreater(max(sample()), 500)

    def test_batches_preserve_pair_order_weights_and_padded_shapes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data.jsonl'
            rows = [['甲乙（AB）丙丁', 1, 'web', 1.5, str(i), 'doc'] for i in range(8)]
            rows += [['甲乙丙丁戊己', 2, 'web', .3, 'short', 'doc']]
            path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
            vocabulary = dict(zip('甲乙丙丁戊己AB（）', range(2, 12)))
            a, b = (load_records(path, vocabulary, arm) for arm in ('context', 'text-only'))
            ia, ib = matched_batches(a, 123, True), matched_batches(b, 123, True)
            self.assertEqual(ia, ib)
            for x, y in zip(batches(a, ia), batches(b, ib)):
                self.assertEqual(x['token_ids'].shape, y['token_ids'].shape)
                self.assertEqual([r['id'] for r in x['records']], [r['id'] for r in y['records']])
                self.assertEqual(x['sample_weights'].tolist(), y['sample_weights'].tolist())
                for batch in (x, y):
                    self.assertEqual(batch['token_ids'][batch['token_mask'] == 0].sum().item(), 0)
                    self.assertTrue(batch['gap_mask'].gather(1, batch['targets'][:, None]).all().item())


if __name__ == '__main__':
    unittest.main()
