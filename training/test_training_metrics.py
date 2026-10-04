"""Reporting invariance, deterministic ties, and position-weight ablation contract."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from train_smoke import evaluate, target_ranks, weighted_loss_totals
from train_sharded import combined_weight_mean, read_training_shard, weight_table


class TrainingMetricsTests(unittest.TestCase):
    def test_weighted_loss_is_invariant_to_batch_partition(self):
        losses = torch.tensor([1., 2., 3., 4.])
        weights = torch.tensor([1., 1., 3., 3.])
        for batches in ([[0, 1], [2, 3]], [[0, 2], [1, 3]], [[0], [1], [2, 3]], [[0, 1, 2, 3]]):
            totals = [weighted_loss_totals(losses[ids], weights[ids]) for ids in batches]
            self.assertAlmostEqual(sum(x for x, _ in totals) / sum(w for _, w in totals), 3.)
        # The originally reported 1.25 counterexample must now report 1.
        a = weighted_loss_totals(torch.ones(2), weights[:2])
        b = weighted_loss_totals(torch.ones(2), weights[2:])
        self.assertEqual((a[0] + b[0]) / (a[1] + b[1]), 1.)

    def test_ranks_agree_with_stable_score_order_and_argmax(self):
        logits = torch.tensor([[0., 0., 0., 0.], [2., 4., 4., 1.], [3., 2., 1., -1e9]])
        order = logits.argsort(dim=1, descending=True, stable=True)
        for target in range(4):
            targets = torch.full((3,), target)
            ranks = target_ranks(logits, targets)
            expected = (order == target).long().argmax(dim=1) + 1
            torch.testing.assert_close(ranks, expected)
            torch.testing.assert_close(ranks == 1, logits.argmax(dim=1) == targets)

    def test_evaluate_tied_model_is_not_full_mrr(self):
        class TiedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))
            def forward(self, ids, token_mask, gap_mask):
                return torch.zeros_like(gap_mask, dtype=torch.float32).masked_fill(~gap_mask, -1e9)
        row = {'tokens': list('甲乙丙丁戊'), 'token_ids': [1, 2, 3, 4, 5],
               'target_index': 3, 'domain': 'test', 'training_weight': 1.}
        result = evaluate(TiedModel(), [row], 512, 8192)
        self.assertEqual(result['accuracy'], 0.)
        self.assertEqual(result['mean_reciprocal_rank'], .25)

    def test_position_ablation_retains_domains_and_normalizes_actual_subset(self):
        manifest = {'statistics': {'cells': [[1, 3], [0, 0]]}}
        on = weight_table(manifest)
        off = weight_table(manifest, 'none')
        self.assertEqual(on, [[2., 2/3], [0., 0.]])
        self.assertEqual(off, [[1., 1.], [1., 1.]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.jsonl'
            path.write_text(json.dumps(['甲乙', 0, 0, 0, 0]) + '\n')
            # A reference table can have population mean one but subset mean two.
            self.assertEqual(combined_weight_mean([path], on, [1.]), 2.)
            self.assertEqual(combined_weight_mean([path], off, [1.]), 1.)
            for table, expected in [(on, 6.), (off, 3.)]:
                rows = read_training_shard(path, {'<unk>': 0, '甲': 1, '乙': 2}, table, ['news'], [3.])
                self.assertEqual(rows[0]['training_weight'], expected)


if __name__ == '__main__':
    unittest.main()
