"""Regression checks for fair channel expansion and pilot selection."""
import random
from pathlib import Path
import tempfile
import unittest

import torch

from run_width_comparison import select_shards
from train_sharded import expanded_initialization
from train_smoke import BoundaryChooser, configure_cpu, configure_mps
from widen_boundary import widen_state, verify, make_inputs


class PilotSelectionTests(unittest.TestCase):
    def test_sampling_preserves_sources_and_is_repeatable(self):
        base = {'statistics': {'samples': 100_000}, 'shards': [
            {'path': f'/corpus/{source}/train-{i:03d}.jsonl', 'bytes': 100}
            for source, size in [('wiki', 100), ('local', 50), ('web-old', 100), ('web-new', 100)]
            for i in range(size)]}
        chosen, groups = select_shards(base, 10_000, 1)
        self.assertEqual((chosen, groups), select_shards(base, 10_000, 1))
        self.assertEqual([g['selected_shards'] for g in groups], [5, 10, 10, 10])
        self.assertEqual(len({s['path'] for s in chosen}), len(chosen))
        self.assertNotEqual(chosen, select_shards(base, 10_000, 2)[0])
        for item in groups:
            selected = {Path(s['path']).name for s in chosen if str(Path(s['path']).parent) == item['source_directory']}
            self.assertNotEqual(selected, {f'train-{i:03d}.jsonl' for i in range(item['selected_shards'])})

    def test_rejects_invalid_targets(self):
        base = {'statistics': {'samples': 100}, 'shards': []}
        for size in (-1, 0, 101):
            with self.assertRaises(ValueError):
                select_shards(base, size, 1)


@unittest.skipUnless(torch.backends.mps.is_available(), 'Metal is required; no CPU training fallback')
class WideningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_cpu(1, 1)
        cls.device = configure_mps()

    def test_learned_affine_norm_padding_and_new_neurons(self):
        torch.manual_seed(8)
        vocabulary = {'<pad>': 0, '<unk>': 1, **{c: i + 2 for i, c in enumerate('这是一条用于比较模型表现的中文测试句子')}}
        old = BoundaryChooser(len(vocabulary), 12, 3)
        with torch.no_grad():
            for block in old.blocks:
                block.normalization.weight.uniform_(.2, 2)
                block.normalization.bias.uniform_(-1, 1)
                block.residual_scale.fill_(.8)
        wide = BoundaryChooser(len(vocabulary), 16, 3)
        widened = widen_state(old.state_dict(), wide.state_dict())
        wide.load_state_dict(widened)
        rng = random.Random(4)
        chars = list(vocabulary)[2:]
        texts = [''.join(rng.choices(chars, k=length)) for length in (2, 3, 15, 16, 17, 34, 80)]
        result = verify(old.to(self.device), wide.to(self.device), make_inputs(texts, vocabulary, self.device))
        self.assertTrue(result['passed'])
        for key, expected in widened.items():
            torch.testing.assert_close(wide.state_dict()[key].cpu(), expected, rtol=0, atol=0)
        # The existing trainer's same-width checkpoint initialization must load
        # every transformed value exactly, not apply another partial transfer.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pt'
            torch.save({'best_state': widened, 'model_state': widened, 'vocabulary': vocabulary, 'best_epoch': 1}, path)
            loaded = BoundaryChooser(len(vocabulary), 16, 3)
            expanded_initialization(loaded, path, vocabulary)
            for key, expected in widened.items():
                torch.testing.assert_close(loaded.state_dict()[key], expected, rtol=0, atol=0)

    def test_zero_variance_preserves_layernorm_epsilon(self):
        torch.manual_seed(3)
        old = BoundaryChooser(3, 6, 2)
        with torch.no_grad():
            old.embedding.weight.fill_(.8)
        wide = BoundaryChooser(3, 11, 2)
        wide.load_state_dict(widen_state(old.state_dict(), wide.state_dict()))
        inputs = make_inputs(['中文测试', '中文'], {'<unk>': 1}, self.device)
        with torch.inference_mode():
            actual = wide.to(self.device)(*inputs)
            expected = old.to(self.device)(*inputs)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-4)

    def test_rejects_structure_or_vocabulary_changes(self):
        old = BoundaryChooser(8, 6, 2)
        for new in (BoundaryChooser(8, 6, 2), BoundaryChooser(9, 12, 2), BoundaryChooser(8, 12, 3)):
            with self.assertRaises(ValueError):
                widen_state(old.state_dict(), new.state_dict())


if __name__ == '__main__':
    unittest.main()
