import argparse
from contextlib import redirect_stdout
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest

from prepare_training_snapshot import sha, snapshot
from text_policy import DATA_POLICY


class TrainingSnapshotTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'source'; source.mkdir()
        evaluation = root / 'evaluation'; evaluation.mkdir()
        for name in ('summary.json', 'vocabulary.json'):
            (evaluation / name).write_text(json.dumps(DATA_POLICY if name.startswith('summary') else {'<pad>': 0}))
        for split in ('validation', 'test'):
            (evaluation / f'{split}.jsonl').write_text(split + '\n')
        plan = {'policy': DATA_POLICY, 'evaluation': str(evaluation),
                'evaluation_summary_sha256': sha(evaluation / 'summary.json'),
                'vocabulary': {'path': str(evaluation / 'vocabulary.json'), 'sha256': sha(evaluation / 'vocabulary.json')},
                'holdouts': {split: {'path': str(evaluation / f'{split}.jsonl'),
                                     'sha256': sha(evaluation / f'{split}.jsonl')} for split in ('validation', 'test')},
                'historical_training_pair_deduplication': False}
        (source / 'plan.json').write_text(json.dumps(plan))
        rows = [['甲乙丙丁', 1, 0, 0, 0], ['甲乙丙丁', 0, 0, 0, 0]] * 4
        for i in range(7):
            folder = source / f'chunk-{i:06d}'; folder.mkdir()
            path = folder / 'data.jsonl.gz'
            with gzip.open(path, 'wt') as handle:
                for row in rows:
                    handle.write(json.dumps(row) + '\n')
            metadata = {'index': i, 'plan_sha256': sha(source / 'plan.json'), 'counts': {'samples': len(rows)},
                'files': {'data': {'name': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)}},
                'statistics': {'samples': len(rows), 'tokens': 32, 'random_baseline_sum': 8/3,
                    'center_correct': 4, 'maximum_sequence_length': 4,
                    'cells': [[8] + [0] * 9] + [[0] * 10 for _ in range(3)], 'position_histogram': [8] + [0] * 9}}
            (folder / 'metadata.json').write_text(json.dumps(metadata))
        # The seventh chunk was published after the captured status: exclude it.
        (source / 'status.json').write_text(json.dumps({'completed_chunks': 6, 'samples': 48}))
        (source / '.chunk-writing').mkdir()
        (source / '.chunk-writing/data.jsonl.gz').write_bytes(b'incomplete')
        return source

    def args(self, source, destination):
        return argparse.Namespace(source_dir=source, output_dir=destination, seed=123,
            target_samples=12, estimate_chunks=3, min_side_characters=2, max_sequence_length=256)

    def test_snapshot_only_uses_committed_data_with_unchanged_holdouts_and_deterministic_selection(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder); source = self.fixture(root)
            before = {str(p): sha(p) for p in source.rglob('*') if p.is_file()}
            manifests = []
            for name in ('one', 'two'):
                output = root / name
                snapshot(self.args(source, output))
                manifests.append(json.loads((output / 'manifest.json').read_text()))
                self.assertEqual((output / 'validation.jsonl').read_bytes(), b'validation\n')
                self.assertEqual((output / 'test.jsonl').read_bytes(), b'test\n')
                self.assertEqual((output / 'summary.json').read_bytes(), (root / 'evaluation/summary.json').read_bytes())
            self.assertEqual(manifests[0]['shards'], manifests[1]['shards'])
            self.assertEqual(len(manifests[0]['shards']), 3)
            self.assertTrue(all(s['preparation_chunk'] < 6 for s in manifests[0]['shards']))
            self.assertEqual(manifests[0]['selection']['estimated_effective_samples'], 12)
            self.assertTrue(manifests[0]['selection']['effective_count_is_estimate'])
            self.assertEqual(before, {str(p): sha(p) for p in source.rglob('*') if p.is_file()})
            with self.assertRaises(FileExistsError):
                snapshot(self.args(source, root / 'one'))

    def test_changed_holdout_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder); source = self.fixture(root)
            (root / 'evaluation/validation.jsonl').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'validation changed'):
                snapshot(self.args(source, root / 'out'))
            self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
