"""Prevent mismatched labels, sample budgets, batching and resumed inputs."""
from collections import Counter
import json
from pathlib import Path
import random
import tempfile
import unittest

from prepare_source_comparison import matched_quotas, sample_pool, stratum, scan, weighting_summary
from run_source_comparison import training_command
from run_web_continuation import write


class SourceComparisonTests(unittest.TestCase):
    def test_capped_quotas_are_exact_repeatable_and_on_common_support(self):
        rng = random.Random(5)
        for _ in range(100):
            a = {str(i): rng.randrange(1, 100) for i in range(30)}
            b = {str(i): rng.randrange(0, 50) for i in range(25)}
            cap = sum(min(n, b.get(k, 0)) for k, n in a.items())
            target = rng.randrange(1, cap + 1)
            quotas = matched_quotas(a, b, target)
            self.assertEqual(sum(quotas.values()), target)
            self.assertEqual(quotas, matched_quotas(a, b, target))
            self.assertTrue(all(0 < n <= min(a[k], b[k]) for k, n in quotas.items()))
        with self.assertRaises(ValueError):
            matched_quotas({'a': 3}, {'a': 1}, 2)
        self.assertEqual(matched_quotas({'a': 90, 'b': 10}, {'a': 10, 'b': 10}, 15), {'a': 10, 'b': 5})

    def test_paired_sampling_preserves_text_target_and_shard_lengths_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            domains = ['web', 'wikipedia', 'synthetic_multistyle']
            base = {'domains': domains, 'position_bins': 10, 'statistics': {'cells': [[0] * 10]},
                    'evaluation_source_dir': '/fixed/eval', 'format': 'super-reader-sharded-training-v1'}
            pools, original, counts = {}, {}, {}
            for arm, domain_ids in [('a', [0, 1, 2]), ('b', [1, 2])]:
                rows = [[('甲' if arm == 'a' else '乙') + str(i).zfill(3) + '中文' * (i % 2 + 1),
                         2, domain_ids[i % len(domain_ids)], 0, 4] for i in range(90)]
                path = run / (arm + '-input.jsonl')
                path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
                result = scan(path, domains, domains, lambda: False)
                pools[arm] = [{k: result[k] for k in ('path', 'bytes', 'sha256')}]
                counts[arm] = result['counts']
                original[arm] = {(row[0], row[1]): domains[row[2]] for row in rows}
            quotas = matched_quotas(counts['a'], counts['b'], 60)
            write(run / 'quotas.json', quotas)
            manifests = {}
            for arm, used_domains in [('a', domains), ('b', domains[1:])]:
                manifests[arm] = sample_pool(run, arm, pools[arm], base, counts[arm], quotas,
                    used_domains, 7, 4, lambda: False, lambda *a, **k: None)
                restored = sample_pool(run, arm, pools[arm], base, counts[arm], quotas,
                    used_domains, 7, 4, lambda: False, lambda *a, **k: None)
                self.assertEqual(manifests[arm], restored)
                seen, histogram = set(), Counter()
                for item in manifests[arm]['shards']:
                    for line in Path(item['path']).read_text().splitlines():
                        row = json.loads(line)
                        self.assertNotIn(row[0], seen)
                        seen.add(row[0])
                        self.assertEqual(used_domains[row[2]], original[arm][(row[0], row[1])])
                        histogram[stratum(row)] += 1
                self.assertEqual(histogram, quotas)
                self.assertAlmostEqual(sum(weighting_summary(manifests[arm])['effective_loss_share'].values()), 1.)
            for a, b in zip(manifests['a']['shards'], manifests['b']['shards']):
                self.assertEqual(a['length_histogram'], b['length_histogram'])
                self.assertEqual(a['batches'], b['batches'])
            with self.assertRaises(InterruptedError):
                sample_pool(run, 'interrupted', pools['a'], base, counts['a'], quotas, domains,
                            7, 4, lambda: True, lambda *a, **k: None)
            self.assertFalse((run / 'interrupted-manifest.json').exists())
            recovered = sample_pool(run, 'interrupted', pools['a'], base, counts['a'], quotas, domains,
                                    7, 4, lambda: False, lambda *a, **k: None)
            self.assertEqual([s['sha256'] for s in recovered['shards']],
                             [s['sha256'] for s in manifests['a']['shards']])
            source = Path(pools['a'][0]['path'])
            source.write_text(source.read_text().replace('甲', '丙'))
            with self.assertRaisesRegex(ValueError, 'Input changed'):
                sample_pool(run, 'changed-input', pools['a'], base, counts['a'], quotas, domains,
                            7, 4, lambda: False, lambda *a, **k: None)
            self.assertFalse((run / 'changed-input-manifest.json').exists())
            first = Path(manifests['a']['shards'][0]['path'])
            first.write_text(first.read_text() + '\n')
            with self.assertRaises(ValueError):
                sample_pool(run, 'a', pools['a'], base, counts['a'], quotas, domains,
                            7, 4, lambda: False, lambda *a, **k: None)

    def test_training_commands_change_only_data_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            plan = {'evaluation_dir': '/fixed/eval', 'seed': 2,
                    'training': {'learning_rate': .00003, 'domain_weight_power': .65}}
            a = training_command(run, run, plan, 'current-mix')
            b = training_command(run, run, plan, 'wiki-synthetic')
            for flag in ('--manifest', '--artifact-dir'):
                b[b.index(flag) + 1] = a[a.index(flag) + 1]
            self.assertEqual(a, b)
            self.assertIn('--defer-test', a)
            self.assertIn('--initialize-from', a)
            with self.assertRaises(FileNotFoundError):
                training_command(run, run, plan, 'current-mix', finalize=True)
            (run / 'current-mix').mkdir()
            (run / 'current-mix/training-state.pt').touch()
            resumed = training_command(run, run, plan, 'current-mix')
            self.assertIn('--resume', resumed)
            self.assertNotIn('--initialize-from', resumed)
            self.assertNotIn('--defer-test', training_command(run, run, plan, 'current-mix', finalize=True))


if __name__ == '__main__':
    unittest.main()
