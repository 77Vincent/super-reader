"""Offline fixtures for label fidelity, inherited-data exclusion and recovery."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from prepare_web_data import (DATA_POLICY, Bloom, add_statistics, document_signature, document_split,
    fresh_statistics, labeled_samples, prepare, projected_key, quota, read, sha, write)
from prepare_synthetic_data import adjacent_samples
import pyarrow as pa
import pyarrow.parquet as pq


class WebPreparationTests(unittest.TestCase):
    def test_same_targets_and_preserved_context(self):
        texts = ["女：我们计划8:30出发，吃苹果、香蕉。请准时到达。", "甲，；乙。丙", "价格3.14元，库存1,200件。明天再买", "甲！）乙，丙。"]
        for text in texts:
            self.assertEqual([(t, i) for t, i, _ in labeled_samples(text)], adjacent_samples(text))
        first = list(labeled_samples(texts[0]))[0]
        self.assertIn("8:30", first[0])
        self.assertIn("、", first[0])
        self.assertEqual(first[2], "，")

    def test_projection_and_bloom(self):
        key = projected_key("甲：8:30乙、丙")
        self.assertEqual(key, projected_key("甲乙丙"))
        bloom = Bloom(8192)
        self.assertFalse(bloom.contains(key))
        bloom.add(key)
        self.assertTrue(bloom.contains(key))
        self.assertFalse(bloom.contains(projected_key("完全不同的文本")))
        self.assertEqual(sum(quota(20_000_000, 256, i) for i in range(256)), 20_000_000)

    def fixture(self, root):
        raw, evaluation, base = (root / name for name in ("raw", "eval", "base"))
        for path in (raw, evaluation, base):
            path.mkdir()
        by_split = {name: [] for name in ("train", "validation", "test")}
        index = 0
        while any(len(values) < 24 for values in by_split.values()):
            nonce = chr(0x6000 + index)
            text = "，".join(f"{nonce}{chr(0x5000 + j)}这是足够长且保留上下文的自然中文句子" for j in range(8)) + "。"
            split = document_split(document_signature(text), 20260915)
            if len(by_split[split]) < 24:
                by_split[split].append(text)
            index += 1
        files = []
        for number in range(2):
            docs = []
            for i in range(12):
                docs += [by_split[split][number * 12 + i] for split in ("validation", "test", "train")]
            path = raw / f"{number}.parquet"
            pq.write_table(pa.table({"content": docs, "source": [f"source-{number}"] * len(docs)}), path)
            files.append({"filename": path.name, "bytes": path.stat().st_size, "sha256": sha(path)})
        write(raw / "download-manifest.json", {"repository": "fixture", "revision": "fixture-revision", "files": files})
        write(raw / "verified-files.json", {"revision": "fixture-revision", "files": {x["filename"]: {"bytes": x["bytes"], "sha256": x["sha256"]} for x in files}})
        forbidden, target, _ = next(labeled_samples(by_split["train"][0]))
        stats = fresh_statistics()
        bucket, position = add_statistics(stats, forbidden, target)
        stats["domain_samples"] = {"news": 1}
        shard = base / "train.jsonl"
        shard.write_text(json.dumps([forbidden, target, 0, bucket, position], ensure_ascii=False) + "\n")
        write(base / "vocabulary.json", {"<pad>": 0, "<unk>": 1})
        manifest = {**DATA_POLICY, "format": "super-reader-sharded-training-v1", "domains": ["news"],
            "statistics": stats, "shards": [{"path": str(shard), "bytes": shard.stat().st_size}],
            "evaluation_source_dir": str(evaluation), "vocabulary_path": str(base / "vocabulary.json")}
        write(base / "manifest.json", manifest)
        splits = {}
        for split, text in [("validation", "旧验证文本保留原样"), ("test", "旧测试文本保留原样")]:
            path = evaluation / f"{split}.jsonl"
            path.write_text(json.dumps({"id": split, "domain": "news", "tokens": list(text), "target_index": 2, "punctuation": "，", "tokenization": "character"}, ensure_ascii=False) + "\n")
            splits[split] = {"count": 1, "per_domain": {"news": 1}, "sha256": sha(path)}
        write(evaluation / "summary.json", {**DATA_POLICY, "splits": splits})
        write(evaluation / "holdout-document-hashes.json", [document_signature(by_split["test"][0])])
        args = argparse.Namespace(base_manifest=base / "manifest.json", evaluation=evaluation, raw_dir=raw,
            output_dir=root / "output", target_samples=100, shards=4, seed=20260915, bloom_bytes=1024 * 1024, checkpoint_documents=1,
            freeze_evaluation=False)
        return args, projected_key(forbidden)

    def run_quietly(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(args)

    def test_full_pipeline_and_recovery_do_not_leak_or_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, forbidden = self.fixture(root)
            self.run_quietly(args)
            manifest = read(args.output_dir / "manifest.json")
            self.assertEqual(manifest["statistics"]["samples"], 101)
            self.assertEqual(len(manifest["web_source"]["files"]), 2)
            seen = {forbidden}
            for item in manifest["shards"][1:]:
                for line in Path(item["path"]).read_text().splitlines():
                    key = projected_key(json.loads(line)[0])
                    self.assertNotIn(key, seen)
                    seen.add(key)
            for split in ("validation", "test"):
                data = (args.output_dir / f"{split}.jsonl").read_bytes()
                self.assertTrue(data.startswith((args.evaluation / f"{split}.jsonl").read_bytes()))
                for line in data.splitlines()[1:]:
                    record = json.loads(line)
                    self.assertEqual(document_split(record["document_id"].removeprefix("web:"), args.seed), split)
                    key = projected_key("".join(record["tokens"]))
                    self.assertNotIn(key, seen)
                    seen.add(key)
            expected = {p.name: p.read_bytes() for p in args.output_dir.glob("web-*.jsonl.part")}
            resumed = argparse.Namespace(**{**vars(args), "output_dir": root / "resumed"})
            import prepare_web_data as implementation
            real_write = implementation.write
            def interrupt_after_checkpoint(path, value):
                real_write(path, value)
                if path.name == "preparation-state.json" and value["documents"] == 4:
                    raise InterruptedError("fixture crash")
            with patch.object(implementation, "write", side_effect=interrupt_after_checkpoint):
                with self.assertRaises(InterruptedError):
                    self.run_quietly(resumed)
            # Bytes beyond the checkpoint must be discarded before restoring dedup state.
            with (resumed.output_dir / "web-train-000.jsonl.part").open("a") as handle:
                handle.write('["不能恢复的半条记录"')
            self.run_quietly(resumed)
            for name, content in expected.items():
                self.assertEqual((resumed.output_dir / name).read_bytes(), content)
            self.run_quietly(resumed)  # Completed run is idempotent.

    def test_expansion_preserves_all_data_and_holds_evaluation_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, forbidden = self.fixture(root)
            self.run_quietly(args)
            previous = read(args.output_dir / "manifest.json")
            extension = argparse.Namespace(**{**vars(args), "output_dir": root / "extension",
                "base_manifest": args.output_dir / "manifest.json", "evaluation": args.output_dir,
                "target_samples": 40, "freeze_evaluation": True})
            # Earlier runs did not copy the original document registry forward.
            (args.output_dir / "holdout-document-hashes.json").unlink()
            self.run_quietly(extension)
            manifest = read(extension.output_dir / "manifest.json")
            self.assertEqual(manifest["statistics"]["samples"], 141)
            self.assertEqual(manifest["statistics"]["domain_samples"], {"news": 1, "web": 140})
            self.assertEqual(manifest["domains"], ["news", "web"])
            self.assertEqual(manifest["shards"][:len(previous["shards"])], previous["shards"])
            self.assertEqual(manifest["web_source"]["added_training_samples"], 40)
            self.assertGreater(manifest["web_statistics"]["reserved_holdout_documents"], 0)
            for split in ("validation", "test"):
                self.assertEqual((extension.output_dir / f"{split}.jsonl").read_bytes(), (args.output_dir / f"{split}.jsonl").read_bytes())
            reserved = set()
            for source in args.raw_dir.glob("*.parquet"):
                for text in pq.read_table(source, columns=["content"]).column(0).to_pylist():
                    if document_split(document_signature(text), args.seed) != "train":
                        reserved.update(projected_key(t) for t, _, _ in labeled_samples(text))
            seen = {forbidden}
            for shard in previous["shards"]:
                seen.update(projected_key(json.loads(line)[0]) for line in Path(shard["path"]).read_text().splitlines())
            for shard in manifest["shards"][len(previous["shards"]):]:
                for line in Path(shard["path"]).read_text().splitlines():
                    text, target, domain, _, _ = json.loads(line)
                    key = projected_key(text)
                    self.assertNotIn(key, seen)
                    self.assertNotIn(key, reserved)
                    self.assertEqual(domain, 1)
                    seen.add(key)
            changed_seed = argparse.Namespace(**{**vars(extension), "seed": 99, "output_dir": root / "wrong-seed"})
            with self.assertRaisesRegex(ValueError, "partition seed"):
                self.run_quietly(changed_seed)


if __name__ == "__main__":
    unittest.main()
