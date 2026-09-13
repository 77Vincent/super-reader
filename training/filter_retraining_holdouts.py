#!/usr/bin/env python3
"""Exclude revised holdout pairs already seen by checkpoints being compared."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def signature(text: str, target: int) -> bytes:
    return hashlib.sha256(text.encode("utf-8") + b"\0" + str(target).encode("ascii")).digest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, action="append", default=[])
    parser.add_argument("--training-jsonl", type=Path, action="append", default=[])
    args = parser.parse_args()
    if not args.training_manifest and not args.training_jsonl:
        raise ValueError("Supply the training data used by the compared checkpoints")
    if args.output_dir.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.output_dir}")

    wanted = set()
    input_counts = {}
    input_hashes = {}
    for split in ("test", "validation"):
        digest = hashlib.sha256()
        count = 0
        with (args.data_dir / f"{split}.jsonl").open("rb") as handle:
            for line in handle:
                digest.update(line)
                if not line.strip():
                    continue
                record = json.loads(line)
                if "、" in record.get("punctuation", ""):
                    raise ValueError(f"Enumeration label remains: {record['id']}")
                key = signature("".join(record["tokens"]), record["target_index"])
                if key in wanted:
                    raise ValueError("Input holdout pairs must be unique within and across splits")
                wanted.add(key)
                count += 1
        input_counts[split] = count
        input_hashes[split] = digest.hexdigest()
    print(f"Checking {len(wanted)} revised holdout pairs for training overlap", flush=True)

    files = {}
    manifests = []
    for path in args.training_manifest:
        content = path.read_bytes()
        manifest = json.loads(content)
        manifests.append({"path": str(path.resolve()), "sha256": hashlib.sha256(content).hexdigest()})
        for item in manifest["shards"]:
            shard = Path(item["path"])
            shard = shard if shard.is_absolute() else ROOT / shard
            if shard.stat().st_size != item["bytes"]:
                raise ValueError(f"Training shard size mismatch: {shard}")
            files[shard.resolve()] = "compact"
    for path in args.training_jsonl:
        files[path.resolve()] = "records"

    overlaps = set()
    training_sources = []
    for index, (path, kind) in enumerate(files.items(), 1):
        digest = hashlib.sha256()
        count = 0
        before = path.stat()
        with path.open("rb") as handle:
            for line in handle:
                digest.update(line)
                if not line.strip():
                    continue
                record = json.loads(line)
                text, target = (record[0], record[1]) if kind == "compact" else ("".join(record["tokens"]), record["target_index"])
                key = signature(text, target)
                if key in wanted:
                    overlaps.add(key)
                count += 1
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Training source changed while being scanned: {path}")
        training_sources.append({"path": str(path), "sha256": digest.hexdigest(), "records": count})
        if index % 16 == 0 or index == len(files):
            print(f"Scanned {index}/{len(files)} training files; overlapping holdout pairs={len(overlaps)}", flush=True)

    args.output_dir.mkdir(parents=True)
    results = {}
    for split in ("test", "validation"):
        kept = Counter()
        removed = Counter()
        digest = hashlib.sha256()
        with (args.data_dir / f"{split}.jsonl").open("rb") as source, (args.output_dir / f"{split}.jsonl").open("wb") as output:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = signature("".join(record["tokens"]), record["target_index"])
                if key in overlaps:
                    removed[record["domain"]] += 1
                    continue
                output.write(line)
                digest.update(line)
                kept[record["domain"]] += 1
        if not kept:
            raise ValueError(f"No clean {split} examples remain")
        results[split] = {"count": sum(kept.values()), "per_domain": dict(kept),
                          "removed_training_overlap": dict(removed), "sha256": digest.hexdigest()}
        print(f"{split}: kept={sum(kept.values())} removed_training_overlap={sum(removed.values())}", flush=True)
    registry = args.data_dir / "holdout-documents.json"
    if registry.exists():
        shutil.copy2(registry, args.output_dir / registry.name)
    summary = {"purpose": "Revised holdouts for fair checkpoint reselection and comparison",
               "tokenization": "character", "excluded_proxy_punctuation": ["、"],
               "source_directory": str(args.data_dir.resolve()), "source_counts": input_counts,
               "source_hashes": input_hashes, "splits": results,
               "overlap_identity": "Han text plus target gap, matching the existing boundary-pair deduplication policy",
               "training_manifests": manifests, "training_files": training_sources}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
