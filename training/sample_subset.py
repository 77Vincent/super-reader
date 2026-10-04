"""Read-time target filtering and resumable statistics; never rewrite the corpus."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def keep_target(length: int, target: int, minimum: int = 1) -> bool:
    if minimum < 1:
        raise ValueError("Minimum side characters must be positive")
    if not isinstance(target, int) or not 0 <= target < length - 1:
        raise ValueError(f"Invalid target {target} for {length} characters")
    cut = target + 1
    return min(cut, length - cut) >= minimum


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def empty_statistics(manifest):
    return {
        "samples": 0, "tokens": 0, "source_samples": 0, "excluded_samples": 0,
        "maximum_sequence_length": 0, "random_baseline_sum": 0., "center_correct": 0,
        "domain_samples": {domain: 0 for domain in manifest["domains"]},
        "cells": [[0] * len(row) for row in manifest["statistics"]["cells"]],
        "domain_cells": [[[0] * len(row) for row in manifest["statistics"]["cells"]]
                         for _ in manifest["domains"]],
    }


def scan_shard(path, manifest, minimum):
    stats = empty_statistics(manifest)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if not line.strip():
                continue
            text, target, domain, bucket, position = json.loads(line)
            stats["source_samples"] += 1
            if not keep_target(len(text), target, minimum):
                stats["excluded_samples"] += 1
                continue
            stats["samples"] += 1
            stats["tokens"] += len(text)
            stats["maximum_sequence_length"] = max(stats["maximum_sequence_length"], len(text))
            stats["random_baseline_sum"] += 1 / (len(text) - 1)
            stats["center_correct"] += target == (len(text) - 1) // 2
            stats["domain_samples"][manifest["domains"][domain]] += 1
            stats["cells"][bucket][position] += 1
            stats["domain_cells"][domain][bucket][position] += 1
    return stats, digest.hexdigest()


def subset_statistics(shards, manifest, minimum, cache_dir):
    """Cache each completed shard; validate file identity before reusing it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    contract = {"version": 1, "minimum_side_characters": minimum,
                "domains": manifest["domains"],
                "cell_shape": [len(row) for row in manifest["statistics"]["cells"]]}
    total = empty_statistics(manifest)
    identities = []
    for index, path in enumerate(shards):
        stat = path.stat()
        identity = {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        cached_path = cache_dir / (hashlib.sha256(identity["path"].encode()).hexdigest() + ".json")
        cached = json.loads(cached_path.read_text()) if cached_path.exists() else None
        if not cached or cached["contract"] != contract or cached["identity"] != identity:
            stats, digest = scan_shard(path, manifest, minimum)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise ValueError(f"Training shard changed during scan: {path}")
            cached = {"contract": contract, "identity": identity, "sha256": digest, "statistics": stats}
            write_json(cached_path, cached)
        stats = cached["statistics"]
        identities.append({**identity, "sha256": cached["sha256"]})
        for key in ("samples", "tokens", "source_samples", "excluded_samples", "random_baseline_sum", "center_correct"):
            total[key] += stats[key]
        total["maximum_sequence_length"] = max(total["maximum_sequence_length"], stats["maximum_sequence_length"])
        for domain in total["domain_samples"]:
            total["domain_samples"][domain] += stats["domain_samples"][domain]
        for b, row in enumerate(total["cells"]):
            for p in range(len(row)):
                row[p] += stats["cells"][b][p]
                for d in range(len(manifest["domains"])):
                    total["domain_cells"][d][b][p] += stats["domain_cells"][d][b][p]
        if index == 0 or (index + 1) % 32 == 0 or index + 1 == len(shards):
            progress = {"stage": "subset-statistics", "shards": index + 1, "total_shards": len(shards),
                        "source_samples": total["source_samples"], "retained_samples": total["samples"],
                        "excluded_samples": total["excluded_samples"]}
            write_json(cache_dir.parent / "subset-progress.json", progress)
            print(json.dumps(progress), flush=True)
    if not total["samples"]:
        raise ValueError("The selected training subset is empty")
    report = {"contract": contract, "statistics": total, "shards": identities}
    write_json(cache_dir.parent / "training-subset.json", report)
    return total


def mean_weight(stats, position_weights, domain_weights):
    total = sum(count * position_weights[b][p] * domain_weights[d]
                for d, cells in enumerate(stats["domain_cells"])
                for b, row in enumerate(cells) for p, count in enumerate(row))
    if not stats["samples"] or total <= 0:
        raise ValueError("Empty or zero-weight training subset")
    return total / stats["samples"]
