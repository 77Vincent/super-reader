#!/usr/bin/env python3
"""Mix a bounded, reproducible sample of downloaded web documents with old shards."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import mmap
import os
from pathlib import Path
import random
import re
import shutil
import signal
import struct
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training/.deps"))
import pyarrow.parquet as pq
from prepare_synthetic_data import clean_document, clean_fragment, quality_document, document_signature
from text_policy import DATA_POLICY, PROXY_PUNCTUATION, require_data_policy, valid_proxy_label

BASE = ROOT / "training/data/processed/unicode-context-192ch-12conv-20260913-combined/manifest.json"
EVAL = ROOT / "training/data/processed/unicode-context-192ch-12conv-20260913-eval"
RAW = ROOT / "training/data/raw/ultra-fineweb-zh"
NON_HAN = re.compile("[^\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f\U00030000-\U000323af]+")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    part = path.with_name(path.name + ".tmp")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    part.replace(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def absolute(path):
    return (ROOT / path).resolve()


def projected_key(text):
    """Ignore punctuation and target position: conservatively reject repeated inputs."""
    projected = NON_HAN.sub("", unicodedata.normalize("NFKC", text))
    # Inputs without Han still need an identity, rather than sharing the empty key.
    value = projected if projected else unicodedata.normalize("NFKC", text)
    return hashlib.sha256(value.encode()).digest()


class Bloom:
    """No false negatives; false positives only discard additional candidates."""
    def __init__(self, size, path=None):
        if size < 1024 or size & (size - 1):
            raise ValueError("Bloom bytes must be a power of two >= 1024")
        self.mask = size * 8 - 1
        self.handle = None
        if path is None:
            self.bits = bytearray(size)
        else:
            path = Path(path)
            self.handle = path.open("r+b" if path.exists() else "w+b")
            if path.stat().st_size not in (0, size):
                raise ValueError("Wrong Bloom file size")
            self.handle.truncate(size)
            self.bits = mmap.mmap(self.handle.fileno(), size)

    def positions(self, key):
        first, second = struct.unpack("<QQ", key[:16])
        second |= 1
        for i in range(7):
            bit = (first + i * second) & self.mask
            yield bit >> 3, 1 << (bit & 7)

    def contains(self, key):
        return all(self.bits[index] & mask for index, mask in self.positions(key))

    def add(self, key):
        for index, mask in self.positions(key):
            self.bits[index] |= mask

    def flush(self):
        if self.handle:
            self.bits.flush()

    def close(self):
        if self.handle:
            self.bits.close()
            self.handle.close()


def labeled_samples(document, rejection_counts=None):
    """Line-local candidates with original proxy glyphs and shared surface filtering."""
    from text_policy import training_pairs
    from prepare_synthetic_data import URL_PATTERN, MARKDOWN_FENCE_PATTERN
    prepared = URL_PATTERN.sub(' ', MARKDOWN_FENCE_PATTERN.sub(' ', str(document or '')))
    yield from training_pairs(prepared, rejection_counts)


def document_split(fingerprint, seed):
    bucket = int.from_bytes(hashlib.sha256(f"{seed}:{fingerprint}".encode()).digest()[:8], "little") % 1000
    return "test" if bucket < 20 else "validation" if bucket < 40 else "train"


def quota(total, files, index):
    return total // files + int(index < total % files)


def fresh_statistics():
    return {"samples": 0, "tokens": 0, "random_baseline_sum": 0.0, "center_correct": 0,
            "domain_samples": {"web": 0}, "cells": [[0] * 10 for _ in range(4)],
            "position_histogram": [0] * 10, "maximum_sequence_length": 0}


def add_statistics(stats, text, target):
    length = len(text)
    bucket = next((i for i, maximum in enumerate((8, 16, 32)) if length <= maximum), 3)
    position = min(9, int((target + 1) / length * 10))
    stats["samples"] += 1
    stats["tokens"] += length
    stats["random_baseline_sum"] += 1 / (length - 1)
    stats["center_correct"] += int(target == (length - 1) // 2)
    stats["domain_samples"]["web"] += 1
    stats["cells"][bucket][position] += 1
    stats["position_histogram"][position] += 1
    stats["maximum_sequence_length"] = max(stats["maximum_sequence_length"], length)
    return bucket, position


def evaluation_lineage(directory):
    directories, seen = [], set()
    while directory not in seen:
        seen.add(directory)
        directories.append(directory)
        summary = read(directory / "summary.json")
        if not summary.get("source_directory"):
            break
        directory = absolute(summary["source_directory"])
    return directories


def holdout_registries(directory):
    return [path / "holdout-document-hashes.json" for path in evaluation_lineage(directory)
            if (path / "holdout-document-hashes.json").exists()]


def source_configuration(args):
    base, evaluation, downloaded = read(args.base_manifest), read(args.evaluation / "summary.json"), read(args.raw_dir / "download-manifest.json")
    require_data_policy(base)
    require_data_policy(evaluation)
    if "web" in base["domains"]:
        if not args.freeze_evaluation:
            raise ValueError("Extending existing web data requires --freeze-evaluation")
        previous_config = read(args.base_manifest.parent / "configuration.json")
        if previous_config["seed"] != args.seed or base["web_source"]["revision"] != downloaded["revision"]:
            raise ValueError("Keep the original document partition seed and source revision")
    verified = read(args.raw_dir / "verified-files.json")
    if verified["revision"] != downloaded["revision"]:
        raise ValueError("Downloaded revision differs from verified sources")
    for item in downloaded["files"]:
        path = args.raw_dir / item["filename"]
        if path.stat().st_size != item["bytes"] or verified["files"].get(item["filename"]) != {"bytes": item["bytes"], "sha256": item["sha256"]}:
            raise ValueError(f"Downloaded file lacks matching verification: {path}")
    manifests, visited = [], set()
    pending = [args.base_manifest]
    for directory in evaluation_lineage(args.evaluation):
        pending.extend(absolute(item["path"]) for item in read(directory / "summary.json").get("training_manifests", []))
    while pending:
        path = pending.pop(0)
        if path in visited:
            continue
        visited.add(path)
        manifests.append(path)
        parent = read(path).get("base_manifest")
        if parent:
            pending.append(absolute(parent))
    files = {}
    for path in manifests:
        for item in read(path)["shards"]:
            resolved = absolute(item["path"])
            if resolved.stat().st_size != item["bytes"]:
                raise ValueError(f"Training shard size changed: {resolved}")
            files[str(resolved)] = {"path": str(resolved), "bytes": item["bytes"], "mtime_ns": resolved.stat().st_mtime_ns, "format": "compact"}
    for split in ("validation", "test"):
        path = args.evaluation / f"{split}.jsonl"
        if sha(path) != evaluation["splits"][split]["sha256"]:
            raise ValueError(f"Frozen {split} changed")
        files[str(path)] = {"path": str(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns, "format": "evaluation"}
    registries = holdout_registries(args.evaluation)
    if not registries:
        raise ValueError("Missing inherited holdout document registry")
    config = {"version": 2, "target_samples": args.target_samples, "seed": args.seed, "shards": args.shards,
              "freeze_evaluation": args.freeze_evaluation,
              "bloom_bytes": args.bloom_bytes, "base_manifest": str(args.base_manifest),
              "manifests": {str(p): sha(p) for p in manifests}, "evaluation": str(args.evaluation),
              "evaluation_summary_sha256": sha(args.evaluation / "summary.json"),
              "download_manifest_sha256": sha(args.raw_dir / "download-manifest.json"),
              "holdout_document_registries": {str(p): sha(p) for p in registries},
              "source_files": [{**x, "mtime_ns": (args.raw_dir / x["filename"]).stat().st_mtime_ns} for x in downloaded["files"]],
              "seed_files": list(files.values()), "policy": DATA_POLICY,
              "code_sha256": {name: sha(Path(__file__).with_name(name)) for name in
                              ("prepare_web_data.py", "prepare_synthetic_data.py", "text_policy.py", "text-policy.json")}}
    return base, evaluation, downloaded, config


def prepare(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare_locked(args)


def prepare_locked(args):
    out = args.output_dir
    base, evaluation, downloaded, config = source_configuration(args)
    plan = out / "configuration.json"
    if plan.exists() and read(plan) != config:
        raise ValueError("Preparation inputs/options changed; use a fresh directory")
    write(plan, config)
    if (out / "manifest.json").exists():
        return
    stop = {"requested": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(requested=True))
    def status(stage, **details):
        value = {"stage": stage, "updated_at": time.time(), **details}
        write(out / "status.json", value)
        print(json.dumps(value), flush=True)
    seeded_path = out / "seeded-files.json"
    seeded = read(seeded_path) if seeded_path.exists() else []
    if seeded and (not (out / "history.bloom").exists() or (out / "history.bloom").stat().st_size != args.bloom_bytes):
        raise ValueError("Historical index is missing; cannot reuse its completion record")
    old = Bloom(args.bloom_bytes, out / "history.bloom")
    try:
        for index, item in enumerate(config["seed_files"]):
            if index < len(seeded):
                continue
            status("indexing_history", file=index + 1, files=len(config["seed_files"]), path=item["path"])
            digest, rows = hashlib.sha256(), 0
            with Path(item["path"]).open("rb") as handle:
                for line in handle:
                    digest.update(line)
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    text = record[0] if item["format"] == "compact" else "".join(record["tokens"])
                    old.add(projected_key(text))
                    rows += 1
            old.flush()
            seeded.append({**item, "sha256": digest.hexdigest(), "rows": rows})
            write(seeded_path, seeded)
            if stop["requested"]:
                status("stopped", phase="indexing_history")
                return
        generate(args, base, evaluation, downloaded, old, stop, status)
    finally:
        old.close()


def generate(args, base, evaluation, downloaded, old, stop, status):
    out = args.output_dir
    state_path = out / "preparation-state.json"
    paths = [out / f"web-train-{i:03d}.jsonl.part" for i in range(args.shards)] + [out / "web-validation.jsonl.part", out / "web-test.jsonl.part"]
    state = read(state_path) if state_path.exists() else {"cursor": {"file": 0, "group": 0, "row": 0},
        "file_training_samples": 0, "sizes": [0] * len(paths), "statistics": fresh_statistics(),
        "evaluation_counts": {"validation": 0, "test": 0}, "documents": 0, "quality_filtered": 0,
        "holdout_documents_filtered": 0, "reserved_holdout_documents": 0,
        "duplicate_or_bloom_filtered": 0, "per_source": {}, "completed_files": []}
    domain_index = base["domains"].index("web") if "web" in base["domains"] else len(base["domains"])
    handles = []
    # Rebuild only from committed bytes: an interrupted write cannot poison later sampling.
    seen = Bloom(max(1024, args.bloom_bytes // 2))
    for path, size in zip(paths, state["sizes"]):
        path.touch(exist_ok=True)
        if path.stat().st_size < size:
            raise ValueError(f"Committed output was truncated: {path}")
        with path.open("r+b") as handle:
            handle.truncate(size)
        with path.open() as handle:
            for line in handle:
                seen.add(projected_key(json.loads(line)[0]))
        handles.append(path.open("ab", buffering=1024 * 1024))
    blocked_documents = {key for path in holdout_registries(args.evaluation) for key in read(path)}
    def checkpoint():
        for handle in handles:
            handle.flush()
        state["sizes"] = [handle.tell() for handle in handles]
        write(state_path, state)
        status("preparing_web", train=state["statistics"]["samples"], target=args.target_samples,
               evaluation=state["evaluation_counts"], documents=state["documents"], cursor=state["cursor"])
    def emit(index, record):
        handles[index].write((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
    try:
        while state["cursor"]["file"] < len(downloaded["files"]):
            file_index = state["cursor"]["file"]
            info = downloaded["files"][file_index]
            per_file_target = quota(args.target_samples, len(downloaded["files"]), file_index)
            parquet = pq.ParquetFile(args.raw_dir / info["filename"])
            groups = list(range(parquet.num_row_groups))
            random.Random(args.seed + file_index).shuffle(groups)
            while state["cursor"]["group"] < len(groups) and state["file_training_samples"] < per_file_target:
                group = groups[state["cursor"]["group"]]
                row = 0
                for batch in parquet.iter_batches(batch_size=256, row_groups=[group], columns=["content", "source"], use_threads=False):
                    for item in batch.to_pylist():
                        row += 1
                        if row <= state["cursor"]["row"]:
                            continue
                        content = item["content"] or ""
                        doc_key = document_signature(content)
                        split = document_split(doc_key, args.seed)
                        source = str(item["source"] or "unknown")
                        state["documents"] += 1
                        if doc_key in blocked_documents:
                            state["holdout_documents_filtered"] += 1
                        elif args.freeze_evaluation and split != "train":
                            state["reserved_holdout_documents"] += 1
                        elif not quality_document(clean_document(content)):
                            state["quality_filtered"] += 1
                        else:
                            for text, target, punctuation in labeled_samples(content, state.setdefault("surface_rejected", {})):
                                key = projected_key(text)
                                if old.contains(key) or seen.contains(key):
                                    state["duplicate_or_bloom_filtered"] += 1
                                    continue
                                seen.add(key)
                                state["per_source"].setdefault(source, {"train": 0, "validation": 0, "test": 0})[split] += 1
                                if split == "train":
                                    bucket, position = add_statistics(state["statistics"], text, target)
                                    emit(int.from_bytes(key[:4], "little") % args.shards, [text, target, domain_index, bucket, position])
                                    state["file_training_samples"] += 1
                                    if state["file_training_samples"] >= per_file_target:
                                        break
                                else:
                                    emit(args.shards + (split == "test"), [text, target, punctuation, doc_key, source])
                                    state["evaluation_counts"][split] += 1
                        state["cursor"]["row"] = row
                        if state["documents"] % args.checkpoint_documents == 0 or stop["requested"]:
                            checkpoint()
                        if stop["requested"]:
                            status("stopped", phase="preparing_web")
                            return
                        if state["file_training_samples"] >= per_file_target:
                            break
                    if state["file_training_samples"] >= per_file_target:
                        break
                if state["file_training_samples"] < per_file_target:
                    state["cursor"]["group"] += 1
                    state["cursor"]["row"] = 0
            if state["file_training_samples"] != per_file_target:
                raise RuntimeError(f"File exhausted before its quota: {info['filename']}")
            state["completed_files"].append({"filename": info["filename"], "train": state["file_training_samples"]})
            state["cursor"] = {"file": file_index + 1, "group": 0, "row": 0}
            state["file_training_samples"] = 0
            checkpoint()
    finally:
        for handle in handles:
            handle.close()
    finalize(args, base, evaluation, downloaded, state, paths)
    status("complete", train=state["statistics"]["samples"], combined=base["statistics"]["samples"] + state["statistics"]["samples"], evaluation=state["evaluation_counts"])


def finalize(args, base, evaluation, downloaded, state, paths):
    out = args.output_dir
    assert state["statistics"]["samples"] == args.target_samples
    splits = {}
    for index, split in enumerate(("validation", "test")):
        output = out / f"{split}.jsonl"
        if args.freeze_evaluation:
            assert state["evaluation_counts"][split] == 0
            temporary = output.with_suffix(".tmp")
            shutil.copyfile(args.evaluation / f"{split}.jsonl", temporary)
            if sha(temporary) != evaluation["splits"][split]["sha256"]:
                raise ValueError(f"Frozen {split} changed while copying")
            temporary.replace(output)
            splits[split] = copy.deepcopy(evaluation["splits"][split])
            continue
        temporary = output.with_suffix(".tmp")
        count, digest = 0, hashlib.sha256()
        with temporary.open("wb") as destination:
            with (args.evaluation / f"{split}.jsonl").open("rb") as source:
                for line in source:
                    destination.write(line)
                    digest.update(line)
            with paths[args.shards + index].open() as source:
                for line in source:
                    text, target, punctuation, document, web_source = json.loads(line)
                    assert valid_proxy_label(punctuation)
                    record = {"id": f"web:{document}:{count}", "document_id": f"web:{document}", "domain": "web",
                              "source": web_source, "tokenization": "character", "tokens": list(text),
                              "target_index": target, "punctuation": punctuation}
                    encoded = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                    destination.write(encoded)
                    digest.update(encoded)
                    count += 1
        assert count == state["evaluation_counts"][split] and count > 0
        temporary.replace(output)
        splits[split] = {"count": evaluation["splits"][split]["count"] + count,
                         "old_count": evaluation["splits"][split]["count"], "new_count": count,
                         "old_sha256": evaluation["splits"][split]["sha256"], "sha256": digest.hexdigest(),
                         "per_domain": {**evaluation["splits"][split]["per_domain"], "web": count}}
    vocabulary = out / "vocabulary.json"
    shutil.copyfile(absolute(base["vocabulary_path"]), vocabulary)
    stats = copy.deepcopy(base["statistics"])
    new = state["statistics"]
    for key in ("samples", "tokens", "random_baseline_sum", "center_correct"):
        stats[key] += new[key]
    stats["domain_samples"]["web"] = stats["domain_samples"].get("web", 0) + new["samples"]
    stats["maximum_sequence_length"] = max(stats["maximum_sequence_length"], new["maximum_sequence_length"])
    stats["cells"] = [[x + y for x, y in zip(a, b)] for a, b in zip(stats["cells"], new["cells"])]
    stats["position_histogram"] = [x + y for x, y in zip(stats["position_histogram"], new["position_histogram"])]
    shards = []
    # Hard links publish committed shards atomically without duplicating their bytes.
    # Keeping .part names permits idempotent finalization after a restart.
    for part in paths[:args.shards]:
        final = part.with_suffix("")
        if not final.exists():
            os.link(part, final)
        if not os.path.samefile(part, final):
            raise ValueError(f"Published shard is not its committed output: {final}")
        shards.append({"path": str(final), "bytes": final.stat().st_size, "sha256": sha(final)})
    protection = {"identity": "NFKC Han-projected input text, independent of target gap", "method": "Bloom filters; false positives discard extra samples; no false negatives",
                  "historical_files": len(read(out / "seeded-files.json")), "document_split": "96% train / 2% validation / 2% test by stable document hash",
                  "original_holdouts": "unchanged whole split" if args.freeze_evaluation else "unchanged byte-for-byte prefix in each combined split",
                  "unused_holdout_documents": "reserved, never added to training" if args.freeze_evaluation else "sampled as holdouts",
                  "near_duplicate_audit": "not performed"}
    write(out / "holdout-document-hashes.json", sorted({key for path in holdout_registries(args.evaluation) for key in read(path)}))
    write(out / "summary.json", {**DATA_POLICY, "splits": splits, "source_directory": str(args.evaluation), "protection": protection})
    write(out / "manifest.json", {**DATA_POLICY, "format": "super-reader-sharded-training-v1",
          "source": "Existing full local corpus plus document-partitioned Ultra-FineWeb Chinese sample",
          "base_manifest": str(args.base_manifest), "base_manifest_sha256": sha(args.base_manifest),
          "domains": base["domains"] if "web" in base["domains"] else [*base["domains"], "web"],
          "evaluation_source_dir": str(out), "vocabulary_path": str(vocabulary),
          "length_bucket_maximums": [8, 16, 32], "position_bins": 10, "maximum_sequence_length": 0,
          "shards": [*base["shards"], *shards], "statistics": stats, "web_statistics": state,
          "web_source": {"repository": downloaded["repository"], "revision": downloaded["revision"],
                         "total_training_samples": stats["domain_samples"]["web"],
                         "added_training_samples": new["samples"], "files": state["completed_files"]},
          "protection": protection, "configuration_sha256": sha(out / "configuration.json")})


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=BASE)
    parser.add_argument("--evaluation", type=Path, default=EVAL)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-samples", type=int, default=20_000_000, help="Additional training pairs on top of --base-manifest")
    parser.add_argument("--freeze-evaluation", action="store_true", help="Keep both complete holdouts identical and only accept train-partition documents")
    parser.add_argument("--shards", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--bloom-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--checkpoint-documents", type=int, default=1000)
    args = parser.parse_args()
    for name in ("base_manifest", "evaluation", "raw_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if min(args.target_samples, args.shards, args.checkpoint_documents) < 1:
        parser.error("Counts must be positive")
    return args


if __name__ == "__main__":
    prepare(arguments())
