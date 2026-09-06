#!/usr/bin/env python3
"""Add a bounded, deduplicated Ultra-FineWeb-L3 sample to existing shards."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any, Iterable, TypeVar

import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_BASE_MANIFEST = (
    SCRIPT_DIR / "data" / "processed" / "wiki-full-sharded-128" / "manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR / "data" / "processed" / "wiki-full-plus-ultra-5m"
)
DEFAULT_RAW_DIR = SCRIPT_DIR / "data" / "raw" / "ultra-fineweb-l3-zh-multistyle"
DATASET = "openbmb/Ultra-FineWeb-L3"
CONFIG = "Ultra-FineWeb-L3-zh-Multi-Style-Synthetic"
SPLIT = "train"
SYNTHETIC_DOMAIN = "synthetic_multistyle"
BASE_DOMAINS = ["news", "academic", "encyclopedia", "dialogue", "wikipedia"]
LENGTH_BUCKET_MAXIMUMS = [8, 16, 32]
PROXY_PUNCTUATION = frozenset("，,。.！!？?；;：:、…")
URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
SPACE_PATTERN = re.compile(r"\s+")
MARKDOWN_FENCE_PATTERN = re.compile(r"```.*?```", re.DOTALL)
BLOOM_BYTES = 256 * 1024 * 1024
BLOOM_BIT_MASK = 0x7FFFFFFF
BLOOM_HASHES = 7
WRITE_BUFFER_BYTES = 1024 * 1024
T = TypeVar("T")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--target-samples", type=int, default=5_000_000)
    parser.add_argument("--synthetic-shards", type=int, default=32)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--max-samples-per-document", type=int, default=128)
    parser.add_argument("--batch-rows", type=int, default=2048)
    parser.add_argument("--checkpoint-documents", type=int, default=10_000)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_DIR / value


def project_relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_DIR))


def is_han(character: str) -> bool:
    code = ord(character)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2FA1F
        or 0x30000 <= code <= 0x323AF
    )


def normalize_document(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = MARKDOWN_FENCE_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    return SPACE_PATTERN.sub(" ", text).strip()


def clean_fragment(text: str) -> str:
    characters = []
    last_was_space = False
    for character in unicodedata.normalize("NFKC", text):
        category = unicodedata.category(character)
        if category.startswith(("P", "S")) or character.isspace():
            if characters and not last_was_space:
                characters.append(" ")
            last_was_space = True
            continue
        characters.append(character)
        last_was_space = False
    return "".join(characters).strip()


def split_into_fragments(text: str) -> list[str]:
    fragments: list[str] = []
    buffer: list[str] = []
    for character in normalize_document(text):
        if character not in PROXY_PUNCTUATION:
            buffer.append(character)
            continue
        cleaned = clean_fragment("".join(buffer))
        if cleaned:
            fragments.append(cleaned)
        buffer.clear()
    tail = clean_fragment("".join(buffer))
    if tail:
        fragments.append(tail)
    return fragments


def adjacent_samples(text: str) -> list[tuple[str, int]]:
    fragments = split_into_fragments(text)
    result: list[tuple[str, int]] = []
    for left_fragment, right_fragment in zip(fragments, fragments[1:]):
        left = "".join(character for character in left_fragment if is_han(character))
        right = "".join(character for character in right_fragment if is_han(character))
        if not left or not right:
            continue
        result.append((left + right, len(left) - 1))
    return result


def quality_document(text: str) -> bool:
    han = sum(is_han(character) for character in text)
    latin = sum(character.isascii() and character.isalpha() for character in text)
    return han >= 80 and han / max(1, han + latin) >= 0.70


def evenly_capped(items: list[T], maximum: int) -> list[T]:
    if len(items) <= maximum:
        return items
    return [
        items[int(((index + 0.5) * len(items)) / maximum)]
        for index in range(maximum)
    ]


def update_hashes(hashes: tuple[int, int], value: str) -> tuple[int, int]:
    first, second = hashes
    for character in value:
        code = ord(character)
        first = ((first ^ code) * 16_777_619) & 0xFFFFFFFF
        second = ((second ^ code) * 2_246_822_519) & 0xFFFFFFFF
    return first, second


def boundary_hashes(text: str, target_index: int) -> tuple[int, int]:
    hashes = (2_166_136_261, 0x9E3779B9)
    for index, character in enumerate(text):
        hashes = update_hashes(hashes, character)
        if index == target_index:
            hashes = (
                ((hashes[0] ^ 0xFFFFFFFF) * 16_777_619) & 0xFFFFFFFF,
                ((hashes[1] ^ 0xFFFFFFFF) * 2_246_822_519) & 0xFFFFFFFF,
            )
    return hashes[0], hashes[1] | 1


class BloomFilter:
    def __init__(self) -> None:
        self.bits = bytearray(BLOOM_BYTES)

    def add(self, hashes: tuple[int, int]) -> bool:
        first, second = hashes
        seen = True
        for index in range(BLOOM_HASHES):
            bit = (first + index * second) & 0xFFFFFFFF
            bit &= BLOOM_BIT_MASK
            byte_index = bit >> 3
            mask = 1 << (bit & 7)
            if not self.bits[byte_index] & mask:
                seen = False
                self.bits[byte_index] |= mask
        return seen


def iter_json_lines(path: Path) -> Iterable[Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def seed_bloom(
    base_manifest: dict[str, Any],
    bloom: BloomFilter,
    output_dir: Path,
    synthetic_shards: int,
) -> int:
    count = 0
    for index, shard in enumerate(base_manifest["shards"]):
        for record in iter_json_lines(project_path(shard["path"])):
            bloom.add(boundary_hashes(record[0], record[1]))
            count += 1
        if (index + 1) % 16 == 0:
            print(f"seeded base bloom shards={index + 1}/{len(base_manifest['shards'])}", flush=True)

    evaluation_dir = project_path(base_manifest["evaluation_source_dir"])
    for split in ("validation", "test"):
        for record in iter_json_lines(evaluation_dir / f"{split}.jsonl"):
            bloom.add(boundary_hashes("".join(record["tokens"]), record["target_index"]))
            count += 1

    restored = 0
    for index in range(synthetic_shards):
        path = output_dir / f"synthetic-train-{index:03d}.jsonl.part"
        if not path.exists():
            continue
        for record in iter_json_lines(path):
            bloom.add(boundary_hashes(record[0], record[1]))
            restored += 1
    print(f"seeded bloom base_and_holdout={count} restored_synthetic={restored}", flush=True)
    return restored


class ShardWriter:
    def __init__(self, output_dir: Path, count: int, initial_sizes: list[int]) -> None:
        self.paths = [
            output_dir / f"synthetic-train-{index:03d}.jsonl.part"
            for index in range(count)
        ]
        self.sizes = initial_sizes
        self.handles: list[Any] = []
        self.buffers = [bytearray() for _ in range(count)]

    def open(self) -> None:
        for path, size in zip(self.paths, self.sizes):
            path.touch(exist_ok=True)
            with path.open("r+b") as handle:
                handle.truncate(size)
            self.handles.append(path.open("ab"))

    def add(self, index: int, record: list[Any]) -> None:
        encoded = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        self.buffers[index].extend(encoded)
        if len(self.buffers[index]) >= WRITE_BUFFER_BYTES:
            self.flush(index)

    def flush(self, index: int) -> None:
        if not self.buffers[index]:
            return
        self.handles[index].write(self.buffers[index])
        self.sizes[index] += len(self.buffers[index])
        self.buffers[index].clear()

    def flush_all(self) -> None:
        for index in range(len(self.paths)):
            self.flush(index)
            self.handles[index].flush()

    def close(self) -> None:
        self.flush_all()
        for handle in self.handles:
            handle.close()


def empty_statistics(position_bins: int) -> dict[str, Any]:
    return {
        "samples": 0,
        "tokens": 0,
        "random_baseline_sum": 0.0,
        "center_correct": 0,
        "duplicate_or_bloom_filtered": 0,
        "overlength_filtered": 0,
        "documents_seen": 0,
        "documents_with_samples": 0,
        "quality_documents_filtered": 0,
        "document_sample_cap_filtered": 0,
        "domain_samples": {SYNTHETIC_DOMAIN: 0},
        "position_histogram": [0] * position_bins,
        "cells": [[0] * position_bins for _ in range(4)],
        "maximum_sequence_length": 0,
    }


def add_sample(
    text: str,
    target: int,
    domain_index: int,
    bloom: BloomFilter,
    writer: ShardWriter,
    statistics: dict[str, Any],
    options: argparse.Namespace,
) -> None:
    hashes = boundary_hashes(text, target)
    if bloom.add(hashes):
        statistics["duplicate_or_bloom_filtered"] += 1
        return
    length = len(text)
    if length > options.max_sequence_length:
        statistics["overlength_filtered"] += 1
        return
    bucket = next(
        (index for index, maximum in enumerate(LENGTH_BUCKET_MAXIMUMS) if length <= maximum),
        len(LENGTH_BUCKET_MAXIMUMS),
    )
    position = min(9, int(((target + 1) / length) * 10))
    shard = hashes[0] & (options.synthetic_shards - 1)
    writer.add(shard, [text, target, domain_index, bucket, position])
    statistics["samples"] += 1
    statistics["tokens"] += length
    statistics["random_baseline_sum"] += 1 / (length - 1)
    statistics["center_correct"] += target == (length - 1) // 2
    statistics["domain_samples"][SYNTHETIC_DOMAIN] += 1
    statistics["position_histogram"][position] += 1
    statistics["cells"][bucket][position] += 1
    statistics["maximum_sequence_length"] = max(
        statistics["maximum_sequence_length"],
        length,
    )


def source_urls() -> list[str]:
    endpoint = (
        f"https://huggingface.co/api/datasets/{DATASET}/parquet/"
        f"{CONFIG}/{SPLIT}"
    )
    with urllib.request.urlopen(endpoint, timeout=60) as response:
        return json.load(response)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_download(url: str, index: int, raw_dir: Path) -> tuple[Path, dict[str, Any]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    final = raw_dir / f"{index:05d}.parquet"
    if not final.exists():
        partial = final.with_suffix(".parquet.part")
        print(f"downloading source parquet {index}: {url}", flush=True)
        subprocess.run(
            [
                "curl",
                "-fL",
                "--retry",
                "8",
                "--retry-all-errors",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                url,
            ],
            check=True,
        )
        os.replace(partial, final)
    parquet = pq.ParquetFile(final)
    return final, {
        "source_index": index,
        "source_url": url,
        "path": project_relative(final),
        "bytes": final.stat().st_size,
        "rows": parquet.metadata.num_rows,
        "sha256": sha256_file(final),
    }


def compatible_state(
    state: dict[str, Any],
    base_sha256: str,
    options: argparse.Namespace,
) -> bool:
    expected = {
        "target_samples": options.target_samples,
        "synthetic_shards": options.synthetic_shards,
        "max_sequence_length": options.max_sequence_length,
        "max_samples_per_document": options.max_samples_per_document,
    }
    return (
        state.get("format_version") == 1
        and state.get("base_manifest_sha256") == base_sha256
        and state.get("options") == expected
    )


def combine_statistics(
    base: dict[str, Any],
    synthetic: dict[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key in (
        "samples",
        "tokens",
        "random_baseline_sum",
        "center_correct",
        "duplicate_or_bloom_filtered",
        "overlength_filtered",
        "documents_seen",
        "documents_with_samples",
        "document_sample_cap_filtered",
    ):
        result[key] = result.get(key, 0) + synthetic.get(key, 0)
    result["quality_documents_filtered"] = synthetic["quality_documents_filtered"]
    result["domain_samples"].update(synthetic["domain_samples"])
    result["position_histogram"] = [
        left + right
        for left, right in zip(result["position_histogram"], synthetic["position_histogram"])
    ]
    result["cells"] = [
        [left + right for left, right in zip(base_row, synthetic_row)]
        for base_row, synthetic_row in zip(result["cells"], synthetic["cells"])
    ]
    result["maximum_sequence_length"] = max(
        result["maximum_sequence_length"],
        synthetic["maximum_sequence_length"],
    )
    return result


def main() -> None:
    options = parse_arguments()
    for name in (
        "target_samples",
        "synthetic_shards",
        "max_sequence_length",
        "max_samples_per_document",
        "batch_rows",
        "checkpoint_documents",
    ):
        if getattr(options, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if options.synthetic_shards & (options.synthetic_shards - 1):
        raise ValueError("--synthetic-shards must be a power of two")

    base_manifest_path = options.base_manifest.resolve()
    output_dir = options.output_dir.resolve()
    raw_dir = options.raw_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        print(manifest_path.read_text(encoding="utf-8"), flush=True)
        return

    base_manifest = load_json(base_manifest_path)
    base_sha256 = hashlib.sha256(base_manifest_path.read_bytes()).hexdigest()
    domains = list(base_manifest.get("domains", BASE_DOMAINS))
    if SYNTHETIC_DOMAIN in domains:
        raise ValueError(f"Base manifest already contains {SYNTHETIC_DOMAIN}")
    domains.append(SYNTHETIC_DOMAIN)
    domain_index = len(domains) - 1

    state_path = output_dir / "preparation-state.json"
    if state_path.exists():
        state = load_json(state_path)
        if not compatible_state(state, base_sha256, options):
            raise ValueError(f"Incompatible preparation state: {state_path}")
    else:
        state = {
            "format_version": 1,
            "base_manifest_sha256": base_sha256,
            "options": {
                "target_samples": options.target_samples,
                "synthetic_shards": options.synthetic_shards,
                "max_sequence_length": options.max_sequence_length,
                "max_samples_per_document": options.max_samples_per_document,
            },
            "source_file_index": 0,
            "rows_processed_in_file": 0,
            "shard_sizes": [0] * options.synthetic_shards,
            "statistics": empty_statistics(10),
            "downloads": [],
        }
        write_json_atomic(state_path, state)

    bloom = BloomFilter()
    restored = seed_bloom(base_manifest, bloom, output_dir, options.synthetic_shards)
    if restored != state["statistics"]["samples"]:
        raise ValueError(
            f"Preparation state contains {state['statistics']['samples']} samples; "
            f"restored {restored}"
        )

    writer = ShardWriter(
        output_dir,
        options.synthetic_shards,
        list(state["shard_sizes"]),
    )
    writer.open()
    urls = source_urls()
    try:
        while state["statistics"]["samples"] < options.target_samples:
            file_index = state["source_file_index"]
            if file_index >= len(urls):
                raise RuntimeError("Synthetic source exhausted before reaching target")
            source_path, download = ensure_download(urls[file_index], file_index, raw_dir)
            if not any(item["source_index"] == file_index for item in state["downloads"]):
                state["downloads"].append(download)
            parquet = pq.ParquetFile(source_path)
            source_row = 0
            checkpoint_at = (
                state["statistics"]["documents_seen"] + options.checkpoint_documents
            )
            for batch in parquet.iter_batches(
                batch_size=options.batch_rows,
                columns=["content"],
            ):
                for content in batch.column(0).to_pylist():
                    if source_row < state["rows_processed_in_file"]:
                        source_row += 1
                        continue
                    source_row += 1
                    state["statistics"]["documents_seen"] += 1
                    state["rows_processed_in_file"] = source_row
                    normalized = normalize_document(content)
                    if not quality_document(normalized):
                        state["statistics"]["quality_documents_filtered"] += 1
                        continue
                    candidates = adjacent_samples(normalized)
                    samples = evenly_capped(candidates, options.max_samples_per_document)
                    state["statistics"]["document_sample_cap_filtered"] += (
                        len(candidates) - len(samples)
                    )
                    before = state["statistics"]["samples"]
                    for text, target in samples:
                        if state["statistics"]["samples"] >= options.target_samples:
                            break
                        add_sample(
                            text,
                            target,
                            domain_index,
                            bloom,
                            writer,
                            state["statistics"],
                            options,
                        )
                    if state["statistics"]["samples"] > before:
                        state["statistics"]["documents_with_samples"] += 1

                    if state["statistics"]["documents_seen"] >= checkpoint_at:
                        writer.flush_all()
                        state["shard_sizes"] = list(writer.sizes)
                        write_json_atomic(state_path, state)
                        checkpoint_at += options.checkpoint_documents
                        print(
                            f"synthetic documents={state['statistics']['documents_seen']} "
                            f"samples={state['statistics']['samples']} "
                            f"file={file_index} row={source_row}",
                            flush=True,
                        )
                    if state["statistics"]["samples"] >= options.target_samples:
                        break
                if state["statistics"]["samples"] >= options.target_samples:
                    break

            if state["statistics"]["samples"] < options.target_samples:
                state["source_file_index"] += 1
                state["rows_processed_in_file"] = 0
            writer.flush_all()
            state["shard_sizes"] = list(writer.sizes)
            write_json_atomic(state_path, state)
    finally:
        writer.close()

    synthetic_shards = []
    for index in range(options.synthetic_shards):
        part = output_dir / f"synthetic-train-{index:03d}.jsonl.part"
        final = output_dir / f"synthetic-train-{index:03d}.jsonl"
        os.replace(part, final)
        synthetic_shards.append({
            "path": project_relative(final),
            "bytes": final.stat().st_size,
        })

    combined = {
        "format": "super-reader-sharded-training-v1",
        "source": "full Chinese Wikipedia plus CLUE and Ultra-FineWeb-L3 Chinese multi-style synthetic data",
        "base_manifest": project_relative(base_manifest_path),
        "base_manifest_sha256": base_sha256,
        "synthetic_source": {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "license": "Apache-2.0; see upstream dataset card for redistribution terms",
            "downloads": state["downloads"],
        },
        "domains": domains,
        "evaluation_source_dir": base_manifest["evaluation_source_dir"],
        "vocabulary_path": base_manifest["vocabulary_path"],
        "length_bucket_maximums": LENGTH_BUCKET_MAXIMUMS,
        "position_bins": 10,
        "maximum_sequence_length": options.max_sequence_length,
        "shards": [*base_manifest["shards"], *synthetic_shards],
        "statistics": combine_statistics(
            base_manifest["statistics"],
            state["statistics"],
        ),
    }
    write_json_atomic(manifest_path, combined)
    print(json.dumps({
        "manifest": str(manifest_path),
        "base_samples": base_manifest["statistics"]["samples"],
        "synthetic_samples": state["statistics"]["samples"],
        "combined_samples": combined["statistics"]["samples"],
        "synthetic_documents": state["statistics"]["documents_seen"],
        "downloads": state["downloads"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
