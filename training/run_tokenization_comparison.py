#!/usr/bin/env python3
"""Run a controlled word-gap versus character-gap medium smoke comparison."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_ROOT = SCRIPT_DIR / "data" / "processed" / "tokenization-medium"
ARTIFACT_ROOT = SCRIPT_DIR / "artifacts" / "tokenization-medium"
SEED = 2026090405
EPOCHS = 3
TRAIN_PER_DOMAIN = 4000
VALIDATION_PER_DOMAIN = 1000
TEST_PER_DOMAIN = 1000


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def verify_same_examples() -> dict[str, int]:
    counts = {}
    for split in ("train", "validation", "test"):
        word = read_json_lines(DATA_ROOT / "word" / f"{split}.jsonl")
        character = read_json_lines(DATA_ROOT / "character" / f"{split}.jsonl")
        if [record["id"] for record in word] != [record["id"] for record in character]:
            raise RuntimeError(f"The {split} sample IDs differ between tokenization modes")

        for word_record, character_record in zip(word, character):
            word_text = "".join(word_record["tokens"])
            character_text = "".join(character_record["tokens"])
            word_left = "".join(word_record["tokens"][: word_record["target_index"] + 1])
            character_left = "".join(
                character_record["tokens"][: character_record["target_index"] + 1]
            )
            if word_text != character_text or word_left != character_left:
                raise RuntimeError(f"Tokenization changed text or target boundary: {word_record['id']}")
        counts[split] = len(word)
    return counts


def metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    test = metrics["test"]
    return {
        "best_epoch": metrics["best_epoch"],
        "parameters": metrics["parameter_count"],
        "vocabulary_size": metrics["vocabulary_size"],
        "maximum_sequence_length": metrics["maximum_sequence_length"],
        "average_test_candidate_gaps": metrics["average_candidate_gaps"]["test"],
        "test_accuracy": test["accuracy"],
        "test_mrr": test["mean_reciprocal_rank"],
        "test_mean_absolute_character_error": test["mean_absolute_character_error"],
        "test_within_one_character_accuracy": test["within_one_character_accuracy"],
        "test_within_two_characters_accuracy": test["within_two_characters_accuracy"],
        "test_per_domain_accuracy": test["per_domain_accuracy"],
        "random_accuracy": metrics["baselines"]["test"]["random_accuracy"],
        "center_accuracy": metrics["baselines"]["test"]["center_accuracy"],
    }


def main() -> None:
    for mode in ("word", "character"):
        data_dir = DATA_ROOT / mode
        artifact_dir = ARTIFACT_ROOT / mode
        run([
            "node",
            "training/prepare_smoke_data.mjs",
            "--output-dir", str(data_dir),
            "--tokenization", mode,
            "--train-per-domain", str(TRAIN_PER_DOMAIN),
            "--validation-per-domain", str(VALIDATION_PER_DOMAIN),
            "--test-per-domain", str(TEST_PER_DOMAIN),
            "--seed", str(SEED),
        ])

    sample_counts = verify_same_examples()

    results = {}
    for mode in ("word", "character"):
        run([
            sys.executable,
            "training/run_smoke.py",
            "--data-dir", str(DATA_ROOT / mode),
            "--artifact-dir", str(ARTIFACT_ROOT / mode),
            "--epochs", str(EPOCHS),
            "--seed", str(SEED),
        ])
        metrics = json.loads(
            (ARTIFACT_ROOT / mode / "smoke-metrics.json").read_text(encoding="utf-8")
        )
        results[mode] = metric_summary(metrics)

    comparison = {
        "seed": SEED,
        "epochs": EPOCHS,
        "sample_counts": sample_counts,
        "same_examples_check": "passed",
        "non_han_policy": "excluded in both modes",
        "word": results["word"],
        "character": results["character"],
        "character_minus_word": {
            "test_accuracy": results["character"]["test_accuracy"] - results["word"]["test_accuracy"],
            "test_mrr": results["character"]["test_mrr"] - results["word"]["test_mrr"],
            "test_mean_absolute_character_error": (
                results["character"]["test_mean_absolute_character_error"]
                - results["word"]["test_mean_absolute_character_error"]
            ),
        },
    }
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = ARTIFACT_ROOT / "comparison.json"
    output_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
