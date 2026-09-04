#!/usr/bin/env python3
"""Train a tiny three-block residual boundary chooser for a smoke test."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from tinygrad import Tensor, nn
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_save


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "processed"
DEFAULT_ARTIFACT_DIR = SCRIPT_DIR / "artifacts"
PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unk>"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-tokens-per-batch", type=int, default=2048)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=2026090405)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    Tensor.manual_seed(seed)


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_vocabulary(
    records: list[dict[str, Any]],
    maximum_size: int,
) -> dict[str, int]:
    counts = Counter(token for record in records for token in record["tokens"])
    vocabulary = {PAD_TOKEN: 0, UNKNOWN_TOKEN: 1}
    for token, _ in counts.most_common(maximum_size - len(vocabulary)):
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_records(
    records: list[dict[str, Any]],
    vocabulary: dict[str, int],
) -> list[dict[str, Any]]:
    unknown = vocabulary[UNKNOWN_TOKEN]
    return [
        {
            "id": record["id"],
            "domain": record["domain"],
            "document_id": record["document_id"],
            "tokens": record["tokens"],
            "target_index": record["target_index"],
            "token_ids": [vocabulary.get(token, unknown) for token in record["tokens"]],
        }
        for record in records
    ]


def bucket_ceiling(length: int) -> int:
    return 1 << (length - 1).bit_length()


def make_batch_indices(
    records: list[dict[str, Any]],
    batch_size: int,
    max_tokens_per_batch: int,
    *,
    shuffle: bool,
    seed: int,
) -> list[list[int]]:
    randomizer = random.Random(seed)
    buckets: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        buckets[bucket_ceiling(len(record["token_ids"]))].append(index)

    batches = []
    for ceiling in sorted(buckets):
        indices = buckets[ceiling]
        if shuffle:
            randomizer.shuffle(indices)
        effective_batch_size = max(
            1,
            min(batch_size, max_tokens_per_batch // ceiling),
        )
        batches.extend(
            indices[offset : offset + effective_batch_size]
            for offset in range(0, len(indices), effective_batch_size)
        )

    if shuffle:
        randomizer.shuffle(batches)
    return batches


def iterate_batches(
    records: list[dict[str, Any]],
    batch_size: int,
    max_tokens_per_batch: int,
    *,
    shuffle: bool,
    seed: int,
) -> Iterator[dict[str, Any]]:
    batches = make_batch_indices(
        records,
        batch_size,
        max_tokens_per_batch,
        shuffle=shuffle,
        seed=seed,
    )
    for indices in batches:
        selected = [records[index] for index in indices]
        batch_length = max(len(record["token_ids"]) for record in selected)
        token_ids = np.zeros((len(selected), batch_length), dtype=np.int32)
        token_mask = np.zeros((len(selected), batch_length), dtype=np.float32)
        gap_mask = np.zeros((len(selected), batch_length - 1), dtype=np.bool_)
        targets = np.zeros(len(selected), dtype=np.int32)

        for row, record in enumerate(selected):
            length = len(record["token_ids"])
            token_ids[row, :length] = record["token_ids"]
            token_mask[row, :length] = 1.0
            gap_mask[row, : length - 1] = True
            targets[row] = record["target_index"]

        yield {
            "token_ids": Tensor(token_ids),
            "token_mask": Tensor(token_mask),
            "gap_mask": Tensor(gap_mask),
            "targets": Tensor(targets),
            "records": selected,
        }


class ResidualConvBlock:
    def __init__(self, channels: int):
        self.normalization = nn.LayerNorm(channels)
        self.first = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 3),
            padding=(0, 1),
        )
        self.second = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 3),
            padding=(0, 1),
        )
        self.residual_scale = Tensor([0.1], requires_grad=True)

    def __call__(self, inputs: Tensor) -> Tensor:
        batch_size, sequence_length, channels = inputs.shape
        hidden = self.normalization(inputs)
        hidden = hidden.permute(0, 2, 1).reshape(
            batch_size,
            channels,
            1,
            sequence_length,
        )
        hidden = self.first(hidden).gelu()
        hidden = self.second(hidden)
        hidden = hidden.reshape(batch_size, channels, sequence_length).permute(0, 2, 1)
        return inputs + self.residual_scale * hidden


class BoundaryChooser:
    def __init__(self, vocabulary_size: int, channels: int):
        self.embedding = nn.Embedding(vocabulary_size, channels)
        self.blocks = [ResidualConvBlock(channels) for _ in range(3)]
        self.boundary_hidden = nn.Linear(channels * 4, channels)
        self.boundary_output = nn.Linear(channels, 1)

    def __call__(
        self,
        token_ids: Tensor,
        token_mask: Tensor,
        gap_mask: Tensor,
    ) -> Tensor:
        hidden = self.embedding(token_ids)
        expanded_mask = token_mask.reshape(*token_mask.shape, 1)
        hidden = hidden * expanded_mask
        for block in self.blocks:
            hidden = block(hidden) * expanded_mask

        left = hidden[:, :-1]
        right = hidden[:, 1:]
        gap_features = left.cat(
            right,
            right - left,
            right * left,
            dim=-1,
        )
        logits = self.boundary_output(self.boundary_hidden(gap_features).gelu()).squeeze(-1)
        return gap_mask.where(logits, -1e9)


def evaluate(
    model: BoundaryChooser,
    records: list[dict[str, Any]],
    batch_size: int,
    max_tokens_per_batch: int,
) -> dict[str, Any]:
    Tensor.training = False
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    reciprocal_rank = 0.0
    domain_counts: dict[str, int] = defaultdict(int)
    domain_correct: dict[str, int] = defaultdict(int)

    for batch in iterate_batches(
        records,
        batch_size,
        max_tokens_per_batch,
        shuffle=False,
        seed=0,
    ):
        logits = model(batch["token_ids"], batch["token_mask"], batch["gap_mask"])
        loss = logits.sparse_categorical_crossentropy(batch["targets"]).mean()
        logits_array = logits.numpy()
        targets = batch["targets"].numpy()
        predictions = logits_array.argmax(axis=1)
        ranks = np.argsort(-logits_array, axis=1)

        count = len(targets)
        total_loss += float(loss.numpy()) * count
        total_count += count
        total_correct += int((predictions == targets).sum())
        for row, target in enumerate(targets):
            rank = int(np.where(ranks[row] == target)[0][0]) + 1
            reciprocal_rank += 1.0 / rank

        for record, prediction, target in zip(batch["records"], predictions, targets):
            domain = record["domain"]
            domain_counts[domain] += 1
            domain_correct[domain] += int(prediction == target)

    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
        "mean_reciprocal_rank": reciprocal_rank / total_count,
        "per_domain_accuracy": {
            domain: domain_correct[domain] / count
            for domain, count in sorted(domain_counts.items())
        },
    }


def random_baseline(records: list[dict[str, Any]]) -> float:
    return sum(1.0 / (len(record["tokens"]) - 1) for record in records) / len(records)


def center_baseline(records: list[dict[str, Any]]) -> float:
    correct = sum(
        record["target_index"] == (len(record["tokens"]) - 1) // 2
        for record in records
    )
    return correct / len(records)


def build_length_prior(records: list[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for record in records:
        counts[len(record["tokens"])][record["target_index"]] += 1

    return {
        length: min(
            target_counts,
            key=lambda target: (
                -target_counts[target],
                abs(target - (length - 1) / 2),
                target,
            ),
        )
        for length, target_counts in counts.items()
    }


def length_prior_baseline(
    records: list[dict[str, Any]],
    prior: dict[int, int],
) -> float:
    correct = 0
    for record in records:
        length = len(record["tokens"])
        prediction = prior.get(length, (length - 1) // 2)
        correct += prediction == record["target_index"]
    return correct / len(records)


def batching_statistics(
    records: list[dict[str, Any]],
    batch_size: int,
    max_tokens_per_batch: int,
) -> dict[str, Any]:
    batches = make_batch_indices(
        records,
        batch_size,
        max_tokens_per_batch,
        shuffle=False,
        seed=0,
    )
    real_tokens = sum(len(record["token_ids"]) for record in records)
    padded_tokens = sum(
        len(indices) * max(len(records[index]["token_ids"]) for index in indices)
        for indices in batches
    )
    return {
        "batch_count": len(batches),
        "smallest_batch_size": min(map(len, batches)),
        "largest_batch_size": max(map(len, batches)),
        "real_tokens": real_tokens,
        "padded_tokens": padded_tokens,
        "padding_efficiency": real_tokens / padded_tokens,
    }


def recursive_tree(
    tokens: list[str],
    scores: list[float],
    start: int = 0,
    end: int | None = None,
) -> dict[str, Any]:
    end = len(tokens) if end is None else end
    if end - start == 1:
        return {"text": tokens[start]}

    boundary = max(range(start, end - 1), key=lambda index: scores[index])
    return {
        "text": "".join(tokens[start:end]),
        "boundary_after": boundary,
        "boundary_score": scores[boundary],
        "left": recursive_tree(tokens, scores, start, boundary + 1),
        "right": recursive_tree(tokens, scores, boundary + 1, end),
    }


def predict_record(
    model: BoundaryChooser,
    record: dict[str, Any],
    max_tokens_per_batch: int,
) -> dict[str, Any]:
    batch = next(iterate_batches(
        [record],
        batch_size=1,
        max_tokens_per_batch=max_tokens_per_batch,
        shuffle=False,
        seed=0,
    ))
    Tensor.training = False
    logits = model(batch["token_ids"], batch["token_mask"], batch["gap_mask"])
    valid_count = len(record["tokens"]) - 1
    valid_logits = logits.numpy()[0, :valid_count]
    shifted = valid_logits - valid_logits.max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    prediction = int(valid_logits.argmax())
    tokens = record["tokens"]
    gold_index = record["target_index"]
    return {
        "id": record["id"],
        "domain": record["domain"],
        "tokens": tokens,
        "gold_index": gold_index,
        "predicted_index": prediction,
        "gold_split": [
            "".join(tokens[: gold_index + 1]),
            "".join(tokens[gold_index + 1 :]),
        ],
        "predicted_split": [
            "".join(tokens[: prediction + 1]),
            "".join(tokens[prediction + 1 :]),
        ],
        "tree": recursive_tree(tokens, probabilities.tolist()),
    }


def clone_state(model: BoundaryChooser) -> dict[str, np.ndarray]:
    return {
        name: value.numpy().copy()
        for name, value in get_state_dict(model).items()
    }


def restore_state(model: BoundaryChooser, state: dict[str, np.ndarray]) -> None:
    load_state_dict(
        model,
        {name: Tensor(value) for name, value in state.items()},
        verbose=False,
    )


def main() -> None:
    args = parse_arguments()
    seed_everything(args.seed)
    raw_records = {
        split: read_json_lines(args.data_dir / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    vocabulary = build_vocabulary(raw_records["train"], args.vocab_size)
    records = {
        split: encode_records(split_records, vocabulary)
        for split, split_records in raw_records.items()
    }
    maximum_length = max(
        len(record["tokens"])
        for split_records in records.values()
        for record in split_records
    )

    model = BoundaryChooser(len(vocabulary), args.channels)
    optimizer = optim.AdamW(
        get_parameters(model),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.artifact_dir / "boundary-smoke.safetensors"
    vocabulary_path = args.artifact_dir / "boundary-smoke-vocabulary.json"
    best_validation = -math.inf
    best_epoch = 0
    best_state: dict[str, np.ndarray] | None = None
    history = []

    for epoch in range(1, args.epochs + 1):
        Tensor.training = True
        running_loss = 0.0
        seen = 0
        for batch in iterate_batches(
            records["train"],
            args.batch_size,
            args.max_tokens_per_batch,
            shuffle=True,
            seed=args.seed + epoch,
        ):
            optimizer.zero_grad()
            logits = model(batch["token_ids"], batch["token_mask"], batch["gap_mask"])
            loss = logits.sparse_categorical_crossentropy(batch["targets"]).mean()
            loss.backward()
            optimizer.step()
            count = len(batch["records"])
            running_loss += float(loss.numpy()) * count
            seen += count

        validation = evaluate(
            model,
            records["validation"],
            args.batch_size,
            args.max_tokens_per_batch,
        )
        row = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['validation_loss']:.4f} "
            f"val_acc={row['validation_accuracy']:.3f}",
            flush=True,
        )

        if validation["accuracy"] > best_validation:
            best_validation = validation["accuracy"]
            best_epoch = epoch
            best_state = clone_state(model)

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    restore_state(model, best_state)
    safe_save(get_state_dict(model), str(checkpoint_path))
    vocabulary_path.write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    validation = evaluate(
        model,
        records["validation"],
        args.batch_size,
        args.max_tokens_per_batch,
    )
    test = evaluate(
        model,
        records["test"],
        args.batch_size,
        args.max_tokens_per_batch,
    )
    predictions = [
        predict_record(model, record, args.max_tokens_per_batch)
        for record in records["test"][:4]
    ]
    length_prior = build_length_prior(records["train"])
    baselines = {
        split: {
            "random_accuracy": random_baseline(items),
            "center_accuracy": center_baseline(items),
            "train_length_lookup_accuracy": length_prior_baseline(items, length_prior),
        }
        for split, items in records.items()
    }
    parameter_count = sum(math.prod(parameter.shape) for parameter in get_parameters(model))
    metrics = {
        "seed": args.seed,
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
        "vocabulary_size": len(vocabulary),
        "maximum_sequence_length": maximum_length,
        "architecture": {
            "channels": args.channels,
            "residual_blocks": 3,
            "convolutions_per_block": 2,
            "kernel_size": 3,
            "dilation": 1,
        },
        "data_sizes": {split: len(items) for split, items in records.items()},
        "batching": {
            "maximum_examples_per_batch": args.batch_size,
            "maximum_tokens_per_batch": args.max_tokens_per_batch,
            "strategy": "power-of-two length buckets with per-batch dynamic padding",
            "splits": {
                split: batching_statistics(items, args.batch_size, args.max_tokens_per_batch)
                for split, items in records.items()
            },
        },
        "validation": validation,
        "test": test,
        "baselines": baselines,
        "test_random_baseline_accuracy": random_baseline(records["test"]),
        "history": history,
        "sample_predictions": predictions,
    }
    metrics_path = args.artifact_dir / "smoke-metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "metrics": str(metrics_path),
        "best_epoch": best_epoch,
        "parameters": parameter_count,
        "validation_accuracy": validation["accuracy"],
        "test_accuracy": test["accuracy"],
        "test_baselines": baselines["test"],
        "test_per_domain": test["per_domain_accuracy"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
