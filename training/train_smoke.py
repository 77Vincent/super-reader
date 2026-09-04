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
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=2026090432)
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
            **record,
            "token_ids": [vocabulary.get(token, unknown) for token in record["tokens"]],
        }
        for record in records
    ]


def iterate_batches(
    records: list[dict[str, Any]],
    batch_size: int,
    maximum_length: int,
    *,
    shuffle: bool,
    seed: int,
) -> Iterator[dict[str, Any]]:
    indices = list(range(len(records)))
    if shuffle:
        random.Random(seed).shuffle(indices)

    for offset in range(0, len(indices), batch_size):
        selected = [records[index] for index in indices[offset : offset + batch_size]]
        token_ids = np.zeros((len(selected), maximum_length), dtype=np.int32)
        token_mask = np.zeros((len(selected), maximum_length), dtype=np.float32)
        gap_mask = np.zeros((len(selected), maximum_length - 1), dtype=np.bool_)
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
    maximum_length: int,
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
        maximum_length,
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
    maximum_length: int,
) -> dict[str, Any]:
    batch = next(iterate_batches(
        [record],
        batch_size=1,
        maximum_length=maximum_length,
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
            maximum_length,
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
            maximum_length,
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
        maximum_length,
    )
    test = evaluate(
        model,
        records["test"],
        args.batch_size,
        maximum_length,
    )
    predictions = [
        predict_record(model, record, maximum_length)
        for record in records["test"][:4]
    ]
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
        "validation": validation,
        "test": test,
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
        "random_baseline": metrics["test_random_baseline_accuracy"],
        "test_per_domain": test["per_domain_accuracy"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
