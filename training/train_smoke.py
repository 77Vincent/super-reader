#!/usr/bin/env python3
"""Train the residual boundary chooser on a multithreaded CPU."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import signal
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "processed"
DEFAULT_ARTIFACT_DIR = SCRIPT_DIR / "artifacts"
PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unk>"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from artifact-dir/training-state.pt",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-tokens-per-batch", type=int, default=8192)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument(
        "--residual-blocks",
        type=int,
        default=4,
        help="Residual blocks; each block contains two kernel-3 convolutions",
    )
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=2026090405)
    parser.add_argument(
        "--threads",
        type=int,
        default=min(5, os.cpu_count() or 1),
        help="PyTorch intra-op CPU threads; five is fastest on the benchmarked M5 Pro",
    )
    parser.add_argument(
        "--interop-threads",
        type=int,
        default=1,
        help="Parallel operators per training process",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_cpu(threads: int, interop_threads: int) -> None:
    if threads < 1 or interop_threads < 1:
        raise ValueError("CPU thread counts must be positive")
    # Set these before constructing tensors or running any operators. More than
    # five intra-op threads slow this short-convolution workload on the M5 Pro.
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(interop_threads)
    torch.set_flush_denormal(True)


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
            "training_weight": float(record.get("training_weight", 1.0)),
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
    batch_indices: list[list[int]] | None = None,
    start_batch: int = 0,
) -> Iterator[dict[str, Any]]:
    batches = (
        batch_indices
        if batch_indices is not None
        else make_batch_indices(
            records,
            batch_size,
            max_tokens_per_batch,
            shuffle=shuffle,
            seed=seed,
        )
    )
    for indices in batches[start_batch:]:
        selected = [records[index] for index in indices]
        batch_length = max(len(record["token_ids"]) for record in selected)
        token_ids = np.zeros((len(selected), batch_length), dtype=np.int64)
        token_mask = np.zeros((len(selected), batch_length), dtype=np.float32)
        gap_mask = np.zeros((len(selected), batch_length - 1), dtype=np.bool_)
        targets = np.zeros(len(selected), dtype=np.int64)
        sample_weights = np.ones(len(selected), dtype=np.float32)

        for row, record in enumerate(selected):
            length = len(record["token_ids"])
            token_ids[row, :length] = record["token_ids"]
            token_mask[row, :length] = 1.0
            gap_mask[row, : length - 1] = True
            targets[row] = record["target_index"]
            sample_weights[row] = record["training_weight"]

        yield {
            "token_ids": torch.from_numpy(token_ids),
            "token_mask": torch.from_numpy(token_mask),
            "gap_mask": torch.from_numpy(gap_mask),
            "targets": torch.from_numpy(targets),
            "sample_weights": torch.from_numpy(sample_weights),
            "sample_weight_sum": float(sample_weights.sum()),
            "records": selected,
        }


class ThreeTapConv1d(nn.Module):
    """Kernel-3 convolution expressed as three CPU-efficient matrix products."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, channels, 3))
        self.bias = nn.Parameter(torch.empty(channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(channels * 3)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        center = F.linear(inputs, self.weight[:, :, 1], self.bias)
        left = F.pad(
            F.linear(inputs[:, :-1], self.weight[:, :, 0]),
            (0, 0, 1, 0),
        )
        right = F.pad(
            F.linear(inputs[:, 1:], self.weight[:, :, 2]),
            (0, 0, 0, 1),
        )
        return center + left + right


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.normalization = nn.LayerNorm(channels)
        self.first = ThreeTapConv1d(channels)
        self.second = ThreeTapConv1d(channels)
        self.residual_scale = nn.Parameter(torch.tensor([0.1]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.normalization(inputs)
        hidden = F.gelu(self.first(hidden), approximate="tanh")
        hidden = self.second(hidden)
        return inputs + self.residual_scale * hidden


class BoundaryChooser(nn.Module):
    def __init__(self, vocabulary_size: int, channels: int, residual_blocks: int):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, channels)
        # tinygrad's Embedding used by the earlier trainer initializes with
        # Glorot uniform; preserve that scale instead of PyTorch's N(0, 1).
        nn.init.xavier_uniform_(self.embedding.weight)
        self.blocks = nn.ModuleList(
            ResidualConvBlock(channels) for _ in range(residual_blocks)
        )
        self.boundary_hidden = nn.Linear(channels * 4, channels)
        self.boundary_output = nn.Linear(channels, 1)

    def forward(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        gap_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.embedding(token_ids)
        expanded_mask = token_mask.unsqueeze(-1)
        hidden = hidden * expanded_mask
        for block in self.blocks:
            hidden = block(hidden) * expanded_mask

        left = hidden[:, :-1]
        right = hidden[:, 1:]
        gap_features = torch.cat((left, right, right - left, right * left), dim=-1)
        hidden_gaps = F.gelu(self.boundary_hidden(gap_features), approximate="tanh")
        logits = self.boundary_output(hidden_gaps).squeeze(-1)
        return logits.masked_fill(~gap_mask, -1e9)


class TrainingStopRequested(Exception):
    """Raised at a safe batch boundary after an interrupt request."""


def evaluate(
    model: BoundaryChooser,
    records: list[dict[str, Any]],
    batch_size: int,
    max_tokens_per_batch: int,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    reciprocal_rank = 0.0
    absolute_character_error = 0
    within_one_character = 0
    within_two_characters = 0
    domain_counts: dict[str, int] = defaultdict(int)
    domain_correct: dict[str, int] = defaultdict(int)

    with torch.inference_mode():
        for batch in iterate_batches(
            records,
            batch_size,
            max_tokens_per_batch,
            shuffle=False,
            seed=0,
        ):
            if should_stop is not None and should_stop():
                raise TrainingStopRequested
            logits = model(batch["token_ids"], batch["token_mask"], batch["gap_mask"])
            loss = F.cross_entropy(logits, batch["targets"])
            predictions_tensor = logits.argmax(dim=1)
            target_logits = logits.gather(1, batch["targets"].unsqueeze(1))
            ranks = 1 + (logits > target_logits).sum(dim=1)
            predictions = predictions_tensor.numpy()
            targets = batch["targets"].numpy()

            count = len(targets)
            total_loss += loss.item() * count
            total_count += count
            total_correct += int((predictions == targets).sum())
            reciprocal_rank += float((1.0 / ranks.float()).sum().item())

            for row, target in enumerate(targets):
                record = batch["records"][row]
                prediction = int(predictions[row])
                predicted_character_position = len("".join(record["tokens"][: prediction + 1]))
                target_character_position = len("".join(record["tokens"][: int(target) + 1]))
                character_error = abs(predicted_character_position - target_character_position)
                absolute_character_error += character_error
                within_one_character += character_error <= 1
                within_two_characters += character_error <= 2

            for record, prediction, target in zip(batch["records"], predictions, targets):
                domain = record["domain"]
                domain_counts[domain] += 1
                domain_correct[domain] += int(prediction == target)

    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
        "mean_reciprocal_rank": reciprocal_rank / total_count,
        "mean_absolute_character_error": absolute_character_error / total_count,
        "within_one_character_accuracy": within_one_character / total_count,
        "within_two_characters_accuracy": within_two_characters / total_count,
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
    model.eval()
    with torch.inference_mode():
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


def clone_state(model: BoundaryChooser) -> dict[str, torch.Tensor]:
    return copy.deepcopy(model.state_dict())


def restore_state(model: BoundaryChooser, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state)


def save_browser_compatible_checkpoint(
    model: BoundaryChooser,
    checkpoint_path: Path,
) -> None:
    """Write PyTorch weights using the existing tinygrad safetensors writer."""
    os.environ["DEBUG"] = "0"
    os.environ.setdefault("CLANG", "1")
    from tinygrad import Tensor as TinyTensor
    from tinygrad.nn.state import safe_save

    checkpoint_state = {}
    for name, value in model.state_dict().items():
        array = value.detach().cpu().numpy().astype(np.float32, copy=True)
        if name.endswith(("first.weight", "second.weight")):
            # Browser inference and the previous tinygrad model use Conv2d's
            # [out, in, 1, kernel] representation for these 1-D convolutions.
            array = np.expand_dims(array, axis=2)
        checkpoint_state[name] = TinyTensor(array, device="CLANG")
    safe_save(checkpoint_state, str(checkpoint_path))


def data_identity(
    raw_records: dict[str, list[dict[str, Any]]],
    summary_path: Path,
) -> dict[str, Any]:
    return {
        "split_sizes": {
            split: len(records)
            for split, records in raw_records.items()
        },
        "summary_sha256": (
            hashlib.sha256(summary_path.read_bytes()).hexdigest()
            if summary_path.exists()
            else None
        ),
    }


def resume_configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "batch_size": args.batch_size,
        "max_tokens_per_batch": args.max_tokens_per_batch,
        "channels": args.channels,
        "residual_blocks": args.residual_blocks,
        "vocab_size": args.vocab_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }


def save_training_state(path: Path, state: dict[str, Any]) -> None:
    """Atomically save all state needed to continue at the next safe batch."""
    temporary_path = path.with_name(f"{path.name}.part")
    try:
        torch.save(state, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_training_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != 1:
        raise ValueError(f"Unsupported resume checkpoint format: {path}")
    return state


def main() -> None:
    args = parse_arguments()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.channels < 1 or args.residual_blocks < 1:
        raise ValueError("--channels and --residual-blocks must be positive")
    configure_cpu(args.threads, args.interop_threads)
    seed_everything(args.seed)
    raw_records = {
        split: read_json_lines(args.data_dir / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    summary_path = args.data_dir / "summary.json"
    data_summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.artifact_dir / "boundary-smoke.safetensors"
    vocabulary_path = args.artifact_dir / "boundary-smoke-vocabulary.json"
    training_state_path = args.artifact_dir / "training-state.pt"
    current_data_identity = data_identity(raw_records, summary_path)
    current_configuration = resume_configuration(args)
    resume_state = load_training_state(training_state_path) if args.resume else None

    if resume_state is not None:
        if resume_state["configuration"] != current_configuration:
            raise ValueError(
                "Resume checkpoint configuration differs from the current arguments"
            )
        if resume_state["data_identity"] != current_data_identity:
            raise ValueError("Resume checkpoint was created from different training data")
        vocabulary = resume_state["vocabulary"]
    else:
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

    model = BoundaryChooser(
        len(vocabulary),
        args.channels,
        args.residual_blocks,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
        foreach=True,
    )
    best_validation = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    start_epoch = 1
    start_batch = 0
    resumed_running_loss = 0.0
    resumed_seen_weight = 0.0
    resumed_seen_examples = 0
    resumed_training_seconds = 0.0

    if resume_state is not None:
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        best_validation = resume_state["best_validation"]
        best_epoch = resume_state["best_epoch"]
        best_state = resume_state["best_state"]
        history = resume_state["history"]
        progress = resume_state["progress"]
        start_epoch = progress["epoch"]
        start_batch = progress["next_batch"]
        resumed_running_loss = progress["running_loss"]
        resumed_seen_weight = progress["seen_weight"]
        resumed_seen_examples = progress["seen_examples"]
        resumed_training_seconds = progress["training_seconds"]
        if start_epoch > args.epochs + 1:
            raise ValueError(
                f"Resume checkpoint is already at epoch {start_epoch - 1}, "
                f"beyond requested --epochs {args.epochs}"
            )
        print(
            f"resumed epoch={start_epoch:02d} next_batch={start_batch} "
            f"completed_epochs={len(history)}",
            flush=True,
        )

    stop_control = {"requested": False, "signals": 0}

    def handle_stop_request(signum: int, _frame: Any) -> None:
        stop_control["signals"] += 1
        if stop_control["signals"] == 1:
            stop_control["requested"] = True
            print(
                "stop requested; saving at the next safe batch boundary "
                "(send the signal again to force exit)",
                flush=True,
            )
            return
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    signal.signal(signal.SIGINT, handle_stop_request)
    signal.signal(signal.SIGTERM, handle_stop_request)

    def persist_training_state(
        *,
        epoch: int,
        next_batch: int,
        running_loss: float,
        seen_weight: float,
        seen_examples: int,
        training_seconds: float,
    ) -> None:
        save_training_state(training_state_path, {
            "format_version": 1,
            "configuration": current_configuration,
            "data_identity": current_data_identity,
            "vocabulary": vocabulary,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_validation": best_validation,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "history": history,
            "progress": {
                "epoch": epoch,
                "next_batch": next_batch,
                "running_loss": running_loss,
                "seen_weight": seen_weight,
                "seen_examples": seen_examples,
                "training_seconds": training_seconds,
            },
        })

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if epoch == start_epoch:
            epoch_start_batch = start_batch
            running_loss = resumed_running_loss
            seen_weight = resumed_seen_weight
            seen_examples = resumed_seen_examples
            previous_training_seconds = resumed_training_seconds
        else:
            epoch_start_batch = 0
            running_loss = 0.0
            seen_weight = 0.0
            seen_examples = 0
            previous_training_seconds = 0.0

        training_batches = make_batch_indices(
            records["train"],
            args.batch_size,
            args.max_tokens_per_batch,
            shuffle=True,
            seed=args.seed + epoch,
        )
        if epoch_start_batch > len(training_batches):
            raise ValueError(
                f"Resume batch {epoch_start_batch} exceeds epoch batch count "
                f"{len(training_batches)}"
            )
        epoch_started = time.perf_counter()
        for batch_index, batch in enumerate(
            iterate_batches(
                records["train"],
                args.batch_size,
                args.max_tokens_per_batch,
                shuffle=True,
                seed=args.seed + epoch,
                batch_indices=training_batches,
                start_batch=epoch_start_batch,
            ),
            start=epoch_start_batch,
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["token_ids"], batch["token_mask"], batch["gap_mask"])
            per_sample_loss = F.cross_entropy(logits, batch["targets"], reduction="none")
            weighted_losses = per_sample_loss * batch["sample_weights"]
            loss = weighted_losses.sum() / batch["sample_weight_sum"]
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch["sample_weight_sum"]
            seen_weight += batch["sample_weight_sum"]
            seen_examples += len(batch["records"])

            if stop_control["requested"]:
                training_seconds = (
                    previous_training_seconds
                    + time.perf_counter() - epoch_started
                )
                persist_training_state(
                    epoch=epoch,
                    next_batch=batch_index + 1,
                    running_loss=running_loss,
                    seen_weight=seen_weight,
                    seen_examples=seen_examples,
                    training_seconds=training_seconds,
                )
                print(
                    f"training stopped; resume checkpoint={training_state_path} "
                    f"epoch={epoch} next_batch={batch_index + 1}",
                    flush=True,
                )
                return

        training_seconds = previous_training_seconds + time.perf_counter() - epoch_started

        try:
            validation = evaluate(
                model,
                records["validation"],
                args.batch_size,
                args.max_tokens_per_batch,
                should_stop=lambda: stop_control["requested"],
            )
        except TrainingStopRequested:
            persist_training_state(
                epoch=epoch,
                next_batch=len(training_batches),
                running_loss=running_loss,
                seen_weight=seen_weight,
                seen_examples=seen_examples,
                training_seconds=training_seconds,
            )
            print(
                f"training stopped during validation; "
                f"resume checkpoint={training_state_path} epoch={epoch}",
                flush=True,
            )
            return
        row = {
            "epoch": epoch,
            "train_loss": running_loss / seen_weight,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "training_seconds": training_seconds,
            "training_examples_per_second": seen_examples / training_seconds,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['validation_loss']:.4f} "
            f"val_acc={row['validation_accuracy']:.3f} "
            f"samples_s={row['training_examples_per_second']:.0f}",
            flush=True,
        )

        if validation["accuracy"] > best_validation:
            best_validation = validation["accuracy"]
            best_epoch = epoch
            best_state = clone_state(model)

        persist_training_state(
            epoch=epoch + 1,
            next_batch=0,
            running_loss=0.0,
            seen_weight=0.0,
            seen_examples=0,
            training_seconds=0.0,
        )
        if stop_control["requested"]:
            print(
                f"training stopped after epoch {epoch}; "
                f"resume checkpoint={training_state_path}",
                flush=True,
            )
            return

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    restore_state(model, best_state)
    save_browser_compatible_checkpoint(model, checkpoint_path)
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
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metrics = {
        "seed": args.seed,
        "tokenization": data_summary.get("tokenization", "unspecified"),
        "candidate_positions": data_summary.get("candidate_positions", "unspecified"),
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
        "vocabulary_size": len(vocabulary),
        "maximum_sequence_length": maximum_length,
        "architecture": {
            "channels": args.channels,
            "residual_blocks": args.residual_blocks,
            "convolutions_per_block": 2,
            "kernel_size": 3,
            "dilation": 1,
        },
        "training_backend": {
            "framework": "PyTorch",
            "framework_version": torch.__version__,
            "device": "cpu",
            "intra_op_threads": torch.get_num_threads(),
            "inter_op_threads": torch.get_num_interop_threads(),
            "logical_cpu_count": os.cpu_count(),
            "gelu_approximation": "tanh",
            "convolution_implementation": (
                "equivalent three-tap left/center/right matrix products"
            ),
            "optimizer_foreach": True,
            "resumable_checkpoint": str(training_state_path),
        },
        "data_sizes": {split: len(items) for split, items in records.items()},
        "training_weighting": {
            "strategy": data_summary.get("training_balance", {}).get(
                "strategy",
                "unit weights",
            ),
            "minimum_weight": min(record["training_weight"] for record in records["train"]),
            "maximum_weight": max(record["training_weight"] for record in records["train"]),
            "mean_weight": sum(
                record["training_weight"] for record in records["train"]
            ) / len(records["train"]),
        },
        "average_candidate_gaps": {
            split: sum(len(record["tokens"]) - 1 for record in items) / len(items)
            for split, items in records.items()
        },
        "batching": {
            "maximum_examples_per_batch": args.batch_size,
            "maximum_tokens_per_batch": args.max_tokens_per_batch,
            "strategy": (
                "power-of-two length buckets with per-batch dynamic padding; "
                "batch cap benchmarked for CPU throughput"
            ),
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
        "resume_checkpoint": str(training_state_path),
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
