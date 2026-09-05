#!/usr/bin/env python3
"""Compare shared L/R gap features with cmpres-style candidate windows."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F

try:
    import train_smoke as base
except ModuleNotFoundError:  # Allow importing this file as training.* in tests.
    from training import train_smoke as base


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "processed" / "tokenization-medium" / "character"
DEFAULT_OUTPUT = SCRIPT_DIR / "artifacts" / "boundary-encoder-comparison.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--residual-blocks", type=int, default=3)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-tokens-per-batch", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=2026090405)
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--interop-threads", type=int, default=1)
    return parser.parse_args()


class SharedPairBoundaryChooser(nn.Module):
    """Current one-pass encoder with explicit [L, R, R-L, R*L] features."""

    def __init__(self, vocabulary_size: int, channels: int, residual_blocks: int):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, channels)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.blocks = nn.ModuleList(
            base.ResidualConvBlock(channels) for _ in range(residual_blocks)
        )
        self.boundary_hidden = nn.Linear(channels * 4, channels)
        self.boundary_output = nn.Linear(channels, 1)

    def forward(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        gap_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.embedding(token_ids) * token_mask.unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden) * token_mask.unsqueeze(-1)

        left = hidden[:, :-1]
        right = hidden[:, 1:]
        features = torch.cat((left, right, right - left, right * left), dim=-1)
        features = F.gelu(self.boundary_hidden(features), approximate="tanh")
        logits = self.boundary_output(features).squeeze(-1)
        return logits.masked_fill(~gap_mask, -1e9)


class ValidThreeTapConv1d(nn.Module):
    """The current kernel-3 convolution without outer padding."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, channels, 3))
        self.bias = nn.Parameter(torch.empty(channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(channels * 3)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[1] < 3:
            raise ValueError("valid kernel-3 convolution needs at least three positions")
        return (
            F.linear(inputs[:, :-2], self.weight[:, :, 0])
            + F.linear(inputs[:, 1:-1], self.weight[:, :, 1], self.bias)
            + F.linear(inputs[:, 2:], self.weight[:, :, 2])
        )


class ValidResidualConvBlock(nn.Module):
    """Current two-convolution residual block, cropped for a valid window."""

    def __init__(self, channels: int):
        super().__init__()
        self.normalization = nn.LayerNorm(channels)
        self.first = ValidThreeTapConv1d(channels)
        self.second = ValidThreeTapConv1d(channels)
        self.residual_scale = nn.Parameter(torch.tensor([0.1]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.normalization(inputs)
        hidden = F.gelu(self.first(hidden), approximate="tanh")
        hidden = self.second(hidden)
        return inputs[:, 2:-2] + self.residual_scale * hidden


class CandidateWindowBoundaryChooser(nn.Module):
    """Cmpres-style centered candidate window with a learned split marker.

    The token unit, data, channel count, convolution count, optimizer, and loss
    stay identical to SharedPairBoundaryChooser. Only candidate representation
    changes: every gap gets its own centered window, which valid convolutions
    reduce to one vector before scoring.
    """

    def __init__(self, vocabulary_size: int, channels: int, residual_blocks: int):
        super().__init__()
        self.radius = residual_blocks * 2
        self.embedding = nn.Embedding(vocabulary_size, channels)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.blocks = nn.ModuleList(
            ValidResidualConvBlock(channels) for _ in range(residual_blocks)
        )
        self.split_marker = nn.Parameter(torch.empty(channels))
        nn.init.normal_(self.split_marker, mean=0.0, std=1 / math.sqrt(channels))
        self.boundary_hidden = nn.Linear(channels, channels)
        self.boundary_output = nn.Linear(channels, 1)

    def forward(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        gap_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.embedding(token_ids) * token_mask.unsqueeze(-1)
        batch_size, sequence_length, channels = hidden.shape
        padded = F.pad(hidden, (0, 0, self.radius, self.radius))

        # For the gap after position i, collect radius tokens from each side.
        raw = padded.unfold(1, self.radius * 2, 1)
        raw = raw[:, 1:sequence_length].permute(0, 1, 3, 2)
        marker = self.split_marker.reshape(1, 1, 1, channels).expand(
            batch_size,
            sequence_length - 1,
            1,
            channels,
        )
        windows = torch.cat(
            (raw[:, :, : self.radius], marker, raw[:, :, self.radius :]),
            dim=2,
        )

        candidate_features = windows[gap_mask]
        for block in self.blocks:
            candidate_features = block(candidate_features)
        if candidate_features.shape[1] != 1:
            raise RuntimeError("candidate window did not reduce to one position")

        candidate_features = candidate_features[:, 0]
        candidate_features = F.gelu(
            self.boundary_hidden(candidate_features),
            approximate="tanh",
        )
        candidate_scores = self.boundary_output(candidate_features).squeeze(-1)
        logits = candidate_scores.new_full(gap_mask.shape, -1e9)
        logits[gap_mask] = candidate_scores
        return logits


def train_model(
    name: str,
    factory: Callable[[], nn.Module],
    records: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    base.seed_everything(args.seed)
    model = factory()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
        foreach=True,
    )
    best_validation = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    total_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen_weight = 0.0
        seen_examples = 0
        epoch_started = time.perf_counter()
        for batch in base.iterate_batches(
            records["train"],
            args.batch_size,
            args.max_tokens_per_batch,
            shuffle=True,
            seed=args.seed + epoch,
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["token_ids"], batch["token_mask"], batch["gap_mask"])
            losses = F.cross_entropy(logits, batch["targets"], reduction="none")
            weighted_losses = losses * batch["sample_weights"]
            loss = weighted_losses.sum() / batch["sample_weight_sum"]
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch["sample_weight_sum"]
            seen_weight += batch["sample_weight_sum"]
            seen_examples += len(batch["records"])

        training_seconds = time.perf_counter() - epoch_started
        validation = base.evaluate(
            model,
            records["validation"],
            args.batch_size,
            args.max_tokens_per_batch,
        )
        row = {
            "epoch": epoch,
            "train_loss": running_loss / seen_weight,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_mean_reciprocal_rank": validation["mean_reciprocal_rank"],
            "training_seconds": training_seconds,
            "training_examples_per_second": seen_examples / training_seconds,
        }
        history.append(row)
        print(
            f"{name} epoch={epoch:02d} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['validation_loss']:.4f} "
            f"val_acc={row['validation_accuracy']:.3f} "
            f"samples_s={row['training_examples_per_second']:.0f}",
            flush=True,
        )
        if validation["accuracy"] > best_validation:
            best_validation = validation["accuracy"]
            best_epoch = epoch
            best_state = base.clone_state(model)

    if best_state is None:
        raise RuntimeError("comparison model did not finish an epoch")
    base.restore_state(model, best_state)
    validation = base.evaluate(
        model,
        records["validation"],
        args.batch_size,
        args.max_tokens_per_batch,
    )
    test = base.evaluate(
        model,
        records["test"],
        args.batch_size,
        args.max_tokens_per_batch,
    )
    return {
        "name": name,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "wall_seconds": time.perf_counter() - total_started,
        "history": history,
        "validation": validation,
        "test": test,
    }


def main() -> None:
    args = parse_arguments()
    if args.epochs < 1 or args.channels < 1 or args.residual_blocks < 1:
        raise ValueError("epochs, channels, and residual blocks must be positive")
    base.configure_cpu(args.threads, args.interop_threads)
    base.seed_everything(args.seed)

    raw_records = {
        split: base.read_json_lines(args.data_dir / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    vocabulary = base.build_vocabulary(raw_records["train"], args.vocab_size)
    records = {
        split: base.encode_records(items, vocabulary)
        for split, items in raw_records.items()
    }
    factories: list[tuple[str, Callable[[], nn.Module]]] = [
        (
            "shared_lr_features",
            lambda: SharedPairBoundaryChooser(
                len(vocabulary), args.channels, args.residual_blocks
            ),
        ),
        (
            "cmpres_candidate_window",
            lambda: CandidateWindowBoundaryChooser(
                len(vocabulary), args.channels, args.residual_blocks
            ),
        ),
    ]
    results = [
        train_model(name, factory, records, args)
        for name, factory in factories
    ]

    by_name = {row["name"]: row for row in results}
    baseline = by_name["shared_lr_features"]
    candidate = by_name["cmpres_candidate_window"]
    output = {
        "experiment": "controlled_boundary_encoder_comparison",
        "data_dir": str(args.data_dir),
        "data_sizes": {split: len(items) for split, items in records.items()},
        "seed": args.seed,
        "epochs": args.epochs,
        "channels": args.channels,
        "residual_blocks": args.residual_blocks,
        "convolution_layers": args.residual_blocks * 2,
        "candidate_positions": "between every adjacent Han character",
        "controlled_constants": [
            "same examples and vocabulary",
            "same character-gap candidates and labels",
            "same channel and convolution counts",
            "same two-convolution residual topology",
            "same optimizer, loss, batches, seed, and checkpoint selection",
        ],
        "changed_variable": (
            "shared full-sequence [L,R,R-L,R*L] features versus a centered "
            "candidate window containing a learned split marker"
        ),
        "results": results,
        "delta": {
            "validation_accuracy": (
                candidate["validation"]["accuracy"]
                - baseline["validation"]["accuracy"]
            ),
            "test_accuracy": (
                candidate["test"]["accuracy"] - baseline["test"]["accuracy"]
            ),
            "training_throughput_ratio": (
                candidate["history"][-1]["training_examples_per_second"]
                / baseline["history"][-1]["training_examples_per_second"]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "baseline_test_accuracy": baseline["test"]["accuracy"],
        "cmpres_test_accuracy": candidate["test"]["accuracy"],
        "test_accuracy_delta": output["delta"]["test_accuracy"],
        "training_throughput_ratio": output["delta"]["training_throughput_ratio"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
