#!/usr/bin/env python3
"""Memory-bounded training over compact on-disk JSONL shards."""

from __future__ import annotations

import argparse
import copy
import gc
import gzip
import hashlib
import json
import math
import os
import random
import signal
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from text_policy import DATA_POLICY, require_data_policy, valid_proxy_label
from sample_subset import (keep_target, source_name, record_source, empty_statistics,
                           count_sample, summarize_inventory, write_json, limit_batch_lengths)

from train_smoke import (
    BoundaryChooser,
    MODEL_EXECUTION_CONTRACT,
    TrainingStopRequested,
    batch_to_device,
    batching_statistics,
    center_baseline,
    clone_state,
    configure_cpu,
    configure_mps,
    evaluate,
    iterate_batches,
    make_batch_indices,
    mps_memory_restart_needed,
    predict_record,
    random_baseline,
    restore_state,
    save_browser_compatible_checkpoint,
    save_training_state,
    seed_everything,
    training_execution,
    weighted_loss_totals,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(os.environ.get('SUPER_READER_PROJECT_ROOT', SCRIPT_DIR.parent))
DEFAULT_MANIFEST = SCRIPT_DIR / "data" / "processed" / "wiki-full-sharded-128" / "manifest.json"
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "processed"
DEFAULT_ARTIFACT_DIR = SCRIPT_DIR / "artifacts" / "wiki-full-128ch-8conv-3ep"
PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unk>"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--vocabulary", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-tokens-per-batch", type=int, default=8192)
    parser.add_argument("--checkpoint-shards", type=int, default=4)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--defer-test", action="store_true",
                        help="Save endpoint validation and stop before test evaluation; resume without this flag after selection")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--position-weighting", choices=("inverse-cell", "none"), default="none")
    parser.add_argument("--min-side-characters", type=int, default=1,
                        help="Keep targets with at least this many Unicode characters on each side, in all splits")
    parser.add_argument("--max-sequence-length", type=int, default=0,
                        help="Discard whole samples exceeding this many Unicode characters in all splits; 0 disables")
    parser.add_argument("--weighting-manifest", type=Path,
                        help="Use position frequencies from this reference manifest (only for explicit position weighting)")
    parser.add_argument("--gradient-clip", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026090405)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.4)
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def open_jsonl(path: Path):
    """Read ordinary and independently compressed training shards identically."""
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")


def weight_table(manifest: dict[str, Any], mode: str = "inverse-cell") -> list[list[float]]:
    cells = manifest["statistics"]["cells"]
    if mode == "none":
        return [[1.0] * len(row) for row in cells]
    if mode != "inverse-cell":
        raise ValueError(f"Unknown position weighting: {mode}")
    result = []
    for row in cells:
        total = sum(row)
        occupied = sum(count > 0 for count in row)
        result.append([
            total / occupied / count if count and occupied else 0.0
            for count in row
        ])
    return result


def selection_score(validation: dict[str, Any]) -> float:
    """Select solely by overall accuracy; corpus-source diagnostics are descriptive."""
    return validation["accuracy"]


def combined_weight_mean(
    shards: list[Path],
    position_weights: list[list[float]],
    minimum_side_characters: int = 1,
    maximum_sequence_length: int = 0,
) -> float:
    """Only explicit nonuniform position weights need a metadata scan."""
    if all(weight == 1.0 for row in position_weights for weight in row):
        return 1.0
    total = 0.0
    samples = 0
    for path in shards:
        with open_jsonl(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                text, target, _, bucket, position = json.loads(line)
                if not keep_target(len(text), target, minimum_side_characters, maximum_sequence_length):
                    continue
                total += position_weights[bucket][position]
                samples += 1
    if samples == 0 or total <= 0:
        raise ValueError("Cannot normalize an empty or zero-weight training set")
    return total / samples


def read_training_shard(
    path: Path,
    vocabulary: dict[str, int],
    weights: list[list[float]],
    sources: list[str],
    minimum_side_characters: int = 1,
    counts: dict[str, Any] | None = None,
    maximum_sequence_length: int = 0,
    include_overlength_for_batching: bool = False,
) -> list[dict[str, Any]]:
    unknown = vocabulary[UNKNOWN_TOKEN]
    records = []
    with open_jsonl(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            text, target, source_index, bucket, position = json.loads(line)
            source = source_name(sources[source_index]) if 0 <= source_index < len(sources) else source_name(None)
            valid_side = keep_target(len(text), target, minimum_side_characters)
            kept = keep_target(len(text), target, minimum_side_characters, maximum_sequence_length)
            if counts is not None:
                count_sample(counts, len(text), target, source, kept)
            # The trainer can retain these host-side records only to construct
            # the original batch plan. limit_batch_lengths removes them before
            # tensor allocation, forward passes, loss and optimizer updates.
            if not kept and not (include_overlength_for_batching and valid_side):
                continue
            token_ids = [vocabulary.get(token, unknown) for token in text]
            records.append({
                "token_ids": token_ids,
                "target_index": target,
                "training_weight": weights[bucket][position],
                "corpus_source": source,
            })
    return records


def read_evaluation_records(
    path: Path,
    vocabulary: dict[str, int],
    minimum_side_characters: int = 1,
    counts: dict[str, Any] | None = None,
    maximum_sequence_length: int = 0,
) -> list[dict[str, Any]]:
    unknown = vocabulary[UNKNOWN_TOKEN]
    records = []
    with open_jsonl(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not valid_proxy_label(record.get("punctuation")):
                raise ValueError(f"Invalid proxy label remains in {path}: {record.get('id')}")
            source = record_source(record)
            kept = keep_target(len(record["tokens"]), record["target_index"], minimum_side_characters,
                               maximum_sequence_length)
            if counts is not None:
                count_sample(counts, len(record["tokens"]), record["target_index"], source, kept)
            if not kept:
                continue
            record["corpus_source"] = source
            record.pop("domain", None)  # serialized legacy field is provenance, not semantics
            record["training_weight"] = 1.0
            record["token_ids"] = [
                vocabulary.get(token, unknown)
                for token in record["tokens"]
            ]
            records.append(record)
    return records


def expanded_initialization(
    model: BoundaryChooser,
    checkpoint_path: Path,
    vocabulary: dict[str, int],
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source = checkpoint["best_state"] or checkpoint["model_state"]
    target = model.state_dict()
    source_channels = source["embedding.weight"].shape[1]
    target_channels = target["embedding.weight"].shape[1]
    channels = min(source_channels, target_channels)
    source_vocabulary = checkpoint["vocabulary"]
    shared_tokens = sorted(set(source_vocabulary) & set(vocabulary))

    with torch.no_grad():
        for token in shared_tokens:
            target["embedding.weight"][vocabulary[token], :channels].copy_(
                source["embedding.weight"][source_vocabulary[token], :channels]
            )
        block = 0
        while f"blocks.{block}.first.weight" in source and f"blocks.{block}.first.weight" in target:
            prefix = f"blocks.{block}"
            target[f"{prefix}.normalization.weight"][:channels].copy_(
                source[f"{prefix}.normalization.weight"][:channels]
            )
            target[f"{prefix}.normalization.bias"][:channels].copy_(
                source[f"{prefix}.normalization.bias"][:channels]
            )
            for layer in ("first", "second"):
                target[f"{prefix}.{layer}.weight"][:channels, :channels].copy_(
                    source[f"{prefix}.{layer}.weight"][:channels, :channels]
                )
                target[f"{prefix}.{layer}.bias"][:channels].copy_(
                    source[f"{prefix}.{layer}.bias"][:channels]
                )
            target[f"{prefix}.residual_scale"].copy_(source[f"{prefix}.residual_scale"])
            block += 1

        target["boundary_hidden.bias"][:channels].copy_(
            source["boundary_hidden.bias"][:channels]
        )
        for group in range(4):
            target_start = group * target_channels
            source_start = group * source_channels
            target["boundary_hidden.weight"][:channels, target_start : target_start + channels].copy_(
                source["boundary_hidden.weight"][:channels, source_start : source_start + channels]
            )
        target["boundary_output.weight"][:, :channels].copy_(
            source["boundary_output.weight"][:, :channels]
        )
        target["boundary_output.bias"].copy_(source["boundary_output.bias"])

        # New residual blocks initially act as identity functions. Their random
        # convolution branches remain trainable through the learned scale, while
        # a depth-only expansion preserves the inherited model's initial logits.
        added_blocks = len(model.blocks) - block
        for index in range(block, len(model.blocks)):
            target[f"blocks.{index}.residual_scale"].zero_()

    model.load_state_dict(target)
    return {
        "path": str(checkpoint_path),
        "source_channels": source_channels,
        "copied_channels": channels,
        "copied_blocks": block,
        "added_identity_blocks": added_blocks,
        "source_best_epoch": checkpoint["best_epoch"],
        "copied_token_embeddings": len(shared_tokens),
        "new_token_embeddings": len(vocabulary) - len(shared_tokens),
        "embedding_mapping": "token identity, not row position",
        "optimizer": "new AdamW",
    }


def identity_for_run(
    manifest_path: Path,
    data_dir: Path,
) -> dict[str, Any]:
    return {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "validation_bytes": (data_dir / "validation.jsonl").stat().st_size,
        "test_bytes": (data_dir / "test.jsonl").stat().st_size,
        "summary_sha256": hashlib.sha256((data_dir / "summary.json").read_bytes()).hexdigest(),
    }


def main() -> None:
    args = parse_arguments()
    if min(args.epochs, args.channels, args.residual_blocks, args.checkpoint_shards) < 1:
        raise ValueError("Epoch, channel, block, and checkpoint counts must be positive")
    if args.min_side_characters < 1:
        raise ValueError("--min-side-characters must be positive")
    if args.max_sequence_length < 0:
        raise ValueError("--max-sequence-length cannot be negative")
    if args.resume and args.initialize_from:
        raise ValueError("Choose either --resume or --initialize-from")
    if min(args.max_shards, args.validation_limit, args.test_limit) < 0:
        raise ValueError("Shard and evaluation limits cannot be negative")
    if args.gradient_clip < 0:
        raise ValueError("--gradient-clip cannot be negative")
    configure_cpu(args.threads, args.interop_threads)
    device = configure_mps(args.mps_memory_fraction)
    seed_everything(args.seed)
    execution = training_execution(args.mps_memory_fraction)
    print("training execution: " + json.dumps(execution), flush=True)

    manifest_path = args.manifest.resolve()
    data_dir = args.data_dir.resolve()
    artifact_dir = args.artifact_dir.resolve()
    manifest = load_json(manifest_path)
    if manifest.get("format") != "super-reader-sharded-training-v1":
        raise ValueError(f"Unsupported shard manifest: {manifest_path}")
    data_summary = load_json(data_dir / "summary.json")
    for source in (manifest, data_summary):
        require_data_policy(source)
    vocabulary_path = (
        args.vocabulary.resolve()
        if args.vocabulary
        else resolve_project_path(manifest["vocabulary_path"])
    )
    vocabulary = load_json(vocabulary_path)
    if any(c not in vocabulary for c in '0123456789:、ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '):
        raise ValueError("Vocabulary lacks context characters; run build_context_vocabulary.py first")
    sources = manifest.get("corpus_sources", manifest.get("domains", []))
    shards = [resolve_project_path(item["path"]) for item in manifest["shards"]]
    if args.max_shards > 0:
        shards = shards[: args.max_shards]
    if not shards or any(not path.exists() for path in shards):
        raise FileNotFoundError("One or more training shards are missing")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if (artifact_dir / "training-state.pt").exists() and not args.resume:
        raise FileExistsError("Training state already exists; pass --resume or use a new artifact directory")
    weighting_manifest = load_json(args.weighting_manifest) if args.weighting_manifest else manifest
    require_data_policy(weighting_manifest)
    if any(weighting_manifest.get(key) != manifest.get(key)
           for key in ("length_bucket_maximums", "position_bins")):
        raise ValueError("Weighting manifest position/length bins differ from training data")
    weights = weight_table(weighting_manifest, args.position_weighting)
    state_path = artifact_dir / "training-state.pt"
    browser_checkpoint_path = artifact_dir / "boundary-smoke.safetensors"
    output_vocabulary_path = artifact_dir / "boundary-smoke-vocabulary.json"
    metrics_path = artifact_dir / "smoke-metrics.json"
    if state_path.exists() and not args.resume:
        raise FileExistsError(
            f"Training state already exists: {state_path}; "
            "pass --resume or choose a different --artifact-dir"
        )
    configuration = {
        "model_execution_contract": MODEL_EXECUTION_CONTRACT,
        "batch_size": args.batch_size,
        "max_tokens_per_batch": args.max_tokens_per_batch,
        "channels": args.channels,
        "residual_blocks": args.residual_blocks,
        "learning_rate": args.learning_rate,
        "source_weighting": "none",
        "position_weighting": args.position_weighting,
        "minimum_side_characters": args.min_side_characters,
        "maximum_sequence_length": args.max_sequence_length,
        "weighting_manifest_sha256": hashlib.sha256(args.weighting_manifest.read_bytes()).hexdigest() if args.weighting_manifest else None,
        "loss_reporting": "sum(weight * example_loss) / sum(weight), v1",
        "selection_metric": "overall_validation_accuracy",
        "gradient_clip": args.gradient_clip,
        "seed": args.seed,
        "max_shards": args.max_shards,
        "validation_limit": args.validation_limit,
        "test_limit": args.test_limit,
    }
    data_identity = identity_for_run(manifest_path, data_dir)
    resume_state = None
    if args.resume:
        resume_state = torch.load(state_path, map_location="cpu", weights_only=True)
        if resume_state.get("format_version") != 2:
            raise ValueError(f"Unsupported sharded checkpoint: {state_path}")
        resumed_configuration = {"minimum_side_characters": 1, "maximum_sequence_length": 0,
                                 **resume_state["configuration"]}
        if resumed_configuration != configuration:
            raise ValueError("Resume checkpoint configuration differs from current arguments")
        if resume_state["data_identity"] != data_identity:
            raise ValueError("Resume checkpoint was created from different data")
        vocabulary = resume_state["vocabulary"]

    # Reuse only after validating both configuration and data identity. Avoid a
    # full corpus scan on every worker refresh needed to release Metal graphs.
    training_weight_mean = (
        resume_state["training_weight_mean"]
        if resume_state is not None and "training_weight_mean" in resume_state
        else combined_weight_mean(shards, weights, args.min_side_characters, args.max_sequence_length)
    )
    if not math.isfinite(training_weight_mean) or training_weight_mean <= 0:
        raise ValueError("Invalid saved training weight mean")
    print(f"combined training weight mean: {training_weight_mean:.6f}", flush=True)

    source_inventory = resume_state.get("source_inventory", {}) if resume_state else {}
    validation_counts = empty_statistics()
    validation_records = read_evaluation_records(data_dir / "validation.jsonl", vocabulary,
                                                args.min_side_characters, validation_counts,
                                                args.max_sequence_length)
    if args.validation_limit > 0:
        validation_records = validation_records[: args.validation_limit]
    if not validation_records:
        raise ValueError("The selected validation subset is empty")
    validation_counts.update(scope="full validation file after target filtering, before evaluation limit",
                             evaluated_samples=len(validation_records))
    print(json.dumps({"stage": "validation-loaded", **validation_counts,
                      "evaluated_samples": len(validation_records)}), flush=True)
    model = BoundaryChooser(len(vocabulary), args.channels, args.residual_blocks)
    initialization = None
    if resume_state is None and args.initialize_from:
        initialization = expanded_initialization(model, args.initialize_from.resolve(), vocabulary)
        print(f"expanded initialization: {json.dumps(initialization)}", flush=True)
    model.to(device)
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
    start_shard = 0
    start_batch = 0
    resumed_loss = 0.0
    resumed_weight = 0.0
    resumed_examples = 0
    resumed_seconds = 0.0
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        best_validation = resume_state["best_validation"]
        best_epoch = resume_state["best_epoch"]
        best_state = resume_state["best_state"]
        history = resume_state["history"]
        initialization = resume_state.get("initialization")
        progress = resume_state["progress"]
        start_epoch = progress["epoch"]
        start_shard = progress["shard"]
        start_batch = progress["next_batch"]
        resumed_loss = progress["running_loss"]
        resumed_weight = progress["seen_weight"]
        resumed_examples = progress["seen_examples"]
        resumed_seconds = progress["training_seconds"]
        print(
            f"resumed epoch={start_epoch} shard={start_shard} next_batch={start_batch}",
            flush=True,
        )

    stop = {"requested": False, "signals": 0}

    def handle_stop(signum: int, _frame: Any) -> None:
        stop["signals"] += 1
        if stop["signals"] == 1:
            stop["requested"] = True
            print("stop requested; checkpointing at the next safe batch", flush=True)
            return
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    def persist(
        epoch: int,
        shard: int,
        next_batch: int,
        running_loss: float,
        seen_weight: float,
        seen_examples: int,
        training_seconds: float,
    ) -> None:
        save_training_state(state_path, {
            "format_version": 2,
            "configuration": configuration,
            # Historical checkpoints are device-independent; data and training
            # configuration above must still match exactly on MPS resume.
            "execution": execution,
            "training_weight_mean": training_weight_mean,
            "source_inventory": source_inventory,
            "data_identity": data_identity,
            "vocabulary": vocabulary,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_validation": best_validation,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "history": history,
            "initialization": initialization,
            "progress": {
                "epoch": epoch,
                "shard": shard,
                "next_batch": next_batch,
                "running_loss": running_loss,
                "seen_weight": seen_weight,
                "seen_examples": seen_examples,
                "training_seconds": training_seconds,
            },
        })

        write_json(artifact_dir / "source-statistics.json", {
            "meaning": "corpus provenance only; not semantic domains or training/selection weights",
            "train": summarize_inventory(source_inventory, len(shards)),
            "validation": validation_counts,
        })

    if resume_state is None and initialization is not None:
        initial_validation = evaluate(model, validation_records, args.batch_size, args.max_tokens_per_batch)
        best_validation = selection_score(initial_validation)
        best_state = clone_state(model)
        initialization["validation_before_training"] = initial_validation
        initialization["selection_score_before_training"] = best_validation
        print(f"initialized validation accuracy={initial_validation['accuracy']:.6f}", flush=True)
        persist(1, 0, 0, 0.0, 0.0, 0, 0.0)
    elif resume_state is None:
        persist(1, 0, 0, 0.0, 0.0, 0, 0.0)
        print("random initialization saved; no pretrained weights or optimizer loaded", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        order = list(range(len(shards)))
        random.Random(args.seed + epoch).shuffle(order)
        if epoch == start_epoch:
            shard_start = start_shard
            batch_start = start_batch
            running_loss = resumed_loss
            seen_weight = resumed_weight
            seen_examples = resumed_examples
            previous_seconds = resumed_seconds
        else:
            shard_start = 0
            batch_start = 0
            running_loss = 0.0
            seen_weight = 0.0
            seen_examples = 0
            previous_seconds = 0.0
        epoch_started = time.perf_counter()
        model.train()

        for shard_position in range(shard_start, len(order)):
            shard_number = order[shard_position]
            shard_counts = empty_statistics()
            records = read_training_shard(
                shards[shard_number],
                vocabulary,
                weights,
                sources,
                args.min_side_characters,
                shard_counts,
                args.max_sequence_length,
                include_overlength_for_batching=True,
            )
            source_inventory[str(shards[shard_number])] = shard_counts
            batches = make_batch_indices(
                records,
                args.batch_size,
                args.max_tokens_per_batch,
                shuffle=True,
                seed=args.seed + epoch * 1000 + shard_number,
            )
            batches = limit_batch_lengths(records, batches, args.max_sequence_length)
            first_batch = batch_start if shard_position == shard_start else 0
            if first_batch > len(batches):
                raise ValueError("Resume batch exceeds the selected shard's batch count")
            for batch_index, batch in enumerate(
                iterate_batches(
                    records,
                    args.batch_size,
                    args.max_tokens_per_batch,
                    shuffle=True,
                    seed=0,
                    batch_indices=batches,
                    start_batch=first_batch,
                ),
                start=first_batch,
            ):
                batch = batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch["token_ids"], batch["token_mask"], batch["gap_mask"])
                losses = F.cross_entropy(logits, batch["targets"], reduction="none")
                weighted = losses * batch["sample_weights"]
                # Corpus provenance never enters optimization. With position
                # weighting disabled, this is the ordinary per-batch mean loss.
                loss = weighted.mean() / training_weight_mean
                loss.backward()
                if args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                loss_total, weight_total = weighted_loss_totals(losses, batch["sample_weights"])
                running_loss += loss_total
                seen_weight += weight_total
                seen_examples += len(batch["records"])
                # Free the completed batch's graph before checking driver usage.
                del logits, losses, weighted, loss
                if batch_index == first_batch or (batch_index + 1) % 500 == 0:
                    elapsed = time.perf_counter() - epoch_started
                    session_samples = seen_examples - (resumed_examples if epoch == start_epoch else 0)
                    print(
                        f"epoch={epoch:02d} shard={shard_position + 1}/{len(shards)} "
                        f"batch={batch_index + 1}/{len(batches)} samples={seen_examples} "
                        f"session_seconds={elapsed:.1f} samples_s={session_samples / elapsed:.0f} "
                        f"mps_driver_mib={torch.mps.driver_allocated_memory() / 2**20:.0f}",
                        flush=True,
                    )
                if stop["requested"]:
                    seconds = previous_seconds + time.perf_counter() - epoch_started
                    persist(
                        epoch,
                        shard_position,
                        batch_index + 1,
                        running_loss,
                        seen_weight,
                        seen_examples,
                        seconds,
                    )
                    print(
                        f"training stopped; checkpoint={state_path} epoch={epoch} "
                        f"shard={shard_position} next_batch={batch_index + 1}",
                        flush=True,
                    )
                    return
                if mps_memory_restart_needed(args.mps_memory_fraction):
                    seconds = previous_seconds + time.perf_counter() - epoch_started
                    persist(epoch, shard_position, batch_index + 1, running_loss,
                            seen_weight, seen_examples, seconds)
                    print("MPS memory pressure; saved next batch, requesting worker refresh (exit 75)", flush=True)
                    raise SystemExit(75)
            del records, batches
            gc.collect()
            torch.mps.empty_cache()
            if (shard_position + 1) % args.checkpoint_shards == 0:
                seconds = previous_seconds + time.perf_counter() - epoch_started
                persist(
                    epoch,
                    shard_position + 1,
                    0,
                    running_loss,
                    seen_weight,
                    seen_examples,
                    seconds,
                )
            if (shard_position + 1) % 8 == 0:
                print(
                    f"epoch={epoch:02d} shards={shard_position + 1}/{len(shards)} "
                    f"samples={seen_examples}",
                    flush=True,
                )

        training_seconds = previous_seconds + time.perf_counter() - epoch_started
        if seen_weight <= 0:
            persist(epoch, len(shards), 0, running_loss, seen_weight, seen_examples, training_seconds)
            raise ValueError("The selected training subset is empty or has zero total weight")
        try:
            validation = evaluate(
                model,
                validation_records,
                args.batch_size,
                args.max_tokens_per_batch,
                should_stop=lambda: stop["requested"],
            )
        except TrainingStopRequested:
            persist(
                epoch,
                len(shards),
                0,
                running_loss,
                seen_weight,
                seen_examples,
                training_seconds,
            )
            print(f"stopped during validation; checkpoint={state_path}", flush=True)
            return
        score = selection_score(validation)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / seen_weight,
            "train_loss_definition": configuration["loss_reporting"],
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "selection_score": score,
            "training_seconds": training_seconds,
            "training_examples_per_second": seen_examples / training_seconds,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['validation_loss']:.4f} "
            f"val_acc={row['validation_accuracy']:.3f} "
            f"selection={row['selection_score']:.3f} "
            f"samples_s={row['training_examples_per_second']:.0f}",
            flush=True,
        )
        if score > best_validation:
            best_validation = score
            best_epoch = epoch
            best_state = clone_state(model)
        persist(epoch + 1, 0, 0, 0.0, 0.0, 0, 0.0)
        if stop["requested"]:
            print(f"stopped after epoch {epoch}; checkpoint={state_path}", flush=True)
            return

    stats = summarize_inventory(source_inventory, len(shards))
    if not stats["complete"] or not stats["samples"]:
        raise RuntimeError("Training did not load a nonempty complete selected corpus")
    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint")
    if args.defer_test:
        # This branch is deliberately after all optimization and checkpointing.
        # Re-evaluate the final endpoint so a resume after the last checkpoint
        # produces the same report even when the training loop is skipped.
        try:
            endpoint_validation = evaluate(
                model, validation_records, args.batch_size, args.max_tokens_per_batch,
                should_stop=lambda: stop["requested"],
            )
        except TrainingStopRequested:
            print(f"stopped during endpoint validation; checkpoint={state_path}", flush=True)
            return
        endpoint_score = selection_score(endpoint_validation)
        report = {
            "stage": "validation-complete-test-deferred",
            "configuration": configuration,
            "data_identity": data_identity,
            "data_sizes": {"train": stats["samples"], "validation": len(validation_records)},
            "corpus_source_statistics": {"train": stats, "validation": validation_counts},
            "sample_subset": {"minimum_side_characters": args.min_side_characters,
                              "maximum_sequence_length": args.max_sequence_length,
                              "train_excluded": stats.get("excluded_samples", 0),
                              "validation": validation_counts},
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "initialization": initialization,
            "history": history,
            "endpoint_validation": endpoint_validation,
            "endpoint_selection_score": endpoint_score,
            "best_epoch": best_epoch,
            "best_selection_score": best_validation,
            "checkpoint": str(state_path),
            "test_evaluated": False,
            "training_weighting": {"position_weighting": args.position_weighting,
                "position_weights": weights, "source_weighting": "none",
                "combined_weight_sample_mean_before_normalization": training_weight_mean},
        }
        destination = artifact_dir / "validation-only.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
        print(json.dumps({"validation_report": str(destination), "selection_score": endpoint_score,
                          "test_evaluated": False}), flush=True)
        return
    restore_state(model, best_state)
    save_browser_compatible_checkpoint(model, browser_checkpoint_path)
    output_vocabulary_path.write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    validation = evaluate(
        model,
        validation_records,
        args.batch_size,
        args.max_tokens_per_batch,
    )
    validation_baselines = {
        "random_accuracy": random_baseline(validation_records),
        "center_accuracy": center_baseline(validation_records),
    }
    validation_batching = batching_statistics(
        validation_records,
        args.batch_size,
        args.max_tokens_per_batch,
    )
    validation_size = len(validation_records)
    validation_average_gaps = sum(len(item["tokens"]) - 1 for item in validation_records) / validation_size
    del validation_records
    gc.collect()

    test_counts = empty_statistics()
    test_records = read_evaluation_records(data_dir / "test.jsonl", vocabulary,
                                          args.min_side_characters, test_counts, args.max_sequence_length)
    if args.test_limit > 0:
        test_records = test_records[: args.test_limit]
    if not test_records:
        raise ValueError("The selected test subset is empty")
    test_counts.update(scope="full test file after target filtering, before evaluation limit",
                       evaluated_samples=len(test_records))
    test = evaluate(model, test_records, args.batch_size, args.max_tokens_per_batch)
    test_baselines = {
        "random_accuracy": random_baseline(test_records),
        "center_accuracy": center_baseline(test_records),
    }
    test_batching = batching_statistics(test_records, args.batch_size, args.max_tokens_per_batch)
    test_average_gaps = sum(len(item["tokens"]) - 1 for item in test_records) / len(test_records)
    predictions = [
        predict_record(model, record, args.max_tokens_per_batch)
        for record in test_records[:4]
    ]
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    nonzero_weights = [value for row in weights for value in row if value]
    metrics = {
        **DATA_POLICY,
        "seed": args.seed,
        "tokenization": "character",
        "candidate_positions": "between every adjacent Unicode code point",
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
        "vocabulary_size": len(vocabulary),
        "maximum_sequence_length": max(
            stats["maximum_sequence_length"],
            max(len(record["tokens"]) for record in test_records),
        ),
        "architecture": {
            "channels": args.channels,
            "residual_blocks": args.residual_blocks,
            "convolutions_per_block": 2,
            "kernel_size": 3,
            "dilation": 1,
        },
        "training_backend": {
            "framework": "PyTorch",
            **execution,
            "sharded_streaming": True,
            "resumable_checkpoint": str(state_path),
        },
        "initialization": initialization,
        "data_manifest": str(manifest_path),
        "corpus_source_statistics": {"train": stats, "validation": validation_counts, "test": test_counts},
        "sample_subset": {"minimum_side_characters": args.min_side_characters,
                          "maximum_sequence_length": args.max_sequence_length,
                          "train_excluded": stats.get("excluded_samples", 0),
                          "validation": validation_counts, "test": test_counts},
        "data_sizes": {
            "train": stats["samples"],
            "validation": validation_size,
            "test": len(test_records),
        },
        "training_weighting": {
            "strategy": "uniform samples" if args.position_weighting == "none" else "position weighting only",
            "position_weighting": args.position_weighting,
            "position_weights": weights,
            "source_weighting": "none",
            "minimum_weight": min(nonzero_weights),
            "maximum_weight": max(nonzero_weights),
            "combined_weight_sample_mean_before_normalization": training_weight_mean,
            "loss_normalization": "fixed example count; weights are not renormalized per batch",
        },
        "checkpoint_selection": {
            "metric": "overall_validation_accuracy",
            "best_score": best_validation,
        },
        "average_candidate_gaps": {
            "train": stats["tokens"] / stats["samples"] - 1,
            "validation": validation_average_gaps,
            "test": test_average_gaps,
        },
        "batching": {
            "maximum_examples_per_batch": args.batch_size,
            "maximum_tokens_per_batch": args.max_tokens_per_batch,
            "strategy": "shuffled compact shards plus power-of-two length buckets",
            "splits": {
                "train": {"shards": len(shards)},
                "validation": validation_batching,
                "test": test_batching,
            },
        },
        "validation": validation,
        "test": test,
        "baselines": {
            "train": {
                "random_accuracy": stats["random_baseline_sum"] / stats["samples"],
                "center_accuracy": stats["center_correct"] / stats["samples"],
            },
            "validation": validation_baselines,
            "test": test_baselines,
        },
        "history": history,
        "sample_predictions": predictions,
    }
    write_json(artifact_dir / "source-statistics.json", metrics["corpus_source_statistics"])
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "checkpoint": str(browser_checkpoint_path),
        "resume_checkpoint": str(state_path),
        "metrics": str(metrics_path),
        "best_epoch": best_epoch,
        "parameters": parameter_count,
        "validation_accuracy": validation["accuracy"],
        "test_accuracy": test["accuracy"],
        "test_per_source": test["per_source_accuracy"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
