#!/usr/bin/env python3
"""Prepare local web data, then continue the frozen best CNN in a separate run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "training/artifacts/web-mix-20m-192ch-16conv-20260915"
DATA = ROOT / "training/data/processed/web-mix-20m-192ch-16conv-20260915"
BEST = ROOT / "training/artifacts/unicode-context-192ch-16conv-20260914/epoch-1-backend/selected-state.pt"
BASE = ROOT / "training/data/processed/unicode-context-192ch-12conv-20260913-combined/manifest.json"
EVAL = ROOT / "training/data/processed/unicode-context-192ch-12conv-20260913-eval"
RAW = ROOT / "training/data/raw/ultra-fineweb-zh"
SOURCES = ["training/" + name for name in ("run_web_continuation.py", "prepare_web_data.py", "prepare_synthetic_data.py", "run_sharded.py",
    "run_smoke.py", "train_sharded.py", "train_smoke.py", "text_policy.py", "text-policy.json", "unicode-symbols.json", "export_browser_model.py")]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def run_with_recovery(stage, command, *, run_once, checkpoint, should_stop, on_refresh):
    """Refresh a Metal worker only after it has saved an advanced checkpoint."""
    while True:
        if should_stop():
            raise InterruptedError("Stopped between stages")
        previous_stamp = checkpoint.stat().st_mtime_ns if checkpoint.exists() else None
        code = run_once(command)
        if should_stop():
            raise InterruptedError("Graceful stop requested")
        if stage != "training" or code != 75:
            if code:
                raise RuntimeError(f"{stage} exited {code}; inspect its log")
            return
        if not checkpoint.exists() or checkpoint.stat().st_mtime_ns == previous_stamp:
            raise RuntimeError("Worker refresh did not advance its checkpoint")
        on_refresh()
        command = list(command)
        if "--initialize-from" in command:
            index = command.index("--initialize-from")
            del command[index:index + 2]
        if "--resume" not in command:
            command.append("--resume")


def validate_training_options(plan, args):
    requested = {"learning_rate": args.learning_rate,
                 "selection_macro_weight": args.selection_macro_weight}
    for key, value in requested.items():
        if plan["training"][key] != value:
            raise ValueError(f"Resume with the original {key}")
    if plan.get("checkpoint_shards", 4) != args.checkpoint_shards:
        raise ValueError("Resume with the original checkpoint frequency")


def extend_completed_run(run, plan, target_epochs, *, resume):
    """Extend only a completed run, preserving the optimizer and frozen trainer.

    Caller holds .run.lock. Publish the archive before the new plan; retrying
    after an interrupted plan update also clears obsolete completion records.
    """
    previous = plan["epochs"]
    if target_epochs < previous:
        raise ValueError("Cannot reduce the recorded epoch target")
    if target_epochs > previous:
        if not resume:
            raise ValueError("Extending training requires --resume")
        status_path, result_path = run / "status.json", run / "result.json"
        if not status_path.exists() or read(status_path).get("stage") != "complete" or not result_path.exists():
            raise ValueError("Finish the recorded training, evaluation and export before extending")
        result = read(result_path)
        if not result.get("history") or result["history"][-1]["epoch"] != previous:
            raise ValueError("Completed result does not match the recorded epoch target")
        sys.path.insert(0, str(ROOT / "training/.deps"))
        import torch
        candidate = run / "candidate"
        checkpoint = candidate / "training-state.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        progress = state["progress"]
        if (progress["epoch"], progress["shard"], progress["next_batch"]) != (previous + 1, 0, 0):
            raise ValueError("Extension requires a completed epoch checkpoint")
        if not state.get("optimizer_state", {}).get("state"):
            raise ValueError("Extension requires saved optimizer state")
        del state
        archive = run / "completed-epochs" / f"epoch-{previous}"
        archive.parent.mkdir(exist_ok=True)
        if not archive.exists():
            with tempfile.TemporaryDirectory(dir=archive.parent, prefix=".archive-") as temporary:
                staging = Path(temporary) / "snapshot"
                staging.mkdir()
                shutil.copytree(candidate, staging / "candidate")
                for name in ("run.json", "result.json", "status.json", "training.log", "export.log"):
                    if (run / name).exists():
                        shutil.copy2(run / name, staging / name)
                staging.rename(archive)
        for source in candidate.iterdir():
            if source.is_file() and sha(source) != sha(archive / "candidate" / source.name):
                raise ValueError(f"Completed archive differs: {source.name}")
        updated = {**plan, "epochs": target_epochs, "epoch_extensions": [
            *plan.get("epoch_extensions", []),
            {"from_epochs": previous, "to_epochs": target_epochs,
             "created_at": datetime.now(timezone.utc).isoformat(), "archive": str(archive),
             "checkpoint_sha256": sha(checkpoint), "optimizer_preserved": True,
             "coordinator_sha256": sha(Path(__file__))},
        ]}
        write(run / "run.json", updated)
        plan.update(updated)
    result_path = run / "result.json"
    if result_path.exists():
        completed_epochs = read(result_path)["history"][-1]["epoch"]
        if completed_epochs < target_epochs:
            if not resume or not plan.get("epoch_extensions"):
                raise ValueError("Incomplete extension requires --resume and its archive")
            # Keep the authoritative result in the archive, not at the active
            # run path where it could be mistaken for the new epoch's result.
            result_path.unlink()
            write(run / "status.json", {"stage": "ready-to-resume", "target_epochs": target_epochs,
                  "updated_at": datetime.now(timezone.utc).isoformat()})
    # Recovery if the process stopped between unlinking result and updating status.
    status_path = run / "status.json"
    if plan.get("epoch_extensions") and not result_path.exists() and status_path.exists() and read(status_path).get("stage") == "complete":
        write(status_path, {"stage": "ready-to-resume", "target_epochs": target_epochs,
              "updated_at": datetime.now(timezone.utc).isoformat()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--target-samples", type=int, default=20_000_000, help="Additional training pairs")
    parser.add_argument("--shards", type=int, default=128, help="Number of additional training shards")
    parser.add_argument("--base-manifest", type=Path, default=BASE)
    parser.add_argument("--evaluation", type=Path, default=EVAL)
    parser.add_argument("--initialize-from", type=Path, default=BEST)
    parser.add_argument("--freeze-evaluation", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--selection-macro-weight", type=float, default=0.5)
    parser.add_argument("--checkpoint-shards", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.target_samples, args.epochs, args.shards, args.checkpoint_shards) < 1:
        parser.error("Counts must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive")
    if not 0 <= args.selection_macro_weight <= 1:
        parser.error("Selection macro weight must be between zero and one")
    run, data = args.run_dir.resolve(), args.data_dir.resolve()
    base, evaluation, initialization = args.base_manifest.resolve(), args.evaluation.resolve(), args.initialize_from.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_path = run / "run.json"
    snapshot = run / "source"
    if not plan_path.exists():
        if shutil.disk_usage(ROOT).free < 40 * 1024 ** 3:
            raise RuntimeError("Need at least 40 GiB free for preparation and checkpoints")
        shutil.copy2(initialization, run / "initialization.pt")
        for name in SOURCES:
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        for name in ("data", ".deps"):
            link = snapshot / "training" / name
            if not link.exists():
                link.symlink_to(ROOT / "training" / name, target_is_directory=True)
        write(plan_path, {"created_at": datetime.now(timezone.utc).isoformat(), "new_train_samples": args.target_samples,
              "additional_shards": args.shards,
              "checkpoint_shards": args.checkpoint_shards,
              "epochs": args.epochs, "data": str(data), "base_manifest": str(base), "base_sha256": sha(base),
              "old_evaluation": str(evaluation), "old_evaluation_summary_sha256": sha(evaluation / "summary.json"),
              "evaluation_sha256": {split: sha(evaluation / f"{split}.jsonl") for split in ("validation", "test")},
              "freeze_evaluation": args.freeze_evaluation,
              "initialization_source": str(initialization), "initialization_sha256": sha(run / "initialization.pt"),
              "architecture": {"channels": 192, "residual_blocks": 8, "vocabulary_size": 8192},
              "source_hashes": {name: sha(snapshot / name) for name in SOURCES},
              "new_evaluation": "Keep complete validation/test splits byte-identical" if args.freeze_evaluation else "Whole documents split 96/2/2; retain all resulting validation/test samples",
              "history_exclusion": "All inherited training manifests plus frozen holdouts; projected input deduplication",
              "training": {"learning_rate": args.learning_rate, "domain_weight_power": 0.65, "selection_macro_weight": args.selection_macro_weight,
                           "gradient_clip": 1.0, "batch_size": 512, "max_tokens_per_batch": 8192, "threads": 1, "device": "mps"},
              "backend_bundle_sha256_at_start": sha(ROOT / "src/boundary-model-data.js")})
    plan = read(plan_path)
    validate_training_options(plan, args)
    if (plan["new_train_samples"], plan["data"]) != (args.target_samples, str(data)):
        raise ValueError("Resume with the original data directory and sample target")
    if plan.get("additional_shards", 128) != args.shards:
        raise ValueError("Resume with the original additional shard count")
    if (plan["base_manifest"], plan["old_evaluation"], plan.get("freeze_evaluation", False)) != (str(base), str(evaluation), args.freeze_evaluation):
        raise ValueError("Resume with the original base corpus and holdout mode")
    for path, expected in [(run / "initialization.pt", plan["initialization_sha256"]), (base, plan["base_sha256"]),
                           (evaluation / "summary.json", plan["old_evaluation_summary_sha256"])]:
        if sha(path) != expected:
            raise ValueError(f"Recorded input changed: {path}")
    for name, expected in plan["source_hashes"].items():
        if sha(snapshot / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    for split, expected in plan.get("evaluation_sha256", {}).items():
        if sha(evaluation / f"{split}.jsonl") != expected:
            raise ValueError(f"Frozen {split} changed")
    extend_completed_run(run, plan, args.epochs, resume=args.resume)
    if (run / "status.json").exists() and read(run / "status.json")["stage"] == "complete":
        print("Run is already complete; see result.json and candidate/smoke-metrics.json", flush=True)
        return
    child, stopped = None, False
    def stop(signum, _frame):
        nonlocal stopped
        if stopped:
            return
        stopped = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    def status(stage, **details):
        value = {"stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(), "coordinator_pid": os.getpid(), **details}
        write(run / "status.json", value)
        print(json.dumps(value), flush=True)
    environment = dict(os.environ, DEBUG="0", PYTHONUNBUFFERED="1", PYTHONPATH=str(ROOT / "training/.deps"),
                       SUPER_READER_PROJECT_ROOT=str(ROOT), PYTORCH_ENABLE_MPS_FALLBACK="0",
                       PYTORCH_MPS_FAST_MATH="0", PYTORCH_MPS_PREFER_METAL="0")
    def execute(stage, command):
        def run_once(current_command):
            nonlocal child
            with (run / f"{stage}.log").open("ab") as log:
                child = subprocess.Popen(current_command, cwd=snapshot, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                status(stage, child_pid=child.pid, command=current_command, log=str(run / f"{stage}.log"))
                return child.wait()
        run_with_recovery(stage, command, run_once=run_once, checkpoint=run / "candidate/training-state.pt",
                          should_stop=lambda: stopped,
                          on_refresh=lambda: status("refreshing-training-worker", reason="MPS memory pressure; resume saved next batch"))
    keep_awake = subprocess.Popen(["/usr/bin/caffeinate", "-is", "-w", str(os.getpid())], stdin=subprocess.DEVNULL)
    try:
        preparation = [sys.executable, str(snapshot / "training/prepare_web_data.py"),
            "--base-manifest", str(base), "--evaluation", str(evaluation), "--raw-dir", str(RAW),
            "--output-dir", str(data), "--target-samples", str(args.target_samples), "--shards", str(args.shards)]
        if args.freeze_evaluation:
            preparation += ["--freeze-evaluation"]
        execute("prepare", preparation)
        if not (data / "manifest.json").exists():
            status("stopped", phase="prepare")
            return
        manifest = read(data / "manifest.json")
        assert manifest["statistics"]["samples"] == read(base)["statistics"]["samples"] + args.target_samples
        summary = read(data / "summary.json")
        for split in ("validation", "test"):
            assert sha(data / f"{split}.jsonl") == summary["splits"][split]["sha256"]
            if args.freeze_evaluation:
                assert summary["splits"][split] == read(evaluation / "summary.json")["splits"][split]
        # Same architecture and vocabulary: every starting parameter comes from best_state.
        sys.path[:0] = [str(snapshot / "training"), str(ROOT / "training/.deps")]
        import torch
        from train_smoke import BoundaryChooser
        from train_sharded import expanded_initialization
        initial = torch.load(run / "initialization.pt", map_location="cpu", weights_only=True)
        vocabulary = read(data / "vocabulary.json")
        assert vocabulary == initial["vocabulary"] and len(vocabulary) == 8192
        model = BoundaryChooser(len(vocabulary), 192, 8)
        inherited = expanded_initialization(model, run / "initialization.pt", vocabulary)
        assert all(torch.equal(value, initial["best_state"][name]) for name, value in model.state_dict().items())
        write(run / "initialization-verification.json", {"passed": True, "parameters": sum(p.numel() for p in model.parameters()),
              "source_best_epoch": initial["best_epoch"], "all_parameters_equal_best_state": True, **inherited})
        del model, initial
        candidate = run / "candidate"
        checkpoint = candidate / "training-state.pt"
        if checkpoint.exists() and not args.resume:
            raise FileExistsError("Training checkpoint exists; use --resume")
        command = [sys.executable, str(snapshot / "training/run_sharded.py"), "--manifest", str(data / "manifest.json"),
            "--data-dir", str(data), "--artifact-dir", str(candidate), "--epochs", str(args.epochs),
            "--channels", "192", "--residual-blocks", "8", "--learning-rate", str(args.learning_rate), "--domain-weight-power", "0.65",
            "--selection-macro-weight", str(args.selection_macro_weight), "--gradient-clip", "1.0", "--batch-size", "512", "--max-tokens-per-batch", "8192",
            "--checkpoint-shards", str(args.checkpoint_shards), "--threads", "1", "--interop-threads", "1", "--device", "mps",
            "--mps-memory-fraction", "0.4"]
        command += ["--resume"] if checkpoint.exists() else ["--initialize-from", str(run / "initialization.pt")]
        execute("training", command)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["progress"]["epoch"] <= args.epochs:
            status("stopped", progress=state["progress"])
            return
        execute("export", [sys.executable, str(snapshot / "training/export_browser_model.py"),
            "--artifact-dir", str(candidate), "--output", str(candidate / "boundary-model-data.js")])
        metrics = read(candidate / "smoke-metrics.json")
        initial_validation = metrics["initialization"]["validation_before_training"]
        write(run / "result.json", {"completed_at": datetime.now(timezone.utc).isoformat(),
              "data_sizes": metrics["data_sizes"], "best_epoch": metrics["best_epoch"],
              "initial_validation_with_fixed_padding": initial_validation,
              "selected_validation": metrics["validation"], "selected_test": metrics["test"],
              "validation_accuracy_gain_pp": 100 * (metrics["validation"]["accuracy"] - initial_validation["accuracy"]),
              "history": metrics["history"], "checkpoint_selection": metrics["checkpoint_selection"],
              "backend_changed": sha(ROOT / "src/boundary-model-data.js") != plan["backend_bundle_sha256_at_start"],
              "candidate_bundle": str(candidate / "boundary-model-data.js"),
              "candidate_bundle_sha256": sha(candidate / "boundary-model-data.js")})
        status("complete", selected_epoch=metrics["best_epoch"], validation_accuracy=metrics["validation"]["accuracy"],
               test_accuracy=metrics["test"]["accuracy"])
    except InterruptedError as error:
        status("stopped", error=str(error))
    except BaseException as error:
        status("failed", error=str(error))
        raise
    finally:
        keep_awake.terminate()
        keep_awake.wait()


if __name__ == "__main__":
    main()
