#!/usr/bin/env python3
"""Continue the completed 12-layer best checkpoint with 16 CNN layers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
PREVIOUS = ROOT / "training/artifacts/unicode-context-192ch-12conv-20260913/candidate"
DEFAULT_RUN = ROOT / "training/artifacts/unicode-context-192ch-16conv-20260914"
MANIFEST = ROOT / "training/data/processed/unicode-context-192ch-12conv-20260913-combined/manifest.json"
EVALUATION = ROOT / "training/data/processed/unicode-context-192ch-12conv-20260913-eval"
SOURCES = ["training/" + name for name in ("run_sharded.py", "run_smoke.py", "train_sharded.py",
    "train_smoke.py", "text_policy.py", "text-policy.json", "unicode-symbols.json", "export_browser_model.py", "preflight_depth_expansion.py")]
SOURCES += ["src/backend/inference.js"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    metadata_path = run / "run.json"
    snapshot = run / "source"
    if not metadata_path.exists():
        previous = read(PREVIOUS / "smoke-metrics.json")
        assert previous["best_epoch"] == 2
        assert previous["architecture"]["channels"] == 192
        assert previous["architecture"]["residual_blocks"] == 6
        shutil.copy2(PREVIOUS / "training-state.pt", run / "initialization.pt")
        shutil.copy2(PREVIOUS / "smoke-metrics.json", run / "source-metrics.json")
        for name in SOURCES:
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        for name in ("data", ".deps"):
            (snapshot / "training" / name).symlink_to(ROOT / "training" / name, target_is_directory=True)
        write(metadata_path, {"created_at": datetime.now(timezone.utc).isoformat(),
              "epochs": args.epochs, "channels": 192, "residual_blocks": 8,
              "source_best_epoch": 2, "initialization": str(PREVIOUS / "training-state.pt"),
              "initialization_sha256": sha(run / "initialization.pt"), "manifest": str(MANIFEST),
              "manifest_sha256": sha(MANIFEST), "evaluation": str(EVALUATION),
              "evaluation_summary_sha256": sha(EVALUATION / "summary.json"),
              "source_hashes": {name: sha(snapshot / name) for name in SOURCES},
              "data_policy": read(ROOT / "training/text-policy.json"),
              "downloaded_corpus_included": False})
    metadata = read(metadata_path)
    if args.epochs != metadata["epochs"]:
        raise ValueError("Use the recorded epoch count when resuming")
    assert sha(run / "initialization.pt") == metadata["initialization_sha256"]
    assert sha(MANIFEST) == metadata["manifest_sha256"]
    assert sha(EVALUATION / "summary.json") == metadata["evaluation_summary_sha256"]
    for name, expected in metadata["source_hashes"].items():
        assert sha(snapshot / name) == expected, name
    candidate = run / "candidate"
    candidate.mkdir(exist_ok=True)
    state = candidate / "training-state.pt"
    if state.exists() and not args.resume:
        raise FileExistsError("Checkpoint exists; use --resume")
    environment = dict(os.environ, DEBUG="0", PYTHONUNBUFFERED="1",
                       PYTHONPATH=str(ROOT / "training/.deps"))
    child = None
    stopped = False
    def stop(signum, _frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)
    for name in (signal.SIGINT, signal.SIGTERM):
        signal.signal(name, stop)
    def status(stage, **details):
        value = {"stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(),
                 "epochs": args.epochs, "convolution_layers": 16, **details}
        write(run / "status.json", value)
        print(json.dumps(value), flush=True)
    def execute(stage, command):
        nonlocal child
        if stopped:
            raise InterruptedError("Stopped between stages")
        with (run / (stage + ".log")).open("ab") as log:
            child = subprocess.Popen(command, cwd=snapshot, env=environment, stdout=log, stderr=subprocess.STDOUT)
            status(stage, pid=child.pid, command=command)
            code = child.wait()
        if stopped:
            raise InterruptedError("Stop requested; inspect checkpoint before resuming")
        if code:
            raise RuntimeError(f"{stage} exited {code}; see {stage}.log")
    try:
        verification = run / "preflight/verification.json"
        if not verification.exists():
            execute("preflight", [sys.executable, str(snapshot / "training/preflight_depth_expansion.py"),
                    "--checkpoint", str(run / "initialization.pt"), "--output-dir", str(run / "preflight")])
        assert read(verification)["passed"]
        command = [sys.executable, str(snapshot / "training/run_sharded.py"), "--manifest", str(MANIFEST),
                   "--data-dir", str(EVALUATION), "--artifact-dir", str(candidate), "--epochs", str(args.epochs),
                   "--channels", "192", "--residual-blocks", "8", "--learning-rate", "0.0003",
                   "--domain-weight-power", "0.65", "--selection-macro-weight", "0.5", "--gradient-clip", "1.0",
                   "--batch-size", "512", "--max-tokens-per-batch", "8192", "--checkpoint-shards", "4",
                   "--threads", "1", "--interop-threads", "1", "--device", "mps"]
        command += ["--resume"] if state.exists() else ["--initialize-from", str(run / "initialization.pt")]
        execute("training", command)
        # A normal trainer return can also represent a graceful interruption.
        sys.path.insert(0, str(ROOT / "training/.deps"))
        import torch
        checkpoint = torch.load(state, map_location="cpu", weights_only=True)
        if checkpoint["progress"]["epoch"] <= args.epochs:
            status("stopped", progress=checkpoint["progress"])
            return
        metrics = read(candidate / "smoke-metrics.json")
        assert abs(checkpoint["initialization"]["validation_before_training"]["accuracy"]
                   - read(run / "source-metrics.json")["validation"]["accuracy"]) < 1e-12
        execute("export", [sys.executable, str(snapshot / "training/export_browser_model.py"),
                "--artifact-dir", str(candidate), "--output", str(candidate / "boundary-model-data.js")])
        status("complete", selected_epoch=metrics["best_epoch"], validation_accuracy=metrics["validation"]["accuracy"],
               test_accuracy=metrics["test"]["accuracy"], model=str(candidate / "boundary-model-data.js"))
    except InterruptedError as error:
        status("stopped", error=str(error))
    except BaseException as error:
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
