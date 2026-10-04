#!/usr/bin/env python3
"""Start an isolated random-initialized run over an existing prepared corpus."""
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

from sample_subset import write_json

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ("train_sharded.py", "train_smoke.py", "sample_subset.py", "text_policy.py", "unicode-symbols.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(run, args):
    if (run / "plan.json").exists():
        raise FileExistsError("Run already prepared; use --resume to continue this run")
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    current_policy = json.loads((ROOT / "training/text-policy.json").read_text())
    # Existing surface selection can be reused, but input/label semantics must
    # match. Keep its actual policy in the snapshot; never relabel v7 data as v8.
    for key in current_policy:
        if key not in ("standard", "sample_filter") and manifest.get(key) != current_policy[key]:
            raise ValueError(f"Prepared corpus changes input/label semantics: {key}")
    policy = {key: manifest[key] for key in current_policy}
    snapshot = run / "source"
    snapshot.mkdir()
    for name in SOURCES:
        shutil.copy2(ROOT / "training" / name, snapshot / name)
    write_json(snapshot / "text-policy.json", policy)
    command = [sys.executable, str(snapshot / "train_sharded.py"),
               "--manifest", str(manifest_path), "--data-dir", str(manifest_path.parent),
               "--artifact-dir", str(run / "candidate"),
               "--channels", "192", "--residual-blocks", "8",
               "--epochs", str(args.epochs), "--learning-rate", str(args.learning_rate),
               "--domain-weight-power", "0.65", "--position-weighting", "none",
               "--min-side-characters", "2", "--gradient-clip", "1",
               "--batch-size", "512", "--max-tokens-per-batch", "8192",
               "--checkpoint-shards", "2", "--threads", "1", "--seed", str(args.seed)]
    plan = {"created_at": datetime.now(timezone.utc).isoformat(),
            "initialization": "random; no pretrained weights or optimizer",
            "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
            "prepared_data_standard": policy["standard"],
            "prepared_policy_preserved": True,
            "new_preparation_filters_applied": False,
            "subset": {"minimum_side_characters": 2, "splits": ["train", "validation", "test"]},
            "position_weighting": "none", "domain_weight_power": .65,
            "epochs": args.epochs, "learning_rate": args.learning_rate, "seed": args.seed,
            "source_hashes": {p.name: sha(p) for p in sorted(snapshot.iterdir())},
            "command": command, "automatic_promotion": False}
    write_json(run / "plan.json", plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "training/data/processed/padding-fixed-web-350m-v7-20260925/manifest.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=.002)
    parser.add_argument("--seed", type=int, default=2026100405)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Use the frozen plan; all training settings come from plan.json")
    args = parser.parse_args()
    if args.epochs < 1 or not 0 < args.learning_rate < float("inf"):
        parser.error("Epochs and learning rate must be positive and finite")
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads((run / "plan.json").read_text()) if args.resume else prepare(run, args)
        if sha(Path(plan["manifest"])) != plan["manifest_sha256"]:
            raise ValueError("Prepared manifest changed")
        for name, digest in plan["source_hashes"].items():
            if sha(run / "source" / name) != digest:
                raise ValueError(f"Frozen training source changed: {name}")
        if args.prepare_only:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        from run_smoke import ensure_dependencies
        ensure_dependencies()
        child, stopped = None, False

        def stop(signum, _frame):
            nonlocal stopped
            stopped = True
            if child is not None and child.poll() is None:
                child.send_signal(signum)

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop)

        def status(stage, **details):
            value = {"stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(),
                     "coordinator_pid": os.getpid(), **details}
            write_json(run / "status.json", value)
            print(json.dumps(value), flush=True)

        environment = dict(os.environ, DEBUG="0", PYTHONUNBUFFERED="1",
                           PYTHONPATH=str(ROOT / "training/.deps"), SUPER_READER_PROJECT_ROOT=str(ROOT),
                           PYTORCH_ENABLE_MPS_FALLBACK="0", PYTORCH_MPS_FAST_MATH="0", PYTORCH_MPS_PREFER_METAL="0")
        awake = subprocess.Popen(["/usr/bin/caffeinate", "-is", "-w", str(os.getpid())], stdin=subprocess.DEVNULL)
        checkpoint = run / "candidate/training-state.pt"
        try:
            while not stopped:
                command = list(plan["command"])
                if checkpoint.exists():
                    command.append("--resume")
                previous_stamp = checkpoint.stat().st_mtime_ns if checkpoint.exists() else None
                with (run / "training.log").open("ab") as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    status("training-worker", child_pid=child.pid, log=str(run / "training.log"))
                    code = child.wait()
                if stopped:
                    status("stopped", returncode=code)
                    return
                if code == 75:
                    if not checkpoint.exists() or checkpoint.stat().st_mtime_ns == previous_stamp:
                        raise RuntimeError("Worker refresh did not save an advanced checkpoint")
                    status("refreshing-worker")
                    continue
                if code:
                    raise RuntimeError(f"Training worker exited {code}; see training.log")
                if not (run / "candidate/smoke-metrics.json").exists():
                    status("stopped", reason="Worker stopped before final evaluation")
                    return
                status("complete", metrics=str(run / "candidate/smoke-metrics.json"), promoted=False)
                return
        except BaseException as error:
            status("failed", error=str(error))
            raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait()
            awake.terminate()
            awake.wait()


if __name__ == "__main__":
    main()
