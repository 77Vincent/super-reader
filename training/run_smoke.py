#!/usr/bin/env python3
"""Install the pinned local training dependencies, then run CPU training."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEPENDENCY_DIR = SCRIPT_DIR / ".deps"
REQUIREMENTS = ("tinygrad==0.9.2", "torch==2.8.0")


def ensure_dependencies() -> None:
    sys.path.insert(0, str(DEPENDENCY_DIR))
    try:
        for name in ("tinygrad", "torch"):
            importlib.import_module(name)
        return
    except (ImportError, OSError):
        pass

    DEPENDENCY_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--upgrade",
            "--target",
            str(DEPENDENCY_DIR),
            *REQUIREMENTS,
        ],
        check=True,
    )
    importlib.invalidate_caches()


def main() -> None:
    os.environ["DEBUG"] = "0"
    ensure_dependencies()
    from train_smoke import main as train

    train()


if __name__ == "__main__":
    main()
