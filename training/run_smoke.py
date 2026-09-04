#!/usr/bin/env python3
"""Install the tiny smoke dependency locally, then run training."""

from __future__ import annotations

import os
import importlib
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEPENDENCY_DIR = SCRIPT_DIR / ".deps"
REQUIREMENT = "tinygrad==0.9.2"


def ensure_dependency() -> None:
    sys.path.insert(0, str(DEPENDENCY_DIR))
    try:
        __import__("tinygrad")
        return
    except ImportError:
        pass

    DEPENDENCY_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--target",
            str(DEPENDENCY_DIR),
            REQUIREMENT,
        ],
        check=True,
    )
    importlib.invalidate_caches()


def main() -> None:
    os.environ["DEBUG"] = "0"
    ensure_dependency()
    from train_smoke import main as train

    train()


if __name__ == "__main__":
    main()
