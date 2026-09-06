#!/usr/bin/env python3
"""Install the Parquet reader locally, then prepare synthetic training shards."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEPENDENCY_DIR = SCRIPT_DIR / ".deps"


def ensure_dependencies() -> None:
    sys.path.insert(0, str(DEPENDENCY_DIR))
    try:
        importlib.import_module("pyarrow.parquet")
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
            "pyarrow==21.0.0",
        ],
        check=True,
    )
    importlib.invalidate_caches()


def main() -> None:
    ensure_dependencies()
    from prepare_synthetic_data import main as prepare

    prepare()


if __name__ == "__main__":
    main()
