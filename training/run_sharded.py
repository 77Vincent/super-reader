#!/usr/bin/env python3
"""Install pinned dependencies, then run memory-bounded sharded training."""

from __future__ import annotations

import os

from run_smoke import ensure_dependencies


def main() -> None:
    # tinygrad 0.9.2 expects DEBUG to be an integer. Some IDE shells export a
    # textual DEBUG value, so keep the training environment deterministic.
    os.environ["DEBUG"] = "0"
    ensure_dependencies()
    from train_sharded import main as train

    train()


if __name__ == "__main__":
    main()
