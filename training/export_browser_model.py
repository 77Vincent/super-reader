#!/usr/bin/env python3
"""Export the trained tinygrad checkpoint as a dependency-free browser script."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEPENDENCY_DIR = SCRIPT_DIR / ".deps"
CHECKPOINT_PATH = SCRIPT_DIR / "artifacts" / "boundary-smoke.safetensors"
VOCABULARY_PATH = SCRIPT_DIR / "artifacts" / "boundary-smoke-vocabulary.json"
METRICS_PATH = SCRIPT_DIR / "artifacts" / "smoke-metrics.json"
OUTPUT_PATH = PROJECT_DIR / "src" / "boundary-model-data.js"


def main() -> None:
    os.environ["DEBUG"] = "0"
    sys.path.insert(0, str(DEPENDENCY_DIR))
    from tinygrad.nn.state import safe_load

    vocabulary = json.loads(VOCABULARY_PATH.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    state = safe_load(str(CHECKPOINT_PATH))
    tensor_names = [
        "embedding.weight",
        *(
            name
            for block in range(3)
            for name in (
                f"blocks.{block}.normalization.weight",
                f"blocks.{block}.normalization.bias",
                f"blocks.{block}.first.weight",
                f"blocks.{block}.first.bias",
                f"blocks.{block}.second.weight",
                f"blocks.{block}.second.bias",
                f"blocks.{block}.residual_scale",
            )
        ),
        "boundary_hidden.weight",
        "boundary_hidden.bias",
        "boundary_output.weight",
        "boundary_output.bias",
    ]

    chunks: list[bytes] = []
    tensors: dict[str, dict[str, object]] = {}
    float_offset = 0
    for name in tensor_names:
        array = np.ascontiguousarray(state[name].numpy(), dtype="<f4")
        flat = array.reshape(-1)
        chunks.append(flat.tobytes())
        tensors[name] = {
            "offset": float_offset,
            "length": int(flat.size),
            "shape": list(array.shape),
        }
        float_offset += int(flat.size)

    checkpoint_bytes = CHECKPOINT_PATH.read_bytes()
    payload = {
        "format": "super-reader-f32-v1",
        "checkpointSha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "bestEpoch": metrics["best_epoch"],
        "testAccuracy": metrics["test"]["accuracy"],
        "tokenization": metrics.get("tokenization", "unspecified"),
        "candidatePositions": metrics.get("candidate_positions", "unspecified"),
        "vocabulary": vocabulary,
        "tensors": tensors,
        "weightsBase64": base64.b64encode(b"".join(chunks)).decode("ascii"),
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    output = (
        "(function exposeBoundaryModelData(root, data) {\n"
        "  if (typeof module === \"object\" && module.exports) module.exports = data;\n"
        "  root.SuperReaderBoundaryModelData = data;\n"
        f"}})(globalThis,{serialized});\n"
    )
    OUTPUT_PATH.write_text(output, encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "checkpoint_sha256": payload["checkpointSha256"],
        "tensor_count": len(tensors),
        "float_count": float_offset,
        "output_bytes": OUTPUT_PATH.stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
