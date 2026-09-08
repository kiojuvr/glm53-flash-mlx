#!/usr/bin/env python3
"""Capture only the model-free compact sparse-prefill QK matmul.

This resolves the remaining exactness blocker in the native selected-K/V
attention region without capturing the checkpoint or a full model execution.
The trace itself must remain outside the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from capture_budget import CaptureBudget, atomic_write, supervise_capture
from capture_sparse_prefill_av_operator import _pipeline_labels, _validate_trace
from capture_steady_packed_decode_critical_path import _trace_identity


DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-sparse-prefill-qk-operator-capture-20260908.json"
)
CONTEXT = 2_048
EXPECTED_QK_PIPELINE = (
    "gemv_bfloat16_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0"
)


def _hash(array, mx) -> str:
    import numpy as np

    return hashlib.sha256(
        np.asarray(array.view(mx.uint16), dtype=np.uint16).tobytes()
    ).hexdigest()


def _child(args: argparse.Namespace) -> int:
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        raise RuntimeError("set MTL_CAPTURE_ENABLED=1 before importing MLX")

    import mlx.core as mx

    import probe_sparse_prefill_reordering_equivalence as equivalence

    fixture = equivalence._fixture(CONTEXT)
    row = equivalence.QUERY_ROWS - 1
    selected_latent = mx.contiguous(
        equivalence._gather_per_query(
            fixture["latent"], fixture["indices"], fixture["valid"]
        )[row]
    )
    selected_key = selected_latent @ fixture["k_weight"]
    scale = mx.array(
        equivalence.LATENT_DIM**-0.5, dtype=mx.bfloat16
    )
    scaled_query = (
        fixture["query"][:, :, row : row + 1, :] * scale
    ).astype(mx.bfloat16)
    selected_key = mx.contiguous(selected_key)
    scaled_query = mx.contiguous(scaled_query)
    mx.eval(selected_key, scaled_query)
    mx.synchronize()

    def operation():
        return scaled_query @ selected_key.swapaxes(-1, -2)

    reference = operation()
    mx.eval(reference)
    reference_hash = _hash(reference, mx)
    for _ in range(2):
        output = operation()
        mx.eval(output)
        mx.synchronize()

    trace = _validate_trace(args.trace)
    started = time.perf_counter()
    mx.metal.start_capture(str(trace))
    try:
        output = operation()
        mx.eval(output)
        mx.synchronize()
    finally:
        mx.metal.stop_capture()
    capture_seconds = time.perf_counter() - started

    identity = _trace_identity(trace)
    identity.update(path=str(trace), stored_in_repository=False)
    labels = _pipeline_labels(trace)
    row = {
        "context_tokens": CONTEXT,
        "heads": equivalence.HEADS,
        "m": 1,
        "n": equivalence.SELECTED_WIDTH,
        "k": equivalence.LATENT_DIM,
        "batch_count": equivalence.HEADS,
        "input_dtype": "bfloat16",
        "output_dtype": str(output.dtype),
        "output_sha256": _hash(output, mx),
        "reference_sha256": reference_hash,
        "output_byte_exact": _hash(output, mx) == reference_hash,
        "trace": identity,
        "capture_process_wall_seconds": capture_seconds,
        "capture_process_wall_is_performance_metric": False,
        "resident_payload_scope": "model-free compact QK operands only",
        "full_model_payload_resident": False,
        "captured_pipeline_resource_labels": labels,
        "pipeline_labels_authoritative": False,
    }
    artifact = {
        "schema": "glm53-sparse-prefill-qk-operator-capture-v1",
        "probe_only": True,
        "case": row,
        "complete": True,
        "accepted": (
            row["output_byte_exact"]
            and EXPECTED_QK_PIPELINE in labels
            and not row["full_model_payload_resident"]
            and not row["trace"]["stored_in_repository"]
        ),
        "decision": (
            "instantiate_captured_compact_qk_topology"
            if EXPECTED_QK_PIPELINE in labels
            else "inspect_qk_capture_in_xcode"
        ),
        "runtime_changes": False,
    }
    atomic_write(args.output, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["accepted"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--negative-output", type=Path)
    parser.add_argument("--capture-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.capture_child:
        return _child(args)
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        raise RuntimeError("set MTL_CAPTURE_ENABLED=1 before starting capture")
    trace = _validate_trace(args.trace)
    negative = args.negative_output or args.output.with_name(
        f"{args.output.stem}-negative.json"
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--trace",
        str(trace),
        "--output",
        str(args.output),
        "--capture-child",
    ]
    return supervise_capture(
        command,
        trace_path=trace,
        evidence_path=negative,
        budget=CaptureBudget(
            max_elapsed_s=180.0,
            max_trace_bytes=2 << 30,
            min_free_bytes=16 << 30,
        ),
        metadata={
            "capture_kind": "model-free-sparse-prefill-qk-operator",
            "trace_path": str(trace),
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
