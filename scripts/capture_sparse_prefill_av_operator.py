#!/usr/bin/env python3
"""Capture the bounded AV matmul that blocks exact sparse native prefill.

The capture is deliberately model-free.  Its largest resident inputs are the
32K attention probabilities and projected values used by the existing exact
localization fixture, so the trace cannot accidentally archive the 320 GB
checkpoint.  Direct and compact arms run in separate processes and traces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from capture_budget import CaptureBudget, atomic_write, supervise_capture
from capture_steady_packed_decode_critical_path import _trace_identity


ARMS = ("direct", "compact")
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-sparse-prefill-av-operator-capture-20260908.json"
)
PIPELINE_PATTERN = re.compile(
    r"(?:steel_gemm|gemm_|gemv|matvec)[A-Za-z0-9_]+"
)


def _validate_trace(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.suffix != ".gputrace":
        raise ValueError("--trace must end in .gputrace")
    if path.is_relative_to(ROOT):
        raise ValueError("AV .gputrace must live outside the repository")
    if path.exists():
        raise FileExistsError(f"trace path already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _trace_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return [item for item in path.rglob("*") if item.is_file()]


def _pipeline_labels(path: Path) -> list[str]:
    """Extract captured pipeline-resource names, excluding embedded libraries."""

    labels: set[str] = set()
    for item in _trace_files(path):
        if not item.read_bytes()[:8] == b"bplist00":
            continue
        try:
            payload = plistlib.loads(item.read_bytes())
        except (OSError, plistlib.InvalidFileException):
            continue

        def visit(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    visit(key)
                    visit(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child)
            elif isinstance(value, str):
                match = PIPELINE_PATTERN.search(value)
                if match:
                    labels.add(match.group(0))

        visit(payload)
    return sorted(labels)


def _hash(array, mx) -> str:
    import numpy as np

    return hashlib.sha256(
        np.asarray(array.view(mx.uint16), dtype=np.uint16).tobytes()
    ).hexdigest()


def _child(args: argparse.Namespace) -> int:
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        raise RuntimeError("set MTL_CAPTURE_ENABLED=1 before importing MLX")

    import mlx.core as mx

    import localize_sparse_prefill_attention_reduction as localization
    import probe_sparse_prefill_reordering_equivalence as equivalence

    fixture = equivalence._fixture(localization.CONTEXT)
    latent = fixture["latent"]
    key = latent @ fixture["k_weight"]
    value = latent @ fixture["v_weight"].swapaxes(-1, -2)
    safe = mx.where(fixture["valid"], fixture["indices"], localization.CONTEXT)
    mask = mx.zeros(
        (1, 1, equivalence.QUERY_ROWS, localization.CONTEXT + 1),
        dtype=mx.bool_,
    )
    mask = mx.put_along_axis(mask, safe[:, None], mx.array(True), axis=-1)[
        ..., : localization.CONTEXT
    ]
    scale = mx.array(equivalence.LATENT_DIM**-0.5, dtype=mx.bfloat16)
    scaled_query = (fixture["query"] * scale).astype(mx.bfloat16)
    full_scores = scaled_query @ key.swapaxes(-1, -2)
    full_scores = mx.where(mask, full_scores, mx.finfo(mx.bfloat16).min)
    full_probabilities = mx.softmax(full_scores, axis=-1, precise=True)

    selected_probabilities = localization._gather_query_axis(
        full_probabilities, safe
    )
    selected_values = equivalence._gather_per_query(
        value, fixture["indices"], fixture["valid"]
    )
    mx.eval(full_probabilities, value, selected_probabilities, selected_values)
    mx.synchronize()

    def operation():
        if args.arm == "direct":
            return full_probabilities @ value
        return selected_probabilities @ selected_values

    direct = full_probabilities @ value
    compact = (selected_probabilities @ selected_values).transpose(2, 1, 0, 3)
    mx.eval(direct, compact)
    direct_hash = _hash(direct, mx)
    compact_hash = _hash(compact, mx)
    differing = int(mx.sum(direct != compact).item())

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
    row = {
        "arm": args.arm,
        "context_tokens": localization.CONTEXT,
        "query_rows": equivalence.QUERY_ROWS,
        "heads": equivalence.HEADS,
        "value_dimension": equivalence.VALUE_DIM,
        "reduction_k": (
            localization.CONTEXT
            if args.arm == "direct"
            else equivalence.SELECTED_WIDTH
        ),
        "input_dtype": "bfloat16",
        "accumulator_dtype_expected": "float32",
        "output_dtype": str(output.dtype),
        "direct_output_sha256": direct_hash,
        "compact_output_sha256": compact_hash,
        "direct_compact_different_elements": differing,
        "expected_exactness_observed": (
            differing == 1 and direct_hash != compact_hash
        ),
        "trace": identity,
        "capture_process_wall_seconds": capture_seconds,
        "capture_process_wall_is_performance_metric": False,
        "resident_payload_scope": "model-free 32K AV operands only",
        "full_model_payload_resident": False,
        "pipeline_labels_best_effort": _pipeline_labels(trace),
        "pipeline_labels_authoritative": False,
    }
    artifact = {
        "schema": "glm53-sparse-prefill-av-operator-capture-v1",
        "probe_only": True,
        "cases": {},
        "complete": False,
        "accepted": False,
        "decision": "capture_dynamic_av_gemm_geometry",
        "runtime_changes": False,
    }
    if args.output.exists():
        artifact = json.loads(args.output.read_text())
    artifact["cases"][args.arm] = row
    artifact["complete"] = set(artifact["cases"]) == set(ARMS)
    artifact["accepted"] = artifact["complete"] and all(
        case["expected_exactness_observed"]
        and not case["full_model_payload_resident"]
        and not case["trace"]["stored_in_repository"]
        for case in artifact["cases"].values()
    )
    if artifact["complete"]:
        artifact["decision"] = "derive_exact_virtual_full_kv_av_topology"
    atomic_write(args.output, artifact)
    print(json.dumps(row, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
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
        f"{args.output.stem}-negative-{args.arm}.json"
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--arm",
        args.arm,
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
            max_elapsed_s=300.0,
            max_trace_bytes=4 << 30,
            min_free_bytes=16 << 30,
        ),
        metadata={
            "capture_kind": "model-free-sparse-prefill-av-operator",
            "arm": args.arm,
            "trace_path": str(trace),
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
