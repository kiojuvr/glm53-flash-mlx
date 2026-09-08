#!/usr/bin/env python3
"""Qualify the complete union-tiled attention island at Q256 geometry."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NATIVE_PACKAGE = ROOT / "native_execution"
if str(NATIVE_PACKAGE) not in sys.path:
    sys.path.insert(0, str(NATIVE_PACKAGE))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from glm53_native_execution import NativeProjectedQKUnionTileLoopPlan
import probe_q256_projected_qk_union_tile_loop as q256


DEFAULT_OUTPUT = ROOT / "bench-results" / (
    "m3ultra512-q256-union-tiled-full-attention-20260909.json"
)
VALUE_DIM = 128
MAX_SCRATCH_BYTES = 10 << 30
MIN_SPEEDUP = 1.20


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _bits(value: mx.array) -> np.ndarray:
    return np.asarray(value.view(mx.uint16), dtype=np.uint16)


def _reference(inputs, value_weight):
    key = (inputs["latent"] @ inputs["key_weight"])[None]
    value = (inputs["latent"] @ value_weight.swapaxes(-1, -2))[None]
    safe = mx.where(
        inputs["selected_valid"], inputs["selected_indices"], q256.PHYSICAL_K
    )
    mask = mx.zeros(
        (q256.QUERY_ROWS, q256.PHYSICAL_K + 1), dtype=mx.bool_
    )
    mask = mx.put_along_axis(mask, safe, mx.array(True), axis=-1)
    mask = mask[:, : q256.PHYSICAL_K].reshape(
        1, 1, q256.QUERY_ROWS, q256.PHYSICAL_K
    )
    output = mx.fast.scaled_dot_product_attention(
        inputs["query"], key, value, scale=q256.ATTENTION_SCALE, mask=mask
    )
    mx.eval(value, output)
    return {"value": value, "output": output}


def _native(plan, inputs, value_weight):
    output = plan.execute_attention(
        inputs["selected_indices"], inputs["selected_valid"], inputs["latent"],
        inputs["key_weight"], value_weight, inputs["query"],
        q256.ATTENTION_SCALE,
    )
    mx.eval(output)
    mx.synchronize()
    return output.transpose(2, 1, 0, 3)


def _timed(operation):
    for _ in range(2):
        mx.eval(operation())
        mx.synchronize()
    samples = []
    for _ in range(3):
        started = time.perf_counter_ns()
        mx.eval(operation())
        mx.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return {"median_wall_ms": statistics.median(samples), "samples_ms": samples}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    print(json.dumps({"phase": "build_q256_full_attention_fixture"}), flush=True)
    inputs, _, _ = q256._fixture()
    value_weight = mx.cos(
        mx.arange(q256.HEADS * VALUE_DIM * q256.LATENT_DIM, dtype=mx.float32).reshape(
            q256.HEADS, VALUE_DIM, q256.LATENT_DIM
        )
        * 0.000019
    ).astype(mx.bfloat16)
    value_weight = mx.contiguous(value_weight)
    mx.eval(value_weight)
    print(json.dumps({"phase": "direct_q256_full_attention"}), flush=True)
    reference = _reference(inputs, value_weight)

    plan = NativeProjectedQKUnionTileLoopPlan(
        q256.PHYSICAL_K, q256.QUERY_ROWS, q256.TILE_ROWS, True, True
    )
    identities = list(plan.buffer_identities)
    print(json.dumps({"phase": "native_q256_full_attention"}), flush=True)
    candidate = _native(plan, inputs, value_weight)
    sample_rows = (0, q256.QUERY_ROWS - 1)
    sample_heads = (0, q256.HEADS - 1)
    sample_slots = (0, q256.VALID_WIDTH - 1)
    selected_value_samples_exact = True
    for row in sample_rows:
        for head in sample_heads:
            for slot in sample_slots:
                physical = int(
                    np.asarray(inputs["selected_indices"][row, slot]).item()
                )
                selected_value_samples_exact &= np.array_equal(
                    _bits(plan.debug_selected_values[row, head, slot]),
                    _bits(reference["value"][0, head, physical]),
                )
    anchors = {
        "selected_value_samples": bool(selected_value_samples_exact),
        "q256_attention_output": np.array_equal(
            _bits(candidate), _bits(reference["output"])
        ),
    }

    print(json.dumps({"phase": "timing"}), flush=True)
    direct_timing = _timed(lambda: _reference(inputs, value_weight)["output"])
    native_timing = _timed(lambda: _native(plan, inputs, value_weight))
    speedup = direct_timing["median_wall_ms"] / native_timing["median_wall_ms"]
    checks = {
        "q256_selected_value_samples_and_output_byte_exact": all(anchors.values()),
        "q256_full_attention_fixed_native_scope": (
            identities == list(plan.buffer_identities)
            and plan.dynamic_allocation_count == 0
            and plan.graph_node_count == 0
            and plan.shape_discovery_count == 0
            and plan.host_synchronization_count == 0
            and plan.returned_intermediate_tensor_bytes == 0
        ),
        "scratch_at_most_10gib": plan.scratch_bytes <= MAX_SCRATCH_BYTES,
        "q256_full_attention_speedup_at_least_1_20x": speedup >= MIN_SPEEDUP,
    }
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-q256-union-tiled-full-attention-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "advance_q256_full_attention_to_long_context_composition"
            if accepted
            else "stop_or_redesign_q256_union_tiled_value_pass"
        ),
        "geometry": {
            "physical_k": q256.PHYSICAL_K,
            "query_rows": q256.QUERY_ROWS,
            "heads": q256.HEADS,
            "selected_width": q256.SELECTED_WIDTH,
            "tile_rows": q256.TILE_ROWS,
            "tile_count": plan.tile_count,
        },
        "anchors": anchors,
        "direct_timing": direct_timing,
        "native_timing": native_timing,
        "speedup": speedup,
        "scratch_bytes": plan.scratch_bytes,
        "selected_value_edge_bytes": plan.debug_selected_values.nbytes,
        "checks": checks,
        "probe_only": True,
        "runtime_changes": False,
    }
    _atomic_write(args.output, artifact)
    print(json.dumps({
        "output": str(args.output),
        "complete": True,
        "accepted": accepted,
        "decision": artifact["decision"],
        "failed_gates": [name for name, passed in checks.items() if not passed],
    }))
    del plan, inputs, value_weight, reference, candidate
    mx.clear_cache()
    gc.collect()
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
