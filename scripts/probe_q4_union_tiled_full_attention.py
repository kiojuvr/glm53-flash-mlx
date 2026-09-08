#!/usr/bin/env python3
"""Qualify projected-QK/softmax/tiled-V/physical-BK16 AV as one island."""

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
import probe_device_count_projected_qk_union_tile_loop as q4


DEFAULT_OUTPUT = ROOT / "bench-results" / (
    "m3ultra512-q4-union-tiled-full-attention-20260909.json"
)
VALUE_DIM = 128
MAX_SCRATCH_BYTES = 600 << 20


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _bits(value: mx.array) -> np.ndarray:
    return np.asarray(value.view(mx.uint16), dtype=np.uint16)


def _reference(inputs: dict[str, mx.array], value_weight: mx.array):
    key = (inputs["latent"] @ inputs["key_weight"])[None]
    value = (inputs["latent"] @ value_weight.swapaxes(-1, -2))[None]
    valid = inputs["selected_valid"][: q4.ATTENTION_QUERY_ROWS]
    indices = inputs["selected_indices"][: q4.ATTENTION_QUERY_ROWS]
    safe = mx.where(valid, indices, q4.PHYSICAL_K)
    mask = mx.zeros(
        (q4.ATTENTION_QUERY_ROWS, q4.PHYSICAL_K + 1), dtype=mx.bool_
    )
    mask = mx.put_along_axis(mask, safe, mx.array(True), axis=-1)
    mask = mask[:, : q4.PHYSICAL_K].reshape(
        1, 1, q4.ATTENTION_QUERY_ROWS, q4.PHYSICAL_K
    )
    output = mx.fast.scaled_dot_product_attention(
        inputs["query"], key, value, scale=q4.ATTENTION_SCALE, mask=mask
    )
    selected_latent = mx.stack([
        mx.take(
            inputs["latent"],
            mx.where(valid[row], indices[row], mx.zeros_like(indices[row])),
            axis=0,
        )
        for row in range(q4.ATTENTION_QUERY_ROWS)
    ])
    selected_values = (
        selected_latent[:, None] @ value_weight.swapaxes(-1, -2)[None]
    )
    mx.eval(key, value, output, selected_values)
    return {"output": output, "selected_values": selected_values}


def _native(plan, inputs, value_weight):
    output = plan.execute_attention(
        inputs["selected_indices"], inputs["selected_valid"], inputs["latent"],
        inputs["key_weight"], value_weight, inputs["query"],
        q4.ATTENTION_SCALE,
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

    print(json.dumps({"phase": "build_q4_full_attention_fixture"}), flush=True)
    inputs, _, _ = q4._fixture()
    value_weight = mx.cos(
        mx.arange(q4.HEADS * VALUE_DIM * q4.LATENT_DIM, dtype=mx.float32).reshape(
            q4.HEADS, VALUE_DIM, q4.LATENT_DIM
        )
        * 0.000019
    ).astype(mx.bfloat16)
    value_weight = mx.contiguous(value_weight)
    mx.eval(value_weight)
    print(json.dumps({"phase": "direct_full_attention"}), flush=True)
    reference = _reference(inputs, value_weight)

    plan = NativeProjectedQKUnionTileLoopPlan(
        q4.PHYSICAL_K, q4.ATTENTION_QUERY_ROWS, q4.TILE_ROWS, True, True
    )
    identities = list(plan.buffer_identities)
    print(json.dumps({"phase": "native_full_attention"}), flush=True)
    candidate = _native(plan, inputs, value_weight)
    anchors = {
        "selected_value_projection": np.array_equal(
            _bits(mx.where(
                inputs["selected_valid"][: q4.ATTENTION_QUERY_ROWS, None, :, None],
                plan.debug_selected_values,
                mx.zeros_like(plan.debug_selected_values),
            )),
            _bits(mx.where(
                inputs["selected_valid"][: q4.ATTENTION_QUERY_ROWS, None, :, None],
                reference["selected_values"],
                mx.zeros_like(reference["selected_values"]),
            )),
        ),
        "attention_output": np.array_equal(
            _bits(candidate), _bits(reference["output"])
        ),
    }

    print(json.dumps({"phase": "timing"}), flush=True)
    direct_timing = _timed(lambda: _reference(inputs, value_weight)["output"])
    native_timing = _timed(lambda: _native(plan, inputs, value_weight))
    speedup = direct_timing["median_wall_ms"] / native_timing["median_wall_ms"]
    checks = {
        "selected_value_and_full_attention_byte_exact": all(anchors.values()),
        "qk_softmax_value_av_single_fixed_scope": (
            identities == list(plan.buffer_identities)
            and plan.dynamic_allocation_count == 0
            and plan.graph_node_count == 0
            and plan.shape_discovery_count == 0
            and plan.host_synchronization_count == 0
            and plan.returned_intermediate_tensor_bytes == 0
        ),
        "scratch_at_most_600mib": plan.scratch_bytes <= MAX_SCRATCH_BYTES,
        "q4_boundary_tax_measured_for_q256_gate": speedup > 0.0,
    }
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-q4-union-tiled-full-attention-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "advance_exact_union_tiled_full_attention_to_q256_reuse_gate"
            if accepted
            else "stop_or_relocalize_union_tiled_value_attention"
        ),
        "geometry": {
            "physical_k": q4.PHYSICAL_K,
            "query_rows": q4.ATTENTION_QUERY_ROWS,
            "heads": q4.HEADS,
            "selected_width": q4.SELECTED_WIDTH,
            "tile_rows": q4.TILE_ROWS,
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
