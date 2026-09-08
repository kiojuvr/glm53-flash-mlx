#!/usr/bin/env python3
"""Probe one exact BM64 shared-physical DSA value tile."""

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

from glm53_native_execution import NativeSharedPhysicalValueTilePlan


DEFAULT_OUTPUT = ROOT / "bench-results" / (
    "m3ultra512-exact-bm64-shared-physical-value-tile-20260909.json"
)
PHYSICAL_K = 8192
QUERY_ROWS = 64
HEADS = 64
LATENT_DIM = 512
VALUE_DIM = 128
SELECTED_WIDTH = 2051
VALID_WIDTH = 2048
MAX_SCRATCH = 256 << 20


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _bits(value: mx.array) -> np.ndarray:
    return np.asarray(value.view(mx.uint16), dtype=np.uint16)


def _fixture() -> dict[str, mx.array]:
    host_indices = np.full((QUERY_ROWS, SELECTED_WIDTH), -1, dtype=np.int32)
    host_valid = np.zeros((QUERY_ROWS, SELECTED_WIDTH), dtype=np.bool_)
    base = np.arange(VALID_WIDTH, dtype=np.int64)
    for row in range(QUERY_ROWS):
        selected = (base * 4 + row % 4) % PHYSICAL_K
        selected.sort()
        host_indices[row, :VALID_WIDTH] = selected.astype(np.int32)
        host_valid[row, :VALID_WIDTH] = True
    scores = mx.sin(
        mx.arange(
            QUERY_ROWS * HEADS * SELECTED_WIDTH, dtype=mx.float32
        ).reshape(QUERY_ROWS, HEADS, SELECTED_WIDTH)
        * 0.00037
    )
    probabilities = mx.softmax(scores, axis=-1, precise=True).astype(mx.bfloat16)
    latent = mx.cos(
        mx.arange(PHYSICAL_K * LATENT_DIM, dtype=mx.float32).reshape(
            PHYSICAL_K, LATENT_DIM
        )
        * 0.000071
    ).astype(mx.bfloat16)
    value_weight = mx.sin(
        mx.arange(HEADS * VALUE_DIM * LATENT_DIM, dtype=mx.float32).reshape(
            HEADS, VALUE_DIM, LATENT_DIM
        )
        * 0.000019
    ).astype(mx.bfloat16)
    result = {
        "probabilities": mx.contiguous(probabilities),
        "indices": mx.contiguous(mx.array(host_indices)),
        "valid": mx.contiguous(mx.array(host_valid)),
        "latent": mx.contiguous(latent),
        "value_weight": mx.contiguous(value_weight),
    }
    mx.eval(*result.values())
    return result


def _reference(inputs: dict[str, mx.array]) -> dict[str, mx.array]:
    indices = inputs["indices"]
    valid = inputs["valid"]
    probabilities = inputs["probabilities"]
    safe = mx.where(valid, indices, mx.zeros_like(indices))
    physical = mx.zeros((HEADS, QUERY_ROWS, PHYSICAL_K + 1), dtype=mx.bfloat16)
    safe = mx.where(valid, indices, PHYSICAL_K)
    physical = mx.put_along_axis(
        physical,
        mx.broadcast_to(safe[None], (HEADS, QUERY_ROWS, SELECTED_WIDTH)),
        mx.where(
            valid[:, None, :], probabilities, mx.zeros_like(probabilities)
        ).transpose(1, 0, 2),
        axis=-1,
    )[..., :PHYSICAL_K]
    projected = inputs["latent"] @ inputs["value_weight"].swapaxes(-1, -2)
    sanitized = mx.where(valid, indices, mx.zeros_like(indices))
    selected_latent = mx.stack([
        mx.take(inputs["latent"], sanitized[row], axis=0)
        for row in range(QUERY_ROWS)
    ])
    selected_values = (
        selected_latent[:, None]
        @ inputs["value_weight"].swapaxes(-1, -2)[None]
    )
    selected_output = (
        probabilities[:, :, None, :] @ selected_values
    ).transpose(1, 0, 2, 3).reshape(HEADS, QUERY_ROWS, VALUE_DIM)
    physical_output = physical @ projected
    mx.eval(physical, projected, selected_output, physical_output)
    return {
        "physical": physical,
        "projected": projected,
        "selected_values": selected_values,
        "selected_output": selected_output,
        "physical_output": physical_output,
    }


def _native(plan, inputs: dict[str, mx.array]) -> mx.array:
    output = plan.execute(
        inputs["probabilities"], inputs["indices"], inputs["valid"],
        inputs["latent"], inputs["value_weight"],
    )
    mx.eval(output)
    mx.synchronize()
    return output


def _timed(operation, samples: int = 3) -> dict[str, object]:
    for _ in range(2):
        mx.eval(operation())
        mx.synchronize()
    values = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        mx.eval(operation())
        mx.synchronize()
        values.append((time.perf_counter_ns() - started) / 1e6)
    return {"median_wall_ms": statistics.median(values), "samples_ms": values}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    print(json.dumps({"phase": "build_bm64_fixture"}), flush=True)
    inputs = _fixture()
    print(json.dumps({"phase": "direct_reference"}), flush=True)
    reference = _reference(inputs)
    plan = NativeSharedPhysicalValueTilePlan(PHYSICAL_K)
    identities = list(plan.buffer_identities)
    print(json.dumps({"phase": "native_shared_physical_value_tile"}), flush=True)
    candidate = _native(plan, inputs)

    anchors = {
        "physical_probability_scatter": np.array_equal(
            _bits(plan.debug_physical_probabilities), _bits(reference["physical"])
        ),
        "shared_value_projection": np.array_equal(
            _bits(plan.debug_projected_values), _bits(reference["projected"])
        ),
        "bm64_physical_attention_output": np.array_equal(
            _bits(candidate), _bits(reference["physical_output"])
        ),
        "query_local_selected_attention_output": np.array_equal(
            _bits(candidate), _bits(reference["selected_output"])
        ),
    }
    print(json.dumps({"phase": "timing"}), flush=True)
    direct_timing = _timed(lambda: _reference(inputs)["selected_output"])
    native_timing = _timed(lambda: _native(plan, inputs))
    speedup = direct_timing["median_wall_ms"] / native_timing["median_wall_ms"]
    checks = {
        "scatter_projection_bm64_output_byte_exact": (
            anchors["physical_probability_scatter"]
            and anchors["shared_value_projection"]
            and anchors["bm64_physical_attention_output"]
        ),
        "query_local_selected_reduction_is_not_the_q256_oracle": (
            not anchors["query_local_selected_attention_output"]
        ),
        "fixed_owned_native_scope": (
            identities == list(plan.buffer_identities)
            and plan.dynamic_allocation_count == 0
            and plan.graph_node_count == 0
            and plan.shape_discovery_count == 0
            and plan.host_synchronization_count == 0
            and plan.returned_intermediate_tensor_bytes == 0
        ),
        "query_local_selected_value_materialization_zero": (
            plan.materialized_query_local_selected_value_bytes == 0
        ),
        "scratch_at_most_256mib": plan.scratch_bytes <= MAX_SCRATCH,
        "native_boundary_has_positive_measured_speedup": speedup > 1.0,
    }
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-exact-bm64-shared-physical-value-tile-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "advance_shared_physical_value_pass_to_multitile_accumulator"
            if accepted
            else "stop_or_relocalize_exact_bm64_shared_physical_value_tile"
        ),
        "geometry": {
            "physical_k": PHYSICAL_K,
            "query_rows": QUERY_ROWS,
            "heads": HEADS,
            "selected_width": SELECTED_WIDTH,
            "value_dimension": VALUE_DIM,
        },
        "anchors": anchors,
        "direct_timing": direct_timing,
        "native_timing": native_timing,
        "speedup": speedup,
        "scratch_bytes": plan.scratch_bytes,
        "materialized_query_local_selected_value_bytes": (
            plan.materialized_query_local_selected_value_bytes
        ),
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
    del plan, inputs, reference, candidate
    mx.clear_cache()
    gc.collect()
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
