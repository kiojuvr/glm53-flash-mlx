#!/usr/bin/env python3
"""Qualify the projected-QK tile loop at full Q256 attention geometry."""

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

from glm53_native_execution import NativeProjectedQKUnionTileLoopPlan


DEFAULT_OUTPUT = ROOT / "bench-results" / (
    "m3ultra512-q256-projected-qk-union-tile-loop-20260909.json"
)
PHYSICAL_K = 8192
TILE_ROWS = 4096
QUERY_ROWS = 256
HEADS = 64
LATENT_DIM = 512
SELECTED_WIDTH = 2051
VALID_WIDTH = 2048
ATTENTION_SCALE = LATENT_DIM**-0.5
MAX_SCRATCH_BYTES = 384 << 20
MIN_SPEEDUP = 1.20


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _bits(value: mx.array) -> np.ndarray:
    return np.asarray(value.view(mx.uint16), dtype=np.uint16)


def _fixture() -> tuple[dict[str, mx.array], np.ndarray, np.ndarray]:
    host_indices = np.full((QUERY_ROWS, SELECTED_WIDTH), -1, dtype=np.int32)
    host_valid = np.zeros((QUERY_ROWS, SELECTED_WIDTH), dtype=np.bool_)
    base = np.arange(VALID_WIDTH, dtype=np.int64)
    for row in range(QUERY_ROWS):
        selected = (base * 4 + row % 4) % PHYSICAL_K
        selected.sort()
        host_indices[row, :VALID_WIDTH] = selected.astype(np.int32)
        host_valid[row, :VALID_WIDTH] = True
    selected_indices = mx.array(host_indices)
    selected_valid = mx.array(host_valid)
    latent = mx.cos(
        mx.arange(PHYSICAL_K * LATENT_DIM, dtype=mx.float32).reshape(
            PHYSICAL_K, LATENT_DIM
        )
        * 0.000071
    ).astype(mx.bfloat16)
    key_weight = mx.sin(
        mx.arange(HEADS * LATENT_DIM * LATENT_DIM, dtype=mx.float32).reshape(
            HEADS, LATENT_DIM, LATENT_DIM
        )
        * 0.000013
    ).astype(mx.bfloat16)
    query = mx.sin(
        mx.arange(HEADS * QUERY_ROWS * LATENT_DIM, dtype=mx.float32).reshape(
            1, HEADS, QUERY_ROWS, LATENT_DIM
        )
        * 0.00031
    ).astype(mx.bfloat16)
    arrays = (selected_indices, selected_valid, latent, key_weight, query)
    mx.eval(*arrays)
    mx.clear_cache()
    return (
        {
            "selected_indices": mx.contiguous(selected_indices),
            "selected_valid": mx.contiguous(selected_valid),
            "latent": mx.contiguous(latent),
            "key_weight": mx.contiguous(key_weight),
            "query": mx.contiguous(query),
        },
        host_indices,
        host_valid,
    )


def _direct(inputs: dict[str, mx.array]) -> dict[str, mx.array]:
    # The oracle deliberately exposes the 32-GiB per-query selected-K
    # materialization that the native tile loop forbids.  It is bounded to an
    # 8K model-free fixture and never enters a production path.
    projected = inputs["latent"] @ inputs["key_weight"]
    sanitized = mx.where(
        inputs["selected_valid"],
        inputs["selected_indices"],
        mx.zeros_like(inputs["selected_indices"]),
    )
    selected_key = mx.stack(
        [mx.take(projected, sanitized[row], axis=1) for row in range(QUERY_ROWS)]
    )
    scale = mx.array(ATTENTION_SCALE, dtype=mx.bfloat16)
    scaled_query = (inputs["query"] * scale).astype(mx.bfloat16)
    score_rows = []
    for row in range(QUERY_ROWS):
        score = (
            scaled_query[:, :, row : row + 1, :]
            @ selected_key[row].swapaxes(-1, -2)
        ).reshape(HEADS, SELECTED_WIDTH)
        score_rows.append(score)
    scores = mx.stack(score_rows)
    scores = mx.where(
        inputs["selected_valid"][:, None, :],
        scores,
        mx.finfo(mx.bfloat16).min,
    )
    mx.eval(projected, scaled_query, scores)
    return {
        "projected": projected,
        "scaled_query": scaled_query,
        "scores": scores,
    }


def _native(plan, inputs: dict[str, mx.array]) -> mx.array:
    output = plan.execute(
        inputs["selected_indices"],
        inputs["selected_valid"],
        inputs["latent"],
        inputs["key_weight"],
        inputs["query"],
        ATTENTION_SCALE,
    )
    mx.eval(output)
    mx.synchronize()
    return output


def _timed(operation, samples: int) -> dict[str, object]:
    values = []
    for _ in range(samples):
        mx.clear_cache()
        started = time.perf_counter_ns()
        output = operation()
        mx.eval(output)
        mx.synchronize()
        values.append((time.perf_counter_ns() - started) / 1e6)
    return {"median_wall_ms": statistics.median(values), "samples_ms": values}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    print(json.dumps({"phase": "build_q256_fixture"}), flush=True)
    inputs, host_indices, host_valid = _fixture()
    print(json.dumps({"phase": "direct_q256_reference"}), flush=True)
    direct_started = time.perf_counter_ns()
    reference = _direct(inputs)
    mx.synchronize()
    direct_reference_ms = (time.perf_counter_ns() - direct_started) / 1e6
    print(json.dumps({"phase": "native_q256_tile_loop"}), flush=True)
    plan = NativeProjectedQKUnionTileLoopPlan(PHYSICAL_K, QUERY_ROWS)
    identities = list(plan.buffer_identities)
    candidate = _native(plan, inputs)

    reference_bits = _bits(reference["scores"]).copy()
    candidate_bits = _bits(candidate).copy()
    union_count = int(np.asarray(plan.union_count, dtype=np.uint32)[0])
    union = np.asarray(plan.union_indices, dtype=np.int32)[:union_count].copy()
    expected_union = np.unique(host_indices[host_valid]).astype(np.int32)
    expected_reverse = np.full(PHYSICAL_K, -1, dtype=np.int32)
    expected_reverse[expected_union] = np.arange(expected_union.size, dtype=np.int32)
    expected_slots = np.full(host_indices.shape, -1, dtype=np.int32)
    expected_slots[host_valid] = expected_reverse[host_indices[host_valid]]
    anchors = {
        "union_indices": union_count == expected_union.size
        and np.array_equal(union, expected_union),
        "query_union_slots": np.array_equal(
            np.asarray(plan.query_union_slots, dtype=np.int32), expected_slots
        ),
        "scaled_queries": np.array_equal(
            _bits(plan.debug_scaled_queries),
            _bits(reference["scaled_query"].transpose(0, 2, 1, 3).reshape(
                QUERY_ROWS, HEADS, LATENT_DIM
            )),
        ),
        "q256_qk_scores": np.array_equal(candidate_bits, reference_bits),
    }
    del reference_bits, candidate_bits
    del reference
    mx.clear_cache()
    gc.collect()

    print(json.dumps({"phase": "timing_direct"}), flush=True)
    direct = _timed(lambda: _direct(inputs)["scores"], 1)
    mx.clear_cache()
    gc.collect()
    print(json.dumps({"phase": "timing_native"}), flush=True)
    native = _timed(lambda: _native(plan, inputs), 3)
    speedup = direct["median_wall_ms"] / native["median_wall_ms"]
    checks = {
        "q256_union_projection_and_scores_byte_exact": all(anchors.values()),
        "all_525056_selected_edges_covered": QUERY_ROWS * SELECTED_WIDTH == 525056,
        "fixed_two_tile_device_count_contract": (
            plan.tile_count == 2
            and identities == list(plan.buffer_identities)
            and plan.dynamic_allocation_count == 0
            and plan.graph_node_count == 0
            and plan.shape_discovery_count == 0
            and plan.host_synchronization_count == 0
            and plan.returned_intermediate_tensor_bytes == 0
        ),
        "selected_key_never_materialized": plan.materialized_selected_key_bytes == 0,
        "scratch_at_most_384mib": plan.scratch_bytes <= MAX_SCRATCH_BYTES,
        "end_to_end_speedup_at_least_1_20x": speedup >= MIN_SPEEDUP,
    }
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-q256-projected-qk-union-tile-loop-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "advance_q256_projected_qk_to_long_context_tiles"
            if accepted
            else "stop_or_redesign_q256_projected_qk_tile_loop"
        ),
        "geometry": {
            "physical_k": PHYSICAL_K,
            "tile_rows": plan.tile_rows,
            "tile_count": plan.tile_count,
            "query_rows": QUERY_ROWS,
            "heads": HEADS,
            "selected_width": SELECTED_WIDTH,
            "selected_edges": QUERY_ROWS * SELECTED_WIDTH,
            "union_count": union_count,
        },
        "anchors": anchors,
        "direct_reference_initial_ms": direct_reference_ms,
        "direct_timing": direct,
        "native_timing": native,
        "speedup": speedup,
        "scratch_bytes": plan.scratch_bytes,
        "materialized_selected_key_bytes": plan.materialized_selected_key_bytes,
        "forbidden_materialized_selected_key_bytes": int(
            QUERY_ROWS * HEADS * SELECTED_WIDTH * LATENT_DIM * 2
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
    del plan, inputs, candidate
    mx.clear_cache()
    gc.collect()
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
