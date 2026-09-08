#!/usr/bin/env python3
"""Qualify a fixed-topology, device-count-driven projected-QK tile loop."""

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
    "m3ultra512-device-count-projected-qk-union-tile-loop-20260909.json"
)
PHYSICAL_K = 8192
TILE_ROWS = 4096
SELECTION_QUERY_ROWS = 256
ATTENTION_QUERY_ROWS = 4
HEADS = 64
LATENT_DIM = 512
SELECTED_WIDTH = 2051
VALID_WIDTH = 2048
ATTENTION_SCALE = LATENT_DIM**-0.5
SAMPLES = 3
MAX_SCRATCH_BYTES = 300 << 20
MIN_SPEEDUP = 1.10


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _bits(value: mx.array) -> np.ndarray:
    return np.asarray(value.view(mx.uint16), dtype=np.uint16)


def _fixture() -> tuple[dict[str, mx.array], np.ndarray, np.ndarray]:
    host_indices = np.full(
        (SELECTION_QUERY_ROWS, SELECTED_WIDTH), -1, dtype=np.int32
    )
    host_valid = np.zeros(
        (SELECTION_QUERY_ROWS, SELECTED_WIDTH), dtype=np.bool_
    )
    base = np.arange(VALID_WIDTH, dtype=np.int64)
    for row in range(SELECTION_QUERY_ROWS):
        # The first four query rows partition every fourth physical position,
        # so each attention query has edges in both tiles and their union is
        # exactly the full 8K physical range.
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
        mx.arange(
            HEADS * ATTENTION_QUERY_ROWS * LATENT_DIM, dtype=mx.float32
        ).reshape(1, HEADS, ATTENTION_QUERY_ROWS, LATENT_DIM)
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


def _union_reference(
    indices: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    union = np.unique(indices[valid]).astype(np.int32)
    reverse = np.full(PHYSICAL_K, -1, dtype=np.int32)
    reverse[union] = np.arange(union.size, dtype=np.int32)
    slots = np.full(indices.shape, -1, dtype=np.int32)
    slots[valid] = reverse[indices[valid]]
    return union, slots


def _direct(inputs: dict[str, mx.array]) -> dict[str, mx.array]:
    projected = inputs["latent"] @ inputs["key_weight"]
    sanitized = mx.where(
        inputs["selected_valid"][:ATTENTION_QUERY_ROWS],
        inputs["selected_indices"][:ATTENTION_QUERY_ROWS],
        mx.zeros_like(inputs["selected_indices"][:ATTENTION_QUERY_ROWS]),
    )
    selected_key = mx.stack(
        [
            mx.take(projected, sanitized[row], axis=1)
            for row in range(ATTENTION_QUERY_ROWS)
        ]
    )
    scale = mx.array(ATTENTION_SCALE, dtype=mx.bfloat16)
    scaled_query = (inputs["query"] * scale).astype(mx.bfloat16)
    score_rows = []
    for row in range(ATTENTION_QUERY_ROWS):
        score = (
            scaled_query[:, :, row : row + 1, :]
            @ selected_key[row].swapaxes(-1, -2)
        ).reshape(HEADS, SELECTED_WIDTH)
        score_rows.append(score)
    scores = mx.stack(score_rows)
    scores = mx.where(
        inputs["selected_valid"][:ATTENTION_QUERY_ROWS, None, :],
        scores,
        mx.finfo(mx.bfloat16).min,
    )
    mx.eval(projected, selected_key, scaled_query, scores)
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
    # Plan-owned buffers are outside the MLX graph.  Diagnostic inspection
    # drains the shared stream here; execute itself never synchronizes.
    mx.synchronize()
    return output


def _timed(operation) -> dict[str, object]:
    for _ in range(2):
        mx.eval(operation())
        mx.synchronize()
    samples = []
    for _ in range(SAMPLES):
        started = time.perf_counter_ns()
        mx.eval(operation())
        mx.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return {"median_wall_ms": statistics.median(samples), "samples_ms": samples}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    print(json.dumps({"phase": "build_fixture", "physical_k": PHYSICAL_K}), flush=True)
    inputs, host_indices, host_valid = _fixture()
    union_reference, slot_reference = _union_reference(host_indices, host_valid)
    print(json.dumps({"phase": "direct_reference"}), flush=True)
    reference = _direct(inputs)
    print(json.dumps({"phase": "native_tile_loop"}), flush=True)
    plan = NativeProjectedQKUnionTileLoopPlan(PHYSICAL_K)
    identities = list(plan.buffer_identities)
    candidate = _native(plan, inputs)
    count = int(np.asarray(plan.union_count, dtype=np.uint32)[0])
    union = np.asarray(plan.union_indices, dtype=np.int32)[:count].copy()
    slots = np.asarray(plan.query_union_slots, dtype=np.int32).copy()
    last_tile_reference = reference["projected"][:, TILE_ROWS:PHYSICAL_K]
    anchors = {
        "union_indices": count == union_reference.size
        and np.array_equal(union, union_reference),
        "query_union_slots": np.array_equal(slots, slot_reference),
        "last_union_latent_tile": np.array_equal(
            _bits(plan.debug_union_latent_tile),
            _bits(inputs["latent"][TILE_ROWS:PHYSICAL_K]),
        ),
        "last_projected_union_key_tile": np.array_equal(
            _bits(plan.debug_projected_union_key_tile),
            _bits(last_tile_reference),
        ),
        "scaled_queries": np.array_equal(
            _bits(plan.debug_scaled_queries),
            _bits(reference["scaled_query"].transpose(0, 2, 1, 3).reshape(
                ATTENTION_QUERY_ROWS, HEADS, LATENT_DIM
            )),
        ),
        "qk_scores": np.array_equal(_bits(candidate), _bits(reference["scores"])),
    }

    print(json.dumps({"phase": "timing"}), flush=True)
    direct = _timed(lambda: _direct(inputs)["scores"])
    native = _timed(lambda: _native(plan, inputs))
    speedup = direct["median_wall_ms"] / native["median_wall_ms"]
    fixed_contract = (
        identities == list(plan.buffer_identities)
        and plan.dynamic_allocation_count == 0
        and plan.graph_node_count == 0
        and plan.shape_discovery_count == 0
        and plan.host_synchronization_count == 0
        and plan.returned_intermediate_tensor_bytes == 0
    )
    checks = {
        "all_union_projection_and_qk_anchors_byte_exact": all(anchors.values()),
        "two_capacity_tiles_encoded_without_count_readback": (
            plan.tile_count == 2 and plan.tile_rows == TILE_ROWS
            and plan.host_synchronization_count == 0
        ),
        "fixed_execution_contract": fixed_contract,
        "selected_key_never_materialized": plan.materialized_selected_key_bytes == 0,
        "scratch_at_most_300mib": plan.scratch_bytes <= MAX_SCRATCH_BYTES,
        "end_to_end_speedup_at_least_1_10x": speedup >= MIN_SPEEDUP,
    }
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-device-count-projected-qk-union-tile-loop-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "advance_projected_qk_tile_loop_to_q256_geometry"
            if accepted
            else "stop_or_redesign_device_count_projected_qk_tile_loop"
        ),
        "geometry": {
            "physical_k": PHYSICAL_K,
            "tile_rows": plan.tile_rows,
            "tile_count": plan.tile_count,
            "selection_query_rows": SELECTION_QUERY_ROWS,
            "attention_query_rows": ATTENTION_QUERY_ROWS,
            "selected_width": SELECTED_WIDTH,
            "union_count": count,
        },
        "anchors": anchors,
        "direct_timing": direct,
        "native_timing": native,
        "speedup": speedup,
        "scratch_bytes": plan.scratch_bytes,
        "materialized_selected_key_bytes": plan.materialized_selected_key_bytes,
        "forbidden_materialized_selected_key_bytes": int(
            ATTENTION_QUERY_ROWS * HEADS * SELECTED_WIDTH * LATENT_DIM * 2
        ),
        "execution_count": plan.execution_count,
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
