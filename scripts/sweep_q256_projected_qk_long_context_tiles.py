#!/usr/bin/env python3
"""Sweep exact Q256 projected-QK tile geometry at 32K history."""

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
    "m3ultra512-q256-projected-qk-long-context-tile-sweep-20260909.json"
)
PHYSICAL_K = 32 << 10
TILE_ROWS = (4096, 8192, 16384, 32768, 65536)
QUERY_ROWS = 256
HEADS = 64
LATENT_DIM = 512
SELECTED_WIDTH = 2051
VALID_WIDTH = 2048
ATTENTION_SCALE = LATENT_DIM**-0.5
MAX_SCRATCH_BYTES = 5 << 30


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
        # The first sixteen residue classes cover all 32K physical rows while
        # every query retains the production selected width.
        selected = (base * 16 + row % 16) % PHYSICAL_K
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
    mx.eval(selected_indices, selected_valid, latent, key_weight, query)
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
    projected = inputs["latent"] @ inputs["key_weight"]
    sanitized = mx.where(
        inputs["selected_valid"], inputs["selected_indices"],
        mx.zeros_like(inputs["selected_indices"]),
    )
    selected_key = mx.stack(
        [mx.take(projected, sanitized[row], axis=1) for row in range(QUERY_ROWS)]
    )
    scale = mx.array(ATTENTION_SCALE, dtype=mx.bfloat16)
    scaled_query = (inputs["query"] * scale).astype(mx.bfloat16)
    scores = mx.stack([
        (
            scaled_query[:, :, row : row + 1, :]
            @ selected_key[row].swapaxes(-1, -2)
        ).reshape(HEADS, SELECTED_WIDTH)
        for row in range(QUERY_ROWS)
    ])
    scores = mx.where(
        inputs["selected_valid"][:, None, :], scores,
        mx.finfo(mx.bfloat16).min,
    )
    mx.eval(projected, scaled_query, scores)
    return {"projected": projected, "scaled_query": scaled_query, "scores": scores}


def _run(plan, inputs: dict[str, mx.array]) -> mx.array:
    output = plan.execute(
        inputs["selected_indices"], inputs["selected_valid"], inputs["latent"],
        inputs["key_weight"], inputs["query"], ATTENTION_SCALE,
    )
    mx.eval(output)
    mx.synchronize()
    return output


def _candidate(
    tile_rows: int,
    inputs: dict[str, mx.array],
    reference_score_bits: np.ndarray,
    projected: mx.array,
) -> dict[str, object]:
    print(json.dumps({"phase": "tile_candidate", "tile_rows": tile_rows}), flush=True)
    plan = NativeProjectedQKUnionTileLoopPlan(PHYSICAL_K, QUERY_ROWS, tile_rows)
    identities = list(plan.buffer_identities)
    output = _run(plan, inputs)
    count = int(np.asarray(plan.union_count, dtype=np.uint32)[0])
    scores_exact = np.array_equal(_bits(output), reference_score_bits)
    last_projected_row_exact = np.array_equal(
        _bits(plan.debug_projected_union_key_tile[:, (PHYSICAL_K - 1) % tile_rows, :]),
        _bits(projected[:, PHYSICAL_K - 1, :]),
    )
    for _ in range(2):
        _run(plan, inputs)
    samples = []
    for _ in range(3):
        started = time.perf_counter_ns()
        _run(plan, inputs)
        samples.append((time.perf_counter_ns() - started) / 1e6)
    row = {
        "tile_rows": tile_rows,
        "tile_count": plan.tile_count,
        "union_count": count,
        "qk_scores_byte_exact": scores_exact,
        "last_projected_row_byte_exact": last_projected_row_exact,
        "median_wall_ms": statistics.median(samples),
        "samples_ms": samples,
        "scratch_bytes": plan.scratch_bytes,
        "buffer_identities_stable": identities == list(plan.buffer_identities),
        "dynamic_allocation_count": plan.dynamic_allocation_count,
        "graph_node_count": plan.graph_node_count,
        "shape_discovery_count": plan.shape_discovery_count,
        "host_synchronization_count": plan.host_synchronization_count,
        "materialized_selected_key_bytes": plan.materialized_selected_key_bytes,
    }
    del plan, output
    mx.clear_cache()
    gc.collect()
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    print(json.dumps({"phase": "build_32k_q256_fixture"}), flush=True)
    inputs, host_indices, host_valid = _fixture()
    print(json.dumps({"phase": "direct_reference"}), flush=True)
    started = time.perf_counter_ns()
    reference = _direct(inputs)
    mx.synchronize()
    direct_ms = (time.perf_counter_ns() - started) / 1e6
    reference_score_bits = _bits(reference["scores"]).copy()
    expected_union = np.unique(host_indices[host_valid]).astype(np.int32)

    candidates = {}
    for tile_rows in TILE_ROWS:
        candidates[str(tile_rows)] = _candidate(
            tile_rows, inputs, reference_score_bits, reference["projected"]
        )
    best = min(candidates.values(), key=lambda row: row["median_wall_ms"])
    checks = {
        "all_tile_geometries_q256_byte_exact": all(
            row["qk_scores_byte_exact"]
            and row["last_projected_row_byte_exact"]
            and row["union_count"] == expected_union.size
            for row in candidates.values()
        ),
        "all_plans_fixed_without_host_extent_readback": all(
            row["buffer_identities_stable"]
            and row["dynamic_allocation_count"] == 0
            and row["graph_node_count"] == 0
            and row["shape_discovery_count"] == 0
            and row["host_synchronization_count"] == 0
            for row in candidates.values()
        ),
        "selected_key_never_materialized": all(
            row["materialized_selected_key_bytes"] == 0
            for row in candidates.values()
        ),
        "all_scratch_at_most_5gib": all(
            row["scratch_bytes"] <= MAX_SCRATCH_BYTES
            for row in candidates.values()
        ),
        "best_geometry_is_faster_than_direct": best["median_wall_ms"] < direct_ms,
    }
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-q256-projected-qk-long-context-tile-sweep-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "advance_best_q256_tile_geometry_to_320k"
            if accepted
            else "stop_or_redesign_q256_long_context_tile_loop"
        ),
        "geometry": {
            "physical_k": PHYSICAL_K,
            "query_rows": QUERY_ROWS,
            "selected_width": SELECTED_WIDTH,
            "selected_edges": QUERY_ROWS * SELECTED_WIDTH,
            "union_count": int(expected_union.size),
        },
        "direct_reference_wall_ms": direct_ms,
        "candidates": candidates,
        "best_tile_rows": best["tile_rows"],
        "best_tile_count": best["tile_count"],
        "best_wall_ms": best["median_wall_ms"],
        "best_speedup": direct_ms / best["median_wall_ms"],
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
        "best_tile_rows": best["tile_rows"],
        "best_wall_ms": best["median_wall_ms"],
        "failed_gates": [name for name, passed in checks.items() if not passed],
    }))
    del inputs, reference, reference_score_bits
    mx.clear_cache()
    gc.collect()
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
