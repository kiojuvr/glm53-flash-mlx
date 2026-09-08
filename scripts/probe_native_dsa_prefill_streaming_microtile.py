#!/usr/bin/env python3
"""Qualify exact Q=4 score/selection microtiles through 512K geometry.

The composed DSA region must preserve the eager BF16 head-score boundary, but
materializing Q256 x 32 heads x 131072 pools would consume 2 GiB. Q=4 makes
the Steel GEMM M dimension 128 and bounds score/selection scratch to about
33 MiB. This probe validates that microtile at 32K, 128K, 320K, and 512K.

The selected outputs remain diagnostic in this probe. Production promotion is
forbidden until the same buffers are consumed by gather/attention inside the
composed native prefill layer plan.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np

from glm53_flash_mlx.cache_geometry import plan_nope_cache_capacity
from glm53_flash_mlx.indexpool import INDEXPOOL_SENTINEL, expand_selected_pools
from glm53_flash_mlx.native_prefill_plan import (
    NATIVE_PREFILL_PLAN_ABI,
    PROFILE_CONTEXTS,
    plan_native_dsa_prefill_streaming_geometry,
)

ROOT = Path(__file__).resolve().parents[1]
NATIVE_PACKAGE = ROOT / "native_execution"
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-dsa-prefill-streaming-microtile-20260908.json"
)
CONTEXTS = PROFILE_CONTEXTS + (512 << 10,)
QUERY_ROWS = 4
HEADS = 32
HEAD_DIM = 128
SELECTED_POOLS = 512
SOFTMAX_SCALE = HEAD_DIM**-0.5


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_plan_type():
    native = str(NATIVE_PACKAGE)
    if native not in sys.path:
        sys.path.insert(0, native)
    from glm53_native_execution import NativeDSAScoreSelectionPlan

    return NativeDSAScoreSelectionPlan


def _exact(left: mx.array, right: mx.array) -> bool:
    mx.eval(left, right)
    return bool(mx.array_equal(left, right).item())


def _fixture(pool_rows: int) -> dict[str, object]:
    query_axis = mx.arange(
        QUERY_ROWS * HEADS * HEAD_DIM, dtype=mx.float32
    ).reshape(1, QUERY_ROWS, HEADS, HEAD_DIM)
    query = mx.sin(query_axis * 0.0007).astype(mx.bfloat16)
    weights = mx.cos(
        mx.arange(QUERY_ROWS * HEADS, dtype=mx.float32).reshape(
            1, QUERY_ROWS, HEADS
        )
        * 0.013
    ).astype(mx.bfloat16)
    pool_keys = mx.cos(
        mx.arange(pool_rows * HEAD_DIM, dtype=mx.float32).reshape(
            1, pool_rows, HEAD_DIM
        )
        * 0.00011
    ).astype(mx.bfloat16)
    pool_indices = mx.arange(pool_rows * 4, dtype=mx.int64).reshape(
        1, pool_rows, 4
    )
    pool_valid = mx.arange(pool_rows)[None] < (pool_rows - 3)
    raw_positions = mx.zeros((1, 4), dtype=mx.int64)
    raw_valid = mx.zeros((1, 4), dtype=mx.bool_)
    current_valid = mx.array([True, True, True, False], dtype=mx.bool_)
    mx.eval(
        query,
        weights,
        pool_keys,
        pool_indices,
        pool_valid,
        raw_positions,
        raw_valid,
        current_valid,
    )
    return {
        "query": query,
        "weights": weights,
        "pool_keys": pool_keys,
        "pool_indices": pool_indices,
        "pool_valid": pool_valid,
        "raw_positions": raw_positions,
        "raw_valid": raw_valid,
        "current_valid": current_valid,
    }


def _reference(fixture: dict[str, object], pool_rows: int):
    query = fixture["query"]
    weights = fixture["weights"]
    pool_valid = fixture["pool_valid"]
    head_scores = (
        query @ fixture["pool_keys"][:, None].swapaxes(-1, -2)
    ).reshape(QUERY_ROWS * HEADS, pool_rows)
    shaped = head_scores.reshape(1, QUERY_ROWS, HEADS, pool_rows)
    scores = mx.maximum(shaped * SOFTMAX_SCALE, 0.0)
    scores = mx.sum(weights[..., None] * scores, axis=2)
    valid_candidates = mx.broadcast_to(
        pool_valid[:, None], (1, QUERY_ROWS, pool_rows)
    )
    scores = mx.where(valid_candidates, scores, -1e30)
    scores = mx.contiguous(scores, allow_col_major=False)
    selected = mx.argsort(-scores, axis=-1)[..., :SELECTED_POOLS]
    selected_valid = mx.take_along_axis(valid_candidates, selected, axis=-1)
    indices, valid = expand_selected_pools(
        selected,
        fixture["pool_indices"],
        selected_valid,
        kv_len=pool_rows * 4,
        index_topk=2_048,
        index_kpool=4,
        tail_positions=mx.zeros((1, 0), dtype=mx.int64),
        tail_valid=mx.zeros((1, 0), dtype=mx.bool_),
        always_select_tail=True,
    )
    valid = valid & fixture["current_valid"][..., None]
    indices = mx.where(valid, indices, INDEXPOOL_SENTINEL)
    return head_scores, scores, indices, valid


def _native(plan, fixture: dict[str, object], pool_rows: int):
    dependencies = tuple(fixture.values())
    mx.async_eval(*dependencies)
    return plan.execute(
        fixture["query"],
        fixture["weights"],
        fixture["pool_keys"],
        fixture["pool_indices"],
        fixture["pool_valid"],
        fixture["raw_positions"],
        fixture["raw_valid"],
        fixture["current_valid"],
        pool_rows,
        pool_rows * 4,
        0,
    )


def _context_case(plan_type, context: int, samples: int) -> dict[str, object]:
    capacity = plan_nope_cache_capacity(context)
    pool_rows = capacity.physical_pool_rows
    geometry = plan_native_dsa_prefill_streaming_geometry(
        physical_pool_rows=pool_rows
    )
    _progress("streaming_microtile", context=context, pools=pool_rows)
    fixture = _fixture(pool_rows)
    reference_head, reference_scores, reference_indices, reference_valid = (
        _reference(fixture, pool_rows)
    )
    plan = plan_type("prefill", QUERY_ROWS, pool_rows, float(SOFTMAX_SCALE))
    identities = list(plan.buffer_identities)
    candidate_indices, candidate_valid = _native(plan, fixture, pool_rows)
    mx.eval(
        reference_head,
        reference_scores,
        reference_indices,
        reference_valid,
        candidate_indices,
        candidate_valid,
        plan.debug_head_scores,
        plan.debug_index_scores,
    )
    exactness = {
        "bf16_head_scores": _exact(reference_head, plan.debug_head_scores),
        "bf16_index_scores": _exact(reference_scores, plan.debug_index_scores),
        "selected_token_indices": _exact(reference_indices, candidate_indices),
        "selected_token_validity": _exact(reference_valid, candidate_valid),
    }
    timings = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        output = _native(plan, fixture, pool_rows)
        submitted = time.perf_counter_ns()
        mx.eval(*output)
        finished = time.perf_counter_ns()
        timings.append(
            {
                "host_submit_ms": (submitted - started) / 1e6,
                "synchronized_wall_ms": (finished - started) / 1e6,
            }
        )
    final_identities = list(plan.buffer_identities)
    return {
        "context_tokens": context,
        "physical_pool_rows": pool_rows,
        "geometry": geometry.descriptor(),
        "exactness": exactness,
        "all_exact": all(exactness.values()),
        "scratch_bytes": plan.scratch_bytes,
        "returned_score_tensor_bytes": plan.returned_score_tensor_bytes,
        "dynamic_allocation_count": plan.dynamic_allocation_count,
        "graph_node_count": plan.graph_node_count,
        "shape_discovery_count": plan.shape_discovery_count,
        "host_synchronization_count": plan.host_synchronization_count,
        "buffer_identities_stable": identities == final_identities,
        "execution_count": plan.execution_count,
        "median_host_submit_ms": statistics.median(
            row["host_submit_ms"] for row in timings
        ),
        "median_synchronized_wall_ms": statistics.median(
            row["synchronized_wall_ms"] for row in timings
        ),
        "timing_is_production_prefill_claim": False,
        "samples": timings,
    }


def main_artifact(contexts: dict[str, object]) -> dict[str, object]:
    rows = list(contexts.values())
    checks = {
        "32k_128k_320k_512k_pool_geometries_measured": (
            set(map(int, contexts)) == set(CONTEXTS)
        ),
        "all_bf16_boundaries_and_selection_outputs_byte_exact": (
            len(rows) == len(CONTEXTS) and all(row["all_exact"] for row in rows)
        ),
        "512k_microtile_score_selection_scratch_at_most_64mib": (
            contexts.get(str(512 << 10), {}).get("scratch_bytes", 1 << 62)
            <= 64 << 20
        ),
        "steel_m128_geometry_is_aligned": all(
            row["geometry"]["matrix_rows"] == 128
            and row["geometry"]["steel_matrix_rows_aligned"]
            for row in rows
        ),
        "q256_is_64_reused_microtiles": all(
            row["geometry"]["microtiles_per_256_row_chunk"] == 64
            for row in rows
        ),
        "no_score_tensor_escapes_native_plan": all(
            row["returned_score_tensor_bytes"] == 0 for row in rows
        ),
        "executor_has_no_allocation_graph_shape_or_sync": all(
            row["dynamic_allocation_count"] == 0
            and row["graph_node_count"] == 0
            and row["shape_discovery_count"] == 0
            and row["host_synchronization_count"] == 0
            for row in rows
        ),
        "all_buffer_addresses_stable": all(
            row["buffer_identities_stable"] for row in rows
        ),
    }
    accepted = all(checks.values())
    return {
        "schema": "glm53-native-dsa-prefill-streaming-microtile-v1",
        "date": date.today().isoformat(),
        "complete": len(contexts) == len(CONTEXTS),
        "accepted": accepted,
        "decision": (
            "embed_exact_q4_score_selection_into_composed_dsa_region"
            if accepted
            else "stop_native_dsa_prefill_streaming_geometry"
        ),
        "probe_only": True,
        "native_prefill_plan_abi": NATIVE_PREFILL_PLAN_ABI,
        "contexts": contexts,
        "checks": checks,
        "promotion": {
            "runtime": False,
            "partial_score_selection": False,
            "requires_composed_gather_attention": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args(argv)
    if args.samples < 2:
        raise ValueError("at least two timing samples are required")
    plan_type = _load_plan_type()
    contexts = {}
    for context in CONTEXTS:
        contexts[str(context)] = _context_case(plan_type, context, args.samples)
        artifact = main_artifact(contexts)
        _atomic_write(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": artifact["complete"],
                "accepted": artifact["accepted"],
                "decision": artifact["decision"],
            }
        )
    )
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
