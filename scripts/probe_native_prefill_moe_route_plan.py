#!/usr/bin/env python3
"""Qualify fixed-arena native route grouping for a Q256/top-8 MoE tile."""

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


ROOT = Path(__file__).resolve().parents[1]
NATIVE_PACKAGE = ROOT / "native_execution"
if str(NATIVE_PACKAGE) not in sys.path:
    sys.path.insert(0, str(NATIVE_PACKAGE))

from glm53_native_execution import NativePrefillMoERoutePlan
from glm53_flash_mlx.grouped_fp8 import build_route_plan


DEFAULT_OUTPUT = ROOT / "bench-results" / (
    "m3ultra512-native-prefill-moe-route-plan-20260909.json"
)
QUERY_ROWS = 256
TOP_K = 8
ROUTES = QUERY_ROWS * TOP_K
EXPERTS = 288
HIDDEN = 4096
TILE_ROWS = 8


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _fixture() -> tuple[mx.array, mx.array, mx.array]:
    routes = np.arange(ROUTES, dtype=np.uint32)
    # Deliberately creates large and irregular tie groups. Original route
    # position is therefore observable in the stable sort oracle.
    expert_ids = ((routes * 73 + (routes // 7) * 19) % EXPERTS).reshape(
        QUERY_ROWS, TOP_K
    )
    scores = (
        np.sin(np.arange(ROUTES, dtype=np.float32) * np.float32(0.0137))
        * np.float32(0.25)
        + np.float32(0.5)
    ).reshape(QUERY_ROWS, TOP_K)
    hidden = mx.sin(
        mx.arange(QUERY_ROWS * HIDDEN, dtype=mx.float32).reshape(
            QUERY_ROWS, HIDDEN
        )
        * 0.00017
    ).astype(mx.bfloat16)
    values = (
        mx.contiguous(hidden),
        mx.contiguous(mx.array(expert_ids, dtype=mx.uint32)),
        mx.contiguous(mx.array(scores, dtype=mx.float32)),
    )
    mx.eval(*values)
    return values


def _reference(hidden: mx.array, expert_ids: mx.array, scores: mx.array):
    sorted_hidden, sorted_experts, sorted_scores, inverse = build_route_plan(
        hidden, expert_ids, scores
    )
    flat_ids = expert_ids.reshape(-1)
    order = mx.argsort(flat_ids)
    values = (sorted_hidden, sorted_experts, sorted_scores, inverse, order)
    mx.eval(*values)
    mx.synchronize()
    return values


def _candidate(plan, expert_ids, scores):
    outputs = tuple(plan.execute(expert_ids, scores))
    mx.eval(*outputs)
    mx.synchronize()
    return outputs


def _timed(operation, samples: int = 9) -> dict[str, object]:
    operation()
    values = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        operation()
        values.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "median_wall_ms": statistics.median(values),
        "samples_ms": values,
    }


def _expected_metadata(expert_ids: mx.array):
    ids = np.asarray(expert_ids, dtype=np.uint32).reshape(-1)
    order = np.argsort(ids, kind="stable").astype(np.uint32)
    sorted_ids = ids[order]
    counts = np.bincount(ids, minlength=EXPERTS).astype(np.uint32)
    offsets = np.concatenate(
        [np.zeros(1, dtype=np.uint32), np.cumsum(counts, dtype=np.uint32)]
    )
    tile_experts = []
    tile_starts = []
    tile_lengths = []
    for expert in range(EXPERTS):
        start = int(offsets[expert])
        stop = int(offsets[expert + 1])
        for tile_start in range(start, stop, TILE_ROWS):
            tile_experts.append(expert)
            tile_starts.append(tile_start)
            tile_lengths.append(min(TILE_ROWS, stop - tile_start))
    return (
        order,
        sorted_ids,
        offsets,
        np.asarray(tile_experts, dtype=np.uint32),
        np.asarray(tile_starts, dtype=np.uint32),
        np.asarray(tile_lengths, dtype=np.uint32),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=9)
    args = parser.parse_args(argv)

    print(json.dumps({"phase": "build_q256_route_fixture"}), flush=True)
    hidden, expert_ids, scores = _fixture()
    plan = NativePrefillMoERoutePlan(EXPERTS)
    identities = list(plan.buffer_identities)

    print(json.dumps({"phase": "exact_reference"}), flush=True)
    reference = _reference(hidden, expert_ids, scores)
    print(json.dumps({"phase": "native_stable_group"}), flush=True)
    actual = _candidate(plan, expert_ids, scores)
    expected = _expected_metadata(expert_ids)
    descriptor_count = int(np.asarray(actual[8], dtype=np.uint32)[0])
    invalid_count = int(np.asarray(actual[9], dtype=np.uint32)[0])
    inverse_expected = np.argsort(expected[0], kind="stable").astype(np.uint32)

    exactness = {
        "route_order": np.array_equal(
            np.asarray(actual[0], dtype=np.uint32), expected[0]
        ),
        "inverse_order": np.array_equal(
            np.asarray(actual[1], dtype=np.uint32), inverse_expected
        ),
        "sorted_experts": np.array_equal(
            np.asarray(actual[2], dtype=np.uint32), expected[1]
        ),
        "sorted_scores": np.array_equal(
            np.asarray(actual[3], dtype=np.float32).view(np.uint32),
            np.asarray(reference[2], dtype=np.float32).view(np.uint32),
        ),
        "expert_offsets": np.array_equal(
            np.asarray(actual[4], dtype=np.uint32), expected[2]
        ),
        "tile_experts": np.array_equal(
            np.asarray(actual[5], dtype=np.uint32)[:descriptor_count], expected[3]
        ),
        "tile_starts": np.array_equal(
            np.asarray(actual[6], dtype=np.uint32)[:descriptor_count], expected[4]
        ),
        "tile_lengths": np.array_equal(
            np.asarray(actual[7], dtype=np.uint32)[:descriptor_count], expected[5]
        ),
        "descriptor_count": descriptor_count == len(expected[3]),
        "invalid_route_count": invalid_count == 0,
        "mlx_argsort_order": np.array_equal(
            np.asarray(reference[4], dtype=np.uint32), expected[0]
        ),
        "indirect_hidden_addressing": np.array_equal(
            np.asarray(hidden.view(mx.uint16), dtype=np.uint16)[
                np.asarray(actual[0], dtype=np.uint32) // TOP_K
            ],
            np.asarray(reference[0].view(mx.uint16)),
        ),
    }

    print(json.dumps({"phase": "invalid_route_fail_closed_flag"}), flush=True)
    invalid_host = np.asarray(expert_ids, dtype=np.uint32).copy()
    invalid_host.reshape(-1)[17] = EXPERTS
    invalid = mx.contiguous(mx.array(invalid_host, dtype=mx.uint32))
    mx.eval(invalid)
    invalid_outputs = _candidate(plan, invalid, scores)
    invalid_flag = int(np.asarray(invalid_outputs[9], dtype=np.uint32)[0])
    # A subsequent valid execute must clear the device flag and reproduce all
    # outputs, proving the failure signal does not poison the persistent plan.
    repeated = _candidate(plan, expert_ids, scores)
    repeat_exact = all(
        np.array_equal(np.asarray(repeated[i]), np.asarray(actual[i]))
        for i in range(len(actual))
    )

    print(json.dumps({"phase": "time_mlx_route_materialization"}), flush=True)
    reference_timing = _timed(
        lambda: _reference(hidden, expert_ids, scores), args.samples
    )
    print(json.dumps({"phase": "time_native_route_metadata"}), flush=True)
    native_timing = _timed(
        lambda: _candidate(plan, expert_ids, scores), args.samples
    )
    speedup = (
        reference_timing["median_wall_ms"] / native_timing["median_wall_ms"]
    )
    checks = {
        "all_route_metadata_byte_exact": all(exactness.values()),
        "stable_equal_expert_order_matches_mlx": exactness["mlx_argsort_order"],
        "invalid_expert_sets_device_failure_flag": invalid_flag == 1,
        "valid_reexecution_clears_failure_and_is_exact": repeat_exact,
        "sorted_hidden_never_materialized": (
            plan.materialized_sorted_hidden_bytes == 0
        ),
        "fixed_arena_and_submission_contract": (
            identities == list(plan.buffer_identities)
            and plan.dynamic_allocation_count == 0
            and plan.graph_node_count == 0
            and plan.shape_discovery_count == 0
            and plan.host_synchronization_count == 0
        ),
        # This boundary is deliberately not promotable alone. Its metadata is
        # consumed by the immediately following expert kernels in one encoder;
        # returning it to MLX would preserve exactly the boundary tax measured
        # here and repeat the rejected partial-kernel design.
        "standalone_boundary_is_explicitly_non_promotable": speedup < 1.0,
    }
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-native-prefill-moe-route-plan-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "advance_route_grouping_only_inside_composed_expert_island"
            if accepted
            else "stop_or_redesign_native_prefill_moe_route_plan"
        ),
        "geometry": {
            "query_rows": QUERY_ROWS,
            "top_k": TOP_K,
            "route_rows": ROUTES,
            "expert_count": EXPERTS,
            "tile_rows": TILE_ROWS,
            "descriptor_count": descriptor_count,
            "descriptor_capacity": plan.descriptor_capacity,
        },
        "exactness": exactness,
        "invalid_route_count": invalid_flag,
        "repeat_exact": repeat_exact,
        "reference_timing": reference_timing,
        "native_timing": native_timing,
        "speedup": speedup,
        "standalone_performance_accepted": False,
        "scratch_bytes": plan.scratch_bytes,
        "avoided_sorted_hidden_bytes": QUERY_ROWS * TOP_K * HIDDEN * 2,
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
        "speedup": speedup,
        "failed_gates": [name for name, passed in checks.items() if not passed],
    }))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
