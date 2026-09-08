#!/usr/bin/env python3
"""Qualify the device-resident Q256 selected-token union substrate."""

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

from glm53_native_execution import NativeSelectedUnionPlan


DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-device-resident-q256-selected-union-20260908.json"
)
CONTEXTS = (32 << 10, 128 << 10, 320 << 10)
QUERY_ROWS = 256
SELECTED_WIDTH = 2051
VALID_WIDTH = 2048
SAMPLES = 5
MAX_SCRATCH_BYTES = 8 << 20
MAX_320K_WALL_MS = 5.0


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _fixture(context: int, shift: int = 0):
    rows = np.full((QUERY_ROWS, SELECTED_WIDTH), -1, dtype=np.int32)
    valid = np.zeros((QUERY_ROWS, SELECTED_WIDTH), dtype=np.bool_)
    base = np.arange(VALID_WIDTH, dtype=np.int64)
    for row in range(QUERY_ROWS):
        selected = (
            base * 8191 + row * 97 + shift * 193
        ) % context
        selected.sort()
        rows[row, :VALID_WIDTH] = selected.astype(np.int32)
        valid[row, :VALID_WIDTH] = True
    indices = mx.array(rows)
    validity = mx.array(valid)
    mx.eval(indices, validity)
    return rows, valid, indices, validity


def _reference(rows: np.ndarray, valid: np.ndarray, context: int):
    union = np.unique(rows[valid])
    physical_to_union = np.full(context, -1, dtype=np.int32)
    physical_to_union[union] = np.arange(union.size, dtype=np.int32)
    slots = np.full(rows.shape, -1, dtype=np.int32)
    slots[valid] = physical_to_union[rows[valid]]
    return union.astype(np.int32), slots


def _run(plan, indices, valid):
    union, count, slots = plan.execute(indices, valid)
    mx.eval(union, count, slots)
    # Native plans encode directly into persistent owned buffers.  The probe
    # must drain that command stream before inspecting those buffers on the
    # CPU; this synchronization is diagnostic-only and is not part of the
    # plan's execution contract.
    mx.synchronize()
    return union, count, slots


def _context_case(context: int) -> dict[str, object]:
    print(json.dumps({"phase": "q256_selected_union", "context": context}), flush=True)
    host_indices, host_valid, indices, valid = _fixture(context)
    reference_union, reference_slots = _reference(
        host_indices, host_valid, context
    )
    plan = NativeSelectedUnionPlan(context)
    identities = list(plan.buffer_identities)
    union, count, slots = _run(plan, indices, valid)
    count_value = int(np.asarray(count, dtype=np.uint32)[0])
    membership_host = np.asarray(
        plan.debug_membership_words, dtype=np.uint32
    ).copy()
    block_counts_host = np.asarray(
        plan.debug_block_counts, dtype=np.uint32
    ).copy()
    union_host = np.asarray(union, dtype=np.int32)[:count_value].copy()
    slots_host = np.asarray(slots, dtype=np.int32).copy()
    first_exact = (
        count_value == reference_union.size
        and np.array_equal(union_host, reference_union)
        and np.array_equal(slots_host, reference_slots)
    )

    # A second, shifted membership set proves that bitmap/count state is
    # cleared rather than accidentally retained across executions.
    second_host, second_valid_host, second_indices, second_valid = _fixture(
        context, shift=1
    )
    second_reference, second_slots_reference = _reference(
        second_host, second_valid_host, context
    )
    second_union, second_count, second_slots = _run(
        plan, second_indices, second_valid
    )
    second_count_value = int(np.asarray(second_count, dtype=np.uint32)[0])
    second_exact = (
        second_count_value == second_reference.size
        and np.array_equal(
            np.asarray(second_union, dtype=np.int32)[:second_count_value],
            second_reference,
        )
        and np.array_equal(
            np.asarray(second_slots, dtype=np.int32), second_slots_reference
        )
    )

    for _ in range(2):
        _run(plan, indices, valid)
    samples = []
    for _ in range(SAMPLES):
        started = time.perf_counter_ns()
        output = plan.execute(indices, valid)
        mx.eval(*output)
        mx.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    result = {
        "context_tokens": context,
        "reference_union_count": int(reference_union.size),
        "candidate_union_count": count_value,
        "membership_nonzero_words": int(np.count_nonzero(membership_host)),
        "membership_popcount": int(
            sum(int(value).bit_count() for value in membership_host)
        ),
        "block_count_sum": int(block_counts_host.sum()),
        "union_indices_byte_exact": np.array_equal(union_host, reference_union),
        "query_union_slots_byte_exact": np.array_equal(slots_host, reference_slots),
        "first_membership_exact": first_exact,
        "replacement_membership_exact": second_exact,
        "median_wall_ms": statistics.median(samples),
        "samples_ms": samples,
        "scratch_bytes": plan.scratch_bytes,
        "buffer_identities_stable": identities == list(plan.buffer_identities),
        "execution_count": plan.execution_count,
        "dynamic_allocation_count": plan.dynamic_allocation_count,
        "graph_node_count": plan.graph_node_count,
        "shape_discovery_count": plan.shape_discovery_count,
        "host_synchronization_count": plan.host_synchronization_count,
        "union_count_is_device_buffer": str(count.dtype) == "mlx.core.uint32",
    }
    del (
        plan, indices, valid, second_indices, second_valid, union, count, slots,
        second_union, second_count, second_slots,
    )
    mx.clear_cache()
    gc.collect()
    return result


def _artifact(contexts: dict[str, object]) -> dict[str, object]:
    complete = set(map(int, contexts)) == set(CONTEXTS)
    long = contexts.get(str(320 << 10), {})
    checks = {
        "32k_128k_320k_measured": complete,
        "physical_order_union_and_query_slots_byte_exact": complete and all(
            row["first_membership_exact"]
            and row["replacement_membership_exact"]
            for row in contexts.values()
        ),
        "union_count_remains_device_resident": complete and all(
            row["union_count_is_device_buffer"] for row in contexts.values()
        ),
        "scratch_at_most_8mib": complete and all(
            row["scratch_bytes"] <= MAX_SCRATCH_BYTES
            for row in contexts.values()
        ),
        "buffers_stable_and_no_alloc_graph_shape_or_host_sync": complete and all(
            row["buffer_identities_stable"]
            and row["dynamic_allocation_count"] == 0
            and row["graph_node_count"] == 0
            and row["shape_discovery_count"] == 0
            and row["host_synchronization_count"] == 0
            for row in contexts.values()
        ),
        "320k_union_build_at_most_5ms": (
            complete and long["median_wall_ms"] <= MAX_320K_WALL_MS
        ),
    }
    accepted = all(checks.values())
    return {
        "schema": "glm53-device-resident-q256-selected-union-v1",
        "date": date.today().isoformat(),
        "complete": complete,
        "accepted": accepted,
        "decision": (
            "advance_q256_union_to_indirect_selected_kv_projection"
            if accepted
            else "stop_or_redesign_device_resident_q256_union"
        ),
        "contexts": contexts,
        "checks": checks,
        "probe_only": True,
        "runtime_changes": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    contexts = {}
    for context in CONTEXTS:
        contexts[str(context)] = _context_case(context)
        artifact = _artifact(contexts)
        _atomic_write(args.output, artifact)
        if not (
            contexts[str(context)]["first_membership_exact"]
            and contexts[str(context)]["replacement_membership_exact"]
        ):
            break
    print(json.dumps({
        "output": str(args.output),
        "complete": artifact["complete"],
        "accepted": artifact["accepted"],
        "decision": artifact["decision"],
        "failed_gates": [
            name for name, passed in artifact["checks"].items() if not passed
        ],
    }))
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
