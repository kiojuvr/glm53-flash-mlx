#!/usr/bin/env python3
"""Qualify the persistent native virtual-BK16 sparse-prefill AV plan."""

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
SCRIPTS = ROOT / "scripts"
NATIVE_PACKAGE = ROOT / "native_execution"
for path in (SCRIPTS, NATIVE_PACKAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import localize_sparse_prefill_attention_reduction as localization
import probe_sparse_prefill_reordering_equivalence as equivalence
from glm53_native_execution import NativeSparsePrefillAVPlan


DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-virtual-bk16-prefill-av-20260908.json"
)
CONTEXTS = (2_048, 32_768)
SAMPLES = 3
MAX_SCRATCH_BYTES = 1 << 20


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _timed(operation) -> dict[str, object]:
    rows = []
    for _ in range(SAMPLES):
        started = time.perf_counter_ns()
        output = operation()
        host_submit_ms = (time.perf_counter_ns() - started) / 1e6
        mx.eval(output)
        mx.synchronize()
        rows.append({
            "host_submit_ms": host_submit_ms,
            "wall_ms": (time.perf_counter_ns() - started) / 1e6,
        })
    return {
        "median_host_submit_ms": statistics.median(
            row["host_submit_ms"] for row in rows
        ),
        "median_wall_ms": statistics.median(row["wall_ms"] for row in rows),
        "samples": rows,
    }


def _context_case(context: int) -> dict[str, object]:
    print(json.dumps({"phase": "native_virtual_bk16_context", "context": context}), flush=True)
    fixture = equivalence._fixture(context)
    latent = fixture["latent"]
    key = latent @ fixture["k_weight"]
    value = latent @ fixture["v_weight"].swapaxes(-1, -2)
    safe = mx.where(fixture["valid"], fixture["indices"], context)
    mask = mx.zeros(
        (1, 1, equivalence.QUERY_ROWS, context + 1), dtype=mx.bool_
    )
    mask = mx.put_along_axis(mask, safe[:, None], mx.array(True), axis=-1)[
        ..., :context
    ]
    scale = mx.array(equivalence.LATENT_DIM**-0.5, dtype=mx.bfloat16)
    scores = (fixture["query"] * scale).astype(mx.bfloat16) @ key.swapaxes(-1, -2)
    scores = mx.where(mask, scores, mx.finfo(mx.bfloat16).min)
    probabilities = mx.softmax(scores, axis=-1, precise=True)
    direct = probabilities @ value
    selected_probabilities = localization._gather_query_axis(
        probabilities, safe
    )
    selected_values = equivalence._gather_per_query(
        value, fixture["indices"], fixture["valid"]
    )
    mx.eval(direct, selected_probabilities, selected_values, value)

    rows = []
    for row in range(equivalence.QUERY_ROWS):
        plan = NativeSparsePrefillAVPlan(context)
        row_probabilities = mx.contiguous(selected_probabilities[row])
        row_values = mx.contiguous(selected_values[row])
        row_indices = mx.contiguous(fixture["indices"][0, row])
        row_valid = mx.contiguous(fixture["valid"][0, row])
        mx.eval(row_probabilities, row_values, row_indices, row_valid)
        identities = list(plan.buffer_identities)
        candidate = plan.execute(
            row_probabilities, row_values, row_indices, row_valid
        )
        reference = direct[:, :, row : row + 1, :]
        mx.eval(candidate, reference)
        different = int(mx.sum(candidate != reference).item())
        mapping = np.asarray(plan.debug_lane_to_selected, dtype=np.int32)
        valid_np = np.asarray(row_valid, dtype=np.bool_)
        indices_np = np.asarray(row_indices, dtype=np.int32)
        mapped = mapping[mapping >= 0]
        map_exact = bool(
            np.array_equal(mapped, np.flatnonzero(valid_np).astype(np.int32))
            and np.array_equal(
                np.flatnonzero(mapping >= 0) % 16,
                indices_np[valid_np] % 16,
            )
        )
        rows.append({
            "row": row,
            "different_output_elements": different,
            "byte_exact": different == 0,
            "lane_map_exact": map_exact,
            "buffer_identities_stable": identities == list(plan.buffer_identities),
            "packed_k": plan.packed_k,
            "scratch_bytes": plan.scratch_bytes,
            "execution_count": plan.execution_count,
            "structural_counters": {
                "dynamic_allocation_count": plan.dynamic_allocation_count,
                "graph_node_count": plan.graph_node_count,
                "shape_discovery_count": plan.shape_discovery_count,
                "host_synchronization_count": plan.host_synchronization_count,
                "returned_intermediate_tensor_bytes": (
                    plan.returned_intermediate_tensor_bytes
                ),
            },
        })

    # Time one stable row after exactness. Direct reads physical K; native
    # gathers selected values directly into BK16 threadgroup tiles. The
    # prepared arm attributes AV independently of lane-map construction.
    row = 3
    plan = NativeSparsePrefillAVPlan(context)
    row_probabilities = mx.contiguous(selected_probabilities[row])
    row_values = mx.contiguous(selected_values[row])
    row_indices = mx.contiguous(fixture["indices"][0, row])
    row_valid = mx.contiguous(fixture["valid"][0, row])
    mx.eval(row_probabilities, row_values, row_indices, row_valid)
    direct_timing = _timed(
        lambda: probabilities[:, :, row : row + 1, :] @ value
    )
    native_timing = _timed(
        lambda: plan.execute(
            row_probabilities, row_values, row_indices, row_valid
        )
    )
    prepared_timing = _timed(
        lambda: plan.execute_prepared(row_probabilities, row_values)
    )
    return {
        "context_tokens": context,
        "rows": rows,
        "all_rows_byte_exact": all(item["byte_exact"] for item in rows),
        "all_lane_maps_exact": all(item["lane_map_exact"] for item in rows),
        "packed_k": plan.packed_k,
        "scratch_bytes": plan.scratch_bytes,
        "direct_timing": direct_timing,
        "native_timing": native_timing,
        "prepared_av_timing": prepared_timing,
        "operator_speedup": (
            direct_timing["median_wall_ms"] / native_timing["median_wall_ms"]
        ),
    }


def _artifact(contexts: dict[str, object]) -> dict[str, object]:
    complete = set(map(int, contexts)) == set(CONTEXTS)
    long = contexts.get(str(32_768), {})
    checks = {
        "2k_and_32k_measured": complete,
        "all_8_query_rows_byte_exact": complete and all(
            row["all_rows_byte_exact"] for row in contexts.values()
        ),
        "native_lane_map_matches_physical_bk16_contract": complete and all(
            row["all_lane_maps_exact"] for row in contexts.values()
        ),
        "fixed_arena_at_most_1mib": complete and all(
            row["scratch_bytes"] <= MAX_SCRATCH_BYTES
            for row in contexts.values()
        ),
        "no_dynamic_allocation_graph_shape_or_sync": complete and all(
            all(value == 0 for value in item["structural_counters"].values())
            for row in contexts.values()
            for item in row["rows"]
        ),
        "buffer_identities_stable": complete and all(
            item["buffer_identities_stable"]
            for row in contexts.values()
            for item in row["rows"]
        ),
        "32k_operator_speedup_at_least_1_20x": (
            complete and long["operator_speedup"] >= 1.20
        ),
    }
    accepted = all(checks.values())
    return {
        "schema": "glm53-native-virtual-bk16-prefill-av-v1",
        "date": date.today().isoformat(),
        "complete": complete,
        "accepted": accepted,
        "decision": (
            "advance_virtual_bk16_av_into_composed_native_prefill_region"
            if accepted
            else "stop_or_redesign_native_virtual_bk16_av"
        ),
        "contexts": contexts,
        "checks": checks,
        "runtime_changes": False,
        "production_admission_changed": False,
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
        if not contexts[str(context)]["all_rows_byte_exact"]:
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
