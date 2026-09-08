#!/usr/bin/env python3
"""Measure the exact selected-projection through virtual-BK16 AV region.

The standalone native AV boundary is slower than Direct Steel GEMM.  This
probe asks the architectural question that matters for prefill: whether
gathering latent rows, projecting only a Q4 microtile's selected K/V, and
finishing with the exact virtual-BK16 reduction still beats Direct full-K/V
projection plus sparse attention.  The candidate deliberately retains the
current MLX/native materialization boundary, so its timing is a conservative
headroom gate rather than a production claim.
"""

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


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
NATIVE_PACKAGE = ROOT / "native_execution"
for path in (SCRIPTS, NATIVE_PACKAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import probe_sparse_prefill_reordering_equivalence as equivalence
from glm53_native_execution import NativeSparsePrefillAVPlan


DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-composed-selected-projection-virtual-bk16-av-20260908.json"
)
CONTEXTS = (2_048, 32_768)
SAMPLES = 3


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _direct(fixture: dict[str, mx.array], context: int) -> mx.array:
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
    return mx.fast.scaled_dot_product_attention(
        fixture["query"], key, value,
        scale=equivalence.LATENT_DIM**-0.5,
        mask=mask,
    )


def _candidate_inputs(fixture: dict[str, mx.array]) -> tuple[mx.array, mx.array]:
    gathered = equivalence._gather_per_query(
        fixture["latent"], fixture["indices"], fixture["valid"]
    )
    key = gathered @ fixture["k_weight"]
    value = gathered @ fixture["v_weight"].swapaxes(-1, -2)
    query = fixture["query"].transpose(2, 1, 0, 3)
    scale = mx.array(equivalence.LATENT_DIM**-0.5, dtype=mx.bfloat16)
    scores = (query * scale).astype(mx.bfloat16) @ key.swapaxes(-1, -2)
    mask = fixture["valid"].reshape(
        equivalence.QUERY_ROWS, 1, 1, equivalence.SELECTED_WIDTH
    )
    scores = mx.where(mask, scores, mx.finfo(mx.bfloat16).min)
    probabilities = mx.softmax(scores, axis=-1, precise=True)
    return probabilities, value


def _candidate(
    fixture: dict[str, mx.array], plans: list[NativeSparsePrefillAVPlan]
) -> tuple[mx.array, mx.array, mx.array]:
    probabilities, value = _candidate_inputs(fixture)
    row_probabilities = [
        mx.contiguous(probabilities[row])
        for row in range(equivalence.QUERY_ROWS)
    ]
    row_values = [
        mx.contiguous(value[row]) for row in range(equivalence.QUERY_ROWS)
    ]
    row_indices = [
        mx.contiguous(fixture["indices"][0, row])
        for row in range(equivalence.QUERY_ROWS)
    ]
    row_valid = [
        mx.contiguous(fixture["valid"][0, row])
        for row in range(equivalence.QUERY_ROWS)
    ]
    # Native inputs must already be scheduled. This explicit boundary is
    # intentionally included in the conservative composed timing.
    mx.eval(*row_probabilities, *row_values, *row_indices, *row_valid)
    outputs = [
        plans[row].execute(
            row_probabilities[row], row_values[row],
            row_indices[row], row_valid[row],
        )
        for row in range(equivalence.QUERY_ROWS)
    ]
    return mx.concatenate(outputs, axis=2), probabilities, value


def _timed(operation) -> dict[str, object]:
    samples = []
    for _ in range(SAMPLES):
        started = time.perf_counter_ns()
        output = operation()
        mx.eval(output)
        mx.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return {"median_wall_ms": statistics.median(samples), "samples_ms": samples}


def _context_case(context: int) -> dict[str, object]:
    print(json.dumps({"phase": "composed_q4_region", "context": context}), flush=True)
    fixture = equivalence._fixture(context)
    plans = [
        NativeSparsePrefillAVPlan(context)
        for _ in range(equivalence.QUERY_ROWS)
    ]
    direct = _direct(fixture, context)
    candidate, probabilities, selected_values = _candidate(fixture, plans)
    mx.eval(direct, candidate, probabilities, selected_values)
    different = int(mx.sum(direct != candidate).item())
    identities = [list(plan.buffer_identities) for plan in plans]

    # Warm the stable plans before timing.
    warm, _, _ = _candidate(fixture, plans)
    mx.eval(warm)
    direct_timing = _timed(lambda: _direct(fixture, context))
    candidate_timing = _timed(lambda: _candidate(fixture, plans)[0])
    final_identities = [list(plan.buffer_identities) for plan in plans]
    row = {
        "context_tokens": context,
        "query_rows": equivalence.QUERY_ROWS,
        "selected_width": equivalence.SELECTED_WIDTH,
        "different_output_elements": different,
        "byte_exact": different == 0,
        "direct_timing": direct_timing,
        "candidate_timing": candidate_timing,
        "speedup": (
            direct_timing["median_wall_ms"]
            / candidate_timing["median_wall_ms"]
        ),
        "native_scratch_bytes": sum(plan.scratch_bytes for plan in plans),
        "buffer_identities_stable": identities == final_identities,
        "dynamic_allocation_count": sum(
            plan.dynamic_allocation_count for plan in plans
        ),
        "graph_node_count": sum(plan.graph_node_count for plan in plans),
        "shape_discovery_count": sum(
            plan.shape_discovery_count for plan in plans
        ),
        "host_synchronization_count": sum(
            plan.host_synchronization_count for plan in plans
        ),
        "candidate_has_explicit_mlx_native_boundary": True,
    }
    del direct, candidate, probabilities, selected_values, warm, plans, fixture
    mx.clear_cache()
    gc.collect()
    return row


def _artifact(contexts: dict[str, object]) -> dict[str, object]:
    complete = set(map(int, contexts)) == set(CONTEXTS)
    long = contexts.get(str(32_768), {})
    checks = {
        "2k_and_32k_q4_regions_measured": complete,
        "all_outputs_byte_exact": complete and all(
            row["byte_exact"] for row in contexts.values()
        ),
        "native_buffers_stable": complete and all(
            row["buffer_identities_stable"] for row in contexts.values()
        ),
        "native_plan_has_no_dynamic_allocation_or_shape_discovery": (
            complete and all(
                row["dynamic_allocation_count"] == 0
                and row["graph_node_count"] == 0
                and row["shape_discovery_count"] == 0
                and row["host_synchronization_count"] == 0
                for row in contexts.values()
            )
        ),
        "32k_composed_region_speedup_at_least_1_20x": (
            complete and long["speedup"] >= 1.20
        ),
    }
    accepted = all(checks.values())
    return {
        "schema": "glm53-composed-selected-projection-virtual-bk16-av-v1",
        "date": date.today().isoformat(),
        "complete": complete,
        "accepted": accepted,
        "decision": (
            "implement_single_encoder_selected_projection_bk16_av_region"
            if accepted
            else "stop_or_redesign_selected_projection_bk16_av_region"
        ),
        "contexts": contexts,
        "checks": checks,
        "timing_is_conservative_boundary_inclusive_headroom": True,
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
        if not contexts[str(context)]["byte_exact"]:
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
