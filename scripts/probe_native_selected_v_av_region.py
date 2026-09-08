#!/usr/bin/env python3
"""Qualify a single-encoder selected-V-projection through exact AV region."""

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
from glm53_native_execution import NativeSelectedVProjectionAVPlan


DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-selected-v-projection-av-region-20260908.json"
)
CONTEXTS = (2_048, 32_768)
SAMPLES = 3
MAX_SCRATCH_BYTES = 40 << 20


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _inputs(context: int) -> dict[str, mx.array]:
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
    full_scores = (fixture["query"] * scale).astype(mx.bfloat16) @ key.swapaxes(
        -1, -2
    )
    full_scores = mx.where(mask, full_scores, mx.finfo(mx.bfloat16).min)
    full_probabilities = mx.softmax(full_scores, axis=-1, precise=True)
    reference = full_probabilities @ value

    selected_latent = mx.contiguous(
        equivalence._gather_per_query(
            latent, fixture["indices"], fixture["valid"]
        )
    )
    selected_key = selected_latent @ fixture["k_weight"]
    query = fixture["query"].transpose(2, 1, 0, 3)
    selected_scores = (query * scale).astype(mx.bfloat16) @ selected_key.swapaxes(
        -1, -2
    )
    selected_mask = fixture["valid"].reshape(
        equivalence.QUERY_ROWS, 1, 1, equivalence.SELECTED_WIDTH
    )
    selected_scores = mx.where(
        selected_mask, selected_scores, mx.finfo(mx.bfloat16).min
    )
    selected_probabilities = mx.contiguous(
        mx.softmax(selected_scores, axis=-1, precise=True)
    )
    indices = mx.contiguous(fixture["indices"])
    valid = mx.contiguous(fixture["valid"])
    mx.eval(
        full_probabilities,
        reference,
        selected_latent,
        selected_probabilities,
        indices,
        valid,
        fixture["v_weight"],
    )
    return {
        "fixture": fixture,
        "full_probabilities": full_probabilities,
        "reference": reference,
        "selected_latent": selected_latent,
        "selected_probabilities": selected_probabilities,
        "indices": indices,
        "valid": valid,
    }


def _native(plan, inputs: dict[str, mx.array]) -> mx.array:
    fixture = inputs["fixture"]
    output = plan.execute(
        inputs["selected_probabilities"],
        inputs["selected_latent"],
        fixture["v_weight"],
        inputs["indices"],
        inputs["valid"],
    )
    return output.transpose(2, 1, 0, 3)


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
    print(json.dumps({"phase": "native_selected_v_av", "context": context}), flush=True)
    inputs = _inputs(context)
    fixture = inputs["fixture"]
    plan = NativeSelectedVProjectionAVPlan(context)
    identities = list(plan.buffer_identities)
    candidate = _native(plan, inputs)
    mx.eval(candidate)
    different = int(mx.sum(candidate != inputs["reference"]).item())

    final_row_reference = (
        inputs["selected_latent"][equivalence.QUERY_ROWS - 1]
        @ fixture["v_weight"].swapaxes(-1, -2)
    )
    mx.eval(final_row_reference, plan.debug_projected_value)
    projection_different = int(
        mx.sum(final_row_reference != plan.debug_projected_value).item()
    )

    # Warm the stable plan, then compare only V projection plus AV. Scores and
    # probabilities are precomputed identically for both arms.
    mx.eval(_native(plan, inputs))
    direct_timing = _timed(
        lambda: inputs["full_probabilities"]
        @ (fixture["latent"] @ fixture["v_weight"].swapaxes(-1, -2))
    )
    native_timing = _timed(lambda: _native(plan, inputs))
    final_identities = list(plan.buffer_identities)
    row = {
        "context_tokens": context,
        "query_rows": equivalence.QUERY_ROWS,
        "different_output_elements": different,
        "byte_exact": different == 0,
        "projection_anchor_different_elements": projection_different,
        "projection_anchor_byte_exact": projection_different == 0,
        "direct_timing": direct_timing,
        "native_timing": native_timing,
        "speedup": (
            direct_timing["median_wall_ms"] / native_timing["median_wall_ms"]
        ),
        "scratch_bytes": plan.scratch_bytes,
        "buffer_identities_stable": identities == final_identities,
        "execution_count": plan.execution_count,
        "dynamic_allocation_count": plan.dynamic_allocation_count,
        "graph_node_count": plan.graph_node_count,
        "shape_discovery_count": plan.shape_discovery_count,
        "host_synchronization_count": plan.host_synchronization_count,
        "returned_intermediate_tensor_bytes": (
            plan.returned_intermediate_tensor_bytes
        ),
    }
    del candidate, final_row_reference, plan, inputs, fixture
    mx.clear_cache()
    gc.collect()
    return row


def _artifact(contexts: dict[str, object]) -> dict[str, object]:
    complete = set(map(int, contexts)) == set(CONTEXTS)
    long = contexts.get(str(32_768), {})
    checks = {
        "2k_and_32k_measured": complete,
        "all_attention_outputs_byte_exact": complete and all(
            row["byte_exact"] for row in contexts.values()
        ),
        "all_selected_v_projection_anchors_byte_exact": complete and all(
            row["projection_anchor_byte_exact"] for row in contexts.values()
        ),
        "single_encoder_scratch_at_most_40mib": complete and all(
            row["scratch_bytes"] <= MAX_SCRATCH_BYTES
            for row in contexts.values()
        ),
        "no_dynamic_allocation_graph_shape_sync_or_intermediate_return": (
            complete and all(
                row["dynamic_allocation_count"] == 0
                and row["graph_node_count"] == 0
                and row["shape_discovery_count"] == 0
                and row["host_synchronization_count"] == 0
                and row["returned_intermediate_tensor_bytes"] == 0
                for row in contexts.values()
            )
        ),
        "all_buffer_addresses_stable": complete and all(
            row["buffer_identities_stable"] for row in contexts.values()
        ),
        "32k_region_speedup_at_least_1_20x": (
            complete and long["speedup"] >= 1.20
        ),
    }
    accepted = all(checks.values())
    return {
        "schema": "glm53-native-selected-v-projection-av-region-v1",
        "date": date.today().isoformat(),
        "complete": complete,
        "accepted": accepted,
        "decision": (
            "advance_native_region_to_qk_softmax_composition"
            if accepted
            else "stop_or_redesign_native_selected_v_av_region"
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
        if not (
            contexts[str(context)]["byte_exact"]
            and contexts[str(context)]["projection_anchor_byte_exact"]
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
