#!/usr/bin/env python3
"""Qualify the complete Q4 selected-K/V native prefill attention region."""

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
    / "m3ultra512-native-selected-kv-attention-region-20260908.json"
)
CONTEXTS = (2_048, 32_768)
SAMPLES = 3
MAX_SCRATCH_BYTES = 192 << 20
ATTENTION_SCALE = equivalence.LATENT_DIM**-0.5


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _fixture_inputs(context: int) -> dict[str, mx.array]:
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
    reference = mx.fast.scaled_dot_product_attention(
        fixture["query"], key, value, scale=ATTENTION_SCALE, mask=mask
    )
    selected_latent = mx.contiguous(
        equivalence._gather_per_query(
            latent, fixture["indices"], fixture["valid"]
        )
    )
    indices = mx.contiguous(fixture["indices"])
    valid = mx.contiguous(fixture["valid"])
    mx.eval(
        reference,
        selected_latent,
        fixture["k_weight"],
        fixture["v_weight"],
        fixture["query"],
        indices,
        valid,
    )
    return {
        "fixture": fixture,
        "reference": reference,
        "selected_latent": selected_latent,
        "indices": indices,
        "valid": valid,
    }


def _native(plan, inputs: dict[str, mx.array]) -> mx.array:
    fixture = inputs["fixture"]
    output = plan.execute_attention(
        inputs["selected_latent"],
        fixture["k_weight"],
        fixture["v_weight"],
        fixture["query"],
        inputs["indices"],
        inputs["valid"],
        ATTENTION_SCALE,
    )
    return output.transpose(2, 1, 0, 3)


def _direct(inputs: dict[str, mx.array]) -> mx.array:
    fixture = inputs["fixture"]
    context = int(fixture["latent"].shape[2])
    key = fixture["latent"] @ fixture["k_weight"]
    value = fixture["latent"] @ fixture["v_weight"].swapaxes(-1, -2)
    safe = mx.where(fixture["valid"], fixture["indices"], context)
    mask = mx.zeros(
        (1, 1, equivalence.QUERY_ROWS, context + 1), dtype=mx.bool_
    )
    mask = mx.put_along_axis(mask, safe[:, None], mx.array(True), axis=-1)[
        ..., :context
    ]
    return mx.fast.scaled_dot_product_attention(
        fixture["query"], key, value, scale=ATTENTION_SCALE, mask=mask
    )


def _timed(operation) -> dict[str, object]:
    samples = []
    for _ in range(SAMPLES):
        started = time.perf_counter_ns()
        output = operation()
        mx.eval(output)
        mx.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return {"median_wall_ms": statistics.median(samples), "samples_ms": samples}


def _different(left: mx.array, right: mx.array) -> int:
    mx.eval(left, right)
    return int(mx.sum(left != right).item())


def _context_case(context: int) -> dict[str, object]:
    print(json.dumps({"phase": "native_selected_kv_attention", "context": context}), flush=True)
    inputs = _fixture_inputs(context)
    fixture = inputs["fixture"]
    plan = NativeSelectedVProjectionAVPlan(context, True)
    identities = list(plan.buffer_identities)
    candidate = _native(plan, inputs)
    mx.eval(candidate)

    row = equivalence.QUERY_ROWS - 1
    selected = inputs["selected_latent"][row]
    key_anchor = selected @ fixture["k_weight"]
    value_anchor = selected @ fixture["v_weight"].swapaxes(-1, -2)
    scale = mx.array(ATTENTION_SCALE, dtype=mx.bfloat16)
    query_anchor = (
        fixture["query"][:, :, row : row + 1, :] * scale
    ).astype(mx.bfloat16)
    score_anchor = query_anchor @ key_anchor.swapaxes(-1, -2)
    score_anchor = mx.where(
        inputs["valid"][:, row : row + 1],
        score_anchor,
        mx.finfo(mx.bfloat16).min,
    )
    probability_anchor = mx.softmax(score_anchor, axis=-1, precise=True)
    anchors = {
        "projected_key": _different(key_anchor, plan.debug_projected_key),
        "projected_value": _different(value_anchor, plan.debug_projected_value),
        "scaled_query": _different(
            query_anchor.reshape(equivalence.HEADS, equivalence.LATENT_DIM),
            plan.debug_scaled_query,
        ),
        "qk_scores": _different(
            score_anchor.reshape(
                equivalence.HEADS, equivalence.SELECTED_WIDTH
            ),
            plan.debug_attention_scores,
        ),
        "precise_softmax": _different(
            probability_anchor.reshape(
                equivalence.HEADS, equivalence.SELECTED_WIDTH
            ),
            plan.debug_attention_probabilities,
        ),
    }
    output_different = _different(candidate, inputs["reference"])

    mx.eval(_native(plan, inputs))
    direct_timing = _timed(lambda: _direct(inputs))
    native_timing = _timed(lambda: _native(plan, inputs))
    final_identities = list(plan.buffer_identities)
    result = {
        "context_tokens": context,
        "query_rows": equivalence.QUERY_ROWS,
        "different_output_elements": output_different,
        "byte_exact": output_different == 0,
        "anchor_different_elements": anchors,
        "all_anchors_byte_exact": all(value == 0 for value in anchors.values()),
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
    del (
        candidate, selected, key_anchor, value_anchor, query_anchor,
        score_anchor, probability_anchor, plan, inputs, fixture,
    )
    mx.clear_cache()
    gc.collect()
    return result


def _artifact(contexts: dict[str, object]) -> dict[str, object]:
    complete = set(map(int, contexts)) == set(CONTEXTS)
    long = contexts.get(str(32_768), {})
    checks = {
        "2k_and_32k_measured": complete,
        "all_projection_qk_softmax_and_output_anchors_byte_exact": (
            complete and all(
                row["byte_exact"] and row["all_anchors_byte_exact"]
                for row in contexts.values()
            )
        ),
        "complete_region_scratch_at_most_192mib": complete and all(
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
        "32k_complete_attention_speedup_at_least_1_50x": (
            complete and long["speedup"] >= 1.50
        ),
    }
    accepted = all(checks.values())
    return {
        "schema": "glm53-native-selected-kv-attention-region-v1",
        "date": date.today().isoformat(),
        "complete": complete,
        "accepted": accepted,
        "decision": (
            "advance_exact_dsa_region_to_score_selection_composition"
            if accepted
            else "stop_or_relocalize_native_selected_kv_attention_region"
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
            and contexts[str(context)]["all_anchors_byte_exact"]
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
