#!/usr/bin/env python3
"""Qualify the complete exact Q256 native DSA prefill island at 320K."""

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
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from glm53_native_execution import (
    NativeProjectedQKUnionTileLoopPlan,
    NativeQ256DSAPrefillPlan,
    NativeSharedPhysicalValueTilePlan,
)
import probe_q256_projected_qk_320k_tiles as qk320


DEFAULT_OUTPUT = ROOT / "bench-results" / (
    "m3ultra512-composed-q256-native-dsa-prefill-320k-20260909.json"
)
VALUE_DIM = 128
TILE_ROWS = 65536
MAX_SCRATCH = 6 << 30
MIN_SPEEDUP = 1.20


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _bits(value: mx.array) -> np.ndarray:
    return np.asarray(value.view(mx.uint16), dtype=np.uint16)


def _value_weight() -> mx.array:
    weight = mx.sin(
        mx.arange(
            qk320.HEADS * VALUE_DIM * qk320.LATENT_DIM, dtype=mx.float32
        ).reshape(qk320.HEADS, VALUE_DIM, qk320.LATENT_DIM)
        * 0.000019
    ).astype(mx.bfloat16)
    weight = mx.contiguous(weight)
    mx.eval(weight)
    return weight


def _qk(plan, inputs):
    probabilities = plan.execute_probabilities(
        inputs["selected_indices"], inputs["selected_valid"], inputs["latent"],
        inputs["key_weight"], inputs["query"], qk320.ATTENTION_SCALE,
    )
    mx.eval(probabilities)
    mx.synchronize()
    return probabilities


def _four_bm64(plan, probabilities, inputs, value_weight):
    blocks = []
    for block in range(4):
        start = block * 64
        stop = start + 64
        block_probabilities = mx.contiguous(probabilities[start:stop])
        block_indices = mx.contiguous(inputs["selected_indices"][start:stop])
        block_valid = mx.contiguous(inputs["selected_valid"][start:stop])
        mx.eval(block_probabilities, block_indices, block_valid)
        output = plan.execute(
            block_probabilities, block_indices, block_valid,
            inputs["latent"], value_weight,
        )
        # The plan reuses one output arena. Preserve each completed BM64 block
        # before the next execution overwrites that arena.
        copied = mx.array(output)
        mx.eval(copied)
        blocks.append(copied)
    output = mx.concatenate(blocks, axis=1)
    mx.eval(output)
    mx.synchronize()
    return output


def _reference(qk_plan, value_plan, inputs, value_weight):
    probabilities = _qk(qk_plan, inputs)
    return probabilities, _four_bm64(
        value_plan, probabilities, inputs, value_weight
    )


def _candidate(plan, inputs, value_weight):
    output = plan.execute(
        inputs["selected_indices"], inputs["selected_valid"], inputs["latent"],
        inputs["key_weight"], value_weight, inputs["query"],
        qk320.ATTENTION_SCALE,
    )
    mx.eval(output)
    mx.synchronize()
    return output


def _timed(operation, samples: int = 3):
    operation()
    values = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        operation()
        values.append((time.perf_counter_ns() - started) / 1e6)
    return {"median_wall_ms": statistics.median(values), "samples_ms": values}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    print(json.dumps({"phase": "build_320k_q256_fixture"}), flush=True)
    inputs, host_indices, host_valid = qk320._fixture()
    # The imported fixture returns contiguous wrappers after evaluating their
    # sources. Schedule those wrappers explicitly before entering the strict
    # native ABI, which correctly rejects unscheduled inputs.
    mx.eval(*inputs.values())
    value_weight = _value_weight()
    expected_union_count = int(np.unique(host_indices[host_valid]).size)
    reference_qk = NativeProjectedQKUnionTileLoopPlan(
        qk320.PHYSICAL_K, 256, TILE_ROWS, True, False
    )
    reference_value = NativeSharedPhysicalValueTilePlan(
        qk320.PHYSICAL_K, TILE_ROWS, 64
    )
    candidate_plan = NativeQ256DSAPrefillPlan(qk320.PHYSICAL_K, TILE_ROWS)
    identities = list(candidate_plan.buffer_identities)

    print(json.dumps({"phase": "reference_qk_softmax_four_bm64"}), flush=True)
    expected_probabilities, expected_output = _reference(
        reference_qk, reference_value, inputs, value_weight
    )
    print(json.dumps({"phase": "composed_q256_native_dsa"}), flush=True)
    actual_output = _candidate(candidate_plan, inputs, value_weight)
    actual_probabilities = candidate_plan.debug_probabilities
    mx.eval(actual_probabilities)
    union_count = int(np.asarray(candidate_plan.debug_union_count, dtype=np.uint32)[0])
    anchors = {
        "union_count": union_count == expected_union_count,
        "qk_precise_probabilities": np.array_equal(
            _bits(actual_probabilities), _bits(expected_probabilities)
        ),
        "q256_attention_output": np.array_equal(
            _bits(actual_output), _bits(expected_output)
        ),
    }

    print(json.dumps({"phase": "timing_reference"}), flush=True)
    reference_timing = _timed(
        lambda: _reference(reference_qk, reference_value, inputs, value_weight)[1]
    )
    print(json.dumps({"phase": "timing_composed"}), flush=True)
    candidate_timing = _timed(
        lambda: _candidate(candidate_plan, inputs, value_weight)
    )
    speedup = reference_timing["median_wall_ms"] / candidate_timing["median_wall_ms"]
    checks = {
        "320k_qk_softmax_q256_attention_byte_exact": all(anchors.values()),
        "single_fixed_native_execution_scope": (
            identities == list(candidate_plan.buffer_identities)
            and candidate_plan.dynamic_allocation_count == 0
            and candidate_plan.graph_node_count == 0
            and candidate_plan.shape_discovery_count == 0
            and candidate_plan.host_synchronization_count == 0
            and candidate_plan.returned_intermediate_tensor_bytes == 0
        ),
        "selected_k_and_query_local_selected_v_never_materialized": (
            candidate_plan.materialized_selected_key_bytes == 0
            and candidate_plan.materialized_query_local_selected_value_bytes == 0
        ),
        "scratch_at_most_6gib": candidate_plan.scratch_bytes <= MAX_SCRATCH,
        "composed_320k_speedup_at_least_1_20x": speedup >= MIN_SPEEDUP,
    }
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-composed-q256-native-dsa-prefill-320k-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "advance_composed_native_dsa_into_prefill_layer_execution_plan"
            if accepted
            else "stop_or_relocalize_composed_q256_native_dsa_prefill"
        ),
        "geometry": {
            "physical_k": qk320.PHYSICAL_K,
            "tile_rows": TILE_ROWS,
            "tile_count": candidate_plan.tile_count,
            "query_rows": 256,
            "query_blocks": 4,
            "union_count": union_count,
        },
        "anchors": anchors,
        "reference_timing": reference_timing,
        "candidate_timing": candidate_timing,
        "speedup": speedup,
        "scratch_bytes": candidate_plan.scratch_bytes,
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
    del reference_qk, reference_value, candidate_plan
    del inputs, value_weight, expected_probabilities, expected_output, actual_output
    mx.clear_cache()
    gc.collect()
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
