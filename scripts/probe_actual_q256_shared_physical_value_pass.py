#!/usr/bin/env python3
"""Qualify the shared physical-value pass at GLM-5.3's actual V=256."""

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
for path in (ROOT / "native_execution", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from glm53_native_execution import NativeSharedPhysicalValueTilePlan
import probe_exact_q256_shared_physical_value_pass as q256


DEFAULT_OUTPUT = ROOT / "bench-results" / (
    "m3ultra512-actual-q256-shared-physical-value-pass-20260909.json"
)
VALUE_DIM = 256
HALF_VALUE_DIM = 128
MIN_COMPOSITION_SPEEDUP = 1.05


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _bits(value: mx.array) -> np.ndarray:
    return np.asarray(value.view(mx.uint16), dtype=np.uint16)


def _value_weight() -> mx.array:
    value = mx.sin(
        mx.arange(
            q256.HEADS * VALUE_DIM * q256.LATENT_DIM, dtype=mx.float32
        ).reshape(q256.HEADS, VALUE_DIM, q256.LATENT_DIM)
        * 0.000019
    ).astype(mx.bfloat16)
    value = mx.contiguous(value)
    mx.eval(value)
    return value


def _execute(plan, inputs, weight) -> mx.array:
    return plan.execute(
        inputs["probabilities"],
        inputs["indices"],
        inputs["valid"],
        inputs["latent"],
        weight,
    )


def _reference(plans, inputs, weights) -> mx.array:
    halves = [_execute(plan, inputs, weight) for plan, weight in zip(plans, weights)]
    output = mx.concatenate(halves, axis=-1)
    mx.eval(output)
    mx.synchronize()
    return output


def _candidate(plan, inputs, weight) -> mx.array:
    output = _execute(plan, inputs, weight)
    mx.eval(output)
    mx.synchronize()
    return output


def _timed(operation, samples: int) -> dict[str, object]:
    operation()
    values = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        operation()
        values.append((time.perf_counter_ns() - started) / 1e6)
    return {"median_wall_ms": statistics.median(values), "samples_ms": values}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    print(json.dumps({"phase": "build_actual_v256_fixture"}), flush=True)
    inputs = q256._fixture()
    weight = _value_weight()
    weights = [
        mx.contiguous(weight[:, :HALF_VALUE_DIM]),
        mx.contiguous(weight[:, HALF_VALUE_DIM:]),
    ]
    mx.eval(*inputs.values(), *weights)

    reference_plans = [
        NativeSharedPhysicalValueTilePlan(
            q256.PHYSICAL_K, q256.TILE_ROWS, q256.QUERY_ROWS, HALF_VALUE_DIM
        )
        for _ in range(2)
    ]
    candidate_plan = NativeSharedPhysicalValueTilePlan(
        q256.PHYSICAL_K, q256.TILE_ROWS, q256.QUERY_ROWS, VALUE_DIM
    )
    identities = list(candidate_plan.buffer_identities)

    print(json.dumps({"phase": "two_v128_oracle"}), flush=True)
    expected = _reference(reference_plans, inputs, weights)
    expected_bits = _bits(expected).copy()
    print(json.dumps({"phase": "single_actual_v256_pass"}), flush=True)
    actual = _candidate(candidate_plan, inputs, weight)
    actual_bits = _bits(actual).copy()
    repeat = _candidate(candidate_plan, inputs, weight)
    repeat_bits = _bits(repeat).copy()

    print(json.dumps({"phase": "timing"}), flush=True)
    reference_timing = _timed(
        lambda: _reference(reference_plans, inputs, weights), args.samples
    )
    candidate_timing = _timed(
        lambda: _candidate(candidate_plan, inputs, weight), args.samples
    )
    speedup = reference_timing["median_wall_ms"] / candidate_timing["median_wall_ms"]
    checks = {
        "actual_v256_matches_two_exact_v128_halves_byte_exact": np.array_equal(
            expected_bits, actual_bits
        ),
        "repeat_is_byte_exact": np.array_equal(actual_bits, repeat_bits),
        "actual_glm53_value_geometry": (
            candidate_plan.value_dim == VALUE_DIM
            and tuple(actual.shape) == (q256.HEADS, q256.QUERY_ROWS, VALUE_DIM)
        ),
        "one_shared_probability_scatter_for_all_four_bn64_value_blocks": (
            candidate_plan.query_blocks == 4 and candidate_plan.value_dim // 64 == 4
        ),
        "fixed_owned_native_scope": (
            identities == list(candidate_plan.buffer_identities)
            and candidate_plan.dynamic_allocation_count == 0
            and candidate_plan.graph_node_count == 0
            and candidate_plan.shape_discovery_count == 0
            and candidate_plan.host_synchronization_count == 0
            and candidate_plan.returned_intermediate_tensor_bytes == 0
        ),
        "composition_speedup_at_least_1_05x": speedup >= MIN_COMPOSITION_SPEEDUP,
    }
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-actual-q256-shared-physical-value-pass-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "advance_actual_dsa_attention_into_45_layer_native_prefill_plan"
            if accepted
            else "stop_or_relocalize_actual_v256_physical_value_pass"
        ),
        "geometry": {
            "physical_k": q256.PHYSICAL_K,
            "tile_rows": q256.TILE_ROWS,
            "query_rows": q256.QUERY_ROWS,
            "heads": q256.HEADS,
            "latent_dim": q256.LATENT_DIM,
            "value_dim": VALUE_DIM,
            "value_bn64_blocks": VALUE_DIM // 64,
        },
        "reference_timing": reference_timing,
        "candidate_timing": candidate_timing,
        "speedup": speedup,
        "scratch_bytes": candidate_plan.scratch_bytes,
        "checks": checks,
        "probe_only": True,
        "runtime_changes": False,
    }
    _write(args.output, artifact)
    print(json.dumps({
        "output": str(args.output),
        "complete": True,
        "accepted": accepted,
        "decision": artifact["decision"],
        "speedup": speedup,
        "failed_gates": [name for name, passed in checks.items() if not passed],
    }))
    del inputs, weight, weights, reference_plans, candidate_plan
    del expected, actual, repeat
    mx.clear_cache()
    gc.collect()
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
