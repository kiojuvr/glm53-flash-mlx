#!/usr/bin/env python3
"""Close the actual V=256 QK/softmax/AV native DSA prefill island."""

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

from glm53_native_execution import (
    NativeProjectedQKUnionTileLoopPlan,
    NativeQ256DSAPrefillPlan,
    NativeSharedPhysicalValueTilePlan,
)
import probe_exact_q256_shared_physical_value_pass as q256


DEFAULT_OUTPUT = ROOT / "bench-results" / (
    "m3ultra512-actual-q256-native-dsa-prefill-island-20260909.json"
)
VALUE_DIM = 256
ATTENTION_SCALE = q256.LATENT_DIM**-0.5


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _bits(value: mx.array) -> np.ndarray:
    return np.asarray(value.view(mx.uint16), dtype=np.uint16)


def _fixture() -> dict[str, mx.array]:
    inputs = q256._fixture()
    inputs.pop("value_weight")
    inputs["key_weight"] = mx.contiguous(
        mx.sin(
            mx.arange(
                q256.HEADS * q256.LATENT_DIM * q256.LATENT_DIM,
                dtype=mx.float32,
            ).reshape(q256.HEADS, q256.LATENT_DIM, q256.LATENT_DIM)
            * 0.000013
        ).astype(mx.bfloat16)
    )
    inputs["value_weight"] = mx.contiguous(
        mx.sin(
            mx.arange(
                q256.HEADS * VALUE_DIM * q256.LATENT_DIM, dtype=mx.float32
            ).reshape(q256.HEADS, VALUE_DIM, q256.LATENT_DIM)
            * 0.000019
        ).astype(mx.bfloat16)
    )
    inputs["query"] = mx.contiguous(
        mx.sin(
            mx.arange(
                q256.HEADS * q256.QUERY_ROWS * q256.LATENT_DIM,
                dtype=mx.float32,
            ).reshape(1, q256.HEADS, q256.QUERY_ROWS, q256.LATENT_DIM)
            * 0.00031
        ).astype(mx.bfloat16)
    )
    mx.eval(*inputs.values())
    return inputs


def _reference(qk, value, inputs) -> tuple[mx.array, mx.array]:
    probabilities = qk.execute_probabilities(
        inputs["indices"], inputs["valid"], inputs["latent"],
        inputs["key_weight"], inputs["query"], ATTENTION_SCALE,
    )
    output = value.execute(
        probabilities, inputs["indices"], inputs["valid"], inputs["latent"],
        inputs["value_weight"],
    )
    mx.eval(probabilities, output)
    mx.synchronize()
    return probabilities, output


def _candidate(plan, inputs) -> mx.array:
    output = plan.execute(
        inputs["indices"], inputs["valid"], inputs["latent"],
        inputs["key_weight"], inputs["value_weight"], inputs["query"],
        ATTENTION_SCALE,
    )
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

    print(json.dumps({"phase": "build_actual_dsa_fixture"}), flush=True)
    inputs = _fixture()
    qk = NativeProjectedQKUnionTileLoopPlan(
        q256.PHYSICAL_K, q256.QUERY_ROWS, q256.TILE_ROWS, True, False
    )
    value = NativeSharedPhysicalValueTilePlan(
        q256.PHYSICAL_K, q256.TILE_ROWS, q256.QUERY_ROWS, VALUE_DIM
    )
    candidate = NativeQ256DSAPrefillPlan(
        q256.PHYSICAL_K, q256.TILE_ROWS, VALUE_DIM
    )
    identities = list(candidate.buffer_identities)

    print(json.dumps({"phase": "split_native_oracle"}), flush=True)
    expected_probabilities, expected = _reference(qk, value, inputs)
    expected_bits = _bits(expected).copy()
    probability_bits = _bits(expected_probabilities).copy()
    print(json.dumps({"phase": "single_scope_actual_dsa"}), flush=True)
    actual = _candidate(candidate, inputs)
    actual_bits = _bits(actual).copy()
    actual_probability_bits = _bits(candidate.debug_probabilities).copy()
    repeat_bits = _bits(_candidate(candidate, inputs)).copy()

    print(json.dumps({"phase": "timing"}), flush=True)
    reference_timing = _timed(lambda: _reference(qk, value, inputs), args.samples)
    candidate_timing = _timed(lambda: _candidate(candidate, inputs), args.samples)
    ratio = reference_timing["median_wall_ms"] / candidate_timing["median_wall_ms"]
    checks = {
        "actual_qk_precise_softmax_is_byte_exact": np.array_equal(
            probability_bits, actual_probability_bits
        ),
        "actual_v256_attention_output_is_byte_exact": np.array_equal(
            expected_bits, actual_bits
        ),
        "repeat_is_byte_exact": np.array_equal(actual_bits, repeat_bits),
        "single_native_scope_returns_only_actual_attention_output": (
            candidate.value_dim == VALUE_DIM
            and candidate.returned_intermediate_tensor_bytes == 0
            and candidate.materialized_selected_key_bytes == 0
            and candidate.materialized_query_local_selected_value_bytes == 0
        ),
        "fixed_owned_buffers": (
            identities == list(candidate.buffer_identities)
            and candidate.dynamic_allocation_count == 0
            and candidate.graph_node_count == 0
            and candidate.shape_discovery_count == 0
            and candidate.host_synchronization_count == 0
        ),
        "composition_does_not_regress_more_than_2_percent": ratio >= 0.98,
    }
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-actual-q256-native-dsa-prefill-island-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "close_dsa_body_and_advance_45_layer_native_prefill_plan"
            if accepted
            else "stop_or_relocalize_actual_q256_native_dsa_island"
        ),
        "geometry": {
            "physical_k": q256.PHYSICAL_K,
            "tile_rows": q256.TILE_ROWS,
            "query_rows": q256.QUERY_ROWS,
            "heads": q256.HEADS,
            "qk_latent_dim": q256.LATENT_DIM,
            "value_dim": VALUE_DIM,
        },
        "reference_timing": reference_timing,
        "candidate_timing": candidate_timing,
        "composition_speedup": ratio,
        "scratch_bytes": candidate.scratch_bytes,
        "checks": checks,
        "coverage": {
            "selected_union": True,
            "key_projection": True,
            "query_scale": True,
            "qk_scores": True,
            "precise_softmax": True,
            "physical_value_projection_v256": True,
            "bm64_av": True,
            "output_projection": False,
            "post_attention_hc": False,
        },
        "probe_only": True,
        "runtime_changes": False,
    }
    _write(args.output, artifact)
    print(json.dumps({
        "output": str(args.output),
        "complete": True,
        "accepted": accepted,
        "decision": artifact["decision"],
        "composition_speedup": ratio,
        "failed_gates": [name for name, passed in checks.items() if not passed],
    }))
    del inputs, qk, value, candidate, expected_probabilities, expected, actual
    mx.clear_cache()
    gc.collect()
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
