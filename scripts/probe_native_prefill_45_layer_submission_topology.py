#!/usr/bin/env python3
"""Measure the 45-layer native submission boundary without changing math."""

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

from glm53_native_execution import NativePrefill45LayerSubmissionPlan


DEFAULT_OUTPUT = ROOT / "bench-results" / (
    "m3ultra512-native-prefill-45-layer-submission-topology-20260909.json"
)
LAYER_COUNT = 45
MIN_HOST_SPEEDUP = 1.20
MAX_WALL_REGRESSION = 1.02


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _hidden() -> mx.array:
    value = mx.sin(
        mx.arange(256 * 4096, dtype=mx.float32).reshape(256, 4096) * 0.00031
    ).astype(mx.bfloat16)
    value = mx.contiguous(value)
    mx.eval(value)
    return value


def _layerwise(plan, hidden):
    current = hidden
    for layer in range(LAYER_COUNT):
        current = plan.execute_layer(current, layer)
    return current


def _timed(operation, samples: int) -> dict[str, object]:
    operation()
    mx.synchronize()
    host_ms = []
    wall_ms = []
    for _ in range(samples):
        mx.synchronize()
        started = time.perf_counter_ns()
        output = operation()
        submitted = time.perf_counter_ns()
        mx.eval(output)
        mx.synchronize()
        finished = time.perf_counter_ns()
        host_ms.append((submitted - started) / 1e6)
        wall_ms.append((finished - started) / 1e6)
    return {
        "median_host_submission_ms": statistics.median(host_ms),
        "median_synchronized_wall_ms": statistics.median(wall_ms),
        "host_samples_ms": host_ms,
        "wall_samples_ms": wall_ms,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    hidden = _hidden()
    plan = NativePrefill45LayerSubmissionPlan()
    identities = list(plan.buffer_identities)

    print(json.dumps({"phase": "layerwise_45_native_calls"}), flush=True)
    layerwise = _layerwise(plan, hidden)
    mx.eval(layerwise)
    mx.synchronize()
    layerwise_bits = np.asarray(layerwise.view(mx.uint16), dtype=np.uint16).copy()
    print(json.dumps({"phase": "single_native_call_45_layers"}), flush=True)
    composed = plan.execute_all(hidden)
    mx.eval(composed)
    mx.synchronize()
    composed_bits = np.asarray(composed.view(mx.uint16), dtype=np.uint16).copy()
    hidden_bits = np.asarray(hidden.view(mx.uint16), dtype=np.uint16)

    print(json.dumps({"phase": "timing"}), flush=True)
    layerwise_timing = _timed(lambda: _layerwise(plan, hidden), args.samples)
    composed_timing = _timed(lambda: plan.execute_all(hidden), args.samples)
    host_speedup = (
        layerwise_timing["median_host_submission_ms"]
        / composed_timing["median_host_submission_ms"]
    )
    wall_ratio = (
        composed_timing["median_synchronized_wall_ms"]
        / layerwise_timing["median_synchronized_wall_ms"]
    )

    attention_types = list(plan.attention_types)
    ffn_types = list(plan.ffn_types)
    checks = {
        "same_45_device_operations_are_byte_exact": (
            np.array_equal(layerwise_bits, composed_bits)
            and np.array_equal(composed_bits, hidden_bits)
        ),
        "official_45_layer_type_map_is_exact": (
            len(attention_types) == LAYER_COUNT
            and attention_types.count("kda") == 34
            and attention_types.count("dsa") == 11
            and all(attention_types[layer] == ("dsa" if layer % 4 == 3 else "kda")
                    for layer in range(LAYER_COUNT))
            and ffn_types[:3] == ["dense"] * 3
            and ffn_types[3:] == ["moe"] * 42
        ),
        "python_native_calls_reduce_from_45_to_1": (
            plan.native_calls_per_layerwise_execute == 45
            and plan.native_calls_per_all_execute == 1
        ),
        "host_submission_speedup_at_least_1_20x": host_speedup >= MIN_HOST_SPEEDUP,
        "same_kernel_topology_wall_does_not_regress_over_2_percent": (
            wall_ratio <= MAX_WALL_REGRESSION
        ),
        "fixed_two_buffer_arena": (
            plan.arena_bytes == 2 * 256 * 4096 * 2
            and identities == list(plan.buffer_identities)
            and plan.buffer_identities_stable
        ),
        "no_allocation_graph_shape_sync_or_intermediate_return": (
            plan.dynamic_allocation_count == 0
            and plan.graph_node_count == 0
            and plan.shape_discovery_count == 0
            and plan.host_synchronization_count == 0
            and plan.returned_intermediate_tensor_bytes == 0
        ),
    }
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-native-prefill-45-layer-submission-topology-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "replace_copy_slots_with_exact_kda_dsa_dense_moe_layer_encoders"
            if accepted
            else "stop_or_relocalize_45_layer_native_submission_topology"
        ),
        "layer_map": {
            "layers": LAYER_COUNT,
            "attention_types": attention_types,
            "ffn_types": ffn_types,
            "kda_layers": plan.kda_layer_count,
            "dsa_layers": plan.dsa_layer_count,
            "dense_ffn_layers": plan.dense_ffn_layer_count,
            "moe_layers": plan.moe_layer_count,
        },
        "layerwise_timing": layerwise_timing,
        "single_call_timing": composed_timing,
        "host_submission_speedup": host_speedup,
        "single_call_over_layerwise_wall_ratio": wall_ratio,
        "checks": checks,
        "arithmetic_coverage": {
            "pass_through_only": True,
            "kda": False,
            "dsa": False,
            "dense_ffn": False,
            "moe": False,
            "final_norm_lm_head": False,
        },
        "qualified_components_ready_for_slot_replacement": {
            "actual_dsa_qk_softmax_av_v256": True,
            "actual_dsa_output_projection_post_attention_hc": True,
            "exact_indirect_packed_moe_final_hc": True,
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
        "host_submission_speedup": host_speedup,
        "wall_ratio": wall_ratio,
        "failed_gates": [name for name, passed in checks.items() if not passed],
    }))
    del hidden, plan, layerwise, composed
    mx.clear_cache()
    gc.collect()
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
