#!/usr/bin/env python3
"""Measure 11 DSA + 42 MoE native stages behind one host boundary."""

from __future__ import annotations

import argparse
import gc
import hashlib
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

from glm53_flash_mlx.grouped_fp8 import SortedGroupedFP8MoE
from glm53_flash_mlx.loader import load
from glm53_native_execution import NativePrefillDominantRegionPlan
import probe_q256_projected_qk_320k_tiles as qk320
import probe_native_prefill_moe_routed_island as moe_probe


DEFAULT_OUTPUT = ROOT / "bench-results" / (
    "m3ultra512-native-prefill-dominant-region-aggregation-20260909.json"
)
PROFILE = ROOT / "bench-results" / (
    "m3ultra512-native-prefill-critical-path-20260908.json"
)
DSA_EVIDENCE = ROOT / "bench-results" / (
    "m3ultra512-composed-q256-native-dsa-prefill-320k-20260909.json"
)
LAYER_EVIDENCE = ROOT / "bench-results" / (
    "m3ultra512-native-final-hc-prefill-layer-execution-20260909.json"
)
TILE_ROWS = 65_536
VALUE_DIM = 256
MIN_HOST_SPEEDUP = 1.20
MIN_WALL_SPEEDUP = 1.02


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _value_weight() -> mx.array:
    value = mx.sin(
        mx.arange(
            qk320.HEADS * VALUE_DIM * qk320.LATENT_DIM,
            dtype=mx.float32,
        ).reshape(qk320.HEADS, VALUE_DIM, qk320.LATENT_DIM)
        * 0.000019
    ).astype(mx.bfloat16)
    value = mx.contiguous(value)
    mx.eval(value)
    return value


def _dsa_arguments(inputs, value_weight, projection):
    return (
        inputs["selected_indices"],
        inputs["selected_valid"],
        inputs["latent"],
        inputs["key_weight"],
        value_weight,
        inputs["query"],
        qk320.ATTENTION_SCALE,
        projection.weight,
        projection.weight_scale_inv,
    )


def _moe_arguments(moe, hidden, indices, scores):
    shared = moe.shared_experts
    return (
        hidden,
        mx.contiguous(indices.reshape(-1).astype(mx.uint32)),
        mx.contiguous(scores.reshape(-1).astype(mx.float32)),
        moe.bank.gate_up_weight,
        moe.bank.gate_up_scale_inv,
        moe.bank.down_weight,
        moe.bank.down_scale_inv,
        shared.gate_proj.weight,
        shared.gate_proj.weight_scale_inv,
        shared.up_proj.weight,
        shared.up_proj.weight_scale_inv,
        shared.down_proj.weight,
        shared.down_proj.weight_scale_inv,
    )


def _layerwise(plan, dsa_arguments, moe_arguments):
    dsa = None
    for _ in range(plan.dsa_layer_count):
        dsa = plan.execute_dsa(*dsa_arguments)
    moe = None
    for _ in range(plan.moe_layer_count):
        moe = plan.execute_moe(*moe_arguments)
    return [dsa, moe]


def _composed(plan, dsa_arguments, moe_arguments):
    return plan.execute_all(*dsa_arguments, *moe_arguments)


def _eval(outputs):
    mx.eval(*outputs)
    mx.synchronize()
    return outputs


def _bits(outputs):
    return [
        np.asarray(value.view(mx.uint16), dtype=np.uint16).copy()
        for value in outputs
    ]


def _timed(operation, samples: int) -> dict[str, object]:
    _eval(operation())
    host = []
    wall = []
    for _ in range(samples):
        mx.synchronize()
        started = time.perf_counter_ns()
        outputs = operation()
        submitted = time.perf_counter_ns()
        _eval(outputs)
        finished = time.perf_counter_ns()
        host.append((submitted - started) / 1e6)
        wall.append((finished - started) / 1e6)
    return {
        "median_host_submission_ms": statistics.median(host),
        "median_synchronized_wall_ms": statistics.median(wall),
        "host_samples_ms": host,
        "wall_samples_ms": wall,
    }


def _dominant_share() -> float:
    profile = json.loads(PROFILE.read_text())
    stages = profile["contexts"]["327680"]["instrumented"]["attribution"][
        "stages"
    ]
    return float(stages["dsa_attention"]["diagnostic_share"]) + float(
        stages["routed_moe"]["diagnostic_share"]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    for evidence in (PROFILE, DSA_EVIDENCE, LAYER_EVIDENCE):
        if not evidence.exists():
            raise FileNotFoundError(evidence)
    if args.samples < 2:
        raise ValueError("at least two samples are required")
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))

    print(json.dumps({"phase": "load_model"}), flush=True)
    model, _ = load(args.model, experimental_packed_grouped_moe=True)
    layer = model.language_model.model.layers[3]
    moe = layer.mlp
    if not isinstance(moe, SortedGroupedFP8MoE):
        raise RuntimeError("layer 3 is not packed grouped MoE")

    print(json.dumps({"phase": "build_320k_dsa_fixture"}), flush=True)
    inputs, _, _ = qk320._fixture()
    mx.eval(*inputs.values())
    value_weight = _value_weight()
    hidden = moe_probe._hidden()
    indices, scores = moe.gate(hidden)
    dsa_arguments = _dsa_arguments(
        inputs, value_weight, layer.self_attn.o_proj
    )
    moe_arguments = _moe_arguments(moe, hidden, indices, scores)
    mx.eval(*dsa_arguments[:6], *dsa_arguments[7:], *moe_arguments)

    plan = NativePrefillDominantRegionPlan(
        qk320.PHYSICAL_K, TILE_ROWS, moe.bank.expert_count
    )
    identities = list(plan.buffer_identities)
    print(json.dumps({"phase": "exact_53_call_topology"}), flush=True)
    layerwise = _eval(_layerwise(plan, dsa_arguments, moe_arguments))
    layerwise_bits = _bits(layerwise)
    print(json.dumps({"phase": "exact_single_call_topology"}), flush=True)
    composed = _eval(_composed(plan, dsa_arguments, moe_arguments))
    composed_bits = _bits(composed)

    print(json.dumps({"phase": "time_53_call_topology"}), flush=True)
    layerwise_timing = _timed(
        lambda: _layerwise(plan, dsa_arguments, moe_arguments), args.samples
    )
    print(json.dumps({"phase": "time_single_call_topology"}), flush=True)
    composed_timing = _timed(
        lambda: _composed(plan, dsa_arguments, moe_arguments), args.samples
    )
    host_speedup = (
        layerwise_timing["median_host_submission_ms"]
        / composed_timing["median_host_submission_ms"]
    )
    wall_speedup = (
        layerwise_timing["median_synchronized_wall_ms"]
        / composed_timing["median_synchronized_wall_ms"]
    )
    exact = all(
        np.array_equal(expected, actual)
        for expected, actual in zip(layerwise_bits, composed_bits, strict=True)
    )
    checks = {
        "identical_11_dsa_42_moe_terminal_anchors_are_byte_exact": exact,
        "dominant_regions_are_at_least_95_percent_of_diagnostic_wall": (
            _dominant_share() >= 0.95
        ),
        "python_native_boundaries_reduce_from_53_to_1": (
            plan.layerwise_native_calls == 53
            and plan.composed_native_calls == 1
        ),
        "host_submission_speedup_at_least_1_20x": (
            host_speedup >= MIN_HOST_SPEEDUP
        ),
        "same_arithmetic_wall_speedup_at_least_1_02x": (
            wall_speedup >= MIN_WALL_SPEEDUP
        ),
        "fixed_owned_arena_without_graph_allocation_or_sync": (
            identities == list(plan.buffer_identities)
            and plan.buffer_identities_stable
            and plan.dynamic_allocation_count == 0
            and plan.graph_node_count == 0
            and plan.shape_discovery_count == 0
            and plan.host_synchronization_count == 0
        ),
    }
    accepted = all(checks.values())
    evidence = {
        path.name: {"sha256": _sha256(path)}
        for path in (PROFILE, DSA_EVIDENCE, LAYER_EVIDENCE)
    }
    artifact = {
        "schema": "glm53-native-prefill-dominant-region-aggregation-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "advance_aggregated_regions_to_dependency_connected_45_layer_plan"
            if accepted
            else (
                "aggregation_only_is_insufficient_build_dependency_connected_plan"
                if exact
                else "stop_and_localize_dominant_region_aggregation_exactness"
            )
        ),
        "geometry": {
            "query_rows": 256,
            "physical_k": qk320.PHYSICAL_K,
            "dsa_layers": plan.dsa_layer_count,
            "moe_layers": plan.moe_layer_count,
            "layerwise_native_calls": plan.layerwise_native_calls,
            "composed_native_calls": plan.composed_native_calls,
        },
        "diagnostic_wall_share": _dominant_share(),
        "layerwise_timing": layerwise_timing,
        "composed_timing": composed_timing,
        "host_submission_speedup": host_speedup,
        "same_arithmetic_wall_speedup": wall_speedup,
        "scratch_bytes": plan.scratch_bytes,
        "returned_diagnostic_anchor_bytes": (
            plan.returned_diagnostic_anchor_bytes
        ),
        "checks": checks,
        "evidence": evidence,
        "scope": {
            "same_native_arithmetic_and_barriers": True,
            "python_native_call_aggregation": True,
            "input_dependent_layer_handoffs": False,
            "production_runtime_change": False,
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
        "same_arithmetic_wall_speedup": wall_speedup,
        "failed_gates": [name for name, passed in checks.items() if not passed],
    }))
    del model, layer, moe, plan, inputs, value_weight, hidden, indices, scores
    mx.clear_cache()
    gc.collect()
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
