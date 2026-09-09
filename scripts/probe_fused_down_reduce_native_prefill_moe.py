#!/usr/bin/env python3
"""Probe the last admissible structural rewrite of Q256 native prefill MoE.

The candidate fuses routed down projection with Direct-order weighted BF16
reduction.  It removes the 2048x4096 routed-down surface and one dispatch;
route, gate/up/SwiGLU, shared expert, and final add remain unchanged.
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
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "native_execution", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from glm53_native_execution import NativePrefillMoEPlan
from glm53_flash_mlx.grouped_fp8 import SortedGroupedFP8MoE
from glm53_flash_mlx.loader import load
import probe_native_prefill_full_moe_island as full_probe
import probe_native_prefill_moe_routed_island as routed_probe


DEFAULT_OUTPUT = ROOT / "bench-results" / (
    "m3ultra512-fused-down-reduce-native-prefill-moe-20260909.json"
)
MIN_SPEEDUP = 1.20
MIN_INCREMENTAL_SAVING_MS = 0.50


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _run(plan, moe, hidden, indices, scores, *, fused: bool):
    shared = moe.shared_experts
    execute = (
        plan.execute_fused_down_reduce if fused else plan.execute
    )
    output = execute(
        hidden, indices, scores,
        moe.bank.gate_up_weight, moe.bank.gate_up_scale_inv,
        moe.bank.down_weight, moe.bank.down_scale_inv,
        shared.gate_proj.weight, shared.gate_proj.weight_scale_inv,
        shared.up_proj.weight, shared.up_proj.weight_scale_inv,
        shared.down_proj.weight, shared.down_proj.weight_scale_inv,
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


def _exact(left, right) -> bool:
    return np.array_equal(
        np.asarray(left.view(mx.uint16), dtype=np.uint16),
        np.asarray(right.view(mx.uint16), dtype=np.uint16),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    print(json.dumps({"phase": "load_model"}), flush=True)
    model, _ = load(args.model, experimental_packed_grouped_moe=True)
    moe = model.language_model.model.layers[args.layer].mlp
    if not isinstance(moe, SortedGroupedFP8MoE) or moe.shared_experts is None:
        raise RuntimeError("target layer does not expose packed and shared experts")
    hidden = routed_probe._hidden()
    expert_ids, scores = moe.gate(hidden)
    expert_ids = mx.contiguous(expert_ids.astype(mx.uint32).reshape(-1))
    scores = mx.contiguous(scores.astype(mx.float32).reshape(-1))
    shared = moe.shared_experts
    weights = (
        moe.bank.gate_up_weight, moe.bank.gate_up_scale_inv,
        moe.bank.down_weight, moe.bank.down_scale_inv,
        shared.gate_proj.weight, shared.gate_proj.weight_scale_inv,
        shared.up_proj.weight, shared.up_proj.weight_scale_inv,
        shared.down_proj.weight, shared.down_proj.weight_scale_inv,
    )
    mx.eval(hidden, expert_ids, scores, *weights)
    plan = NativePrefillMoEPlan(moe.bank.expert_count)
    identities = list(plan.buffer_identities)

    print(json.dumps({"phase": "exact_direct_full_moe"}), flush=True)
    matrix_ids = expert_ids.reshape(256, 8)
    matrix_scores = scores.reshape(256, 8)
    reference = full_probe._reference(moe, hidden, matrix_ids, matrix_scores)
    print(json.dumps({"phase": "materialized_native"}), flush=True)
    materialized = _run(plan, moe, hidden, expert_ids, scores, fused=False)
    print(json.dumps({"phase": "fused_down_reduce_native"}), flush=True)
    fused = _run(plan, moe, hidden, expert_ids, scores, fused=True)
    repeat = _run(plan, moe, hidden, expert_ids, scores, fused=True)

    print(json.dumps({"phase": "timing"}), flush=True)
    reference_timing = _timed(
        lambda: full_probe._reference(
            moe, hidden, matrix_ids, matrix_scores
        ),
        args.samples,
    )
    materialized_timing = _timed(
        lambda: _run(plan, moe, hidden, expert_ids, scores, fused=False),
        args.samples,
    )
    fused_timing = _timed(
        lambda: _run(plan, moe, hidden, expert_ids, scores, fused=True),
        args.samples,
    )
    fused_ms = fused_timing["median_wall_ms"]
    speedup = reference_timing["median_wall_ms"] / fused_ms
    incremental = materialized_timing["median_wall_ms"] - fused_ms
    checks = {
        "materialized_native_remains_byte_exact": _exact(reference, materialized),
        "fused_down_reduce_full_moe_is_byte_exact": _exact(reference, fused),
        "repeat_is_byte_exact": _exact(fused, repeat),
        "fused_path_does_not_materialize_routed_down": (
            plan.fused_materialized_routed_down_bytes == 0
        ),
        "fixed_arena_submission_contract": (
            identities == list(plan.buffer_identities)
            and plan.dynamic_allocation_count == 0
            and plan.graph_node_count == 0
            and plan.shape_discovery_count == 0
            and plan.host_synchronization_count == 0
            and plan.returned_intermediate_tensor_bytes == 0
        ),
        "full_native_moe_speedup_at_least_1_20x": speedup >= MIN_SPEEDUP,
        "fused_boundary_saves_at_least_0_50ms": (
            incremental >= MIN_INCREMENTAL_SAVING_MS
        ),
    }
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-fused-down-reduce-native-prefill-moe-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "advance_exact_native_moe_into_complete_prefill_layer_plan"
            if accepted
            else "stop_exact_native_prefill_moe_on_structural_performance"
        ),
        "layer": args.layer,
        "reference_timing": reference_timing,
        "materialized_native_timing": materialized_timing,
        "fused_native_timing": fused_timing,
        "speedup": speedup,
        "incremental_saving_ms": incremental,
        "eliminated_routed_down_bytes": 2048 * 4096 * 2,
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
        "incremental_saving_ms": incremental,
        "failed_gates": [name for name, passed in checks.items() if not passed],
    }))
    del model, moe, plan, reference, materialized, fused, repeat
    mx.clear_cache()
    gc.collect()
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
