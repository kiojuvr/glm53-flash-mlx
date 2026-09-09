#!/usr/bin/env python3
"""Qualify exact device-sized indirect dispatch for Q256 native prefill MoE."""

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
    "m3ultra512-indirect-native-prefill-moe-dispatch-20260909.json"
)
MIN_INCREMENTAL_SAVING_MS = 1.70
MIN_DIRECT_SPEEDUP = 1.20


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _candidate(plan, moe, hidden, indices, scores, *, indirect):
    shared = moe.shared_experts
    method = plan.execute_indirect if indirect else plan.execute
    output = method(
        hidden,
        indices,
        scores,
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
    mx.eval(output)
    mx.synchronize()
    return output


def _bits(value):
    return np.asarray(value.view(mx.uint16), dtype=np.uint16)


def _timed(operation, samples):
    operation()
    values = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        operation()
        values.append((time.perf_counter_ns() - started) / 1e6)
    return {"median_wall_ms": statistics.median(values), "samples_ms": values}


def main(argv=None):
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
    indices, scores = moe.gate(hidden)
    indices = mx.contiguous(indices.astype(mx.uint32).reshape(-1))
    scores = mx.contiguous(scores.astype(mx.float32).reshape(-1))
    mx.eval(indices, scores)
    shaped_indices = indices.reshape(256, 8)
    shaped_scores = scores.reshape(256, 8)
    plan = NativePrefillMoEPlan(moe.bank.expert_count)
    identities = list(plan.buffer_identities)

    print(json.dumps({"phase": "direct_exact_oracle"}), flush=True)
    expected = full_probe._reference(
        moe, hidden, shaped_indices, shaped_scores
    )
    expected_bits = _bits(expected).copy()
    print(json.dumps({"phase": "fixed_capacity_native"}), flush=True)
    fixed = _candidate(plan, moe, hidden, indices, scores, indirect=False)
    fixed_bits = _bits(fixed).copy()
    print(json.dumps({"phase": "indirect_native"}), flush=True)
    indirect = _candidate(plan, moe, hidden, indices, scores, indirect=True)
    indirect_bits = _bits(indirect).copy()
    repeat = _candidate(plan, moe, hidden, indices, scores, indirect=True)
    repeat_bits = _bits(repeat).copy()

    print(json.dumps({"phase": "timing"}), flush=True)
    direct_timing = _timed(
        lambda: full_probe._reference(moe, hidden, shaped_indices, shaped_scores),
        args.samples,
    )
    fixed_timing = _timed(
        lambda: _candidate(plan, moe, hidden, indices, scores, indirect=False),
        args.samples,
    )
    indirect_timing = _timed(
        lambda: _candidate(plan, moe, hidden, indices, scores, indirect=True),
        args.samples,
    )
    saving = fixed_timing["median_wall_ms"] - indirect_timing["median_wall_ms"]
    direct_speedup = (
        direct_timing["median_wall_ms"] / indirect_timing["median_wall_ms"]
    )
    checks = {
        "fixed_capacity_native_is_byte_exact": np.array_equal(
            expected_bits.reshape(fixed_bits.shape), fixed_bits
        ),
        "indirect_native_is_byte_exact": np.array_equal(
            expected_bits.reshape(indirect_bits.shape), indirect_bits
        ),
        "indirect_repeat_is_byte_exact": np.array_equal(
            indirect_bits, repeat_bits
        ),
        "indirect_plan_buffers_are_stable": identities == list(plan.buffer_identities),
        "indirect_saves_at_least_1_70ms_vs_fixed_capacity": (
            saving >= MIN_INCREMENTAL_SAVING_MS
        ),
        "indirect_exact_moe_speedup_at_least_1_20x": (
            direct_speedup >= MIN_DIRECT_SPEEDUP
        ),
    }
    exact = all(
        checks[name]
        for name in (
            "fixed_capacity_native_is_byte_exact",
            "indirect_native_is_byte_exact",
            "indirect_repeat_is_byte_exact",
            "indirect_plan_buffers_are_stable",
        )
    )
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-indirect-native-prefill-moe-dispatch-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "advance_indirect_exact_moe_into_complete_layer_requalification"
            if accepted
            else (
                "stop_indirect_descriptor_dispatch_on_performance"
                if exact
                else "stop_and_localize_indirect_descriptor_dispatch"
            )
        ),
        "layer": args.layer,
        "direct_timing": direct_timing,
        "fixed_capacity_timing": fixed_timing,
        "indirect_timing": indirect_timing,
        "indirect_saving_ms": saving,
        "indirect_speedup_vs_direct": direct_speedup,
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
        "indirect_saving_ms": saving,
        "indirect_speedup_vs_direct": direct_speedup,
        "failed_gates": [name for name, passed in checks.items() if not passed],
    }))
    del model, moe, plan, expected, fixed, indirect, repeat
    mx.clear_cache()
    gc.collect()
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
