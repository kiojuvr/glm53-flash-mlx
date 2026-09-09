#!/usr/bin/env python3
"""Qualify an exact Q256 native shared-expert prefill island."""

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
if str(ROOT / "native_execution") not in sys.path:
    sys.path.insert(0, str(ROOT / "native_execution"))

from glm53_native_execution import NativePrefillSharedExpertPlan
from glm53_flash_mlx.loader import load
from glm53_flash_mlx.grouped_fp8 import SortedGroupedFP8MoE


DEFAULT_OUTPUT = ROOT / "bench-results" / (
    "m3ultra512-native-prefill-shared-expert-island-20260909.json"
)
MIN_SPEEDUP = 1.20


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _hidden():
    value = mx.sin(
        mx.arange(256 * 4096, dtype=mx.float32).reshape(256, 4096) * 0.000031
    ).astype(mx.bfloat16)
    value = mx.contiguous(value)
    mx.eval(value)
    return value


def _reference(shared, hidden):
    output = shared(hidden)
    mx.eval(output); mx.synchronize()
    return output


def _candidate(plan, shared, hidden):
    output = plan.execute(
        hidden, shared.gate_proj.weight, shared.gate_proj.weight_scale_inv,
        shared.up_proj.weight, shared.up_proj.weight_scale_inv,
        shared.down_proj.weight, shared.down_proj.weight_scale_inv,
    )
    mx.eval(output); mx.synchronize()
    return output


def _timed(operation, samples):
    operation(); values = []
    for _ in range(samples):
        started = time.perf_counter_ns(); operation()
        values.append((time.perf_counter_ns() - started) / 1e6)
    return {"median_wall_ms": statistics.median(values), "samples_ms": values}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps({"phase": "load_model"}), flush=True)
    model, _ = load(args.model, experimental_packed_grouped_moe=True)
    moe = model.language_model.model.layers[args.layer].mlp
    if not isinstance(moe, SortedGroupedFP8MoE) or moe.shared_experts is None:
        raise RuntimeError("target layer does not expose a shared expert")
    shared = moe.shared_experts
    hidden = _hidden()
    weights = (
        shared.gate_proj.weight, shared.gate_proj.weight_scale_inv,
        shared.up_proj.weight, shared.up_proj.weight_scale_inv,
        shared.down_proj.weight, shared.down_proj.weight_scale_inv,
    )
    mx.eval(*weights)
    plan = NativePrefillSharedExpertPlan()
    identities = list(plan.buffer_identities)
    print(json.dumps({"phase": "exact_direct_shared"}), flush=True)
    expected = _reference(shared, hidden)
    print(json.dumps({"phase": "native_shared"}), flush=True)
    actual = _candidate(plan, shared, hidden)
    exact = np.array_equal(
        np.asarray(expected.view(mx.uint16), dtype=np.uint16),
        np.asarray(actual.view(mx.uint16), dtype=np.uint16),
    )
    repeat = _candidate(plan, shared, hidden)
    repeat_exact = np.array_equal(
        np.asarray(actual.view(mx.uint16), dtype=np.uint16),
        np.asarray(repeat.view(mx.uint16), dtype=np.uint16),
    )
    print(json.dumps({"phase": "time_direct_shared"}), flush=True)
    reference = _timed(lambda: _reference(shared, hidden), args.samples)
    print(json.dumps({"phase": "time_native_shared"}), flush=True)
    candidate = _timed(lambda: _candidate(plan, shared, hidden), args.samples)
    speedup = reference["median_wall_ms"] / candidate["median_wall_ms"]
    checks = {
        "real_checkpoint_shared_output_byte_exact": exact,
        "repeat_byte_exact": repeat_exact,
        "intermediate_never_returned": plan.returned_intermediate_tensor_bytes == 0,
        "fixed_arena_submission_contract": (
            identities == list(plan.buffer_identities)
            and plan.dynamic_allocation_count == 0
            and plan.graph_node_count == 0
            and plan.shape_discovery_count == 0
            and plan.host_synchronization_count == 0
        ),
        "native_shared_speedup_at_least_1_20x": speedup >= MIN_SPEEDUP,
    }
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-native-prefill-shared-expert-island-v1",
        "date": date.today().isoformat(), "complete": True,
        "accepted": accepted,
        "decision": (
            "advance_shared_expert_into_full_native_prefill_moe"
            if accepted else "stop_native_prefill_shared_expert_on_gate"
        ),
        "layer": args.layer, "reference_timing": reference,
        "candidate_timing": candidate, "speedup": speedup,
        "scratch_bytes": plan.scratch_bytes, "checks": checks,
        "probe_only": True, "runtime_changes": False,
    }
    _write(args.output, artifact)
    print(json.dumps({
        "output": str(args.output), "complete": True, "accepted": accepted,
        "decision": artifact["decision"], "speedup": speedup,
        "failed_gates": [name for name, passed in checks.items() if not passed],
    }))
    del model, moe, shared, plan, expected, actual, repeat
    mx.clear_cache(); gc.collect()
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
