#!/usr/bin/env python3
"""Probe exact native Q256 route + BM8 gate/up/SwiGLU composition."""

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
import mlx.nn as nn
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NATIVE_PACKAGE = ROOT / "native_execution"
if str(NATIVE_PACKAGE) not in sys.path:
    sys.path.insert(0, str(NATIVE_PACKAGE))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from glm53_native_execution import NativePrefillMoEGateUpPlan
from glm53_flash_mlx.loader import load
from glm53_flash_mlx.grouped_fp8 import SortedGroupedFP8MoE
import probe_grouped_fp8_parity_ladder as parity


DEFAULT_OUTPUT = ROOT / "bench-results" / (
    "m3ultra512-native-prefill-moe-gate-up-island-20260909.json"
)
QUERY_ROWS = 256
HIDDEN = 4096
EXPERTS = 288
MIN_SPEEDUP = 1.20


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _hidden() -> mx.array:
    value = mx.sin(
        mx.arange(QUERY_ROWS * HIDDEN, dtype=mx.float32).reshape(
            QUERY_ROWS, HIDDEN
        )
        * 0.000031
    ).astype(mx.bfloat16)
    value = mx.contiguous(value)
    mx.eval(value)
    return value


def _reference(moe, hidden, indices, scores):
    route = parity._stable_route_plan(hidden, indices, scores)
    tile = parity._build_tile_plan(
        route["sorted_experts"], moe.bank.expert_count, tile_rows=8
    )
    gate = parity.direct_order_grouped_linear(
        route["sorted_x"], tile, moe.bank.gate_up_weight,
        moe.bank.gate_up_scale_inv, row_offset=0, scale_row_offset=0,
        out_features=moe.bank.intermediate_size,
    )
    up = parity.direct_order_grouped_linear(
        route["sorted_x"], tile, moe.bank.gate_up_weight,
        moe.bank.gate_up_scale_inv,
        row_offset=moe.bank.intermediate_size,
        scale_row_offset=moe.bank.intermediate_scale_rows,
        out_features=moe.bank.intermediate_size,
    )
    activated = nn.silu(mx.minimum(gate, moe.config.swiglu_limit)) * mx.clip(
        up, -moe.config.swiglu_limit, moe.config.swiglu_limit
    )
    mx.eval(activated)
    mx.synchronize()
    return activated


def _candidate(plan, moe, hidden, indices, scores):
    output = plan.execute(
        hidden, indices, scores, moe.bank.gate_up_weight,
        moe.bank.gate_up_scale_inv,
    )
    mx.eval(output)
    mx.synchronize()
    return output


def _timed(operation, samples: int):
    operation()
    values = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        operation()
        values.append((time.perf_counter_ns() - started) / 1e6)
    return {"median_wall_ms": statistics.median(values), "samples_ms": values}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    print(json.dumps({"phase": "load_model"}), flush=True)
    model, _ = load(args.model, experimental_packed_grouped_moe=True)
    moe = model.language_model.model.layers[args.layer].mlp
    if not isinstance(moe, SortedGroupedFP8MoE):
        raise RuntimeError("target layer is not a packed grouped MoE")
    hidden = _hidden()
    indices, scores = moe.gate(hidden)
    indices = mx.contiguous(indices.astype(mx.uint32))
    scores = mx.contiguous(scores.astype(mx.float32))
    mx.eval(
        indices, scores, moe.bank.gate_up_weight, moe.bank.gate_up_scale_inv
    )
    flat_indices = mx.contiguous(indices.reshape(-1))
    flat_scores = mx.contiguous(scores.reshape(-1))
    mx.eval(flat_indices, flat_scores)
    plan = NativePrefillMoEGateUpPlan(moe.bank.expert_count)
    identities = list(plan.buffer_identities)

    print(json.dumps({"phase": "exact_direct_order_bm8"}), flush=True)
    expected = _reference(moe, hidden, indices, scores)
    print(json.dumps({"phase": "native_route_gate_up_swiglu"}), flush=True)
    actual = _candidate(plan, moe, hidden, flat_indices, flat_scores)
    exact = np.array_equal(
        np.asarray(expected.view(mx.uint16), dtype=np.uint16),
        np.asarray(actual.view(mx.uint16), dtype=np.uint16),
    )
    repeat = _candidate(plan, moe, hidden, flat_indices, flat_scores)
    repeat_exact = np.array_equal(
        np.asarray(actual.view(mx.uint16), dtype=np.uint16),
        np.asarray(repeat.view(mx.uint16), dtype=np.uint16),
    )

    print(json.dumps({"phase": "time_exact_mlx_ingress"}), flush=True)
    reference_timing = _timed(
        lambda: _reference(moe, hidden, indices, scores), args.samples
    )
    print(json.dumps({"phase": "time_composed_native_ingress"}), flush=True)
    candidate_timing = _timed(
        lambda: _candidate(plan, moe, hidden, flat_indices, flat_scores), args.samples
    )
    speedup = (
        reference_timing["median_wall_ms"] / candidate_timing["median_wall_ms"]
    )
    checks = {
        "real_checkpoint_gate_up_swiglu_byte_exact": exact,
        "repeat_byte_exact": repeat_exact,
        "route_metadata_remains_internal": plan.returned_route_metadata_bytes == 0,
        "sorted_hidden_never_materialized": plan.materialized_sorted_hidden_bytes == 0,
        "fixed_arena_submission_contract": (
            identities == list(plan.buffer_identities)
            and plan.dynamic_allocation_count == 0
            and plan.graph_node_count == 0
            and plan.shape_discovery_count == 0
            and plan.host_synchronization_count == 0
        ),
        "composed_gate_up_speedup_at_least_1_20x": speedup >= MIN_SPEEDUP,
    }
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-native-prefill-moe-gate-up-island-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "advance_exact_native_prefill_moe_to_down_reduce_shared"
            if accepted
            else "stop_or_relocalize_native_prefill_moe_gate_up_island"
        ),
        "layer": args.layer,
        "geometry": {
            "query_rows": QUERY_ROWS,
            "route_rows": QUERY_ROWS * 8,
            "hidden_size": HIDDEN,
            "intermediate_size": moe.bank.intermediate_size,
            "expert_count": moe.bank.expert_count,
        },
        "reference_timing": reference_timing,
        "candidate_timing": candidate_timing,
        "speedup": speedup,
        "scratch_bytes": plan.scratch_bytes,
        "avoided_sorted_hidden_bytes": QUERY_ROWS * 8 * HIDDEN * 2,
        "checks": checks,
        "probe_only": True,
        "runtime_changes": False,
    }
    _atomic_write(args.output, artifact)
    print(json.dumps({
        "output": str(args.output), "complete": True, "accepted": accepted,
        "decision": artifact["decision"], "speedup": speedup,
        "failed_gates": [name for name, passed in checks.items() if not passed],
    }))
    del model, moe, plan, hidden, indices, scores, flat_indices, flat_scores
    del expected, actual, repeat
    mx.clear_cache()
    gc.collect()
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
