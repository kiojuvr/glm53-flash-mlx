#!/usr/bin/env python3
"""Qualify the exact Q256 R=4 KDA recurrence as a native plan slot."""

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

from glm53_native_execution import NativePrefillKDARecurrentPlan
import probe_row_blocked_vector_kda_prefill as kda_probe


DEFAULT_OUTPUT = ROOT / "bench-results" / (
    "m3ultra512-native-prefill-kda-recurrent-plan-20260909.json"
)
MIN_OPERATOR_SPEEDUP = 1.15


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _bits(value: mx.array) -> np.ndarray:
    dtype = mx.uint16 if value.dtype == mx.bfloat16 else mx.uint32
    return np.asarray(value.view(dtype)).copy()


def _inputs(initial: str):
    q, k, v, g, beta, state = kda_probe._operator_inputs(256)
    if initial == "zero":
        state = mx.zeros_like(state)
    values = tuple(mx.contiguous(value) for value in (q, k, v, g, beta, state))
    mx.eval(*values)
    return values


def _reference(inputs, mask):
    output, state = kda_probe._reference(*inputs, mask)
    mx.eval(output, state)
    mx.synchronize()
    return output, state


def _candidate_prepared(plan, inputs, mask_input, has_mask):
    output, state = plan.execute(*inputs, mask_input, has_mask)
    mx.eval(output, state)
    mx.synchronize()
    return output, state


def _candidate(plan, inputs, mask):
    mask_input = (
        mx.ones((1,), dtype=mx.bool_)
        if mask is None
        else mx.contiguous(mask)
    )
    mx.eval(mask_input)
    return _candidate_prepared(plan, inputs, mask_input, mask is not None)


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
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    plan = NativePrefillKDARecurrentPlan()
    identities = list(plan.buffer_identities)
    fixtures = []
    timing_inputs = None
    for initial in ("zero", "nonzero"):
        inputs = _inputs(initial)
        if initial == "nonzero":
            timing_inputs = inputs
        for mask_kind in ("none", "tail", "internal_gap"):
            mask = None
            if mask_kind != "none":
                mask = mx.ones((1, 256), dtype=mx.bool_)
                if mask_kind == "tail":
                    mask[:, -1] = False
                else:
                    mask[:, 127] = False
                mask = mx.contiguous(mask)
                mx.eval(mask)
            print(json.dumps({
                "phase": "fixture", "initial": initial, "mask": mask_kind
            }), flush=True)
            expected_output, expected_state = _reference(inputs, mask)
            actual_output, actual_state = _candidate(plan, inputs, mask)
            repeat_output, repeat_state = _candidate(plan, inputs, mask)
            fixtures.append({
                "initial": initial,
                "mask": mask_kind,
                "output_byte_exact": np.array_equal(
                    _bits(expected_output), _bits(actual_output)
                ),
                "state_byte_exact": np.array_equal(
                    _bits(expected_state), _bits(actual_state)
                ),
                "repeat_output_byte_exact": np.array_equal(
                    _bits(actual_output), _bits(repeat_output)
                ),
                "repeat_state_byte_exact": np.array_equal(
                    _bits(actual_state), _bits(repeat_state)
                ),
            })

    assert timing_inputs is not None
    resident_no_mask = mx.ones((1,), dtype=mx.bool_)
    mx.eval(resident_no_mask)
    print(json.dumps({"phase": "timing"}), flush=True)
    reference_timing = _timed(
        lambda: _reference(timing_inputs, None), args.samples
    )
    candidate_timing = _timed(
        lambda: _candidate_prepared(
            plan, timing_inputs, resident_no_mask, False
        ),
        args.samples,
    )
    speedup = reference_timing["median_wall_ms"] / candidate_timing["median_wall_ms"]
    checks = {
        "zero_nonzero_and_mask_fixtures_byte_exact": all(
            all(value for key, value in row.items() if key.endswith("exact"))
            for row in fixtures
        ),
        "fixed_q256_h64_d128_r4_geometry": (
            plan.query_rows == 256
            and plan.heads == 64
            and plan.key_dim == 128
            and plan.value_dim == 128
            and plan.row_block == 4
        ),
        "operator_speedup_at_least_1_15x": speedup >= MIN_OPERATOR_SPEEDUP,
        "fixed_owned_native_arena": (
            identities == list(plan.buffer_identities)
            and plan.buffer_identities_stable
            and plan.dynamic_allocation_count == 0
            and plan.graph_node_count == 0
            and plan.shape_discovery_count == 0
            and plan.host_synchronization_count == 0
            and plan.returned_intermediate_tensor_bytes == 0
        ),
    }
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-native-prefill-kda-recurrent-plan-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "advance_native_kda_recurrence_into_complete_kda_layer_plan"
            if accepted
            else "stop_or_relocalize_native_kda_recurrence"
        ),
        "fixtures": fixtures,
        "reference_timing": reference_timing,
        "candidate_timing": candidate_timing,
        "speedup": speedup,
        "arena_bytes": plan.arena_bytes,
        "checks": checks,
        "coverage": {
            "qkv_projection": False,
            "conv": False,
            "gate_preparation": False,
            "exact_recurrence": True,
            "gated_norm": False,
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
        "speedup": speedup,
        "failed_gates": [name for name, passed in checks.items() if not passed],
    }))
    del plan, timing_inputs
    mx.clear_cache()
    gc.collect()
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
