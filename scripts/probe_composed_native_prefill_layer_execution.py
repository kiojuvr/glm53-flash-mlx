#!/usr/bin/env python3
"""Measure the exact currently-composable Q256 native prefill layer path."""

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

from glm53_native_execution import NativePrefillDSAOutputPlan, NativePrefillMoEPlan
from glm53_flash_mlx.grouped_fp8 import SortedGroupedFP8MoE
from glm53_flash_mlx.loader import load
from mlx_vlm.models.deepseek_v4.hyper_connection import hc_expand
import probe_native_prefill_dsa_output_projection as dsa_probe
import probe_native_prefill_full_moe_island as moe_probe


DEFAULT_OUTPUT = ROOT / "bench-results" / (
    "m3ultra512-composed-native-prefill-layer-execution-20260909.json"
)
MIN_COMPLETE_LAYER_SPEEDUP = 1.20


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _residual() -> mx.array:
    value = mx.cos(
        mx.arange(256 * 4 * 4096, dtype=mx.float32).reshape(1, 256, 4, 4096)
        * 0.000031
    ).astype(mx.bfloat16)
    value = mx.contiguous(value)
    mx.eval(value)
    return value


def _direct_dsa_hc(projection, head_major, residual, post, comb):
    row_major = mx.contiguous(
        head_major.transpose(1, 0, 2).reshape(1, 256, 16_384),
        allow_col_major=False,
    )
    projected = projection(row_major)
    state = hc_expand(projected, residual, post, comb)
    mx.eval(state)
    mx.synchronize()
    return state


def _ffn_entry(layer, state):
    collapsed, post, comb = layer.ffn_hc(state)
    normalized = layer.post_attention_layernorm(collapsed)
    indices, scores = layer.mlp.gate(normalized)
    indices = mx.contiguous(indices.astype(mx.uint32))
    scores = mx.contiguous(scores.astype(mx.float32))
    mx.eval(collapsed, post, comb, normalized, indices, scores)
    mx.synchronize()
    return normalized, post, comb, indices, scores


def _direct(layer, head_major, residual, attn_post, attn_comb):
    state = _direct_dsa_hc(
        layer.self_attn.o_proj, head_major, residual, attn_post, attn_comb
    )
    normalized, post, comb, indices, scores = _ffn_entry(layer, state)
    moe = moe_probe._reference(layer.mlp, normalized, indices, scores)
    output = hc_expand(moe, state, post, comb)
    mx.eval(output)
    mx.synchronize()
    return output


def _native_moe(plan, moe, hidden, indices, scores, *, indirect=False):
    shared = moe.shared_experts
    flat_indices = mx.contiguous(indices.reshape(-1))
    flat_scores = mx.contiguous(scores.reshape(-1))
    mx.eval(flat_indices, flat_scores)
    method = plan.execute_indirect if indirect else plan.execute
    output = method(
        hidden,
        flat_indices,
        flat_scores,
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


def _candidate(
    dsa_plan, moe_plan, layer, head_major, residual, attn_post, attn_comb,
    *, indirect_moe=False
):
    state = dsa_plan.execute_hc(
        head_major,
        layer.self_attn.o_proj.weight,
        layer.self_attn.o_proj.weight_scale_inv,
        residual,
        attn_post,
        attn_comb,
    )
    mx.eval(state)
    mx.synchronize()
    state = state.reshape(1, 256, 4, 4096)
    normalized, post, comb, indices, scores = _ffn_entry(layer, state)
    moe = _native_moe(
        moe_plan,
        layer.mlp,
        normalized,
        indices,
        scores,
        indirect=indirect_moe,
    )
    output = hc_expand(moe[None], state, post, comb)
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--indirect-moe", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    print(json.dumps({"phase": "load_model"}), flush=True)
    model, _ = load(args.model, experimental_packed_grouped_moe=True)
    layer = model.language_model.model.layers[args.layer]
    if not isinstance(layer.mlp, SortedGroupedFP8MoE):
        raise RuntimeError("target layer is not a packed grouped MoE")
    head_major = dsa_probe._head_major()
    residual = _residual()
    _, attn_post, attn_comb = layer.attn_hc(residual)
    attn_post = mx.contiguous(attn_post)
    attn_comb = mx.contiguous(attn_comb)
    mx.eval(attn_post, attn_comb)
    dsa_plan = NativePrefillDSAOutputPlan()
    moe_plan = NativePrefillMoEPlan(layer.mlp.bank.expert_count)
    dsa_ids = list(dsa_plan.buffer_identities)
    moe_ids = list(moe_plan.buffer_identities)

    print(json.dumps({"phase": "direct_complete_layer"}), flush=True)
    expected = _direct(layer, head_major, residual, attn_post, attn_comb)
    expected_bits = _bits(expected).copy()
    print(json.dumps({"phase": "native_composed_complete_layer"}), flush=True)
    actual = _candidate(
        dsa_plan, moe_plan, layer, head_major, residual, attn_post, attn_comb,
        indirect_moe=args.indirect_moe,
    )
    actual_bits = _bits(actual).copy()
    repeat = _candidate(
        dsa_plan, moe_plan, layer, head_major, residual, attn_post, attn_comb,
        indirect_moe=args.indirect_moe,
    )
    repeat_bits = _bits(repeat).copy()

    print(json.dumps({"phase": "timing"}), flush=True)
    direct_timing = _timed(
        lambda: _direct(layer, head_major, residual, attn_post, attn_comb),
        args.samples,
    )
    candidate_timing = _timed(
        lambda: _candidate(
            dsa_plan,
            moe_plan,
            layer,
            head_major,
            residual,
            attn_post,
            attn_comb,
            indirect_moe=args.indirect_moe,
        ),
        args.samples,
    )
    speedup = direct_timing["median_wall_ms"] / candidate_timing["median_wall_ms"]
    required_candidate_ms = direct_timing["median_wall_ms"] / MIN_COMPLETE_LAYER_SPEEDUP
    remaining_to_gate_ms = candidate_timing["median_wall_ms"] - required_candidate_ms
    checks = {
        "complete_layer_output_is_byte_exact": np.array_equal(
            expected_bits.reshape(actual_bits.shape), actual_bits
        ),
        "repeat_is_byte_exact": np.array_equal(actual_bits, repeat_bits),
        "native_component_buffers_are_stable": (
            dsa_ids == list(dsa_plan.buffer_identities)
            and moe_ids == list(moe_plan.buffer_identities)
        ),
        "currently_composed_layer_speedup_at_least_1_20x": (
            speedup >= MIN_COMPLETE_LAYER_SPEEDUP
        ),
    }
    exact = all(
        checks[name]
        for name in (
            "complete_layer_output_is_byte_exact",
            "repeat_is_byte_exact",
            "native_component_buffers_are_stable",
        )
    )
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-composed-native-prefill-layer-execution-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "implement_single_scope_complete_native_prefill_layer"
            if accepted
            else (
                "redesign_complete_layer_dataflow_before_native_glue"
                if exact
                else "localize_composed_native_layer_exactness"
            )
        ),
        "layer": args.layer,
        "direct_timing": direct_timing,
        "candidate_timing": candidate_timing,
        "speedup": speedup,
        "required_candidate_ms_for_1_20x": required_candidate_ms,
        "remaining_wall_ms_to_1_20x": remaining_to_gate_ms,
        "checks": checks,
        "composition": {
            "native_dsa_output_projection_and_attn_hc": True,
            "mlx_ffn_entry_glue": True,
            "native_exact_materialized_moe": True,
            "device_sized_indirect_moe_dispatch": args.indirect_moe,
            "mlx_final_hc_expand": True,
            "single_native_scope": False,
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
        "remaining_wall_ms_to_1_20x": remaining_to_gate_ms,
        "failed_gates": [name for name, passed in checks.items() if not passed],
    }))
    del model, layer, dsa_plan, moe_plan, expected, actual, repeat
    del head_major, residual, attn_post, attn_comb
    mx.clear_cache()
    gc.collect()
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
