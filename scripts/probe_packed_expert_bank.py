#!/usr/bin/env python3
"""Measure one-layer packed expert migration without changing the runtime default."""

from __future__ import annotations

import argparse
import gc
import json
import time

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from glm53_flash_mlx.abi import KERNEL_ABI_VERSION
from glm53_flash_mlx.fp8 import DirectFP8MoE
from glm53_flash_mlx.loader import load_model, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint
from glm53_flash_mlx.packed import PackedFP8ExpertBank, PackedFP8MoE


def _snapshot() -> dict[str, int]:
    mx.synchronize()
    return {
        "active_bytes": mx.get_active_memory(),
        "cache_bytes": mx.get_cache_memory(),
        "peak_bytes": mx.get_peak_memory(),
    }


def _materialize_bank(bank: PackedFP8ExpertBank) -> None:
    mx.eval(*(value for _, value in tree_flatten(bank.parameters())))
    mx.synchronize()


def _verify_all_slices(bank: PackedFP8ExpertBank, experts) -> tuple[bool, int]:
    checks = 0
    scale_rows = bank.intermediate_scale_rows
    intermediate = bank.intermediate_size
    for expert_id, expert in enumerate(experts):
        pairs = (
            (bank.gate_up_weight[expert_id, :intermediate], expert.gate_proj.weight),
            (bank.gate_up_weight[expert_id, intermediate:], expert.up_proj.weight),
            (bank.down_weight[expert_id], expert.down_proj.weight),
            (
                bank.gate_up_scale_inv[expert_id, :scale_rows],
                expert.gate_proj.weight_scale_inv,
            ),
            (
                bank.gate_up_scale_inv[expert_id, scale_rows:],
                expert.up_proj.weight_scale_inv,
            ),
            (bank.down_scale_inv[expert_id], expert.down_proj.weight_scale_inv),
        )
        equalities = [mx.array_equal(packed, original) for packed, original in pairs]
        mx.eval(*equalities)
        checks += len(equalities)
        if not all(bool(value.item()) for value in equalities):
            return False, checks
    return True, checks


def _selected_parity(direct, packed, hidden_size: int) -> tuple[bool, float]:
    values = np.linspace(-1.0, 1.0, hidden_size, dtype=np.float32)
    x = mx.array(values.reshape(1, 1, hidden_size)).astype(mx.bfloat16)
    expected = direct(x)
    actual = packed(x)
    mx.eval(expected, actual)
    error = mx.max(
        mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32))
    ).item()
    return bool(mx.allclose(actual, expected, rtol=0.02, atol=0.02).item()), float(error)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    parser.add_argument("--max-steady-delta-gib", type=float, default=1.0)
    parser.add_argument("--max-peak-gb", type=float, default=340.0)
    args = parser.parse_args()

    report = inspect_checkpoint(args.model, require_server_ready=True)
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    model, raw_config = load_model(args.model, strict=True)
    warm_residency(model)
    layers = model.language_model.model.layers
    if not 0 <= args.layer < len(layers):
        raise ValueError(f"layer {args.layer} is outside 0..{len(layers) - 1}")
    layer = layers[args.layer]
    old_moe = layer.mlp
    if not isinstance(old_moe, DirectFP8MoE):
        raise ValueError(f"layer {args.layer} is not a routed DirectFP8MoE layer")

    baseline = _snapshot()
    mx.reset_peak_memory()
    started = time.perf_counter()
    bank = PackedFP8ExpertBank.pack(old_moe.experts)
    _materialize_bank(bank)
    packing_seconds = time.perf_counter() - started
    packed_resident = _snapshot()

    byte_identity, identity_checks = _verify_all_slices(bank, old_moe.experts)
    packed_moe = PackedFP8MoE(
        bank, old_moe.config, old_moe.gate, old_moe.shared_experts
    )
    selected_parity, selected_max_abs_error = _selected_parity(
        old_moe, packed_moe, old_moe.config.hidden_size
    )

    layer.mlp = packed_moe
    del old_moe
    gc.collect()
    clear_started = time.perf_counter()
    mx.clear_cache()
    mx.synchronize()
    clear_seconds = time.perf_counter() - clear_started
    steady = _snapshot()

    steady_delta = steady["active_bytes"] - baseline["active_bytes"]
    original_released = steady_delta <= int(args.max_steady_delta_gib * 2**30)
    peak_ok = steady["peak_bytes"] <= int(args.max_peak_gb * 1e9)
    dtype_ok = (
        bank.gate_up_weight.dtype == mx.uint8
        and bank.down_weight.dtype == mx.uint8
        and bank.gate_up_scale_inv.dtype == mx.float32
        and bank.down_scale_inv.dtype == mx.float32
    )
    accepted = all(
        (byte_identity, selected_parity, original_released, peak_ok, dtype_ok)
    )
    result = {
        "schema": "glm53-packed-expert-bank-feasibility-v1",
        "official_hf_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "kernel_abi": KERNEL_ABI_VERSION,
        "layer": args.layer,
        "expert_count": bank.expert_count,
        "hidden_size": int(raw_config["text_config"]["hidden_size"]),
        "intermediate_size": bank.intermediate_size,
        "bank_shapes": {
            "gate_up_weight": list(bank.gate_up_weight.shape),
            "gate_up_scale_inv": list(bank.gate_up_scale_inv.shape),
            "down_weight": list(bank.down_weight.shape),
            "down_scale_inv": list(bank.down_scale_inv.shape),
        },
        "bank_bytes": bank.nbytes,
        "bank_gib": bank.nbytes / 2**30,
        "packing_seconds": packing_seconds,
        "clear_cache_seconds": clear_seconds,
        "identity_checks": identity_checks,
        "all_slices_byte_identical": byte_identity,
        "selected_top8_parity": selected_parity,
        "selected_top8_max_abs_error": selected_max_abs_error,
        "canonical_dtype_preserved": dtype_ok,
        "baseline": baseline,
        "packed_resident": packed_resident,
        "steady_after_replace_and_clear": steady,
        "steady_delta_bytes": steady_delta,
        "original_expert_tensors_released": original_released,
        "full_model_peak_below_limit": peak_ok,
        "acceptance": {
            "max_steady_delta_gib": args.max_steady_delta_gib,
            "max_peak_gb": args.max_peak_gb,
            "accepted": accepted,
        },
        "runtime_default_changed": False,
    }
    print(json.dumps(result, indent=2), flush=True)
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
