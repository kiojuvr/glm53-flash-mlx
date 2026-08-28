#!/usr/bin/env python3
"""Benchmark layer-local sorted grouped FP8 MoE against DirectFP8MoE."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from glm53_flash_mlx.abi import KERNEL_ABI_VERSION
from glm53_flash_mlx.fp8 import DirectFP8MoE
from glm53_flash_mlx.grouped_fp8 import (
    GROUPED_KERNEL_ABI,
    GROUPED_MIN_ROUTES,
    GROUPED_TILE_ROWS,
    SortedGroupedFP8MoE,
    activate_gate_up,
    build_grouped_tile_plan,
    build_route_plan,
    grouped_fp8_linear,
    restore_and_reduce,
)
from glm53_flash_mlx.loader import load_model, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint
from glm53_flash_mlx.packed import PackedFP8ExpertBank, PackedFP8MoE

SEQUENCES = (32, 64, 128, 256, 512)
CROSSOVER_SEQUENCES = (*range(1, 9), 16, 24, 32)


def _snapshot() -> dict[str, int]:
    mx.synchronize()
    return {
        "active_bytes": mx.get_active_memory(),
        "cache_bytes": mx.get_cache_memory(),
        "peak_bytes": mx.get_peak_memory(),
    }


def _eval_timed(factory, *, warmups: int, repeats: int) -> tuple[float, list[float]]:
    for _ in range(warmups):
        output = factory()
        mx.eval(output)
        mx.synchronize()
        del output
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = factory()
        mx.eval(output)
        mx.synchronize()
        samples.append(time.perf_counter() - started)
        del output
    return statistics.median(samples), samples


def _time_value(factory):
    mx.synchronize()
    started = time.perf_counter()
    value = factory()
    values = value if isinstance(value, tuple) else (value,)
    mx.eval(*values)
    mx.synchronize()
    return value, time.perf_counter() - started


def _error_metrics(expected, actual) -> dict[str, float]:
    diff = actual.astype(mx.float32) - expected.astype(mx.float32)
    absolute = mx.abs(diff)
    metrics = (mx.max(absolute), mx.mean(absolute), mx.sqrt(mx.mean(diff * diff)))
    mx.eval(*metrics)
    return {
        "max_abs": float(metrics[0].item()),
        "mean_abs": float(metrics[1].item()),
        "rms": float(metrics[2].item()),
        "allclose_rtol_0_02_atol_0_02": bool(
            mx.allclose(actual, expected, rtol=0.02, atol=0.02).item()
        ),
    }


def _make_input(sequence: int, hidden_size: int):
    raw = np.arange(sequence * hidden_size, dtype=np.float32)
    raw = np.sin(raw * np.float32(0.0009765625)).reshape(
        1, sequence, hidden_size
    )
    x = mx.array(raw).astype(mx.bfloat16)
    mx.eval(x)
    return x


def _forced_grouped(grouped, x):
    indices, scores = grouped.gate(x)
    return grouped.grouped_from_routes(x, indices, scores)


def _route_metrics(
    sorted_experts,
    tile_plan,
    expert_count: int,
    tile_rows: int = GROUPED_TILE_ROWS,
) -> dict:
    mx.eval(sorted_experts, *tile_plan)
    values = np.asarray(sorted_experts, dtype=np.uint32)
    counts = np.bincount(values, minlength=expert_count)
    tile_experts, tile_starts, tile_lengths, route_offsets, tile_offsets = (
        np.asarray(value, dtype=np.uint32) for value in tile_plan
    )
    expected_route_offsets = np.concatenate(
        [np.zeros(1, dtype=np.uint32), np.cumsum(counts, dtype=np.uint32)]
    )
    expected_tile_counts = (counts + tile_rows - 1) // tile_rows
    expected_tile_offsets = np.concatenate(
        [
            np.zeros(1, dtype=np.uint32),
            np.cumsum(expected_tile_counts, dtype=np.uint32),
        ]
    )
    assert np.array_equal(route_offsets, expected_route_offsets)
    assert np.array_equal(tile_offsets, expected_tile_offsets)

    valid = tile_starts < values.size
    assert int(valid.sum()) == int(expected_tile_offsets[-1])
    assert np.all(tile_lengths[~valid] == 0)
    coverage = np.zeros(values.size, dtype=np.uint8)
    boundary_tiles = 0
    for expert, start, length in zip(
        tile_experts[valid], tile_starts[valid], tile_lengths[valid], strict=True
    ):
        expert = int(expert)
        start = int(start)
        length = int(length)
        assert 0 <= expert < expert_count
        assert 0 < length <= tile_rows
        assert start + length <= values.size
        descriptor_values = values[start : start + length]
        boundary_tiles += int(np.any(descriptor_values != expert))
        coverage[start : start + length] += 1
    assert boundary_tiles == 0
    assert np.all(coverage == 1)

    naive_boundary_tiles = 0
    for start in range(0, values.size, tile_rows):
        if np.unique(values[start : start + tile_rows]).size > 1:
            naive_boundary_tiles += 1
    used = counts[counts > 0]
    aligned_tiles = int(valid.sum())
    descriptor_slots = int(tile_experts.size)
    return {
        "unique_experts": int(used.size),
        "zero_route_experts": int(np.sum(counts == 0)),
        "routes_per_expert_mean": float(used.mean()) if used.size else 0.0,
        "routes_per_expert_max": int(used.max()) if used.size else 0,
        "expert_boundary_tiles": boundary_tiles,
        "descriptor_routes_covered_once": bool(np.all(coverage == 1)),
        "naive_fixed_grid_boundary_tiles": naive_boundary_tiles,
        "aligned_route_tiles": aligned_tiles,
        "descriptor_slots": descriptor_slots,
        "unused_descriptor_slots": descriptor_slots - aligned_tiles,
    }


def _phase_profile(grouped, x) -> tuple[dict, object, tuple]:
    (indices, scores), router_seconds = _time_value(lambda: grouped.gate(x))
    plan, sort_seconds = _time_value(lambda: build_route_plan(x, indices, scores))
    sorted_x, experts, sorted_scores, inverse = plan
    tile_plan, tile_plan_seconds = _time_value(
        lambda: build_grouped_tile_plan(experts, grouped.bank.expert_count)
    )
    gate_up, gate_up_seconds = _time_value(
        lambda: grouped_fp8_linear(
            sorted_x,
            tile_plan,
            grouped.bank.gate_up_weight,
            grouped.bank.gate_up_scale_inv,
        )
    )
    hidden, activation_seconds = _time_value(
        lambda: activate_gate_up(
            gate_up, grouped.bank.intermediate_size, grouped.config.swiglu_limit
        )
    )
    down, down_seconds = _time_value(
        lambda: grouped_fp8_linear(
            hidden,
            tile_plan,
            grouped.bank.down_weight,
            grouped.bank.down_scale_inv,
        )
    )
    restored, restore_seconds = _time_value(
        lambda: restore_and_reduce(
            down,
            sorted_scores,
            inverse,
            x.shape,
            grouped.config.num_experts_per_tok,
        )
    )
    if grouped.shared_experts is None:
        shared_seconds = 0.0
    else:
        _, shared_seconds = _time_value(lambda: grouped.shared_experts(x))
    return (
        {
            "router_seconds": router_seconds,
            "sort_seconds": sort_seconds,
            "tile_plan_seconds": tile_plan_seconds,
            "gate_up_seconds": gate_up_seconds,
            "activation_seconds": activation_seconds,
            "down_seconds": down_seconds,
            "restore_reduce_seconds": restore_seconds,
            "shared_expert_seconds": shared_seconds,
        },
        experts,
        tile_plan,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--sequences", type=int, nargs="+", default=list(SEQUENCES))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    args = parser.parse_args()

    report = inspect_checkpoint(args.model, require_server_ready=True)
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    model, _ = load_model(args.model, strict=True)
    warm_residency(model)
    layer = model.language_model.model.layers[args.layer]
    direct = layer.mlp
    if not isinstance(direct, DirectFP8MoE):
        raise ValueError(f"layer {args.layer} is not a routed DirectFP8MoE layer")
    bank = PackedFP8ExpertBank.pack(direct.experts)
    mx.eval(*(value for _, value in tree_flatten(bank.parameters())))
    mx.synchronize()
    grouped = SortedGroupedFP8MoE(
        bank, direct.config, direct.gate, direct.shared_experts
    )
    packed = PackedFP8MoE(
        bank, direct.config, direct.gate, direct.shared_experts
    )
    storage_baseline = _snapshot()

    crossover_results = []
    for sequence in CROSSOVER_SEQUENCES:
        x = _make_input(sequence, direct.config.hidden_size)
        expected = packed(x)
        actual = _forced_grouped(grouped, x)
        mx.eval(expected, actual)
        errors = _error_metrics(expected, actual)
        packed_median, packed_samples = _eval_timed(
            lambda: packed(x), warmups=args.warmups, repeats=args.repeats
        )
        grouped_median, grouped_samples = _eval_timed(
            lambda: _forced_grouped(grouped, x),
            warmups=args.warmups,
            repeats=args.repeats,
        )
        crossover_results.append(
            {
                "sequence_tokens": sequence,
                "routes": sequence * direct.config.num_experts_per_tok,
                "forced_grouped": True,
                "packed_fallback_seconds": {
                    "median": packed_median,
                    "samples": packed_samples,
                },
                "grouped_seconds": {
                    "median": grouped_median,
                    "samples": grouped_samples,
                },
                "speedup": packed_median / grouped_median,
                "error": errors,
            }
        )
        del expected, actual, x
        gc.collect()

    break_even = next(
        (row for row in crossover_results if row["speedup"] >= 1.0), None
    )
    break_even_index = (
        crossover_results.index(break_even) if break_even is not None else None
    )
    last_slower = (
        crossover_results[break_even_index - 1]
        if break_even_index is not None and break_even_index > 0
        else None
    )
    if break_even is not None:
        grouped.min_routes = break_even["routes"]

    results = []
    grouped_peak_bytes = storage_baseline["active_bytes"]
    for sequence in args.sequences:
        x = _make_input(sequence, direct.config.hidden_size)

        expected = direct(x)
        actual = grouped(x)
        mx.eval(expected, actual)
        errors = _error_metrics(expected, actual)
        direct_median, direct_samples = _eval_timed(
            lambda: direct(x), warmups=args.warmups, repeats=args.repeats
        )

        mx.clear_cache()
        mx.reset_peak_memory()
        grouped_median, grouped_samples = _eval_timed(
            lambda: grouped(x), warmups=args.warmups, repeats=args.repeats
        )
        phase, sorted_experts, tile_plan = _phase_profile(grouped, x)
        grouped_peak_bytes = max(grouped_peak_bytes, mx.get_peak_memory())
        routes = sequence * direct.config.num_experts_per_tok
        results.append(
            {
                "sequence_tokens": sequence,
                "routes": routes,
                **_route_metrics(
                    sorted_experts, tile_plan, bank.expert_count
                ),
                "direct_seconds": {
                    "median": direct_median,
                    "samples": direct_samples,
                },
                "grouped_seconds": {
                    "median": grouped_median,
                    "samples": grouped_samples,
                },
                "speedup": direct_median / grouped_median,
                "error": errors,
                "phase_seconds": phase,
            }
        )
        del expected, actual, x
        gc.collect()

    mx.clear_cache()
    steady = _snapshot()
    working_peak_delta = grouped_peak_bytes - storage_baseline["active_bytes"]
    row_256 = next(row for row in results if row["sequence_tokens"] == 256)
    parity_ok = all(row["error"]["allclose_rtol_0_02_atol_0_02"] for row in results)
    crossover_parity_ok = all(
        row["error"]["allclose_rtol_0_02_atol_0_02"]
        for row in crossover_results
    )
    crossover_bracketed = last_slower is not None
    threshold_matches = (
        break_even is not None
        and grouped.min_routes == GROUPED_MIN_ROUTES == break_even["routes"]
    )
    performance_ok = row_256["speedup"] >= 1.5
    memory_ok = working_peak_delta <= 512 * 2**20
    dtype_ok = (
        bank.gate_up_weight.dtype == mx.uint8
        and bank.down_weight.dtype == mx.uint8
        and bank.gate_up_scale_inv.dtype == mx.float32
        and bank.down_scale_inv.dtype == mx.float32
    )
    accepted = (
        parity_ok
        and crossover_parity_ok
        and break_even is not None
        and crossover_bracketed
        and threshold_matches
        and performance_ok
        and memory_ok
        and dtype_ok
    )
    output = {
        "schema": "glm53-sorted-grouped-fp8-moe-probe-v2",
        "official_hf_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "runtime_kernel_abi": KERNEL_ABI_VERSION,
        "grouped_kernel_abi": GROUPED_KERNEL_ABI,
        "layer": args.layer,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "bank_bytes": bank.nbytes,
        "canonical_dtype_preserved": dtype_ok,
        "weight_bf16_expansion": False,
        "hot_path_host_sync_free": True,
        "decode_selected_top8_fallback_below_routes": grouped.min_routes,
        "storage_baseline": storage_baseline,
        "grouped_working_peak_bytes": grouped_peak_bytes,
        "grouped_working_peak_delta_bytes": working_peak_delta,
        "steady_after_clear": steady,
        "measured_break_even": (
            {
                "last_slower": {
                    "sequence_tokens": last_slower["sequence_tokens"],
                    "routes": last_slower["routes"],
                    "speedup": last_slower["speedup"],
                },
                "first_faster": {
                    "sequence_tokens": break_even["sequence_tokens"],
                    "routes": break_even["routes"],
                    "speedup": break_even["speedup"],
                },
                "selected_min_routes": grouped.min_routes,
            }
            if break_even is not None and last_slower is not None
            else None
        ),
        "forced_grouped_crossover": crossover_results,
        "results": results,
        "acceptance": {
            "all_sequences_parity": parity_ok,
            "forced_crossover_parity": crossover_parity_ok,
            "measured_break_even_bracketed": crossover_bracketed,
            "selected_threshold_matches_measurement": threshold_matches,
            "speedup_256_at_least_1_5": performance_ok,
            "working_peak_delta_at_most_512_mib": memory_ok,
            "accepted": accepted,
        },
        "runtime_default_changed": False,
    }
    print(json.dumps(output, indent=2), flush=True)
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
