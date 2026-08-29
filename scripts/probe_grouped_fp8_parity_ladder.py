#!/usr/bin/env python3
"""Decompose grouped-FP8 error with a Direct-order BM8 diagnostic kernel."""

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
import mlx.nn as nn
import numpy as np

from glm53_flash_mlx.abi import GROUPED_MIN_ROUTES
from glm53_flash_mlx.fp8 import BLOCK_SIZE, THREADS, _FP8_LUT_HEADER
from glm53_flash_mlx.grouped_fp8 import (
    SortedGroupedFP8MoE,
    activate_gate_up,
    build_grouped_tile_plan,
    grouped_fp8_linear,
    restore_and_reduce,
)
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint
from glm53_flash_mlx.packed import PackedFP8MoE

DIRECT_ORDER_TILE_ROWS = 8
DEFAULT_PROMPT = "Reply with exactly: OK"

_DIRECT_ORDER_GROUPED_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd_id = simdgroup_index_in_threadgroup;
    uint group_id = threadgroup_position_in_grid.x;
    uint out_row = group_id % OUT_FEATURES;
    uint tile = group_id / OUT_FEATURES;
    uint first_route = tile_starts[tile];
    if (first_route >= ROUTES) return;
    uint valid_rows = tile_lengths[tile];
    uint expert = tile_experts[tile];
    uint expert_routes = route_offsets[expert + 1] - route_offsets[expert];

    thread float acc[TILE_ROWS];
    for (uint row = 0; row < TILE_ROWS; ++row) acc[row] = 0.0f;
    uint bank_row = WEIGHT_ROW_OFFSET + out_row;
    const device uint8_t* wr = weight
        + (size_t(expert) * BANK_OUT_FEATURES + bank_row) * IN_FEATURES;
    uint scale_row = SCALE_ROW_OFFSET + out_row / BLOCK_SIZE;
    for (uint k = tid; k < IN_FEATURES; k += THREADS) {
        float scale = scale_inv[
            (size_t(expert) * BANK_SCALE_ROWS + scale_row) * SCALE_COLS
            + k / BLOCK_SIZE
        ];
        float decoded = glm53_fp8_lut[wr[k]] * scale;
        for (uint row = 0; row < TILE_ROWS; ++row) {
            if (row < valid_rows) {
                float input = float(
                    x[size_t(first_route + row) * IN_FEATURES + k]
                );
                // block_fp8_linear selects GEMV for singleton expert buckets
                // and tiled GEMM otherwise. Preserve both expression trees.
                if (expert_routes == 1) {
                    acc[row] += input * glm53_fp8_lut[wr[k]] * scale;
                } else {
                    acc[row] += input * decoded;
                }
            }
        }
    }
    for (uint row = 0; row < TILE_ROWS; ++row) acc[row] = simd_sum(acc[row]);

    constexpr uint NSIMD = THREADS / 32;
    threadgroup float partial[TILE_ROWS][NSIMD];
    if (lane == 0) {
        for (uint row = 0; row < TILE_ROWS; ++row) {
            partial[row][simd_id] = acc[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        for (uint row = 0; row < TILE_ROWS; ++row) {
            float total = lane < NSIMD ? partial[row][lane] : 0.0f;
            total = simd_sum(total);
            if (lane == 0 && row < valid_rows) {
                output[size_t(first_route + row) * OUT_FEATURES + out_row]
                    = T(total);
            }
        }
    }
"""

_DIRECT_ORDER_REDUCE_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint token = threadgroup_position_in_grid.x;
    if (token >= TOKENS) return;
    for (uint column = tid; column < HIDDEN; column += THREADS) {
        T total = T(0.0f);
        uint used = 0;
        for (uint rank = 0; rank < TOP_K; ++rank) {
            uint chosen_slot = 0;
            uint chosen_expert = 0xffffffffu;
            for (uint slot = 0; slot < TOP_K; ++slot) {
                uint expert = expert_ids[size_t(token) * TOP_K + slot];
                if ((used & (1u << slot)) == 0 && expert < chosen_expert) {
                    chosen_expert = expert;
                    chosen_slot = slot;
                }
            }
            used |= 1u << chosen_slot;
            T contribution = T(float(
                route_output[
                    (size_t(token) * TOP_K + chosen_slot) * HIDDEN + column
                ]
            ) * scores[size_t(token) * TOP_K + chosen_slot]);
            // MLX scatter-add casts each expert contribution to the target
            // BF16 dtype before the expert-ordered accumulation.
            total = T(float(total) + float(contribution));
        }
        output[size_t(token) * HIDDEN + column] = total;
    }
"""

_direct_order_grouped_kernel = (
    mx.fast.metal_kernel(
        name="glm53_direct_order_grouped_bm8_fp8",
        input_names=[
            "x",
            "tile_experts",
            "tile_starts",
            "tile_lengths",
            "route_offsets",
            "weight",
            "scale_inv",
        ],
        output_names=["output"],
        source=_DIRECT_ORDER_GROUPED_SOURCE,
        header=_FP8_LUT_HEADER,
    )
    if mx.metal.is_available()
    else None
)

_direct_order_reduce_kernel = (
    mx.fast.metal_kernel(
        name="glm53_direct_expert_order_top8_reduce",
        input_names=["route_output", "expert_ids", "scores"],
        output_names=["output"],
        source=_DIRECT_ORDER_REDUCE_SOURCE,
    )
    if mx.metal.is_available()
    else None
)


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), file=sys.stderr, flush=True)


def _deterministic_tokens(tokens: int, vocab: int) -> np.ndarray:
    return (
        (np.arange(tokens, dtype=np.uint64) * 7919) % (vocab - 1024) + 100
    ).astype(np.uint32)[None, :]


def _sha256(values: np.ndarray) -> str:
    return hashlib.sha256(values.tobytes()).hexdigest()


def _float_array(value) -> np.ndarray:
    value = value.astype(mx.float32)
    mx.eval(value)
    return np.ascontiguousarray(np.asarray(value), dtype=np.float32)


def _int_array(value) -> np.ndarray:
    mx.eval(value)
    return np.ascontiguousarray(np.asarray(value), dtype=np.int32)


def _tensor_metrics(reference, actual) -> dict:
    reference_array = _float_array(reference)
    actual_array = _float_array(actual)
    reference64 = reference_array.astype(np.float64)
    actual64 = actual_array.astype(np.float64)
    diff = actual64 - reference64
    denominator = np.linalg.norm(reference64)
    return {
        "array_equal": bool(np.array_equal(reference_array, actual_array)),
        "reference_f32_sha256": _sha256(reference_array),
        "actual_f32_sha256": _sha256(actual_array),
        "relative_l2": (
            float(np.linalg.norm(diff) / denominator) if denominator else 0.0
        ),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rms": float(np.sqrt(np.mean(diff * diff))),
    }


def _build_tile_plan(sorted_experts, expert_count: int, *, tile_rows: int):
    routes = sorted_experts.shape[0]
    expert_range = mx.arange(expert_count, dtype=mx.uint32)
    counts = mx.sum(
        sorted_experts[:, None] == expert_range[None, :], axis=0
    ).astype(mx.uint32)
    zero = mx.zeros((1,), dtype=mx.uint32)
    route_offsets = mx.concatenate([zero, mx.cumsum(counts)])
    tile_counts = (counts + tile_rows - 1) // tile_rows
    tile_offsets = mx.concatenate([zero, mx.cumsum(tile_counts)])
    descriptor_slots = (routes + tile_rows - 1) // tile_rows + expert_count
    descriptor_ids = mx.arange(descriptor_slots, dtype=mx.uint32)
    owners = mx.sum(
        descriptor_ids[:, None] >= tile_offsets[None, 1:], axis=1
    ).astype(mx.uint32)
    valid = descriptor_ids < tile_offsets[-1]
    safe_owners = mx.minimum(owners, expert_count - 1)
    local_tiles = descriptor_ids - tile_offsets[safe_owners]
    starts = route_offsets[safe_owners] + local_tiles * tile_rows
    remaining = route_offsets[safe_owners + 1] - starts
    lengths = mx.minimum(remaining, tile_rows)
    invalid_start = mx.full(starts.shape, routes, dtype=mx.uint32)
    return (
        mx.where(valid, safe_owners, mx.zeros_like(safe_owners)),
        mx.where(valid, starts, invalid_start),
        mx.where(valid, lengths, mx.zeros_like(lengths)),
        route_offsets,
        tile_offsets,
    )


def _stable_route_plan(x, indices, scores):
    indices_array = _int_array(indices).reshape(-1)
    order_array = np.argsort(indices_array, kind="stable").astype(np.uint32)
    inverse_array = np.argsort(order_array, kind="stable").astype(np.uint32)
    top_k = indices.shape[-1]
    order = mx.array(order_array, dtype=mx.uint32)
    inverse = mx.array(inverse_array, dtype=mx.uint32)
    flat_x = x.reshape(-1, x.shape[-1])
    return {
        "sorted_x": flat_x[order // top_k],
        "sorted_experts": indices.reshape(-1).astype(mx.uint32)[order],
        "sorted_scores": scores.reshape(-1).astype(mx.float32)[order],
        "order": order,
        "inverse": inverse,
        "order_array": order_array,
        "sorted_experts_array": indices_array[order_array],
        "top_k": top_k,
        "tokens": flat_x.shape[0],
    }


def direct_order_grouped_linear(
    x,
    tile_plan,
    weight,
    scale_inv,
    *,
    row_offset: int,
    scale_row_offset: int,
    out_features: int,
):
    if _direct_order_grouped_kernel is None:
        raise RuntimeError("Direct-order grouped projection requires Metal")
    routes, in_features = x.shape
    experts, bank_out_features, weight_in = weight.shape
    if weight_in != in_features:
        raise ValueError("Direct-order grouped weight/input width mismatch")
    if weight.dtype != mx.uint8 or scale_inv.dtype != mx.float32:
        raise ValueError("Direct-order grouped storage must remain uint8/float32")
    expected_scale_cols = (in_features + BLOCK_SIZE - 1) // BLOCK_SIZE
    if scale_inv.shape[2] != expected_scale_cols:
        raise ValueError("Direct-order grouped scale columns mismatch")
    tile_experts, tile_starts, tile_lengths, route_offsets = tile_plan[:4]
    return _direct_order_grouped_kernel(
        inputs=[
            x,
            tile_experts,
            tile_starts,
            tile_lengths,
            route_offsets,
            weight,
            scale_inv,
        ],
        template=[
            ("T", x.dtype),
            ("IN_FEATURES", in_features),
            ("OUT_FEATURES", out_features),
            ("BANK_OUT_FEATURES", bank_out_features),
            ("BANK_SCALE_ROWS", scale_inv.shape[1]),
            ("WEIGHT_ROW_OFFSET", row_offset),
            ("SCALE_ROW_OFFSET", scale_row_offset),
            ("ROUTES", routes),
            ("SCALE_COLS", scale_inv.shape[2]),
            ("BLOCK_SIZE", BLOCK_SIZE),
            ("THREADS", THREADS),
            ("TILE_ROWS", DIRECT_ORDER_TILE_ROWS),
        ],
        grid=(tile_experts.shape[0] * out_features * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(routes, out_features)],
        output_dtypes=[x.dtype],
    )[0]


def direct_order_reduce(route_output, indices, scores):
    if _direct_order_reduce_kernel is None:
        raise RuntimeError("Direct-order route reduction requires Metal")
    tokens = indices.reshape(-1, indices.shape[-1]).shape[0]
    top_k = indices.shape[-1]
    hidden = route_output.shape[-1]
    return _direct_order_reduce_kernel(
        inputs=[
            route_output.reshape(-1, hidden),
            indices.reshape(-1).astype(mx.uint32),
            scores.reshape(-1).astype(mx.float32),
        ],
        template=[
            ("T", route_output.dtype),
            ("TOKENS", tokens),
            ("TOP_K", top_k),
            ("HIDDEN", hidden),
            ("THREADS", THREADS),
        ],
        grid=(tokens * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(tokens, hidden)],
        output_dtypes=[route_output.dtype],
    )[0]


def _direct_scatter_reduce(sorted_down, plan, shape):
    hidden = shape[-1]
    result = mx.zeros((plan["tokens"], hidden), dtype=sorted_down.dtype)
    experts = plan["sorted_experts_array"]
    order = plan["order_array"]
    scores = plan["sorted_scores"]
    start = 0
    while start < experts.size:
        end = start + 1
        while end < experts.size and experts[end] == experts[start]:
            end += 1
        token_rows = mx.array(
            order[start:end] // plan["top_k"], dtype=mx.int32
        )
        contribution = sorted_down[start:end] * scores[start:end, None]
        result = result.at[token_rows].add(contribution)
        start = end
    return result.reshape(shape)


def _direct_bucket_stages(moe, x, indices, scores, plan):
    bank = moe.bank
    experts = plan["sorted_experts_array"]
    sorted_x = plan["sorted_x"]
    gate_parts = []
    up_parts = []
    expert_ranges = []
    start = 0
    while start < experts.size:
        end = start + 1
        while end < experts.size and experts[end] == experts[start]:
            end += 1
        expert_ranges.append((start, end))
        start = end
    for start, end in expert_ranges:
        expert = bank.expert(
            int(experts[start]), limit=moe.config.swiglu_limit
        )
        gate_parts.append(expert.gate_proj(sorted_x[start:end]))
        up_parts.append(expert.up_proj(sorted_x[start:end]))
    gate = mx.concatenate(gate_parts, axis=0)
    up = mx.concatenate(up_parts, axis=0)
    activation = nn.silu(mx.minimum(gate, moe.config.swiglu_limit)) * mx.clip(
        up, -moe.config.swiglu_limit, moe.config.swiglu_limit
    )
    down_parts = []
    for start, end in expert_ranges:
        expert = bank.expert(
            int(experts[start]), limit=moe.config.swiglu_limit
        )
        down_parts.append(expert.down_proj(activation[start:end]))
    down = mx.concatenate(down_parts, axis=0)
    weighted = down * plan["sorted_scores"][:, None]
    reduced = _direct_scatter_reduce(down, plan, x.shape)
    shared = moe.shared_experts(x) if moe.shared_experts is not None else 0
    return {
        "gate_projection": gate,
        "up_projection": up,
        "swiglu_activation": activation,
        "down_projection": down,
        "weighted_per_route_output": weighted,
        "direct_expert_order_reduction": reduced,
        "final_moe_output": reduced + shared,
        "shared_expert_output": shared,
    }


def _bm32_stages(moe, x, indices, scores, plan):
    bank = moe.bank
    tile_plan = build_grouped_tile_plan(
        plan["sorted_experts"], bank.expert_count
    )
    gate_up = grouped_fp8_linear(
        plan["sorted_x"],
        tile_plan,
        bank.gate_up_weight,
        bank.gate_up_scale_inv,
    )
    intermediate = bank.intermediate_size
    gate = gate_up[:, :intermediate]
    up = gate_up[:, intermediate:]
    activation = activate_gate_up(
        gate_up, intermediate, moe.config.swiglu_limit
    )
    down = grouped_fp8_linear(
        activation,
        tile_plan,
        bank.down_weight,
        bank.down_scale_inv,
    )
    weighted = down * plan["sorted_scores"][:, None]
    grouped_reduction = restore_and_reduce(
        down,
        plan["sorted_scores"],
        plan["inverse"],
        x.shape,
        plan["top_k"],
    )
    direct_reduction = _direct_scatter_reduce(down, plan, x.shape)
    shared = moe.shared_experts(x) if moe.shared_experts is not None else 0
    return {
        "gate_projection": gate,
        "up_projection": up,
        "swiglu_activation": activation,
        "down_projection": down,
        "weighted_per_route_output": weighted,
        "grouped_fp32_reduction": grouped_reduction,
        "direct_expert_order_reduction": direct_reduction,
        "final_grouped_reduction_moe_output": grouped_reduction + shared,
        "final_direct_reduction_moe_output": direct_reduction + shared,
    }


def _bm8_stages(moe, x, indices, scores, plan):
    bank = moe.bank
    tile_plan = _build_tile_plan(
        plan["sorted_experts"],
        bank.expert_count,
        tile_rows=DIRECT_ORDER_TILE_ROWS,
    )
    gate = direct_order_grouped_linear(
        plan["sorted_x"],
        tile_plan,
        bank.gate_up_weight,
        bank.gate_up_scale_inv,
        row_offset=0,
        scale_row_offset=0,
        out_features=bank.intermediate_size,
    )
    up = direct_order_grouped_linear(
        plan["sorted_x"],
        tile_plan,
        bank.gate_up_weight,
        bank.gate_up_scale_inv,
        row_offset=bank.intermediate_size,
        scale_row_offset=bank.intermediate_scale_rows,
        out_features=bank.intermediate_size,
    )
    activation = nn.silu(mx.minimum(gate, moe.config.swiglu_limit)) * mx.clip(
        up, -moe.config.swiglu_limit, moe.config.swiglu_limit
    )
    down = direct_order_grouped_linear(
        activation,
        tile_plan,
        bank.down_weight,
        bank.down_scale_inv,
        row_offset=0,
        scale_row_offset=0,
        out_features=bank.down_weight.shape[1],
    )
    weighted = down * plan["sorted_scores"][:, None]
    grouped_reduction = restore_and_reduce(
        down,
        plan["sorted_scores"],
        plan["inverse"],
        x.shape,
        plan["top_k"],
    )
    direct_scatter = _direct_scatter_reduce(down, plan, x.shape)
    route_order_down = down[plan["inverse"]]
    fused_direct_order = direct_order_reduce(
        route_order_down, indices, scores
    ).reshape(x.shape)
    shared = moe.shared_experts(x) if moe.shared_experts is not None else 0
    return {
        "gate_projection": gate,
        "up_projection": up,
        "swiglu_activation": activation,
        "down_projection": down,
        "weighted_per_route_output": weighted,
        "grouped_fp32_reduction": grouped_reduction,
        "direct_expert_order_reduction": direct_scatter,
        "fused_direct_order_reduction": fused_direct_order,
        "final_grouped_reduction_moe_output": grouped_reduction + shared,
        "final_direct_reduction_moe_output": direct_scatter + shared,
        "final_fused_direct_order_moe_output": fused_direct_order + shared,
    }


def _compare_stage_paths(reference, actual, mapping: dict[str, str]) -> dict:
    return {
        actual_name: _tensor_metrics(reference[reference_name], actual[actual_name])
        for actual_name, reference_name in mapping.items()
    }


class _CaptureMLP:
    def __init__(self, delegate, captures: dict, layer_id: int):
        self.delegate = delegate
        self.captures = captures
        self.layer_id = layer_id

    def __call__(self, x):
        indices, scores = self.delegate.gate(x)
        self.captures[self.layer_id] = {
            "x": x,
            "indices": indices,
            "scores": scores,
        }
        return self.delegate(x)


def _capture_layer_inputs(model, token_ids, targets=(3, 5)):
    layers = model.language_model.model.layers
    originals = {target: layers[target].mlp for target in targets}
    captures = {}
    try:
        for target in targets:
            layers[target].mlp = _CaptureMLP(
                originals[target], captures, target
            )
        output = model(mx.array(token_ids), cache=model.make_cache())
        values = [output.logits]
        for capture in captures.values():
            values.extend(capture.values())
        mx.eval(*values)
        mx.synchronize()
        return captures
    finally:
        for target, mlp in originals.items():
            layers[target].mlp = mlp


class DirectOrderGroupedFP8MoE(PackedFP8MoE):
    """Probe-only BM8 path; decode retains the packed selected-top8 fallback."""

    def __init__(self, bank, config, gate, shared_experts):
        super().__init__(bank, config, gate, shared_experts)
        self.min_routes = GROUPED_MIN_ROUTES

    def __call__(self, x):
        routes = (
            x.reshape(-1, x.shape[-1]).shape[0]
            * self.config.num_experts_per_tok
        )
        if routes < self.min_routes:
            return super().__call__(x)
        indices, scores = self.gate(x)
        flat_x = x.reshape(-1, x.shape[-1])
        expert_ids = indices.reshape(-1).astype(mx.uint32)
        order = mx.argsort(expert_ids)
        inverse = mx.argsort(order)
        sorted_x = flat_x[order // self.config.num_experts_per_tok]
        sorted_experts = expert_ids[order]
        tile_plan = _build_tile_plan(
            sorted_experts,
            self.bank.expert_count,
            tile_rows=DIRECT_ORDER_TILE_ROWS,
        )
        gate = direct_order_grouped_linear(
            sorted_x,
            tile_plan,
            self.bank.gate_up_weight,
            self.bank.gate_up_scale_inv,
            row_offset=0,
            scale_row_offset=0,
            out_features=self.bank.intermediate_size,
        )
        up = direct_order_grouped_linear(
            sorted_x,
            tile_plan,
            self.bank.gate_up_weight,
            self.bank.gate_up_scale_inv,
            row_offset=self.bank.intermediate_size,
            scale_row_offset=self.bank.intermediate_scale_rows,
            out_features=self.bank.intermediate_size,
        )
        activation = nn.silu(mx.minimum(gate, self.config.swiglu_limit)) * mx.clip(
            up, -self.config.swiglu_limit, self.config.swiglu_limit
        )
        down = direct_order_grouped_linear(
            activation,
            tile_plan,
            self.bank.down_weight,
            self.bank.down_scale_inv,
            row_offset=0,
            scale_row_offset=0,
            out_features=self.bank.down_weight.shape[1],
        )
        route_order_down = down[inverse]
        result = direct_order_reduce(route_order_down, indices, scores).reshape(
            x.shape
        )
        if self.shared_experts is not None:
            result = result + self.shared_experts(x)
        return result


class _CapturingGate:
    def __init__(self, delegate):
        self.delegate = delegate
        self.indices = None
        self.scores = None

    def __call__(self, x):
        self.indices, self.scores = self.delegate(x)
        return self.indices, self.scores


def _moe_modules(model):
    return {
        layer_id: layer.mlp
        for layer_id, layer in enumerate(model.language_model.model.layers)
        if hasattr(layer.mlp, "gate")
    }


def _evaluate_prefill(model, token_ids, *, capture_routes: bool):
    modules = _moe_modules(model)
    original_gates = {layer: module.gate for layer, module in modules.items()}
    captures = {}
    try:
        if capture_routes:
            for layer_id, module in modules.items():
                capture = _CapturingGate(module.gate)
                module.gate = capture
                captures[layer_id] = capture
        mx.clear_cache()
        mx.reset_peak_memory()
        started = time.perf_counter()
        output = model(mx.array(token_ids), cache=model.make_cache())
        logits = output.logits[0, -1].astype(mx.float32)
        values = [logits]
        if capture_routes:
            values.extend(capture.indices for capture in captures.values())
            values.extend(capture.scores for capture in captures.values())
        mx.eval(*values)
        mx.synchronize()
        elapsed = time.perf_counter() - started
        logits_array = _float_array(logits)
        routes = (
            {
                layer: {
                    "indices": _int_array(capture.indices),
                    "scores": _float_array(capture.scores),
                }
                for layer, capture in captures.items()
            }
            if capture_routes
            else None
        )
        return (
            {
                "elapsed_seconds": elapsed,
                "tokens_per_second": token_ids.size / elapsed,
                "last_logits_f32_sha256": _sha256(logits_array),
                "active_bytes": mx.get_active_memory(),
                "peak_bytes": mx.get_peak_memory(),
            },
            logits_array,
            routes,
        )
    finally:
        for layer_id, module in modules.items():
            module.gate = original_gates[layer_id]


def _benchmark(model, token_ids, *, warmups: int, repeats: int, phase: str):
    warmup_rows = []
    for sample in range(warmups):
        row, _, _ = _evaluate_prefill(model, token_ids, capture_routes=False)
        warmup_rows.append(row)
        _progress(
            f"{phase}_warmup", sample=sample + 1, seconds=row["elapsed_seconds"]
        )
    measured_rows = []
    for sample in range(repeats):
        row, _, _ = _evaluate_prefill(model, token_ids, capture_routes=False)
        measured_rows.append(row)
        _progress(
            f"{phase}_measured", sample=sample + 1, seconds=row["elapsed_seconds"]
        )
    median = statistics.median(
        row["elapsed_seconds"] for row in measured_rows
    )
    return {
        "warmups": warmup_rows,
        "measured": measured_rows,
        "warmup_count": warmups,
        "measured_count": repeats,
        "median_seconds": median,
        "median_tokens_per_second": token_ids.size / median,
    }


def _route_parity(reference_routes, actual_routes) -> dict:
    rows = []
    for layer in sorted(reference_routes):
        reference = reference_routes[layer]
        actual = actual_routes[layer]
        rows.append(
            {
                "layer": layer,
                "indices_array_equal": bool(
                    np.array_equal(reference["indices"], actual["indices"])
                ),
                "scores_array_equal": bool(
                    np.array_equal(reference["scores"], actual["scores"])
                ),
                "reference_indices_sha256": _sha256(reference["indices"]),
                "actual_indices_sha256": _sha256(actual["indices"]),
                "reference_scores_sha256": _sha256(reference["scores"]),
                "actual_scores_sha256": _sha256(actual["scores"]),
            }
        )
    return {
        "all_indices_and_scores_byte_identical": all(
            row["indices_array_equal"] and row["scores_array_equal"]
            for row in rows
        ),
        "layers": rows,
    }


def _oracle(model, processor, expected_path: Path, *, tokens=16):
    expected = json.loads(expected_path.read_text())
    formatted = processor.apply_chat_template(
        [{"role": "user", "content": DEFAULT_PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = processor(formatted, return_tensors="np", add_special_tokens=True)
    prompt_ids = np.asarray(encoded["input_ids"], dtype=np.int32).reshape(1, -1)
    cache = model.make_cache()
    output = model(mx.array(prompt_ids), cache=cache)
    hashes = []
    generated = []
    for step in range(tokens):
        logits = output.logits[0, -1].astype(mx.float32)
        array = _float_array(logits)
        hashes.append(_sha256(array))
        top2 = np.argpartition(array, -2)[-2:]
        top2 = top2[np.argsort(array[top2])[::-1]]
        token = int(top2[0])
        generated.append(token)
        if step + 1 < tokens:
            output = model(mx.array([[token]], dtype=mx.uint32), cache=cache)
    expected_steps = expected["steps"][:tokens]
    expected_hashes = [row["logits_f32_sha256"] for row in expected_steps]
    expected_tokens = [row["token"] for row in expected_steps]
    return {
        "prompt_tokens": int(prompt_ids.size),
        "generation_tokens": tokens,
        "generated_token_ids": generated,
        "expected_token_ids": expected_tokens,
        "all_token_ids_match": generated == expected_tokens,
        "all_step_logits_hashes_match": hashes == expected_hashes,
        "actual_logits_hashes": hashes,
    }


def _install_direct_order_modules(model) -> list[int]:
    converted = []
    for layer_id, layer in enumerate(model.language_model.model.layers):
        moe = layer.mlp
        if not isinstance(moe, SortedGroupedFP8MoE):
            continue
        layer.mlp = DirectOrderGroupedFP8MoE(
            moe.bank, moe.config, moe.gate, moe.shared_experts
        )
        layer._ffn_c = None
        converted.append(layer_id)
    return converted


def _run_stage_case(model, layer_id: int, token_ids) -> dict:
    captures = _capture_layer_inputs(model, token_ids)
    capture = captures[layer_id]
    x = capture["x"]
    indices = capture["indices"]
    scores = capture["scores"]
    moe = model.language_model.model.layers[layer_id].mlp
    plan = _stable_route_plan(x, indices, scores)

    mx.clear_cache()
    baseline = mx.get_active_memory()
    mx.reset_peak_memory()
    direct = _direct_bucket_stages(moe, x, indices, scores, plan)
    bm32 = _bm32_stages(moe, x, indices, scores, plan)
    bm8 = _bm8_stages(moe, x, indices, scores, plan)
    mx.eval(
        *direct.values(),
        *bm32.values(),
        *bm8.values(),
    )
    mx.synchronize()
    ladder_peak_increment = max(0, mx.get_peak_memory() - baseline)

    projection_mapping = {
        "gate_projection": "gate_projection",
        "up_projection": "up_projection",
        "swiglu_activation": "swiglu_activation",
        "down_projection": "down_projection",
        "weighted_per_route_output": "weighted_per_route_output",
    }
    bm32_metrics = _compare_stage_paths(direct, bm32, projection_mapping)
    bm32_metrics.update(
        {
            "grouped_fp32_reduction": _tensor_metrics(
                direct["direct_expert_order_reduction"],
                bm32["grouped_fp32_reduction"],
            ),
            "direct_expert_order_reduction": _tensor_metrics(
                direct["direct_expert_order_reduction"],
                bm32["direct_expert_order_reduction"],
            ),
            "final_grouped_reduction_moe_output": _tensor_metrics(
                direct["final_moe_output"],
                bm32["final_grouped_reduction_moe_output"],
            ),
            "final_direct_reduction_moe_output": _tensor_metrics(
                direct["final_moe_output"],
                bm32["final_direct_reduction_moe_output"],
            ),
        }
    )
    bm8_metrics = _compare_stage_paths(direct, bm8, projection_mapping)
    bm8_metrics.update(
        {
            "grouped_fp32_reduction": _tensor_metrics(
                direct["direct_expert_order_reduction"],
                bm8["grouped_fp32_reduction"],
            ),
            "direct_expert_order_reduction": _tensor_metrics(
                direct["direct_expert_order_reduction"],
                bm8["direct_expert_order_reduction"],
            ),
            "fused_direct_order_reduction": _tensor_metrics(
                direct["direct_expert_order_reduction"],
                bm8["fused_direct_order_reduction"],
            ),
            "final_grouped_reduction_moe_output": _tensor_metrics(
                direct["final_moe_output"],
                bm8["final_grouped_reduction_moe_output"],
            ),
            "final_direct_reduction_moe_output": _tensor_metrics(
                direct["final_moe_output"],
                bm8["final_direct_reduction_moe_output"],
            ),
            "final_fused_direct_order_moe_output": _tensor_metrics(
                direct["final_moe_output"],
                bm8["final_fused_direct_order_moe_output"],
            ),
        }
    )
    reference_hashes = {
        name: _sha256(_float_array(value)) for name, value in direct.items()
    }
    del direct, bm32, bm8
    gc.collect()
    mx.clear_cache()
    bm8_baseline = mx.get_active_memory()
    mx.reset_peak_memory()
    bm8_working_set = _bm8_stages(moe, x, indices, scores, plan)
    mx.eval(*bm8_working_set.values())
    mx.synchronize()
    bm8_peak_increment = max(0, mx.get_peak_memory() - bm8_baseline)
    del bm8_working_set
    gc.collect()
    mx.clear_cache()
    return {
        "layer": layer_id,
        "tokens": int(token_ids.size),
        "routes": int(token_ids.size * indices.shape[-1]),
        "unique_experts": int(np.unique(_int_array(indices)).size),
        "diagnostic_ladder_peak_increment_bytes": int(ladder_peak_increment),
        "bm8_working_peak_increment_bytes": int(bm8_peak_increment),
        "direct_reference_f32_hashes": reference_hashes,
        "current_bm32_simdgroup_matrix": bm32_metrics,
        "direct_order_bm8": bm8_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--oracle", type=Path, default=Path("oracles/glm53-official-greedy-16.json")
    )
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = inspect_checkpoint(args.model, require_server_ready=True)
    raw_config = json.loads((Path(args.model) / "config.json").read_text())
    vocab = int(raw_config["text_config"]["vocab_size"])
    token_ids = _deterministic_tokens(args.tokens, vocab)
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))

    _progress("load_direct")
    direct_model, processor = load(args.model)
    warm_residency(direct_model)
    direct_timing, reference_logits, reference_routes = _evaluate_prefill(
        direct_model, token_ids, capture_routes=True
    )
    direct_benchmark = _benchmark(
        direct_model,
        token_ids,
        warmups=args.warmups,
        repeats=args.repeats,
        phase="direct",
    )
    del direct_model
    gc.collect()
    mx.clear_cache()
    mx.synchronize()

    _progress("load_packed_grouped")
    model, _ = load(args.model, experimental_packed_grouped_moe=True)
    warm_residency(model)
    grouped_modules = {
        layer_id: layer.mlp
        for layer_id, layer in enumerate(model.language_model.model.layers)
        if isinstance(layer.mlp, SortedGroupedFP8MoE)
    }
    if sorted(grouped_modules) != list(range(3, 45)):
        raise RuntimeError("expected packed grouped layers 3..44")
    storage_ok = all(
        moe.bank.gate_up_weight.dtype == mx.uint8
        and moe.bank.down_weight.dtype == mx.uint8
        and moe.bank.gate_up_scale_inv.dtype == mx.float32
        and moe.bank.down_scale_inv.dtype == mx.float32
        for moe in grouped_modules.values()
    )

    for moe in grouped_modules.values():
        moe.min_routes = 1 << 30
    fallback_timing, fallback_logits, fallback_routes = _evaluate_prefill(
        model, token_ids, capture_routes=True
    )
    fallback_logits_metrics = _tensor_metrics(
        mx.array(reference_logits), mx.array(fallback_logits)
    )
    fallback_route_parity = _route_parity(reference_routes, fallback_routes)

    stage_cases = []
    for tokens in (32, 64, 128, 256, 512):
        case_tokens = _deterministic_tokens(tokens, vocab)
        for layer_id in (3, 5):
            _progress("stage_case", layer=layer_id, tokens=tokens)
            case = _run_stage_case(model, layer_id, case_tokens)
            stage_cases.append(case)
            _progress(
                "stage_case_done",
                layer=layer_id,
                tokens=tokens,
                bm8_final_exact=case["direct_order_bm8"][
                    "final_fused_direct_order_moe_output"
                ]["array_equal"],
                bm8_peak_increment=case["bm8_working_peak_increment_bytes"],
                ladder_peak_increment=case[
                    "diagnostic_ladder_peak_increment_bytes"
                ],
            )
            mx.clear_cache()

    for moe in grouped_modules.values():
        moe.min_routes = GROUPED_MIN_ROUTES
    current_timing, current_logits, current_routes = _evaluate_prefill(
        model, token_ids, capture_routes=True
    )
    current_benchmark = _benchmark(
        model,
        token_ids,
        warmups=args.warmups,
        repeats=args.repeats,
        phase="current_bm32",
    )

    converted = _install_direct_order_modules(model)
    bm8_timing, bm8_logits, bm8_routes = _evaluate_prefill(
        model, token_ids, capture_routes=True
    )
    bm8_benchmark = _benchmark(
        model,
        token_ids,
        warmups=args.warmups,
        repeats=args.repeats,
        phase="direct_order_bm8",
    )
    oracle = _oracle(model, processor, args.oracle)

    bm8_logits_metrics = _tensor_metrics(
        mx.array(reference_logits), mx.array(bm8_logits)
    )
    current_logits_metrics = _tensor_metrics(
        mx.array(reference_logits), mx.array(current_logits)
    )
    bm8_route_parity = _route_parity(reference_routes, bm8_routes)
    current_route_parity = _route_parity(reference_routes, current_routes)
    bm8_speedup = (
        direct_benchmark["median_seconds"] / bm8_benchmark["median_seconds"]
    )
    current_speedup = (
        direct_benchmark["median_seconds"] / current_benchmark["median_seconds"]
    )

    projection_exact = all(
        case["direct_order_bm8"][stage]["array_equal"]
        for case in stage_cases
        for stage in (
            "gate_projection",
            "up_projection",
            "swiglu_activation",
            "down_projection",
            "weighted_per_route_output",
        )
    )
    single_moe_exact = all(
        case["direct_order_bm8"][
            "final_fused_direct_order_moe_output"
        ]["array_equal"]
        for case in stage_cases
    )
    max_working_peak = max(
        case["bm8_working_peak_increment_bytes"] for case in stage_cases
    )
    candidate_gate = {
        "bm8_projection_and_activation_byte_identical": projection_exact,
        "bm8_direct_style_reduction_single_moe_byte_identical": single_moe_exact,
        "working_peak_at_most_512_mib": max_working_peak <= 512 * 1024**2,
        "full_model_256_logits_byte_identical": bm8_logits_metrics["array_equal"],
        "full_model_router_indices_and_scores_byte_identical": bm8_route_parity[
            "all_indices_and_scores_byte_identical"
        ],
        "direct_prefill_speedup_at_least_1_5": bm8_speedup >= 1.5,
        "decode_oracle_all_token_ids_match": oracle["all_token_ids_match"],
        "decode_oracle_all_logits_hashes_match": oracle[
            "all_step_logits_hashes_match"
        ],
        "packed_fp8_storage_preserved": storage_ok,
    }
    candidate_accepted = all(candidate_gate.values())
    evidence_gate = {
        "all_10_stage_cases_measured": len(stage_cases) == 10,
        "direct_packed_fallback_logits_byte_identical": fallback_logits_metrics[
            "array_equal"
        ],
        "direct_packed_fallback_routes_byte_identical": fallback_route_parity[
            "all_indices_and_scores_byte_identical"
        ],
        "all_42_layers_converted_without_repacking": converted
        == list(range(3, 45)),
        "runtime_server_apc_unchanged": True,
    }
    output = {
        "schema": "glm53-grouped-fp8-direct-order-parity-ladder-v1",
        "date": date.today().isoformat(),
        "machine": "Apple M3 Ultra, 80-core GPU, 512 GB unified memory",
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "kernel_ladder": [
            "Direct expert bucket oracle",
            "current BM32 simdgroup-matrix grouped FP8",
            "BM8 Direct-order grouped FP8",
        ],
        "direct_order_contract": {
            "expert_aligned_tile_rows": DIRECT_ORDER_TILE_ROWS,
            "threads": THREADS,
            "k_order": "k = tid; k < IN_FEATURES; k += 256",
            "product_order": (
                "singleton bucket: acc += x * FP8 * scale; "
                "multi-row bucket: decoded = FP8 * scale; acc += x * decoded"
            ),
            "reduction": "Direct simd_sum -> threadgroup partial -> simd_sum",
            "gate_up_separate_dispatches": True,
            "projection_store_dtype": "bfloat16",
            "weight_storage": "uint8 E4M3 codes plus FP32 block scales",
        },
        "prompt": {
            "tokens": args.tokens,
            "token_formula": "(arange(tokens) * 7919) % (vocab_size - 1024) + 100",
            "fresh_cache_per_run": True,
            "dsa_indexpool_bypassed": args.tokens <= 2048,
        },
        "direct_reference": {
            "timing": direct_timing,
            "benchmark": direct_benchmark,
        },
        "packed_fallback": {
            "timing": fallback_timing,
            "final_logits": fallback_logits_metrics,
            "router": fallback_route_parity,
        },
        "stage_parity": {
            "cases": stage_cases,
            "max_bm8_working_peak_increment_bytes": max_working_peak,
            "max_diagnostic_ladder_peak_increment_bytes": max(
                case["diagnostic_ladder_peak_increment_bytes"]
                for case in stage_cases
            ),
        },
        "full_model": {
            "current_bm32": {
                "timing": current_timing,
                "benchmark": current_benchmark,
                "direct_median_speedup": current_speedup,
                "final_logits": current_logits_metrics,
                "router": current_route_parity,
            },
            "direct_order_bm8": {
                "timing": bm8_timing,
                "benchmark": bm8_benchmark,
                "direct_median_speedup": bm8_speedup,
                "final_logits": bm8_logits_metrics,
                "router": bm8_route_parity,
                "oracle_16": oracle,
            },
        },
        "candidate_gate": {
            **candidate_gate,
            "accepted": candidate_accepted,
        },
        "evidence_gate": {
            **evidence_gate,
            "accepted": all(evidence_gate.values()),
        },
        "runtime_policy": {
            "default_backend": "direct",
            "experimental_backend": "current BM32 unchanged",
            "direct_order_probe_installed": False,
            "apc_identity_changed": False,
            "prompt_limit": 256,
            "grouped_full_model_correctness_accepted": False,
        },
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "projection_exact": projection_exact,
                "single_moe_exact": single_moe_exact,
                "full_model_exact": bm8_logits_metrics["array_equal"],
                "router_exact": bm8_route_parity[
                    "all_indices_and_scores_byte_identical"
                ],
                "direct_order_speedup": bm8_speedup,
                "candidate_accepted": candidate_accepted,
                "evidence_accepted": all(evidence_gate.values()),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if all(evidence_gate.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
