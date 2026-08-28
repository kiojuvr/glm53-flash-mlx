"""Sorted grouped block-FP8 MoE kernels for the layer-local prefill probe."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .abi import (
    GROUPED_KERNEL_ABI,
    GROUPED_MEASURED_CROSSOVER_ROUTES,
    GROUPED_MIN_ROUTES,
)
from .fp8 import (
    BLOCK_SIZE,
    _FP8_LUT_HEADER,
)
from .packed import PackedFP8MoE

GROUPED_TILE_ROWS = 32
GROUPED_OUTPUT_COLS = 32
GROUPED_K_TILE = 32

_GROUPED_FP8_HEADER = r"""
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;
""" + _FP8_LUT_HEADER

_GROUPED_FP8_SOURCE = r"""
    constexpr short BM = 32;
    constexpr short BK = 32;
    constexpr short BN = 32;
    constexpr short TILE_STRIDE = BK + 4;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint thread_index = thread_index_in_threadgroup;
    uint output_tiles = OUT_FEATURES / BN;
    uint group_id = threadgroup_position_in_grid.x;
    uint output_tile = group_id % output_tiles;
    uint tile = group_id / output_tiles;
    uint first_route = tile_starts[tile];
    if (first_route >= ROUTES) return;
    uint valid_rows = tile_lengths[tile];
    uint expert = tile_experts[tile];
    uint tile_column = output_tile * BN;
    uint load_row = thread_index / 4;
    uint load_column = (thread_index % 4) * 8;
    uint simd_row = simd_group / 2;
    uint simd_column = simd_group % 2;

    short quad = lane / 4;
    short fragment_row = (quad & 4) + ((lane / 2) % 4);
    short fragment_column = (quad & 2) * 2 + (lane % 2) * 2;

    threadgroup float x_tile[BM * TILE_STRIDE];
    threadgroup float weight_tile[BN * TILE_STRIDE];

    metal::simdgroup_matrix<float, 8, 8> accumulators[2][2];
    #pragma clang loop unroll(full)
    for (short row = 0; row < 2; ++row) {
        #pragma clang loop unroll(full)
        for (short column = 0; column < 2; ++column) {
            accumulators[row][column].thread_elements()[0] = 0.0f;
            accumulators[row][column].thread_elements()[1] = 0.0f;
        }
    }

    for (uint k_start = 0; k_start < IN_FEATURES; k_start += BK) {
        uint input_row = load_row;
        uint weight_row = tile_column + load_row;
        uint input_column = k_start + load_column;
        #pragma clang loop unroll(full)
        for (short element = 0; element < 8; ++element) {
            uint k = input_column + element;
            x_tile[load_row * TILE_STRIDE + load_column + element] =
                input_row < valid_rows
                ? float(x[size_t(first_route + input_row) * IN_FEATURES + k])
                : 0.0f;
            size_t weight_offset =
                (size_t(expert) * OUT_FEATURES + weight_row) * IN_FEATURES + k;
            size_t scale_offset =
                (size_t(expert) * SCALE_ROWS + weight_row / BLOCK_SIZE)
                * SCALE_COLS + k / BLOCK_SIZE;
            weight_tile[load_row * TILE_STRIDE + load_column + element] =
                glm53_fp8_lut[weight[weight_offset]] * scale_inv[scale_offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (short k_fragment = 0; k_fragment < BK; k_fragment += 8) {
            metal::simdgroup_matrix<float, 8, 8> a[2];
            metal::simdgroup_matrix<float, 8, 8> b[2];
            #pragma clang loop unroll(full)
            for (short frag = 0; frag < 2; ++frag) {
                uint a_row = simd_row * 16 + frag * 8 + fragment_row;
                uint b_row = simd_column * 16 + frag * 8 + fragment_column;
                #pragma clang loop unroll(full)
                for (short element = 0; element < 2; ++element) {
                    a[frag].thread_elements()[element] = float(
                        x_tile[a_row * TILE_STRIDE + k_fragment
                               + fragment_column + element]);
                    b[frag].thread_elements()[element] = float(
                        weight_tile[(b_row + element) * TILE_STRIDE
                                    + k_fragment + fragment_row]);
                }
            }
            #pragma clang loop unroll(full)
            for (short row = 0; row < 2; ++row) {
                #pragma clang loop unroll(full)
                for (short column = 0; column < 2; ++column) {
                    metal::simdgroup_matrix<float, 8, 8> result;
                    simdgroup_multiply_accumulate(
                        result, a[row], b[column], accumulators[row][column]);
                    accumulators[row][column] = result;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (short row = 0; row < 2; ++row) {
        #pragma clang loop unroll(full)
        for (short column = 0; column < 2; ++column) {
            uint output_row = simd_row * 16 + row * 8 + fragment_row;
            uint output_column =
                tile_column + simd_column * 16 + column * 8 + fragment_column;
            #pragma clang loop unroll(full)
            for (short element = 0; element < 2; ++element) {
                if (output_row < valid_rows) {
                    output[size_t(first_route + output_row) * OUT_FEATURES
                           + output_column + element] = T(
                        accumulators[row][column].thread_elements()[element]);
                }
            }
        }
    }
"""

_grouped_fp8_kernel = (
    mx.fast.metal_kernel(
        name="glm53_sorted_grouped_block128_e4m3",
        input_names=[
            "x", "tile_experts", "tile_starts", "tile_lengths", "weight", "scale_inv"
        ],
        output_names=["output"],
        source=_GROUPED_FP8_SOURCE,
        header=_GROUPED_FP8_HEADER,
    )
    if mx.metal.is_available()
    else None
)


def build_route_plan(x, topk_indices, topk_scores):
    """Return a fully GPU-resident expert-sorted route plan."""
    flat_x = x.reshape(-1, x.shape[-1])
    top_k = topk_indices.shape[-1]
    expert_ids = topk_indices.reshape(-1).astype(mx.uint32)
    order = mx.argsort(expert_ids)
    inverse_order = mx.argsort(order)
    token_ids = order // top_k
    return (
        flat_x[token_ids],
        expert_ids[order],
        topk_scores.reshape(-1)[order],
        inverse_order,
    )


def build_grouped_tile_plan(sorted_experts, expert_count: int):
    """Build expert-aligned fixed-capacity tile descriptors entirely on GPU."""
    routes = sorted_experts.shape[0]
    expert_range = mx.arange(expert_count, dtype=mx.uint32)
    counts = mx.sum(
        sorted_experts[:, None] == expert_range[None, :], axis=0
    ).astype(mx.uint32)
    zero = mx.zeros((1,), dtype=mx.uint32)
    route_offsets = mx.concatenate([zero, mx.cumsum(counts)])
    tile_counts = (counts + GROUPED_TILE_ROWS - 1) // GROUPED_TILE_ROWS
    tile_offsets = mx.concatenate([zero, mx.cumsum(tile_counts)])

    descriptor_slots = (routes + GROUPED_TILE_ROWS - 1) // GROUPED_TILE_ROWS + expert_count
    descriptor_ids = mx.arange(descriptor_slots, dtype=mx.uint32)
    owners = mx.sum(
        descriptor_ids[:, None] >= tile_offsets[None, 1:], axis=1
    ).astype(mx.uint32)
    valid = descriptor_ids < tile_offsets[-1]
    safe_owners = mx.minimum(owners, expert_count - 1)
    local_tiles = descriptor_ids - tile_offsets[safe_owners]
    starts = route_offsets[safe_owners] + local_tiles * GROUPED_TILE_ROWS
    remaining = route_offsets[safe_owners + 1] - starts
    lengths = mx.minimum(remaining, GROUPED_TILE_ROWS)
    invalid_start = mx.full(starts.shape, routes, dtype=mx.uint32)
    return (
        mx.where(valid, safe_owners, mx.zeros_like(safe_owners)),
        mx.where(valid, starts, invalid_start),
        mx.where(valid, lengths, mx.zeros_like(lengths)),
        route_offsets,
        tile_offsets,
    )


def grouped_fp8_linear(x, tile_plan, weight, scale_inv):
    """Project sorted route rows from a packed block-FP8 expert bank."""
    if _grouped_fp8_kernel is None:
        raise RuntimeError("sorted grouped FP8 execution requires Metal")
    if weight.dtype != mx.uint8 or weight.ndim != 3:
        raise ValueError("grouped weight must be [experts, out, in] uint8 E4M3")
    if scale_inv.dtype != mx.float32 or scale_inv.ndim != 3:
        raise ValueError("grouped scale must be [experts, out_blocks, in_blocks] FP32")
    routes, in_features = x.shape
    experts, out_features, weight_in = weight.shape
    if (
        out_features % BLOCK_SIZE != 0
        or in_features % BLOCK_SIZE != 0
        or out_features % GROUPED_OUTPUT_COLS != 0
        or in_features % GROUPED_K_TILE != 0
    ):
        raise ValueError(
            "grouped probe requires block-aligned projection dimensions; "
            f"got out={out_features}, in={in_features}"
        )
    expected_scales = (
        experts,
        (out_features + BLOCK_SIZE - 1) // BLOCK_SIZE,
        (in_features + BLOCK_SIZE - 1) // BLOCK_SIZE,
    )
    if weight_in != in_features or scale_inv.shape != expected_scales:
        raise ValueError(
            f"incompatible grouped shapes x={x.shape}, weight={weight.shape}, "
            f"scale={scale_inv.shape}, expected_scale={expected_scales}"
        )
    tile_experts, tile_starts, tile_lengths = tile_plan[:3]
    descriptor_slots = tile_experts.shape[0]
    output_tiles = out_features // GROUPED_OUTPUT_COLS
    return _grouped_fp8_kernel(
        inputs=[
            x, tile_experts, tile_starts, tile_lengths, weight, scale_inv
        ],
        template=[
            ("T", x.dtype),
            ("IN_FEATURES", in_features),
            ("OUT_FEATURES", out_features),
            ("ROUTES", routes),
            ("SCALE_ROWS", expected_scales[1]),
            ("SCALE_COLS", expected_scales[2]),
            ("BLOCK_SIZE", BLOCK_SIZE),
        ],
        grid=(descriptor_slots * output_tiles * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(routes, out_features)],
        output_dtypes=[x.dtype],
    )[0]


def activate_gate_up(gate_up, intermediate_size: int, limit: float):
    if intermediate_size % BLOCK_SIZE != 0:
        raise ValueError("gate/up split must align to the block-128 scale boundary")
    gate = mx.minimum(gate_up[:, :intermediate_size], limit)
    up = mx.clip(gate_up[:, intermediate_size:], -limit, limit)
    return nn.silu(gate) * up


def restore_and_reduce(sorted_output, sorted_scores, inverse_order, shape, top_k):
    weighted = sorted_output * sorted_scores[:, None]
    restored = weighted[inverse_order].reshape(-1, top_k, shape[-1])
    return restored.sum(axis=1).astype(sorted_output.dtype).reshape(shape)


class SortedGroupedFP8MoE(PackedFP8MoE):
    """Prefill-only grouped path with the existing packed/decode fallback."""

    def __init__(
        self,
        bank,
        config,
        gate,
        shared_experts,
        *,
        min_routes: int = GROUPED_MIN_ROUTES,
    ):
        super().__init__(bank, config, gate, shared_experts)
        self.min_routes = int(min_routes)

    def grouped_from_routes(self, x, indices, scores):
        sorted_x, experts, sorted_scores, inverse = build_route_plan(x, indices, scores)
        tile_plan = build_grouped_tile_plan(experts, self.bank.expert_count)
        gate_up = grouped_fp8_linear(
            sorted_x,
            tile_plan,
            self.bank.gate_up_weight,
            self.bank.gate_up_scale_inv,
        )
        hidden = activate_gate_up(
            gate_up, self.bank.intermediate_size, self.config.swiglu_limit
        )
        down = grouped_fp8_linear(
            hidden,
            tile_plan,
            self.bank.down_weight,
            self.bank.down_scale_inv,
        )
        result = restore_and_reduce(
            down,
            sorted_scores,
            inverse,
            x.shape,
            self.config.num_experts_per_tok,
        )
        if self.shared_experts is not None:
            result = result + self.shared_experts(x)
        return result

    def __call__(self, x):
        routes = x.reshape(-1, x.shape[-1]).shape[0] * self.config.num_experts_per_tok
        if routes < self.min_routes:
            return super().__call__(x)
        indices, scores = self.gate(x)
        return self.grouped_from_routes(x, indices, scores)
