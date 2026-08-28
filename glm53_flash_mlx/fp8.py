"""Direct E4M3 block-FP8 operators for the official GLM-5.3 checkpoint."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .abi import KERNEL_ABI_VERSION

BLOCK_SIZE = 128
THREADS = 256
DECODE_TOP_K = 8
PREFILL_TILE_ROWS = 8


def _e4m3_value(byte: int) -> float:
    sign = -1.0 if byte & 0x80 else 1.0
    exponent = (byte >> 3) & 15
    mantissa = byte & 7
    value = mantissa * 2.0**-9 if exponent == 0 else (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
    return sign * value


def _metal_float(value: float) -> str:
    literal = f"{value:.9g}"
    if "." not in literal and "e" not in literal:
        literal += ".0"
    return literal + "f"


_FP8_LUT_HEADER = "constant float glm53_fp8_lut[256] = {" + ",".join(
    _metal_float(_e4m3_value(code)) for code in range(256)
) + "};"

_FP8_GEMV_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd_id = simdgroup_index_in_threadgroup;
    uint group_id = threadgroup_position_in_grid.x;

    uint out_row = group_id % OUT_FEATURES;
    uint batch_row = group_id / OUT_FEATURES;
    if (batch_row >= BATCH_ROWS) return;

    float acc = 0.0f;
    const device T* xr = x + size_t(batch_row) * IN_FEATURES;
    const device uint8_t* wr = weight + size_t(out_row) * IN_FEATURES;
    uint scale_row = out_row / BLOCK_SIZE;
    for (uint k = tid; k < IN_FEATURES; k += THREADS) {
        float scale = scale_inv[scale_row * SCALE_COLS + k / BLOCK_SIZE];
        acc += float(xr[k]) * glm53_fp8_lut[wr[k]] * scale;
    }
    acc = simd_sum(acc);

    constexpr uint NSIMD = THREADS / 32;
    threadgroup float partial[NSIMD];
    if (lane == 0) partial[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        float total = lane < NSIMD ? partial[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0) {
            output[size_t(batch_row) * OUT_FEATURES + out_row] = T(total);
        }
    }
"""

_FP8_GEMM_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd_id = simdgroup_index_in_threadgroup;
    uint group_id = threadgroup_position_in_grid.x;
    uint out_row = group_id % OUT_FEATURES;
    uint tile = group_id / OUT_FEATURES;
    uint first_row = tile * TILE_ROWS;
    if (first_row >= BATCH_ROWS) return;

    thread float acc[TILE_ROWS];
    for (uint row = 0; row < TILE_ROWS; ++row) acc[row] = 0.0f;
    const device uint8_t* wr = weight + size_t(out_row) * IN_FEATURES;
    uint scale_row = out_row / BLOCK_SIZE;
    for (uint k = tid; k < IN_FEATURES; k += THREADS) {
        float decoded = glm53_fp8_lut[wr[k]]
            * scale_inv[scale_row * SCALE_COLS + k / BLOCK_SIZE];
        for (uint row = 0; row < TILE_ROWS; ++row) {
            uint batch_row = first_row + row;
            if (batch_row < BATCH_ROWS) {
                acc[row] += float(x[size_t(batch_row) * IN_FEATURES + k]) * decoded;
            }
        }
    }
    for (uint row = 0; row < TILE_ROWS; ++row) acc[row] = simd_sum(acc[row]);

    constexpr uint NSIMD = THREADS / 32;
    threadgroup float partial[TILE_ROWS][NSIMD];
    if (lane == 0) {
        for (uint row = 0; row < TILE_ROWS; ++row) partial[row][simd_id] = acc[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        for (uint row = 0; row < TILE_ROWS; ++row) {
            float total = lane < NSIMD ? partial[row][lane] : 0.0f;
            total = simd_sum(total);
            uint batch_row = first_row + row;
            if (lane == 0 && batch_row < BATCH_ROWS) {
                output[size_t(batch_row) * OUT_FEATURES + out_row] = T(total);
            }
        }
    }
"""

_fp8_gemv_kernel = (
    mx.fast.metal_kernel(
        name="glm53_block128_e4m3_gemv",
        input_names=["x", "weight", "scale_inv"],
        output_names=["output"],
        source=_FP8_GEMV_SOURCE,
        header=_FP8_LUT_HEADER,
    )
    if mx.metal.is_available()
    else None
)
_fp8_gemm_kernel = (
    mx.fast.metal_kernel(
        name="glm53_block128_e4m3_tiled8_gemm",
        input_names=["x", "weight", "scale_inv"],
        output_names=["output"],
        source=_FP8_GEMM_SOURCE,
        header=_FP8_LUT_HEADER,
    )
    if mx.metal.is_available()
    else None
)


def _selected_weight_source(*, down: bool) -> str:
    weight_bindings = "\n".join(
        f"""
        case {i}:
            wr = weight_{i} + size_t(out_row) * IN_FEATURES;
            break;"""
        for i in range(DECODE_TOP_K)
    )
    scale_bindings = "\n".join(
        f"case {i}: scale = scale_{i}[scale_offset]; break;"
        for i in range(DECODE_TOP_K)
    )
    if down:
        prologue = r"""
    uint group_id = threadgroup_position_in_grid.x;
    uint expert = group_id / OUT_FEATURES;
    uint out_row = group_id % OUT_FEATURES;
    if (expert >= TOP_K) return;
    """
        expert_loop = f"""
    const device uint8_t* wr = weight_0;
    switch (expert) {{{weight_bindings}
    }}
    const device T* xr = hidden + size_t(expert) * IN_FEATURES;
    float acc = 0.0f;
    for (uint k = tid; k < IN_FEATURES; k += THREADS) {{
        uint scale_offset = size_t(out_row / BLOCK_SIZE) * SCALE_COLS
            + k / BLOCK_SIZE;
        float scale = 0.0f;
        switch (expert) {{{scale_bindings}
        }}
        acc += float(xr[k]) * glm53_fp8_lut[wr[k]] * scale;
    }}
    acc = simd_sum(acc);
    if (lane == 0) partial[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {{
        float reduced = lane < NSIMD ? partial[lane] : 0.0f;
        reduced = simd_sum(reduced);
        if (lane == 0) {{
            output[size_t(expert) * OUT_FEATURES + out_row] = T(reduced);
        }}
    }}
    """
    else:
        prologue = r"""
    uint group_id = threadgroup_position_in_grid.x;
    uint expert = group_id / OUT_FEATURES;
    uint out_row = group_id % OUT_FEATURES;
    if (expert >= TOP_K) return;
    """
        expert_loop = f"""
    const device uint8_t* wr = weight_0;
    switch (expert) {{{weight_bindings}
    }}
    float acc = 0.0f;
    for (uint k = tid; k < IN_FEATURES; k += THREADS) {{
        uint scale_offset = size_t(out_row / BLOCK_SIZE) * SCALE_COLS
            + k / BLOCK_SIZE;
        float scale = 0.0f;
        switch (expert) {{{scale_bindings}
        }}
        acc += float(x[k]) * glm53_fp8_lut[wr[k]] * scale;
    }}
    acc = simd_sum(acc);
    if (lane == 0) partial[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {{
        float reduced = lane < NSIMD ? partial[lane] : 0.0f;
        reduced = simd_sum(reduced);
        if (lane == 0) output[size_t(expert) * OUT_FEATURES + out_row] = T(reduced);
    }}
    """
    return r"""
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd_id = simdgroup_index_in_threadgroup;
    constexpr uint NSIMD = THREADS / 32;
    threadgroup float partial[NSIMD];
    """ + prologue + expert_loop


_selected_input_names = ["x"] + [
    name for i in range(DECODE_TOP_K) for name in (f"weight_{i}", f"scale_{i}")
]
_down_input_names = ["hidden"] + [
    name for i in range(DECODE_TOP_K) for name in (f"weight_{i}", f"scale_{i}")
]
_selected_projection_kernel = (
    mx.fast.metal_kernel(
        name="glm53_selected8_fp8_projection",
        input_names=_selected_input_names,
        output_names=["output"],
        source=_selected_weight_source(down=False),
        header=_FP8_LUT_HEADER,
    )
    if mx.metal.is_available()
    else None
)
_selected_down_kernel = (
    mx.fast.metal_kernel(
        name="glm53_selected8_fp8_down",
        input_names=_down_input_names,
        output_names=["output"],
        source=_selected_weight_source(down=True),
        header=_FP8_LUT_HEADER,
    )
    if mx.metal.is_available()
    else None
)


def block_fp8_linear(x: mx.array, weight: mx.array, scale_inv: mx.array) -> mx.array:
    """Compute ``x @ dequant(weight, scale_inv).T`` without expanding weight."""
    if _fp8_gemv_kernel is None or mx.default_device() != mx.gpu:
        raise RuntimeError("direct GLM-5.3 FP8 execution requires the Metal GPU backend")
    if weight.dtype != mx.uint8 or weight.ndim != 2:
        raise ValueError(f"expected 2-D uint8 E4M3 weight, got {weight.shape} {weight.dtype}")
    if scale_inv.dtype != mx.float32 or scale_inv.ndim != 2:
        raise ValueError("weight_scale_inv must be a 2-D float32 block table")
    in_features = weight.shape[1]
    out_features = weight.shape[0]
    if x.shape[-1] != in_features:
        raise ValueError(f"input width {x.shape[-1]} != weight width {in_features}")
    expected_scales = (
        (out_features + BLOCK_SIZE - 1) // BLOCK_SIZE,
        (in_features + BLOCK_SIZE - 1) // BLOCK_SIZE,
    )
    if scale_inv.shape != expected_scales:
        raise ValueError(
            f"scale shape {scale_inv.shape} != block128 shape {expected_scales}"
        )
    original_shape = x.shape
    flat = x.reshape(-1, in_features)
    batch_rows = flat.shape[0]
    kernel = _fp8_gemv_kernel if batch_rows == 1 else _fp8_gemm_kernel
    tile_rows = 1 if batch_rows == 1 else PREFILL_TILE_ROWS
    groups = (batch_rows + tile_rows - 1) // tile_rows
    output = kernel(
        inputs=[flat, weight, scale_inv],
        template=[
            ("T", flat.dtype),
            ("IN_FEATURES", in_features),
            ("OUT_FEATURES", out_features),
            ("BATCH_ROWS", batch_rows),
            ("SCALE_COLS", expected_scales[1]),
            ("BLOCK_SIZE", BLOCK_SIZE),
            ("THREADS", THREADS),
            ("TILE_ROWS", tile_rows),
        ],
        grid=(groups * out_features * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(batch_rows, out_features)],
        output_dtypes=[flat.dtype],
    )[0]
    return output.reshape(*original_shape[:-1], out_features)


class BlockFP8Linear(nn.Module):
    """``nn.Linear`` ABI backed by canonical block-128 E4M3 tensors."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = mx.zeros((out_features, in_features), dtype=mx.uint8)
        self.weight_scale_inv = mx.ones(
            (
                (out_features + BLOCK_SIZE - 1) // BLOCK_SIZE,
                (in_features + BLOCK_SIZE - 1) // BLOCK_SIZE,
            ),
            dtype=mx.float32,
        )
        self.bias = mx.zeros((out_features,)) if bias else None

    @classmethod
    def from_linear(cls, linear: nn.Module) -> "BlockFP8Linear":
        out_features, in_features = linear.weight.shape
        return cls(in_features, out_features, bias=getattr(linear, "bias", None) is not None)

    def __call__(self, x):
        y = block_fp8_linear(x, self.weight, self.weight_scale_inv)
        return y + self.bias if self.bias is not None else y


class DirectFP8Expert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, limit: float):
        super().__init__()
        self.gate_proj = BlockFP8Linear(hidden_size, intermediate_size)
        self.up_proj = BlockFP8Linear(hidden_size, intermediate_size)
        self.down_proj = BlockFP8Linear(intermediate_size, hidden_size)
        self.limit = float(limit)

    def __call__(self, x):
        gate = mx.minimum(self.gate_proj(x), self.limit)
        up = mx.clip(self.up_proj(x), -self.limit, self.limit)
        return self.down_proj(nn.silu(gate) * up)


class DirectFP8MoE(nn.Module):
    """Bucketed top-k MoE over separate canonical expert tensors.

    The official checkpoint stores experts separately. Keeping that layout
    avoids the 283 GiB duplicate created by stacking all expert banks.
    """

    def __init__(self, config, gate, shared_experts):
        super().__init__()
        self.config = config
        self.gate = gate
        self.shared_experts = shared_experts
        self.experts = [
            DirectFP8Expert(
                config.hidden_size,
                config.moe_intermediate_size,
                config.swiglu_limit,
            )
            for _ in range(config.n_routed_experts)
        ]

    def __call__(self, x):
        indices, scores = self.gate(x)
        # Routing is a phase boundary. A single readback gives the host the
        # stable expert buckets while the 306 GiB canonical bank stays mapped.
        mx.eval(indices)
        routes = np.asarray(indices).reshape(-1, self.config.num_experts_per_tok)
        flat_x = x.reshape(-1, x.shape[-1])
        flat_scores = scores.reshape(-1, self.config.num_experts_per_tok)
        if flat_x.shape[0] == 1 and routes.shape == (1, DECODE_TOP_K):
            selected = [self.experts[int(expert_id)] for expert_id in routes[0]]
            gate = _selected_projection(flat_x[0], selected, "gate_proj")
            up = _selected_projection(flat_x[0], selected, "up_proj")
            hidden = nn.silu(mx.minimum(gate, self.config.swiglu_limit)) * mx.clip(
                up, -self.config.swiglu_limit, self.config.swiglu_limit
            )
            result = _selected_down(hidden, flat_scores[0], selected)[None, :]
            result = result.reshape(x.shape)
            if self.shared_experts is not None:
                result = result + self.shared_experts(x)
            return result
        result = mx.zeros_like(flat_x)
        for expert_id in np.unique(routes):
            positions = np.argwhere(routes == expert_id)
            rows_np = positions[:, 0]
            slots_np = positions[:, 1]
            rows = mx.array(rows_np, dtype=mx.int32)
            slots = mx.array(slots_np, dtype=mx.int32)
            expert_out = self.experts[int(expert_id)](flat_x[rows])
            route_weight = flat_scores[rows, slots][..., None]
            result = result.at[rows].add(expert_out * route_weight)
        result = result.reshape(x.shape)
        if self.shared_experts is not None:
            result = result + self.shared_experts(x)
        return result


def _selected_projection(x, experts, projection: str):
    if _selected_projection_kernel is None:
        raise RuntimeError("selected expert path requires Metal")
    modules = [getattr(expert, projection) for expert in experts]
    out_features, in_features = modules[0].weight.shape
    inputs = [x]
    for module in modules:
        inputs.extend((module.weight, module.weight_scale_inv))
    return _selected_projection_kernel(
        inputs=inputs,
        template=[
            ("T", x.dtype),
            ("IN_FEATURES", in_features),
            ("OUT_FEATURES", out_features),
            ("TOP_K", DECODE_TOP_K),
            ("SCALE_COLS", (in_features + BLOCK_SIZE - 1) // BLOCK_SIZE),
            ("BLOCK_SIZE", BLOCK_SIZE),
            ("THREADS", THREADS),
        ],
        grid=(DECODE_TOP_K * out_features * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(DECODE_TOP_K, out_features)],
        output_dtypes=[x.dtype],
    )[0]


def _selected_down(hidden, scores, experts):
    if _selected_down_kernel is None:
        raise RuntimeError("selected expert path requires Metal")
    modules = [expert.down_proj for expert in experts]
    out_features, in_features = modules[0].weight.shape
    inputs = [hidden]
    for module in modules:
        inputs.extend((module.weight, module.weight_scale_inv))
    output = _selected_down_kernel(
        inputs=inputs,
        template=[
            ("T", hidden.dtype),
            ("IN_FEATURES", in_features),
            ("OUT_FEATURES", out_features),
            ("TOP_K", DECODE_TOP_K),
            ("SCALE_COLS", (in_features + BLOCK_SIZE - 1) // BLOCK_SIZE),
            ("BLOCK_SIZE", BLOCK_SIZE),
            ("THREADS", THREADS),
        ],
        grid=(DECODE_TOP_K * out_features * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(DECODE_TOP_K, out_features)],
        output_dtypes=[hidden.dtype],
    )[0]
    return mx.sum(output.astype(mx.float32) * scores[:, None], axis=0).astype(hidden.dtype)
