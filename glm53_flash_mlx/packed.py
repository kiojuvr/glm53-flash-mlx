"""Storage-only packed FP8 expert bank used by the feasibility probe."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .abi import PACKED_EXPERT_BANK_ABI
from .fp8 import (
    BLOCK_SIZE,
    DECODE_TOP_K,
    PREFILL_TILE_ROWS,
    THREADS,
    DirectFP8MoE,
    _FP8_LUT_HEADER,
    _metal_input,
)
from .ownership import (
    ResidentTensor,
    TensorLease,
    TensorLayout,
    borrowed_stable_tensor,
    owned_tensor,
    require_resident,
    resident_concatenate,
    storage_descriptor,
)

_PACKED_SELECTED_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd_id = simdgroup_index_in_threadgroup;
    uint group_id = threadgroup_position_in_grid.x;
    uint selected = group_id / OUT_FEATURES;
    uint out_row = group_id % OUT_FEATURES;
    if (selected >= TOP_K) return;

    uint expert = expert_ids[selected];
    uint bank_row = WEIGHT_ROW_OFFSET + out_row;
    const device uint8_t* wr = weight
        + (size_t(expert) * BANK_OUT_FEATURES + bank_row) * IN_FEATURES;
    float acc = 0.0f;
    for (uint k = tid; k < IN_FEATURES; k += THREADS) {
        size_t scale_offset =
            (size_t(expert) * BANK_SCALE_ROWS + SCALE_ROW_OFFSET
             + out_row / BLOCK_SIZE)
            * SCALE_COLS + k / BLOCK_SIZE;
        acc += float(x[k]) * glm53_fp8_lut[wr[k]] * scale_inv[scale_offset];
    }
    acc = simd_sum(acc);
    constexpr uint NSIMD = THREADS / 32;
    threadgroup float partial[NSIMD];
    if (lane == 0) partial[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        float reduced = lane < NSIMD ? partial[lane] : 0.0f;
        reduced = simd_sum(reduced);
        if (lane == 0) {
            output[size_t(selected) * OUT_FEATURES + out_row] = T(reduced);
        }
    }
"""

_PACKED_SELECTED_DOWN_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd_id = simdgroup_index_in_threadgroup;
    uint group_id = threadgroup_position_in_grid.x;
    uint selected = group_id / OUT_FEATURES;
    uint out_row = group_id % OUT_FEATURES;
    if (selected >= TOP_K) return;

    uint expert = expert_ids[selected];
    const device uint8_t* wr = weight
        + (size_t(expert) * OUT_FEATURES + out_row) * IN_FEATURES;
    const device T* xr = hidden + size_t(selected) * IN_FEATURES;
    float acc = 0.0f;
    for (uint k = tid; k < IN_FEATURES; k += THREADS) {
        size_t scale_offset =
            (size_t(expert) * SCALE_ROWS + out_row / BLOCK_SIZE)
            * SCALE_COLS + k / BLOCK_SIZE;
        acc += float(xr[k]) * glm53_fp8_lut[wr[k]] * scale_inv[scale_offset];
    }
    acc = simd_sum(acc);
    constexpr uint NSIMD = THREADS / 32;
    threadgroup float partial[NSIMD];
    if (lane == 0) partial[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        float reduced = lane < NSIMD ? partial[lane] : 0.0f;
        reduced = simd_sum(reduced);
        if (lane == 0) {
            output[size_t(selected) * OUT_FEATURES + out_row] = T(reduced);
        }
    }
"""

_packed_selected_kernel = (
    mx.fast.metal_kernel(
        name="glm53_packed_selected8_fp8_projection",
        input_names=["x", "expert_ids", "weight", "scale_inv"],
        output_names=["output"],
        source=_PACKED_SELECTED_SOURCE,
        header=_FP8_LUT_HEADER,
    )
    if mx.metal.is_available()
    else None
)

_packed_selected_down_kernel = (
    mx.fast.metal_kernel(
        name="glm53_packed_selected8_fp8_down",
        input_names=["hidden", "expert_ids", "weight", "scale_inv"],
        output_names=["output"],
        source=_PACKED_SELECTED_DOWN_SOURCE,
        header=_FP8_LUT_HEADER,
    )
    if mx.metal.is_available()
    else None
)


_PACKED_BANK_GEMV_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd_id = simdgroup_index_in_threadgroup;
    uint group_id = threadgroup_position_in_grid.x;
    uint out_row = group_id % OUT_FEATURES;
    uint batch_row = group_id / OUT_FEATURES;
    if (batch_row >= BATCH_ROWS) return;

    const device T* xr = x + size_t(batch_row) * IN_FEATURES;
    uint expert = expert_id[0];
    const device uint8_t* wr = weight
        + (size_t(expert) * BANK_OUT_FEATURES + ROW_OFFSET + out_row)
        * IN_FEATURES;
    uint scale_row = SCALE_ROW_OFFSET + out_row / BLOCK_SIZE;
    float acc = 0.0f;
    for (uint k = tid; k < IN_FEATURES; k += THREADS) {
        size_t scale_offset =
            (size_t(expert) * BANK_SCALE_ROWS + scale_row) * SCALE_COLS
            + k / BLOCK_SIZE;
        acc += float(xr[k]) * glm53_fp8_lut[wr[k]] * scale_inv[scale_offset];
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


_PACKED_BANK_GEMM_SOURCE = r"""
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
    uint expert = expert_id[0];
    const device uint8_t* wr = weight
        + (size_t(expert) * BANK_OUT_FEATURES + ROW_OFFSET + out_row)
        * IN_FEATURES;
    uint scale_row = SCALE_ROW_OFFSET + out_row / BLOCK_SIZE;
    for (uint k = tid; k < IN_FEATURES; k += THREADS) {
        size_t scale_offset =
            (size_t(expert) * BANK_SCALE_ROWS + scale_row) * SCALE_COLS
            + k / BLOCK_SIZE;
        float decoded = glm53_fp8_lut[wr[k]] * scale_inv[scale_offset];
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


_packed_bank_gemv_kernel = (
    mx.fast.metal_kernel(
        name="glm53_packed_bank_direct_gemv",
        input_names=[
            "x",
            "expert_id",
            "weight",
            "scale_inv",
        ],
        output_names=["output"],
        source=_PACKED_BANK_GEMV_SOURCE,
        header=_FP8_LUT_HEADER,
    )
    if mx.metal.is_available()
    else None
)
_packed_bank_gemm_kernel = (
    mx.fast.metal_kernel(
        name="glm53_packed_bank_direct_tiled8_gemm",
        input_names=[
            "x",
            "expert_id",
            "weight",
            "scale_inv",
        ],
        output_names=["output"],
        source=_PACKED_BANK_GEMM_SOURCE,
        header=_FP8_LUT_HEADER,
    )
    if mx.metal.is_available()
    else None
)


def _packed_bank_linear(
    x,
    weight,
    scale_inv,
    *,
    expert_id,
    row_offset: int,
    scale_row_offset: int,
    out_features: int,
):
    """Direct-order FP8 linear over one expert without materializing a slice."""
    if _packed_bank_gemv_kernel is None or _packed_bank_gemm_kernel is None:
        raise RuntimeError("packed bank Direct path requires Metal")
    in_features = int(weight.shape[-1])
    original_shape = x.shape
    flat = _metal_input(x.reshape(-1, in_features))
    batch_rows = int(flat.shape[0])
    tile_rows = 1 if batch_rows == 1 else PREFILL_TILE_ROWS
    kernel = _packed_bank_gemv_kernel if batch_rows == 1 else _packed_bank_gemm_kernel
    groups = (batch_rows + tile_rows - 1) // tile_rows
    output = kernel(
        inputs=[
            flat,
            expert_id,
            _metal_input(weight),
            _metal_input(scale_inv),
        ],
        template=[
            ("T", flat.dtype),
            ("IN_FEATURES", in_features),
            ("OUT_FEATURES", int(out_features)),
            ("BATCH_ROWS", batch_rows),
            ("BANK_OUT_FEATURES", int(weight.shape[1])),
            ("BANK_SCALE_ROWS", int(scale_inv.shape[1])),
            ("ROW_OFFSET", int(row_offset)),
            ("SCALE_ROW_OFFSET", int(scale_row_offset)),
            ("SCALE_COLS", int(scale_inv.shape[2])),
            ("BLOCK_SIZE", BLOCK_SIZE),
            ("THREADS", THREADS),
            ("TILE_ROWS", tile_rows),
        ],
        grid=(groups * int(out_features) * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(batch_rows, int(out_features))],
        output_dtypes=[flat.dtype],
    )[0]
    return output.reshape(*original_shape[:-1], int(out_features))


def _packed_selected_projection(x, expert_ids, bank, *, row_offset: int):
    if _packed_selected_kernel is None:
        raise RuntimeError("packed selected expert path requires Metal")
    out_features = bank.intermediate_size
    in_features = x.shape[-1]
    return _packed_selected_kernel(
        inputs=[
            _metal_input(x),
            _metal_input(expert_ids),
            _metal_input(bank.gate_up_weight),
            _metal_input(bank.gate_up_scale_inv),
        ],
        template=[
            ("T", x.dtype),
            ("IN_FEATURES", in_features),
            ("OUT_FEATURES", out_features),
            ("BANK_OUT_FEATURES", bank.gate_up_weight.shape[1]),
            ("BANK_SCALE_ROWS", bank.gate_up_scale_inv.shape[1]),
            ("WEIGHT_ROW_OFFSET", row_offset),
            (
                "SCALE_ROW_OFFSET",
                0 if row_offset == 0 else bank.intermediate_scale_rows,
            ),
            ("TOP_K", DECODE_TOP_K),
            ("SCALE_COLS", bank.gate_up_scale_inv.shape[2]),
            ("BLOCK_SIZE", BLOCK_SIZE),
            ("THREADS", THREADS),
        ],
        grid=(DECODE_TOP_K * out_features * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(DECODE_TOP_K, out_features)],
        output_dtypes=[x.dtype],
    )[0]


def _packed_selected_down(hidden, scores, expert_ids, bank):
    if _packed_selected_down_kernel is None:
        raise RuntimeError("packed selected expert path requires Metal")
    out_features = bank.down_weight.shape[1]
    in_features = bank.down_weight.shape[2]
    output = _packed_selected_down_kernel(
        inputs=[
            _metal_input(hidden),
            _metal_input(expert_ids),
            _metal_input(bank.down_weight),
            _metal_input(bank.down_scale_inv),
        ],
        template=[
            ("T", hidden.dtype),
            ("IN_FEATURES", in_features),
            ("OUT_FEATURES", out_features),
            ("TOP_K", DECODE_TOP_K),
            ("SCALE_ROWS", bank.down_scale_inv.shape[1]),
            ("SCALE_COLS", bank.down_scale_inv.shape[2]),
            ("BLOCK_SIZE", BLOCK_SIZE),
            ("THREADS", THREADS),
        ],
        grid=(DECODE_TOP_K * out_features * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(DECODE_TOP_K, out_features)],
        output_dtypes=[hidden.dtype],
    )[0]
    return mx.sum(output.astype(mx.float32) * scores[:, None], axis=0).astype(
        hidden.dtype
    )


class _PackedLinearView:
    def __init__(
        self,
        weight: mx.array,
        scale_inv: mx.array,
        *,
        expert_id: int,
        expert_id_input: mx.array,
        row_offset: int,
        scale_row_offset: int,
        out_features: int,
    ):
        self._bank_weight = weight
        self._bank_scale_inv = scale_inv
        self.expert_id = int(expert_id)
        self.row_offset = int(row_offset)
        self.scale_row_offset = int(scale_row_offset)
        self.out_features = int(out_features)
        self._expert_id_input = expert_id_input

    @property
    def weight(self):
        start = self.row_offset
        end = start + self.out_features
        return self._bank_weight[self.expert_id, start:end]

    @property
    def weight_scale_inv(self):
        rows = math.ceil(self.out_features / BLOCK_SIZE)
        start = self.scale_row_offset
        return self._bank_scale_inv[self.expert_id, start : start + rows]

    def __call__(self, x):
        return _packed_bank_linear(
            x,
            self._bank_weight,
            self._bank_scale_inv,
            expert_id=self._expert_id_input,
            row_offset=self.row_offset,
            scale_row_offset=self.scale_row_offset,
            out_features=self.out_features,
        )


class _PackedExpertView:
    def __init__(
        self,
        bank: "PackedFP8ExpertBank",
        expert_id: int,
        expert_id_input: mx.array,
        limit: float,
    ):
        intermediate = bank.intermediate_size
        scale_rows = bank.intermediate_scale_rows
        self.gate_proj = _PackedLinearView(
            bank.gate_up_weight,
            bank.gate_up_scale_inv,
            expert_id=expert_id,
            expert_id_input=expert_id_input,
            row_offset=0,
            scale_row_offset=0,
            out_features=intermediate,
        )
        self.up_proj = _PackedLinearView(
            bank.gate_up_weight,
            bank.gate_up_scale_inv,
            expert_id=expert_id,
            expert_id_input=expert_id_input,
            row_offset=intermediate,
            scale_row_offset=scale_rows,
            out_features=intermediate,
        )
        self.down_proj = _PackedLinearView(
            bank.down_weight,
            bank.down_scale_inv,
            expert_id=expert_id,
            expert_id_input=expert_id_input,
            row_offset=0,
            scale_row_offset=0,
            out_features=bank.down_weight.shape[1],
        )
        self.limit = float(limit)

    def __call__(self, x):
        gate = mx.minimum(self.gate_proj(x), self.limit)
        up = mx.clip(self.up_proj(x), -self.limit, self.limit)
        return self.down_proj(nn.silu(gate) * up)


class PackedFP8ExpertBank(nn.Module):
    """Four contiguous canonical-FP8 buffers for one routed MoE layer."""

    def __init__(
        self,
        gate_up_weight: TensorLease | ResidentTensor,
        gate_up_scale_inv: TensorLease | ResidentTensor,
        down_weight: TensorLease | ResidentTensor,
        down_scale_inv: TensorLease | ResidentTensor,
        *,
        intermediate_size: int,
    ):
        super().__init__()
        resident = {
            "gate_up_weight": require_resident(gate_up_weight),
            "gate_up_scale_inv": require_resident(gate_up_scale_inv),
            "down_weight": require_resident(down_weight),
            "down_scale_inv": require_resident(down_scale_inv),
        }
        self.gate_up_weight = resident["gate_up_weight"].value
        self.gate_up_scale_inv = resident["gate_up_scale_inv"].value
        self.down_weight = resident["down_weight"].value
        self.down_scale_inv = resident["down_scale_inv"].value
        self.storage_contracts = {
            name: storage_descriptor(value) for name, value in resident.items()
        }
        self._storage_owners = tuple(
            value.owner for value in resident.values() if value.owner is not None
        )
        self.intermediate_size = int(intermediate_size)
        self.intermediate_scale_rows = math.ceil(intermediate_size / BLOCK_SIZE)
        self._expert_ids = mx.arange(
            self.gate_up_weight.shape[0], dtype=mx.uint32
        )

    @classmethod
    def pack(cls, experts) -> "PackedFP8ExpertBank":
        experts = list(experts)
        if not experts:
            raise ValueError("cannot pack an empty expert list")
        hidden_size = experts[0].gate_proj.weight.shape[1]
        intermediate_size = experts[0].gate_proj.weight.shape[0]
        expert_count = len(experts)
        scale_rows = math.ceil(intermediate_size / BLOCK_SIZE)
        hidden_scale_rows = math.ceil(hidden_size / BLOCK_SIZE)

        for expert in experts:
            expected = (
                (expert.gate_proj.weight.shape, (intermediate_size, hidden_size)),
                (expert.up_proj.weight.shape, (intermediate_size, hidden_size)),
                (expert.down_proj.weight.shape, (hidden_size, intermediate_size)),
            )
            if any(actual != wanted for actual, wanted in expected):
                raise ValueError("expert projection shapes are not uniform")
            for projection in (expert.gate_proj, expert.up_proj, expert.down_proj):
                if projection.weight.dtype != mx.uint8:
                    raise ValueError("packed expert weights must remain uint8 E4M3")
                if projection.weight_scale_inv.dtype != mx.float32:
                    raise ValueError("packed expert scales must remain float32")

        def stable(projection, name: str):
            return borrowed_stable_tensor(
                getattr(projection, name),
                owner=projection,
                layout=TensorLayout.ROW_MAJOR_CONTIGUOUS,
            )

        gate_up_weight = resident_concatenate(
            [
                stable(projection, "weight")
                for expert in experts
                for projection in (expert.gate_proj, expert.up_proj)
            ],
            axis=0,
        ).value.reshape(expert_count, 2 * intermediate_size, hidden_size)
        gate_up_scale_inv = resident_concatenate(
            [
                stable(projection, "weight_scale_inv")
                for expert in experts
                for projection in (expert.gate_proj, expert.up_proj)
            ],
            axis=0,
        ).value.reshape(expert_count, 2 * scale_rows, hidden_scale_rows)
        down_weight = resident_concatenate(
            [stable(expert.down_proj, "weight") for expert in experts],
            axis=0,
        ).value.reshape(expert_count, hidden_size, intermediate_size)
        down_scale_inv = resident_concatenate(
            [stable(expert.down_proj, "weight_scale_inv") for expert in experts],
            axis=0,
        ).value.reshape(expert_count, hidden_scale_rows, scale_rows)
        return cls(
            owned_tensor(
                gate_up_weight, layout=TensorLayout.ROW_MAJOR_CONTIGUOUS
            ),
            owned_tensor(
                gate_up_scale_inv, layout=TensorLayout.ROW_MAJOR_CONTIGUOUS
            ),
            owned_tensor(down_weight, layout=TensorLayout.ROW_MAJOR_CONTIGUOUS),
            owned_tensor(
                down_scale_inv, layout=TensorLayout.ROW_MAJOR_CONTIGUOUS
            ),
            intermediate_size=intermediate_size,
        )

    @property
    def expert_count(self) -> int:
        return self.gate_up_weight.shape[0]

    @property
    def nbytes(self) -> int:
        return sum(
            value.nbytes
            for value in (
                self.gate_up_weight,
                self.gate_up_scale_inv,
                self.down_weight,
                self.down_scale_inv,
            )
        )

    def expert(self, expert_id: int, *, limit: float) -> _PackedExpertView:
        if not 0 <= expert_id < self.expert_count:
            raise IndexError(expert_id)
        return _PackedExpertView(
            self,
            expert_id,
            self._expert_ids[expert_id : expert_id + 1],
            limit,
        )


class PackedFP8MoE(DirectFP8MoE):
    """Existing MoE execution semantics backed by a contiguous expert bank."""

    def __init__(self, bank, config, gate, shared_experts):
        nn.Module.__init__(self)
        self.bank = bank
        self.config = config
        self.gate = gate
        self.shared_experts = shared_experts
        self._expert_views = tuple(
            bank.expert(expert_id, limit=config.swiglu_limit)
            for expert_id in range(bank.expert_count)
        )

    def _expert(self, expert_id: int):
        return self._expert_views[expert_id]

    def __call__(self, x):
        flat_x = x.reshape(-1, x.shape[-1])
        if flat_x.shape[0] != 1:
            return super().__call__(x)
        indices, scores = self.gate(x)
        if indices.shape[-1] != DECODE_TOP_K:
            return super().__call__(x)
        expert_ids = indices.reshape(-1).astype(mx.uint32)
        flat_scores = scores.reshape(-1)
        gate = _packed_selected_projection(
            flat_x[0], expert_ids, self.bank, row_offset=0
        )
        up = _packed_selected_projection(
            flat_x[0],
            expert_ids,
            self.bank,
            row_offset=self.bank.intermediate_size,
        )
        hidden = nn.silu(mx.minimum(gate, self.config.swiglu_limit)) * mx.clip(
            up, -self.config.swiglu_limit, self.config.swiglu_limit
        )
        result = _packed_selected_down(
            hidden, flat_scores, expert_ids, self.bank
        ).reshape(x.shape)
        if self.shared_experts is not None:
            result = result + self.shared_experts(x)
        return result
