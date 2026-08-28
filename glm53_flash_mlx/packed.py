"""Storage-only packed FP8 expert bank used by the feasibility probe."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .fp8 import BLOCK_SIZE, DirectFP8MoE, block_fp8_linear


class _PackedLinearView:
    def __init__(self, weight: mx.array, scale_inv: mx.array):
        self.weight = weight
        self.weight_scale_inv = scale_inv

    def __call__(self, x):
        return block_fp8_linear(x, self.weight, self.weight_scale_inv)


class _PackedExpertView:
    def __init__(self, bank: "PackedFP8ExpertBank", expert_id: int, limit: float):
        intermediate = bank.intermediate_size
        scale_rows = bank.intermediate_scale_rows
        self.gate_proj = _PackedLinearView(
            bank.gate_up_weight[expert_id, :intermediate],
            bank.gate_up_scale_inv[expert_id, :scale_rows],
        )
        self.up_proj = _PackedLinearView(
            bank.gate_up_weight[expert_id, intermediate:],
            bank.gate_up_scale_inv[expert_id, scale_rows:],
        )
        self.down_proj = _PackedLinearView(
            bank.down_weight[expert_id], bank.down_scale_inv[expert_id]
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
        gate_up_weight: mx.array,
        gate_up_scale_inv: mx.array,
        down_weight: mx.array,
        down_scale_inv: mx.array,
        *,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_up_weight = gate_up_weight
        self.gate_up_scale_inv = gate_up_scale_inv
        self.down_weight = down_weight
        self.down_scale_inv = down_scale_inv
        self.intermediate_size = int(intermediate_size)
        self.intermediate_scale_rows = math.ceil(intermediate_size / BLOCK_SIZE)

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

        gate_up_weight = mx.concatenate(
            [
                projection.weight
                for expert in experts
                for projection in (expert.gate_proj, expert.up_proj)
            ],
            axis=0,
        ).reshape(expert_count, 2 * intermediate_size, hidden_size)
        gate_up_scale_inv = mx.concatenate(
            [
                projection.weight_scale_inv
                for expert in experts
                for projection in (expert.gate_proj, expert.up_proj)
            ],
            axis=0,
        ).reshape(expert_count, 2 * scale_rows, hidden_scale_rows)
        down_weight = mx.concatenate(
            [expert.down_proj.weight for expert in experts], axis=0
        ).reshape(expert_count, hidden_size, intermediate_size)
        down_scale_inv = mx.concatenate(
            [expert.down_proj.weight_scale_inv for expert in experts], axis=0
        ).reshape(expert_count, hidden_scale_rows, scale_rows)
        return cls(
            gate_up_weight,
            gate_up_scale_inv,
            down_weight,
            down_scale_inv,
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
        return _PackedExpertView(self, expert_id, limit)


class PackedFP8MoE(DirectFP8MoE):
    """Existing MoE execution semantics backed by a contiguous expert bank."""

    def __init__(self, bank, config, gate, shared_experts):
        nn.Module.__init__(self)
        self.bank = bank
        self.config = config
        self.gate = gate
        self.shared_experts = shared_experts

    def _expert(self, expert_id: int):
        return self.bank.expert(expert_id, limit=self.config.swiglu_limit)
