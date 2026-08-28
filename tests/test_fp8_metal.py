import math
from types import SimpleNamespace

import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    mx = None
    nn = None


def _require_metal():
    if mx is None:
        pytest.skip("MLX/Metal is unavailable in this session")
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")


def test_block_fp8_kernel_matches_dequantized_matmul():
    _require_metal()
    from glm53_flash_mlx.fp8 import block_fp8_linear

    # Dimensions cross both output and reduction block boundaries.
    m, k = 129, 257
    codes = mx.random.randint(0, 247, shape=(m, k)).astype(mx.uint8)
    scales = mx.random.uniform(shape=(math.ceil(m / 128), math.ceil(k / 128))).astype(mx.float32) * 0.02
    x = mx.random.normal(shape=(2, k)).astype(mx.bfloat16)
    actual = block_fp8_linear(x, codes, scales)

    dense = mx.from_fp8(codes, dtype=mx.float32)
    expanded = mx.repeat(mx.repeat(scales, 128, axis=0), 128, axis=1)[:m, :k]
    expected = (x.astype(mx.float32) @ (dense * expanded).T).astype(mx.bfloat16)
    mx.eval(actual, expected)
    error = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()
    assert error <= 0.03125


def test_selected_top8_moe_matches_explicit_routing_and_scores():
    _require_metal()
    from glm53_flash_mlx.fp8 import DirectFP8MoE

    routes = mx.array([[[7, 1, 5, 0, 3, 6, 2, 4]]], dtype=mx.int32)
    scores = mx.array([[[0.24, 0.18, 0.15, 0.13, 0.11, 0.08, 0.06, 0.05]]])

    class FixedGate(nn.Module):
        def __call__(self, _):
            return routes, scores

    config = SimpleNamespace(
        hidden_size=257,
        moe_intermediate_size=129,
        swiglu_limit=3.0,
        n_routed_experts=8,
        num_experts_per_tok=8,
    )
    moe = DirectFP8MoE(config, FixedGate(), None)
    for expert in moe.experts:
        for projection in (expert.gate_proj, expert.up_proj, expert.down_proj):
            projection.weight = mx.random.randint(
                0, 247, shape=projection.weight.shape
            ).astype(mx.uint8)
            projection.weight_scale_inv = (
                mx.random.uniform(shape=projection.weight_scale_inv.shape) * 0.002
            ).astype(mx.float32)

    x = (mx.random.normal(shape=(1, 1, 257)) * 2).astype(mx.bfloat16)
    actual = moe(x)
    flat = x.reshape(1, 257)
    expected = mx.zeros_like(flat)
    for slot, expert_id in enumerate([7, 1, 5, 0, 3, 6, 2, 4]):
        expert = moe.experts[expert_id]
        gate = mx.minimum(expert.gate_proj(flat), config.swiglu_limit)
        up = mx.clip(
            expert.up_proj(flat), -config.swiglu_limit, config.swiglu_limit
        )
        expected = expected + expert.down_proj(nn.silu(gate) * up) * scores[0, 0, slot]
    expected = expected.reshape(actual.shape)
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, rtol=0.02, atol=0.02).item()
