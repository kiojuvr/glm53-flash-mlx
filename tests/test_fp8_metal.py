import importlib.util
import math
from pathlib import Path
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
        def __call__(self, x):
            shape = (*x.shape[:-1], 8)
            return mx.broadcast_to(routes, shape), mx.broadcast_to(scores, shape)

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


def test_packed_expert_bank_preserves_bytes_and_selected_top8_output():
    _require_metal()
    from mlx.utils import tree_flatten

    from glm53_flash_mlx.fp8 import DirectFP8MoE
    from glm53_flash_mlx.packed import PackedFP8ExpertBank, PackedFP8MoE

    routes = mx.array([[[7, 1, 5, 0, 3, 6, 2, 4]]], dtype=mx.int32)
    scores = mx.array([[[0.24, 0.18, 0.15, 0.13, 0.11, 0.08, 0.06, 0.05]]])

    class FixedGate(nn.Module):
        def __call__(self, x):
            shape = (*x.shape[:-1], 8)
            return mx.broadcast_to(routes, shape), mx.broadcast_to(scores, shape)

    config = SimpleNamespace(
        hidden_size=257,
        moe_intermediate_size=129,
        swiglu_limit=3.0,
        n_routed_experts=8,
        num_experts_per_tok=8,
    )
    direct = DirectFP8MoE(config, FixedGate(), None)
    for expert in direct.experts:
        for projection in (expert.gate_proj, expert.up_proj, expert.down_proj):
            projection.weight = mx.random.randint(
                0, 247, shape=projection.weight.shape
            ).astype(mx.uint8)
            projection.weight_scale_inv = (
                mx.random.uniform(shape=projection.weight_scale_inv.shape) * 0.002
            ).astype(mx.float32)

    bank = PackedFP8ExpertBank.pack(direct.experts)
    mx.eval(*[value for _, value in tree_flatten(bank.parameters())])
    assert bank.gate_up_weight.shape == (8, 258, 257)
    assert bank.gate_up_scale_inv.shape == (8, 4, 3)
    assert bank.down_weight.shape == (8, 257, 129)
    assert bank.down_scale_inv.shape == (8, 3, 2)
    for expert_id, expert in enumerate(direct.experts):
        packed = bank.expert(expert_id, limit=config.swiglu_limit)
        assert mx.array_equal(packed.gate_proj.weight, expert.gate_proj.weight).item()
        assert mx.array_equal(packed.up_proj.weight, expert.up_proj.weight).item()
        assert mx.array_equal(packed.down_proj.weight, expert.down_proj.weight).item()
        assert mx.array_equal(
            packed.gate_proj.weight_scale_inv, expert.gate_proj.weight_scale_inv
        ).item()
        assert mx.array_equal(
            packed.up_proj.weight_scale_inv, expert.up_proj.weight_scale_inv
        ).item()
        assert mx.array_equal(
            packed.down_proj.weight_scale_inv, expert.down_proj.weight_scale_inv
        ).item()

    packed_moe = PackedFP8MoE(bank, config, direct.gate, None)
    x = (mx.random.normal(shape=(1, 1, 257)) * 2).astype(mx.bfloat16)
    expected = direct(x)
    actual = packed_moe(x)
    mx.eval(expected, actual)
    assert mx.array_equal(actual, expected).item()

    prefill_x = (mx.random.normal(shape=(1, 9, 257)) * 2).astype(mx.bfloat16)
    expected_prefill = direct(prefill_x)
    actual_prefill = packed_moe(prefill_x)
    mx.eval(expected_prefill, actual_prefill)
    assert mx.array_equal(actual_prefill, expected_prefill).item()


def test_sorted_grouped_fp8_moe_matches_direct_prefill_and_keeps_decode_fallback():
    _require_metal()
    from glm53_flash_mlx.fp8 import DirectFP8MoE
    from glm53_flash_mlx.grouped_fp8 import SortedGroupedFP8MoE
    from glm53_flash_mlx.packed import PackedFP8ExpertBank

    class DeterministicGate(nn.Module):
        def __call__(self, x):
            rows = x.reshape(-1, x.shape[-1]).shape[0]
            routes = mx.arange(rows * 8, dtype=mx.uint32).reshape(rows, 8)
            indices = ((routes * 5 + 3) % 8).reshape(*x.shape[:-1], 8)
            scores = mx.full(indices.shape, 0.125, dtype=x.dtype)
            return indices, scores

    config = SimpleNamespace(
        hidden_size=256,
        moe_intermediate_size=128,
        swiglu_limit=3.0,
        n_routed_experts=8,
        num_experts_per_tok=8,
    )
    direct = DirectFP8MoE(config, DeterministicGate(), None)
    for expert in direct.experts:
        for projection in (expert.gate_proj, expert.up_proj, expert.down_proj):
            projection.weight = mx.random.randint(
                0, 247, shape=projection.weight.shape
            ).astype(mx.uint8)
            projection.weight_scale_inv = (
                mx.random.uniform(shape=projection.weight_scale_inv.shape) * 0.002
            ).astype(mx.float32)
    bank = PackedFP8ExpertBank.pack(direct.experts)
    grouped = SortedGroupedFP8MoE(bank, config, direct.gate, None)
    assert grouped.min_routes == 256

    prefill_x = mx.random.normal(shape=(1, 32, 256)).astype(mx.bfloat16)
    expected = direct(prefill_x)
    actual = grouped(prefill_x)
    mx.eval(expected, actual)
    assert mx.allclose(actual, expected, rtol=0.02, atol=0.02).item()

    decode_x = mx.random.normal(shape=(1, 1, 256)).astype(mx.bfloat16)
    expected_decode = direct(decode_x)
    actual_decode = grouped(decode_x)
    mx.eval(expected_decode, actual_decode)
    assert mx.array_equal(actual_decode, expected_decode).item()


def test_grouped_tile_descriptors_cover_sparse_edge_buckets_without_boundaries():
    _require_metal()
    import numpy as np

    from glm53_flash_mlx.grouped_fp8 import build_grouped_tile_plan

    spec = importlib.util.spec_from_file_location(
        "probe_grouped_fp8_moe",
        Path(__file__).parents[1] / "scripts" / "probe_grouped_fp8_moe.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    route_metrics = module._route_metrics

    sorted_experts = mx.array(
        [0] * 31 + [1] * 32 + [287] * 33, dtype=mx.uint32
    )
    tile_plan = build_grouped_tile_plan(sorted_experts, expert_count=288)
    metrics = route_metrics(sorted_experts, tile_plan, expert_count=288)

    assert metrics["unique_experts"] == 3
    assert metrics["zero_route_experts"] == 285
    assert metrics["routes_per_expert_max"] == 33
    assert metrics["expert_boundary_tiles"] == 0
    assert metrics["descriptor_routes_covered_once"]
    assert metrics["aligned_route_tiles"] == 4

    experts, starts, lengths = (
        np.asarray(value) for value in tile_plan[:3]
    )
    valid = starts < sorted_experts.shape[0]
    assert list(zip(experts[valid], starts[valid], lengths[valid], strict=True)) == [
        (0, 0, 31),
        (1, 31, 32),
        (287, 63, 32),
        (287, 95, 1),
    ]


def test_installer_replaces_every_direct_moe_layer_one_at_a_time():
    _require_metal()
    from glm53_flash_mlx.fp8 import DirectFP8MoE
    from glm53_flash_mlx.grouped_fp8 import SortedGroupedFP8MoE
    from glm53_flash_mlx.loader import install_packed_grouped_moe

    class Gate(nn.Module):
        def __call__(self, x):
            shape = (*x.shape[:-1], 2)
            return mx.zeros(shape, dtype=mx.uint32), mx.ones(shape, dtype=x.dtype)

    config = SimpleNamespace(
        hidden_size=128,
        moe_intermediate_size=128,
        swiglu_limit=3.0,
        n_routed_experts=2,
        num_experts_per_tok=2,
    )
    layers = []
    for _ in range(3):
        layer = SimpleNamespace(
            mlp=DirectFP8MoE(config, Gate(), None), compile_ffn=True
        )
        for expert in layer.mlp.experts:
            for projection in (expert.gate_proj, expert.up_proj, expert.down_proj):
                projection.weight = mx.zeros(projection.weight.shape, dtype=mx.uint8)
        layers.append(layer)
    model = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(layers=layers)
        )
    )

    report = install_packed_grouped_moe(model)
    assert report["converted_layers"] == [0, 1, 2]
    assert report["converted_count"] == 3
    assert report["remaining_direct_layers"] == []
    assert report["all_old_expert_modules_detached"]
    assert all(isinstance(layer.mlp, SortedGroupedFP8MoE) for layer in layers)
    assert all(not hasattr(layer.mlp, "experts") for layer in layers)
    assert all(not layer.compile_ffn for layer in layers)
    second = install_packed_grouped_moe(model)
    assert second["converted_count"] == 0
    assert second["remaining_direct_layers"] == []


def test_packed_decode_installer_never_installs_grouped_modules():
    _require_metal()
    from glm53_flash_mlx.fp8 import DirectFP8MoE
    from glm53_flash_mlx.grouped_fp8 import SortedGroupedFP8MoE
    from glm53_flash_mlx.loader import install_packed_decode_moe
    from glm53_flash_mlx.packed import PackedFP8MoE

    class Gate(nn.Module):
        def __call__(self, x):
            shape = (*x.shape[:-1], 2)
            return mx.zeros(shape, dtype=mx.uint32), mx.ones(shape, dtype=x.dtype)

    config = SimpleNamespace(
        hidden_size=128,
        moe_intermediate_size=128,
        swiglu_limit=3.0,
        n_routed_experts=2,
        num_experts_per_tok=2,
    )
    layers = []
    for _ in range(2):
        layer = SimpleNamespace(
            mlp=DirectFP8MoE(config, Gate(), None), compile_ffn=True
        )
        for expert in layer.mlp.experts:
            for projection in (expert.gate_proj, expert.up_proj, expert.down_proj):
                projection.weight = mx.zeros(projection.weight.shape, dtype=mx.uint8)
        layers.append(layer)
    model = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=layers))
    )

    report = install_packed_decode_moe(model)
    assert report["backend"] == "packed-decode"
    assert report["grouped_enabled"] is False
    assert report["converted_count"] == 2
    assert report["all_old_expert_modules_detached"]
    assert all(isinstance(layer.mlp, PackedFP8MoE) for layer in layers)
    assert all(not isinstance(layer.mlp, SortedGroupedFP8MoE) for layer in layers)
    assert model._glm53_moe_backend == "packed-decode"
    assert model._glm53_packed_decode_report is report
    second = install_packed_decode_moe(model)
    assert second["converted_count"] == 0
