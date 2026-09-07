from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKED = ROOT / "glm53_flash_mlx" / "packed.py"


def _mlx():
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("MLX/Metal is unavailable")
    if not mx.metal.is_available():
        pytest.skip("MLX/Metal is unavailable")
    return mx


def _bank(mx):
    from glm53_flash_mlx.ownership import TensorLayout, owned_tensor
    from glm53_flash_mlx.packed import PackedFP8ExpertBank

    experts = 8
    hidden = 128
    intermediate = 128
    gate_up = (
        (mx.arange(experts * 2 * intermediate * hidden, dtype=mx.uint32) * 7 + 3)
        % 256
    ).astype(mx.uint8).reshape(experts, 2 * intermediate, hidden)
    gate_up_scale = mx.full((experts, 2, 1), 0.015625, dtype=mx.float32)
    down = (
        (mx.arange(experts * hidden * intermediate, dtype=mx.uint32) * 13 + 5)
        % 256
    ).astype(mx.uint8).reshape(experts, hidden, intermediate)
    down_scale = mx.full((experts, 1, 1), 0.0078125, dtype=mx.float32)
    return PackedFP8ExpertBank(
        owned_tensor(gate_up, layout=TensorLayout.ROW_MAJOR_CONTIGUOUS),
        owned_tensor(gate_up_scale, layout=TensorLayout.ROW_MAJOR_CONTIGUOUS),
        owned_tensor(down, layout=TensorLayout.ROW_MAJOR_CONTIGUOUS),
        owned_tensor(down_scale, layout=TensorLayout.ROW_MAJOR_CONTIGUOUS),
        intermediate_size=intermediate,
    )


def test_runtime_uses_the_qualified_exact_fused_topology():
    source = PACKED.read_text()
    body = source[source.index("class PackedFP8MoE") :]
    assert "_packed_selected_fused_gate_up_swiglu(" in body
    assert "_packed_selected_down_raw(" in body
    assert "_packed_selected_weighted_reduction(" in body
    assert "_shared_fused_gate_up_swiglu(" in body
    assert "return super().__call__(x)" in body
    assert "_packed_selected_projection(" not in body


def test_packed_decode_abi_names_all_exact_fused_boundaries():
    from glm53_flash_mlx.abi import PACKED_DECODE_KERNEL_ABI

    assert PACKED_DECODE_KERNEL_ABI.startswith("glm53-packed-selected8-fp8-v2-")
    assert "exact-gate-up-swiglu" in PACKED_DECODE_KERNEL_ABI
    assert "exact-weighted-reduction" in PACKED_DECODE_KERNEL_ABI
    assert "exact-shared-gate-up-swiglu" in PACKED_DECODE_KERNEL_ABI


def test_routed_fused_topology_is_byte_exact_to_existing_mlx_composition():
    mx = _mlx()
    import mlx.nn as nn

    from glm53_flash_mlx.packed import (
        _packed_selected_down_raw,
        _packed_selected_fused_gate_up_swiglu,
        _packed_selected_projection,
        _packed_selected_weighted_reduction,
    )

    bank = _bank(mx)
    x = (mx.sin(mx.arange(128, dtype=mx.float32) * 0.03125) * 0.25).astype(
        mx.bfloat16
    )
    ids = mx.arange(8, dtype=mx.uint32)
    scores = mx.array(
        [0.28125, 0.203125, 0.15625, 0.125, 0.09375, 0.0625, 0.046875, 0.03125],
        dtype=mx.float32,
    )
    gate = _packed_selected_projection(x, ids, bank, row_offset=0)
    up = _packed_selected_projection(x, ids, bank, row_offset=128)
    expected_hidden = nn.silu(mx.minimum(gate, 10.0)) * mx.clip(up, -10.0, 10.0)
    actual_hidden = _packed_selected_fused_gate_up_swiglu(
        x, ids, bank, limit=10.0
    )
    raw = _packed_selected_down_raw(actual_hidden, ids, bank)
    expected = mx.sum(raw.astype(mx.float32) * scores[:, None], axis=0).astype(
        mx.bfloat16
    )
    actual = _packed_selected_weighted_reduction(raw, scores)
    mx.eval(expected_hidden, actual_hidden, expected, actual)
    assert mx.array_equal(expected_hidden, actual_hidden).item()
    assert mx.array_equal(expected, actual).item()


def test_shared_fused_gate_up_swiglu_is_byte_exact():
    mx = _mlx()
    import mlx.nn as nn

    from glm53_flash_mlx.fp8 import BlockFP8Linear
    from glm53_flash_mlx.packed import _shared_fused_gate_up_swiglu

    shared = SimpleNamespace(
        gate_proj=BlockFP8Linear(128, 128),
        up_proj=BlockFP8Linear(128, 128),
    )
    shared.gate_proj.weight = (
        mx.arange(128 * 128, dtype=mx.uint32) % 256
    ).astype(mx.uint8).reshape(128, 128)
    shared.up_proj.weight = (
        (mx.arange(128 * 128, dtype=mx.uint32) * 7 + 3) % 256
    ).astype(mx.uint8).reshape(128, 128)
    shared.gate_proj.weight_scale_inv = mx.full(
        (1, 1), 0.015625, dtype=mx.float32
    )
    shared.up_proj.weight_scale_inv = mx.full(
        (1, 1), 0.0078125, dtype=mx.float32
    )
    x = (mx.cos(mx.arange(128, dtype=mx.float32) * 0.03125) * 0.25).astype(
        mx.bfloat16
    )
    gate = shared.gate_proj(x)
    up = shared.up_proj(x)
    expected = nn.silu(mx.minimum(gate, 10.0)) * mx.clip(up, -10.0, 10.0)
    actual = _shared_fused_gate_up_swiglu(x, shared, limit=10.0)
    mx.eval(expected, actual)
    assert mx.array_equal(expected, actual).item()
