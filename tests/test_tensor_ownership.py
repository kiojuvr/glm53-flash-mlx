import gc
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    pytest.skip("MLX/Metal is unavailable", allow_module_level=True)

from glm53_flash_mlx.ownership import (
    TensorLayout,
    TensorOwnership,
    TensorOwnershipError,
    borrowed_ephemeral_tensor,
    borrowed_stable_tensor,
    materialize_owned,
    owned_tensor,
    require_resident,
    resident_concatenate,
    storage_descriptor,
)


def _require_metal() -> None:
    if not mx.metal.is_available():
        pytest.skip("MLX/Metal is unavailable")


def test_row_contiguous_does_not_promote_ephemeral_storage_to_resident():
    _require_metal()
    staging = np.arange(64, dtype=np.float32).reshape(8, 8)
    lease = borrowed_ephemeral_tensor(
        mx.asarray(staging),
        owner=staging,
        layout=TensorLayout.ROW_MAJOR_CONTIGUOUS,
    )

    with pytest.raises(TensorOwnershipError, match="materialize_owned"):
        require_resident(lease)

    resident = materialize_owned(lease)
    assert storage_descriptor(resident) == {
        "ownership": "owned",
        "layout": "row-major-contiguous",
    }
    expected = staging.copy()
    staging.fill(-91.0)
    assert np.array_equal(np.asarray(resident.value), expected)


def test_layout_and_ownership_contracts_are_independent():
    _require_metal()
    wide = mx.arange(8 * 16, dtype=mx.float32).reshape(8, 16)
    strided = wide[:, 1::2]

    resident = require_resident(owned_tensor(strided))
    assert resident.ownership is TensorOwnership.OWNED
    assert resident.layout is TensorLayout.ROW_MAJOR_CONTIGUOUS
    assert np.array_equal(
        np.asarray(resident.value), np.asarray(mx.contiguous(strided))
    )

    with pytest.raises(TypeError, match="explicit TensorLease"):
        require_resident(mx.contiguous(strided))


def test_reusable_staging_views_cannot_corrupt_resident_fused_projection():
    _require_metal()
    staging = np.empty((3, 32), dtype=np.float32)

    q_expected = np.arange(staging.size, dtype=np.float32).reshape(staging.shape)
    staging[...] = q_expected
    unsafe_q = mx.asarray(staging)
    mx.eval(unsafe_q)
    q = materialize_owned(
        borrowed_ephemeral_tensor(
            unsafe_q,
            owner=staging,
            layout=TensorLayout.ROW_MAJOR_CONTIGUOUS,
        )
    )

    kv_expected = (q_expected * -0.25) + 17.0
    staging[...] = kv_expected
    kv = materialize_owned(
        borrowed_ephemeral_tensor(
            mx.asarray(staging),
            owner=staging,
            layout=TensorLayout.ROW_MAJOR_CONTIGUOUS,
        )
    )
    # Reproduce the external failure mode: an unowned view observes reuse.
    assert np.array_equal(np.asarray(unsafe_q), kv_expected)

    fused = resident_concatenate(
        [
            owned_tensor(q.value, layout=q.layout),
            owned_tensor(kv.value, layout=kv.layout),
        ],
        axis=0,
    )
    expected = np.concatenate([q_expected, kv_expected], axis=0)
    staging.fill(12345.0)
    del unsafe_q, staging
    gc.collect()

    assert np.array_equal(np.asarray(q.value), q_expected)
    assert np.array_equal(np.asarray(kv.value), kv_expected)
    assert np.array_equal(np.asarray(fused.value), expected)


def test_packed_fp8_bank_owns_reusable_staging_weights_and_scales():
    _require_metal()
    from glm53_flash_mlx.packed import PackedFP8ExpertBank, PackedFP8MoE

    experts = 8
    hidden = 128
    intermediate = 128

    weight_staging = np.empty(
        (experts, 2 * intermediate, hidden), dtype=np.uint8
    )
    gate_up_expected = (
        np.arange(weight_staging.size, dtype=np.uint32).reshape(
            weight_staging.shape
        )
        % 120
    ).astype(np.uint8)
    weight_staging[...] = gate_up_expected
    gate_up = materialize_owned(
        borrowed_ephemeral_tensor(
            mx.asarray(weight_staging),
            owner=weight_staging,
            layout=TensorLayout.ROW_MAJOR_CONTIGUOUS,
        )
    )

    down_expected = (
        np.arange(experts * hidden * intermediate, dtype=np.uint32).reshape(
            experts, hidden, intermediate
        )
        * 7
        % 120
    ).astype(np.uint8)
    weight_staging[:, :hidden, :] = down_expected
    down = materialize_owned(
        borrowed_ephemeral_tensor(
            mx.asarray(weight_staging[:, :hidden, :]),
            owner=weight_staging,
            layout=TensorLayout.ROW_MAJOR_CONTIGUOUS,
        )
    )

    scale_staging = np.empty((experts, 2, 1), dtype=np.float32)
    gate_up_scale_expected = np.linspace(
        0.001, 0.008, scale_staging.size, dtype=np.float32
    ).reshape(scale_staging.shape)
    scale_staging[...] = gate_up_scale_expected
    gate_up_scale = materialize_owned(
        borrowed_ephemeral_tensor(
            mx.asarray(scale_staging),
            owner=scale_staging,
            layout=TensorLayout.ROW_MAJOR_CONTIGUOUS,
        )
    )

    down_scale_expected = np.linspace(
        0.002, 0.009, experts, dtype=np.float32
    ).reshape(experts, 1, 1)
    scale_staging[:, :1, :] = down_scale_expected
    down_scale = materialize_owned(
        borrowed_ephemeral_tensor(
            mx.asarray(scale_staging[:, :1, :]),
            owner=scale_staging,
            layout=TensorLayout.ROW_MAJOR_CONTIGUOUS,
        )
    )

    bank = PackedFP8ExpertBank(
        gate_up,
        gate_up_scale,
        down,
        down_scale,
        intermediate_size=intermediate,
    )
    assert set(tuple(row.values()) for row in bank.storage_contracts.values()) == {
        ("owned", "row-major-contiguous")
    }

    routes = mx.arange(experts, dtype=mx.uint32).reshape(1, 1, experts)
    scores = mx.full((1, 1, experts), 1.0 / experts, dtype=mx.float32)

    class FixedGate(nn.Module):
        def __call__(self, x):
            return routes, scores.astype(x.dtype)

    config = SimpleNamespace(
        hidden_size=hidden,
        moe_intermediate_size=intermediate,
        swiglu_limit=3.0,
        n_routed_experts=experts,
        num_experts_per_tok=experts,
    )
    moe = PackedFP8MoE(bank, config, FixedGate(), None)
    x = mx.linspace(-0.25, 0.25, hidden).reshape(1, 1, hidden).astype(
        mx.bfloat16
    )
    output_before = moe(x)
    mx.eval(output_before)

    weight_staging.fill(255)
    scale_staging.fill(-7.0)
    del weight_staging, scale_staging
    gc.collect()
    output_after = moe(x)
    mx.eval(output_after)

    assert bank.gate_up_weight.dtype == mx.uint8
    assert bank.gate_up_scale_inv.dtype == mx.float32
    assert bank.down_weight.dtype == mx.uint8
    assert bank.down_scale_inv.dtype == mx.float32
    assert np.array_equal(np.asarray(bank.gate_up_weight), gate_up_expected)
    assert np.array_equal(
        np.asarray(bank.gate_up_scale_inv), gate_up_scale_expected
    )
    assert np.array_equal(np.asarray(bank.down_weight), down_expected)
    assert np.array_equal(np.asarray(bank.down_scale_inv), down_scale_expected)
    assert mx.array_equal(output_before, output_after).item()


def test_packed_bank_rejects_ephemeral_and_bare_storage():
    _require_metal()
    from glm53_flash_mlx.packed import PackedFP8ExpertBank

    weight = mx.zeros((8, 256, 128), dtype=mx.uint8)
    scale = mx.ones((8, 2, 1), dtype=mx.float32)
    down = mx.zeros((8, 128, 128), dtype=mx.uint8)
    down_scale = mx.ones((8, 1, 1), dtype=mx.float32)

    with pytest.raises(TensorOwnershipError, match="materialize_owned"):
        PackedFP8ExpertBank(
            borrowed_ephemeral_tensor(weight),
            owned_tensor(scale),
            owned_tensor(down),
            owned_tensor(down_scale),
            intermediate_size=128,
        )
    with pytest.raises(TypeError, match="explicit TensorLease"):
        PackedFP8ExpertBank(
            weight,
            owned_tensor(scale),
            owned_tensor(down),
            owned_tensor(down_scale),
            intermediate_size=128,
        )


def test_borrowed_stable_requires_a_retained_owner():
    _require_metal()
    value = mx.zeros((4, 4), dtype=mx.float32)
    with pytest.raises(TensorOwnershipError, match="requires an owner"):
        borrowed_stable_tensor(value, owner=None)
