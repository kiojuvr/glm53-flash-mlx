from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    if not mx.metal.is_available():
        raise ImportError("Metal is unavailable")
except ImportError:
    pytest.skip("MLX/Metal is unavailable", allow_module_level=True)

from glm53_flash_mlx.kda_digest import (
    compare_layerwise_digests,
    first_kda_state_difference,
    layerwise_kda_digests,
)
from glm53_flash_mlx.patch import apply_runtime_patch


def _cache(delta=0):
    apply_runtime_patch()
    from mlx_vlm.models.cache import ArraysCache

    result = [None] * 3
    for layer in (0, 2):
        entry = ArraysCache(size=2)
        wide = mx.arange(128, dtype=mx.float32).reshape(4, 32)
        entry[0] = wide[:, 1::2].astype(mx.bfloat16)
        recurrent = wide[::2, :].astype(mx.float32)
        if layer == 2 and delta:
            recurrent = recurrent + mx.array(delta, dtype=mx.float32)
        entry[1] = recurrent
        result[layer] = entry
    mx.eval([result[layer].state for layer in (0, 2)])
    return result


def test_actual_bfloat16_and_float32_state_hashes_are_bit_exact_and_nonmutating():
    cache = _cache()
    bindings = tuple(
        id(value) for layer in (0, 2) for value in cache[layer].cache
    )
    first = layerwise_kda_digests(
        cache, kda_layers=(0, 2), mx_module=mx
    )
    second = layerwise_kda_digests(
        cache, kda_layers=(0, 2), mx_module=mx
    )
    assert first == second
    assert bindings == tuple(
        id(value) for layer in (0, 2) for value in cache[layer].cache
    )


def test_actual_state_difference_localizes_layer_and_raw_coordinate():
    left = _cache()
    right = _cache(delta=1)
    left_rows = layerwise_kda_digests(left, kda_layers=(0, 2), mx_module=mx)
    right_rows = layerwise_kda_digests(right, kda_layers=(0, 2), mx_module=mx)
    digest_difference = compare_layerwise_digests(left_rows, right_rows)
    assert digest_difference["layer"] == 2
    assert digest_difference["state_kind"] == "recurrent"
    detail = first_kda_state_difference(
        left, right, kda_layers=(0, 2), mx_module=mx
    )
    assert detail["layer"] == 2
    assert detail["state_kind"] == "recurrent"
    assert detail["coordinate"] == [0, 0]
    assert detail["dtype"] == "mlx.core.float32"
