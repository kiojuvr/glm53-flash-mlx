from __future__ import annotations

from types import SimpleNamespace

import pytest

try:
    import mlx.core as mx
except ImportError:
    pytest.skip("MLX/Metal is unavailable", allow_module_level=True)

from glm53_flash_mlx.cache_geometry import plan_nope_cache_capacity
from glm53_flash_mlx.nope_cache import make_compact_nope_dsa_cache
from glm53_flash_mlx.patch import apply_runtime_patch


def _indexer():
    apply_runtime_patch()
    from mlx_vlm.models.glm5_next.language import Glm5NextIndexer

    config = SimpleNamespace(
        hidden_size=8,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2048,
        index_kpool=4,
        index_kpool_always_select_tail=True,
        q_lora_rank=4,
    )
    mx.random.seed(911)
    value = Glm5NextIndexer(config)
    value.set_dtype(mx.bfloat16)
    return value


def _inputs(start: int, tokens: int):
    positions = mx.arange(start, start + tokens, dtype=mx.float32)[:, None]
    x = mx.sin(positions * 0.03125 + mx.arange(8)[None] * 0.0625)[None]
    qr = mx.cos(positions * 0.046875 + mx.arange(4)[None] * 0.09375)[None]
    latent = mx.sin(
        positions * 0.015625 + mx.arange(4)[None] * 0.125
    ).astype(mx.bfloat16).reshape(1, 1, tokens, 4)
    return x.astype(mx.bfloat16), qr.astype(mx.bfloat16), latent


def _append(indexer, cache, start: int, tokens: int):
    x, qr, latent = _inputs(start, tokens)
    selected = indexer(x, qr, None, cache=cache[1])
    cache[0].update_and_fetch(latent, latent)
    return selected


@pytest.mark.parametrize("logical_capacity", [255, 256, 257, 511, 512, 513])
def test_latent_and_indexpool_share_the_authoritative_physical_geometry(
    logical_capacity,
):
    indexer = _indexer()
    cache = make_compact_nope_dsa_cache(
        indexer, capacity_tokens=logical_capacity
    )
    _append(indexer, cache, 0, 1)
    plan = plan_nope_cache_capacity(logical_capacity)
    assert cache[0].physical_capacity_tokens == plan.physical_capacity_tokens
    assert cache[1].physical_capacity_rows == plan.physical_pool_rows
    assert cache[1].logical_pool_count == 1

    padding_start = cache[1].logical_pool_count
    assert mx.all(cache[1].pool_indices[:, padding_start:] == -1).item()
    assert not mx.any(cache[1].pool_valid[:, padding_start:]).item()
    assert mx.all(cache[1].pool_keys[:, padding_start:] == 0).item()


def test_trim_retires_pool_rows_to_sentinel_padding_and_replay_is_exact():
    from mlx_vlm.apc_adapters import clone_cache_entry

    indexer = _indexer()
    oracle = make_compact_nope_dsa_cache(indexer, capacity_tokens=513)
    _append(indexer, oracle, 0, 256)
    expected = []
    for position in range(256, 273):
        expected.append(_append(indexer, oracle, position, 1))

    targets = []
    replay = clone_cache_entry(
        oracle, min_capacity_tokens=513, eval_targets=targets
    )
    mx.eval(*targets)
    replay.trim(16)
    assert replay[1].total_tokens == 257
    assert replay[1].logical_pool_count == 65
    assert mx.all(replay[1].pool_indices[:, 65:] == -1).item()
    assert not mx.any(replay[1].pool_valid[:, 65:]).item()
    assert mx.all(replay[1].pool_keys[:, 65:] == 0).item()

    for position, left in zip(range(257, 273), expected[1:], strict=True):
        right = _append(indexer, replay, position, 1)
        if left is None:
            assert right is None
        else:
            assert mx.array_equal(left, right).item()
    for left, right in zip(oracle.state, replay.state, strict=True):
        for left_child, right_child in zip(left, right, strict=True):
            if left_child is None:
                assert right_child is None
            else:
                assert mx.array_equal(left_child, right_child).item()
    assert oracle.meta_state == replay.meta_state
    assert replay[0].physical_capacity_tokens == 768
    assert replay[1].physical_capacity_rows == 192


def test_incompatible_restored_geometry_is_rejected_before_reallocation():
    indexer = _indexer()
    cache = make_compact_nope_dsa_cache(indexer, capacity_tokens=257)
    _append(indexer, cache, 0, 1)
    latent_meta = list(cache[0].meta_state)
    pool_meta = list(cache[1].meta_state)
    latent_meta[-1] = "128"
    pool_meta[-1] = "128"
    with pytest.raises(ValueError, match="tile alignment"):
        cache[0].meta_state = tuple(latent_meta)
    with pytest.raises(ValueError, match="tile alignment"):
        cache[1].meta_state = tuple(pool_meta)
