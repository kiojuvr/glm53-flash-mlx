import importlib
import inspect
from types import SimpleNamespace

import pytest

try:
    import mlx.core as mx
except ImportError:
    pytest.skip("MLX/Metal is unavailable", allow_module_level=True)

from glm53_flash_mlx.nope_cache import (
    CompactIndexPoolCache,
    SingleNoPELatentCache,
    make_compact_nope_dsa_cache,
)
from glm53_flash_mlx.patch import apply_runtime_patch


def _make_indexer(*, topk=32, kpool=4):
    apply_runtime_patch()
    from mlx_vlm.models.glm5_next.language import Glm5NextIndexer

    config = SimpleNamespace(
        hidden_size=8,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=topk,
        index_kpool=kpool,
        index_kpool_always_select_tail=True,
        q_lora_rank=4,
    )
    mx.random.seed(73)
    indexer = Glm5NextIndexer(config)
    indexer.set_dtype(mx.bfloat16)
    return indexer


def _inputs(start: int, tokens: int):
    positions = mx.arange(start, start + tokens, dtype=mx.float32)[:, None]
    x = mx.sin(positions * 0.03125 + mx.arange(8)[None] * 0.0625)[None]
    qr = mx.cos(positions * 0.046875 + mx.arange(4)[None] * 0.09375)[None]
    return x.astype(mx.bfloat16), qr.astype(mx.bfloat16)


def _assert_equal(left, right):
    assert left.shape == right.shape
    assert mx.array_equal(left, right).item()


def _assert_pool(indexer, cache, x, mask=None):
    keys = indexer.k_norm(indexer.wk(x)).reshape(1, x.shape[1], indexer.head_dim)
    gates = x @ indexer.index_kpool_compress_gate.swapaxes(-1, -2)
    valid = (
        mask
        if mask is not None
        else mx.ones((1, x.shape[1]), dtype=mx.bool_)
    )
    expected = indexer._pooled_states(keys, gates, valid)
    for left, right in zip(expected, cache.logical_pool(), strict=True):
        _assert_equal(left, right)


@pytest.mark.parametrize("prompt_tokens", [1, 16, 128, 256])
def test_prefill_builds_compact_state_without_packed_history(prompt_tokens):
    indexer = _make_indexer(topk=2048)
    cache = CompactIndexPoolCache(indexer, reserve_tokens=4096)
    x, qr = _inputs(0, prompt_tokens)

    assert indexer(x, qr, None, cache=cache) is None
    _assert_pool(indexer, cache, x)
    assert cache.total_tokens == prompt_tokens
    assert cache.raw_token_count == min(prompt_tokens, cache.raw_state_window)
    assert cache.raw_state_window == 16 + indexer.index_kpool - 1
    assert not hasattr(cache, "keys")
    assert not hasattr(cache, "packed_token_history")


def test_left_padded_prefill_preserves_pool_validity_and_sentinel():
    indexer = _make_indexer(topk=2048)
    cache = CompactIndexPoolCache(indexer, reserve_tokens=16)
    x, qr = _inputs(0, 16)
    mask = mx.arange(16)[None] >= 3

    assert indexer(x, qr, mask, cache=cache) is None
    _assert_pool(indexer, cache, x, mask)
    values = cache.pool_indices[:, : cache.logical_pool_count]
    assert mx.all((values == -1) | ((values >= 0) & (values < 16))).item()
    assert not mx.any(
        mx.isnan(cache.pool_keys[:, : cache.logical_pool_count].astype(mx.float32))
    ).item()


def test_sparse_decode_indices_match_full_history_cache_for_16_steps():
    from mlx_vlm.models.cache import KVCache

    indexer = _make_indexer(topk=32)
    direct = KVCache()
    compact = CompactIndexPoolCache(indexer, reserve_tokens=64)
    x, qr = _inputs(0, 32)
    assert indexer(x, qr, None, cache=direct) is None
    assert indexer(x, qr, None, cache=compact) is None

    for step in range(16):
        x, qr = _inputs(32 + step, 1)
        expected = indexer(x, qr, None, cache=direct)
        actual = indexer(x, qr, None, cache=compact)
        _assert_equal(expected, actual)
        for left, right in zip(direct._pool[:3], compact.logical_pool(), strict=True):
            _assert_equal(left, right)
        assert compact.raw_token_count <= compact.raw_state_window
        assert mx.all(
            (actual == -1) | ((actual >= 0) & (actual < compact.total_tokens))
        ).item()
        assert not mx.any(
            mx.isnan(compact.pool_keys[:, : compact.logical_pool_count].astype(mx.float32))
        ).item()


@pytest.mark.parametrize("target_mod", range(4))
def test_ram_apc_clone_restores_each_pool_tail_position(target_mod):
    from mlx_vlm.apc_adapters import clone_cache_entry

    indexer = _make_indexer(topk=32)
    original = make_compact_nope_dsa_cache(indexer, reserve_tokens=64)
    x, qr = _inputs(0, 32)
    indexer(x, qr, None, cache=original[1])
    for position in range(32, 32 + target_mod):
        x, qr = _inputs(position, 1)
        indexer(x, qr, None, cache=original[1])
    latent = mx.zeros((1, 1, 32 + target_mod, 4), dtype=mx.bfloat16)
    original[0].update_and_fetch(latent, latent)

    eval_targets = []
    restored = clone_cache_entry(
        original, min_capacity_tokens=32 + target_mod + 16, eval_targets=eval_targets
    )
    mx.eval(*eval_targets)
    assert restored is not None

    for step in range(16):
        position = 32 + target_mod + step
        x, qr = _inputs(position, 1)
        expected = indexer(x, qr, None, cache=original[1])
        actual = indexer(x, qr, None, cache=restored[1])
        _assert_equal(expected, actual)
        token = mx.full((1, 1, 1, 4), position, dtype=mx.bfloat16)
        left, _ = original[0].update_and_fetch(token, token)
        right, _ = restored[0].update_and_fetch(token, token)
        _assert_equal(left, right)


def test_capacity_growth_covers_255_256_257_and_4095_4096():
    indexer = _make_indexer(topk=8192)
    cache = CompactIndexPoolCache(indexer, reserve_tokens=0)
    for start, count in ((0, 255), (255, 1), (256, 1), (257, 3838), (4095, 1)):
        x, qr = _inputs(start, count)
        assert indexer(x, qr, None, cache=cache) is None
        assert cache.total_tokens == start + count
        assert cache.raw_token_count <= cache.raw_state_window
        assert cache.pool_capacity >= cache.logical_pool_count
    assert cache.total_tokens == 4096

    latent = SingleNoPELatentCache(reserve_tokens=0)
    for start, count in ((0, 255), (255, 1), (256, 1), (257, 3838), (4095, 1)):
        value = mx.zeros((1, 1, count, 4), dtype=mx.bfloat16)
        keys, values = latent.update_and_fetch(value, value)
        assert keys is values
        assert latent.size() == start + count
    assert latent.nbytes == latent.keys.nbytes


def test_trim_over_window_fails_before_either_cache_changes():
    indexer = _make_indexer(topk=2048)
    combined = make_compact_nope_dsa_cache(indexer, reserve_tokens=16)
    x, qr = _inputs(0, 32)
    indexer(x, qr, None, cache=combined[1])
    latent = mx.zeros((1, 1, 32, 4), dtype=mx.bfloat16)
    combined[0].update_and_fetch(latent, latent)
    before = (combined[0].size(), combined[1].size())

    with pytest.raises(ValueError, match="within"):
        combined.trim(17)

    assert (combined[0].size(), combined[1].size()) == before


def test_batch_greater_than_one_is_rejected():
    indexer = _make_indexer(topk=2048)
    cache = CompactIndexPoolCache(indexer)
    x, qr = _inputs(0, 1)
    with pytest.raises(ValueError, match="batch size 1"):
        cache.update(indexer, mx.broadcast_to(x, (2, 1, 8)), qr, None)

    latent = SingleNoPELatentCache()
    value = mx.zeros((2, 1, 1, 4), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="batch size 1"):
        latent.update_and_fetch(value, value)


def test_production_hot_paths_do_not_use_host_materialization():
    module = importlib.import_module("glm53_flash_mlx.nope_cache")
    source = inspect.getsource(module)
    assert "numpy" not in source
    for method in (
        CompactIndexPoolCache.update,
        CompactIndexPoolCache.trim,
        SingleNoPELatentCache.update_and_fetch,
    ):
        body = inspect.getsource(method)
        assert ".item()" not in body
        assert "mx.eval" not in body
