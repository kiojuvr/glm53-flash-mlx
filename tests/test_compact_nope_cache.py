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


def _make_attention(*, topk=32, kpool=4):
    apply_runtime_patch()
    from mlx_vlm.models.glm5_next.language import Glm5NextSparseAttention

    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        q_lora_rank=4,
        qk_rope_head_dim=0,
        kv_lora_rank=4,
        v_head_dim=4,
        qk_nope_head_dim=4,
        mla_use_nope=True,
        attention_bias=False,
        rms_norm_eps=1e-6,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=topk,
        index_kpool=kpool,
        index_kpool_always_select_tail=True,
    )
    mx.random.seed(79)
    attention = Glm5NextSparseAttention(config)
    attention.set_dtype(mx.bfloat16)
    return attention


def _inputs(start: int, tokens: int):
    positions = mx.arange(start, start + tokens, dtype=mx.float32)[:, None]
    x = mx.sin(positions * 0.03125 + mx.arange(8)[None] * 0.0625)[None]
    qr = mx.cos(positions * 0.046875 + mx.arange(4)[None] * 0.09375)[None]
    return x.astype(mx.bfloat16), qr.astype(mx.bfloat16)


def _assert_equal(left, right):
    assert left.shape == right.shape
    assert mx.array_equal(left, right).item()


def _copy_tree(value):
    if isinstance(value, tuple):
        return tuple(_copy_tree(item) for item in value)
    if isinstance(value, list):
        return [_copy_tree(item) for item in value]
    if value is None or isinstance(value, str):
        return value
    copied = mx.array(value)
    mx.eval(copied)
    return copied


def _assert_tree_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_tree_equal(left_item, right_item)
    elif left is None or isinstance(left, str):
        assert left == right
    else:
        _assert_equal(left, right)


def _append_combined(indexer, cache, start: int, tokens: int):
    x, qr = _inputs(start, tokens)
    selected = indexer(x, qr, None, cache=cache[1])
    latent = mx.sin(
        mx.arange(start, start + tokens, dtype=mx.float32)[:, None] * 0.015625
        + mx.arange(4, dtype=mx.float32)[None] * 0.125
    ).astype(mx.bfloat16)
    latent = latent.reshape(1, 1, tokens, 4)
    cache[0].update_and_fetch(latent, latent)
    return selected


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
    cache = CompactIndexPoolCache(indexer, capacity_tokens=4352)
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
    cache = CompactIndexPoolCache(indexer, capacity_tokens=16)
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
    compact = CompactIndexPoolCache(indexer, capacity_tokens=64)
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
    original = make_compact_nope_dsa_cache(indexer, capacity_tokens=64)
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
    cache = CompactIndexPoolCache(indexer, capacity_tokens=0)
    for start, count in ((0, 255), (255, 1), (256, 1), (257, 3838), (4095, 1)):
        x, qr = _inputs(start, count)
        assert indexer(x, qr, None, cache=cache) is None
        assert cache.total_tokens == start + count
        assert cache.raw_token_count <= cache.raw_state_window
        assert cache.pool_capacity >= cache.logical_pool_count
    assert cache.total_tokens == 4096

    latent = SingleNoPELatentCache(capacity_tokens=0)
    for start, count in ((0, 255), (255, 1), (256, 1), (257, 3838), (4095, 1)):
        value = mx.zeros((1, 1, count, 4), dtype=mx.bfloat16)
        keys, values = latent.update_and_fetch(value, value)
        assert keys is values
        assert latent.size() == start + count
    assert latent.nbytes == latent.keys.nbytes


def test_absolute_capacity_is_fixed_through_4352_and_grows_once_at_4353():
    indexer = _make_indexer(topk=8192)
    combined = make_compact_nope_dsa_cache(indexer, capacity_tokens=4352)

    _append_combined(indexer, combined, 0, 1)
    latent, pool = combined
    assert latent.capacity_tokens == pool.capacity_tokens == 4352
    assert latent.physical_capacity_tokens == 4352
    assert pool.physical_capacity_rows == 1088
    initial_latent = latent.keys
    initial_pool = pool.pool_keys

    for start, count in ((1, 15), (16, 112), (128, 128), (256, 4096)):
        _append_combined(indexer, combined, start, count)
        assert latent.physical_capacity_tokens == 4352
        assert pool.physical_capacity_rows == 1088
        assert latent.keys is initial_latent
        assert pool.pool_keys is initial_pool

    _append_combined(indexer, combined, 4352, 1)
    assert latent.physical_capacity_tokens == 4608
    assert pool.physical_capacity_rows == 1152
    grown_latent = latent.keys
    grown_pool = pool.pool_keys
    _append_combined(indexer, combined, 4353, 1)
    assert latent.keys is grown_latent
    assert pool.pool_keys is grown_pool


@pytest.mark.parametrize("tail_mod", range(4))
def test_absolute_capacity_pool_tail_contract_matches_latent(tail_mod):
    indexer = _make_indexer(topk=2048)
    combined = make_compact_nope_dsa_cache(indexer, capacity_tokens=65)
    total = 64 + tail_mod
    _append_combined(indexer, combined, 0, total)
    latent, pool = combined
    assert latent.capacity_tokens == pool.capacity_tokens == 65
    assert latent.physical_capacity_tokens == 256
    assert pool.physical_capacity_rows == 64
    assert pool.logical_pool_count == (total + 3) // 4
    assert pool.active_tail_count == total % 4


def test_reserve_until_is_absolute_and_allocates_only_once():
    indexer = _make_indexer(topk=8192)
    combined = make_compact_nope_dsa_cache(indexer, capacity_tokens=4352)
    _append_combined(indexer, combined, 0, 32)
    latent, pool = combined

    latent.reserve_until(8192)
    pool.reserve_until(8192)
    assert latent.capacity_tokens == pool.capacity_tokens == 8192
    assert latent.physical_capacity_tokens == 8192
    assert pool.physical_capacity_rows == 2048
    latent_buffer = latent.keys
    pool_buffer = pool.pool_keys

    latent.reserve_until(8192)
    pool.reserve_until(8192)
    _append_combined(indexer, combined, 32, 4096)
    assert latent.keys is latent_buffer
    assert pool.pool_keys is pool_buffer


def test_clone_trim_replay_preserves_absolute_physical_capacity():
    from mlx_vlm.apc_adapters import clone_cache_entry

    indexer = _make_indexer(topk=2048)
    original = make_compact_nope_dsa_cache(indexer, capacity_tokens=4352)
    _append_combined(indexer, original, 0, 65)
    eval_targets = []
    restored = clone_cache_entry(
        original, min_capacity_tokens=81, eval_targets=eval_targets
    )
    mx.eval(*eval_targets)

    assert restored[0].capacity_tokens == restored[1].capacity_tokens == 4352
    assert restored[0].physical_capacity_tokens == 4352
    assert restored[1].physical_capacity_rows == 1088
    latent_buffer = restored[0].keys
    pool_buffer = restored[1].pool_keys
    restored.trim(1)
    _append_combined(indexer, restored, 64, 1)
    assert restored[0].keys is latent_buffer
    assert restored[1].pool_keys is pool_buffer
    _assert_tree_equal(
        (original.state, original.meta_state),
        (restored.state, restored.meta_state),
    )


def test_trim_over_window_fails_before_either_cache_changes():
    indexer = _make_indexer(topk=2048)
    combined = make_compact_nope_dsa_cache(indexer, capacity_tokens=16)
    x, qr = _inputs(0, 32)
    indexer(x, qr, None, cache=combined[1])
    latent = mx.zeros((1, 1, 32, 4), dtype=mx.bfloat16)
    combined[0].update_and_fetch(latent, latent)
    before = _copy_tree((combined.state, combined.meta_state))

    with pytest.raises(ValueError, match="within"):
        combined.trim(17)

    _assert_tree_equal(before, (combined.state, combined.meta_state))


def test_cache_list_preflight_keeps_latent_atomic_when_pool_rejects():
    indexer = _make_indexer(topk=2048)
    combined = make_compact_nope_dsa_cache(indexer, capacity_tokens=16)
    _append_combined(indexer, combined, 0, 32)
    pool = combined[1]
    pool.raw_keys = pool.raw_keys[:, -1:]
    pool.raw_gates = pool.raw_gates[:, -1:]
    pool.raw_valid = pool.raw_valid[:, -1:]
    pool.raw_positions = pool.raw_positions[:, -1:]
    before = _copy_tree((combined.state, combined.meta_state))

    with pytest.raises(ValueError, match="cannot reconstruct"):
        combined.trim(2)

    _assert_tree_equal(before, (combined.state, combined.meta_state))


@pytest.mark.parametrize("target_mod", range(4))
@pytest.mark.parametrize("tokens", range(1, 17))
def test_apc_clone_trim_replay_is_self_contained(target_mod, tokens):
    from mlx_vlm.apc_adapters import clone_cache_entry

    indexer = _make_indexer(topk=32)
    target = 48 + target_mod
    total = target + tokens
    original = make_compact_nope_dsa_cache(indexer, capacity_tokens=96)
    _append_combined(indexer, original, 0, 32)
    for position in range(32, total):
        _append_combined(indexer, original, position, 1)

    eval_targets = []
    restored = clone_cache_entry(
        original, min_capacity_tokens=total + 16, eval_targets=eval_targets
    )
    mx.eval(*eval_targets)
    assert not hasattr(restored[1], "_indexer")
    restored.trim(tokens)

    oracle = make_compact_nope_dsa_cache(indexer, capacity_tokens=96)
    _append_combined(indexer, oracle, 0, 32)
    for position in range(32, target):
        _append_combined(indexer, oracle, position, 1)
    _assert_tree_equal(oracle[0].state, restored[0].state)
    _assert_tree_equal(oracle[1].logical_pool(), restored[1].logical_pool())
    assert restored[1].total_tokens == target
    assert restored[1].raw_token_count <= restored[1].raw_state_window

    for position in range(target, total):
        _append_combined(indexer, restored, position, 1)
    _assert_tree_equal(
        (original.state, original.meta_state),
        (restored.state, restored.meta_state),
    )

    next_original = _append_combined(indexer, original, total, 1)
    next_restored = _append_combined(indexer, restored, total, 1)
    _assert_equal(next_original, next_restored)


@pytest.mark.parametrize("target_mod", range(4))
@pytest.mark.parametrize("tokens", range(1, 17))
def test_apc_clone_trim_replay_attention_output_hash_matches(target_mod, tokens):
    from mlx_vlm.apc_adapters import clone_cache_entry

    attention = _make_attention(topk=32)
    target = 48 + target_mod
    total = target + tokens
    original = make_compact_nope_dsa_cache(attention.indexer, capacity_tokens=96)
    x, _ = _inputs(0, 32)
    attention(x, cache=original)
    replay_outputs = {}
    for position in range(32, total):
        x, _ = _inputs(position, 1)
        replay_outputs[position] = attention(x, cache=original)

    eval_targets = []
    restored = clone_cache_entry(
        original, min_capacity_tokens=total + 16, eval_targets=eval_targets
    )
    mx.eval(*eval_targets)
    restored.trim(tokens)
    for position in range(target, total):
        x, _ = _inputs(position, 1)
        actual = attention(x, cache=restored)
        _assert_equal(replay_outputs[position], actual)

    x, _ = _inputs(total, 1)
    _assert_equal(attention(x, cache=original), attention(x, cache=restored))


def test_long_sparse_prefill_rejection_is_atomic_and_precedes_latent_update():
    from mlx_vlm.models.glm5_next.language import Glm5NextSparseAttention

    attention = _make_attention(topk=32)
    cache = make_compact_nope_dsa_cache(attention.indexer, capacity_tokens=16)
    x, _ = _inputs(0, 32)
    attention(x, cache=cache)
    before = _copy_tree((cache.state, cache.meta_state))
    x, _ = _inputs(32, 2)

    with pytest.raises(ValueError, match="long sparse prefill"):
        attention(x, cache=cache)

    _assert_tree_equal(before, (cache.state, cache.meta_state))
    sparse_source = inspect.getsource(Glm5NextSparseAttention.__call__)
    assert sparse_source.index("validate_update") < sparse_source.index(
        "update_and_fetch"
    )


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
