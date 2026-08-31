import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import mlx.core as mx
except ImportError:
    mx = None


def _require_metal():
    if mx is None:
        pytest.skip("MLX/Metal is unavailable in this session")
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")


def _make_indexer(*, topk=32, kpool=4, bypass_short=False):
    _require_metal()
    from glm53_flash_mlx.patch import apply_runtime_patch

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
    mx.random.seed(7)
    indexer = Glm5NextIndexer(config)
    indexer.bypass_short = bypass_short
    return indexer


def _inputs(tokens, *, batch=1):
    x = mx.sin(mx.arange(batch * tokens * 8).reshape(batch, tokens, 8) * 0.03125)
    qr = mx.cos(mx.arange(batch * tokens * 4).reshape(batch, tokens, 4) * 0.0625)
    return x.astype(mx.float32), qr.astype(mx.float32)


def _array(value):
    mx.eval(value)
    return np.ascontiguousarray(np.asarray(value))


def _assert_sentinel_contract(indices, kv_len):
    values = _array(indices)
    assert not np.issubdtype(values.dtype, np.floating)
    valid = values != -1
    assert np.all(values[valid] >= 0)
    assert np.all(values[valid] < kv_len)
    assert not np.any((values < -1) | (values >= kv_len))
    return values


def test_sanitize_and_attention_gather_reject_positive_out_of_range_indices():
    _require_metal()
    from glm53_flash_mlx.indexpool import (
        build_prefill_indexpool_mask,
        prepare_decode_indexpool_gather,
        sanitize_indexpool_indices,
    )

    kv_len = 4
    raw = mx.array([-1, 0, kv_len - 1, kv_len, kv_len + 7], dtype=mx.int32)
    sanitized = sanitize_indexpool_indices(raw, kv_len)
    safe, valid = prepare_decode_indexpool_gather(raw, kv_len)
    sparse, sparse_valid = build_prefill_indexpool_mask(
        raw.reshape(1, 1, 1, -1), kv_len
    )
    mx.eval(sanitized, safe, valid, sparse, sparse_valid)

    assert _array(sanitized).tolist() == [-1, 0, 3, -1, -1]
    assert _array(safe).tolist() == [0, 0, 3, 0, 0]
    assert _array(valid).tolist() == [False, True, True, False, False]
    assert _array(sparse).reshape(-1).tolist() == [True, False, False, True]

    q = mx.array([[[[0.25, -0.5]]]], dtype=mx.float32)
    keys = mx.array([[[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-1.0, 0.0]]]])
    values = keys * 0.25
    gathered_k = keys[:, :, safe, :]
    gathered_v = values[:, :, safe, :]
    selection_mask = valid.reshape(1, 1, 1, -1)
    expected = mx.fast.scaled_dot_product_attention(
        q, gathered_k, gathered_v, scale=2**-0.5, mask=selection_mask
    )
    kv_valid = valid.reshape(1, 1, -1, 1)
    changed_k = mx.where(kv_valid, gathered_k, 10_000.0)
    changed_v = mx.where(kv_valid, gathered_v, -10_000.0)
    actual = mx.fast.scaled_dot_product_attention(
        q, changed_k, changed_v, scale=2**-0.5, mask=selection_mask
    )
    mx.eval(expected, actual)
    assert mx.array_equal(expected, actual).item()


@pytest.mark.parametrize("tail_count", [0, 1, 2, 3])
def test_expand_selected_pools_keeps_tail_and_validity_independent(tail_count):
    _require_metal()
    from glm53_flash_mlx.indexpool import expand_selected_pools

    kv_len = 8 + tail_count
    selected = mx.array([[[0, 1]]], dtype=mx.int32)
    pool_indices = mx.arange(8, dtype=mx.int64).reshape(1, 2, 4)
    selected_valid = mx.ones((1, 1, 2), dtype=mx.bool_)
    tail_positions = mx.arange(8, kv_len, dtype=mx.int64)[None]
    tail_valid = mx.ones((1, tail_count), dtype=mx.bool_)
    indices, valid = expand_selected_pools(
        selected,
        pool_indices,
        selected_valid,
        kv_len=kv_len,
        index_topk=16,
        index_kpool=4,
        tail_positions=tail_positions,
        tail_valid=tail_valid,
        always_select_tail=True,
    )
    values = _array(indices)
    validity = _array(valid)
    assert values.shape == validity.shape == (1, 1, 19)
    assert values[validity].tolist() == list(range(kv_len))
    assert np.all(values[~validity] == -1)
    assert int(validity.sum()) == kv_len


@pytest.mark.parametrize("valid_pools", [7, 512, 513])
def test_expand_selected_pools_handles_below_at_and_above_512(valid_pools):
    _require_metal()
    from glm53_flash_mlx.indexpool import expand_selected_pools

    pool_indices = mx.arange(valid_pools * 4, dtype=mx.int64).reshape(
        1, valid_pools, 4
    )
    selected_count = min(valid_pools, 512)
    selected = mx.arange(
        valid_pools - selected_count, valid_pools, dtype=mx.int32
    ).reshape(1, 1, selected_count)
    indices, valid = expand_selected_pools(
        selected,
        pool_indices,
        mx.ones(selected.shape, dtype=mx.bool_),
        kv_len=valid_pools * 4,
        index_topk=2048,
        index_kpool=4,
        tail_positions=mx.zeros((1, 0), dtype=mx.int64),
        tail_valid=mx.zeros((1, 0), dtype=mx.bool_),
        always_select_tail=True,
    )
    values = _array(indices)
    validity = _array(valid)
    assert values.shape == validity.shape == (1, 1, 2051)
    assert np.all(values[~validity] == -1)
    assert int(validity.sum()) == selected_count * 4
    if valid_pools == 512:
        assert values[validity].tolist() == list(range(2048))
    if valid_pools > 512:
        assert set(values[validity].tolist()) == set(range(4, 2052))


def test_expand_selected_pools_sentinelizes_token_and_pool_oob():
    _require_metal()
    from glm53_flash_mlx.indexpool import expand_selected_pools

    indices, valid = expand_selected_pools(
        mx.array([[[0, 9]]], dtype=mx.int32),
        mx.array([[[-1, 0, 7, 15]]], dtype=mx.int64),
        mx.ones((1, 1, 2), dtype=mx.bool_),
        kv_len=8,
        index_topk=8,
        index_kpool=4,
        tail_positions=mx.array([[8, 3]], dtype=mx.int64),
        tail_valid=mx.ones((1, 2), dtype=mx.bool_),
        always_select_tail=True,
    )
    values = _array(indices).reshape(-1)
    validity = _array(valid).reshape(-1)
    assert values.tolist() == [-1, 0, 7, -1, -1, -1, -1, -1, -1, 3, -1]
    assert validity.tolist() == [
        False, True, True, False, False, False, False, False, False, True, False
    ]


@pytest.mark.parametrize("kv_len", [4351, 4352, 16384, 65536, 131072, 262144])
def test_expand_selected_pools_width_is_bounded_at_long_context(kv_len):
    _require_metal()
    from glm53_flash_mlx.indexpool import expand_selected_pools

    complete = kv_len // 4
    selected_count = min(complete, 512)
    start = complete - selected_count
    selected = mx.arange(start, complete, dtype=mx.int32).reshape(
        1, 1, selected_count
    )
    pool_indices = mx.arange(complete * 4, dtype=mx.int64).reshape(1, complete, 4)
    tail_count = kv_len % 4
    tail = mx.arange(complete * 4, kv_len, dtype=mx.int64)[None]
    indices, valid = expand_selected_pools(
        selected,
        pool_indices,
        mx.ones(selected.shape, dtype=mx.bool_),
        kv_len=kv_len,
        index_topk=2048,
        index_kpool=4,
        tail_positions=tail,
        tail_valid=mx.ones((1, tail_count), dtype=mx.bool_),
        always_select_tail=True,
    )
    values = _assert_sentinel_contract(indices, kv_len)
    assert values.shape == _array(valid).shape == (1, 1, 2051)
    assert int(_array(valid).sum()) == selected_count * 4 + tail_count


def test_expand_selected_pools_repeats_byte_identically():
    _require_metal()
    from glm53_flash_mlx.indexpool import expand_selected_pools

    args = (
        mx.array([[[1, 0]]], dtype=mx.int32),
        mx.arange(8, dtype=mx.int64).reshape(1, 2, 4),
        mx.array([[[True, False]]]),
    )
    kwargs = {
        "kv_len": 9,
        "index_topk": 8,
        "index_kpool": 4,
        "tail_positions": mx.array([[8]], dtype=mx.int64),
        "tail_valid": mx.array([[True]]),
        "always_select_tail": True,
    }
    first = tuple(_array(value) for value in expand_selected_pools(*args, **kwargs))
    restored = tuple(
        _array(value)
        for value in expand_selected_pools(
            *(mx.array(_array(value)) for value in args), **kwargs
        )
    )
    assert all(np.array_equal(left, right) for left, right in zip(first, restored))


@pytest.mark.parametrize("tokens", [2047, 2048, 2049])
def test_compact_indexpool_preserves_2048_sparse_bypass_boundary(tokens):
    _require_metal()
    from glm53_flash_mlx.nope_cache import CompactIndexPoolCache

    indexer = _make_indexer(topk=2048, kpool=4, bypass_short=True)
    cache = CompactIndexPoolCache(indexer, capacity_tokens=2049)
    x, qr = _inputs(tokens)
    if tokens <= 2048:
        output = indexer(
            x, qr, mx.ones((1, tokens), dtype=mx.bool_), cache=cache
        )
        assert output is None
    else:
        assert indexer(
            x[:, :2048],
            qr[:, :2048],
            mx.ones((1, 2048), dtype=mx.bool_),
            cache=cache,
        ) is None
        output = indexer(
            x[:, 2048:], qr[:, 2048:], mx.ones((1, 1), dtype=mx.bool_), cache=cache
        )
        values = _assert_sentinel_contract(output, tokens)
        assert values.shape == (1, 1, 1, 2051)


@pytest.mark.parametrize("tokens", [31, 32, 33])
def test_indexpool_topk_and_partial_pool_boundaries(tokens):
    indexer = _make_indexer(topk=32, kpool=4, bypass_short=True)
    x, qr = _inputs(tokens)
    mask = mx.ones((1, tokens), dtype=mx.bool_)
    if tokens <= 32:
        assert indexer(x, qr, mask) is None
    else:
        assert indexer(x, qr, mask) is not None

    indexer.bypass_short = False
    output = indexer(x, qr, mask)
    values = _assert_sentinel_contract(output, tokens)
    assert values.shape == (1, 1, tokens, 35)

    final = values[0, 0, -1]
    valid_final = final[final >= 0]
    expected_valid = min(tokens // 4, 8) * 4 + tokens % 4
    assert valid_final.size == expected_valid
    assert set(valid_final.tolist()) == set(range(tokens))
    assert np.all(final[valid_final.size :] == -1) or np.count_nonzero(final == -1) == (
        35 - expected_valid
    )
    assert 0 in valid_final
    assert tokens - 1 in valid_final


def test_indexpool_zero_valid_and_left_padded_rows_are_sentinel_clean():
    indexer = _make_indexer(topk=32, kpool=4)
    tokens = 33
    x, qr = _inputs(tokens, batch=2)
    mask = mx.array(
        [[False] * tokens, [False] * 5 + [True] * (tokens - 5)], dtype=mx.bool_
    )
    output = indexer(x, qr, mask)
    values = _assert_sentinel_contract(output, tokens)

    assert np.all(values[0] == -1)
    assert np.all(values[1, :, :5] == -1)
    valid = values[1][values[1] >= 0]
    assert valid.size > 0
    assert valid.min() >= 5
    assert valid.max() == tokens - 1


def test_indexpool_batch_cache_uses_physical_kv_columns_not_row_offsets():
    _require_metal()
    from mlx_vlm.models.cache import BatchKVCache

    indexer = _make_indexer(topk=32, kpool=4)
    tokens = 5
    x, qr = _inputs(tokens, batch=2)
    mask = mx.array(
        [[False, False, True, True, True], [True] * tokens], dtype=mx.bool_
    )
    cache = BatchKVCache(left_padding=[2, 0])

    output = indexer(x, qr, mask, cache=cache)
    values = _assert_sentinel_contract(output, tokens)

    assert values.shape == (2, 1, tokens, 35)
    assert np.all(values[0, :, :2] == -1)
    assert set(values[0][values[0] >= 0].tolist()) == {2, 3, 4}
    assert set(values[1][values[1] >= 0].tolist()) == set(range(tokens))


@pytest.mark.parametrize("tokens", [511, 512, 513])
def test_indexpool_query_chunk_boundaries_repeat_byte_identically(tokens):
    indexer = _make_indexer(topk=32, kpool=4)
    x, qr = _inputs(tokens)
    mask = mx.ones((1, tokens), dtype=mx.bool_)

    first = _assert_sentinel_contract(indexer(x, qr, mask), tokens)
    second = _assert_sentinel_contract(indexer(x, qr, mask), tokens)

    assert first.shape == (1, 1, tokens, 35)
    assert hashlib.sha256(first.tobytes()).digest() == hashlib.sha256(
        second.tobytes()
    ).digest()


def test_indexpool_one_shot_chunked_and_incremental_valid_sets_match():
    _require_metal()
    from mlx_vlm.models.cache import KVCache

    tokens = 33
    indexer = _make_indexer(topk=32, kpool=4)
    x, qr = _inputs(tokens)
    mask = mx.ones((1, tokens), dtype=mx.bool_)
    one_shot = _assert_sentinel_contract(indexer(x, qr, mask), tokens)

    def run_chunked():
        cache = KVCache()
        parts = []
        for start, end in ((0, 17), (17, tokens)):
            part = indexer(
                x[:, start:end],
                qr[:, start:end],
                mask[:, start:end],
                cache=cache,
            )
            parts.append(_assert_sentinel_contract(part, end))
        return np.concatenate(parts, axis=2)

    def run_incremental():
        cache = KVCache()
        parts = []
        for position in range(tokens):
            part = indexer(
                x[:, position : position + 1],
                qr[:, position : position + 1],
                mask[:, position : position + 1],
                cache=cache,
            )
            parts.append(_assert_sentinel_contract(part, position + 1))
        return np.concatenate(parts, axis=2)

    chunked = run_chunked()
    incremental = run_incremental()

    for row in range(tokens):
        expected = set(one_shot[0, 0, row][one_shot[0, 0, row] >= 0].tolist())
        chunked_set = set(chunked[0, 0, row][chunked[0, 0, row] >= 0].tolist())
        incremental_set = set(
            incremental[0, 0, row][incremental[0, 0, row] >= 0].tolist()
        )
        assert expected == chunked_set == incremental_set

    assert np.array_equal(chunked, run_chunked())
    assert np.array_equal(incremental, run_incremental())
    assert set(incremental[0, 0, 2][incremental[0, 0, 2] >= 0]) == {0, 1, 2}
    assert set(incremental[0, 0, 3][incremental[0, 0, 3] >= 0]) == {0, 1, 2, 3}
