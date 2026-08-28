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
