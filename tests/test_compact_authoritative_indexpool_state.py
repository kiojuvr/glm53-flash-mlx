import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import mlx.core as mx
except ImportError:
    pytest.skip("MLX/Metal is unavailable", allow_module_level=True)

_SPEC = importlib.util.spec_from_file_location(
    "probe_compact_authoritative_indexpool_state",
    Path(__file__).parents[1]
    / "scripts"
    / "probe_compact_authoritative_indexpool_state.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _make_indexer():
    from glm53_flash_mlx.patch import apply_runtime_patch

    apply_runtime_patch()
    from mlx_vlm.models.glm5_next.language import Glm5NextIndexer

    config = SimpleNamespace(
        hidden_size=8,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=32,
        index_kpool=4,
        index_kpool_always_select_tail=True,
        q_lora_rank=4,
    )
    mx.random.seed(41)
    indexer = Glm5NextIndexer(config)
    indexer.bypass_short = False
    return indexer


def _inputs(start: int, tokens: int):
    positions = mx.arange(start, start + tokens, dtype=mx.float32)[:, None]
    x = mx.sin(positions * 0.03125 + mx.arange(8)[None] * 0.0625)[None]
    qr = mx.cos(positions * 0.046875 + mx.arange(4)[None] * 0.09375)[None]
    return x, qr


def _oracle_cache(indexer, history=32):
    from mlx_vlm.models.cache import KVCache

    capacity = 256
    keys, gates, valid = _MODULE._history_rows(indexer, 3, capacity)
    valid = mx.arange(capacity)[None] < history
    packed = mx.concatenate(
        [keys, gates, valid.astype(keys.dtype)[..., None]], axis=-1
    )
    logical = packed[:, :history]
    logical_keys, logical_gates, valid_channel = mx.split(
        logical, [indexer.head_dim, 2 * indexer.head_dim], axis=-1
    )
    cache = KVCache()
    cache.keys = packed[:, None]
    cache.values = mx.zeros((1, 1, capacity, 0), dtype=mx.bfloat16)
    cache.offset = history
    cache._no_pad = True
    cache._pool = (
        *indexer._pooled_states(
            logical_keys, logical_gates, valid_channel[..., 0] > 0
        ),
        history,
    )
    mx.eval(packed, *cache._pool[:3])
    return cache


def _step(indexer, cache, state, step):
    x, qr = _inputs(31 + step, 1)
    pooled = _MODULE._pool_update(indexer, x, qr, cache)
    pooled["x"] = x
    expected = _MODULE._expand_phase(
        indexer,
        pooled,
        _MODULE._selection_phase(indexer, _MODULE._score_phase(indexer, pooled)),
    )

    _, append_copy, carry_copy = _MODULE._compact_pool_update(
        state, indexer, x
    )
    selected = _MODULE._selection_phase(
        indexer, _MODULE._compact_score(indexer, state, x, qr)
    )
    actual = _MODULE._compact_expand(indexer, state, selected)
    mx.eval(expected, actual, *cache._pool[:3], *state.logical_pool())
    return expected, actual, append_copy, carry_copy


def _assert_pool_equal(cache, state):
    for expected, actual in zip(cache._pool[:3], state.logical_pool(), strict=True):
        assert mx.array_equal(expected, actual).item()


def test_compact_state_matches_full_history_for_16_steps():
    indexer = _make_indexer()
    cache = _oracle_cache(indexer)
    state = _MODULE._build_compact_state(indexer, 3, 32)
    append_copies = set()
    mods = set()

    for step in range(1, 17):
        expected, actual, append_copy, carry_copy = _step(
            indexer, cache, state, step
        )
        assert mx.array_equal(expected, actual).item()
        _assert_pool_equal(cache, state)
        assert state.raw_token_count <= state.rollback_window
        assert state.active_tail_count == (32 + step) % 4
        assert carry_copy > 0
        append_copies.add(append_copy)
        mods.add((32 + step) % 4)

    assert mods == {0, 1, 2, 3}
    assert len(append_copies) == 1
    assert not hasattr(state, "packed_token_history")


@pytest.mark.parametrize("tokens", _MODULE.ROLLBACK_CASES)
def test_compact_trim_replay_matches_full_history(tokens):
    indexer = _make_indexer()
    cache = _oracle_cache(indexer)
    state = _MODULE._build_compact_state(indexer, 3, 32)
    first = []
    for step in range(1, tokens + 1):
        first.append(_step(indexer, cache, state, step)[:2])

    cache.trim(tokens)
    cache._pool = None
    session = _MODULE._CompactSession(
        3,
        SimpleNamespace(indexer=indexer),
        SimpleNamespace(offset=32 + tokens),
        state,
    )
    _MODULE._compact_trim(session, tokens)
    assert state.total_tokens == 32
    assert state.active_tail_count == 0

    for step, before in enumerate(first, 1):
        expected, actual, _, _ = _step(indexer, cache, state, step)
        assert mx.array_equal(expected, before[0]).item()
        assert mx.array_equal(actual, before[1]).item()
        assert mx.array_equal(expected, actual).item()
        _assert_pool_equal(cache, state)


def test_compact_state_reduction_exceeds_eighty_percent_at_256k():
    indexer = _make_indexer()
    state = _MODULE._build_compact_state(indexer, 3, 262143)
    capacity = _MODULE._capacity(262143 + 16)
    pool_capacity = (262143 + 16 + 3) // 4
    # Official layout: BF16 key/gate and pool key, int64 four-token indices.
    oracle = capacity * (2 * 128 + 1) * 2
    compact = pool_capacity * (128 * 2 + 4 * 8 + 1)
    compact += _MODULE.ROLLBACK_WINDOW * (2 * 128 * 2 + 1 + 8)

    assert 1.0 - compact / oracle >= 0.8
    assert state.raw_token_count == 16
