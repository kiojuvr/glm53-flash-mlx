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
    "probe_incremental_indexpool_update_copies",
    Path(__file__).parents[1]
    / "scripts"
    / "probe_incremental_indexpool_update_copies.py",
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
    mx.random.seed(29)
    indexer = Glm5NextIndexer(config)
    indexer.bypass_short = False
    return indexer


def _inputs(start: int, tokens: int):
    positions = mx.arange(start, start + tokens, dtype=mx.float32)[:, None]
    x = mx.sin(positions * 0.03125 + mx.arange(8)[None] * 0.0625)[None]
    qr = mx.cos(positions * 0.046875 + mx.arange(4)[None] * 0.09375)[None]
    return x, qr


def _history(indexer):
    from mlx_vlm.models.cache import KVCache

    cache = KVCache()
    x, qr = _inputs(0, 32)
    indexer(x, qr, mx.ones((1, 32), dtype=mx.bool_), cache=cache)
    mx.eval(cache.keys, *cache._pool[:3])
    return cache


def _session(indexer, arm):
    cache = _history(indexer)
    pool = _MODULE._make_pool(arm, cache._pool[:3], 32, 48)
    return _MODULE._Session(
        layer_id=3,
        attention=None,
        capture=SimpleNamespace(delegate=indexer),
        latent_cache=None,
        indexer_cache=cache,
        pool=pool,
        arm=arm,
    )


def _manual_step(session, x, qr):
    projection, _ = _MODULE._current_projection(session, x)
    packed, append_traffic = _MODULE._append_token(session, projection)
    suffix, _ = _MODULE._recompute_partial(session, packed)
    _, carry_traffic, sync = _MODULE._carry_prefix(session, suffix)
    _MODULE._eval(sync)
    _MODULE._publish_pool(session)
    scored = _MODULE._score_segments(session, x, qr)
    selected = _MODULE._select_pools(session, scored)
    indices = _MODULE._expand_selection(session, selected)
    _MODULE._eval(indices)
    return indices, append_traffic, carry_traffic


def _contiguous_pool(pool):
    segments = pool.segments()
    return tuple(mx.concatenate([row[i] for row in segments], axis=1) for i in range(3))


def test_incremental_pool_arms_match_reference_for_all_tail_shapes():
    indexer = _make_indexer()
    oracle_cache = _history(indexer)
    sessions = {
        arm: _session(indexer, arm)
        for arm in _MODULE.ARMS
    }
    mods = set()

    for step in range(1, 17):
        x, qr = _inputs(31 + step, 1)
        expected = indexer(x, qr, None, cache=oracle_cache)
        actual = {
            arm: _manual_step(session, x, qr)
            for arm, session in sessions.items()
        }
        mx.eval(expected, *[row[0] for row in actual.values()])
        mods.add((32 + step) % 4)

        for arm, (indices, append_traffic, carry_traffic) in actual.items():
            assert mx.array_equal(indices, expected).item(), (step, arm)
            assert append_traffic["copy_bytes"] > 0
            assert sessions[arm].pool.total_tokens == 32 + step
            assert sessions[arm].pool.logical_count == (32 + step + 3) // 4

            pool = _contiguous_pool(sessions[arm].pool)
            for observed, reference in zip(pool, oracle_cache._pool[:3], strict=True):
                assert mx.array_equal(observed, reference).item(), (step, arm)

            if arm == "reference_concat":
                assert carry_traffic["copy_bytes"] > 0
                assert sum(carry_traffic["pool_copy_bytes"].values()) > 0
            elif arm == "preallocated_pool_row":
                assert carry_traffic["copy_bytes"] > 0
                assert sum(carry_traffic["pool_copy_bytes"].values()) > 0
            else:
                assert carry_traffic["copy_bytes"] == 0
                assert carry_traffic["pool_copy_bytes"] == {
                    "keys": 0,
                    "indices": 0,
                    "valid": 0,
                }

    assert mods == {0, 1, 2, 3}


def test_expand_selection_has_runtime_indexer_shape():
    indexer = _make_indexer()
    session = _session(indexer, "preallocated_pool_row")
    x, qr = _inputs(32, 1)
    indices, _, _ = _manual_step(session, x, qr)

    assert indices.shape == (1, 1, 1, indexer.index_topk + 3)
    assert indices.dtype == mx.int32
