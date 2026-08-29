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
    "probe_persistent_all_dsa_session_frontier",
    Path(__file__).parents[1]
    / "scripts"
    / "probe_persistent_all_dsa_session_frontier.py",
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
    mx.random.seed(17)
    indexer = Glm5NextIndexer(config)
    indexer.bypass_short = False
    return indexer


def _inputs(start: int, tokens: int):
    positions = mx.arange(start, start + tokens, dtype=mx.float32)[:, None]
    x = mx.sin(positions * 0.03125 + mx.arange(8)[None] * 0.0625)[None]
    qr = mx.cos(positions * 0.046875 + mx.arange(4)[None] * 0.09375)[None]
    return x, qr


def _history_cache(indexer):
    from mlx_vlm.models.cache import KVCache

    cache = KVCache()
    x, qr = _inputs(0, 32)
    output = indexer(x, qr, mx.ones((1, 32), dtype=mx.bool_), cache=cache)
    mx.eval(output, cache.keys, cache.values, *cache._pool[:3])
    # Force the next token through KVCache's capacity-extension branch.
    cache.keys = mx.contiguous(cache.keys[..., :32, :])
    cache.values = mx.contiguous(cache.values[..., :32, :])
    mx.eval(cache.keys, cache.values)
    return cache


def test_persistent_resident_and_first_rebuild_match_for_16_steps():
    indexer = _make_indexer()
    resident = _history_cache(indexer)
    restored = _history_cache(indexer)
    restored._pool = None
    original_keys = resident.keys
    mods = set()

    for step in range(1, 17):
        x, qr = _inputs(31 + step, 1)
        resident_indices = indexer(x, qr, None, cache=resident)
        restored_indices = indexer(x, qr, None, cache=restored)
        mx.eval(resident_indices, restored_indices)

        assert mx.array_equal(resident_indices, restored_indices).item()
        assert resident._pool is not None
        assert restored._pool is not None
        assert resident._pool[3] == 32 + step
        assert restored._pool[3] == 32 + step
        mods.add((32 + step) % 4)

        if step == 1:
            assert resident.keys.shape[2] > 32
            assert mx.array_equal(original_keys, resident.keys[..., :32, :]).item()

    assert mods == {0, 1, 2, 3}


def test_pool_bytes_reports_actual_dtype_storage():
    pool = (
        mx.zeros((1, 3, 4), dtype=mx.bfloat16),
        mx.zeros((1, 3, 4), dtype=mx.int32),
        mx.zeros((1, 3), dtype=mx.bool_),
        12,
    )
    session = SimpleNamespace(indexer_cache=SimpleNamespace(_pool=pool))

    assert _MODULE._pool_bytes([session]) == {
        "bool": pool[2].nbytes,
        "bfloat16": pool[0].nbytes,
        "int32": pool[1].nbytes,
    }


def test_idle_snapshot_accepts_bfloat16_cache_state():
    latent = SimpleNamespace(
        keys=mx.zeros((1, 1, 8, 4), dtype=mx.bfloat16),
        values=mx.zeros((1, 1, 8, 4), dtype=mx.bfloat16),
        offset=7,
    )
    indexer = SimpleNamespace(
        keys=mx.zeros((1, 1, 8, 9), dtype=mx.bfloat16),
        values=mx.zeros((1, 1, 8, 0), dtype=mx.bfloat16),
        offset=7,
        _no_pad=True,
        _pool=None,
    )

    first = _MODULE._cache_snapshot(latent, indexer)
    second = _MODULE._cache_snapshot(latent, indexer)

    assert first == second


def test_zero_width_cache_values_have_no_dtype_state():
    before = mx.zeros((1, 1, 32, 0), dtype=mx.bfloat16)
    after = mx.zeros((1, 1, 32, 0), dtype=mx.float32)

    assert _MODULE._array_equal(before, after)


def test_expected_tail_sentinel_count_wraps_at_complete_pool():
    assert [
        _MODULE._expected_tail_sentinels(tokens, 4)
        for tokens in (2049, 2050, 2051, 2052)
    ] == [2, 1, 0, 3]
