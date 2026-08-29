import importlib.util
import sys
from pathlib import Path

import pytest

try:
    import mlx.core as mx
except ImportError:
    pytest.skip("MLX/Metal is unavailable", allow_module_level=True)

_SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location(
    "probe_single_buffer_nope_latent_cache_frontier",
    _SCRIPTS / "probe_single_buffer_nope_latent_cache_frontier.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _row(step: int, width: int = 8):
    return mx.sin(mx.arange(width)[None, None, None] * 0.125 + step).astype(
        mx.bfloat16
    )


def test_single_latent_cache_matches_dual_kvcache_across_capacity_extension():
    from mlx_vlm.models.cache import KVCache

    history = mx.sin(mx.arange(256 * 8).reshape(1, 1, 256, 8) * 0.01).astype(
        mx.bfloat16
    )
    dual = KVCache()
    dual.keys = history
    dual.values = history
    dual.offset = 256
    single = _MODULE.ProbeNoPELatentCache(history, 256)

    for step in range(1, 17):
        current = _row(step)
        dual_keys, dual_values = dual.update_and_fetch(current, current)
        single_keys, single_values = single.update_and_fetch(current, current)
        mx.eval(dual_keys, dual_values, single_keys, single_values)

        assert mx.array_equal(dual_keys, single_keys).item()
        assert mx.array_equal(dual_values, single_values).item()
        assert single.keys is single.values

    assert single.extension_count == 1
    assert single.total_copy_bytes == history.nbytes
    assert single.nbytes * 2 == dual.keys.nbytes + dual.values.nbytes


def test_preallocated_single_latent_avoids_extension_and_copy():
    latent = mx.zeros((1, 1, 512, 8), dtype=mx.bfloat16)
    cache = _MODULE.ProbeNoPELatentCache(latent, 256)

    for step in range(1, 17):
        cache.update_and_fetch(_row(step), _row(step))
    mx.eval(cache.keys)

    assert cache.offset == 272
    assert cache.extension_count == 0
    assert cache.total_copy_bytes == 0
    assert cache.keys is cache.values


def test_single_latent_dependency_assignment_preserves_alias_and_bytes():
    latent = mx.ones((1, 1, 8, 4), dtype=mx.bfloat16)
    cache = _MODULE.ProbeNoPELatentCache(latent, 8)
    dependency = mx.zeros((1,), dtype=mx.float32)
    before = mx.array(cache.keys)

    cache.keys = mx.depends(cache.keys, (dependency,))
    mx.eval(cache.keys, before)

    assert cache.keys is cache.values
    assert mx.array_equal(cache.keys, before).item()
