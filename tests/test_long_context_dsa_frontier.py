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
    "probe_long_context_dsa_decode_frontier",
    Path(__file__).parents[1]
    / "scripts"
    / "probe_long_context_dsa_decode_frontier.py",
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
    mx.random.seed(11)
    indexer = Glm5NextIndexer(config)
    indexer.bypass_short = False
    return indexer


def _inputs(tokens):
    x = mx.sin(mx.arange(tokens * 8).reshape(1, tokens, 8) * 0.03125)
    qr = mx.cos(mx.arange(tokens * 4).reshape(1, tokens, 4) * 0.0625)
    return x.astype(mx.float32), qr.astype(mx.float32)


def _cache_at_history(indexer, x, qr, *, rebuild):
    from mlx_vlm.models.cache import KVCache

    cache = KVCache()
    mask = mx.ones((1, x.shape[1]), dtype=mx.bool_)
    previous = indexer(x, qr, mask, cache=cache)
    mx.eval(previous, cache.keys, *cache._pool[:3])
    if rebuild:
        cache._pool = None
    return cache


@pytest.mark.parametrize("rebuild", [False, True])
def test_manual_decode_phases_match_fixed_indexer(rebuild):
    indexer = _make_indexer()
    x, qr = _inputs(33)
    manual_cache = _cache_at_history(
        indexer, x[:, :32], qr[:, :32], rebuild=rebuild
    )
    reference_cache = _cache_at_history(
        indexer, x[:, :32], qr[:, :32], rebuild=rebuild
    )
    current_mask = mx.ones((1, 1), dtype=mx.bool_)

    actual = _MODULE._manual_indexer(
        indexer, x[:, 32:], qr[:, 32:], manual_cache
    )
    expected = indexer(
        x[:, 32:], qr[:, 32:], current_mask, cache=reference_cache
    )
    mx.eval(actual, expected)

    assert mx.array_equal(actual, expected).item()


def test_deterministic_cache_rows_repeat_byte_identically():
    first = _MODULE._deterministic_rows(65, 17, 0.625, mx.bfloat16)
    second = _MODULE._deterministic_rows(65, 17, 0.625, mx.bfloat16)
    mx.eval(first, second)

    assert mx.array_equal(first, second).item()
