from __future__ import annotations

import hashlib

import numpy as np
import pytest

try:
    import mlx.core as mx
    if not mx.metal.is_available():
        raise ImportError("Metal is unavailable")
except ImportError:
    pytest.skip("MLX/Metal is unavailable", allow_module_level=True)

from glm53_flash_mlx.kda_state import (
    KDAStateIndexError,
    KDAStateTypeError,
    materialization_sources,
    normalize_kda_state_index,
    restore_indexed_state,
)
from glm53_flash_mlx.patch import apply_runtime_patch


def _hash(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        if value is None:
            digest.update(b"none")
            continue
        original_dtype = str(value.dtype)
        host_value = value.astype(mx.float32) if value.dtype == mx.bfloat16 else value
        mx.eval(host_value)
        host = np.ascontiguousarray(np.asarray(host_value))
        digest.update(original_dtype.encode())
        digest.update(repr(host.shape).encode())
        digest.update(host.tobytes())
    return digest.hexdigest()


def test_real_arrays_cache_guards_mlx_scalar_and_strided_state():
    apply_runtime_patch()
    from mlx_vlm.models.cache import ArraysCache

    cache = ArraysCache(size=2)
    wide = mx.arange(64, dtype=mx.float32).reshape(4, 16)
    conv = wide[:, 1::2].astype(mx.bfloat16)
    recurrent = wide[::2, :].astype(mx.float32)
    cache[0] = conv
    cache[1] = recurrent
    before = _hash(cache.state)

    assert normalize_kda_state_index(mx.array(0, dtype=mx.int32), capacity=2) == 0
    assert normalize_kda_state_index(mx.array(1, dtype=mx.int64), capacity=2) == 1
    for invalid in (
        mx.array(-2, dtype=mx.int32),
        mx.array(2, dtype=mx.int64),
    ):
        with pytest.raises(KDAStateIndexError):
            _ = cache[invalid]
        assert _hash(cache.state) == before
    with pytest.raises(KDAStateTypeError):
        _ = cache[mx.array(1.0, dtype=mx.float32)]
    assert _hash(cache.state) == before

    assert cache[mx.array(-1, dtype=mx.int32)] is None
    cache[mx.array(-1, dtype=mx.int64)] = mx.zeros_like(conv)
    assert _hash(cache.state) == before
    assert _hash(materialization_sources(cache, (0, 1))) == before


def test_ram_apc_clone_and_restore_remain_guarded_and_atomic():
    apply_runtime_patch()
    from mlx_vlm.apc_adapters import clone_cache_entry
    from mlx_vlm.models.cache import ArraysCache

    source = ArraysCache(size=2)
    source[0] = mx.arange(24, dtype=mx.float32).reshape(2, 3, 4)
    source[1] = mx.arange(32, dtype=mx.float32).reshape(2, 4, 4)
    targets = []
    cloned = clone_cache_entry(
        source,
        min_capacity_tokens=0,
        eval_targets=targets,
    )
    mx.eval(targets)
    before_source = _hash(source.state)
    before_clone = _hash(cloned.state)
    assert before_source == before_clone

    with pytest.raises(KDAStateIndexError):
        restore_indexed_state(
            cloned,
            ((0, mx.zeros_like(cloned[0])), (2, mx.zeros_like(cloned[1]))),
        )
    assert _hash(source.state) == before_source
    assert _hash(cloned.state) == before_clone

    with pytest.raises(KDAStateIndexError):
        cloned.state = [mx.zeros_like(cloned[0])]
    assert _hash(cloned.state) == before_clone
