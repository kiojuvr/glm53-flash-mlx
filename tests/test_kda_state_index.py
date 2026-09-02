from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pytest

from glm53_flash_mlx.kda_state import (
    KDA_ROLLBACK_WINDOW,
    KDA_STATE_INDEX_CONTRACT,
    KDA_STATE_SENTINEL,
    KDAStateIndexError,
    KDAStateTypeError,
    install_kda_state_index_guards,
    materialization_sources,
    normalize_kda_state_index,
    restore_indexed_state,
    rollback_restore_state,
    validate_kda_rollback,
)


class FakeArraysCache:
    def __init__(self, size):
        self.cache = [None] * size

    def __getitem__(self, index):
        return self.cache[index]

    def __setitem__(self, index, value):
        self.cache[index] = value

    @property
    def state(self):
        return self.cache

    @state.setter
    def state(self, value):
        self.cache = value


install_kda_state_index_guards(FakeArraysCache)


def _digest(cache, metadata) -> str:
    payload = repr((cache.state, metadata)).encode()
    return hashlib.sha256(payload).hexdigest()


def _cache():
    cache = FakeArraysCache(2)
    cache[0] = b"conv"
    cache[1] = b"recurrent"
    return cache


@pytest.mark.parametrize("index", [-2, 2, 3, np.int32(-2), np.int64(2)])
@pytest.mark.parametrize("operation", ["read", "write", "materialize", "restore"])
def test_invalid_read_write_restore_and_materialization_are_atomic(index, operation):
    cache = _cache()
    metadata = {
        "state_index": 7,
        "materialization_epoch": 3,
        "decode_token_counter": 19,
        "lifecycle_accounting": (4096, 0, 1),
        "apc_namespace": "unchanged",
    }
    before = _digest(cache, metadata)
    with pytest.raises(KDAStateIndexError):
        if operation == "read":
            _ = cache[index]
        elif operation == "write":
            cache[index] = b"bad"
        elif operation == "materialize":
            materialization_sources(cache, (0, index, 1))
        else:
            restore_indexed_state(cache, ((0, b"new-conv"), (index, b"bad")))
    assert _digest(cache, metadata) == before


@pytest.mark.parametrize("operation", ["read", "write", "materialize", "restore"])
def test_minus_one_is_a_no_access_sentinel(operation):
    cache = _cache()
    metadata = {
        "materialization_epoch": 3,
        "decode_token_counter": 19,
        "allocation_count": 2,
    }
    before = _digest(cache, metadata)
    if operation == "read":
        assert cache[KDA_STATE_SENTINEL] is None
    elif operation == "write":
        cache[KDA_STATE_SENTINEL] = b"bad"
    elif operation == "materialize":
        assert materialization_sources(cache, (-1,)) == ()
    else:
        restore_indexed_state(cache, ((-1, b"bad"),))
    assert _digest(cache, metadata) == before


def test_last_valid_slot_is_readable_and_writable():
    cache = _cache()
    assert cache[1] == b"recurrent"
    cache[1] = b"updated"
    assert cache[1] == b"updated"
    assert materialization_sources(cache, (0, 1)) == (b"conv", b"updated")


def test_restore_validates_every_destination_before_any_write():
    cache = _cache()
    before = tuple(cache.state)
    with pytest.raises(KDAStateIndexError):
        restore_indexed_state(cache, ((0, b"mutated"), (2, b"bad")))
    assert tuple(cache.state) == before
    with pytest.raises(KDAStateIndexError, match="duplicate"):
        restore_indexed_state(cache, ((0, b"a"), (0, b"b")))
    assert tuple(cache.state) == before


def test_state_restore_shape_is_atomic_and_from_state_construction_still_works():
    cache = _cache()
    before = tuple(cache.state)
    with pytest.raises(KDAStateIndexError):
        cache.state = [b"only-one"]
    assert tuple(cache.state) == before

    uninitialized = FakeArraysCache.__new__(FakeArraysCache)
    uninitialized.state = [b"conv", b"recurrent"]
    assert uninitialized.state == [b"conv", b"recurrent"]


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 8, 15, 16])
def test_rollback_one_through_sixteen_is_exact(tokens):
    cache = _cache()
    snapshot = list(cache.state)
    cache.state = [b"changed-conv", b"changed-recurrent"]
    rollback_restore_state(cache, snapshot, tokens=tokens)
    assert cache.state == snapshot


def test_rollback_seventeen_and_malformed_snapshot_fail_before_mutation():
    cache = _cache()
    before = tuple(cache.state)
    with pytest.raises(KDAStateIndexError):
        rollback_restore_state(cache, [b"a", b"b"], tokens=17)
    assert tuple(cache.state) == before
    with pytest.raises(KDAStateIndexError):
        rollback_restore_state(cache, [b"a"], tokens=1)
    assert tuple(cache.state) == before


@pytest.mark.parametrize("index", [0, 1, np.int32(0), np.int64(1)])
def test_python_and_numpy_signed_integer_types_are_explicitly_supported(index):
    assert normalize_kda_state_index(index, capacity=2) == int(index)


@pytest.mark.parametrize(
    "index",
    [True, False, 1.0, 1.9, float("nan"), float("inf"), "1", None],
)
def test_lossy_or_ambiguous_index_types_are_rejected(index):
    with pytest.raises(KDAStateTypeError):
        normalize_kda_state_index(index, capacity=2)


def test_signed_mlx_scalar_contract_without_requiring_metal():
    def as_array(self, dtype=None, copy=None):
        return np.asarray(self.value, dtype=np.int64 if "64" in self.dtype else np.int32)

    scalar_type = type(
        "array",
        (),
        {
            "__module__": "mlx.core",
            "shape": (),
            "__array__": as_array,
        },
    )
    for dtype, value in (("int32", 0), ("mlx.core.int64", 1)):
        scalar = scalar_type()
        scalar.dtype = dtype
        scalar.value = value
        assert normalize_kda_state_index(scalar, capacity=2) == value

    float_scalar = scalar_type()
    float_scalar.dtype = "float32"
    float_scalar.value = 1.0
    with pytest.raises(KDAStateTypeError):
        normalize_kda_state_index(float_scalar, capacity=2)


def test_contract_has_no_clip_modulo_or_implicit_integer_coercion():
    source = inspect.getsource(normalize_kda_state_index)
    implementation = inspect.getsource(
        type(normalize_kda_state_index.__globals__["INDEX_CONTRACT"]).normalize
    )
    combined = source + implementation
    assert "clip" not in combined
    assert "%" not in combined
    assert KDA_STATE_INDEX_CONTRACT.endswith("sentinel-minus1")
    assert KDA_ROLLBACK_WINDOW == 16
