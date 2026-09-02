"""Fail-closed index contract for GLM-5.3 KDA state slots.

The pinned GLM implementation stores convolution and recurrent state in an
``ArraysCache`` with two slots.  Python's ordinary list semantics would make
``-1`` alias the recurrent slot and would permit accidental negative indexing.
This module makes the state-index ABI explicit without changing the cache type,
its serialized state, or the runtime cache ABI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


KDA_STATE_SENTINEL = -1
KDA_CONV_STATE_SLOT = 0
KDA_RECURRENT_STATE_SLOT = 1
KDA_STATE_SLOT_CAPACITY = 2
KDA_ROLLBACK_WINDOW = 16
KDA_STATE_INDEX_CONTRACT = "glm53-kda-state-index-v1-sentinel-minus1"


class KDAStateIndexError(IndexError):
    """Raised before an invalid KDA state slot can be read or written."""


class KDAStateTypeError(TypeError):
    """Raised when an index could be silently truncated or coerced."""


def _is_mlx_scalar(value: object) -> bool:
    cls = type(value)
    return cls.__module__.startswith("mlx") and cls.__name__ == "array"


def _normalize_integer(index: object) -> int:
    # bool is an int subclass.  It is never a valid state-slot spelling.
    if isinstance(index, (bool, np.bool_)):
        raise KDAStateTypeError("KDA state index must not be bool")
    if isinstance(index, (int, np.integer)):
        return int(index)
    if _is_mlx_scalar(index):
        # MLX scalar support is a host/API boundary convenience, not a decode
        # hot-path operation.  Production KDA calls use literal Python slots.
        shape = tuple(getattr(index, "shape", ()))
        dtype = str(getattr(index, "dtype", ""))
        if shape != ():
            raise KDAStateTypeError("KDA state MLX index must be a scalar")
        if dtype not in {"mlx.core.int32", "mlx.core.int64", "int32", "int64"}:
            raise KDAStateTypeError(
                "KDA state MLX index must have signed int32/int64 dtype"
            )
        return int(np.asarray(index))
    raise KDAStateTypeError(
        "KDA state index must be Python int, NumPy integer, or signed MLX scalar"
    )


@dataclass(frozen=True)
class KDAStateIndexContract:
    """Canonical host/native contract for one fixed-capacity state array."""

    sentinel: int = KDA_STATE_SENTINEL

    def normalize(self, index: object, *, capacity: int) -> int | None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise KDAStateTypeError("KDA state capacity must be a Python int")
        if capacity < 0:
            raise ValueError("KDA state capacity must be non-negative")
        normalized = _normalize_integer(index)
        if normalized == self.sentinel:
            return None
        if normalized < self.sentinel:
            raise KDAStateIndexError(
                f"KDA state index {normalized} is below sentinel {self.sentinel}"
            )
        if normalized >= capacity:
            raise KDAStateIndexError(
                f"KDA state index {normalized} is outside capacity {capacity}"
            )
        return normalized


INDEX_CONTRACT = KDAStateIndexContract()


def normalize_kda_state_index(index: object, *, capacity: int) -> int | None:
    return INDEX_CONTRACT.normalize(index, capacity=capacity)


def validate_kda_rollback(tokens: object, *, maximum: int = KDA_ROLLBACK_WINDOW) -> int:
    value = _normalize_integer(tokens)
    if value < 1 or value > maximum:
        raise KDAStateIndexError(
            f"KDA rollback must be within [1, {maximum}], got {value}"
        )
    return value


def materialization_sources(cache, indices: Iterable[object]):
    """Validate every source index before returning any materialization leaf."""

    requested = tuple(indices)
    normalized = tuple(
        normalize_kda_state_index(index, capacity=len(cache.cache))
        for index in requested
    )
    return tuple(cache[index] for index in normalized if index is not None)


def restore_indexed_state(cache, updates: Iterable[tuple[object, object]]) -> None:
    """Atomically restore indexed KDA state after validating all destinations."""

    pending = tuple(updates)
    normalized = tuple(
        (normalize_kda_state_index(index, capacity=len(cache.cache)), value)
        for index, value in pending
    )
    destinations = [index for index, _ in normalized if index is not None]
    if len(destinations) != len(set(destinations)):
        raise KDAStateIndexError("duplicate KDA state restore destination")

    replacement = list(cache.state)
    for index, value in normalized:
        if index is not None:
            replacement[index] = value
    cache.state = replacement


def rollback_restore_state(
    cache,
    snapshot: Sequence[object],
    *,
    tokens: object,
) -> None:
    """Atomically restore a validated short rollback snapshot."""

    validate_kda_rollback(tokens)
    if not isinstance(snapshot, (tuple, list)):
        raise KDAStateTypeError("KDA rollback snapshot must be a tuple or list")
    if len(snapshot) != len(cache.cache):
        raise KDAStateIndexError(
            "KDA rollback snapshot slot count does not match live capacity"
        )
    cache.state = list(snapshot)


def install_kda_state_index_guards(arrays_cache_cls) -> None:
    """Install the contract on the pinned ``ArraysCache`` class once.

    Keeping the original class preserves the RAM/disk APC schema and lets
    clones restored by mlx-vlm inherit the same guard automatically.
    """

    if getattr(arrays_cache_cls, "_glm53_kda_state_index_guard", False):
        return

    original_getitem = arrays_cache_cls.__getitem__
    original_setitem = arrays_cache_cls.__setitem__

    def guarded_getitem(self, index):
        normalized = normalize_kda_state_index(index, capacity=len(self.cache))
        if normalized is None:
            return None
        return original_getitem(self, normalized)

    def guarded_setitem(self, index, value):
        normalized = normalize_kda_state_index(index, capacity=len(self.cache))
        if normalized is None:
            return None
        original_setitem(self, normalized, value)
        return None

    def guarded_state_getter(self):
        # Materialization/APC reads traverse only validated positive slots.
        return [guarded_getitem(self, index) for index in range(len(self.cache))]

    def guarded_state_setter(self, value):
        # Snapshot restore validates its complete shape before replacing live
        # state, so a malformed restore cannot partially mutate the cache.
        if not isinstance(value, (tuple, list)):
            raise KDAStateTypeError("ArraysCache state must be a tuple or list")
        capacity = len(self.cache) if hasattr(self, "cache") else len(value)
        if len(value) != capacity:
            raise KDAStateIndexError(
                f"restored state has {len(value)} slots; expected {capacity}"
            )
        replacement = list(value)
        self.cache = replacement

    arrays_cache_cls.__getitem__ = guarded_getitem
    arrays_cache_cls.__setitem__ = guarded_setitem
    arrays_cache_cls.state = property(guarded_state_getter, guarded_state_setter)
    arrays_cache_cls._glm53_kda_state_index_guard = True
    arrays_cache_cls._glm53_kda_state_index_contract = KDA_STATE_INDEX_CONTRACT
