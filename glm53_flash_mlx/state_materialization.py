"""Orthogonal value-materialization and persistent-cache write contracts.

Cache ownership answers whether a computed value may be persisted.  It must
never be used to infer whether a downstream consumer still needs that value.
This small host-side contract is intentionally independent of tensor layout,
dtype, cache lifecycle, and the runtime cache ABI.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar


CACHE_WRITE_SENTINEL = -1
STATE_MATERIALIZATION_CONTRACT = (
    "glm53-state-materialization-v1"
    "-value-independent-of-write-owner"
    "-sentinel-minus1"
    "-preflight-before-producer"
)


class StateMaterializationError(ValueError):
    """Raised before computation or state mutation for an invalid request."""


def normalize_cache_write_slot(
    slot: object | None,
    *,
    capacity: int,
) -> int | None:
    """Resolve an owned destination without clamp, modulo, or float coercion."""

    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise StateMaterializationError("cache write capacity must be a Python int")
    if capacity < 0:
        raise StateMaterializationError("cache write capacity must be non-negative")
    if slot is None:
        return None
    if isinstance(slot, bool):
        raise StateMaterializationError("cache write slot must not be bool")
    try:
        normalized = operator.index(slot)
    except TypeError as error:
        raise StateMaterializationError(
            "cache write slot must be an integer, None, or the -1 sentinel"
        ) from error
    if normalized == CACHE_WRITE_SENTINEL:
        return None
    if normalized < CACHE_WRITE_SENTINEL:
        raise StateMaterializationError(
            f"cache write slot {normalized} is below sentinel {CACHE_WRITE_SENTINEL}"
        )
    if normalized >= capacity:
        raise StateMaterializationError(
            f"cache write slot {normalized} is outside capacity {capacity}"
        )
    return normalized


@dataclass(frozen=True)
class MaterializationRequest:
    """A value requirement plus a logically independent persistence owner."""

    require_value: bool
    cache_write_slot: object | None

    def __post_init__(self) -> None:
        if not isinstance(self.require_value, bool):
            raise StateMaterializationError("require_value must be bool")

    def resolve_write_slot(self, *, capacity: int) -> int | None:
        return normalize_cache_write_slot(self.cache_write_slot, capacity=capacity)


ValueT = TypeVar("ValueT")


@dataclass(frozen=True)
class MaterializationResult(Generic[ValueT]):
    value: ValueT | None
    materialized: bool
    cache_written: bool
    cache_write_slot: int | None


def execute_state_materialization(
    request: MaterializationRequest,
    *,
    cache_capacity: int,
    producer: Callable[[], ValueT],
    cache_writer: Callable[[int, ValueT], None] | None = None,
) -> MaterializationResult[ValueT]:
    """Materialize when required, then independently persist when owned.

    Destination and writer validation happen before ``producer`` is invoked.
    Consequently, a malformed destination cannot allocate a temporary value or
    partially mutate authoritative cache state.  ``None`` and ``-1`` suppress
    only the write: they never suppress a required computation.
    """

    if not isinstance(request, MaterializationRequest):
        raise TypeError("state materialization requires MaterializationRequest")
    if not callable(producer):
        raise TypeError("state materialization producer must be callable")
    slot = request.resolve_write_slot(capacity=cache_capacity)
    if slot is not None and cache_writer is None:
        raise StateMaterializationError(
            "an owned cache write destination requires an explicit writer"
        )
    if cache_writer is not None and not callable(cache_writer):
        raise TypeError("cache writer must be callable")
    if not request.require_value:
        if slot is not None:
            raise StateMaterializationError(
                "cache write cannot be requested without value materialization"
            )
        return MaterializationResult(
            value=None,
            materialized=False,
            cache_written=False,
            cache_write_slot=None,
        )

    value = producer()
    if slot is not None:
        assert cache_writer is not None
        cache_writer(slot, value)
    return MaterializationResult(
        value=value,
        materialized=True,
        cache_written=slot is not None,
        cache_write_slot=slot,
    )
