from __future__ import annotations

import hashlib

import pytest

from glm53_flash_mlx.state_materialization import (
    CACHE_WRITE_SENTINEL,
    STATE_MATERIALIZATION_CONTRACT,
    MaterializationRequest,
    StateMaterializationError,
    execute_state_materialization,
    normalize_cache_write_slot,
)


class _Slots:
    def __init__(self, *values: bytes):
        self.values = list(values)
        self.write_count = 0
        self.allocated_bytes = 0

    def write(self, slot: int, value: bytes) -> None:
        self.values[slot] = bytes(value)
        self.write_count += 1
        self.allocated_bytes += len(value)

    def digest(self) -> str:
        value = b"\0".join(self.values)
        return hashlib.sha256(value).hexdigest()

    def accounting(self) -> tuple[int, int]:
        return self.write_count, self.allocated_bytes


def _producer(calls: list[str], value: bytes = b"materialized"):
    def produce() -> bytes:
        calls.append("produce")
        return value

    return produce


def test_materialize_and_owned_write_are_both_exact():
    cache = _Slots(b"old")
    calls = []
    result = execute_state_materialization(
        MaterializationRequest(require_value=True, cache_write_slot=0),
        cache_capacity=1,
        producer=_producer(calls),
        cache_writer=cache.write,
    )
    assert result.value == b"materialized"
    assert result.materialized is True
    assert result.cache_written is True
    assert result.cache_write_slot == 0
    assert cache.values == [b"materialized"]
    assert cache.accounting() == (1, len(b"materialized"))
    assert calls == ["produce"]


@pytest.mark.parametrize("no_owner", [None, CACHE_WRITE_SENTINEL])
def test_required_value_is_materialized_without_cache_ownership(no_owner):
    cache = _Slots(b"authoritative")
    before = cache.digest(), cache.accounting()
    calls = []
    result = execute_state_materialization(
        MaterializationRequest(require_value=True, cache_write_slot=no_owner),
        cache_capacity=1,
        producer=_producer(calls),
        cache_writer=cache.write,
    )
    assert result.value == b"materialized"
    assert result.materialized is True
    assert result.cache_written is False
    assert result.cache_write_slot is None
    assert calls == ["produce"]
    assert (cache.digest(), cache.accounting()) == before


@pytest.mark.parametrize("no_owner", [None, CACHE_WRITE_SENTINEL])
def test_no_materialization_and_no_owner_is_an_allocation_free_noop(no_owner):
    cache = _Slots(b"authoritative")
    before = cache.digest(), cache.accounting()
    calls = []
    result = execute_state_materialization(
        MaterializationRequest(require_value=False, cache_write_slot=no_owner),
        cache_capacity=1,
        producer=_producer(calls),
        cache_writer=cache.write,
    )
    assert result.value is None
    assert result.materialized is False
    assert result.cache_written is False
    assert calls == []
    assert (cache.digest(), cache.accounting()) == before


@pytest.mark.parametrize("slot", [-2, 1, 2])
def test_invalid_destination_fails_before_producer_or_authoritative_write(slot):
    cache = _Slots(b"authoritative")
    before = cache.digest(), cache.accounting()
    calls = []
    with pytest.raises(StateMaterializationError):
        execute_state_materialization(
            MaterializationRequest(require_value=True, cache_write_slot=slot),
            cache_capacity=1,
            producer=_producer(calls),
            cache_writer=cache.write,
        )
    assert calls == []
    assert (cache.digest(), cache.accounting()) == before


def test_owned_destination_requires_writer_before_producer_runs():
    calls = []
    with pytest.raises(StateMaterializationError, match="explicit writer"):
        execute_state_materialization(
            MaterializationRequest(require_value=True, cache_write_slot=0),
            cache_capacity=1,
            producer=_producer(calls),
        )
    assert calls == []


def test_write_without_materialization_is_rejected_before_state_mutation():
    cache = _Slots(b"authoritative")
    before = cache.digest(), cache.accounting()
    calls = []
    with pytest.raises(StateMaterializationError, match="without value"):
        execute_state_materialization(
            MaterializationRequest(require_value=False, cache_write_slot=0),
            cache_capacity=1,
            producer=_producer(calls),
            cache_writer=cache.write,
        )
    assert calls == []
    assert (cache.digest(), cache.accounting()) == before


@pytest.mark.parametrize("slot", [True, False, 0.0, 1.9, float("nan")])
def test_slot_never_uses_bool_float_or_implicit_numeric_coercion(slot):
    with pytest.raises(StateMaterializationError):
        normalize_cache_write_slot(slot, capacity=2)


def test_contract_identity_names_both_independent_axes_and_preflight():
    assert "value-independent-of-write-owner" in STATE_MATERIALIZATION_CONTRACT
    assert "sentinel-minus1" in STATE_MATERIALIZATION_CONTRACT
    assert "preflight-before-producer" in STATE_MATERIALIZATION_CONTRACT
