from __future__ import annotations

import pytest

from glm53_flash_mlx.churn import (
    BASELINE_256K_CUMULATIVE_ALLOCATION_BYTES,
    accounting_balance_errors,
    churn_tier,
    distributed_churn_schedule,
    required_churn_cycles,
    rollback_schedule,
)


def test_churn_tiers_separate_logical_length_from_allocation_pressure():
    screen = churn_tier("screen")
    qualification = churn_tier("qualification")
    extended = churn_tier("extended")
    assert screen.logical_tokens == 4_096
    assert screen.cumulative_allocation_target_bytes == 50_000_000_000
    assert qualification.logical_tokens == extended.logical_tokens == 16_384
    assert qualification.cumulative_allocation_target_bytes == (
        BASELINE_256K_CUMULATIVE_ALLOCATION_BYTES
    )
    assert extended.cumulative_allocation_target_bytes == 1 << 40


def test_required_cycles_and_schedule_are_deterministic_and_dense():
    assert required_churn_cycles(1_000, 100, 400) == 3
    assert required_churn_cycles(100, 100, 400) == 0
    assert distributed_churn_schedule(8, 4) == (2, 4, 6, 8)
    assert distributed_churn_schedule(2, 4) == (1, 1, 1, 2)
    with pytest.raises(ValueError):
        required_churn_cycles(1, 0, 0)


def test_rollback_schedule_rotates_all_supported_depths():
    rows = rollback_schedule(churn_tier("qualification"))
    assert len(rows) == 32
    assert rows[:6] == (
        (512, 1),
        (1_024, 8),
        (1_536, 16),
        (2_048, 1),
        (2_560, 8),
        (3_072, 16),
    )
    assert {depth for _, depth in rows} == {1, 8, 16}


def test_accounting_balance_detects_per_class_and_total_errors():
    row = {
        "cumulative_allocated_bytes": 100,
        "cumulative_released_bytes": 30,
        "transfer_in_bytes": 10,
        "transfer_out_bytes": 20,
        "resident_bytes": 60,
    }
    snapshot = {
        "by_lifecycle": {"snapshot-state": dict(row)},
        "resident_bytes": 60,
        "ownership_balance_exact": True,
    }
    assert accounting_balance_errors(snapshot) == ()
    snapshot["by_lifecycle"]["snapshot-state"]["resident_bytes"] = 61
    assert accounting_balance_errors(snapshot)
