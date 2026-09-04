"""Deterministic schedules and accounting gates for cache churn qualification."""

from __future__ import annotations

from dataclasses import dataclass


BASELINE_256K_CUMULATIVE_ALLOCATION_BYTES = 446_825_650_554


@dataclass(frozen=True)
class ChurnTier:
    name: str
    logical_tokens: int
    cumulative_allocation_target_bytes: int
    rollback_interval: int


CHURN_TIERS = {
    # Dense enough to force APC churn while a rollback source snapshot is live.
    "developer-smoke": ChurnTier("developer-smoke", 256, 10_000_000_000, 64),
    "screen": ChurnTier("screen", 4_096, 50_000_000_000, 1_024),
    "qualification": ChurnTier(
        "qualification",
        16_384,
        BASELINE_256K_CUMULATIVE_ALLOCATION_BYTES,
        512,
    ),
    "extended": ChurnTier("extended", 16_384, 1 << 40, 512),
}


def churn_tier(name: str) -> ChurnTier:
    try:
        return CHURN_TIERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown churn tier: {name}") from exc


def required_churn_cycles(
    target_bytes: int,
    current_bytes: int,
    allocation_bytes_per_cycle: int,
) -> int:
    if target_bytes < 0 or current_bytes < 0:
        raise ValueError("allocation byte counts must be non-negative")
    if allocation_bytes_per_cycle <= 0:
        raise ValueError("allocation bytes per cycle must be positive")
    remaining = max(0, target_bytes - current_bytes)
    return (remaining + allocation_bytes_per_cycle - 1) // allocation_bytes_per_cycle


def distributed_churn_schedule(
    logical_tokens: int,
    cycles: int,
) -> tuple[int, ...]:
    """Return a deterministic step for every cycle, allowing dense repeats."""

    if logical_tokens < 1 or cycles < 0:
        raise ValueError("logical tokens must be positive and cycles non-negative")
    return tuple(
        max(1, min(logical_tokens, ((index + 1) * logical_tokens) // cycles))
        for index in range(cycles)
    ) if cycles else ()


def rollback_schedule(tier: ChurnTier) -> tuple[tuple[int, int], ...]:
    depths = (1, 8, 16)
    targets = range(tier.rollback_interval, tier.logical_tokens + 1, tier.rollback_interval)
    return tuple(
        (target, depths[index % len(depths)])
        for index, target in enumerate(targets)
    )


def accounting_balance_errors(snapshot: dict) -> tuple[str, ...]:
    errors = []
    resident_total = 0
    for lifecycle, row in snapshot["by_lifecycle"].items():
        expected = (
            int(row["cumulative_allocated_bytes"])
            + int(row["transfer_in_bytes"])
            - int(row["cumulative_released_bytes"])
            - int(row["transfer_out_bytes"])
        )
        resident = int(row["resident_bytes"])
        resident_total += resident
        if expected != resident:
            errors.append(f"{lifecycle}: expected {expected}, observed {resident}")
    if resident_total != int(snapshot["resident_bytes"]):
        errors.append(
            f"total: expected {resident_total}, observed {snapshot['resident_bytes']}"
        )
    if not snapshot.get("ownership_balance_exact", False):
        errors.append("snapshot ownership_balance_exact is false")
    return tuple(errors)


def temporary_storage_returned(before: dict, after: dict) -> bool:
    """Require a transition to preserve any pre-existing temporary owners."""

    for lifecycle in ("snapshot-state", "draft-transient"):
        if (
            int(after["by_lifecycle"][lifecycle]["resident_bytes"])
            != int(before["by_lifecycle"][lifecycle]["resident_bytes"])
        ):
            return False
    return True
