"""Atomic winner commit for isolated semantic trajectory branches.

Evaluation policy is deliberately outside this module.  The transaction only
accepts a selected branch and binds that selection to an exact branch
generation, position, and semantic digest before promoting its already-owned
cache into the active root.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, Sequence

from .materialization import MATERIALIZATION_INTERVAL_TOKENS
from .semantic_branch import (
    PromotedSemanticBranch,
    SemanticBranch,
    SemanticBranchManager,
)
from .semantic_snapshot import (
    HybridSemanticPrefixSnapshot,
    SemanticCacheHandle,
    SemanticSnapshotIdentity,
    inspect_semantic_boundary,
    prepare_snapshot_restore,
    semantic_cache_digest,
    semantic_cache_resident_bytes,
)


TRAJECTORY_TRANSACTION_SCHEMA = "glm53-trajectory-transaction-v1-cas-winner-promotion"
TRANSACTION_STATE_LIFECYCLE = "trajectory-transaction-state"


class TrajectoryTransactionError(ValueError):
    """Base class for fail-closed transaction state errors."""


class StaleTrajectoryTransaction(TrajectoryTransactionError):
    """The active root changed after this transaction began."""


class StaleTrajectoryEvaluation(TrajectoryTransactionError):
    """The evaluated branch changed before winner commit."""


class TransactionState(str, Enum):
    OPEN = "open"
    EXECUTED = "executed"
    EVALUATED = "evaluated"
    COMMITTED = "committed"
    ABORTED = "aborted"


TERMINAL_TRANSACTION_STATES = {
    TransactionState.COMMITTED,
    TransactionState.ABORTED,
}


@dataclass(frozen=True)
class TrajectoryEvaluation:
    transaction_id: str
    winner_branch_id: int
    branch_generation: int
    terminal_position: int
    terminal_state_digest: str
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.transaction_id, str) or not self.transaction_id:
            raise TrajectoryTransactionError("evaluation requires a transaction id")
        if (
            isinstance(self.winner_branch_id, bool)
            or not isinstance(self.winner_branch_id, int)
            or self.winner_branch_id < 1
        ):
            raise TrajectoryTransactionError("evaluation requires a valid winner branch")
        if self.branch_generation < 0 or self.terminal_position < 0:
            raise TrajectoryTransactionError("evaluation generation/position is invalid")
        if not self.terminal_state_digest:
            raise TrajectoryTransactionError("evaluation requires a terminal state digest")
        if not math.isfinite(self.score):
            raise TrajectoryTransactionError("evaluation score must be finite")

    def descriptor(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WinnerCommitResult:
    transaction_id: str
    winner_branch_id: int
    active_generation_before: int
    active_generation_after: int
    winner_semantic_digest: str
    active_semantic_digest: str
    winner_cache_id_before: int
    active_cache_id_after: int
    winner_promoted_bytes: int
    loser_released_bytes: int
    previous_active_released_bytes: int
    mixed_generation_count: int

    @property
    def winner_storage_promoted_without_copy(self) -> bool:
        return self.winner_cache_id_before == self.active_cache_id_after

    def descriptor(self) -> dict[str, object]:
        return {
            **asdict(self),
            "winner_storage_promoted_without_copy": (
                self.winner_storage_promoted_without_copy
            ),
        }


class ActiveSemanticState:
    """Single owned active-root reference with generation-based CAS identity."""

    def __init__(
        self,
        cache: Sequence[object],
        *,
        compatibility_identity: SemanticSnapshotIdentity,
        position: int,
    ) -> None:
        if not cache:
            raise TrajectoryTransactionError("active semantic state requires cache state")
        self._handle: SemanticCacheHandle | None = SemanticCacheHandle(cache)
        self.compatibility_identity = compatibility_identity
        self.position = int(position)
        self.generation = 0
        initial = semantic_cache_resident_bytes(self.cache)
        self._resident_bytes = initial
        self._peak_bytes = initial
        self._cumulative_allocated_bytes = initial
        self._cumulative_released_bytes = 0
        self._winner_promoted_bytes = 0
        self._promotion_count = 0
        self._restore_count = 0

    @classmethod
    def from_snapshot(
        cls,
        snapshot: HybridSemanticPrefixSnapshot,
        *,
        expected_identity: SemanticSnapshotIdentity,
        clone_entry: Callable | None = None,
    ) -> "ActiveSemanticState":
        replacement = prepare_snapshot_restore(
            snapshot,
            expected_identity=expected_identity,
            clone_entry=clone_entry,
        )
        return cls(
            replacement,
            compatibility_identity=snapshot.identity,
            position=snapshot.boundary.absolute_token_position,
        )

    @property
    def released(self) -> bool:
        return self._handle is None

    @property
    def cache(self) -> list[object]:
        if self._handle is None:
            raise TrajectoryTransactionError("active semantic state is released")
        return self._handle.cache

    @property
    def state_sha256(self) -> str:
        return semantic_cache_digest(self.cache)

    @property
    def resident_bytes(self) -> int:
        return semantic_cache_resident_bytes(self.cache) if not self.released else 0

    def commit_position(self, absolute_token_position: int) -> None:
        if (
            isinstance(absolute_token_position, bool)
            or not isinstance(absolute_token_position, int)
            or absolute_token_position < self.position
        ):
            raise TrajectoryTransactionError("active position must advance monotonically")
        inspect_semantic_boundary(
            self.cache,
            absolute_token_position=absolute_token_position,
            materialization_epoch=(
                absolute_token_position // MATERIALIZATION_INTERVAL_TOKENS
            ),
        )
        if absolute_token_position != self.position:
            self.position = absolute_token_position
            self.generation += 1
        self._observe_resident()

    def restore_snapshot(
        self,
        snapshot: HybridSemanticPrefixSnapshot,
        *,
        expected_identity: SemanticSnapshotIdentity,
        clone_entry: Callable | None = None,
    ) -> None:
        replacement = prepare_snapshot_restore(
            snapshot,
            expected_identity=expected_identity,
            clone_entry=clone_entry,
        )
        previous = self._observe_resident()
        replacement_bytes = semantic_cache_resident_bytes(replacement)
        replacement_handle = SemanticCacheHandle(replacement)
        self._handle = replacement_handle
        self.compatibility_identity = snapshot.identity
        self.position = snapshot.boundary.absolute_token_position
        self.generation += 1
        self._resident_bytes = replacement_bytes
        self._peak_bytes = max(self._peak_bytes, replacement_bytes)
        self._cumulative_allocated_bytes += replacement_bytes
        self._cumulative_released_bytes += previous
        self._restore_count += 1

    def _observe_resident(self) -> int:
        current = self.resident_bytes
        if current > self._resident_bytes:
            self._cumulative_allocated_bytes += current - self._resident_bytes
        elif current < self._resident_bytes:
            self._cumulative_released_bytes += self._resident_bytes - current
        self._resident_bytes = current
        self._peak_bytes = max(self._peak_bytes, current)
        return current

    def _commit_promoted(self, promoted: PromotedSemanticBranch) -> tuple[int, int]:
        previous = self._observe_resident()
        self._handle = promoted._handle
        self.compatibility_identity = promoted.compatibility_identity
        self.position = promoted.position
        self.generation += 1
        self._resident_bytes = promoted.resident_bytes
        self._peak_bytes = max(self._peak_bytes, promoted.resident_bytes)
        self._cumulative_released_bytes += previous
        self._winner_promoted_bytes += promoted.resident_bytes
        self._promotion_count += 1
        return previous, promoted.resident_bytes

    def release(self) -> int:
        previous = self._observe_resident()
        self._handle = None
        self._resident_bytes = 0
        self._cumulative_released_bytes += previous
        return previous

    def accounting(self) -> dict[str, object]:
        self._observe_resident()
        return {
            "generation": self.generation,
            "position": self.position,
            "resident_bytes": self._resident_bytes,
            "peak_bytes": self._peak_bytes,
            "cumulative_allocated_bytes": self._cumulative_allocated_bytes,
            "cumulative_released_bytes": self._cumulative_released_bytes,
            "winner_promoted_bytes": self._winner_promoted_bytes,
            "cumulative_promoted_in_bytes": self._winner_promoted_bytes,
            "ownership_balance_bytes": (
                self._cumulative_allocated_bytes
                + self._winner_promoted_bytes
                - self._cumulative_released_bytes
                - self._resident_bytes
            ),
            "promotion_count": self._promotion_count,
            "restore_count": self._restore_count,
            "anonymous_allocation_count": 0,
        }


@dataclass
class TrajectoryTransaction:
    transaction_id: str
    parent_snapshot_id: str
    candidate_branch_ids: tuple[int, ...]
    base_active_generation: int
    base_active_position: int
    base_active_state_digest: str
    state: TransactionState = TransactionState.OPEN
    evaluation: TrajectoryEvaluation | None = None
    commit_result: WinnerCommitResult | None = None

    def descriptor(self) -> dict[str, object]:
        return {
            "transaction_id": self.transaction_id,
            "parent_snapshot_id": self.parent_snapshot_id,
            "candidate_branch_ids": list(self.candidate_branch_ids),
            "base_active_generation": self.base_active_generation,
            "base_active_position": self.base_active_position,
            "base_active_state_digest": self.base_active_state_digest,
            "state": self.state.value,
            "evaluation": self.evaluation.descriptor() if self.evaluation else None,
            "commit_result": (
                self.commit_result.descriptor() if self.commit_result else None
            ),
        }


class TrajectoryTransactionManager:
    """State machine and CAS commit over an active root and branch manager."""

    def __init__(
        self,
        active: ActiveSemanticState,
        branches: SemanticBranchManager | None = None,
    ) -> None:
        self.active = active
        self.branches = branches or SemanticBranchManager()
        self._transactions: dict[str, TrajectoryTransaction] = {}
        self._transaction_count = 0
        self._commit_count = 0
        self._abort_count = 0
        self._stale_commit_reject_count = 0
        self._winner_promoted_bytes = 0
        self._loser_released_bytes = 0

    def _get(self, transaction_id: str) -> TrajectoryTransaction:
        try:
            return self._transactions[transaction_id]
        except KeyError as error:
            raise TrajectoryTransactionError("unknown trajectory transaction") from error

    @staticmethod
    def _require_state(
        transaction: TrajectoryTransaction, allowed: set[TransactionState]
    ) -> None:
        if transaction.state not in allowed:
            raise TrajectoryTransactionError(
                f"transaction is {transaction.state.value}; operation is not allowed"
            )

    def begin(
        self,
        *,
        transaction_id: str,
        parent_snapshot: HybridSemanticPrefixSnapshot,
        expected_identity: SemanticSnapshotIdentity,
        candidate_count: int = 2,
    ) -> TrajectoryTransaction:
        if not isinstance(transaction_id, str) or not transaction_id:
            raise TrajectoryTransactionError("transaction id must be non-empty")
        if transaction_id in self._transactions:
            raise TrajectoryTransactionError("duplicate trajectory transaction id")
        if (
            isinstance(candidate_count, bool)
            or not isinstance(candidate_count, int)
            or candidate_count < 2
        ):
            raise TrajectoryTransactionError("candidate_count must be at least two")
        active_digest = self.active.state_sha256
        if (
            self.active.position != parent_snapshot.boundary.absolute_token_position
            or active_digest != parent_snapshot.state_sha256
            or self.active.compatibility_identity != expected_identity
        ):
            raise StaleTrajectoryTransaction(
                "active root does not match the transaction parent snapshot"
            )
        created = []
        try:
            for _ in range(candidate_count):
                created.append(
                    self.branches.fork(
                        parent_snapshot, expected_identity=expected_identity
                    ).identity.branch_id
                )
        except Exception:
            self.branches.deactivate()
            for branch_id in created:
                self.branches.delete_branch(branch_id)
            raise
        transaction = TrajectoryTransaction(
            transaction_id=transaction_id,
            parent_snapshot_id=parent_snapshot.snapshot_id,
            candidate_branch_ids=tuple(created),
            base_active_generation=self.active.generation,
            base_active_position=self.active.position,
            base_active_state_digest=active_digest,
        )
        self._transactions[transaction_id] = transaction
        self._transaction_count += 1
        return transaction

    def mark_executed(self, transaction_id: str) -> None:
        transaction = self._get(transaction_id)
        self._require_state(transaction, {TransactionState.OPEN})
        if any(
            self.branches.branch(branch_id).position
            <= transaction.base_active_position
            for branch_id in transaction.candidate_branch_ids
        ):
            raise TrajectoryTransactionError(
                "every candidate must advance before execution is complete"
            )
        transaction.state = TransactionState.EXECUTED

    def evaluate(
        self,
        transaction_id: str,
        *,
        winner_branch_id: int,
        score: float,
    ) -> TrajectoryEvaluation:
        transaction = self._get(transaction_id)
        self._require_state(transaction, {TransactionState.EXECUTED})
        if winner_branch_id not in transaction.candidate_branch_ids:
            raise TrajectoryTransactionError("winner is not a transaction candidate")
        branch = self.branches.branch(winner_branch_id)
        before = (
            branch.identity,
            branch.position,
            branch.state_sha256,
            self.branches.accounting()["by_branch"][str(winner_branch_id)],
        )
        evaluation = TrajectoryEvaluation(
            transaction_id=transaction_id,
            winner_branch_id=winner_branch_id,
            branch_generation=branch.identity.generation,
            terminal_position=branch.position,
            terminal_state_digest=before[2],
            score=float(score),
        )
        after = (
            branch.identity,
            branch.position,
            branch.state_sha256,
            self.branches.accounting()["by_branch"][str(winner_branch_id)],
        )
        if after != before:
            raise AssertionError("trajectory evaluation mutated branch state")
        transaction.evaluation = evaluation
        transaction.state = TransactionState.EVALUATED
        return evaluation

    def commit(self, transaction_id: str) -> WinnerCommitResult:
        transaction = self._get(transaction_id)
        self._require_state(transaction, {TransactionState.EVALUATED})
        evaluation = transaction.evaluation
        if evaluation is None:
            raise AssertionError("evaluated transaction has no evaluation record")
        if (
            self.active.generation != transaction.base_active_generation
            or self.active.position != transaction.base_active_position
            or self.active.state_sha256 != transaction.base_active_state_digest
        ):
            self._stale_commit_reject_count += 1
            raise StaleTrajectoryTransaction("active root changed after transaction begin")
        winner = self.branches.branch(evaluation.winner_branch_id)
        winner_digest = winner.state_sha256
        if (
            winner.identity.generation != evaluation.branch_generation
            or winner.position != evaluation.terminal_position
            or winner_digest != evaluation.terminal_state_digest
        ):
            self._stale_commit_reject_count += 1
            raise StaleTrajectoryEvaluation("evaluated winner changed before commit")
        loser_ids = tuple(
            branch_id
            for branch_id in transaction.candidate_branch_ids
            if branch_id != evaluation.winner_branch_id
        )
        loser_bytes = sum(
            self.branches.branch(branch_id).resident_bytes for branch_id in loser_ids
        )
        winner_cache_id = id(winner.cache)
        generation_before = self.active.generation
        # No callback or validation occurs after this point.  The owned winner
        # handle is transferred, the active root is swapped once, and only then
        # are losers disposed.
        self.branches.deactivate()
        promoted = self.branches.promote_branch(evaluation.winner_branch_id)
        previous_active, promoted_bytes = self.active._commit_promoted(promoted)
        for branch_id in loser_ids:
            self.branches.delete_branch(branch_id)
        active_digest = self.active.state_sha256
        result = WinnerCommitResult(
            transaction_id=transaction_id,
            winner_branch_id=evaluation.winner_branch_id,
            active_generation_before=generation_before,
            active_generation_after=self.active.generation,
            winner_semantic_digest=winner_digest,
            active_semantic_digest=active_digest,
            winner_cache_id_before=winner_cache_id,
            active_cache_id_after=id(self.active.cache),
            winner_promoted_bytes=promoted_bytes,
            loser_released_bytes=loser_bytes,
            previous_active_released_bytes=previous_active,
            mixed_generation_count=0,
        )
        if active_digest != winner_digest:
            raise AssertionError("winner promotion changed semantic state")
        transaction.commit_result = result
        transaction.state = TransactionState.COMMITTED
        self._commit_count += 1
        self._winner_promoted_bytes += promoted_bytes
        self._loser_released_bytes += loser_bytes
        return result

    def abort(self, transaction_id: str) -> None:
        transaction = self._get(transaction_id)
        self._require_state(
            transaction,
            {
                TransactionState.OPEN,
                TransactionState.EXECUTED,
                TransactionState.EVALUATED,
            },
        )
        active_before = (
            self.active.generation,
            self.active.position,
            self.active.state_sha256,
            id(self.active.cache),
        )
        if self.branches.active_branch_id in transaction.candidate_branch_ids:
            self.branches.deactivate()
        for branch_id in transaction.candidate_branch_ids:
            if self.branches.has_branch(branch_id):
                self.branches.delete_branch(branch_id)
        active_after = (
            self.active.generation,
            self.active.position,
            self.active.state_sha256,
            id(self.active.cache),
        )
        if active_after != active_before:
            raise AssertionError("transaction abort changed active semantic state")
        transaction.state = TransactionState.ABORTED
        self._abort_count += 1

    def transaction(self, transaction_id: str) -> TrajectoryTransaction:
        return self._get(transaction_id)

    def accounting(self) -> dict[str, object]:
        branch_accounting = self.branches.accounting()
        resident_by_transaction = {}
        for transaction_id, transaction in self._transactions.items():
            resident = 0
            for branch_id in transaction.candidate_branch_ids:
                row = branch_accounting["by_branch"].get(str(branch_id))
                if row is not None:
                    resident += row["resident_bytes"]
            resident_by_transaction[transaction_id] = resident
        return {
            "schema": TRAJECTORY_TRANSACTION_SCHEMA,
            "lifecycle": TRANSACTION_STATE_LIFECYCLE,
            "transaction_count": self._transaction_count,
            "commit_count": self._commit_count,
            "abort_count": self._abort_count,
            "stale_commit_reject_count": self._stale_commit_reject_count,
            "resident_bytes_by_transaction": resident_by_transaction,
            "transaction_resident_bytes": sum(resident_by_transaction.values()),
            "winner_promoted_bytes": self._winner_promoted_bytes,
            "loser_released_bytes": self._loser_released_bytes,
            "ownership_balance_exact": (
                branch_accounting["ownership_balance_bytes"] == 0
                and self.active.accounting()["ownership_balance_bytes"] == 0
            ),
            "branch": branch_accounting,
            "active": self.active.accounting(),
            "anonymous_allocation_count": 0,
            "transactions": {
                key: value.descriptor()
                for key, value in sorted(self._transactions.items())
            },
        }
