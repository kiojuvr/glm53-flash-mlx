"""First-class, eager-copy semantic branch ownership and isolation.

Branches are mutable hybrid live states forked from immutable semantic
snapshots.  Version one intentionally shares no mutable tensor storage and
executes only one explicitly activated branch at a time.  Compatibility
identity remains a snapshot concern; branch lineage is diagnostic provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Callable

from .kda_state import KDA_ROLLBACK_WINDOW
from .materialization import MATERIALIZATION_INTERVAL_TOKENS
from .semantic_snapshot import (
    HybridSemanticPrefixSnapshot,
    SemanticCacheHandle,
    SemanticSnapshotIdentity,
    SemanticSnapshotStore,
    inspect_semantic_boundary,
    prepare_snapshot_restore,
    semantic_cache_digest,
    semantic_cache_resident_bytes,
    semantic_cache_storage_alias_count,
)


SEMANTIC_BRANCH_SCHEMA = "glm53-semantic-branch-v1-eager-owned-isolated"
SEMANTIC_BRANCH_LIFECYCLE = "semantic-branch-state"


class SemanticBranchError(ValueError):
    """Raised before a branch publish, activation, restore, or deletion mutates state."""


def _branch_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SemanticBranchError("branch_id must be a positive Python int")
    return value


def _position(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SemanticBranchError("branch position must be a non-negative Python int")
    return value


def _nonempty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticBranchError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class BranchIdentity:
    branch_id: int
    parent_snapshot_id: str
    generation: int = 0

    def __post_init__(self) -> None:
        _branch_id(self.branch_id)
        _nonempty("parent_snapshot_id", self.parent_snapshot_id)
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise SemanticBranchError("branch generation must be non-negative")

    def descriptor(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BranchSnapshotLineage:
    snapshot_id: str
    source_branch_id: int
    source_branch_generation: int
    parent_snapshot_id: str
    absolute_token_position: int

    def __post_init__(self) -> None:
        _nonempty("snapshot_id", self.snapshot_id)
        _branch_id(self.source_branch_id)
        if self.source_branch_generation < 0:
            raise SemanticBranchError("source branch generation must be non-negative")
        _nonempty("parent_snapshot_id", self.parent_snapshot_id)
        _position(self.absolute_token_position)

    def descriptor(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class _BranchAccounting:
    resident_bytes: int = 0
    peak_bytes: int = 0
    cumulative_allocated_bytes: int = 0
    cumulative_released_bytes: int = 0
    allocation_count: int = 0
    release_count: int = 0

    def allocate(self, nbytes: int) -> None:
        self.resident_bytes += nbytes
        self.peak_bytes = max(self.peak_bytes, self.resident_bytes)
        self.cumulative_allocated_bytes += nbytes
        self.allocation_count += 1

    def release(self, nbytes: int) -> None:
        self.resident_bytes -= nbytes
        self.cumulative_released_bytes += nbytes
        self.release_count += 1
        if self.resident_bytes < 0:
            raise AssertionError("semantic branch accounting became negative")

    def descriptor(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class SemanticBranch:
    identity: BranchIdentity
    compatibility_identity: SemanticSnapshotIdentity
    position: int
    _handle: SemanticCacheHandle | None = field(repr=False)
    _accounting: _BranchAccounting = field(repr=False)

    @property
    def released(self) -> bool:
        return self._handle is None

    @property
    def cache(self) -> list[object]:
        if self._handle is None:
            raise SemanticBranchError("semantic branch has been released")
        return self._handle.cache

    @property
    def state_sha256(self) -> str:
        return semantic_cache_digest(self.cache)

    @property
    def resident_bytes(self) -> int:
        return semantic_cache_resident_bytes(self.cache) if not self.released else 0

    def component_generation_tags(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (layer, self.identity.branch_id, self.identity.generation)
            for layer in range(len(self.cache))
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": SEMANTIC_BRANCH_SCHEMA,
            "identity": self.identity.descriptor(),
            "compatibility_identity_namespace_sha256": (
                self.compatibility_identity.namespace_sha256
            ),
            "position": self.position,
            "resident_bytes": self.resident_bytes,
            "component_generation_tags": [
                list(row) for row in self.component_generation_tags()
            ],
            "released": self.released,
        }


class SemanticBranchManager:
    """Own isolated live branches and atomically select one execution state."""

    def __init__(self) -> None:
        self._branches: dict[int, SemanticBranch] = {}
        self._history: dict[int, _BranchAccounting] = {}
        self._snapshot_lineage: dict[str, BranchSnapshotLineage] = {}
        self._snapshot_bytes_by_lineage: dict[int, int] = {}
        self._next_branch_id = 1
        self._active_branch_id: int | None = None
        self._create_count = 0
        self._delete_count = 0
        self._switch_count = 0
        self._restore_count = 0
        self._peak_resident_bytes = 0

    def _get(self, branch_id: int) -> SemanticBranch:
        branch_id = _branch_id(branch_id)
        try:
            return self._branches[branch_id]
        except KeyError as error:
            raise SemanticBranchError(f"unknown semantic branch: {branch_id}") from error

    def _total_resident(self) -> int:
        return sum(branch.resident_bytes for branch in self._branches.values())

    def _observe_peak(self) -> None:
        self._peak_resident_bytes = max(
            self._peak_resident_bytes, self._total_resident()
        )

    @staticmethod
    def _synchronize_branch_accounting(branch: SemanticBranch) -> int:
        current = branch.resident_bytes
        recorded = branch._accounting.resident_bytes
        if current > recorded:
            branch._accounting.allocate(current - recorded)
        elif current < recorded:
            branch._accounting.release(recorded - current)
        return current

    def fork(
        self,
        snapshot: HybridSemanticPrefixSnapshot,
        *,
        expected_identity: SemanticSnapshotIdentity,
        branch_id: int | None = None,
        clone_entry: Callable | None = None,
    ) -> SemanticBranch:
        candidate_id = self._next_branch_id if branch_id is None else _branch_id(branch_id)
        if candidate_id in self._branches or candidate_id in self._history:
            raise SemanticBranchError(f"duplicate semantic branch: {candidate_id}")
        replacement = prepare_snapshot_restore(
            snapshot,
            expected_identity=expected_identity,
            clone_entry=clone_entry,
        )
        if semantic_cache_storage_alias_count(replacement, snapshot._cache):
            raise SemanticBranchError("fork retained mutable snapshot storage aliases")
        for existing in self._branches.values():
            if semantic_cache_storage_alias_count(replacement, existing.cache):
                raise SemanticBranchError("fork retained mutable cross-branch aliases")
        nbytes = semantic_cache_resident_bytes(replacement)
        accounting = _BranchAccounting()
        accounting.allocate(nbytes)
        branch = SemanticBranch(
            identity=BranchIdentity(candidate_id, snapshot.snapshot_id),
            compatibility_identity=snapshot.identity,
            position=snapshot.boundary.absolute_token_position,
            _handle=SemanticCacheHandle(replacement),
            _accounting=accounting,
        )
        # Publish only after clone, compatibility, and alias validation pass.
        self._branches[candidate_id] = branch
        self._history[candidate_id] = accounting
        self._next_branch_id = max(self._next_branch_id, candidate_id + 1)
        self._create_count += 1
        self._observe_peak()
        return branch

    def branch(self, branch_id: int) -> SemanticBranch:
        return self._get(branch_id)

    @property
    def active_branch_id(self) -> int | None:
        return self._active_branch_id

    @property
    def active_branch(self) -> SemanticBranch:
        if self._active_branch_id is None:
            raise SemanticBranchError("no semantic branch is active")
        return self._get(self._active_branch_id)

    @property
    def active_cache(self) -> list[object]:
        return self.active_branch.cache

    def activate(
        self, branch_id: int, *, validator: Callable[[SemanticBranch], None] | None = None
    ) -> SemanticBranch:
        target = self._get(branch_id)
        before = self._active_branch_id
        try:
            inspect_semantic_boundary(
                target.cache,
                absolute_token_position=target.position,
                materialization_epoch=(
                    target.position // MATERIALIZATION_INTERVAL_TOKENS
                ),
            )
            if validator is not None:
                validator(target)
        except Exception:
            if self._active_branch_id != before:
                raise AssertionError("branch activation mutated before validation")
            raise
        self._active_branch_id = target.identity.branch_id
        self._switch_count += 1
        return target

    def deactivate(self) -> None:
        if self._active_branch_id is not None:
            self._active_branch_id = None
            self._switch_count += 1

    def commit_active_position(self, absolute_token_position: int) -> None:
        target = self.active_branch
        position = _position(absolute_token_position)
        if position < target.position:
            raise SemanticBranchError("branch position cannot move backwards without restore")
        inspect_semantic_boundary(
            target.cache,
            absolute_token_position=position,
            materialization_epoch=position // MATERIALIZATION_INTERVAL_TOKENS,
        )
        target.position = position
        self._synchronize_branch_accounting(target)
        self._observe_peak()

    def restore_into_branch(
        self,
        branch_id: int,
        snapshot: HybridSemanticPrefixSnapshot,
        *,
        expected_identity: SemanticSnapshotIdentity,
        rollback_tokens: int | None = None,
        clone_entry: Callable | None = None,
    ) -> SemanticBranch:
        target = self._get(branch_id)
        if rollback_tokens is not None:
            if (
                isinstance(rollback_tokens, bool)
                or not isinstance(rollback_tokens, int)
                or not 1 <= rollback_tokens <= KDA_ROLLBACK_WINDOW
            ):
                raise SemanticBranchError(
                    f"branch rollback must be within [1, {KDA_ROLLBACK_WINDOW}]"
                )
            if target.position - snapshot.boundary.absolute_token_position != rollback_tokens:
                raise SemanticBranchError("branch rollback snapshot boundary mismatch")
        replacement = prepare_snapshot_restore(
            snapshot,
            expected_identity=expected_identity,
            clone_entry=clone_entry,
        )
        if semantic_cache_storage_alias_count(replacement, snapshot._cache):
            raise SemanticBranchError("branch restore retained snapshot storage aliases")
        for other_id, other in self._branches.items():
            if other_id != branch_id and semantic_cache_storage_alias_count(
                replacement, other.cache
            ):
                raise SemanticBranchError("branch restore retained cross-branch aliases")
        previous_bytes = self._synchronize_branch_accounting(target)
        replacement_bytes = semantic_cache_resident_bytes(replacement)
        replacement_handle = SemanticCacheHandle(replacement)
        next_identity = replace(
            target.identity,
            parent_snapshot_id=snapshot.snapshot_id,
            generation=target.identity.generation + 1,
        )
        # The only mutation point follows complete validation and owned cloning.
        target._handle = replacement_handle
        target.identity = next_identity
        target.compatibility_identity = snapshot.identity
        target.position = snapshot.boundary.absolute_token_position
        target._accounting.release(previous_bytes)
        target._accounting.allocate(replacement_bytes)
        self._restore_count += 1
        self._observe_peak()
        return target

    def capture_branch_snapshot(
        self,
        branch_id: int,
        store: SemanticSnapshotStore,
        *,
        snapshot_id: str,
        identity: SemanticSnapshotIdentity,
        rollback_epoch: int = 0,
        clone_entry: Callable | None = None,
    ) -> HybridSemanticPrefixSnapshot:
        branch = self._get(branch_id)
        snapshot = store.capture(
            branch._handle,
            snapshot_id=snapshot_id,
            identity=identity,
            absolute_token_position=branch.position,
            materialization_epoch=(
                branch.position // MATERIALIZATION_INTERVAL_TOKENS
            ),
            rollback_epoch=rollback_epoch,
            clone_entry=clone_entry,
        )
        lineage = BranchSnapshotLineage(
            snapshot_id=snapshot_id,
            source_branch_id=branch.identity.branch_id,
            source_branch_generation=branch.identity.generation,
            parent_snapshot_id=branch.identity.parent_snapshot_id,
            absolute_token_position=branch.position,
        )
        self._snapshot_lineage[snapshot_id] = lineage
        self._snapshot_bytes_by_lineage[branch.identity.branch_id] = (
            self._snapshot_bytes_by_lineage.get(branch.identity.branch_id, 0)
            + snapshot.resident_bytes
        )
        return snapshot

    def snapshot_lineage(self, snapshot_id: str) -> BranchSnapshotLineage:
        try:
            return self._snapshot_lineage[snapshot_id]
        except KeyError as error:
            raise SemanticBranchError("snapshot has no semantic branch lineage") from error

    def delete_branch_snapshot(
        self, store: SemanticSnapshotStore, snapshot_id: str
    ) -> None:
        lineage = self.snapshot_lineage(snapshot_id)
        snapshot = store.get(snapshot_id)
        store.delete(snapshot_id)
        current = self._snapshot_bytes_by_lineage[lineage.source_branch_id]
        remaining = current - snapshot.resident_bytes
        if remaining < 0:
            raise AssertionError("branch snapshot lineage accounting became negative")
        if remaining:
            self._snapshot_bytes_by_lineage[lineage.source_branch_id] = remaining
        else:
            del self._snapshot_bytes_by_lineage[lineage.source_branch_id]
        del self._snapshot_lineage[snapshot_id]

    def delete_branch(self, branch_id: int) -> None:
        target = self._get(branch_id)
        if self._active_branch_id == branch_id:
            raise SemanticBranchError("active semantic branch must be switched before delete")
        nbytes = self._synchronize_branch_accounting(target)
        target._accounting.release(nbytes)
        target._handle = None
        del self._branches[branch_id]
        self._delete_count += 1

    def accounting(self) -> dict[str, object]:
        for branch in self._branches.values():
            self._synchronize_branch_accounting(branch)
        by_branch = {}
        for branch_id, accounting in sorted(self._history.items()):
            live = self._branches.get(branch_id)
            row = accounting.descriptor()
            row.update(
                {
                    "position": live.position if live is not None else None,
                    "generation": (
                        live.identity.generation if live is not None else None
                    ),
                    "parent_snapshot_id": (
                        live.identity.parent_snapshot_id if live is not None else None
                    ),
                }
            )
            by_branch[str(branch_id)] = row
        resident = self._total_resident()
        mixed_component_generation_count = sum(
            1
            for branch in self._branches.values()
            for _, branch_id, generation in branch.component_generation_tags()
            if branch_id != branch.identity.branch_id
            or generation != branch.identity.generation
        )
        return {
            "schema": SEMANTIC_BRANCH_SCHEMA,
            "lifecycle": SEMANTIC_BRANCH_LIFECYCLE,
            "branch_count": len(self._branches),
            "active_branch_id": self._active_branch_id,
            "resident_bytes": resident,
            "peak_bytes": self._peak_resident_bytes,
            "cumulative_allocated_bytes": sum(
                row.cumulative_allocated_bytes for row in self._history.values()
            ),
            "cumulative_released_bytes": sum(
                row.cumulative_released_bytes for row in self._history.values()
            ),
            "branch_create_count": self._create_count,
            "branch_delete_count": self._delete_count,
            "branch_switch_count": self._switch_count,
            "branch_restore_count": self._restore_count,
            "mixed_component_generation_count": mixed_component_generation_count,
            "resident_bytes_by_branch": {
                key: row["resident_bytes"] for key, row in by_branch.items()
            },
            "peak_bytes_by_branch": {
                key: row["peak_bytes"] for key, row in by_branch.items()
            },
            "cumulative_allocated_by_branch": {
                key: row["cumulative_allocated_bytes"]
                for key, row in by_branch.items()
            },
            "snapshot_bytes_by_lineage": {
                str(key): value
                for key, value in sorted(self._snapshot_bytes_by_lineage.items())
            },
            "by_branch": by_branch,
            "anonymous_allocation_count": 0,
        }
