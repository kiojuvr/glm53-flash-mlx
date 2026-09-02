"""Logical cache lifecycle, retention, and accounting contracts.

This module deliberately does not allocate model cache tensors or implement a
draft model.  It defines the ownership and eviction domains that future cache
allocators must preserve, and provides a deterministic byte-storage simulator
for policy and accounting fixtures.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class CacheLifecycle(str, Enum):
    TARGET_PREFIX = "target-prefix"
    ACTIVE_RECURRENT = "active-recurrent"
    SNAPSHOT_STATE = "snapshot-state"
    DRAFT_TRANSIENT = "draft-transient"


class RetentionPolicy(str, Enum):
    LONG_REUSE = "long-reuse"
    PINNED_REQUEST = "pinned-request"
    SNAPSHOT_OWNED = "snapshot-owned"
    SHORT_BOUNDED = "short-bounded"


class EvictionReason(str, Enum):
    TARGET_PRESSURE = "target-pressure"
    DRAFT_PRESSURE = "draft-pressure"
    APC_POLICY = "apc-policy"
    REQUEST_COMPLETE = "request-complete"
    EXPLICIT_RELEASE = "explicit-release"


class CacheLifecycleError(ValueError):
    """Raised when a cache allocation violates a lifecycle contract."""


@dataclass(frozen=True)
class LifecycleContract:
    contents: tuple[str, ...]
    lifetime: str
    eviction_domain: str


LIFECYCLE_CONTRACTS = {
    CacheLifecycle.TARGET_PREFIX: LifecycleContract(
        contents=("dsa-latent-kv", "indexpool-prefix-state"),
        lifetime="prefix-identity",
        eviction_domain="target-pressure-only",
    ),
    CacheLifecycle.ACTIVE_RECURRENT: LifecycleContract(
        contents=("kda-conv-state", "kda-recurrent-state"),
        lifetime="active-request",
        eviction_domain="request-complete-only",
    ),
    CacheLifecycle.SNAPSHOT_STATE: LifecycleContract(
        contents=("apc-kda-state", "apc-indexpool-state", "apc-kv-state"),
        lifetime="apc-snapshot",
        eviction_domain="apc-policy-only",
    ),
    CacheLifecycle.DRAFT_TRANSIENT: LifecycleContract(
        contents=("draft-sliding-window-kv", "draft-mutable-state"),
        lifetime="draft-session",
        eviction_domain="draft-pressure-only",
    ),
}


_RETENTION_CONTRACT = {
    CacheLifecycle.TARGET_PREFIX: {RetentionPolicy.LONG_REUSE},
    CacheLifecycle.ACTIVE_RECURRENT: {RetentionPolicy.PINNED_REQUEST},
    CacheLifecycle.SNAPSHOT_STATE: {RetentionPolicy.SNAPSHOT_OWNED},
    CacheLifecycle.DRAFT_TRANSIENT: {RetentionPolicy.SHORT_BOUNDED},
}

_EVICTION_CONTRACT = {
    CacheLifecycle.TARGET_PREFIX: {
        EvictionReason.TARGET_PRESSURE,
        EvictionReason.EXPLICIT_RELEASE,
    },
    CacheLifecycle.ACTIVE_RECURRENT: {
        EvictionReason.REQUEST_COMPLETE,
        EvictionReason.EXPLICIT_RELEASE,
    },
    CacheLifecycle.SNAPSHOT_STATE: {
        EvictionReason.APC_POLICY,
        EvictionReason.EXPLICIT_RELEASE,
    },
    CacheLifecycle.DRAFT_TRANSIENT: {
        EvictionReason.DRAFT_PRESSURE,
        EvictionReason.EXPLICIT_RELEASE,
    },
}


@dataclass(frozen=True)
class PrefixIdentity:
    """Complete reuse identity; token equality alone is insufficient."""

    model_revision: str
    checkpoint_fingerprint: str
    backend_policy: str
    attention_cache_abi: str
    kda_state_abi: str
    indexpool_abi: str
    prefix_token_sha256: str

    def __post_init__(self) -> None:
        missing = [
            name
            for name, value in self.canonical_descriptor().items()
            if not isinstance(value, str) or not value
        ]
        if missing:
            raise CacheLifecycleError(
                f"prefix identity fields must be non-empty strings: {missing}"
            )

    def canonical_descriptor(self) -> dict[str, str]:
        return {
            "model_revision": self.model_revision,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "backend_policy": self.backend_policy,
            "attention_cache_abi": self.attention_cache_abi,
            "kda_state_abi": self.kda_state_abi,
            "indexpool_abi": self.indexpool_abi,
            "prefix_token_sha256": self.prefix_token_sha256,
        }

    @property
    def namespace_sha256(self) -> str:
        payload = json.dumps(
            self.canonical_descriptor(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass
class CacheStorage:
    """Physical byte storage, intentionally independent of lifecycle metadata."""

    storage_id: int
    _payload: bytearray

    @classmethod
    def owned_copy(cls, storage_id: int, payload) -> "CacheStorage":
        # bytearray always severs aliases to bytes, memoryview, and bytearray.
        return cls(storage_id=storage_id, _payload=bytearray(payload))

    @property
    def nbytes(self) -> int:
        return len(self._payload)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self._payload).hexdigest()

    def read(self) -> bytes:
        return bytes(self._payload)

    def overwrite(self, payload) -> None:
        replacement = bytearray(payload)
        if len(replacement) != len(self._payload):
            raise CacheLifecycleError(
                "in-place cache state mutation cannot change physical size"
            )
        self._payload[:] = replacement


@dataclass
class CacheEntry:
    entry_id: str
    storage: CacheStorage
    lifecycle: CacheLifecycle
    retention: RetentionPolicy
    owner_id: str
    prefix_identity: PrefixIdentity | None = None

    @property
    def nbytes(self) -> int:
        return self.storage.nbytes

    @property
    def state_digest(self) -> str:
        return self.storage.digest


@dataclass
class _ClassAccounting:
    resident_bytes: int = 0
    peak_bytes: int = 0
    allocation_count: int = 0
    eviction_count: int = 0
    cumulative_allocated_bytes: int = 0

    def allocate(self, nbytes: int) -> None:
        self.resident_bytes += nbytes
        self.peak_bytes = max(self.peak_bytes, self.resident_bytes)
        self.allocation_count += 1
        self.cumulative_allocated_bytes += nbytes

    def evict(self, nbytes: int) -> None:
        self.resident_bytes -= nbytes
        self.eviction_count += 1
        if self.resident_bytes < 0:
            raise AssertionError("cache lifecycle accounting became negative")

    def replace(self, previous: int, current: int) -> None:
        self.resident_bytes += current - previous
        self.peak_bytes = max(self.peak_bytes, self.resident_bytes)
        self.allocation_count += 1
        self.cumulative_allocated_bytes += current

    def snapshot(self) -> dict[str, int]:
        return {
            "resident_bytes": self.resident_bytes,
            "peak_bytes": self.peak_bytes,
            "allocation_count": self.allocation_count,
            "eviction_count": self.eviction_count,
            "cumulative_allocated_bytes": self.cumulative_allocated_bytes,
        }


class CacheLifecycleManager:
    """Logical policy simulator with isolated eviction/accounting domains."""

    def __init__(self, budgets: Mapping[CacheLifecycle, int]):
        if set(budgets) != set(CacheLifecycle):
            raise CacheLifecycleError(
                "budgets must explicitly name all cache lifecycle classes"
            )
        if any(not isinstance(key, CacheLifecycle) for key in budgets):
            raise CacheLifecycleError("cache lifecycle must be explicit")
        if any(int(value) < 0 for value in budgets.values()):
            raise CacheLifecycleError("cache budgets must be non-negative")
        self._budgets = {key: int(value) for key, value in budgets.items()}
        self._entries: dict[str, CacheEntry] = {}
        self._lru = {
            CacheLifecycle.TARGET_PREFIX: OrderedDict(),
            CacheLifecycle.SNAPSHOT_STATE: OrderedDict(),
            CacheLifecycle.DRAFT_TRANSIENT: OrderedDict(),
        }
        self._prefix_index: dict[str, str] = {}
        self._accounting = {
            lifecycle: _ClassAccounting() for lifecycle in CacheLifecycle
        }
        self._next_storage_id = 1

    @property
    def entries(self) -> tuple[CacheEntry, ...]:
        return tuple(self._entries.values())

    def get(self, entry_id: str) -> CacheEntry:
        try:
            return self._entries[entry_id]
        except KeyError as error:
            raise CacheLifecycleError(f"unknown cache entry: {entry_id}") from error

    def _new_storage(self, payload) -> CacheStorage:
        storage = CacheStorage.owned_copy(self._next_storage_id, payload)
        self._next_storage_id += 1
        return storage

    def _validate_contract(
        self,
        lifecycle: CacheLifecycle,
        retention: RetentionPolicy,
        owner_id: str,
        prefix_identity: PrefixIdentity | None,
    ) -> None:
        if not isinstance(lifecycle, CacheLifecycle):
            raise CacheLifecycleError("cache lifecycle cannot be inferred")
        if not isinstance(retention, RetentionPolicy):
            raise CacheLifecycleError("retention policy cannot be inferred")
        if retention not in _RETENTION_CONTRACT[lifecycle]:
            allowed = ", ".join(
                sorted(policy.value for policy in _RETENTION_CONTRACT[lifecycle])
            )
            raise CacheLifecycleError(
                f"{lifecycle.value} requires one of: {allowed}"
            )
        if not owner_id:
            raise CacheLifecycleError("cache entries require an explicit owner")
        if lifecycle is CacheLifecycle.TARGET_PREFIX and prefix_identity is None:
            raise CacheLifecycleError("target prefix entries require a reuse identity")
        if lifecycle is not CacheLifecycle.TARGET_PREFIX and prefix_identity is not None:
            raise CacheLifecycleError(
                "prefix identity is only valid for target prefix entries"
            )

    def _evict_to_fit(self, lifecycle: CacheLifecycle, incoming: int) -> None:
        budget = self._budgets[lifecycle]
        if incoming > budget:
            raise CacheLifecycleError(
                f"{lifecycle.value} allocation exceeds its dedicated budget"
            )
        accounting = self._accounting[lifecycle]
        reason = {
            CacheLifecycle.TARGET_PREFIX: EvictionReason.TARGET_PRESSURE,
            CacheLifecycle.SNAPSHOT_STATE: EvictionReason.APC_POLICY,
            CacheLifecycle.DRAFT_TRANSIENT: EvictionReason.DRAFT_PRESSURE,
        }.get(lifecycle)
        while accounting.resident_bytes + incoming > budget:
            if reason is None:
                raise CacheLifecycleError(
                    "active recurrent state is pinned for the request lifetime"
                )
            lru = self._lru[lifecycle]
            if not lru:
                raise CacheLifecycleError(
                    f"{lifecycle.value} has no evictable allocation"
                )
            entry_id = next(iter(lru))
            self.evict(entry_id, reason=reason)

    def allocate(
        self,
        *,
        entry_id: str,
        payload,
        lifecycle: CacheLifecycle,
        retention: RetentionPolicy,
        owner_id: str,
        prefix_identity: PrefixIdentity | None = None,
    ) -> CacheEntry:
        if not entry_id or entry_id in self._entries:
            raise CacheLifecycleError(f"duplicate or empty cache entry: {entry_id!r}")
        self._validate_contract(lifecycle, retention, owner_id, prefix_identity)
        if prefix_identity is not None:
            namespace = prefix_identity.namespace_sha256
            if namespace in self._prefix_index:
                raise CacheLifecycleError("duplicate target prefix identity")
        try:
            payload_bytes = memoryview(payload).nbytes
        except TypeError as error:
            raise CacheLifecycleError("cache payload must be bytes-like") from error
        self._evict_to_fit(lifecycle, payload_bytes)
        storage = self._new_storage(payload)
        entry = CacheEntry(
            entry_id=entry_id,
            storage=storage,
            lifecycle=lifecycle,
            retention=retention,
            owner_id=owner_id,
            prefix_identity=prefix_identity,
        )
        self._entries[entry_id] = entry
        if lifecycle in self._lru:
            self._lru[lifecycle][entry_id] = None
        if prefix_identity is not None:
            self._prefix_index[prefix_identity.namespace_sha256] = entry_id
        self._accounting[lifecycle].allocate(storage.nbytes)
        return entry

    def evict(self, entry_id: str, *, reason: EvictionReason) -> None:
        entry = self.get(entry_id)
        if not isinstance(reason, EvictionReason):
            raise CacheLifecycleError("eviction reason must be explicit")
        if reason not in _EVICTION_CONTRACT[entry.lifecycle]:
            raise CacheLifecycleError(
                f"{reason.value} cannot evict {entry.lifecycle.value}"
            )
        del self._entries[entry_id]
        if entry.lifecycle in self._lru:
            self._lru[entry.lifecycle].pop(entry_id, None)
        if entry.prefix_identity is not None:
            self._prefix_index.pop(entry.prefix_identity.namespace_sha256, None)
        self._accounting[entry.lifecycle].evict(entry.nbytes)

    def lookup_target_prefix(
        self, identity: PrefixIdentity
    ) -> CacheEntry | None:
        entry_id = self._prefix_index.get(identity.namespace_sha256)
        if entry_id is None:
            return None
        entry = self._entries[entry_id]
        self._lru[CacheLifecycle.TARGET_PREFIX].move_to_end(entry_id)
        return entry

    def mutate_active(self, entry_id: str, payload) -> None:
        entry = self.get(entry_id)
        if entry.lifecycle is not CacheLifecycle.ACTIVE_RECURRENT:
            raise CacheLifecycleError("only active recurrent state is mutable here")
        entry.storage.overwrite(payload)

    def capture_snapshot(
        self,
        source_entry_id: str,
        *,
        snapshot_entry_id: str,
        snapshot_owner_id: str,
    ) -> CacheEntry:
        source = self.get(source_entry_id)
        snapshot = self.allocate(
            entry_id=snapshot_entry_id,
            payload=source.storage.read(),
            lifecycle=CacheLifecycle.SNAPSHOT_STATE,
            retention=RetentionPolicy.SNAPSHOT_OWNED,
            owner_id=snapshot_owner_id,
        )
        if snapshot.storage.storage_id == source.storage.storage_id:
            raise AssertionError("snapshot storage borrowed active state")
        return snapshot

    def restore_snapshot(self, snapshot_entry_id: str, active_entry_id: str) -> None:
        snapshot = self.get(snapshot_entry_id)
        active = self.get(active_entry_id)
        if snapshot.lifecycle is not CacheLifecycle.SNAPSHOT_STATE:
            raise CacheLifecycleError("restore source is not snapshot-owned state")
        if active.lifecycle is not CacheLifecycle.ACTIVE_RECURRENT:
            raise CacheLifecycleError("snapshot restore target is not active state")
        previous = active.nbytes
        replacement_bytes = snapshot.nbytes
        resulting_resident = (
            self._accounting[CacheLifecycle.ACTIVE_RECURRENT].resident_bytes
            - previous
            + replacement_bytes
        )
        if resulting_resident > self._budgets[CacheLifecycle.ACTIVE_RECURRENT]:
            raise CacheLifecycleError("restored active state exceeds request budget")
        replacement = self._new_storage(snapshot.storage.read())
        active.storage = replacement
        self._accounting[CacheLifecycle.ACTIVE_RECURRENT].replace(
            previous, replacement.nbytes
        )
        if active.storage.storage_id == snapshot.storage.storage_id:
            raise AssertionError("restored active state borrowed snapshot storage")

    def lru_entry_ids(self, lifecycle: CacheLifecycle) -> tuple[str, ...]:
        if lifecycle is CacheLifecycle.ACTIVE_RECURRENT:
            return ()
        return tuple(self._lru[lifecycle])

    def accounting_snapshot(self) -> dict[str, dict[str, int]]:
        return {
            lifecycle.value: self._accounting[lifecycle].snapshot()
            for lifecycle in CacheLifecycle
        }

    def budget_snapshot(self) -> dict[str, int]:
        return {
            lifecycle.value: self._budgets[lifecycle]
            for lifecycle in CacheLifecycle
        }

    def audit(self) -> dict[str, object]:
        actual = {lifecycle: 0 for lifecycle in CacheLifecycle}
        for entry in self._entries.values():
            actual[entry.lifecycle] += entry.nbytes
        accounting_exact = all(
            actual[lifecycle] == self._accounting[lifecycle].resident_bytes
            for lifecycle in CacheLifecycle
        )
        within_budget = all(
            actual[lifecycle] <= self._budgets[lifecycle]
            for lifecycle in CacheLifecycle
        )
        targets = [
            entry
            for entry in self._entries.values()
            if entry.lifecycle is CacheLifecycle.TARGET_PREFIX
        ]
        prefix_index_exact = len(targets) == len(self._prefix_index) and all(
            entry.prefix_identity is not None
            and self._prefix_index.get(entry.prefix_identity.namespace_sha256)
            == entry.entry_id
            for entry in targets
        )
        active_absent_from_lru = all(
            entry_id not in lru
            for entry_id, entry in self._entries.items()
            if entry.lifecycle is CacheLifecycle.ACTIVE_RECURRENT
            for lru in self._lru.values()
        )
        no_anonymous_entries = all(
            entry.entry_id
            and entry.owner_id
            and entry.retention in _RETENTION_CONTRACT[entry.lifecycle]
            for entry in self._entries.values()
        )
        storage_ids = [entry.storage.storage_id for entry in self._entries.values()]
        return {
            "accounting_exact": accounting_exact,
            "within_dedicated_budgets": within_budget,
            "prefix_index_exact": prefix_index_exact,
            "active_recurrent_absent_from_all_lru": active_absent_from_lru,
            "no_anonymous_entries": no_anonymous_entries,
            "unique_physical_storage": len(storage_ids) == len(set(storage_ids)),
            "anonymous_entry_count": sum(
                not entry.entry_id or not entry.owner_id
                for entry in self._entries.values()
            ),
        }
