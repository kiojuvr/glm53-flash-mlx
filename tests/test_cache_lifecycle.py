from dataclasses import replace

import pytest

from glm53_flash_mlx.cache_lifecycle import (
    CacheLifecycle,
    CacheLifecycleError,
    CacheLifecycleManager,
    EvictionReason,
    LIFECYCLE_CONTRACTS,
    PrefixIdentity,
    RetentionPolicy,
)


def _budgets(**overrides):
    values = {
        CacheLifecycle.TARGET_PREFIX: 200,
        CacheLifecycle.ACTIVE_RECURRENT: 256,
        CacheLifecycle.SNAPSHOT_STATE: 256,
        CacheLifecycle.DRAFT_TRANSIENT: 32,
    }
    values.update(overrides)
    return values


def _identity(**overrides):
    values = {
        "model_revision": "04c4e9e",
        "checkpoint_fingerprint": "checkpoint-a",
        "backend_policy": "packed-decode+compact-nope-dsa",
        "attention_cache_abi": "glm53-nope-dsa-v4",
        "kda_state_abi": "glm53-kda-state-v1",
        "indexpool_abi": "compact-indexpool-v4",
        "prefix_token_sha256": "tokens-a",
    }
    values.update(overrides)
    return PrefixIdentity(**values)


def _allocate_target(manager, *, entry_id="prefix-a", payload=b"T" * 100):
    return manager.allocate(
        entry_id=entry_id,
        payload=payload,
        lifecycle=CacheLifecycle.TARGET_PREFIX,
        retention=RetentionPolicy.LONG_REUSE,
        owner_id="prefix-owner-a",
        prefix_identity=_identity(prefix_token_sha256=entry_id),
    )


def _allocate_active(manager, *, entry_id="active-a", payload=b"K" * 64):
    return manager.allocate(
        entry_id=entry_id,
        payload=payload,
        lifecycle=CacheLifecycle.ACTIVE_RECURRENT,
        retention=RetentionPolicy.PINNED_REQUEST,
        owner_id="request-a",
    )


def _audit_passes(manager):
    audit = manager.audit()
    return audit["anonymous_entry_count"] == 0 and all(
        value for key, value in audit.items() if key != "anonymous_entry_count"
    )


def test_lifecycle_contracts_name_authoritative_state_contents():
    assert LIFECYCLE_CONTRACTS[CacheLifecycle.TARGET_PREFIX].contents == (
        "dsa-latent-kv",
        "indexpool-prefix-state",
    )
    assert LIFECYCLE_CONTRACTS[CacheLifecycle.ACTIVE_RECURRENT].contents == (
        "kda-conv-state",
        "kda-recurrent-state",
    )
    assert LIFECYCLE_CONTRACTS[CacheLifecycle.SNAPSHOT_STATE].contents == (
        "apc-kda-state",
        "apc-indexpool-state",
        "apc-kv-state",
    )
    assert LIFECYCLE_CONTRACTS[CacheLifecycle.DRAFT_TRANSIENT].contents == (
        "draft-sliding-window-kv",
        "draft-mutable-state",
    )


def test_draft_pressure_isolated_from_target_prefix_retention():
    manager = CacheLifecycleManager(_budgets())
    target = _allocate_target(manager)
    target_storage_id = target.storage.storage_id
    target_digest = target.state_digest

    for index in range(4096):
        manager.allocate(
            entry_id=f"draft-{index}",
            payload=bytes([index % 251]),
            lifecycle=CacheLifecycle.DRAFT_TRANSIENT,
            retention=RetentionPolicy.SHORT_BOUNDED,
            owner_id="draft-session",
        )

    target_after = manager.get("prefix-a")
    metrics = manager.accounting_snapshot()
    assert target_after.storage.storage_id == target_storage_id
    assert target_after.state_digest == target_digest
    assert metrics["target-prefix"]["eviction_count"] == 0
    assert metrics["draft-transient"]["resident_bytes"] <= 32
    assert metrics["draft-transient"]["eviction_count"] == 4096 - 32
    assert metrics["draft-transient"]["cumulative_allocated_bytes"] == 4096
    assert _audit_passes(manager)


def test_active_recurrent_state_is_pinned_and_absent_from_prefix_lru():
    manager = CacheLifecycleManager(_budgets())
    active = _allocate_active(manager)
    active_storage_id = active.storage.storage_id
    active_digest = active.state_digest
    _allocate_target(manager, entry_id="prefix-0", payload=b"0" * 100)
    _allocate_target(manager, entry_id="prefix-1", payload=b"1" * 100)
    _allocate_target(manager, entry_id="prefix-2", payload=b"2" * 100)
    for index in range(128):
        manager.allocate(
            entry_id=f"draft-{index}",
            payload=b"D",
            lifecycle=CacheLifecycle.DRAFT_TRANSIENT,
            retention=RetentionPolicy.SHORT_BOUNDED,
            owner_id="draft-session",
        )

    active_after = manager.get("active-a")
    assert active_after.storage.storage_id == active_storage_id
    assert active_after.state_digest == active_digest
    assert manager.lru_entry_ids(CacheLifecycle.ACTIVE_RECURRENT) == ()
    with pytest.raises(CacheLifecycleError, match="cannot evict"):
        manager.evict("active-a", reason=EvictionReason.DRAFT_PRESSURE)
    with pytest.raises(CacheLifecycleError, match="cannot evict"):
        manager.evict("active-a", reason=EvictionReason.TARGET_PRESSURE)
    assert manager.accounting_snapshot()["active-recurrent"]["eviction_count"] == 0
    assert _audit_passes(manager)


def test_snapshot_is_owned_and_independent_of_active_prefix_and_draft_pressure():
    manager = CacheLifecycleManager(_budgets())
    active = _allocate_active(manager, payload=b"S0" * 32)
    snapshot = manager.capture_snapshot(
        "active-a", snapshot_entry_id="snapshot-a", snapshot_owner_id="apc-a"
    )
    snapshot_digest = snapshot.state_digest
    assert snapshot.storage.storage_id != active.storage.storage_id

    manager.mutate_active("active-a", b"S1" * 32)
    active_mutated_storage_id = manager.get("active-a").storage.storage_id
    _allocate_target(manager, entry_id="prefix-0", payload=b"0" * 100)
    _allocate_target(manager, entry_id="prefix-1", payload=b"1" * 100)
    _allocate_target(manager, entry_id="prefix-2", payload=b"2" * 100)
    for index in range(128):
        manager.allocate(
            entry_id=f"draft-{index}",
            payload=b"D",
            lifecycle=CacheLifecycle.DRAFT_TRANSIENT,
            retention=RetentionPolicy.SHORT_BOUNDED,
            owner_id="draft-session",
        )
    assert manager.get("active-a").storage.storage_id == active_mutated_storage_id
    assert manager.get("snapshot-a").state_digest == snapshot_digest

    manager.restore_snapshot("snapshot-a", "active-a")
    restored = manager.get("active-a")
    assert restored.state_digest == snapshot_digest
    assert restored.storage.storage_id != manager.get("snapshot-a").storage.storage_id
    manager.mutate_active("active-a", b"S2" * 32)
    assert manager.get("snapshot-a").state_digest == snapshot_digest
    manager.restore_snapshot("snapshot-a", "active-a")
    assert manager.get("active-a").state_digest == snapshot_digest

    metrics = manager.accounting_snapshot()
    assert metrics["snapshot-state"]["resident_bytes"] == 64
    assert metrics["snapshot-state"]["eviction_count"] == 0
    assert metrics["active-recurrent"]["resident_bytes"] == 64
    assert _audit_passes(manager)


def test_snapshot_and_active_allocations_are_owned_copies_of_caller_storage():
    manager = CacheLifecycleManager(_budgets())
    source = bytearray(b"A" * 64)
    active = _allocate_active(manager, payload=source)
    source[:] = b"B" * 64
    assert active.storage.read() == b"A" * 64
    snapshot = manager.capture_snapshot(
        "active-a", snapshot_entry_id="snapshot-a", snapshot_owner_id="apc-a"
    )
    manager.mutate_active("active-a", b"C" * 64)
    assert snapshot.storage.read() == b"A" * 64


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_revision", "different-revision"),
        ("checkpoint_fingerprint", "different-checkpoint"),
        ("backend_policy", "direct+direct-cache"),
        ("attention_cache_abi", "different-attention-abi"),
        ("kda_state_abi", "different-kda-abi"),
        ("indexpool_abi", "different-indexpool-abi"),
        ("prefix_token_sha256", "different-tokens"),
    ],
)
def test_prefix_identity_fields_prevent_accidental_sharing(field, value):
    manager = CacheLifecycleManager(_budgets())
    identity = _identity()
    manager.allocate(
        entry_id="prefix-a",
        payload=b"P" * 32,
        lifecycle=CacheLifecycle.TARGET_PREFIX,
        retention=RetentionPolicy.LONG_REUSE,
        owner_id="prefix-owner-a",
        prefix_identity=identity,
    )
    assert manager.lookup_target_prefix(identity).entry_id == "prefix-a"
    assert manager.lookup_target_prefix(replace(identity, **{field: value})) is None


def test_lifecycle_retention_and_owner_are_never_inferred():
    manager = CacheLifecycleManager(_budgets())
    with pytest.raises(CacheLifecycleError, match="cannot be inferred"):
        manager.allocate(
            entry_id="anonymous",
            payload=b"x",
            lifecycle="target-prefix",
            retention=RetentionPolicy.LONG_REUSE,
            owner_id="owner",
            prefix_identity=_identity(),
        )
    with pytest.raises(CacheLifecycleError, match="cannot be inferred"):
        manager.allocate(
            entry_id="anonymous",
            payload=b"x",
            lifecycle=CacheLifecycle.TARGET_PREFIX,
            retention="long-reuse",
            owner_id="owner",
            prefix_identity=_identity(),
        )
    with pytest.raises(CacheLifecycleError, match="explicit owner"):
        manager.allocate(
            entry_id="anonymous",
            payload=b"x",
            lifecycle=CacheLifecycle.DRAFT_TRANSIENT,
            retention=RetentionPolicy.SHORT_BOUNDED,
            owner_id="",
        )
    with pytest.raises(CacheLifecycleError, match="requires one of: long-reuse"):
        manager.allocate(
            entry_id="wrong-policy",
            payload=b"x",
            lifecycle=CacheLifecycle.TARGET_PREFIX,
            retention=RetentionPolicy.SHORT_BOUNDED,
            owner_id="owner",
            prefix_identity=_identity(),
        )


def test_rejected_allocation_does_not_evict_or_register_storage():
    manager = CacheLifecycleManager(_budgets())
    target = _allocate_target(manager)
    baseline_metrics = manager.accounting_snapshot()
    baseline_audit = manager.audit()
    baseline_entries = tuple(
        (entry.entry_id, entry.storage.storage_id, entry.state_digest)
        for entry in manager.entries
    )

    with pytest.raises(CacheLifecycleError, match="duplicate target prefix identity"):
        manager.allocate(
            entry_id="duplicate-prefix",
            payload=b"X" * 150,
            lifecycle=CacheLifecycle.TARGET_PREFIX,
            retention=RetentionPolicy.LONG_REUSE,
            owner_id="prefix-owner-b",
            prefix_identity=target.prefix_identity,
        )
    with pytest.raises(CacheLifecycleError, match="exceeds its dedicated budget"):
        manager.allocate(
            entry_id="oversized-draft",
            payload=b"D" * 33,
            lifecycle=CacheLifecycle.DRAFT_TRANSIENT,
            retention=RetentionPolicy.SHORT_BOUNDED,
            owner_id="draft-session",
        )

    assert manager.accounting_snapshot() == baseline_metrics
    assert manager.audit() == baseline_audit
    assert tuple(
        (entry.entry_id, entry.storage.storage_id, entry.state_digest)
        for entry in manager.entries
    ) == baseline_entries


def test_eviction_domains_fail_closed():
    manager = CacheLifecycleManager(_budgets())
    target = _allocate_target(manager)
    active = _allocate_active(manager)
    snapshot = manager.capture_snapshot(
        active.entry_id,
        snapshot_entry_id="snapshot-a",
        snapshot_owner_id="apc-a",
    )
    draft = manager.allocate(
        entry_id="draft-a",
        payload=b"D",
        lifecycle=CacheLifecycle.DRAFT_TRANSIENT,
        retention=RetentionPolicy.SHORT_BOUNDED,
        owner_id="draft-a",
    )
    for entry, reason in (
        (target, EvictionReason.DRAFT_PRESSURE),
        (active, EvictionReason.APC_POLICY),
        (snapshot, EvictionReason.TARGET_PRESSURE),
        (draft, EvictionReason.TARGET_PRESSURE),
    ):
        with pytest.raises(CacheLifecycleError, match="cannot evict"):
            manager.evict(entry.entry_id, reason=reason)


def test_allocation_and_eviction_accounting_remains_exact():
    manager = CacheLifecycleManager(_budgets())
    _allocate_target(manager)
    _allocate_active(manager)
    manager.capture_snapshot(
        "active-a", snapshot_entry_id="snapshot-a", snapshot_owner_id="apc-a"
    )
    for index in range(100):
        manager.allocate(
            entry_id=f"draft-{index}",
            payload=b"D",
            lifecycle=CacheLifecycle.DRAFT_TRANSIENT,
            retention=RetentionPolicy.SHORT_BOUNDED,
            owner_id="draft-a",
        )
    manager.evict("active-a", reason=EvictionReason.REQUEST_COMPLETE)
    manager.evict("snapshot-a", reason=EvictionReason.APC_POLICY)
    metrics = manager.accounting_snapshot()
    assert metrics["target-prefix"] == {
        "resident_bytes": 100,
        "peak_bytes": 100,
        "allocation_count": 1,
        "eviction_count": 0,
        "cumulative_allocated_bytes": 100,
    }
    assert metrics["active-recurrent"]["resident_bytes"] == 0
    assert metrics["active-recurrent"]["eviction_count"] == 1
    assert metrics["snapshot-state"]["resident_bytes"] == 0
    assert metrics["snapshot-state"]["eviction_count"] == 1
    assert metrics["draft-transient"]["resident_bytes"] == 32
    assert metrics["draft-transient"]["peak_bytes"] == 32
    assert metrics["draft-transient"]["allocation_count"] == 100
    assert metrics["draft-transient"]["eviction_count"] == 68
    assert metrics["draft-transient"]["cumulative_allocated_bytes"] == 100
    assert _audit_passes(manager)
