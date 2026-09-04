from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

try:
    import mlx.core as mx
except ImportError:
    pytest.skip("MLX/Metal is unavailable", allow_module_level=True)

from glm53_flash_mlx.abi import NOPE_DSA_CACHE_ABI_COMPACT
from glm53_flash_mlx.cache_lifecycle import (
    CacheLifecycle,
    CacheLifecycleManager,
    PrefixIdentity,
    RetentionPolicy,
)
from glm53_flash_mlx.nope_cache import make_compact_nope_dsa_cache
from glm53_flash_mlx.patch import apply_runtime_patch
from glm53_flash_mlx.semantic_snapshot import (
    COMPACT_INDEXPOOL_ABI,
    KDA_STATE_ABI,
    SEMANTIC_PREFIX_SNAPSHOT_SCHEMA,
    SemanticCacheHandle,
    SemanticSnapshotError,
    SemanticSnapshotIdentity,
    SemanticSnapshotStore,
    inspect_semantic_boundary,
    semantic_cache_digest,
    semantic_cache_schema,
    semantic_component_digests,
)


def _indexer():
    apply_runtime_patch()
    from mlx_vlm.models.glm5_next.language import Glm5NextIndexer

    config = SimpleNamespace(
        hidden_size=8,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2048,
        index_kpool=4,
        index_kpool_always_select_tail=True,
        q_lora_rank=4,
    )
    mx.random.seed(121)
    value = Glm5NextIndexer(config)
    value.set_dtype(mx.bfloat16)
    return value


def _inputs(start: int, count: int):
    positions = mx.arange(start, start + count, dtype=mx.float32)[:, None]
    x = mx.sin(positions * 0.03125 + mx.arange(8)[None] * 0.0625)[None]
    qr = mx.cos(positions * 0.046875 + mx.arange(4)[None] * 0.09375)[None]
    return x.astype(mx.bfloat16), qr.astype(mx.bfloat16)


def _new_cache(tokens: int = 8):
    from mlx_vlm.models.cache import ArraysCache

    indexer = _indexer()
    left = ArraysCache(size=2)
    right = ArraysCache(size=2)
    left[0] = mx.zeros((1, 8, 2), dtype=mx.bfloat16)
    left[1] = mx.zeros((1, 2, 4, 4), dtype=mx.float32)
    right[0] = mx.ones((1, 8, 2), dtype=mx.bfloat16)
    right[1] = mx.ones((1, 2, 4, 4), dtype=mx.float32)
    dsa = make_compact_nope_dsa_cache(indexer, capacity_tokens=512)
    x, qr = _inputs(0, tokens)
    assert indexer(x, qr, None, cache=dsa[1]) is None
    latent = mx.sin(
        mx.arange(tokens, dtype=mx.float32)[:, None] * 0.015625
        + mx.arange(4, dtype=mx.float32)[None] * 0.125
    ).astype(mx.bfloat16)
    latent = latent.reshape(1, 1, tokens, 4)
    dsa[0].update_and_fetch(latent, latent)
    mx.eval(left.state, right.state, dsa.state)
    return [left, dsa, right], indexer


def _advance(cache, indexer, start: int, count: int) -> None:
    for position in range(start, start + count):
        for layer in (0, 2):
            cache[layer][0] = cache[layer][0] + mx.array(1, dtype=mx.bfloat16)
            cache[layer][1] = cache[layer][1] + mx.array(0.25, dtype=mx.float32)
        x, qr = _inputs(position, 1)
        assert indexer(x, qr, None, cache=cache[1][1]) is None
        latent = mx.full((1, 1, 1, 4), position, dtype=mx.bfloat16)
        cache[1][0].update_and_fetch(latent, latent)
    mx.eval([entry.state for entry in cache])


def _identity(**overrides) -> SemanticSnapshotIdentity:
    values = {
        "checkpoint_revision": "04c4e9e",
        "checkpoint_fingerprint": "checkpoint-fingerprint",
        "moe_backend": "packed-decode",
        "cache_backend": "compact-nope-dsa",
        "attention_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
        "kda_state_abi": KDA_STATE_ABI,
        "indexpool_abi": COMPACT_INDEXPOOL_ABI,
        "prefix_token_sha256": "prefix-token-digest",
    }
    values.update(overrides)
    return SemanticSnapshotIdentity(**values)


def _capture(store, handle, *, tokens=8, snapshot_id="snapshot-a", clone_entry=None):
    return store.capture(
        handle,
        snapshot_id=snapshot_id,
        identity=_identity(),
        absolute_token_position=tokens,
        materialization_epoch=tokens // 256,
        clone_entry=clone_entry,
    )


def _arrays(value):
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _arrays(item)


def _cache_arrays(cache):
    for entry in cache:
        yield from _arrays(entry.state)


def test_contract_names_complete_identity_boundary_and_ram_ownership():
    cache, _ = _new_cache()
    handle = SemanticCacheHandle(cache)
    store = SemanticSnapshotStore()
    snapshot = _capture(store, handle)

    descriptor = snapshot.descriptor()
    assert descriptor["schema"] == SEMANTIC_PREFIX_SNAPSHOT_SCHEMA
    assert descriptor["identity"] == _identity().descriptor()
    assert descriptor["boundary"]["absolute_token_position"] == 8
    assert descriptor["boundary"]["logical_prefix_length"] == 8
    assert descriptor["boundary"]["materialization_epoch"] == 0
    assert descriptor["boundary"]["rollback_epoch"] == 0
    assert descriptor["boundary"]["kv_logical_extents"] == [[1, 8]]
    assert descriptor["boundary"]["indexpool_logical_extents"] == [[1, 8]]
    assert descriptor["boundary"]["active_state_slots"] == [
        [0, [0, 1]],
        [2, [0, 1]],
    ]
    assert descriptor["ownership"] == {
        "lifecycle": "snapshot-state",
        "retention": "snapshot-owned",
        "tensor_ownership": "owned",
        "physical_storage_alias_with_live": False,
        "prefix_lru_member": False,
        "persistence": "ram-only",
    }


def test_capture_is_observation_only_and_all_snapshot_tensors_are_detached():
    cache, _ = _new_cache()
    handle = SemanticCacheHandle(cache)
    store = SemanticSnapshotStore()
    live_before = semantic_cache_digest(handle.cache)
    schema_before = semantic_cache_schema(handle.cache)

    snapshot = _capture(store, handle)

    assert semantic_cache_digest(handle.cache) == live_before
    assert semantic_cache_schema(handle.cache) == schema_before
    live_arrays = list(_cache_arrays(handle.cache))
    snapshot_arrays = list(_cache_arrays(snapshot._cache))
    assert len(live_arrays) == len(snapshot_arrays) == snapshot.cache_leaf_count
    assert all(left is not right for left, right in zip(live_arrays, snapshot_arrays))
    assert snapshot.state_sha256 == live_before
    assert snapshot.component_digests == semantic_component_digests(handle.cache)


def test_snapshot_is_immutable_under_live_mutation_and_reusable_restore():
    cache, indexer = _new_cache()
    handle = SemanticCacheHandle(cache)
    store = SemanticSnapshotStore()
    snapshot = _capture(store, handle)
    snapshot_digest = snapshot.state_sha256
    snapshot_components = dict(snapshot.component_digests)

    _advance(handle.cache, indexer, 8, 8)
    assert semantic_cache_digest(handle.cache) != snapshot_digest
    assert snapshot.state_sha256 == snapshot_digest
    with pytest.raises(TypeError):
        snapshot.component_digests["kda_state_sha256"] = "mutated"
    assert dict(snapshot.component_digests) == snapshot_components

    store.restore("snapshot-a", handle, expected_identity=_identity())
    assert semantic_cache_digest(handle.cache) == snapshot_digest
    _advance(handle.cache, indexer, 8, 4)
    store.restore("snapshot-a", handle, expected_identity=_identity())
    assert semantic_cache_digest(handle.cache) == snapshot_digest
    assert snapshot.state_sha256 == snapshot_digest


def test_restore_branch_and_replay_are_byte_exact():
    cache, indexer = _new_cache()
    store = SemanticSnapshotStore()
    source = SemanticCacheHandle(cache)
    snapshot = _capture(store, source)
    left = SemanticCacheHandle(_new_cache()[0])
    right = SemanticCacheHandle(_new_cache()[0])

    store.restore("snapshot-a", left, expected_identity=_identity())
    store.restore("snapshot-a", right, expected_identity=_identity())
    _advance(left.cache, indexer, 8, 16)
    _advance(right.cache, indexer, 8, 16)

    assert semantic_cache_digest(left.cache) == semantic_cache_digest(right.cache)
    assert semantic_component_digests(left.cache) == semantic_component_digests(
        right.cache
    )
    assert snapshot.state_sha256 == semantic_cache_digest(snapshot._cache)


def test_capture_failure_does_not_publish_or_mutate_live_state_or_accounting():
    from mlx_vlm.apc_adapters import clone_cache_entry

    cache, _ = _new_cache()
    handle = SemanticCacheHandle(cache)
    store = SemanticSnapshotStore()
    before_cache = handle.cache
    before_digest = semantic_cache_digest(handle.cache)
    before_accounting = store.accounting()
    calls = 0

    def fail_after_one(entry, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected capture failure")
        return clone_cache_entry(entry, **kwargs)

    with pytest.raises(RuntimeError, match="injected capture failure"):
        _capture(store, handle, clone_entry=fail_after_one)
    assert handle.cache is before_cache
    assert semantic_cache_digest(handle.cache) == before_digest
    assert store.snapshot_ids == ()
    assert store.accounting() == before_accounting


def test_capture_rejects_live_state_change_before_publish():
    from mlx_vlm.apc_adapters import clone_cache_entry

    cache, _ = _new_cache()
    handle = SemanticCacheHandle(cache)
    store = SemanticSnapshotStore()
    calls = 0

    def mutate_during_clone(entry, **kwargs):
        nonlocal calls
        replacement = clone_cache_entry(entry, **kwargs)
        calls += 1
        if calls == 1:
            handle.cache[0][0] = handle.cache[0][0] + mx.array(
                1, dtype=mx.bfloat16
            )
        return replacement

    with pytest.raises(SemanticSnapshotError, match="changed during snapshot capture"):
        _capture(store, handle, clone_entry=mutate_during_clone)
    assert store.snapshot_ids == ()
    assert store.accounting()["resident_bytes"] == 0


def test_restore_failure_is_atomic_after_preparing_one_component():
    from mlx_vlm.apc_adapters import clone_cache_entry

    cache, indexer = _new_cache()
    handle = SemanticCacheHandle(cache)
    store = SemanticSnapshotStore()
    _capture(store, handle)
    _advance(handle.cache, indexer, 8, 4)
    live_reference = handle.cache
    live_digest = semantic_cache_digest(handle.cache)
    calls = 0

    def fail_after_one(entry, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected restore failure")
        return clone_cache_entry(entry, **kwargs)

    with pytest.raises(RuntimeError, match="injected restore failure"):
        store.restore(
            "snapshot-a",
            handle,
            expected_identity=_identity(),
            clone_entry=fail_after_one,
        )
    assert handle.cache is live_reference
    assert semantic_cache_digest(handle.cache) == live_digest


def test_identity_mismatch_rejected_before_clone_and_live_state_unchanged():
    cache, _ = _new_cache()
    handle = SemanticCacheHandle(cache)
    store = SemanticSnapshotStore()
    _capture(store, handle)
    before = semantic_cache_digest(handle.cache)
    clone_called = False

    def forbidden(*args, **kwargs):
        nonlocal clone_called
        clone_called = True
        raise AssertionError("identity must be checked before cloning")

    with pytest.raises(SemanticSnapshotError, match="identity mismatch"):
        store.restore(
            "snapshot-a",
            handle,
            expected_identity=replace(_identity(), checkpoint_revision="wrong"),
            clone_entry=forbidden,
        )
    assert clone_called is False
    assert semantic_cache_digest(handle.cache) == before


def test_inconsistent_boundary_and_materialization_epoch_fail_before_publish():
    cache, _ = _new_cache(tokens=8)
    handle = SemanticCacheHandle(cache)
    store = SemanticSnapshotStore()
    before = semantic_cache_digest(handle.cache)

    with pytest.raises(SemanticSnapshotError, match="logical extent"):
        _capture(store, handle, tokens=7)
    with pytest.raises(SemanticSnapshotError, match="materialization epoch"):
        store.capture(
            handle,
            snapshot_id="wrong-epoch",
            identity=_identity(),
            absolute_token_position=8,
            materialization_epoch=1,
        )
    assert store.snapshot_ids == ()
    assert store.accounting()["resident_bytes"] == 0
    assert semantic_cache_digest(handle.cache) == before


def test_snapshot_accounting_returns_to_baseline_only_on_explicit_delete():
    cache, _ = _new_cache()
    handle = SemanticCacheHandle(cache)
    store = SemanticSnapshotStore()
    snapshot = _capture(store, handle)
    allocated = store.accounting()
    assert allocated["resident_bytes"] == snapshot.resident_bytes > 0
    assert allocated["allocation_count"] == 1
    assert allocated["anonymous_allocation_count"] == 0
    assert allocated["prefix_lru_member"] is False
    assert allocated["disk_persistence"] is False

    store.delete("snapshot-a")
    released = store.accounting()
    assert snapshot.released is True
    assert snapshot._cache == ()
    assert released["resident_bytes"] == 0
    assert released["cumulative_allocated_bytes"] == snapshot.resident_bytes
    assert released["cumulative_released_bytes"] == snapshot.resident_bytes
    assert released["release_count"] == 1


def test_target_prefix_lru_pressure_cannot_evict_semantic_snapshot():
    cache, _ = _new_cache()
    store = SemanticSnapshotStore()
    snapshot = _capture(store, SemanticCacheHandle(cache))
    digest = snapshot.state_sha256
    budgets = {lifecycle: 1024 for lifecycle in CacheLifecycle}
    manager = CacheLifecycleManager(budgets)
    for index in range(64):
        identity = PrefixIdentity(
            model_revision="04c4e9e",
            checkpoint_fingerprint="checkpoint",
            backend_policy="packed-decode",
            attention_cache_abi=NOPE_DSA_CACHE_ABI_COMPACT,
            kda_state_abi=KDA_STATE_ABI,
            indexpool_abi=COMPACT_INDEXPOOL_ABI,
            prefix_token_sha256=f"prefix-{index}",
        )
        manager.allocate(
            entry_id=f"target-{index}",
            payload=b"P" * 128,
            lifecycle=CacheLifecycle.TARGET_PREFIX,
            retention=RetentionPolicy.LONG_REUSE,
            owner_id="prefix-cache",
            prefix_identity=identity,
        )
    assert manager.accounting_snapshot()["target-prefix"]["eviction_count"] > 0
    assert store.get("snapshot-a").state_sha256 == digest
    assert store.accounting()["resident_bytes"] == snapshot.resident_bytes


@pytest.mark.parametrize("tokens", [1, 255, 256, 257, 1023, 1024])
def test_boundary_contract_covers_materialization_edges(tokens):
    cache, _ = _new_cache(tokens=tokens)
    boundary = inspect_semantic_boundary(
        cache,
        absolute_token_position=tokens,
        materialization_epoch=tokens // 256,
    )
    assert boundary.absolute_token_position == tokens
    assert boundary.logical_prefix_length == tokens
    assert boundary.materialization_epoch == tokens // 256
    assert boundary.kv_logical_extents == ((1, tokens),)
    assert boundary.indexpool_logical_extents == ((1, tokens),)
