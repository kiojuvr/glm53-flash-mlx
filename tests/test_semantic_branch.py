from __future__ import annotations

from types import SimpleNamespace

import pytest

try:
    import mlx.core as mx
except ImportError:
    pytest.skip("MLX/Metal is unavailable", allow_module_level=True)

from glm53_flash_mlx.abi import NOPE_DSA_CACHE_ABI_COMPACT
from glm53_flash_mlx.nope_cache import make_compact_nope_dsa_cache
from glm53_flash_mlx.patch import apply_runtime_patch
from glm53_flash_mlx.semantic_branch import (
    SEMANTIC_BRANCH_LIFECYCLE,
    SEMANTIC_BRANCH_SCHEMA,
    SemanticBranchError,
    SemanticBranchManager,
)
from glm53_flash_mlx.semantic_snapshot import (
    COMPACT_INDEXPOOL_ABI,
    KDA_STATE_ABI,
    SemanticCacheHandle,
    SemanticSnapshotIdentity,
    SemanticSnapshotStore,
    semantic_cache_digest,
    semantic_cache_storage_alias_count,
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
    mx.random.seed(197)
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
    latent = mx.arange(tokens * 4, dtype=mx.float32).reshape(1, 1, tokens, 4)
    latent = (latent * 0.015625).astype(mx.bfloat16)
    dsa[0].update_and_fetch(latent, latent)
    mx.eval(left.state, right.state, dsa.state)
    return [left, dsa, right], indexer


def _advance(cache, indexer, start: int, count: int, *, salt: float = 0.0) -> None:
    for position in range(start, start + count):
        for layer in (0, 2):
            cache[layer][0] = cache[layer][0] + mx.array(
                1 + salt, dtype=mx.bfloat16
            )
            cache[layer][1] = cache[layer][1] + mx.array(
                0.25 + salt, dtype=mx.float32
            )
        x, qr = _inputs(position, 1)
        x = (x.astype(mx.float32) + salt).astype(mx.bfloat16)
        assert indexer(x, qr, None, cache=cache[1][1]) is None
        latent = mx.full((1, 1, 1, 4), position + salt, dtype=mx.bfloat16)
        cache[1][0].update_and_fetch(latent, latent)
    mx.eval([entry.state for entry in cache])


def _identity(token_digest: str = "prefix") -> SemanticSnapshotIdentity:
    return SemanticSnapshotIdentity(
        checkpoint_revision="04c4e9e",
        checkpoint_fingerprint="checkpoint-fingerprint",
        moe_backend="packed-decode",
        cache_backend="compact-nope-dsa",
        attention_cache_abi=NOPE_DSA_CACHE_ABI_COMPACT,
        kda_state_abi=KDA_STATE_ABI,
        indexpool_abi=COMPACT_INDEXPOOL_ABI,
        prefix_token_sha256=token_digest,
    )


def _source_snapshot():
    cache, indexer = _new_cache()
    handle = SemanticCacheHandle(cache)
    store = SemanticSnapshotStore()
    snapshot = store.capture(
        handle,
        snapshot_id="S0",
        identity=_identity(),
        absolute_token_position=8,
        materialization_epoch=0,
    )
    return store, snapshot, indexer


def test_fork_creates_owned_alias_free_twin_branches():
    _, snapshot, indexer = _source_snapshot()
    manager = SemanticBranchManager()
    left = manager.fork(snapshot, expected_identity=_identity())
    right = manager.fork(snapshot, expected_identity=_identity())

    assert left.identity.branch_id == 1
    assert right.identity.branch_id == 2
    assert left.identity.parent_snapshot_id == right.identity.parent_snapshot_id == "S0"
    assert left.identity.generation == right.identity.generation == 0
    assert semantic_cache_storage_alias_count(left.cache, right.cache) == 0
    assert semantic_cache_storage_alias_count(left.cache, snapshot._cache) == 0
    assert semantic_cache_storage_alias_count(right.cache, snapshot._cache) == 0

    manager.activate(1)
    _advance(left.cache, indexer, 8, 16)
    manager.commit_active_position(24)
    manager.activate(2)
    _advance(right.cache, indexer, 8, 16)
    manager.commit_active_position(24)
    assert left.state_sha256 == right.state_sha256
    assert snapshot.state_sha256 == semantic_cache_digest(snapshot._cache)
    assert {branch for _, branch, _ in left.component_generation_tags()} == {1}
    assert {generation for _, _, generation in left.component_generation_tags()} == {0}


def test_divergent_mutation_and_failed_activation_are_branch_local():
    _, snapshot, indexer = _source_snapshot()
    manager = SemanticBranchManager()
    left = manager.fork(snapshot, expected_identity=_identity())
    right = manager.fork(snapshot, expected_identity=_identity())
    snapshot_before = snapshot.state_sha256
    right_before = right.state_sha256
    right_accounting_before = manager.accounting()["by_branch"]["2"].copy()

    manager.activate(1)
    _advance(left.cache, indexer, 8, 4, salt=1.0)
    manager.commit_active_position(12)
    assert right.state_sha256 == right_before
    assert manager.accounting()["by_branch"]["2"] == right_accounting_before
    assert snapshot.state_sha256 == snapshot_before

    active_before = manager.active_branch_id
    cache_before = manager.active_cache
    left_before = left.state_sha256
    with pytest.raises(RuntimeError, match="injected activation failure"):
        manager.activate(
            2,
            validator=lambda _: (_ for _ in ()).throw(
                RuntimeError("injected activation failure")
            ),
        )
    assert manager.active_branch_id == active_before == 1
    assert manager.active_cache is cache_before
    assert left.state_sha256 == left_before
    assert right.state_sha256 == right_before


def test_branch_local_snapshot_rollback_lineage_and_failure_atomicity():
    store, snapshot, indexer = _source_snapshot()
    manager = SemanticBranchManager()
    left = manager.fork(snapshot, expected_identity=_identity())
    right = manager.fork(snapshot, expected_identity=_identity())
    manager.activate(1)
    _advance(left.cache, indexer, 8, 4)
    manager.commit_active_position(12)
    rollback_identity = _identity("left-at-12")
    rollback = manager.capture_branch_snapshot(
        1,
        store,
        snapshot_id="A-rollback",
        identity=rollback_identity,
    )
    lineage = manager.snapshot_lineage("A-rollback")
    assert lineage.source_branch_id == 1
    assert lineage.source_branch_generation == 0
    assert lineage.parent_snapshot_id == "S0"
    assert lineage.absolute_token_position == 12

    _advance(left.cache, indexer, 12, 16)
    manager.commit_active_position(28)
    expected = left.state_sha256
    right_before = right.state_sha256
    snapshot_before = snapshot.state_sha256
    manager.restore_into_branch(
        1,
        rollback,
        expected_identity=rollback_identity,
        rollback_tokens=16,
    )
    assert left.identity.generation == 1
    assert left.identity.parent_snapshot_id == "A-rollback"
    _advance(left.cache, indexer, 12, 16)
    manager.commit_active_position(28)
    assert left.state_sha256 == expected
    assert right.state_sha256 == right_before
    assert snapshot.state_sha256 == snapshot_before

    live_before = left.state_sha256
    identity_before = left.identity
    accounting_before = manager.accounting()
    with pytest.raises(SemanticBranchError, match=r"within \[1, 16\]"):
        manager.restore_into_branch(
            1,
            rollback,
            expected_identity=rollback_identity,
            rollback_tokens=17,
        )
    assert left.state_sha256 == live_before
    assert left.identity == identity_before
    assert manager.accounting() == accounting_before
    assert right.state_sha256 == right_before
    assert snapshot.state_sha256 == snapshot_before


def test_branch_delete_releases_only_its_owned_state_and_preserves_snapshots():
    store, snapshot, _ = _source_snapshot()
    manager = SemanticBranchManager()
    left = manager.fork(snapshot, expected_identity=_identity())
    right = manager.fork(snapshot, expected_identity=_identity())
    left_snapshot = manager.capture_branch_snapshot(
        1,
        store,
        snapshot_id="SA",
        identity=_identity("left"),
    )
    right_snapshot = manager.capture_branch_snapshot(
        2,
        store,
        snapshot_id="SB",
        identity=_identity("right"),
    )
    right_before = right.state_sha256
    s0_before = snapshot.state_sha256
    sa_before = left_snapshot.state_sha256
    sb_before = right_snapshot.state_sha256

    manager.activate(2)
    manager.delete_branch(1)
    after_left = manager.accounting()
    assert left.released is True
    assert after_left["resident_bytes_by_branch"]["1"] == 0
    assert right.state_sha256 == right_before
    assert snapshot.state_sha256 == s0_before
    assert left_snapshot.state_sha256 == sa_before
    assert right_snapshot.state_sha256 == sb_before

    manager.deactivate()
    manager.delete_branch(2)
    final = manager.accounting()
    assert final["schema"] == SEMANTIC_BRANCH_SCHEMA
    assert final["lifecycle"] == SEMANTIC_BRANCH_LIFECYCLE
    assert final["resident_bytes"] == 0
    assert final["branch_count"] == 0
    assert final["branch_create_count"] == final["branch_delete_count"] == 2
    assert final["cumulative_allocated_bytes"] == final["cumulative_released_bytes"]
    assert final["anonymous_allocation_count"] == 0

    manager.delete_branch_snapshot(store, "SA")
    manager.delete_branch_snapshot(store, "SB")
    assert manager.accounting()["snapshot_bytes_by_lineage"] == {}
    store.delete("S0")
    assert store.accounting()["resident_bytes"] == 0


def test_failed_fork_does_not_publish_branch_or_accounting():
    _, snapshot, _ = _source_snapshot()
    manager = SemanticBranchManager()
    before = manager.accounting()
    with pytest.raises(Exception, match="identity mismatch"):
        manager.fork(
            snapshot,
            expected_identity=_identity("wrong-prefix"),
        )
    assert manager.accounting() == before
