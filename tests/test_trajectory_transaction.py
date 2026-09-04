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
from glm53_flash_mlx.semantic_branch import SemanticBranchManager
from glm53_flash_mlx.semantic_snapshot import (
    COMPACT_INDEXPOOL_ABI,
    KDA_STATE_ABI,
    SemanticCacheHandle,
    SemanticSnapshotIdentity,
    SemanticSnapshotStore,
    semantic_cache_digest,
)
from glm53_flash_mlx.trajectory_transaction import (
    TRAJECTORY_TRANSACTION_SCHEMA,
    ActiveSemanticState,
    StaleTrajectoryEvaluation,
    StaleTrajectoryTransaction,
    TrajectoryTransactionError,
    TrajectoryTransactionManager,
    TransactionState,
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
    mx.random.seed(211)
    value = Glm5NextIndexer(config)
    value.set_dtype(mx.bfloat16)
    return value


def _inputs(start: int, count: int):
    positions = mx.arange(start, start + count, dtype=mx.float32)[:, None]
    x = mx.sin(positions * 0.03125 + mx.arange(8)[None] * 0.0625)[None]
    qr = mx.cos(positions * 0.046875 + mx.arange(4)[None] * 0.09375)[None]
    return x.astype(mx.bfloat16), qr.astype(mx.bfloat16)


def _identity() -> SemanticSnapshotIdentity:
    return SemanticSnapshotIdentity(
        checkpoint_revision="04c4e9e",
        checkpoint_fingerprint="checkpoint-fingerprint",
        moe_backend="packed-decode",
        cache_backend="compact-nope-dsa",
        attention_cache_abi=NOPE_DSA_CACHE_ABI_COMPACT,
        kda_state_abi=KDA_STATE_ABI,
        indexpool_abi=COMPACT_INDEXPOOL_ABI,
        prefix_token_sha256="transaction-parent",
    )


def _source():
    from mlx_vlm.models.cache import ArraysCache

    indexer = _indexer()
    left = ArraysCache(size=2)
    right = ArraysCache(size=2)
    left[0] = mx.zeros((1, 8, 2), dtype=mx.bfloat16)
    left[1] = mx.zeros((1, 2, 4, 4), dtype=mx.float32)
    right[0] = mx.ones((1, 8, 2), dtype=mx.bfloat16)
    right[1] = mx.ones((1, 2, 4, 4), dtype=mx.float32)
    dsa = make_compact_nope_dsa_cache(indexer, capacity_tokens=512)
    x, qr = _inputs(0, 8)
    assert indexer(x, qr, None, cache=dsa[1]) is None
    latent = mx.arange(32, dtype=mx.float32).reshape(1, 1, 8, 4)
    dsa[0].update_and_fetch((latent * 0.015625).astype(mx.bfloat16), latent)
    cache = [left, dsa, right]
    mx.eval([entry.state for entry in cache])
    store = SemanticSnapshotStore()
    identity = _identity()
    snapshot = store.capture(
        SemanticCacheHandle(cache),
        snapshot_id="S0",
        identity=identity,
        absolute_token_position=8,
        materialization_epoch=0,
    )
    active = ActiveSemanticState.from_snapshot(snapshot, expected_identity=identity)
    return store, snapshot, identity, indexer, active


def _advance(cache, indexer, start: int, count: int, *, salt: float) -> None:
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


def _execute_candidates(manager, transaction, indexer) -> dict[int, dict[str, object]]:
    observations = {}
    for offset, branch_id in enumerate(transaction.candidate_branch_ids):
        branch = manager.branches.activate(branch_id)
        _advance(branch.cache, indexer, branch.position, 4, salt=float(offset))
        manager.branches.commit_active_position(branch.position + 4)
        observations[branch_id] = {
            "digest": branch.state_sha256,
            "position": branch.position,
            "cache_id": id(branch.cache),
            "resident_bytes": branch.resident_bytes,
        }
    manager.branches.deactivate()
    manager.mark_executed(transaction.transaction_id)
    return observations


@pytest.mark.parametrize("winner_offset", [0, 1])
def test_a_and_b_winner_commit_promotes_owned_state_without_copy(winner_offset):
    _, snapshot, identity, indexer, active = _source()
    branches = SemanticBranchManager()
    manager = TrajectoryTransactionManager(active, branches)
    transaction = manager.begin(
        transaction_id=f"winner-{winner_offset}",
        parent_snapshot=snapshot,
        expected_identity=identity,
    )
    observations = _execute_candidates(manager, transaction, indexer)
    winner_id = transaction.candidate_branch_ids[winner_offset]
    loser_id = transaction.candidate_branch_ids[1 - winner_offset]
    evaluation = manager.evaluate(
        transaction.transaction_id, winner_branch_id=winner_id, score=0.5
    )
    assert evaluation.winner_branch_id == winner_id

    result = manager.commit(transaction.transaction_id)

    assert transaction.state is TransactionState.COMMITTED
    assert result.winner_storage_promoted_without_copy is True
    assert result.mixed_generation_count == 0
    assert result.winner_semantic_digest == result.active_semantic_digest
    assert active.state_sha256 == observations[winner_id]["digest"]
    assert active.position == observations[winner_id]["position"]
    assert id(active.cache) == observations[winner_id]["cache_id"]
    assert not branches.has_branch(winner_id)
    assert not branches.has_branch(loser_id)
    accounting = manager.accounting()
    assert accounting["transaction_resident_bytes"] == 0
    assert accounting["branch"]["resident_bytes"] == 0
    assert accounting["branch"]["by_branch"][str(winner_id)]["promotion_count"] == 1
    assert accounting["branch"]["by_branch"][str(loser_id)]["release_count"] == 1
    assert accounting["winner_promoted_bytes"] == observations[winner_id]["resident_bytes"]
    assert accounting["loser_released_bytes"] == observations[loser_id]["resident_bytes"]
    assert accounting["ownership_balance_exact"] is True
    assert accounting["branch"]["ownership_balance_bytes"] == 0
    assert accounting["active"]["ownership_balance_bytes"] == 0
    assert accounting["anonymous_allocation_count"] == 0

    before = active.state_sha256
    _advance(active.cache, indexer, active.position, 4, salt=float(winner_offset))
    active.commit_position(active.position + 4)
    assert active.state_sha256 != before
    with pytest.raises(TrajectoryTransactionError, match="operation is not allowed"):
        manager.commit(transaction.transaction_id)
    with pytest.raises(TrajectoryTransactionError, match="operation is not allowed"):
        manager.abort(transaction.transaction_id)


def test_evaluation_is_observation_only_and_stale_winner_rejects_before_mutation():
    _, snapshot, identity, indexer, active = _source()
    manager = TrajectoryTransactionManager(active)
    transaction = manager.begin(
        transaction_id="stale-evaluation",
        parent_snapshot=snapshot,
        expected_identity=identity,
    )
    _execute_candidates(manager, transaction, indexer)
    winner_id = transaction.candidate_branch_ids[0]
    before_evaluate = {
        branch_id: (
            manager.branches.branch(branch_id).state_sha256,
            manager.branches.branch(branch_id).position,
            id(manager.branches.branch(branch_id).cache),
        )
        for branch_id in transaction.candidate_branch_ids
    }
    manager.evaluate(transaction.transaction_id, winner_branch_id=winner_id, score=1.0)
    after_evaluate = {
        branch_id: (
            manager.branches.branch(branch_id).state_sha256,
            manager.branches.branch(branch_id).position,
            id(manager.branches.branch(branch_id).cache),
        )
        for branch_id in transaction.candidate_branch_ids
    }
    assert after_evaluate == before_evaluate

    winner = manager.branches.activate(winner_id)
    _advance(winner.cache, indexer, winner.position, 1, salt=2.0)
    manager.branches.commit_active_position(winner.position + 1)
    manager.branches.deactivate()
    active_before = (active.generation, active.position, active.state_sha256, id(active.cache))
    branches_before = {
        branch_id: (
            manager.branches.branch(branch_id).state_sha256,
            manager.branches.branch(branch_id).position,
            id(manager.branches.branch(branch_id).cache),
        )
        for branch_id in transaction.candidate_branch_ids
    }
    with pytest.raises(StaleTrajectoryEvaluation):
        manager.commit(transaction.transaction_id)
    assert (active.generation, active.position, active.state_sha256, id(active.cache)) == active_before
    assert {
        branch_id: (
            manager.branches.branch(branch_id).state_sha256,
            manager.branches.branch(branch_id).position,
            id(manager.branches.branch(branch_id).cache),
        )
        for branch_id in transaction.candidate_branch_ids
    } == branches_before
    assert transaction.state is TransactionState.EVALUATED
    manager.abort(transaction.transaction_id)
    assert transaction.state is TransactionState.ABORTED
    assert manager.accounting()["transaction_resident_bytes"] == 0


def test_stale_base_cas_rejects_and_abort_preserves_new_active_root():
    _, snapshot, identity, indexer, active = _source()
    manager = TrajectoryTransactionManager(active)
    transaction = manager.begin(
        transaction_id="stale-base",
        parent_snapshot=snapshot,
        expected_identity=identity,
    )
    _execute_candidates(manager, transaction, indexer)
    manager.evaluate(
        transaction.transaction_id,
        winner_branch_id=transaction.candidate_branch_ids[1],
        score=-1.0,
    )

    _advance(active.cache, indexer, active.position, 1, salt=3.0)
    active.commit_position(active.position + 1)
    active_before = (active.generation, active.position, active.state_sha256, id(active.cache))
    branch_ids_before = manager.branches.branch_ids
    with pytest.raises(StaleTrajectoryTransaction):
        manager.commit(transaction.transaction_id)
    assert (active.generation, active.position, active.state_sha256, id(active.cache)) == active_before
    assert manager.branches.branch_ids == branch_ids_before
    assert manager.accounting()["stale_commit_reject_count"] == 1
    manager.abort(transaction.transaction_id)
    assert (active.generation, active.position, active.state_sha256, id(active.cache)) == active_before
    assert manager.branches.branch_ids == ()


def test_abort_and_n_candidate_state_machine_release_all_transaction_state():
    _, snapshot, identity, _, active = _source()
    manager = TrajectoryTransactionManager(active)
    active_before = (active.generation, active.position, active.state_sha256, id(active.cache))
    transaction = manager.begin(
        transaction_id="abort-three",
        parent_snapshot=snapshot,
        expected_identity=identity,
        candidate_count=3,
    )
    assert transaction.state is TransactionState.OPEN
    assert len(transaction.candidate_branch_ids) == 3
    assert manager.accounting()["transaction_resident_bytes"] > 0
    manager.abort(transaction.transaction_id)
    assert transaction.state is TransactionState.ABORTED
    assert manager.branches.branch_ids == ()
    assert manager.accounting()["transaction_resident_bytes"] == 0
    assert (active.generation, active.position, active.state_sha256, id(active.cache)) == active_before
    with pytest.raises(TrajectoryTransactionError, match="operation is not allowed"):
        manager.abort(transaction.transaction_id)
    with pytest.raises(TrajectoryTransactionError, match="operation is not allowed"):
        manager.evaluate(transaction.transaction_id, winner_branch_id=1, score=0.0)
    assert manager.accounting()["schema"] == TRAJECTORY_TRANSACTION_SCHEMA
    assert manager.accounting()["ownership_balance_exact"] is True


def test_invalid_evaluation_does_not_select_or_mutate_runtime_state():
    _, snapshot, identity, indexer, active = _source()
    manager = TrajectoryTransactionManager(active)
    transaction = manager.begin(
        transaction_id="invalid-evaluation",
        parent_snapshot=snapshot,
        expected_identity=identity,
    )
    _execute_candidates(manager, transaction, indexer)
    before = manager.accounting()
    with pytest.raises(TrajectoryTransactionError, match="not a transaction candidate"):
        manager.evaluate(transaction.transaction_id, winner_branch_id=999, score=1.0)
    assert manager.accounting() == before
    assert transaction.state is TransactionState.EXECUTED
    manager.abort(transaction.transaction_id)
