#!/usr/bin/env python3
"""Qualify atomic A/B trajectory winner commit on the real hybrid cache.

The evaluator is deliberately external to the transaction mechanism: this
probe supplies an already-selected winner id and score, while the runtime only
validates generation/digest freshness and performs an owned-state promotion.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import weakref
from datetime import date
from pathlib import Path

import mlx.core as mx

from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint
from glm53_flash_mlx.semantic_branch import SemanticBranchManager
from glm53_flash_mlx.semantic_snapshot import (
    SEMANTIC_PREFIX_SNAPSHOT_SCHEMA,
    SemanticCacheHandle,
    SemanticSnapshotStore,
    semantic_cache_resident_bytes,
)
from glm53_flash_mlx.trajectory_transaction import (
    TRAJECTORY_TRANSACTION_SCHEMA,
    ActiveSemanticState,
    StaleTrajectoryEvaluation,
    StaleTrajectoryTransaction,
    TrajectoryTransactionError,
    TrajectoryTransactionManager,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from qualify_hybrid_semantic_prefix_snapshot_replay import (  # noqa: E402
    CHECKPOINT_INTERVAL,
    _atomic_write,
    _compare_trajectory,
    _identity,
    _materialize,
    _memory,
    _official_oracle,
    _progress,
    _run_trajectory,
    _tokens,
)


DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-trajectory-transaction-winner-commit-20260905.json"
)
DEFAULT_PREFIX_TOKENS = 256
DEFAULT_TRAJECTORY_STEPS = 256
DEFAULT_CONTINUATION_STEPS = 64
DEFAULT_FAILURE_STEPS = 16
MAX_PEAK_BYTES = 340_000_000_000


def _summary(value: dict) -> dict:
    return {
        key: row
        for key, row in value.items()
        if key not in {"logits_hashes", "checkpoints"}
    } | {"checkpoint_count": len(value["checkpoints"])}


def _run_branch(
    model,
    manager: TrajectoryTransactionManager,
    branch_id: int,
    token_ids: list[int],
    *,
    steps: int,
) -> dict:
    branch = manager.branches.activate(branch_id)
    start = branch.position
    trajectory = _run_trajectory(
        model,
        branch,
        token_ids,
        start_position=start,
        steps=steps,
    )
    manager.branches.commit_active_position(start + steps)
    manager.branches.deactivate()
    return trajectory


def _run_active(
    model,
    active: ActiveSemanticState,
    token_ids: list[int],
    *,
    steps: int,
) -> dict:
    start = active.position
    trajectory = _run_trajectory(
        model,
        SemanticCacheHandle(active.cache),
        token_ids,
        start_position=start,
        steps=steps,
    )
    active.commit_position(start + steps)
    return trajectory


def _branch_observation(manager: SemanticBranchManager, branch_id: int) -> dict:
    branch = manager.branch(branch_id)
    return {
        "generation": branch.identity.generation,
        "position": branch.position,
        "state_sha256": branch.state_sha256,
        "cache_id": id(branch.cache),
        "resident_bytes": branch.resident_bytes,
        "accounting": manager.accounting()["by_branch"][str(branch_id)],
    }


def _active_observation(active: ActiveSemanticState) -> dict:
    return {
        "generation": active.generation,
        "position": active.position,
        "state_sha256": active.state_sha256,
        "cache_id": id(active.cache),
        "resident_bytes": active.resident_bytes,
        "accounting": active.accounting(),
    }


def _run_winner_case(
    *,
    case_name: str,
    winner_offset: int,
    model,
    manager: TrajectoryTransactionManager,
    store: SemanticSnapshotStore,
    checkpoint_report,
    s0,
    s0_digest: str,
    identity_s0,
    tokens_a: list[int],
    tokens_b: list[int],
    trajectory_steps: int,
    continuation_steps: int,
) -> dict:
    manager.active.restore_snapshot(s0, expected_identity=identity_s0)
    active_at_begin = _active_observation(manager.active)
    transaction = manager.begin(
        transaction_id=case_name,
        parent_snapshot=s0,
        expected_identity=identity_s0,
    )
    branch_ids = transaction.candidate_branch_ids
    trajectories = {}
    sibling_isolation = {}
    token_sets = (tokens_a, tokens_b)
    for offset, branch_id in enumerate(branch_ids):
        sibling_id = branch_ids[1 - offset]
        sibling_before = _branch_observation(manager.branches, sibling_id)
        trajectories[str(branch_id)] = _run_branch(
            model,
            manager,
            branch_id,
            token_sets[offset],
            steps=trajectory_steps,
        )
        sibling_isolation[str(branch_id)] = (
            sibling_before == _branch_observation(manager.branches, sibling_id)
        )
    manager.mark_executed(transaction.transaction_id)
    winner_id = branch_ids[winner_offset]
    loser_id = branch_ids[1 - winner_offset]
    winner = manager.branches.branch(winner_id)
    winner_observation = _branch_observation(manager.branches, winner_id)
    loser_observation = _branch_observation(manager.branches, loser_id)
    evaluation_before = {
        branch_id: _branch_observation(manager.branches, branch_id)
        for branch_id in branch_ids
    }
    evaluation = manager.evaluate(
        transaction.transaction_id,
        winner_branch_id=winner_id,
        score=1.0,
    )
    evaluation_after = {
        branch_id: _branch_observation(manager.branches, branch_id)
        for branch_id in branch_ids
    }
    winner_tokens = token_sets[winner_offset]
    winner_identity = _identity(
        checkpoint_report,
        winner_tokens,
        winner.position,
    )
    oracle_snapshot_id = f"{case_name}-winner-oracle"
    oracle_snapshot = manager.branches.capture_branch_snapshot(
        winner_id,
        store,
        snapshot_id=oracle_snapshot_id,
        identity=winner_identity,
    )
    winner_cache_id = id(winner.cache)
    old_active_refs = [weakref.ref(entry) for entry in manager.active.cache]
    loser_refs = [weakref.ref(entry) for entry in manager.branches.branch(loser_id).cache]
    result = manager.commit(transaction.transaction_id)
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    post_commit = _active_observation(manager.active)

    committed_continuation = _run_active(
        model,
        manager.active,
        winner_tokens,
        steps=continuation_steps,
    )
    oracle_branch = manager.branches.fork(
        oracle_snapshot, expected_identity=winner_identity
    )
    oracle_continuation = _run_branch(
        model,
        manager,
        oracle_branch.identity.branch_id,
        winner_tokens,
        steps=continuation_steps,
    )
    comparison = _compare_trajectory(oracle_continuation, committed_continuation)
    final_state_exact = (
        manager.active.state_sha256
        == manager.branches.branch(oracle_branch.identity.branch_id).state_sha256
    )
    manager.branches.delete_branch(oracle_branch.identity.branch_id)
    manager.branches.delete_branch_snapshot(store, oracle_snapshot_id)

    terminal_rejections = {}
    for name, operation in {
        "double_commit": lambda: manager.commit(transaction.transaction_id),
        "commit_then_abort": lambda: manager.abort(transaction.transaction_id),
    }.items():
        rejected = False
        try:
            operation()
        except TrajectoryTransactionError:
            rejected = True
        terminal_rejections[name] = rejected
    return {
        "winner_offset": winner_offset,
        "candidate_branch_ids": list(branch_ids),
        "active_at_begin": active_at_begin,
        "trajectories": {
            key: _summary(value) for key, value in trajectories.items()
        },
        "sibling_isolation": sibling_isolation,
        "evaluation": evaluation.descriptor(),
        "evaluation_observation_only": evaluation_before == evaluation_after,
        "winner_before_commit": winner_observation,
        "loser_before_commit": loser_observation,
        "commit": result.descriptor(),
        "post_commit": post_commit,
        "winner_cache_id_before": winner_cache_id,
        "winner_cache_id_after": id(manager.active.cache),
        "old_active_stale_entry_count": sum(
            reference() is not None for reference in old_active_refs
        ),
        "loser_stale_entry_count": sum(
            reference() is not None for reference in loser_refs
        ),
        "committed_continuation": _summary(committed_continuation),
        "oracle_continuation": _summary(oracle_continuation),
        "continuation_comparison": comparison,
        "continuation_final_state_exact": final_state_exact,
        "transaction_state": transaction.state.value,
        "terminal_rejections": terminal_rejections,
        "s0_immutable": s0.state_sha256 == s0_digest,
    }


def _run_stale_evaluation_case(
    model,
    manager: TrajectoryTransactionManager,
    s0,
    identity_s0,
    tokens_a,
    tokens_b,
    *,
    steps: int,
) -> dict:
    manager.active.restore_snapshot(s0, expected_identity=identity_s0)
    transaction = manager.begin(
        transaction_id="stale-evaluation",
        parent_snapshot=s0,
        expected_identity=identity_s0,
    )
    for branch_id, tokens in zip(
        transaction.candidate_branch_ids, (tokens_a, tokens_b), strict=True
    ):
        _run_branch(model, manager, branch_id, tokens, steps=steps)
    manager.mark_executed(transaction.transaction_id)
    winner_id = transaction.candidate_branch_ids[0]
    manager.evaluate(transaction.transaction_id, winner_branch_id=winner_id, score=1.0)
    _run_branch(model, manager, winner_id, tokens_a, steps=1)
    active_before = _active_observation(manager.active)
    branches_before = {
        branch_id: _branch_observation(manager.branches, branch_id)
        for branch_id in transaction.candidate_branch_ids
    }
    rejected = False
    try:
        manager.commit(transaction.transaction_id)
    except StaleTrajectoryEvaluation:
        rejected = True
    unchanged = active_before == _active_observation(manager.active) and all(
        branches_before[branch_id]
        == _branch_observation(manager.branches, branch_id)
        for branch_id in transaction.candidate_branch_ids
    )
    manager.abort(transaction.transaction_id)
    return {
        "rejected": rejected,
        "all_state_and_references_unchanged": unchanged,
        "transaction_state_after_abort": transaction.state.value,
        "resident_after_abort": manager.accounting()["transaction_resident_bytes"],
    }


def _run_stale_base_case(
    model,
    manager: TrajectoryTransactionManager,
    s0,
    identity_s0,
    tokens_a,
    tokens_b,
    *,
    steps: int,
) -> dict:
    manager.active.restore_snapshot(s0, expected_identity=identity_s0)
    transaction = manager.begin(
        transaction_id="stale-base",
        parent_snapshot=s0,
        expected_identity=identity_s0,
    )
    for branch_id, tokens in zip(
        transaction.candidate_branch_ids, (tokens_a, tokens_b), strict=True
    ):
        _run_branch(model, manager, branch_id, tokens, steps=steps)
    manager.mark_executed(transaction.transaction_id)
    manager.evaluate(
        transaction.transaction_id,
        winner_branch_id=transaction.candidate_branch_ids[1],
        score=1.0,
    )
    _run_active(model, manager.active, tokens_a, steps=1)
    active_before = _active_observation(manager.active)
    branches_before = {
        branch_id: _branch_observation(manager.branches, branch_id)
        for branch_id in transaction.candidate_branch_ids
    }
    rejected = False
    try:
        manager.commit(transaction.transaction_id)
    except StaleTrajectoryTransaction:
        rejected = True
    unchanged = active_before == _active_observation(manager.active) and all(
        branches_before[branch_id]
        == _branch_observation(manager.branches, branch_id)
        for branch_id in transaction.candidate_branch_ids
    )
    manager.abort(transaction.transaction_id)
    return {
        "rejected": rejected,
        "all_state_and_references_unchanged": unchanged,
        "transaction_state_after_abort": transaction.state.value,
        "resident_after_abort": manager.accounting()["transaction_resident_bytes"],
    }


def _run_abort_case(
    manager: TrajectoryTransactionManager,
    s0,
    identity_s0,
) -> dict:
    manager.active.restore_snapshot(s0, expected_identity=identity_s0)
    active_before = _active_observation(manager.active)
    transaction = manager.begin(
        transaction_id="explicit-abort",
        parent_snapshot=s0,
        expected_identity=identity_s0,
    )
    manager.abort(transaction.transaction_id)
    active_after = _active_observation(manager.active)
    terminal_rejected = False
    try:
        manager.commit(transaction.transaction_id)
    except TrajectoryTransactionError:
        terminal_rejected = True
    return {
        "active_unchanged": active_before == active_after,
        "state": transaction.state.value,
        "terminal_commit_rejected": terminal_rejected,
        "resident_after_abort": manager.accounting()["transaction_resident_bytes"],
    }


def _validate_args(args) -> None:
    for name in (
        "prefix_tokens",
        "trajectory_steps",
        "continuation_steps",
        "failure_steps",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be positive")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix-tokens", type=int, default=DEFAULT_PREFIX_TOKENS)
    parser.add_argument(
        "--trajectory-steps", type=int, default=DEFAULT_TRAJECTORY_STEPS
    )
    parser.add_argument(
        "--continuation-steps", type=int, default=DEFAULT_CONTINUATION_STEPS
    )
    parser.add_argument("--failure-steps", type=int, default=DEFAULT_FAILURE_STEPS)
    args = parser.parse_args()
    _validate_args(args)
    if not mx.metal.is_available():
        raise RuntimeError("trajectory transaction probe requires MLX/Metal")

    report = inspect_checkpoint(args.model, require_server_ready=True)
    capacity = (
        args.prefix_tokens + args.trajectory_steps + args.continuation_steps + 16
    )
    mx.set_wired_limit(int(440e9))
    mx.set_cache_limit(int(32e9))
    model, processor = load(
        args.model,
        experimental_packed_decode_moe=True,
        experimental_compact_nope_dsa_cache=True,
        compact_cache_capacity_tokens=capacity,
    )
    warm_residency(model)
    vocab = int(model.language_model.lm_head.weight.shape[0])
    token_count = args.prefix_tokens + args.trajectory_steps + args.continuation_steps + 2
    tokens_a = _tokens(token_count, vocab)
    tokens_b = list(tokens_a)
    tokens_a[args.prefix_tokens] = (tokens_a[args.prefix_tokens] + 17) % vocab
    tokens_b[args.prefix_tokens] = (tokens_b[args.prefix_tokens] + 31) % vocab

    source = SemanticCacheHandle(model.make_cache())
    prefix = mx.array(tokens_a[: args.prefix_tokens], dtype=mx.uint32)[None]
    output = model(prefix, cache=source.cache)
    mx.eval(output.logits)
    del output
    _materialize(source.cache)
    store = SemanticSnapshotStore()
    identity_s0 = _identity(report, tokens_a, args.prefix_tokens)
    s0 = store.capture(
        source,
        snapshot_id="S0",
        identity=identity_s0,
        absolute_token_position=args.prefix_tokens,
        materialization_epoch=args.prefix_tokens // CHECKPOINT_INTERVAL,
    )
    s0_digest = s0.state_sha256
    source_bytes = semantic_cache_resident_bytes(source.cache)
    source_refs = [weakref.ref(entry) for entry in source.cache]
    del source
    gc.collect()
    mx.clear_cache()
    mx.synchronize()

    active = ActiveSemanticState.from_snapshot(s0, expected_identity=identity_s0)
    branches = SemanticBranchManager()
    manager = TrajectoryTransactionManager(active, branches)
    artifact = {
        "schema": TRAJECTORY_TRANSACTION_SCHEMA,
        "snapshot_schema": SEMANTIC_PREFIX_SNAPSHOT_SCHEMA,
        "date": str(date.today()),
        "complete": False,
        "last_completed_phase": "loaded",
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "model_path": str(args.model.resolve()),
        "backend": {"moe": "packed-decode", "cache": "compact-nope-dsa"},
        "configuration": {
            "prefix_tokens": args.prefix_tokens,
            "trajectory_steps": args.trajectory_steps,
            "continuation_steps": args.continuation_steps,
            "failure_steps": args.failure_steps,
            "candidate_count": 2,
            "evaluation_policy": "external-deterministic-fixture",
            "winner_selection_is_runtime_policy": False,
            "capacity_tokens": capacity,
        },
        "source_release": {
            "resident_bytes": source_bytes,
            "stale_entry_reference_count": sum(
                reference() is not None for reference in source_refs
            ),
        },
        "runtime_changes": {
            "server": False,
            "disk_apc": False,
            "cache_abi": False,
            "kernel_abi": False,
            "backend": False,
            "admission": False,
            "parallel_branch_execution": False,
        },
        "acceptance": {},
    }
    _atomic_write(args.output, artifact)

    for name, winner_offset in (("a-wins", 0), ("b-wins", 1)):
        _progress(name, steps=args.trajectory_steps)
        artifact[name.replace("-", "_")] = _run_winner_case(
            case_name=name,
            winner_offset=winner_offset,
            model=model,
            manager=manager,
            store=store,
            checkpoint_report=report,
            s0=s0,
            s0_digest=s0_digest,
            identity_s0=identity_s0,
            tokens_a=tokens_a,
            tokens_b=tokens_b,
            trajectory_steps=args.trajectory_steps,
            continuation_steps=args.continuation_steps,
        )
        artifact["last_completed_phase"] = name
        _atomic_write(args.output, artifact)

    artifact["stale_evaluation"] = _run_stale_evaluation_case(
        model,
        manager,
        s0,
        identity_s0,
        tokens_a,
        tokens_b,
        steps=args.failure_steps,
    )
    artifact["stale_base"] = _run_stale_base_case(
        model,
        manager,
        s0,
        identity_s0,
        tokens_a,
        tokens_b,
        steps=args.failure_steps,
    )
    artifact["abort"] = _run_abort_case(manager, s0, identity_s0)
    artifact["last_completed_phase"] = "failure-domains"
    _atomic_write(args.output, artifact)

    artifact["official_oracle"] = _official_oracle(model, processor, report)
    accounting_before_release = manager.accounting()
    active.release()
    store.delete("S0")
    gc.collect()
    mx.clear_cache()
    final_memory = _memory()
    final_accounting = manager.accounting()
    final_snapshot_accounting = store.accounting()
    artifact["accounting_before_release"] = accounting_before_release
    artifact["final_accounting"] = {
        "transaction": final_accounting,
        "snapshot": final_snapshot_accounting,
        "memory": final_memory,
    }

    exact_keys = (
        "full_vocab_logits_exact",
        "all_checkpoint_state_exact",
        "final_state_exact",
        "kda_state_exact",
        "dsa_kv_exact",
        "indexpool_exact",
        "slot_index_metadata_exact",
    )
    winner_rows = (artifact["a_wins"], artifact["b_wins"])
    oracle = artifact["official_oracle"]
    acceptance = {
        "explicit_transaction_state_machine": all(
            row["transaction_state"] == "committed" for row in winner_rows
        )
        and artifact["abort"]["state"] == "aborted",
        "terminal_transactions_fail_closed": all(
            all(row["terminal_rejections"].values()) for row in winner_rows
        )
        and artifact["abort"]["terminal_commit_rejected"],
        "evaluation_policy_is_external": (
            artifact["configuration"]["winner_selection_is_runtime_policy"] is False
        ),
        "evaluation_is_observation_only": all(
            row["evaluation_observation_only"] for row in winner_rows
        ),
        "a_and_b_winner_commit_symmetric": (
            artifact["a_wins"]["winner_offset"] == 0
            and artifact["b_wins"]["winner_offset"] == 1
        ),
        "winner_semantic_state_promoted_exact": all(
            row["commit"]["winner_semantic_digest"]
            == row["commit"]["active_semantic_digest"]
            == row["post_commit"]["state_sha256"]
            for row in winner_rows
        ),
        "winner_owned_storage_promoted_without_copy": all(
            row["commit"]["winner_storage_promoted_without_copy"]
            and row["winner_cache_id_before"] == row["winner_cache_id_after"]
            for row in winner_rows
        ),
        "winner_continuation_full_state_exact": all(
            all(row["continuation_comparison"][key] for key in exact_keys)
            and row["continuation_final_state_exact"]
            for row in winner_rows
        ),
        "sibling_mutation_isolated": all(
            all(row["sibling_isolation"].values()) for row in winner_rows
        ),
        "loser_released_only_after_commit": all(
            row["commit"]["loser_released_bytes"]
            == row["loser_before_commit"]["resident_bytes"]
            for row in winner_rows
        ),
        "old_active_and_loser_stale_references_zero": all(
            row["old_active_stale_entry_count"] == 0
            and row["loser_stale_entry_count"] == 0
            for row in winner_rows
        ),
        "mixed_generation_zero": all(
            row["commit"]["mixed_generation_count"] == 0 for row in winner_rows
        ),
        "stale_evaluation_rejected_atomically": (
            artifact["stale_evaluation"]["rejected"]
            and artifact["stale_evaluation"]["all_state_and_references_unchanged"]
        ),
        "stale_base_rejected_atomically": (
            artifact["stale_base"]["rejected"]
            and artifact["stale_base"]["all_state_and_references_unchanged"]
        ),
        "abort_preserves_active_and_releases_candidates": (
            artifact["abort"]["active_unchanged"]
            and artifact["abort"]["resident_after_abort"] == 0
        ),
        "all_transaction_and_branch_resident_released": (
            final_accounting["transaction_resident_bytes"] == 0
            and final_accounting["branch"]["resident_bytes"] == 0
            and final_accounting["branch"]["branch_count"] == 0
        ),
        "all_snapshot_and_active_resident_released": (
            final_snapshot_accounting["resident_bytes"] == 0
            and final_accounting["active"]["resident_bytes"] == 0
        ),
        "anonymous_allocation_zero": (
            final_accounting["anonymous_allocation_count"] == 0
            and final_snapshot_accounting["anonymous_allocation_count"] == 0
        ),
        "ownership_accounting_balanced": (
            final_accounting["ownership_balance_exact"]
            and final_accounting["branch"]["ownership_balance_bytes"] == 0
            and final_accounting["active"]["ownership_balance_bytes"] == 0
        ),
        "source_cache_released": (
            artifact["source_release"]["stale_entry_reference_count"] == 0
        ),
        "parent_snapshot_immutable": all(row["s0_immutable"] for row in winner_rows),
        "nan_zero": all(
            row[name]["nan_count"] == 0
            for row in winner_rows
            for name in ("committed_continuation", "oracle_continuation")
        ),
        "official_16_128_oracle_exact": (
            oracle["first_16_match"] and oracle["full_128_match"]
        ),
        "peak_within_340gb": final_memory["peak_bytes"] <= MAX_PEAK_BYTES,
        "runtime_scope_unchanged": all(
            value is False for value in artifact["runtime_changes"].values()
        ),
    }
    artifact["acceptance"] = acceptance
    artifact["complete"] = all(acceptance.values())
    artifact["last_completed_phase"] = "complete"
    _atomic_write(args.output, artifact)
    print(json.dumps({"complete": artifact["complete"], "acceptance": acceptance}))
    return 0 if artifact["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
