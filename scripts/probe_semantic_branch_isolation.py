#!/usr/bin/env python3
"""Qualify first-class eager-copy semantic branch isolation on real weights.

The default screen performs several thousand sequential full-model forwards.
It is intentionally user-launched; the artifact is atomically updated after
each phase, while live branch state itself is RAM-only and not resumable.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import weakref
from dataclasses import replace
from datetime import date
from pathlib import Path

import mlx.core as mx

from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint
from glm53_flash_mlx.semantic_branch import (
    SEMANTIC_BRANCH_SCHEMA,
    SemanticBranchError,
    SemanticBranchManager,
)
from glm53_flash_mlx.semantic_snapshot import (
    SEMANTIC_PREFIX_SNAPSHOT_SCHEMA,
    SemanticCacheHandle,
    SemanticSnapshotStore,
    semantic_cache_digest,
    semantic_cache_resident_bytes,
    semantic_cache_storage_alias_count,
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
    / "m3ultra512-semantic-branch-isolation-20260904.json"
)
DEFAULT_PREFIX_TOKENS = 256
DEFAULT_TWIN_STEPS = 1024
DEFAULT_DIVERGENT_STEPS = 512
DEFAULT_REPLAY_STEPS = 64
ROLLBACK_TOKENS = 16
MAX_PEAK_BYTES = 340_000_000_000


def _trajectory_summary(value: dict) -> dict:
    return {
        key: row
        for key, row in value.items()
        if key not in {"logits_hashes", "checkpoints"}
    } | {"checkpoint_count": len(value["checkpoints"])}


def _branch_observation(manager: SemanticBranchManager, branch_id: int) -> dict:
    branch = manager.branch(branch_id)
    return {
        "state_sha256": branch.state_sha256,
        "identity": branch.identity.descriptor(),
        "position": branch.position,
        "resident_bytes": branch.resident_bytes,
        "accounting": manager.accounting()["by_branch"][str(branch_id)],
        "component_generation_tags": [
            list(row) for row in branch.component_generation_tags()
        ],
    }


def _observation_equal(left: dict, right: dict) -> bool:
    return left == right


def _restore_with_stale_check(
    manager: SemanticBranchManager,
    branch_id: int,
    snapshot,
    *,
    identity,
    rollback_tokens: int | None = None,
) -> dict:
    branch = manager.branch(branch_id)
    old_cache = branch.cache
    cache_reference_before = id(old_cache)
    old_entries = [weakref.ref(entry) for entry in old_cache]
    before = branch.identity.descriptor()
    manager.restore_into_branch(
        branch_id,
        snapshot,
        expected_identity=identity,
        rollback_tokens=rollback_tokens,
    )
    del old_cache
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    after = branch.identity.descriptor()
    return {
        "identity_before": before,
        "identity_after": after,
        "generation_advanced_once": after["generation"] == before["generation"] + 1,
        "cache_reference_replaced": id(branch.cache) != cache_reference_before,
        "stale_entry_reference_count": sum(ref() is not None for ref in old_entries),
    }


def _run_active(
    model,
    manager: SemanticBranchManager,
    branch_id: int,
    token_ids: list[int],
    *,
    start_position: int,
    steps: int,
    nested_step: int | None = None,
    nested_capture=None,
) -> dict:
    manager.activate(branch_id)
    trajectory = _run_trajectory(
        model,
        manager.active_branch,
        token_ids,
        start_position=start_position,
        steps=steps,
        nested_step=nested_step,
        nested_capture=nested_capture,
    )
    manager.commit_active_position(start_position + steps)
    return trajectory


def _failure_isolation(
    manager: SemanticBranchManager,
    *,
    active_branch_id: int,
    other_branch_id: int,
    rollback_snapshot,
    rollback_identity,
    parent_snapshot,
) -> dict:
    rows = {}

    def run(name: str, operation) -> None:
        active_before = manager.active_branch_id
        cache_before = manager.active_cache
        branches_before = {
            branch_id: _branch_observation(manager, branch_id)
            for branch_id in (active_branch_id, other_branch_id)
        }
        parent_before = parent_snapshot.state_sha256
        accounting_before = manager.accounting()
        rejected = False
        message = None
        try:
            operation()
        except (SemanticBranchError, RuntimeError, ValueError) as error:
            rejected = True
            message = str(error)
        rows[name] = {
            "rejected": rejected,
            "message": message,
            "active_branch_unchanged": manager.active_branch_id == active_before,
            "active_cache_reference_unchanged": manager.active_cache is cache_before,
            "all_branch_state_and_accounting_unchanged": all(
                _branch_observation(manager, branch_id) == before
                for branch_id, before in branches_before.items()
            ),
            "parent_snapshot_unchanged": parent_snapshot.state_sha256 == parent_before,
            "manager_accounting_unchanged": manager.accounting() == accounting_before,
        }

    run(
        "activation_validator_failure",
        lambda: manager.activate(
            other_branch_id,
            validator=lambda _: (_ for _ in ()).throw(
                RuntimeError("injected activation failure")
            ),
        ),
    )
    run("unknown_branch_activation", lambda: manager.activate(999_999))
    run(
        "rollback_17",
        lambda: manager.restore_into_branch(
            active_branch_id,
            rollback_snapshot,
            expected_identity=rollback_identity,
            rollback_tokens=17,
        ),
    )
    run(
        "snapshot_identity_mismatch",
        lambda: manager.restore_into_branch(
            active_branch_id,
            rollback_snapshot,
            expected_identity=replace(
                rollback_identity, checkpoint_revision="wrong-revision"
            ),
        ),
    )
    return rows


def _validate_args(args) -> None:
    for name in (
        "prefix_tokens",
        "twin_steps",
        "divergent_steps",
        "replay_steps",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be positive")
    if args.divergent_steps <= ROLLBACK_TOKENS:
        raise ValueError("divergent_steps must exceed the 16-token rollback window")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix-tokens", type=int, default=DEFAULT_PREFIX_TOKENS)
    parser.add_argument("--twin-steps", type=int, default=DEFAULT_TWIN_STEPS)
    parser.add_argument(
        "--divergent-steps", type=int, default=DEFAULT_DIVERGENT_STEPS
    )
    parser.add_argument("--replay-steps", type=int, default=DEFAULT_REPLAY_STEPS)
    args = parser.parse_args()
    _validate_args(args)
    if not mx.metal.is_available():
        raise RuntimeError("semantic branch isolation requires MLX/Metal")

    report = inspect_checkpoint(args.model, require_server_ready=True)
    capacity = args.prefix_tokens + max(
        args.twin_steps, args.divergent_steps + args.replay_steps
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
    twin_tokens = _tokens(args.prefix_tokens + args.twin_steps, vocab)
    branch_tokens_a = _tokens(
        args.prefix_tokens + args.divergent_steps + args.replay_steps, vocab
    )
    branch_tokens_b = list(branch_tokens_a)
    branch_tokens_a[args.prefix_tokens] = (
        branch_tokens_a[args.prefix_tokens] + 17
    ) % vocab
    branch_tokens_b[args.prefix_tokens] = (
        branch_tokens_b[args.prefix_tokens] + 31
    ) % vocab

    source_handle = SemanticCacheHandle(model.make_cache())
    prefix = mx.array(twin_tokens[: args.prefix_tokens], dtype=mx.uint32)[None]
    prefix_output = model(prefix, cache=source_handle.cache)
    mx.eval(prefix_output.logits)
    del prefix_output
    _materialize(source_handle.cache)
    store = SemanticSnapshotStore()
    identity_s0 = _identity(report, twin_tokens, args.prefix_tokens)
    s0 = store.capture(
        source_handle,
        snapshot_id="S0",
        identity=identity_s0,
        absolute_token_position=args.prefix_tokens,
        materialization_epoch=args.prefix_tokens // CHECKPOINT_INTERVAL,
    )
    s0_digest = s0.state_sha256
    manager = SemanticBranchManager()
    branch_a = manager.fork(s0, expected_identity=identity_s0)
    branch_b = manager.fork(s0, expected_identity=identity_s0)
    initial_aliases = {
        "a_b": semantic_cache_storage_alias_count(branch_a.cache, branch_b.cache),
        "a_s0": semantic_cache_storage_alias_count(branch_a.cache, s0._cache),
        "b_s0": semantic_cache_storage_alias_count(branch_b.cache, s0._cache),
    }

    artifact = {
        "schema": SEMANTIC_BRANCH_SCHEMA,
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
            "twin_steps": args.twin_steps,
            "divergent_steps": args.divergent_steps,
            "rollback_tokens": ROLLBACK_TOKENS,
            "replay_steps": args.replay_steps,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "capacity_tokens": capacity,
            "decode_model_forward_calls": (
                3 * args.twin_steps
                + 2 * args.divergent_steps
                + ROLLBACK_TOKENS
                + 4 * args.replay_steps
            ),
            "prefill_model_forward_calls": 1,
        },
        "initial_storage_alias_count": initial_aliases,
        "runtime_changes": {
            "parallel_execution": False,
            "scheduler": False,
            "continuous_batching": False,
            "kernel_abi": False,
            "cache_abi": False,
            "server": False,
            "disk_apc": False,
            "admission": False,
        },
        "acceptance": {},
    }
    _atomic_write(args.output, artifact)

    _progress("twin-baseline", steps=args.twin_steps)
    baseline = _run_trajectory(
        model,
        source_handle,
        twin_tokens,
        start_position=args.prefix_tokens,
        steps=args.twin_steps,
    )
    b_before_a = _branch_observation(manager, branch_b.identity.branch_id)
    twin_a = _run_active(
        model,
        manager,
        branch_a.identity.branch_id,
        twin_tokens,
        start_position=args.prefix_tokens,
        steps=args.twin_steps,
    )
    b_after_a = _branch_observation(manager, branch_b.identity.branch_id)
    a_before_b = _branch_observation(manager, branch_a.identity.branch_id)
    twin_b = _run_active(
        model,
        manager,
        branch_b.identity.branch_id,
        twin_tokens,
        start_position=args.prefix_tokens,
        steps=args.twin_steps,
    )
    a_after_b = _branch_observation(manager, branch_a.identity.branch_id)
    twin_comparison_a = _compare_trajectory(baseline, twin_a)
    twin_comparison_b = _compare_trajectory(baseline, twin_b)
    artifact["twin"] = {
        "baseline": _trajectory_summary(baseline),
        "branch_a": _trajectory_summary(twin_a),
        "branch_b": _trajectory_summary(twin_b),
        "baseline_vs_a": twin_comparison_a,
        "baseline_vs_b": twin_comparison_b,
        "a_b_logits_sequence_exact": (
            twin_a["logits_sequence_sha256"] == twin_b["logits_sequence_sha256"]
        ),
        "b_unchanged_while_a_advanced": _observation_equal(b_before_a, b_after_a),
        "a_unchanged_while_b_advanced": _observation_equal(a_before_b, a_after_b),
        "s0_immutable": s0.state_sha256 == s0_digest,
    }
    reference_bytes = semantic_cache_resident_bytes(source_handle.cache)
    reference_entries = [weakref.ref(entry) for entry in source_handle.cache]
    del source_handle
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    artifact["twin"]["reference_baseline_release"] = {
        "resident_bytes_before_release": reference_bytes,
        "stale_entry_reference_count": sum(
            ref() is not None for ref in reference_entries
        ),
        "memory_after_release": _memory(),
    }
    artifact["last_completed_phase"] = "twin"
    _atomic_write(args.output, artifact)

    restore_a_s0 = _restore_with_stale_check(
        manager, branch_a.identity.branch_id, s0, identity=identity_s0
    )
    restore_b_s0 = _restore_with_stale_check(
        manager, branch_b.identity.branch_id, s0, identity=identity_s0
    )
    rollback_holder = {}

    def capture_rollback(branch, absolute_position):
        manager.commit_active_position(absolute_position)
        identity = _identity(report, branch_tokens_a, absolute_position)
        snapshot = manager.capture_branch_snapshot(
            branch_a.identity.branch_id,
            store,
            snapshot_id="A-rollback",
            identity=identity,
        )
        rollback_holder.update(snapshot=snapshot, identity=identity)

    b_before_divergent_a = _branch_observation(manager, branch_b.identity.branch_id)
    divergent_a = _run_active(
        model,
        manager,
        branch_a.identity.branch_id,
        branch_tokens_a,
        start_position=args.prefix_tokens,
        steps=args.divergent_steps,
        nested_step=args.divergent_steps - ROLLBACK_TOKENS,
        nested_capture=capture_rollback,
    )
    b_after_divergent_a = _branch_observation(manager, branch_b.identity.branch_id)
    final_a_digest = branch_a.state_sha256
    rollback_restore = _restore_with_stale_check(
        manager,
        branch_a.identity.branch_id,
        rollback_holder["snapshot"],
        identity=rollback_holder["identity"],
        rollback_tokens=ROLLBACK_TOKENS,
    )
    rollback_replay = _run_active(
        model,
        manager,
        branch_a.identity.branch_id,
        branch_tokens_a,
        start_position=args.prefix_tokens + args.divergent_steps - ROLLBACK_TOKENS,
        steps=ROLLBACK_TOKENS,
    )
    rollback_comparison = _compare_trajectory(
        divergent_a,
        rollback_replay,
        reference_logits_offset=args.divergent_steps - ROLLBACK_TOKENS,
    )
    rollback_exact = branch_a.state_sha256 == final_a_digest and all(
        rollback_comparison[key]
        for key in (
            "full_vocab_logits_exact",
            "all_checkpoint_state_exact",
            "kda_state_exact",
            "dsa_kv_exact",
            "indexpool_exact",
            "slot_index_metadata_exact",
        )
    )
    identity_sa = _identity(
        report, branch_tokens_a, args.prefix_tokens + args.divergent_steps
    )
    sa = manager.capture_branch_snapshot(
        branch_a.identity.branch_id,
        store,
        snapshot_id="SA",
        identity=identity_sa,
    )
    a_before_divergent_b = _branch_observation(manager, branch_a.identity.branch_id)
    divergent_b = _run_active(
        model,
        manager,
        branch_b.identity.branch_id,
        branch_tokens_b,
        start_position=args.prefix_tokens,
        steps=args.divergent_steps,
    )
    a_after_divergent_b = _branch_observation(manager, branch_a.identity.branch_id)
    identity_sb = _identity(
        report, branch_tokens_b, args.prefix_tokens + args.divergent_steps
    )
    sb = manager.capture_branch_snapshot(
        branch_b.identity.branch_id,
        store,
        snapshot_id="SB",
        identity=identity_sb,
    )
    artifact["divergent"] = {
        "restore_from_s0": {"branch_a": restore_a_s0, "branch_b": restore_b_s0},
        "branch_a": _trajectory_summary(divergent_a),
        "branch_b": _trajectory_summary(divergent_b),
        "first_input_tokens_differ": (
            branch_tokens_a[args.prefix_tokens]
            != branch_tokens_b[args.prefix_tokens]
        ),
        "final_states_differ": (
            divergent_a["final_state_sha256"] != divergent_b["final_state_sha256"]
        ),
        "b_unchanged_while_a_advanced": _observation_equal(
            b_before_divergent_a, b_after_divergent_a
        ),
        "a_unchanged_while_b_advanced": _observation_equal(
            a_before_divergent_b, a_after_divergent_b
        ),
        "rollback_restore": rollback_restore,
        "rollback_replay": _trajectory_summary(rollback_replay),
        "rollback_replay_exact": rollback_exact,
        "s0_immutable": s0.state_sha256 == s0_digest,
    }
    artifact["branch_snapshots"] = {
        "SA": {
            "lineage": manager.snapshot_lineage("SA").descriptor(),
            "state_sha256": sa.state_sha256,
            "resident_bytes": sa.resident_bytes,
        },
        "SB": {
            "lineage": manager.snapshot_lineage("SB").descriptor(),
            "state_sha256": sb.state_sha256,
            "resident_bytes": sb.resident_bytes,
        },
        "storage_alias_count": {
            "sa_sb": semantic_cache_storage_alias_count(sa._cache, sb._cache),
            "sa_s0": semantic_cache_storage_alias_count(sa._cache, s0._cache),
            "sb_s0": semantic_cache_storage_alias_count(sb._cache, s0._cache),
            "sa_branch_a": semantic_cache_storage_alias_count(
                sa._cache, branch_a.cache
            ),
            "sb_branch_b": semantic_cache_storage_alias_count(
                sb._cache, branch_b.cache
            ),
        },
    }
    artifact["last_completed_phase"] = "divergent"
    _atomic_write(args.output, artifact)

    reference_a = _run_active(
        model,
        manager,
        branch_a.identity.branch_id,
        branch_tokens_a,
        start_position=args.prefix_tokens + args.divergent_steps,
        steps=args.replay_steps,
    )
    reference_b = _run_active(
        model,
        manager,
        branch_b.identity.branch_id,
        branch_tokens_b,
        start_position=args.prefix_tokens + args.divergent_steps,
        steps=args.replay_steps,
    )
    restore_b_sb = _restore_with_stale_check(
        manager, branch_b.identity.branch_id, sb, identity=identity_sb
    )
    replay_b = _run_active(
        model,
        manager,
        branch_b.identity.branch_id,
        branch_tokens_b,
        start_position=args.prefix_tokens + args.divergent_steps,
        steps=args.replay_steps,
    )
    comparison_b = _compare_trajectory(reference_b, replay_b)

    manager.activate(branch_b.identity.branch_id)
    b_before_delete_a = _branch_observation(manager, branch_b.identity.branch_id)
    snapshots_before_delete_a = {
        "S0": s0.state_sha256,
        "SA": sa.state_sha256,
        "SB": sb.state_sha256,
    }
    manager.delete_branch(branch_a.identity.branch_id)
    b_after_delete_a = _branch_observation(manager, branch_b.identity.branch_id)
    snapshots_after_delete_a = {
        "S0": s0.state_sha256,
        "SA": sa.state_sha256,
        "SB": sb.state_sha256,
    }
    a2 = manager.fork(sa, expected_identity=identity_sa)
    a2_b_alias_count = semantic_cache_storage_alias_count(a2.cache, branch_b.cache)
    a2_sa_alias_count = semantic_cache_storage_alias_count(a2.cache, sa._cache)
    b_before_a2 = _branch_observation(manager, branch_b.identity.branch_id)
    replay_a2 = _run_active(
        model,
        manager,
        a2.identity.branch_id,
        branch_tokens_a,
        start_position=args.prefix_tokens + args.divergent_steps,
        steps=args.replay_steps,
    )
    comparison_a2 = _compare_trajectory(reference_a, replay_a2)
    b_after_a2 = _branch_observation(manager, branch_b.identity.branch_id)
    artifact["branch_snapshot_replay"] = {
        "branch_a_reference": _trajectory_summary(reference_a),
        "branch_b_reference": _trajectory_summary(reference_b),
        "restore_b": restore_b_sb,
        "branch_b_replay": _trajectory_summary(replay_b),
        "branch_b_replay_comparison": comparison_b,
        "deleted_a_resident_bytes": manager.accounting()["by_branch"]["1"][
            "resident_bytes"
        ],
        "b_unchanged_by_a_delete": _observation_equal(
            b_before_delete_a, b_after_delete_a
        ),
        "snapshots_unchanged_by_a_delete": (
            snapshots_before_delete_a == snapshots_after_delete_a
        ),
        "a2_parent_snapshot_id": a2.identity.parent_snapshot_id,
        "a2_b_storage_alias_count": a2_b_alias_count,
        "a2_sa_storage_alias_count": a2_sa_alias_count,
        "branch_a2_replay": _trajectory_summary(replay_a2),
        "branch_a2_replay_comparison": comparison_a2,
        "b_unchanged_while_a2_advanced": _observation_equal(b_before_a2, b_after_a2),
    }

    manager.activate(branch_b.identity.branch_id)
    failures = _failure_isolation(
        manager,
        active_branch_id=branch_b.identity.branch_id,
        other_branch_id=a2.identity.branch_id,
        rollback_snapshot=rollback_holder["snapshot"],
        rollback_identity=rollback_holder["identity"],
        parent_snapshot=s0,
    )
    artifact["failure_isolation"] = failures
    artifact["branch_accounting_before_delete"] = manager.accounting()
    artifact["snapshot_accounting_before_delete"] = store.accounting()

    manager.deactivate()
    manager.delete_branch(branch_b.identity.branch_id)
    manager.delete_branch(a2.identity.branch_id)
    for snapshot_id in ("A-rollback", "SA", "SB"):
        manager.delete_branch_snapshot(store, snapshot_id)
    store.delete("S0")
    final_branch_accounting = manager.accounting()
    final_snapshot_accounting = store.accounting()
    artifact["final_accounting"] = {
        "branch": final_branch_accounting,
        "snapshot": final_snapshot_accounting,
        "memory": _memory(),
    }
    artifact["official_oracle"] = _official_oracle(model, processor, report)

    exact_keys = (
        "full_vocab_logits_exact",
        "all_checkpoint_state_exact",
        "final_state_exact",
        "kda_state_exact",
        "dsa_kv_exact",
        "indexpool_exact",
        "slot_index_metadata_exact",
    )
    all_failures_atomic = all(
        row["rejected"]
        and row["active_branch_unchanged"]
        and row["active_cache_reference_unchanged"]
        and row["all_branch_state_and_accounting_unchanged"]
        and row["parent_snapshot_unchanged"]
        and row["manager_accounting_unchanged"]
        for row in failures.values()
    )
    oracle = artifact["official_oracle"]
    acceptance = {
        "forked_mutable_storage_alias_zero": all(
            value == 0 for value in initial_aliases.values()
        ),
        "twin_baseline_a_b_byte_exact": all(
            twin_comparison_a[key] and twin_comparison_b[key] for key in exact_keys
        )
        and artifact["twin"]["a_b_logits_sequence_exact"],
        "twin_branch_mutation_isolated": (
            artifact["twin"]["b_unchanged_while_a_advanced"]
            and artifact["twin"]["a_unchanged_while_b_advanced"]
        ),
        "reference_baseline_cache_released": artifact["twin"][
            "reference_baseline_release"
        ]["stale_entry_reference_count"]
        == 0,
        "divergent_branches_are_independent": (
            artifact["divergent"]["first_input_tokens_differ"]
            and artifact["divergent"]["final_states_differ"]
            and artifact["divergent"]["b_unchanged_while_a_advanced"]
            and artifact["divergent"]["a_unchanged_while_b_advanced"]
        ),
        "branch_local_rollback_replay_exact": rollback_exact,
        "rollback_and_activation_failures_are_branch_local": all_failures_atomic,
        "branch_snapshot_lineage_exact": (
            artifact["branch_snapshots"]["SA"]["lineage"]["source_branch_id"]
            == 1
        )
        and artifact["branch_snapshots"]["SB"]["lineage"]["source_branch_id"] == 2,
        "branch_snapshots_owned_and_alias_free": all(
            value == 0
            for value in artifact["branch_snapshots"]["storage_alias_count"].values()
        )
        and artifact["branch_snapshot_replay"]["a2_b_storage_alias_count"] == 0
        and artifact["branch_snapshot_replay"]["a2_sa_storage_alias_count"] == 0,
        "branch_snapshot_restore_replay_exact": all(
            comparison_a2[key] and comparison_b[key] for key in exact_keys
        ),
        "delete_is_branch_local": (
            artifact["branch_snapshot_replay"]["deleted_a_resident_bytes"] == 0
            and artifact["branch_snapshot_replay"]["b_unchanged_by_a_delete"]
            and artifact["branch_snapshot_replay"][
                "snapshots_unchanged_by_a_delete"
            ]
            and artifact["branch_snapshot_replay"][
                "b_unchanged_while_a2_advanced"
            ]
        ),
        "all_branch_storage_released": (
            final_branch_accounting["branch_count"] == 0
            and final_branch_accounting["resident_bytes"] == 0
            and final_branch_accounting["snapshot_bytes_by_lineage"] == {}
            and final_branch_accounting["branch_create_count"]
            == final_branch_accounting["branch_delete_count"]
            and final_branch_accounting["cumulative_allocated_bytes"]
            == final_branch_accounting["cumulative_released_bytes"]
        ),
        "mixed_branch_generation_zero": (
            artifact["branch_accounting_before_delete"][
                "mixed_component_generation_count"
            ]
            == 0
            and final_branch_accounting["mixed_component_generation_count"] == 0
        ),
        "all_snapshot_storage_released": (
            final_snapshot_accounting["snapshot_count"] == 0
            and final_snapshot_accounting["resident_bytes"] == 0
            and final_snapshot_accounting["cumulative_allocated_bytes"]
            == final_snapshot_accounting["cumulative_released_bytes"]
        ),
        "anonymous_allocation_zero": (
            final_branch_accounting["anonymous_allocation_count"] == 0
            and final_snapshot_accounting["anonymous_allocation_count"] == 0
        ),
        "nan_invalid_access_metal_error_zero": (
            baseline["nan_count"]
            + twin_a["nan_count"]
            + twin_b["nan_count"]
            + divergent_a["nan_count"]
            + divergent_b["nan_count"]
            + reference_a["nan_count"]
            + reference_b["nan_count"]
            + replay_a2["nan_count"]
            + replay_b["nan_count"]
            == 0
        ),
        "peak_memory_within_340gb": artifact["final_accounting"]["memory"][
            "peak_bytes"
        ]
        <= MAX_PEAK_BYTES,
        "official_16_token_oracle_exact": oracle["first_16_match"],
        "official_128_token_oracle_exact": oracle["full_128_match"],
        "runtime_server_cache_abi_scheduler_unchanged": not any(
            artifact["runtime_changes"].values()
        ),
    }
    artifact["acceptance"] = acceptance
    artifact["complete"] = all(acceptance.values())
    artifact["last_completed_phase"] = "complete"
    artifact["decision"] = (
        "first_class_semantic_branch_isolation_defined"
        if artifact["complete"]
        else "first_class_semantic_branch_isolation_not_defined"
    )
    _atomic_write(args.output, artifact)
    _progress("complete", accepted=artifact["complete"], output=str(args.output))
    return 0 if artifact["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
