from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-semantic-branch-isolation-20260904.json"
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


EXACT_KEYS = (
    "full_vocab_logits_exact",
    "all_checkpoint_state_exact",
    "final_state_exact",
    "kda_state_exact",
    "dsa_kv_exact",
    "indexpool_exact",
    "slot_index_metadata_exact",
)


def test_semantic_branch_isolation_artifact_passes_every_gate():
    artifact = _artifact()
    assert artifact["schema"] == "glm53-semantic-branch-v1-eager-owned-isolated"
    assert artifact["snapshot_schema"] == (
        "glm53-hybrid-semantic-prefix-snapshot-v1-ram-owned-transactional"
    )
    assert artifact["complete"] is True
    assert artifact["last_completed_phase"] == "complete"
    assert artifact["decision"] == "first_class_semantic_branch_isolation_defined"
    assert all(artifact["acceptance"].values())
    assert artifact["backend"] == {
        "moe": "packed-decode",
        "cache": "compact-nope-dsa",
    }
    assert not any(artifact["runtime_changes"].values())


def test_twin_branches_match_baseline_at_all_four_checkpoints():
    artifact = _artifact()
    assert artifact["configuration"] == {
        "capacity_tokens": 1280,
        "checkpoint_interval": 256,
        "decode_model_forward_calls": 4368,
        "divergent_steps": 512,
        "prefill_model_forward_calls": 1,
        "prefix_tokens": 256,
        "replay_steps": 64,
        "rollback_tokens": 16,
        "twin_steps": 1024,
    }
    assert artifact["initial_storage_alias_count"] == {
        "a_b": 0,
        "a_s0": 0,
        "b_s0": 0,
    }
    twin = artifact["twin"]
    for trajectory in ("baseline", "branch_a", "branch_b"):
        assert twin[trajectory]["steps"] == 1024
        assert twin[trajectory]["checkpoint_count"] == 4
        assert twin[trajectory]["materialization_count"] == 4
        assert twin[trajectory]["nan_count"] == 0
    for comparison in ("baseline_vs_a", "baseline_vs_b"):
        assert all(twin[comparison][key] for key in EXACT_KEYS)
        assert twin[comparison]["checkpoint_mismatch_positions"] == []
        assert twin[comparison]["logits_mismatch_steps"] == []
    assert twin["a_b_logits_sequence_exact"] is True
    assert twin["a_unchanged_while_b_advanced"] is True
    assert twin["b_unchanged_while_a_advanced"] is True
    assert twin["s0_immutable"] is True
    assert twin["reference_baseline_release"]["resident_bytes_before_release"] > 0
    assert twin["reference_baseline_release"]["stale_entry_reference_count"] == 0


def test_divergent_branch_mutation_rollback_and_snapshot_replay_are_isolated():
    artifact = _artifact()
    divergent = artifact["divergent"]
    assert divergent["first_input_tokens_differ"] is True
    assert divergent["final_states_differ"] is True
    assert divergent["a_unchanged_while_b_advanced"] is True
    assert divergent["b_unchanged_while_a_advanced"] is True
    assert divergent["rollback_replay_exact"] is True
    assert divergent["s0_immutable"] is True
    assert divergent["branch_a"]["checkpoint_count"] == 2
    assert divergent["branch_b"]["checkpoint_count"] == 2

    snapshots = artifact["branch_snapshots"]
    assert snapshots["SA"]["lineage"] == {
        "absolute_token_position": 768,
        "parent_snapshot_id": "A-rollback",
        "snapshot_id": "SA",
        "source_branch_generation": 2,
        "source_branch_id": 1,
    }
    assert snapshots["SB"]["lineage"] == {
        "absolute_token_position": 768,
        "parent_snapshot_id": "S0",
        "snapshot_id": "SB",
        "source_branch_generation": 1,
        "source_branch_id": 2,
    }
    assert all(value == 0 for value in snapshots["storage_alias_count"].values())

    replay = artifact["branch_snapshot_replay"]
    for comparison in ("branch_a2_replay_comparison", "branch_b_replay_comparison"):
        assert all(replay[comparison][key] for key in EXACT_KEYS)
    assert replay["a2_parent_snapshot_id"] == "SA"
    assert replay["a2_b_storage_alias_count"] == 0
    assert replay["a2_sa_storage_alias_count"] == 0
    assert replay["deleted_a_resident_bytes"] == 0
    assert replay["b_unchanged_by_a_delete"] is True
    assert replay["b_unchanged_while_a2_advanced"] is True
    assert replay["snapshots_unchanged_by_a_delete"] is True


def test_branch_generations_and_all_failure_domains_are_atomic():
    artifact = _artifact()
    restores = [
        artifact["divergent"]["restore_from_s0"]["branch_a"],
        artifact["divergent"]["restore_from_s0"]["branch_b"],
        artifact["divergent"]["rollback_restore"],
        artifact["branch_snapshot_replay"]["restore_b"],
    ]
    for row in restores:
        assert row["cache_reference_replaced"] is True
        assert row["generation_advanced_once"] is True
        assert row["identity_after"]["generation"] == (
            row["identity_before"]["generation"] + 1
        )
        assert row["stale_entry_reference_count"] == 0

    failures = artifact["failure_isolation"]
    assert set(failures) == {
        "activation_validator_failure",
        "unknown_branch_activation",
        "rollback_17",
        "snapshot_identity_mismatch",
    }
    for row in failures.values():
        assert row["rejected"] is True
        assert row["active_branch_unchanged"] is True
        assert row["active_cache_reference_unchanged"] is True
        assert row["all_branch_state_and_accounting_unchanged"] is True
        assert row["parent_snapshot_unchanged"] is True
        assert row["manager_accounting_unchanged"] is True


def test_branch_and_snapshot_resources_are_exact_and_fully_released():
    artifact = _artifact()
    before = artifact["branch_accounting_before_delete"]
    assert before["branch_count"] == 2
    assert before["branch_create_count"] == 3
    assert before["branch_delete_count"] == 1
    assert before["branch_restore_count"] == 4
    assert before["mixed_component_generation_count"] == 0
    assert before["resident_bytes"] == sum(
        before["resident_bytes_by_branch"].values()
    )
    assert before["resident_bytes"] == 326_350_386
    assert before["peak_bytes"] == 326_350_386
    assert before["anonymous_allocation_count"] == 0

    final_branch = artifact["final_accounting"]["branch"]
    assert final_branch["active_branch_id"] is None
    assert final_branch["branch_count"] == 0
    assert final_branch["branch_create_count"] == final_branch[
        "branch_delete_count"
    ] == 3
    assert final_branch["resident_bytes"] == 0
    assert final_branch["snapshot_bytes_by_lineage"] == {}
    assert final_branch["cumulative_allocated_bytes"] == 1_142_226_351
    assert final_branch["cumulative_released_bytes"] == 1_142_226_351
    assert all(
        value == 0 for value in final_branch["resident_bytes_by_branch"].values()
    )

    final_snapshot = artifact["final_accounting"]["snapshot"]
    assert final_snapshot["capture_count"] == final_snapshot["delete_count"] == 4
    assert final_snapshot["resident_bytes"] == 0
    assert final_snapshot["snapshot_count"] == 0
    assert final_snapshot["cumulative_allocated_bytes"] == 652_700_772
    assert final_snapshot["cumulative_released_bytes"] == 652_700_772
    assert final_snapshot["anonymous_allocation_count"] == 0
    assert artifact["final_accounting"]["memory"]["peak_bytes"] <= 340_000_000_000

    oracle = artifact["official_oracle"]
    assert oracle["first_16_match"] is True
    assert oracle["full_128_match"] is True
    assert oracle["all_full_vocab_logits_hashes_match"] is True
