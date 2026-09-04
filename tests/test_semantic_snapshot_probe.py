from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/probe_hybrid_semantic_prefix_snapshot_contract.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-hybrid-semantic-prefix-snapshot-contract-20260904.json"
)


def _artifact():
    return json.loads(ARTIFACT.read_text())


def test_probe_is_ram_only_contract_work_and_does_not_change_runtime_policy():
    source = SCRIPT.read_text()
    assert "SemanticSnapshotStore" in source
    assert "capture_observation_only" in source
    assert "capture_mutate_restore_replay_exact" in source
    assert "same_snapshot_second_restore_exact" in source
    assert '"server_api": False' in source
    assert '"disk_apc": False' in source
    assert '"cache_abi": False' in source
    assert '"backend": False' in source
    assert '"admission": False' in source


def test_artifact_passes_all_semantic_snapshot_contract_gates():
    artifact = _artifact()
    assert artifact["schema"] == (
        "glm53-hybrid-semantic-prefix-snapshot-v1-ram-owned-transactional"
    )
    assert artifact["complete"] is True
    assert artifact["probe_only"] is True
    assert all(artifact["acceptance"].values())
    assert artifact["decision"] == (
        "hybrid_semantic_prefix_snapshot_contract_ready_for_replay_qualification"
    )
    assert artifact["backend"] == {
        "moe": "packed-decode",
        "cache": "compact-nope-dsa",
    }


def test_all_materialization_boundaries_and_64_step_continuations_are_exact():
    screen = _artifact()["screen"]
    assert screen["boundaries"] == [1, 255, 256, 257, 1023, 1024]
    assert screen["continuation_steps"] == 64
    assert screen["mutation_steps"] == 8
    rows = screen["rows"]
    assert [row["position"] for row in rows] == screen["boundaries"]
    for row in rows:
        assert row["materialization_epoch"] == row["position"] // 256
        assert row["full_vocab_logits_steps_compared"] == 64
        assert row["nan_count"] == 0
        assert all(
            row[key]
            for key in (
                "capture_observation_only",
                "capture_only_continuation_exact",
                "mutated_state_differs",
                "capture_mutate_restore_replay_exact",
                "same_snapshot_second_restore_exact",
                "snapshot_immutable",
                "restored_source_exact",
                "kda_state_exact",
                "dsa_kv_exact",
                "indexpool_exact",
                "slot_index_metadata_exact",
            )
        )
        boundary = row["snapshot_descriptor"]["boundary"]
        assert boundary["absolute_token_position"] == row["position"]
        assert boundary["logical_prefix_length"] == row["position"]
        assert boundary["materialization_epoch"] == row["materialization_epoch"]
        assert len(boundary["kv_logical_extents"]) == 11
        assert len(boundary["indexpool_logical_extents"]) == 11
        assert all(value == row["position"] for _, value in boundary["kv_logical_extents"])
        assert all(
            value == row["position"]
            for _, value in boundary["indexpool_logical_extents"]
        )


def test_snapshot_storage_is_owned_separate_and_fully_released():
    artifact = _artifact()
    rows = artifact["screen"]["rows"]
    for row in rows:
        ownership = row["snapshot_descriptor"]["ownership"]
        assert ownership == {
            "lifecycle": "snapshot-state",
            "retention": "snapshot-owned",
            "tensor_ownership": "owned",
            "physical_storage_alias_with_live": False,
            "prefix_lru_member": False,
            "persistence": "ram-only",
        }
        before = row["accounting_before_delete"]
        after = row["accounting_after_delete"]
        assert before["resident_bytes"] == row["snapshot_descriptor"][
            "resident_bytes"
        ]
        assert after["resident_bytes"] == 0
        assert after["anonymous_allocation_count"] == 0

    accounting = artifact["snapshot_accounting"]
    assert accounting["peak_single_snapshot_resident_bytes"] > 0
    final = accounting["final"]
    assert final["resident_bytes"] == 0
    assert final["allocation_count"] == final["release_count"] == 6
    assert final["cumulative_allocated_bytes"] == final[
        "cumulative_released_bytes"
    ]
    assert final["anonymous_allocation_count"] == 0
    assert final["prefix_lru_member"] is False
    assert final["disk_persistence"] is False


def test_identity_boundary_and_contract_are_complete():
    artifact = _artifact()
    contract = artifact["contract"]
    assert contract["persistence"] == "ram-only"
    assert contract["capture_point"] == "post-forward-quiescent-materialized"
    assert contract["capture_commit"] == "validate-all-then-publish"
    assert contract["restore_commit"] == (
        "validate-all-then-single-cache-reference-swap"
    )
    assert contract["snapshot_consumed_by_restore"] is False
    assert contract["partial_component_restore"] is False
    assert contract["prefix_lru_member"] is False
    assert contract["materialization_interval_tokens"] == 256
    for row in artifact["screen"]["rows"]:
        identity = row["snapshot_descriptor"]["identity"]
        assert identity["checkpoint_revision"] == artifact["checkpoint_revision"]
        assert identity["checkpoint_fingerprint"] == artifact[
            "checkpoint_fingerprint"
        ]
        assert identity["moe_backend"] == "packed-decode"
        assert identity["cache_backend"] == "compact-nope-dsa"
        assert identity["attention_cache_abi"] == contract["attention_cache_abi"]
        assert identity["kda_state_abi"] == contract["kda_state_abi"]
        assert identity["indexpool_abi"] == contract["indexpool_abi"]
        assert len(identity["prefix_token_sha256"]) == 64


def test_official_oracle_and_runtime_scope_remain_exact():
    artifact = _artifact()
    oracle = artifact["official_oracle"]
    assert oracle["first_16_match"] is True
    assert oracle["full_128_match"] is True
    assert oracle["all_full_vocab_logits_hashes_match"] is True
    assert oracle["failures_16"] == []
    assert oracle["failures_128"] == []
    assert artifact["runtime_changes"] == {
        "server_api": False,
        "disk_apc": False,
        "cache_abi": False,
        "backend": False,
        "admission": False,
    }
