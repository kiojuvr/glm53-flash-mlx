from __future__ import annotations

from pathlib import Path

import json


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stress_state_cumulative_allocation_churn.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-state-cumulative-allocation-churn-screen-20260904.json"
)


def test_churn_probe_decouples_logical_tokens_from_allocation_pressure():
    source = SCRIPT.read_text()
    assert "BASELINE_256K_CUMULATIVE_ALLOCATION_BYTES" in source
    assert "required_churn_cycles" in source
    assert "distributed_churn_schedule" in source
    assert '"logical_tokens_at_most_16k"' in source
    assert '"cumulative_allocation_target_reached"' in source


def test_churn_probe_covers_exact_and_rejected_state_transitions():
    source = SCRIPT.read_text()
    for operation in (
        "wrong-cache-identity-restore",
        "invalid-kda-state-index",
        "ephemeral-resident-promotion",
        "rollback-17",
    ):
        assert operation in source
    for gate in (
        "all_full_vocab_logits_exact",
        "all_34_kda_layer_digests_exact",
        "all_full_cache_digests_exact",
        "apc_capture_restore_exact",
        "rollback_1_8_16_exact",
        "all_rejected_operations_atomic",
        "snapshot_owned_storage_immutable",
    ):
        assert gate in source


def test_churn_probe_records_failure_by_operation_and_allocation_sequence():
    source = SCRIPT.read_text()
    for field in (
        "operation_sequence",
        "logical_token",
        "allocation_sequence",
        "lifecycle",
        "ownership_state",
        "apc_generation",
        "rollback_depth",
        "resident_bytes_before",
        "resident_bytes_after",
        "cumulative_allocated_bytes",
        "first_state_difference",
    ):
        assert field in source


def test_churn_probe_keeps_runtime_and_server_out_of_scope():
    source = SCRIPT.read_text()
    for key in ('"abi": False', '"backend": False', '"server": False', '"admission": False', '"apc": False'):
        assert key in source


def test_churn_screen_reaches_allocation_target_with_exact_state():
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "glm53-state-cumulative-allocation-churn-v1"
    assert artifact["complete"] is True
    assert artifact["tier"] == "screen"
    assert artifact["logical_tokens"] == artifact["last_completed_step"] == 4_096
    assert all(artifact["acceptance"].values())
    assert artifact["first_divergence"] is None
    assert artifact["nan_count"] == 0
    assert artifact["metal_error"] is None

    summary = artifact["summary"]
    assert summary["actual_model_forwards"] == {
        "uninterrupted": 4_096,
        "eventful": 4_122,
    }
    assert summary["apc_generation_count"] == 124
    assert summary["operation_count"] == 128
    assert summary["event_counts"] == {
        "apc-ownership-churn": 124,
        "rollback-replay": 4,
    }
    assert summary["checkpoint_count"] == 17
    assert summary["materialization_count"] == 16
    assert summary["cumulative_allocated_bytes"] == 51_656_675_634
    assert summary["cumulative_allocated_bytes"] >= 50_000_000_000
    assert summary["cumulative_allocated_tokens"] == 1_122_816
    assert summary["authoritative_state_drift_bytes"] == 0
    assert summary["active_memory_drift_bytes"] == 230_015
    assert summary["active_memory_drift_bytes"] <= 64 * 2**20
    assert summary["peak_memory_bytes"] <= 340_000_000_000
    assert set(summary["rejected_operation_counts"]) == {
        "wrong-cache-identity-restore",
        "invalid-kda-state-index",
        "ephemeral-resident-promotion",
        "rollback-17",
    }
    assert set(summary["rejected_operation_counts"].values()) == {31}


def test_churn_screen_lifecycle_balance_and_temporary_release_are_exact():
    artifact = json.loads(ARTIFACT.read_text())
    lifecycle = artifact["summary"]["lifecycle_accounting"]["end"]
    assert lifecycle["ownership_balance_exact"] is True
    assert lifecycle["anonymous_allocation_count"] == 0
    assert lifecycle["resident_bytes"] == 400_439_346
    assert lifecycle["cumulative_released_bytes"] == 51_256_236_288
    assert lifecycle["cumulative_allocated_bytes"] - lifecycle[
        "cumulative_released_bytes"
    ] == lifecycle["resident_bytes"]
    assert lifecycle["by_lifecycle"]["snapshot-state"]["resident_bytes"] == 0
    assert all(
        value == 0
        for value in lifecycle["by_lifecycle"]["draft-transient"].values()
    )
    for row in lifecycle["by_lifecycle"].values():
        assert (
            row["cumulative_allocated_bytes"]
            + row["transfer_in_bytes"]
            - row["cumulative_released_bytes"]
            - row["transfer_out_bytes"]
            == row["resident_bytes"]
        )


def test_churn_screen_all_checkpoints_and_events_preserve_invariants():
    artifact = json.loads(ARTIFACT.read_text())
    assert all(
        row["layerwise_exact"]
        and row["full_cache_digest_exact"]
        and row["full_vocab_logits_exact"]
        and row["lifecycle_balance_exact"]
        and set(row["state_leaf_count"].values()) == {167}
        for row in artifact["checkpoints"].values()
    )
    rollback = [row for row in artifact["events"] if row["kind"] == "rollback-replay"]
    assert {row["rollback_depth"] for row in rollback} == {1, 8, 16}
    assert all(
        row["state_exact"]
        and row["logits_exact"]
        and row["source_snapshot_immutable"]
        and row["lifecycle_balance_exact"]
        and row["nan_count"] == 0
        for row in rollback
    )
    churn = [
        row for row in artifact["events"] if row["kind"] == "apc-ownership-churn"
    ]
    assert all(
        row["restore_exact"]
        and row["snapshot_owned_storage_immutable"]
        and row["temporary_storage_returned"]
        and row["lifecycle_balance_exact"]
        and all(
            row["rejected_operation"][key]
            for key in (
                "rejected",
                "authoritative_state_unchanged",
                "snapshot_unchanged",
                "accounting_unchanged",
                "bindings_unchanged",
            )
        )
        for row in churn
    )
