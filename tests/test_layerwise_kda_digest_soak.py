from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "soak_layerwise_kda_state_digests.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-layerwise-kda-state-digest-screen-20260902.json"
)


def test_soak_script_defines_staged_tiers_events_and_failure_localization():
    source = SCRIPT.read_text()
    for value in ("SCREEN_STEPS = 4_096", "QUALIFICATION_STEPS = 100_000", "EXTENDED_STEPS = 256_000"):
        assert value in source
    for event in (
        "ram-apc-save-load",
        "rollback-replay",
        "rejected-rollback",
    ):
        assert event in source
    for failure_field in (
        "first_divergent_token",
        "previous_exact_checkpoint",
        "materialization_count",
        "slot_index",
        "cumulative_allocated_tokens",
        "lifecycle",
    ):
        assert failure_field in source
    assert "server_admission_bypassed_inside_probe_only" in source
    assert '"process_resume_supported": False' in source


def test_screen_artifact_has_exact_layerwise_authoritative_evidence():
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["tier"] == "screen"
    assert artifact["steps"] == 4_096
    assert artifact["first_divergence"] is None
    assert all(artifact["acceptance"].values())
    assert artifact["summary"]["checkpoint_count"] == 21
    assert artifact["summary"]["periodic_256_checkpoint_count"] == 16
    assert artifact["summary"]["materialization_count"] == 16
    assert artifact["summary"]["authoritative_state_drift_bytes"] == 0
    assert artifact["summary"]["state_leaf_counts"] == [167]
    assert artifact["summary"]["observer_overhead"]["uninterrupted_ratio"] <= 0.01
    assert artifact["summary"]["observer_overhead"]["eventful_ratio"] <= 0.01

    for step, checkpoint in artifact["checkpoints"].items():
        left = checkpoint["uninterrupted"]
        right = checkpoint["eventful"]
        assert int(step) == checkpoint["step"]
        assert checkpoint["layerwise_exact"] is True
        assert checkpoint["lifecycle_accounting_exact"] is True
        assert len(left["layers"]) == len(right["layers"]) == 34
        for left_layer, right_layer in zip(
            left["layers"], right["layers"], strict=True
        ):
            assert left_layer == right_layer
            assert set(
                ("conv_digest", "recurrent_digest", "index_digest")
            ).issubset(left_layer)


def test_screen_exercises_all_event_classes_and_allocation_accounting():
    artifact = json.loads(ARTIFACT.read_text())
    rollback = {
        row["tokens"]: row
        for row in artifact["events"]
        if row["kind"] == "rollback-replay"
    }
    assert set(rollback) == {1, 8, 16}
    assert all(
        row["layerwise_state_exact"]
        and row["final_logits_exact"]
        and row["source_snapshot_immutable"]
        for row in rollback.values()
    )
    rejected = next(
        row for row in artifact["events"] if row["kind"] == "rejected-rollback"
    )
    assert rejected["tokens"] == 17
    assert rejected["kda_rejected"] and rejected["dsa_rejected"]
    assert rejected["state_unchanged"]
    apc = next(row for row in artifact["events"] if row["kind"] == "ram-apc-save-load")
    assert apc["snapshot_immutable_after_next_decode"] is True

    lifecycle = artifact["summary"]["lifecycle"]
    assert lifecycle["anonymous_allocation_count"] == 0
    assert lifecycle["cumulative_allocated_tokens"] > artifact["steps"]
    assert lifecycle["cumulative_allocated_bytes"] > lifecycle["resident_bytes"]
    assert lifecycle["by_lifecycle"]["draft-transient"]["resident_bytes"] == 0
    assert lifecycle["by_lifecycle"]["snapshot-state"]["resident_bytes"] == 0

