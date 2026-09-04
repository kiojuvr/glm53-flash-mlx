from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-hybrid-semantic-prefix-snapshot-replay-20260904.json"
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def test_replay_qualification_is_complete_and_all_gates_pass():
    artifact = _artifact()
    assert artifact["schema"] == "glm53-hybrid-semantic-prefix-snapshot-replay-v1"
    assert artifact["snapshot_contract_schema"] == (
        "glm53-hybrid-semantic-prefix-snapshot-v1-ram-owned-transactional"
    )
    assert artifact["complete"] is True
    assert artifact["last_completed_phase"] == "complete"
    assert artifact["decision"] == (
        "hybrid_semantic_prefix_snapshot_replay_qualified"
    )
    assert all(artifact["acceptance"].values())
    assert artifact["backend"] == {
        "moe": "packed-decode",
        "cache": "compact-nope-dsa",
    }
    assert not any(artifact["runtime_changes"].values())


def test_eight_4k_replays_and_nested_2k_replay_are_byte_exact():
    artifact = _artifact()
    config = artifact["configuration"]
    assert config == {
        "checkpoint_interval": 256,
        "decode_model_forward_calls": 38_912,
        "expected_checkpoints_per_replay": 16,
        "model_forward_calls": 38_913,
        "model_token_positions": 39_168,
        "nested_step": 2_048,
        "prefill_model_forward_calls": 1,
        "prefix_tokens": 256,
        "replays": 8,
        "steps": 4_096,
    }
    assert artifact["baseline"]["steps"] == 4_096
    assert len(artifact["baseline"]["logits_hashes"]) == 4_096
    assert len(artifact["baseline"]["checkpoints"]) == 16
    assert artifact["baseline"]["materialization_count"] == 16

    assert len(artifact["replays"]) == 8
    exact_keys = (
        "full_vocab_logits_exact",
        "all_checkpoint_state_exact",
        "final_state_exact",
        "kda_state_exact",
        "dsa_kv_exact",
        "indexpool_exact",
        "slot_index_metadata_exact",
    )
    for index, row in enumerate(artifact["replays"], start=1):
        assert row["replay"] == index
        assert row["checkpoint_count"] == 16
        assert row["trajectory"]["steps"] == 4_096
        assert row["trajectory"]["materialization_count"] == 16
        assert row["comparison"]["nan_count"] == 0
        assert all(row["comparison"][key] for key in exact_keys)
        assert row["logits_sequence_sha256"] == artifact["baseline"][
            "logits_sequence_sha256"
        ]

    nested = artifact["nested_replay"]
    assert nested["checkpoint_count"] == 8
    assert nested["trajectory"]["steps"] == 2_048
    assert nested["trajectory"]["materialization_count"] == 8
    assert nested["comparison"]["nan_count"] == 0
    assert all(nested["comparison"][key] for key in exact_keys)


def test_nested_snapshots_and_all_ten_restores_are_isolated():
    artifact = _artifact()
    capture = artifact["nested_capture"]
    assert capture == {
        "s0_generation": 1,
        "s0_immutable_during_s1_capture": True,
        "s0_s1_storage_alias_count": 0,
        "s1_generation": 2,
    }

    restores = [
        *(row["restore"] for row in artifact["replays"]),
        artifact["nested_replay"]["restore"],
        artifact["deletion"]["final_s0_restore"],
    ]
    assert len(restores) == 10
    for generation, row in enumerate(restores, start=1):
        assert row["cache_reference_replaced"] is True
        assert row["generation_before"] == generation - 1
        assert row["generation_after"] == generation
        assert row["stale_entry_reference_count"] == 0
        assert row["snapshot_resident_before"] == row["snapshot_resident_after"]

    for row in artifact["failure_injections"].values():
        assert row["rejected"] is True
        assert row["live_reference_unchanged"] is True
        assert row["live_generation_unchanged"] is True
        assert row["live_state_unchanged"] is True
        assert row["snapshot_unchanged"] is True


def test_snapshot_and_live_replacement_resources_are_fully_accounted():
    artifact = _artifact()
    final = artifact["deletion"]["final_snapshot_accounting"]
    assert final["capture_count"] == final["delete_count"] == 2
    assert final["allocation_count"] == final["release_count"] == 2
    assert final["cumulative_allocated_bytes"] == 400_439_346
    assert final["cumulative_released_bytes"] == 400_439_346
    assert final["snapshot_count"] == 0
    assert final["snapshot_owned_bytes"] == final["resident_bytes"] == 0
    assert final["anonymous_allocation_count"] == 0

    live = artifact["restore_accounting"]
    assert live["live_generation"] == live["restore_count"] == 10
    assert live["cumulative_replacement_allocated_bytes"] == 2_002_196_730
    assert live["cumulative_replaced_live_bytes"] == 2_002_196_730
    assert live["resident_live_bytes"] == live["peak_live_bytes"] == 200_219_673

    resource = artifact["resource"]
    assert resource["replay_endpoint_active_drift_bytes"] == 96
    assert resource["replay_endpoint_active_drift_bytes"] <= 64 * 2**20
    assert resource["memory"]["peak_bytes"] <= 340_000_000_000
    oracle = artifact["official_oracle"]
    assert oracle["first_16_match"] is True
    assert oracle["full_128_match"] is True
    assert oracle["all_full_vocab_logits_hashes_match"] is True
