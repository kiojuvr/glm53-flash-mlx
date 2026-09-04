from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/probe_state_materialization_cache_write_ownership.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-state-materialization-cache-write-ownership-20260904.json"
)


def test_probe_covers_compact_apc_and_prefill_decode_paths():
    source = SCRIPT.read_text()
    for path in ("compact_cache", "ram_apc_restore", "prefill_decode_transition"):
        assert f'"{path}"' in source
    for symbol in (
        "MaterializationRequest(True, 0)",
        "MaterializationRequest(True, CACHE_WRITE_SENTINEL)",
        "MaterializationRequest(False, None)",
        "invalid_destination_preflight_atomic",
    ):
        assert symbol in source
    for key in (
        "abi",
        "admission",
        "apc_namespace",
        "backend",
        "cache_implementation",
        "server",
    ):
        assert f'"{key}": False' in source


def test_artifact_proves_materialization_and_write_ownership_are_independent():
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == (
        "glm53-state-materialization-cache-write-ownership-v1"
    )
    assert artifact["complete"] is True
    assert artifact["probe_only"] is True
    assert all(artifact["acceptance"].values())
    assert artifact["contract"] == {
        "cache_write_sentinel": -1,
        "identity": (
            "glm53-state-materialization-v1"
            "-value-independent-of-write-owner"
            "-sentinel-minus1"
            "-preflight-before-producer"
        ),
        "invalid_destination_checked_before_producer": True,
        "materialization_requirement_independent_of_cache_ownership": True,
    }

    for fixture in artifact["fixtures"].values():
        assert fixture["producer_calls"] == {
            "a": 1,
            "b": 1,
            "c": 0,
            "invalid": 0,
        }
        assert fixture["arm_c_allocation_free_noop"] is True
        assert fixture["invalid_destination_preflight_atomic"] is True


def test_compact_apc_and_transition_outputs_and_state_are_exact():
    fixtures = json.loads(ARTIFACT.read_text())["fixtures"]
    compact = fixtures["compact_cache"]
    assert compact["arm_a_materialized_and_written"] is True
    assert compact["arm_a_value_exact"] is True
    assert compact["arm_a_cache_exact"] is True
    assert compact["arm_b_materialized_without_write"] is True
    assert compact["arm_b_value_exact"] is True
    assert compact["arm_b_cache_and_accounting_unchanged"] is True

    apc = fixtures["ram_apc_restore"]
    assert apc["arm_a_materialized_and_restored"] is True
    assert apc["arm_a_restore_exact"] is True
    assert apc["arm_b_materialized_without_restore"] is True
    assert apc["arm_b_value_exact"] is True
    assert apc["arm_b_live_cache_unchanged"] is True
    assert apc["snapshot_immutable"] is True

    transition = fixtures["prefill_decode_transition"]
    assert transition["transition_context"] == {
        "prefill_tokens": 32,
        "decode_tokens": 1,
    }
    assert transition["arm_a_projection_exact"] is True
    assert transition["arm_a_selected_indices_exact"] is True
    assert transition["arm_a_cache_exact"] is True
    assert transition["arm_b_materialized_without_write"] is True
    assert transition["arm_b_cache_and_accounting_unchanged"] is True


def test_full_model_oracle_and_ram_apc_remain_exact_without_identity_changes():
    artifact = json.loads(ARTIFACT.read_text())
    full_model = artifact["full_model"]
    assert full_model["moe_backend"] == "packed-decode"
    assert full_model["cache_backend"] == "compact-nope-dsa"
    assert full_model["official_oracle"]["first_16_match"] is True
    assert full_model["official_oracle"]["full_128_match"] is True
    assert full_model["official_oracle"][
        "all_full_vocab_logits_hashes_match"
    ] is True
    assert full_model["ram_apc"] == {
        "all_logits_hashes_match": True,
        "post_state_exact": True,
        "snapshot_immutable": True,
        "steps": 16,
    }
    assert artifact["runtime_changes"] == {
        "abi": False,
        "admission": False,
        "apc_namespace": False,
        "backend": False,
        "cache_implementation": False,
        "server": False,
    }
