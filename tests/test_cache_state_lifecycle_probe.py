import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/probe_cache_state_lifecycle.py"
ARTIFACT = ROOT / "bench-results/m3ultra512-cache-state-lifecycle-20260902.json"


def test_probe_scope_is_policy_only_and_names_all_lifecycle_classes():
    source = SCRIPT.read_text()
    for name in (
        "TARGET_PREFIX",
        "ACTIVE_RECURRENT",
        "SNAPSHOT_STATE",
        "DRAFT_TRANSIENT",
    ):
        assert name in source
    assert "DRAFT_ROTATIONS = 4096" in source
    assert '"logical_lifecycle_separate_from_physical_storage": True' in source
    for key in (
        "abi",
        "admission",
        "apc_namespace",
        "backend",
        "cache_implementation",
        "server",
    ):
        assert f'"{key}": False' in source


def test_artifact_proves_eviction_isolation_and_accounting():
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "glm53-cache-state-lifecycle-v1"
    assert artifact["complete"] is True
    assert artifact["probe_only"] is True
    assert all(artifact["acceptance"].values())
    simulation = artifact["simulation"]
    assert set(simulation["classes"]) == {
        "target-prefix",
        "active-recurrent",
        "snapshot-state",
        "draft-transient",
    }
    draft = simulation["draft_pressure_isolation"]
    assert draft["target_evictions"] == 0
    assert draft["draft_resident_bytes"] <= draft["draft_budget_bytes"]
    assert draft["draft_evictions"] > 0
    assert simulation["active_recurrent_pinning"]["active_lru_entry_ids"] == []
    assert all(
        simulation["prefix_identity"][
            "same_tokens_different_identity_misses"
        ].values()
    )


def test_artifact_preserves_runtime_oracle_and_ram_apc_semantics():
    artifact = json.loads(ARTIFACT.read_text())
    full_model = artifact["full_model"]
    assert full_model["official_oracle"]["first_16_match"] is True
    assert full_model["official_oracle"]["full_128_match"] is True
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
