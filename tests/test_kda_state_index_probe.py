from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_kda_state_index_guards.py"
ARTIFACT = (
    ROOT / "bench-results" / "m3ultra512-kda-state-index-guards-20260902.json"
)


def test_probe_declares_all_state_boundaries_and_runtime_non_changes():
    source = SCRIPT.read_text()
    for boundary in (
        '"read"',
        '"write"',
        '"materialization_source"',
        '"materialized_state_replacement"',
        '"restore_destination"',
        '"conv_read"',
        '"conv_write"',
        '"recurrent_read"',
        '"recurrent_write"',
    ):
        assert boundary in source
    for unchanged in ("abi", "admission", "apc_namespace", "backend", "server"):
        assert f'"{unchanged}": False' in source


def test_committed_artifact_passes_all_guards():
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert all(artifact["acceptance"].values())
    assert artifact["contract"] == {
        "identity": "glm53-kda-state-index-v1-sentinel-minus1",
        "invalid": "index < -1 or index >= capacity",
        "rollback_window": 16,
        "sentinel": -1,
        "valid": "0 <= index < capacity",
    }
    matrix = artifact["boundary_matrix"]
    assert all(row["rejected"] and row["state_unchanged"] for row in matrix["invalid"])
    assert all(row["no_access"] and row["state_unchanged"] for row in matrix["sentinel"])
    assert matrix["rollback_17_rejected_atomically"] is True
    assert all(
        row["rejected"] and row["state_unchanged"]
        for row in matrix["materialized_state_replacement"]
    )
    assert artifact["full_model"]["cache_layout"]["direct_kda_arrays_cache_count"] == 34
    assert artifact["full_model"]["cache_layout"]["compact_kda_arrays_cache_count"] == 34
