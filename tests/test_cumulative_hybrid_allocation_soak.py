import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "soak_cumulative_hybrid_allocation_1m.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-cumulative-hybrid-allocation-1m-20260831.json"
)


def _load_probe():
    spec = importlib.util.spec_from_file_location("hybrid_allocation_soak", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arm_matrix_and_cycle_counts_are_explicit():
    probe = _load_probe()
    assert probe.ARMS == (
        "direct-moe_direct-cache",
        "packed-moe_direct-cache",
        "direct-moe_compact-cache",
        "packed-moe_compact-cache",
    )
    assert probe.expected_cycles(256) == 3907
    assert probe.expected_cycles(4352) == 230
    assert probe.MILESTONES == (0, 100_000, 500_000, 1_000_000)
    assert probe.EXPECTED_LEAVES == {"direct": 112, "compact-nope-dsa": 167}


def test_materialization_schedule_uses_actual_forward_count():
    probe = _load_probe()
    assert probe.expected_materializations(255) == 0
    assert probe.expected_materializations(256) == 1
    assert probe.expected_materializations(511) == 1
    assert probe.expected_materializations(512) == 2


def test_probe_is_process_isolated_atomic_and_non_runtime():
    source = SCRIPT.read_text()
    assert "subprocess.run(command" in source
    assert '"separate_process_per_arm": True' in source
    assert "temporary.replace(path)" in source
    assert "reusable_fingerprints <= {checkpoint.fingerprint}" in source
    assert "cumulative_physical_sequence_capacity" in source
    assert "cumulative_dsa_layer_token_slots" in source
    assert "cumulative_kda_state_bytes" in source
    assert "live_cache_reference_failures" in source
    assert '"backend_default": False' in source
    assert '"cache_abi": False' in source
    assert '"admission": False' in source


def test_artifact_is_complete_when_present():
    if not ARTIFACT.exists():
        pytest.skip("M3 Ultra allocation-soak artifact has not been generated yet")
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "glm53-cumulative-hybrid-allocation-1m-v1"
    if not artifact["complete"]:
        pytest.skip("M3 Ultra allocation soak is still in progress")
    assert artifact["complete"] is True
    assert artifact["checkpoint_revision"]
    assert artifact["checkpoint_fingerprint"]
    assert set(artifact["arms"]) == set(_load_probe().ARMS)
    assert artifact["acceptance"]["accepted"] == all(
        value
        for key, value in artifact["acceptance"].items()
        if key != "accepted"
    )
    for arm in artifact["arms"].values():
        assert arm["complete"] is True
        assert arm["cumulative_physical_sequence_capacity"] >= 1_000_000
        assert arm["acceptance"]["accepted"] is True
        assert arm["scheduled_materialization_count"] == arm[
            "expected_scheduled_materialization_count"
        ]
        assert arm["live_cache_reference_failures"] == 0
