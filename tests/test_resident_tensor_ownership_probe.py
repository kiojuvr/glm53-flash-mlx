import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/probe_resident_tensor_ownership.py"
ARTIFACT = (
    ROOT / "bench-results/m3ultra512-resident-tensor-ownership-20260902.json"
)


def test_probe_keeps_ownership_and_layout_as_independent_contracts():
    source = SCRIPT.read_text()
    assert '"ownership_layout_independent": True' in source
    assert "borrowed_ephemeral_tensor" in source
    assert "materialize_owned" in source
    assert "unsafe_alias_mutation_reproduced" in source
    assert "all_42_packed_banks_owned_row_major" in source


def test_probe_does_not_change_runtime_policy_or_abi():
    source = SCRIPT.read_text()
    for key in ("abi", "admission", "backend_policy", "cache", "server"):
        assert f'"{key}": False' in source


def test_artifact_proves_resident_lifetime_safety():
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "glm53-resident-tensor-ownership-v1"
    assert artifact["complete"] is True
    assert artifact["ownership_layout_independent"] is True
    assert all(artifact["acceptance"].values())
    assert artifact["staging_fixture"]["unsafe_alias_mutation_reproduced"] is True
    assert artifact["packed_fp8_fixture"]["weight_dtype"] == "uint8"
    assert artifact["packed_fp8_fixture"]["scale_dtype"] == "float32"
    assert artifact["full_model"]["packed_bank_count"] == 42
    assert artifact["full_model"]["official_oracle"]["first_16_match"] is True
    assert artifact["full_model"]["official_oracle"]["full_128_match"] is True
