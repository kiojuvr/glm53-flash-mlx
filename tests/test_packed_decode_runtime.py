import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "probe_packed_decode_runtime.py"
ARTIFACT = (
    ROOT / "bench-results" / "m3ultra512-packed-decode-runtime-20260831.json"
)


def test_probe_covers_required_correctness_and_performance_frontiers():
    source = SCRIPT.read_text()
    assert "PROMPTS = (1, 16, 128, 256)" in source
    assert "DECODE_STEPS = 4096" in source
    assert "FRONTIER_CONTEXTS = (2049, 262144)" in source
    assert "materialization_count_16" in source
    assert "final_kda_dsa_state_exact" in source
    assert "synthetic_256k_continuation_exact" in source
    assert "decode_2k_speedup_at_least_1_12" in source
    assert "decode_256k_speedup_at_least_1_10" in source
    assert "server_ready_at_most_190_seconds" in source


def test_packed_decode_policy_has_no_grouped_prefill_dispatch():
    packed_source = (ROOT / "glm53_flash_mlx" / "packed.py").read_text()
    loader_source = (ROOT / "glm53_flash_mlx" / "loader.py").read_text()
    assert "if flat_x.shape[0] != 1:\n            return super().__call__(x)" in packed_source
    assert "module_type = SortedGroupedFP8MoE if grouped else PackedFP8MoE" in loader_source
    assert 'backend = "packed-grouped" if grouped else "packed-decode"' in loader_source
    assert "def install_packed_decode_moe" in loader_source


def test_packed_decode_apc_identity_is_storage_and_decode_only(monkeypatch):
    from glm53_flash_mlx.server import _disk_cache_descriptor

    monkeypatch.setenv("GLM53_MOE_BACKEND", "packed-decode")
    descriptor = _disk_cache_descriptor("digest")
    assert descriptor["moe_backend"] == "packed-decode"
    assert "packed_bank_abi" in descriptor
    assert "packed_decode_kernel_abi" in descriptor
    assert "grouped_kernel_abi" not in descriptor
    assert "grouped_min_routes" not in descriptor


def test_artifact_is_complete_when_present():
    if not ARTIFACT.exists():
        pytest.skip("M3 Ultra artifact has not been generated yet")
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["moe_backend"] == "packed-decode"
    assert artifact["acceptance"]["accepted"] is True
    assert artifact["packed_decode"]["prefill_256"]["grouped_kernel_calls"] == 0
    assert artifact["packed_decode"]["decode_4096"]["materialization_count"] == 16
    assert artifact["runtime_changes"] == {
        "default_backend": False,
        "prompt_admission": False,
        "cache_abi": False,
        "grouped_backend": False,
    }
