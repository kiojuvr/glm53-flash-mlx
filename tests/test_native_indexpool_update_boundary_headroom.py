import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_native_indexpool_update_boundary_headroom.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-indexpool-update-boundary-headroom-20260907.json"
)


def test_probe_is_a_strict_counterfactual_not_a_runtime_backend():
    source = PROBE.read_text()
    assert "exact post-update state replay" in source
    assert "_snapshot_pool" in source
    assert "_install_snapshot" in source
    assert "A_tier1_normal_update" in source
    assert "B_precomputed_update_tier1_selection" in source
    assert "with tier1._native_arm" in source
    assert '"profiling_only": True' in source
    for component in ("runtime", "server", "apc", "cache_abi", "kernel_abi"):
        assert f'"{component}": False' in source


def test_probe_has_fixed_full_model_headroom_gate_and_exactness_gate():
    source = PROBE.read_text()
    assert "CONTEXTS = (2_048, 262_144)" in source
    assert "MIN_256K_HEADROOM_MS = 0.75" in source
    assert '"all_logits_and_post_state_byte_exact"' in source
    assert '"256k_update_boundary_headroom_at_least_0_75ms"' in source
    assert '"implement_native_indexpool_update_submission_island"' in source
    assert '"stop_native_indexpool_update_boundary"' in source


def test_replay_validates_boundary_before_state_installation():
    source = PROBE.read_text()
    update = source[source.index("class _UpdateReplay") :]
    validate = update.index("pool.validate_update")
    stale = update.index("precomputed IndexPool state has a stale boundary")
    install = update.index("_install_snapshot(pool, snapshot)")
    assert validate < stale < install


def test_m3ultra_update_boundary_has_enough_full_model_headroom():
    result = json.loads(ARTIFACT.read_text())
    assert result["complete"] is True
    assert result["accepted"] is True
    assert result["decision"] == "implement_native_indexpool_update_submission_island"
    assert result["acceptance"] == {
        "256k_update_boundary_headroom_at_least_0_75ms": True,
        "2k_counterfactual_regression_at_most_1_percent": True,
        "all_counterfactual_logits_and_state_byte_exact": True,
        "official_oracle_exact": True,
        "process_peak_at_most_340GB": True,
        "production_abi_unchanged": True,
    }

    short = result["contexts"]["2048"]
    long = result["contexts"]["262144"]
    assert short["all_logits_and_post_state_byte_exact"] is True
    assert long["all_logits_and_post_state_byte_exact"] is True
    assert short["update_boundary_headroom_ms"] >= 0.75
    assert long["update_boundary_headroom_ms"] >= 0.75
    assert (
        long["timing"]["B_precomputed_update_tier1_selection"]
        ["median_host_submit_ms"]
        < long["timing"]["A_tier1_normal_update"]["median_host_submit_ms"]
    )
    assert result["official_oracle"]["all_full_vocab_logits_hashes_match"] is True
    assert result["process_peak_memory_bytes"] <= 340_000_000_000
    assert all(value is False for value in result["runtime_changes"].values())
