from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from glm53_flash_mlx.native_prefill_plan import (
    NATIVE_PREFILL_PLAN_ABI,
    NativePrefillPlanError,
    build_native_prefill_execution_plan,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-prefill-critical-path-20260908.json"
)
SCRIPT = ROOT / "scripts" / "define_native_prefill_execution_plan.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-prefill-execution-plan-20260908.json"
)


def _script_module():
    spec = importlib.util.spec_from_file_location("native_prefill_plan_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _profile():
    return json.loads(PROFILE.read_text())


def test_measured_resource_failure_is_preserved_not_relaxed():
    profile = _profile()
    assert profile["accepted"] is False
    assert profile["acceptance"]["process_peak_at_most_340gb"] is False
    assert all(
        passed
        for name, passed in profile["acceptance"].items()
        if name != "process_peak_at_most_340gb"
    )
    plan = build_native_prefill_execution_plan(profile)
    assert plan.measured_profile_peak_bytes == 367_183_048_783
    assert plan.measured_profile_peak_over_budget_bytes == 27_183_048_783


def test_plan_targets_complete_dataflow_not_individual_kernel_promotion():
    plan = build_native_prefill_execution_plan(_profile())
    assert plan.abi == NATIVE_PREFILL_PLAN_ABI
    assert plan.tile_rows == 256
    assert plan.combined_structural_share_320k > 0.95
    assert plan.dsa_32k_to_320k_scaling > 8.0
    assert 0.99 < plan.routed_moe_32k_to_320k_scaling < 1.02
    assert any("partial kernels cannot be promoted" in row for row in plan.invariants)
    assert any("does not return intermediates to MLX" in row for row in plan.invariants)


def test_100_tps_is_checkpoint_with_200_and_300_targets_retained():
    plan = build_native_prefill_execution_plan(_profile())
    targets = {row.tokens_per_second: row for row in plan.throughput_targets}
    assert tuple(targets) == (100, 200, 300)
    assert targets[100].chunk_budget_ms == 2_560.0
    assert targets[200].chunk_budget_ms == 1_280.0
    assert targets[300].chunk_budget_ms == pytest.approx(853.3333333333334)
    assert targets[100].required_speedup_from_320k > 6.5
    assert (
        targets[100].required_structural_region_speedup_if_other_fixed > 9.0
    )
    assert targets[100].fixed_other_region_must_also_improve is False
    assert targets[200].fixed_other_region_must_also_improve is True
    assert targets[300].fixed_other_region_must_also_improve is True
    assert targets[300].required_speedup_from_320k > 19.0


def test_wall_projection_matches_measured_dsa_moe_other_breakdown():
    plan = build_native_prefill_execution_plan(_profile())
    assert plan.measured_320k_ms_per_token == pytest.approx(66.028810)
    assert plan.projected_dsa_ms_per_token_320k == pytest.approx(43.494, abs=0.01)
    assert plan.projected_routed_moe_ms_per_token_320k == pytest.approx(
        19.391, abs=0.01
    )
    assert plan.projected_other_ms_per_token_320k == pytest.approx(3.146, abs=0.01)


def test_non_resource_profile_failure_stops_plan():
    profile = _profile()
    profile["acceptance"]["instrumentation_logits_state_offsets_exact"] = False
    with pytest.raises(NativePrefillPlanError, match="non-resource failures"):
        build_native_prefill_execution_plan(profile)


def test_execution_plan_artifact_is_accepted_without_runtime_promotion():
    artifact = _script_module().build_artifact(_profile())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert artifact["decision"] == "implement_composed_native_prefill_layer_plan"
    assert artifact["source_profile_accepted"] is False
    assert artifact["source_profile_failed_gates"] == [
        "process_peak_at_most_340gb"
    ]
    assert all(artifact["checks"].values())
    assert not any(artifact["runtime_changes"].values())


def test_recorded_execution_plan_artifact_matches_builder():
    if not ARTIFACT.exists():
        pytest.skip("native prefill execution-plan artifact is pending")
    recorded = json.loads(ARTIFACT.read_text())
    expected = json.loads(
        json.dumps(_script_module().build_artifact(_profile()))
    )
    assert recorded == expected
