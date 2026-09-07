import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_shape_specialized_native_moe_repair.py"
METAL = ROOT / "native_execution" / "native_indexer_plan.metal"
PLAN = ROOT / "native_execution" / "native_packed_moe_plan.cpp"
HEADER = ROOT / "native_execution" / "native_packed_moe_plan.h"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-shape-specialized-native-moe-repair-20260907.json"
)


def test_repair_probe_requires_the_recorded_first_divergence():
    source = PROBE.read_text()
    ast.parse(source)
    assert '"step": 29' in source
    assert '"layer": 41' in source
    assert '"stage": "routed_hidden"' in source
    assert "42 * TARGET_STEP" in source
    assert "all_2856_owned_layer_steps_byte_exact" in source
    assert "official_16_128_oracle_exact_both_arms" in source


def test_glm53_routed_kernel_uses_compile_time_oracle_geometry():
    metal = METAL.read_text()
    header = HEADER.read_text()
    plan = PLAN.read_text()
    assert "glm53_native_glm53_packed_selected8_gate_up_swiglu" in metal
    for declaration in (
        "constexpr uint kHiddenSize = 4096u;",
        "constexpr uint kIntermediateSize = 2048u;",
        "constexpr uint kIntermediateScaleRows = 16u;",
        "constexpr uint kHiddenScaleRows = 32u;",
        "constexpr float kSwiGLULimit = 10.0f;",
    ):
        assert declaration in metal
    assert "uses_shape_specialized_routed_gate_up" in header
    assert '"glm53_native_glm53_packed_selected8_gate_up_swiglu"' in plan
    assert '"glm53_native_packed_selected8_gate_up_swiglu"' in plan


def test_repair_remains_probe_only_until_oracle_and_performance_requalification():
    source = PROBE.read_text()
    assert '"probe_only": True' in source
    assert "advance_shape_specialized_native_moe_to_performance_requalification" in source
    for component in (
        "runtime",
        "server",
        "apc",
        "cache_abi",
        "production_kernel_abi",
    ):
        assert f'"{component}": False' in source


def test_shape_specialization_is_archived_as_an_exact_negative_result():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is False
    assert artifact["decision"] == "stop_or_relocalize_native_routed_hidden_repair"
    replay = artifact["owned_activation_replay"]
    assert replay["exact_layer_step_comparisons"] == 1214
    assert replay["specialization"] == {
        "all_fixed_arena_invariants": True,
        "all_shape_specialized": True,
        "plan_count": 42,
    }
    first = replay["first_divergence"]
    assert (first["step"], first["layer"]) == (29, 41)
    assert first["stage_localization"]["first_differing_stage"] == "routed_hidden"
