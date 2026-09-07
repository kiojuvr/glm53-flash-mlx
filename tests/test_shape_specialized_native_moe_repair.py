import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_shape_specialized_native_moe_repair.py"
METAL = ROOT / "native_execution" / "native_indexer_plan.metal"
PLAN = ROOT / "native_execution" / "native_packed_moe_plan.cpp"
HEADER = ROOT / "native_execution" / "native_packed_moe_plan.h"


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
