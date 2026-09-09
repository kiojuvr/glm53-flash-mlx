import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "native_execution" / "native_prefill_dominant_region_plan.h"
SOURCE = ROOT / "native_execution" / "native_prefill_dominant_region_plan.cpp"
BINDINGS = ROOT / "native_execution" / "bindings.cpp"
PACKAGE = ROOT / "native_execution" / "glm53_native_execution" / "__init__.py"
SCRIPT = ROOT / "scripts" / "probe_native_prefill_dominant_region_aggregation.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-native-prefill-dominant-region-aggregation-20260909.json"
)


def test_plan_aggregates_the_dominant_11_dsa_and_42_moe_stages():
    header = HEADER.read_text()
    source = SOURCE.read_text()
    assert "kDSALayerCount = 11" in header
    assert "kMoELayerCount = 42" in header
    assert "for (int layer = 0; layer < kDSALayerCount; ++layer)" in source
    assert "for (int layer = 0; layer < kMoELayerCount; ++layer)" in source
    assert "return {dsa_terminal, moe_terminal};" in source


def test_plan_preserves_explicit_reuse_hazards_and_one_call_contract():
    header = HEADER.read_text()
    source = SOURCE.read_text()
    assert "void barrier();" in header
    assert "dsa_body_.execute(" in source
    assert "dsa_output_.execute(" in source
    assert "moe_.execute_indirect(" in source
    assert "dynamic_allocation_count() const { return 0; }" in header
    assert "composed_native_calls() const { return 1; }" in header


def test_binding_package_and_probe_are_fail_closed_about_scope():
    assert "NativePrefillDominantRegionPlan" in BINDINGS.read_text()
    assert "NativePrefillDominantRegionPlan" in PACKAGE.read_text()
    ast.parse(SCRIPT.read_text())
    text = SCRIPT.read_text()
    assert "same_arithmetic_wall_speedup_at_least_1_02x" in text
    assert '"input_dependent_layer_handoffs": False' in text
    assert "aggregation_only_is_insufficient_build_dependency_connected_plan" in text
    if ARTIFACT.exists():
        artifact = json.loads(ARTIFACT.read_text())
        assert artifact["complete"] is True
        assert artifact["checks"][
            "identical_11_dsa_42_moe_terminal_anchors_are_byte_exact"
        ] is True
        assert artifact["scope"]["production_runtime_change"] is False
