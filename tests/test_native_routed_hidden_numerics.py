import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "localize_native_routed_hidden_numerics.py"
METAL = ROOT / "native_execution" / "native_indexer_plan.metal"
HEADER = ROOT / "native_execution" / "native_packed_moe_plan.h"
BINDINGS = ROOT / "native_execution" / "bindings.cpp"
PACKAGE = ROOT / "native_execution" / "glm53_native_execution" / "__init__.py"


def test_diagnostic_targets_the_first_reproducible_routed_hidden_difference():
    source = PROBE.read_text()
    ast.parse(source)
    assert "TARGET_STEP = 29" in source
    assert "TARGET_LAYER = 41" in source
    assert '"source_moe_output_mismatch_reproduced"' in source
    assert '"diagnostic_hidden_matches_plan_hidden"' in source
    assert '"jit_activation_diagnostic_matches_exact_hidden"' in source


def test_diagnostic_exposes_every_durable_projection_activation_boundary():
    source = PROBE.read_text()
    metal = METAL.read_text()
    header = HEADER.read_text()
    bindings = BINDINGS.read_text()
    package = PACKAGE.read_text()
    assert "NativePackedMoERoutedDiagnostic" in header
    assert '"NativePackedMoERoutedDiagnostic"' in bindings
    assert '"NativePackedMoERoutedDiagnostic"' in package
    assert "glm53_native_glm53_packed_selected8_gate_up_swiglu_diagnostic" in metal
    for stage in (
        "gate_projection_bf16",
        "up_projection_bf16",
        "sigmoid_bf16",
        "silu_bf16",
        "activated_hidden_bf16",
    ):
        assert f'"{stage}"' in source
    for output in ("gate_output", "up_output", "sigmoid_output", "silu_output"):
        assert output in metal


def test_diagnostic_is_probe_only_and_has_explicit_next_decisions():
    source = PROBE.read_text()
    assert '"diagnostic_only": True' in source
    assert "repair_native_routed_projection_reduction" in source
    assert "repair_native_routed_sigmoid_rounding" in source
    assert "repair_native_routed_activation_rounding" in source
    assert "stop_unreliable_routed_hidden_diagnostic" in source
    for component in (
        "runtime",
        "server",
        "apc",
        "cache_abi",
        "production_kernel_abi",
    ):
        assert f'"{component}": False' in source
