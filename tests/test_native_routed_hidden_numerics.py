import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "localize_native_routed_hidden_numerics.py"
METAL = ROOT / "native_execution" / "native_indexer_plan.metal"
HEADER = ROOT / "native_execution" / "native_packed_moe_plan.h"
BINDINGS = ROOT / "native_execution" / "bindings.cpp"
PACKAGE = ROOT / "native_execution" / "glm53_native_execution" / "__init__.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-routed-hidden-numerical-localization-20260907.json"
)


def test_diagnostic_targets_the_first_reproducible_routed_hidden_difference():
    source = PROBE.read_text()
    ast.parse(source)
    assert "TARGET_STEP = 29" in source
    assert "TARGET_LAYER = 41" in source
    assert '"source_moe_output_mismatch_reproduced"' in source
    assert '"diagnostic_hidden_matches_plan_hidden"' in source
    assert '"jit_activation_diagnostic_matches_exact_hidden"' in source
    assert "for value in np.unravel_index" in source


def test_diagnostic_serializes_before_publishing_atomic_artifact():
    source = PROBE.read_text()
    assert "if isinstance(item, np.generic)" in source
    assert "return item.item()" in source
    assert "payload = json.dumps(" in source
    assert source.index("payload = json.dumps(") < source.index(
        "tempfile.NamedTemporaryFile("
    )


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


def test_native_routed_hidden_localization_is_archived_at_sigmoid_bf16():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert artifact["decision"] == "repair_native_routed_sigmoid_rounding"
    evidence = artifact["evidence"]
    assert evidence["diagnostic_hidden_matches_plan_hidden"]
    assert evidence["jit_activation_diagnostic_matches_exact_hidden"]
    assert evidence["first_differing_stage"] == "sigmoid_bf16"
    assert evidence["stages"]["gate_projection_bf16"]["byte_identical"]
    assert evidence["stages"]["up_projection_bf16"]["byte_identical"]
    sigmoid = evidence["stages"]["sigmoid_bf16"]
    assert sigmoid["different_elements"] == 1
    assert sigmoid["first_difference"]["coordinate"] == [3, 2017]
