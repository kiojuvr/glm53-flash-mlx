import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_composed_native_prefill_layer_execution.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-composed-native-prefill-layer-execution-20260909.json"
)
INDIRECT_ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-composed-indirect-native-prefill-layer-execution-20260909.json"
)


def test_composition_includes_both_accepted_native_regions_and_glue():
    ast.parse(SCRIPT.read_text())
    text = SCRIPT.read_text()
    assert "NativePrefillDSAOutputPlan" in text
    assert "NativePrefillMoEPlan" in text
    assert "mlx_ffn_entry_glue" in text
    assert "mlx_final_hc_expand" in text
    assert "MIN_COMPLETE_LAYER_SPEEDUP = 1.20" in text


def test_artifact_never_weakens_exactness_or_complete_layer_speed_gate():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["checks"]["complete_layer_output_is_byte_exact"]
    assert artifact["checks"]["repeat_is_byte_exact"]
    assert artifact["checks"]["native_component_buffers_are_stable"]
    assert artifact["accepted"] == artifact["checks"][
        "currently_composed_layer_speedup_at_least_1_20x"
    ]


def test_indirect_dispatch_requalification_passes_complete_layer_gate():
    if not INDIRECT_ARTIFACT.exists():
        return
    artifact = json.loads(INDIRECT_ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert artifact["checks"]["complete_layer_output_is_byte_exact"]
    assert artifact["composition"]["device_sized_indirect_moe_dispatch"]
    assert artifact["speedup"] >= 1.20
