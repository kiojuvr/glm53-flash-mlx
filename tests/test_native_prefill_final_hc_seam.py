import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "native_execution" / "native_prefill_moe_plan.h"
SOURCE = ROOT / "native_execution" / "native_prefill_moe_plan.cpp"
BINDINGS = ROOT / "native_execution" / "bindings.cpp"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-native-final-hc-prefill-layer-execution-20260909.json"
)


def test_indirect_moe_can_finish_directly_in_four_branch_hc_state():
    header = HEADER.read_text()
    source = SOURCE.read_text()
    bindings = BINDINGS.read_text()
    assert "execute_indirect_hc" in header
    assert "hc_output_" in header
    assert "glm53_native_prefill_post_attention_hc_expand" in source
    assert "return finish_hc(residual, post, comb);" in source
    assert "execute_indirect_hc" in bindings


def test_native_final_hc_complete_layer_is_exact_and_keeps_speed_gate():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert artifact["checks"]["complete_layer_output_is_byte_exact"]
    assert artifact["composition"]["native_final_hc_expand"]
    assert artifact["composition"]["mlx_final_hc_expand"] is False
    assert artifact["speedup"] >= 1.20
