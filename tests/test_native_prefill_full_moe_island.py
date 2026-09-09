import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "native_execution" / "native_prefill_moe_plan.h"
SOURCE = ROOT / "native_execution" / "native_prefill_moe_plan.cpp"
SCRIPT = ROOT / "scripts" / "probe_native_prefill_full_moe_island.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-native-prefill-full-moe-island-20260909.json"
)


def test_full_moe_plan_composes_routed_shared_and_final_add():
    header = HEADER.read_text(); source = SOURCE.read_text()
    assert "NativePrefillMoEGateUpPlan routed_plan_" in header
    assert "NativePrefillSharedExpertPlan shared_plan_" in header
    assert "returned_intermediate_tensor_bytes() const { return 0; }" in header
    assert "routed_plan_.execute_routed" in source
    assert "shared_plan_.execute" in source
    assert "glm53_native_add_routed_shared" in source


def test_full_moe_probe_contract_and_artifact():
    ast.parse(SCRIPT.read_text())
    text = SCRIPT.read_text()
    assert "real_checkpoint_full_moe_output_byte_exact" in text
    assert "advance_exact_native_moe_into_prefill_layer_plan" in text
    if ARTIFACT.exists():
        artifact = json.loads(ARTIFACT.read_text())
        assert artifact["complete"] is True
        assert artifact["accepted"] is False
        assert artifact["decision"] == "stop_or_redesign_native_prefill_full_moe"
        assert artifact["checks"]["real_checkpoint_full_moe_output_byte_exact"]
        assert not artifact["checks"]["full_native_moe_speedup_at_least_1_20x"]
