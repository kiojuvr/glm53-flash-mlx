import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "native_execution" / "native_q256_dsa_prefill_plan.h"
SOURCE = ROOT / "native_execution" / "native_q256_dsa_prefill_plan.cpp"
SCRIPT = ROOT / "scripts" / "probe_composed_q256_native_dsa_prefill_320k.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-composed-q256-native-dsa-prefill-320k-20260909.json"
)


def test_composed_plan_keeps_qk_softmax_and_value_pass_native():
    header = HEADER.read_text()
    source = SOURCE.read_text()
    assert "NativeProjectedQKUnionTileLoopPlan qk_plan_" in header
    assert "NativeSharedPhysicalValueTilePlan value_plan_" in header
    assert "qk_plan_.execute_probabilities" in source
    assert "value_plan_.execute" in source
    assert "materialized_selected_key_bytes() const { return 0; }" in header
    assert "materialized_query_local_selected_value_bytes() const { return 0; }" in header


def test_320k_probe_contract_and_artifact():
    ast.parse(SCRIPT.read_text())
    text = SCRIPT.read_text()
    assert "320k_qk_softmax_q256_attention_byte_exact" in text
    assert "selected_k_and_query_local_selected_v_never_materialized" in text
    assert "advance_composed_native_dsa_into_prefill_layer_execution_plan" in text
    if ARTIFACT.exists():
        artifact = json.loads(ARTIFACT.read_text())
        assert artifact["complete"] is True
        assert artifact["accepted"] is True
        assert all(artifact["checks"].values())
