import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "native_execution" / "native_prefill_moe_gate_up_plan.h"
SOURCE = ROOT / "native_execution" / "native_prefill_moe_gate_up_plan.cpp"
METAL = ROOT / "native_execution" / "native_indexer_plan.metal"
SCRIPT = ROOT / "scripts" / "probe_native_prefill_moe_routed_island.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-native-prefill-moe-routed-island-20260909.json"
)


def test_routed_island_composes_down_and_exact_expert_order_reduction():
    assert "execute_routed" in HEADER.read_text()
    source = SOURCE.read_text()
    metal = METAL.read_text()
    assert "glm53_native_prefill_moe_bm8_down" in source
    assert "glm53_native_prefill_moe_direct_order_reduce" in source
    assert "inverse_route_order[route]" in metal
    assert "total = bfloat16_t(float(total) + float(contribution))" in metal


def test_routed_island_probe_contract_and_artifact():
    ast.parse(SCRIPT.read_text())
    text = SCRIPT.read_text()
    assert "real_checkpoint_routed_output_byte_exact" in text
    assert "advance_exact_native_routed_moe_to_shared_and_layer_composition" in text
    if ARTIFACT.exists():
        artifact = json.loads(ARTIFACT.read_text())
        assert artifact["complete"] is True
        assert artifact["accepted"] is False
        assert artifact["decision"] == "stop_or_relocalize_native_prefill_routed_moe"
        assert artifact["checks"]["real_checkpoint_routed_output_byte_exact"]
        assert not artifact["checks"]["composed_routed_moe_speedup_at_least_1_20x"]
