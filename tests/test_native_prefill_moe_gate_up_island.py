import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "native_execution" / "native_prefill_moe_gate_up_plan.h"
SOURCE = ROOT / "native_execution" / "native_prefill_moe_gate_up_plan.cpp"
METAL = ROOT / "native_execution" / "native_indexer_plan.metal"
SCRIPT = ROOT / "scripts" / "probe_native_prefill_moe_gate_up_island.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-native-prefill-moe-gate-up-island-20260909.json"
)


def test_composed_moe_ingress_keeps_routes_internal_and_hidden_indirect():
    header = HEADER.read_text()
    source = SOURCE.read_text()
    metal = METAL.read_text()
    assert "NativePrefillMoERoutePlan route_plan_" in header
    assert "returned_route_metadata_bytes() const { return 0; }" in header
    assert "materialized_sorted_hidden_bytes() const { return 0; }" in header
    assert "route_plan_.execute" in source
    assert "sorted_route_order[first_sorted + row]" in metal
    assert "metal::fast::exp" in metal


def test_composed_moe_ingress_probe_contract_and_artifact():
    ast.parse(SCRIPT.read_text())
    text = SCRIPT.read_text()
    assert "real_checkpoint_gate_up_swiglu_byte_exact" in text
    assert "advance_exact_native_prefill_moe_to_down_reduce_shared" in text
    if ARTIFACT.exists():
        artifact = json.loads(ARTIFACT.read_text())
        assert artifact["complete"] is True
        assert artifact["accepted"] is False
        assert artifact["decision"] == (
            "stop_or_relocalize_native_prefill_moe_gate_up_island"
        )
        assert artifact["checks"]["real_checkpoint_gate_up_swiglu_byte_exact"]
        assert not artifact["checks"]["composed_gate_up_speedup_at_least_1_20x"]
