import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "native_execution" / "native_prefill_moe_gate_up_plan.h"
SOURCE = ROOT / "native_execution" / "native_prefill_moe_gate_up_plan.cpp"
METAL = ROOT / "native_execution" / "native_indexer_plan.metal"
SCRIPT = ROOT / "scripts" / "probe_indirect_native_prefill_moe_dispatch.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-indirect-native-prefill-moe-dispatch-20260909.json"
)


def test_descriptor_count_drives_gate_up_and_down_indirect_dispatch():
    header = HEADER.read_text()
    source = SOURCE.read_text()
    metal = METAL.read_text()
    assert "execute_routed_indirect" in header
    assert "indirect_arguments_" in header
    assert "route_plan_.descriptor_count()" in source
    assert source.count("dispatchThreadgroups(") >= 2
    assert "glm53_native_prefill_moe_build_indirect_arguments" in metal
    assert "descriptor_count[0] * uint(output_rows)" in metal


def test_probe_requires_exact_output_and_complete_layer_sized_saving():
    ast.parse(SCRIPT.read_text())
    text = SCRIPT.read_text()
    assert "indirect_native_is_byte_exact" in text
    assert "MIN_INCREMENTAL_SAVING_MS = 1.70" in text
    assert "MIN_DIRECT_SPEEDUP = 1.20" in text
    if ARTIFACT.exists():
        artifact = json.loads(ARTIFACT.read_text())
        assert artifact["complete"] is True
        assert artifact["checks"]["indirect_native_is_byte_exact"]
