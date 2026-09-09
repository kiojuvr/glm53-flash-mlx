import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "native_execution" / "native_prefill_moe_gate_up_plan.h"
SOURCE = ROOT / "native_execution" / "native_prefill_moe_gate_up_plan.cpp"
METAL = ROOT / "native_execution" / "native_indexer_plan.metal"
SCRIPT = ROOT / "scripts" / "probe_fused_down_reduce_native_prefill_moe.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-fused-down-reduce-native-prefill-moe-20260909.json"
)


def test_fused_down_reduce_eliminates_the_route_surface_boundary():
    assert "execute_routed_fused" in HEADER.read_text()
    assert "fused_materialized_routed_down_bytes" in HEADER.read_text()
    assert "fused_down_reduce_pipeline_" in HEADER.read_text()
    assert "glm53_native_prefill_moe_fused_down_reduce" in SOURCE.read_text()
    metal = METAL.read_text()
    assert "glm53_native_prefill_moe_fused_down_reduce" in metal
    assert "bfloat16_t contribution" in metal
    assert "running = bfloat16_t(float(running) + float(contribution))" in metal


def test_probe_keeps_fixed_exactness_and_performance_gates():
    ast.parse(SCRIPT.read_text())
    text = SCRIPT.read_text()
    assert "fused_down_reduce_full_moe_is_byte_exact" in text
    assert "full_native_moe_speedup_at_least_1_20x" in text
    assert "fused_boundary_saves_at_least_0_50ms" in text
    if ARTIFACT.exists():
        artifact = json.loads(ARTIFACT.read_text())
        assert artifact["complete"] is True
        assert artifact["checks"]["fused_down_reduce_full_moe_is_byte_exact"]
