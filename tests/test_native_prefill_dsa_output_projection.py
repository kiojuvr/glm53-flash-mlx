import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "native_execution" / "native_prefill_dsa_output_plan.h"
SOURCE = ROOT / "native_execution" / "native_prefill_dsa_output_plan.cpp"
METAL = ROOT / "native_execution" / "native_indexer_plan.metal"
SCRIPT = ROOT / "scripts" / "probe_native_prefill_dsa_output_projection.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-native-prefill-dsa-output-projection-20260909.json"
)


def test_actual_dsa_output_geometry_and_aot_projection_are_fixed():
    header = HEADER.read_text()
    source = SOURCE.read_text()
    metal = METAL.read_text()
    assert "kHeads = 64" in header
    assert "kValueDim = 256" in header
    assert "kInputSize = kHeads * kValueDim" in header
    assert "glm53_native_prefill_dsa_head_to_row_bfloat16" in source
    assert "glm53_native_prefill_dsa_o_proj_e4m3" in source
    assert "glm53_native_prefill_dsa_o_proj_e4m3" in metal
    assert "constexpr uint kInput = 16384u" in metal


def test_probe_keeps_exactness_and_fixed_speed_gate():
    ast.parse(SCRIPT.read_text())
    text = SCRIPT.read_text()
    assert "official_fp8_o_projection_is_byte_exact" in text
    assert "output_projection_speedup_at_least_1_20x" in text
    if ARTIFACT.exists():
        artifact = json.loads(ARTIFACT.read_text())
        assert artifact["complete"] is True
        assert artifact["checks"]["official_fp8_o_projection_is_byte_exact"]
