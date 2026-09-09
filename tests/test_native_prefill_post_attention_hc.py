import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "native_execution" / "native_prefill_dsa_output_plan.h"
SOURCE = ROOT / "native_execution" / "native_prefill_dsa_output_plan.cpp"
METAL = ROOT / "native_execution" / "native_indexer_plan.metal"
SCRIPT = ROOT / "scripts" / "probe_native_prefill_post_attention_hc.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-native-prefill-post-attention-hc-20260909.json"
)


def test_post_attention_hc_is_part_of_the_fixed_native_plan():
    header = HEADER.read_text()
    source = SOURCE.read_text()
    metal = METAL.read_text()
    assert "execute_hc" in header
    assert "hc_output_" in header
    assert "glm53_native_prefill_post_attention_hc_expand" in source
    assert "glm53_native_prefill_post_attention_hc_expand" in metal
    assert "comb[(size_t(row) * kBranches + source)" in metal
    assert "post[size_t(row) * kBranches + branch]" in metal


def test_probe_anchors_projection_and_requires_exact_hc_speedup():
    ast.parse(SCRIPT.read_text())
    text = SCRIPT.read_text()
    assert "official_fp8_o_projection_anchor_is_byte_exact" in text
    assert "post_attention_hc_output_is_byte_exact" in text
    assert "post_attention_hc_speedup_at_least_1_20x" in text
    if ARTIFACT.exists():
        artifact = json.loads(ARTIFACT.read_text())
        assert artifact["complete"] is True
        assert artifact["checks"][
            "official_fp8_o_projection_anchor_is_byte_exact"
        ]
