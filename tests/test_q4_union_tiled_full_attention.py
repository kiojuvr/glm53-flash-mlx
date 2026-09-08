import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_q4_union_tiled_full_attention.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-q4-union-tiled-full-attention-20260909.json"
)


def test_native_plan_reuses_union_for_tiled_value_and_physical_av():
    header = (
        ROOT / "native_execution" / "native_projected_qk_union_tile_loop_plan.h"
    ).read_text()
    source = (
        ROOT / "native_execution" / "native_projected_qk_union_tile_loop_plan.cpp"
    ).read_text()
    metal = (ROOT / "native_execution" / "native_indexer_plan.metal").read_text()
    assert "execute_attention" in header
    assert "projected_union_value_tile_" in header
    assert "selected_values_" in header
    assert "glm53_native_scatter_projected_union_value_tile_bfloat16" in metal
    assert '"glm53_native_build_virtual_bk16_map"' in source
    assert '"glm53_native_virtual_bk16_av_bfloat16"' in source


def test_probe_requires_selected_value_and_final_output_exactness():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert '"selected_value_projection"' in source
    assert '"attention_output"' in source
    assert '"q4_boundary_tax_measured_for_q256_gate"' in source


def test_artifact_is_accepted_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert all(artifact["anchors"].values())
    assert all(artifact["checks"].values())
