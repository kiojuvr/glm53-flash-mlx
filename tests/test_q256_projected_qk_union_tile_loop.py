import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_q256_projected_qk_union_tile_loop.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-q256-projected-qk-union-tile-loop-20260909.json"
)


def test_native_tile_loop_admits_only_q4_or_full_q256_geometry():
    header = (
        ROOT / "native_execution" / "native_projected_qk_union_tile_loop_plan.h"
    ).read_text()
    source = (
        ROOT / "native_execution" / "native_projected_qk_union_tile_loop_plan.cpp"
    ).read_text()
    metal = (ROOT / "native_execution" / "native_indexer_plan.metal").read_text()
    assert "attention_query_rows_" in header
    assert "value != 4 && value != 256" in source
    assert "row < attention_query_rows_" in source
    assert "query_rows" in metal
    assert "glm53_native_prepare_union_prefill_attention_query_bfloat16" in metal


def test_probe_requires_full_edge_exactness_and_no_selected_key():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert '"q256_qk_scores"' in source
    assert "525056" in source
    assert '"selected_key_never_materialized"' in source
    assert '"end_to_end_speedup_at_least_1_20x"' in source


def test_artifact_is_accepted_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert all(artifact["anchors"].values())
    assert all(artifact["checks"].values())
