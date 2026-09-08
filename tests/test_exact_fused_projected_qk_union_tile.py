import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_exact_fused_projected_qk_union_tile.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-exact-fused-projected-qk-union-tile-20260908.json"
)


def test_native_plan_projects_once_and_consumes_exact_qk_in_one_scope():
    header = (ROOT / "native_execution" / "native_projected_qk_union_tile_plan.h").read_text()
    source = (ROOT / "native_execution" / "native_projected_qk_union_tile_plan.cpp").read_text()
    metal = (ROOT / "native_execution" / "native_indexer_plan.metal").read_text()
    assert "projected_union_key_" in header
    assert "attention_scores_" in header
    assert source.count("set_compute_pipeline_state(projection_pipeline_)") == 1
    assert "selected_key_" not in header
    assert "glm53_native_projected_union_qk_bfloat16" in metal
    assert "projected_qk_pipeline_" in source
    assert "host_synchronization_count() const { return 0; }" in header


def test_probe_requires_projection_and_qk_bits_plus_structural_gate():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert '"projected_union_key"' in source
    assert '"qk_scores"' in source
    assert '"materialized_selected_key_bytes": 0' in source
    assert '"structural_speedup_at_least_0_90x"' in source


def test_artifact_is_accepted_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert all(artifact["anchors"].values())
    assert all(artifact["checks"].values())
