import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_device_count_projected_qk_union_tile_loop.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-device-count-projected-qk-union-tile-loop-20260909.json"
)


def test_tile_loop_uses_device_union_extent_and_fixed_capacity_topology():
    header = (
        ROOT
        / "native_execution"
        / "native_projected_qk_union_tile_loop_plan.h"
    ).read_text()
    source = (
        ROOT
        / "native_execution"
        / "native_projected_qk_union_tile_loop_plan.cpp"
    ).read_text()
    metal = (ROOT / "native_execution" / "native_indexer_plan.metal").read_text()
    assert "NativeSelectedUnionPlan union_plan_" in header
    assert "for (int tile = 0; tile < tile_count_; ++tile)" in source
    assert "union_count[0]" in metal
    assert "glm53_native_gather_selected_union_latent_tile_bfloat16" in metal
    assert "glm53_native_projected_union_qk_tile_bfloat16" in metal
    assert "materialized_selected_key_bytes() const { return 0; }" in header
    assert "host_synchronization_count() const { return 0; }" in header


def test_probe_requires_exact_multi_tile_anchors_and_wall_gain():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert '"last_projected_union_key_tile"' in source
    assert '"qk_scores"' in source
    assert '"two_capacity_tiles_encoded_without_count_readback"' in source
    assert '"end_to_end_speedup_at_least_1_10x"' in source


def test_artifact_is_accepted_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert all(artifact["anchors"].values())
    assert all(artifact["checks"].values())
