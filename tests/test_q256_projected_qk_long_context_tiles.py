import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sweep_q256_projected_qk_long_context_tiles.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-q256-projected-qk-long-context-tile-sweep-20260909.json"
)


def test_tile_rows_are_fixed_plan_geometry_not_runtime_shape_discovery():
    header = (
        ROOT / "native_execution" / "native_projected_qk_union_tile_loop_plan.h"
    ).read_text()
    source = (
        ROOT / "native_execution" / "native_projected_qk_union_tile_loop_plan.cpp"
    ).read_text()
    assert "int tile_rows = 4096" in header
    assert "checked_tile_rows" in source
    assert 'physical K must be in (4096, 512K]' in source
    assert "power of two in [4096, 65536]" in source
    assert "tile_count_((physical_k_ + tile_rows_ - 1) / tile_rows_)" in source
    assert "shape_discovery_count() const { return 0; }" in header


def test_sweep_requires_all_q256_geometries_exact_and_bounded():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert "TILE_ROWS = (4096, 8192, 16384, 32768, 65536)" in source
    assert '"all_tile_geometries_q256_byte_exact"' in source
    assert '"selected_key_never_materialized"' in source
    assert '"best_tile_rows"' in source


def test_artifact_is_accepted_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert all(artifact["checks"].values())
    assert set(artifact["candidates"]) == {
        "4096", "8192", "16384", "32768", "65536"
    }
