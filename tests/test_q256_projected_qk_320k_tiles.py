import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_q256_projected_qk_320k_tiles.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-q256-projected-qk-320k-tiles-20260909.json"
)


def test_320k_probe_compares_planned_32k_and_65k_tiles():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert "PHYSICAL_K = 320 << 10" in source
    assert "TILE_ROWS = (32768, 65536)" in source
    assert "partial_final_tile_projected_row" in source
    assert '"qk_scores"' in source
    assert '"selected_key_never_materialized"' in source


def test_artifact_is_accepted_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert set(artifact["candidates"]) == {"32768", "65536"}
    assert all(artifact["checks"].values())
    for row in artifact["candidates"].values():
        assert all(row["anchors"].values())
