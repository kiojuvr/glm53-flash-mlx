import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "profile_selected_kv_projection_reuse_frontier.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-selected-kv-projection-reuse-frontier-20260908.json"
)


def test_profile_uses_real_q256_indexer_outputs_and_authoritative_reference():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert "_CaptureIndexer" in source
    assert "EXPECTED_DSA" in source
    assert "m3ultra512-native-prefill-critical-path-20260908.json" in source
    assert "repeat_logits_exact" in source
    assert "repeat_state_exact" in source
    assert "final_logits_matches_authoritative_profile" in source
    assert "state_matches_authoritative_profile" in source


def test_profile_compares_full_per_query_and_block_union_projection_rows():
    source = SCRIPT.read_text()
    assert "BLOCK_ROWS = (1, 4, 8, 16, 32, 64, 128, 256)" in source
    assert '"full_history": context' in source
    assert "projection_rows_by_strategy" in source
    assert "aggregate_minimum_strategy" in source
    assert "compose_full_history_projection_with_native_sparse_attention" in source
    assert "selected_projection_reuse_plan" in source


def test_artifact_selects_q256_union_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert all(artifact["checks"].values())
    assert artifact["decision"] == "implement_union_q256_selected_projection_reuse_plan"
    long = artifact["contexts"][str(320 << 10)]
    assert long["aggregate_minimum_strategy"] == "union_q256"
    assert long["aggregate_minimum_vs_full_history"] < 0.40
