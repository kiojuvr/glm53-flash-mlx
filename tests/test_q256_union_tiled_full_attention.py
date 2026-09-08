import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_q256_union_tiled_full_attention.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-q256-union-tiled-full-attention-20260909.json"
)


def test_probe_uses_full_q256_output_and_bounded_selected_value_arena():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert '"q256_attention_output"' in source
    assert '"selected_value_samples"' in source
    assert '"scratch_at_most_10gib"' in source
    assert '"q256_full_attention_speedup_at_least_1_20x"' in source


def test_artifact_records_the_q256_query_local_value_rejection_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is False
    assert artifact["decision"] == "stop_or_redesign_q256_union_tiled_value_pass"
    assert artifact["anchors"]["selected_value_samples"] is True
    assert artifact["anchors"]["q256_attention_output"] is False
    assert artifact["speedup"] < 1.0
