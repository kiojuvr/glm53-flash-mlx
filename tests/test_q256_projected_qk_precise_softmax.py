import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_q256_projected_qk_precise_softmax.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-q256-projected-qk-precise-softmax-20260909.json"
)


def test_native_plan_keeps_score_and_precise_softmax_in_owned_arena():
    header = (
        ROOT / "native_execution" / "native_projected_qk_union_tile_loop_plan.h"
    ).read_text()
    source = (
        ROOT / "native_execution" / "native_projected_qk_union_tile_loop_plan.cpp"
    ).read_text()
    assert "attention_probabilities_" in header
    assert "execute_probabilities" in header
    assert '"glm53_native_block_softmax_precise_bfloat16"' in source
    assert "row * row_bytes" in source
    assert "returned_intermediate_tensor_bytes() const { return 0; }" in header


def test_probe_requires_exact_probabilities_and_incremental_wall_gate():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert '"precise_softmax_probabilities"' in source
    assert '"incremental_precise_softmax_at_most_10ms"' in source
    assert '"scratch_at_most_450mib"' in source


def test_artifact_is_accepted_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert all(artifact["anchors"].values())
    assert all(artifact["checks"].values())
