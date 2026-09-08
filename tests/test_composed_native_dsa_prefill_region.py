import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_composed_native_dsa_prefill_region.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-composed-native-dsa-prefill-region-20260908.json"
)


def test_composed_plan_owns_selection_gather_and_attention_boundary():
    header = (
        ROOT
        / "native_execution"
        / "native_selected_kv_attention_selection_plan.h"
    ).read_text()
    source = (
        ROOT
        / "native_execution"
        / "native_selected_kv_attention_selection_plan.cpp"
    ).read_text()
    metal = (ROOT / "native_execution" / "native_indexer_plan.metal").read_text()
    assert "NativeDSAScoreSelectionPlan score_plan_" in header
    assert "NativeSelectedVProjectionAVPlan attention_plan_" in header
    assert "selected_latent_" in header
    assert "score_plan_.execute" in source
    assert "attention_plan_.execute_attention" in source
    assert "glm53_native_gather_prefill_selected_latent_bfloat16" in metal
    assert "returned_intermediate_tensor_bytes() const { return 0; }" in header


def test_probe_has_exact_anchors_crossover_and_long_context_gate():
    source = PROBE.read_text()
    ast.parse(source)
    for anchor in (
        "score_order_indices", "selected_indices", "selected_valid",
        "selected_latent",
    ):
        assert f'"{anchor}"' in source
    assert '"32k_composed_region_speedup_at_least_2x"' in source
    assert '"2k_requires_direct_crossover"' in source


def test_artifact_is_accepted_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert all(artifact["checks"].values())
