import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_native_selected_v_av_region.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-selected-v-projection-av-region-20260908.json"
)


def test_native_plan_keeps_projected_v_inside_one_encoder_region():
    header = (ROOT / "native_execution" / "native_selected_v_av_plan.h").read_text()
    source = (ROOT / "native_execution" / "native_selected_v_av_plan.cpp").read_text()
    assert "immediately consumes that plan-owned buffer" in header
    assert "returned_intermediate_tensor_bytes() const { return 0; }" in header
    assert "projection_pipeline_" in source
    assert "map_pipeline_" in source
    assert "av_pipeline_" in source
    assert "for (int row = 0; row < kQueryRows; ++row)" in source


def test_probe_requires_projection_output_and_region_performance():
    source = PROBE.read_text()
    assert '"all_selected_v_projection_anchors_byte_exact"' in source
    assert '"all_attention_outputs_byte_exact"' in source
    assert '"32k_region_speedup_at_least_1_20x"' in source
    assert '"runtime_changes": False' in source


def test_artifact_is_accepted_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert all(artifact["checks"].values())
    assert artifact["contexts"]["32768"]["speedup"] >= 1.20
