import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_native_selected_kv_attention_region.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-selected-kv-attention-region-20260908.json"
)


def test_native_attention_region_prebinds_complete_qk_softmax_av_path():
    header = (ROOT / "native_execution" / "native_selected_v_av_plan.h").read_text()
    source = (ROOT / "native_execution" / "native_selected_v_av_plan.cpp").read_text()
    assert "execute_attention" in header
    for stage in (
        "key_projection_pipeline_",
        "query_scale_pipeline_",
        "qk_pipeline_",
        "mask_pipeline_",
        "softmax_pipeline_",
        "projection_pipeline_",
        "av_pipeline_",
    ):
        assert stage in source
    assert "returned_intermediate_tensor_bytes() const { return 0; }" in header


def test_probe_requires_all_intermediate_anchors_and_region_speedup():
    source = PROBE.read_text()
    assert '"projected_key"' in source
    assert '"projected_value"' in source
    assert '"scaled_query"' in source
    assert '"qk_scores"' in source
    assert '"precise_softmax"' in source
    assert '"32k_complete_attention_speedup_at_least_1_50x"' in source


def test_artifact_is_accepted_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert all(artifact["checks"].values())
    assert artifact["contexts"]["32768"]["all_anchors_byte_exact"] is True
