import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "profile_native_prefill_ffn_entry_seam.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-native-prefill-ffn-entry-seam-profile-20260909.json"
)


def test_profile_covers_the_complete_post_attention_to_route_boundary():
    ast.parse(SCRIPT.read_text())
    text = SCRIPT.read_text()
    assert "ffn_hc_collapse" in text
    assert "post_attention_norm" in text
    assert "router_logits" in text
    assert "router_selection" in text
    assert "MIN_STANDALONE_WALL_HEADROOM_MS = 0.75" in text


def test_measured_submillisecond_boundary_is_not_a_standalone_native_target():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["checks"]["all_stage_anchors_repeat_byte_exact"]
    assert artifact["maximum_standalone_wall_headroom_ms"] < 0.75
    assert artifact["standalone_native_implementation_warranted"] is False
    assert artifact["decision"] == (
        "stop_standalone_ffn_entry_native_island_and_move_to_cross_layer_plan"
    )
