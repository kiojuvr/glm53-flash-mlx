import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_native_virtual_bk16_prefill_av.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-virtual-bk16-prefill-av-20260908.json"
)


def test_native_plan_has_fixed_arena_and_captured_steel_geometry():
    header = (ROOT / "native_execution" / "native_prefill_av_plan.h").read_text()
    source = (ROOT / "native_execution" / "native_prefill_av_plan.cpp").read_text()
    metal = (ROOT / "native_execution" / "native_indexer_plan.metal").read_text()
    assert "kMaximumPackedK = kSelectedWidth * kBK" in header
    assert "The Steel-compatible kernel gathers selected V rows directly" in header
    assert "glm53_native_virtual_bk16_av_bfloat16" in source
    assert "glm53_native_build_virtual_bk16_map" in metal
    assert "glm53_native_virtual_bk16_av_bfloat16" in metal
    assert "simd_any" in metal
    assert "previous_block" in metal


def test_native_probe_requires_exactness_structure_and_operator_gain():
    source = PROBE.read_text()
    assert '"all_8_query_rows_byte_exact"' in source
    assert '"native_lane_map_matches_physical_bk16_contract"' in source
    assert '"no_dynamic_allocation_graph_shape_or_sync"' in source
    assert '"32k_operator_speedup_at_least_1_20x"' in source
    assert '"runtime_changes": False' in source


def test_native_artifact_records_exact_but_slow_boundary_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is False
    assert artifact["decision"] == "stop_or_redesign_native_virtual_bk16_av"
    assert artifact["checks"]["all_8_query_rows_byte_exact"] is True
    assert artifact["checks"]["native_lane_map_matches_physical_bk16_contract"] is True
    assert artifact["checks"]["no_dynamic_allocation_graph_shape_or_sync"] is True
    assert artifact["checks"]["32k_operator_speedup_at_least_1_20x"] is False
    assert artifact["contexts"]["32768"]["scratch_bytes"] <= 1 << 20
