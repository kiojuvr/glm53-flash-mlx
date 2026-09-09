import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "native_execution" / "native_prefill_moe_route_plan.h"
SOURCE = ROOT / "native_execution" / "native_prefill_moe_route_plan.cpp"
METAL = ROOT / "native_execution" / "native_indexer_plan.metal"
SCRIPT = ROOT / "scripts" / "probe_native_prefill_moe_route_plan.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-native-prefill-moe-route-plan-20260909.json"
)


def test_native_route_plan_is_fixed_arena_and_avoids_sorted_hidden_copy():
    header = HEADER.read_text()
    source = SOURCE.read_text()
    metal = METAL.read_text()
    assert "materialized_sorted_hidden_bytes() const { return 0; }" in header
    assert "kQueryRows = 256" in header
    assert "kTopK = 8" in header
    assert "kTileRows = 8" in header
    assert "glm53_native_prefill_moe_count_routes" in source
    assert "glm53_native_prefill_moe_stable_scatter_routes" in source
    assert "One work item owns one expert bucket" in metal
    assert "glm53_native_prefill_moe_tile_descriptors" in metal


def test_native_route_probe_contract_and_artifact():
    ast.parse(SCRIPT.read_text())
    text = SCRIPT.read_text()
    assert "all_route_metadata_byte_exact" in text
    assert "invalid_expert_sets_device_failure_flag" in text
    assert "advance_route_grouping_only_inside_composed_expert_island" in text
    if ARTIFACT.exists():
        artifact = json.loads(ARTIFACT.read_text())
        assert artifact["complete"] is True
        assert artifact["accepted"] is True
        assert all(artifact["checks"].values())
