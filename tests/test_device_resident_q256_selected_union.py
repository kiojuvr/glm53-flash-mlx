import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_device_resident_q256_selected_union.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-device-resident-q256-selected-union-20260908.json"
)


def test_union_plan_uses_fixed_device_bitset_prefix_and_mapping_arena():
    header = (ROOT / "native_execution" / "native_selected_union_plan.h").read_text()
    source = (ROOT / "native_execution" / "native_selected_union_plan.cpp").read_text()
    metal = (ROOT / "native_execution" / "native_indexer_plan.metal").read_text()
    for name in (
        "membership_words_", "block_counts_", "block_prefix_",
        "union_indices_", "physical_to_union_", "query_union_slots_",
        "union_count_",
    ):
        assert name in header
    for kernel in (
        "mark_selected_union",
        "count_selected_union_blocks", "prefix_selected_union_blocks",
        "scatter_selected_union", "map_queries_to_selected_union",
    ):
        assert f"glm53_native_{kernel}" in metal
        assert f'"glm53_native_{kernel}"' in source
    assert "host_synchronization_count() const { return 0; }" in header
    assert "clear_selected_union" not in source
    assert "clear_selected_union" not in metal


def test_probe_requires_exact_replacement_and_bounded_320k_wall():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert "replacement_membership_exact" in source
    assert "union_indices_byte_exact" in source
    assert "query_union_slots_byte_exact" in source
    assert '"320k_union_build_at_most_5ms"' in source


def test_artifact_is_accepted_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert all(artifact["checks"].values())
