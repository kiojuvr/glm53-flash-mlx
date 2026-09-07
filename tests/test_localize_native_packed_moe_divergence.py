import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "localize_native_packed_moe_divergence.py"
HEADER = ROOT / "native_execution" / "native_packed_moe_plan.h"
BINDINGS = ROOT / "native_execution" / "bindings.cpp"


def test_localizer_replays_owned_real_activations_through_persistent_plans():
    source = PROBE.read_text()
    ast.parse(source)
    assert "DEFAULT_TARGET_STEP = 68" in source
    assert "captured_input = mx.array(x)" in source
    assert "captured_output = mx.array(result)" in source
    assert "for step in range(1, target_step + 1)" in source
    assert "native_probe._native_call(" in source
    assert '"exact_layer_step_comparisons_before_divergence"' in source


def test_localizer_records_all_durable_moe_stage_boundaries():
    source = PROBE.read_text()
    header = HEADER.read_text()
    bindings = BINDINGS.read_text()
    for stage in (
        "routed_hidden",
        "routed_down",
        "routed_output",
        "shared_hidden",
        "shared_down",
        "final_output",
    ):
        assert f'"{stage}"' in source
        if stage != "final_output":
            assert f"debug_{stage}" in header
            assert f'"debug_{stage}"' in bindings
    assert '"router_indices_hash"' in source
    assert '"router_scores_hash"' in source
    assert '"first_differing_stage"' in source


def test_localizer_is_diagnostic_only_and_changes_no_production_abi():
    source = PROBE.read_text()
    assert '"diagnostic_only": True' in source
    assert "repair_localized_native_moe_stage" in source
    assert "investigate_full_graph_or_fixed_arena_lifetime" in source
    for component in ("runtime", "server", "apc", "cache_abi", "kernel_abi"):
        assert f'"{component}": False' in source
