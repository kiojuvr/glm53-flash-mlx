import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "requalify_exact_native_packed_moe_performance.py"
EXACT_SOURCE = (
    ROOT
    / "bench-results"
    / "m3ultra512-exact-native-routed-fast-sigmoid-repair-20260907.json"
)
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-exact-native-packed-moe-performance-requalification-20260907.json"
)


def test_requalification_requires_the_completed_exact_repair():
    source = PROBE.read_text()
    ast.parse(source)
    assert "EXACT_REPAIR_ARTIFACT" in source
    assert "advance_exact_native_moe_to_performance_requalification" in source
    assert "exact_layer_step_comparisons\"] != 2_856" in source
    assert "first_divergence\"] is not None" in source
    repair = json.loads(EXACT_SOURCE.read_text())
    assert repair["complete"] is True
    assert repair["accepted"] is True
    assert all(repair["acceptance"].values())


def test_requalification_keeps_the_original_fixed_performance_gates():
    source = PROBE.read_text()
    assert "CONTEXTS = (2_048, 262_144)" in source
    assert "TARGET_2K_TPS = 15.0" in source
    assert "MIN_NATIVE_WALL_SAVING_MS = 0.50" in source
    assert "MIN_NATIVE_HOST_SAVING_MS = 0.50" in source
    assert "MIN_CONTEXT_RETENTION = 0.90" in source
    assert "MAX_PROCESS_PEAK_BYTES = 340_000_000_000" in source
    assert '"performance_only": True' in source


def test_requalification_measures_wall_host_exactness_and_fast_plan_usage():
    source = PROBE.read_text()
    for field in (
        "native_wall_saving_ms",
        "native_host_saving_ms",
        "all_full_vocab_logits_byte_exact",
        "all_generated_tokens_exact",
        "post_state_byte_exact",
        "uses_fast_bf16_routed_sigmoid",
        "fixed_arena_zero_execute_allocation_graph_shape_sync",
    ):
        assert field in source
    assert "keep_exact_native_packed_moe_execution_plan" in source
    assert "stop_exact_native_packed_moe_execution_plan_on_performance" in source


def test_requalification_is_probe_only_and_changes_no_production_abi():
    source = PROBE.read_text()
    assert '"probe_only": True' in source
    for component in ("runtime", "server", "apc", "cache_abi", "kernel_abi"):
        assert f'"{component}": False' in source


def test_performance_result_is_archived_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["exact_repair_source"]["accepted"] is True
    assert artifact["failed_gates"] == [
        name for name, passed in artifact["acceptance"].items() if not passed
    ]
