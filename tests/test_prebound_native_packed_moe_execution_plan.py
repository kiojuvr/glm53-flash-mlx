import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_prebound_native_packed_moe_execution_plan.py"
HEADER = ROOT / "native_execution" / "native_packed_moe_plan.h"
SOURCE = ROOT / "native_execution" / "native_packed_moe_plan.cpp"
BINDINGS = ROOT / "native_execution" / "bindings.cpp"
REJECTED = (
    ROOT
    / "bench-results"
    / "m3ultra512-exact-native-packed-moe-performance-requalification-20260907.json"
)
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-prebound-native-packed-moe-execution-plan-20260907.json"
)


def test_prebound_plan_binds_static_weights_and_pipelines_once():
    header = HEADER.read_text()
    source = SOURCE.read_text()
    bindings = BINDINGS.read_text()
    assert "void bind_weights(" in header
    assert "mx::array execute_bound(" in header
    assert "bound_weights_" in header
    assert "bound_pipelines_" in header
    assert "pipeline_lookup_count_ += 6" in source
    assert 'throw std::logic_error("native packed MoE weights are already bound")' in source
    assert 'throw std::logic_error("native packed MoE weights are not bound")' in source
    assert '.def("bind_weights"' in bindings
    assert '.def("execute_bound"' in bindings


def test_bound_execute_validates_only_three_dynamic_inputs():
    source = SOURCE.read_text()
    start = source.index("NativePackedMoEDecodePlan::execute_bound(")
    stop = source.index("NativePackedMoEDecodePlan::encode(", start)
    body = source[start:stop]
    assert 'validate_input(x, "x"' in body
    assert 'validate_input(expert_ids, "expert_ids"' in body
    assert 'validate_input(scores, "scores"' in body
    assert "validate_static_weights" not in body
    assert "resolve_pipelines" not in body
    assert "dynamic_input_validation_count_ += 3" in body


def test_probe_reuses_the_scheduled_input_row_handle_for_native_submission():
    source = PROBE.read_text()
    start = source.index("def _bound_call(")
    stop = source.index("@contextlib.contextmanager", start)
    body = source[start:stop]
    assert "input_row = flat[0]" in body
    assert "mx.async_eval(input_row, expert_ids, flat_scores)" in body
    assert "plan.execute_bound(input_row, expert_ids, flat_scores)" in body
    assert body.count("flat[0]") == 1


def test_probe_attributes_unbound_tax_and_keeps_fixed_performance_gates():
    source = PROBE.read_text()
    ast.parse(source)
    assert "CONTEXTS = (2_048, 262_144)" in source
    assert "TARGET_2K_TPS = 15.0" in source
    assert "MIN_WALL_SAVING_VS_EXACT_MS = 0.50" in source
    assert "MIN_HOST_SAVING_VS_EXACT_MS = 0.50" in source
    assert "MIN_HOST_SAVING_VS_UNBOUND_MS = 1.50" in source
    for arm in (
        "A_exact_composition",
        "B_unbound_native_plan",
        "C_prebound_native_plan",
    ):
        assert arm in source
    for metric in (
        "prebound_vs_exact_wall_saving_ms",
        "prebound_vs_exact_host_saving_ms",
        "prebound_vs_unbound_wall_saving_ms",
        "prebound_vs_unbound_host_saving_ms",
    ):
        assert metric in source


def test_probe_requires_exact_repair_and_recorded_boundary_only_failure():
    source = PROBE.read_text()
    assert "EXACT_REPAIR_ARTIFACT" in source
    assert "REJECTED_PERFORMANCE_ARTIFACT" in source
    rejected = json.loads(REJECTED.read_text())
    assert rejected["accepted"] is False
    assert set(rejected["failed_gates"]) == {
        "2k_native_wall_saving_at_least_0_50ms",
        "256k_native_wall_saving_at_least_0_50ms",
        "2k_native_host_saving_at_least_0_50ms",
        "256k_native_host_saving_at_least_0_50ms",
    }


def test_probe_is_probe_only_and_changes_no_production_abi():
    source = PROBE.read_text()
    assert '"probe_only": True' in source
    assert '"kernels": False' in source
    for component in (
        "runtime",
        "server",
        "apc",
        "cache_abi",
        "production_kernel_abi",
    ):
        assert f'"{component}": False' in source


def test_prebound_result_is_archived_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    if artifact["decision"] == "abort_prebound_native_packed_moe_execution_plan":
        assert artifact["accepted"] is False
        assert artifact["error"]["type"]
        return
    assert artifact["failed_gates"] == [
        name for name, passed in artifact["acceptance"].items() if not passed
    ]
