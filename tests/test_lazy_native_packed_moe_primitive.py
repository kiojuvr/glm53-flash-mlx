import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_lazy_native_packed_moe_primitive.py"
HEADER = ROOT / "native_execution" / "native_packed_moe_plan.h"
SOURCE = ROOT / "native_execution" / "native_packed_moe_plan.cpp"
BINDINGS = ROOT / "native_execution" / "bindings.cpp"
PREBOUND = (
    ROOT
    / "bench-results"
    / "m3ultra512-prebound-native-packed-moe-execution-plan-20260907.json"
)
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-lazy-native-packed-moe-primitive-20260907.json"
)


def test_lazy_primitive_defers_native_encoding_to_graph_evaluation():
    header = HEADER.read_text()
    source = SOURCE.read_text()
    bindings = BINDINGS.read_text()
    assert "class NativePackedMoELazyPrimitive : public mx::Primitive" in source
    assert "void eval_gpu(" in source
    assert "plan_->encode(inputs[0], inputs[1], inputs[2]" in source
    assert "mx::array execute_lazy(" in header
    assert '.def("execute_lazy"' in bindings
    assert 'return "NativePackedMoELazyPrimitive"' in source


def test_lazy_construction_accepts_unscheduled_inputs_without_async_eval():
    source = SOURCE.read_text()
    start = source.index("NativePackedMoEDecodePlan::execute_lazy(")
    stop = source.index("NativePackedMoEDecodePlan::encode(", start)
    body = source[start:stop]
    assert "validate_input_descriptor" in body
    assert "validate_input(" not in body
    assert "async_eval" not in body
    python = PROBE.read_text()
    start = python.index("def _lazy_call(")
    stop = python.index("@contextlib.contextmanager", start)
    assert "async_eval" not in python[start:stop]
    assert ".execute_lazy(" in python[start:stop]


def test_probe_compares_exact_eager_and_lazy_at_fixed_gates():
    source = PROBE.read_text()
    ast.parse(source)
    assert "CONTEXTS = (2_048, 262_144)" in source
    assert "TARGET_2K_TPS = 15.0" in source
    assert "MIN_WALL_SAVING_VS_EXACT_MS = 0.50" in source
    assert "MIN_HOST_SAVING_VS_EXACT_MS = 0.50" in source
    assert "MIN_WALL_RECOVERY_VS_EAGER_MS = 1.00" in source
    assert "MIN_HOST_RECOVERY_VS_EAGER_MS = 1.50" in source
    for arm in (
        "A_exact_composition",
        "B_eager_prebound_native",
        "C_lazy_prebound_native",
    ):
        assert arm in source


def test_probe_requires_the_exact_but_slow_prebound_source():
    source = json.loads(PREBOUND.read_text())
    assert source["accepted"] is False
    assert source["acceptance"][
        "all_three_arms_logits_tokens_and_state_byte_exact"
    ]
    assert source["acceptance"][
        "all_42_layers_prebound_once_and_execute_dynamic_only"
    ]
    assert "PREBOUND_ARTIFACT" in PROBE.read_text()


def test_probe_is_probe_only_and_preserves_production_abi():
    source = PROBE.read_text()
    assert '"probe_only": True' in source
    assert '"kernels": False' in source
    assert '"forced_async_eval_per_layer": False' in source
    for component in (
        "runtime",
        "server",
        "apc",
        "cache_abi",
        "production_kernel_abi",
    ):
        assert f'"{component}": False' in source


def test_lazy_result_is_archived_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is False
    assert artifact["decision"] == "stop_lazy_native_packed_moe_primitive"
    assert set(artifact["failed_gates"]) == {
        name for name, passed in artifact["acceptance"].items() if not passed
    }
    acceptance = artifact["acceptance"]
    assert acceptance["all_three_arms_logits_tokens_and_state_byte_exact"]
    assert acceptance["all_42_layers_use_one_lazy_graph_node_per_execution"]
    assert acceptance["lazy_plans_keep_bound_resources_and_fixed_scratch"]
    assert acceptance["2k_lazy_recovers_at_least_1_50ms_host_from_eager"]
    assert acceptance["256k_lazy_recovers_at_least_1_50ms_host_from_eager"]
    assert not acceptance["2k_lazy_wall_saving_vs_exact_at_least_0_50ms"]
    assert not acceptance["256k_lazy_wall_saving_vs_exact_at_least_0_50ms"]

    short = artifact["contexts"]["2048"]
    long = artifact["contexts"]["262144"]
    assert short["lazy_vs_eager_host_saving_ms"] >= 1.50
    assert long["lazy_vs_eager_host_saving_ms"] >= 1.50
    assert short["lazy_vs_exact_host_saving_ms"] > 0
    assert long["lazy_vs_exact_host_saving_ms"] > 0
    assert short["lazy_vs_exact_wall_saving_ms"] < -1.0
    assert long["lazy_vs_exact_wall_saving_ms"] < -1.0
