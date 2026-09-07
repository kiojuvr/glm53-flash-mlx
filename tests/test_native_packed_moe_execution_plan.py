import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_native_packed_moe_execution_plan.py"
METAL = ROOT / "native_execution" / "native_indexer_plan.metal"
HEADER = ROOT / "native_execution" / "native_packed_moe_plan.h"
SOURCE = ROOT / "native_execution" / "native_packed_moe_plan.cpp"
BINDINGS = ROOT / "native_execution" / "bindings.cpp"
CMAKE = ROOT / "native_execution" / "CMakeLists.txt"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-packed-moe-execution-plan-20260907.json"
)


def test_native_packed_moe_plan_is_bound_and_built_probe_only():
    header = HEADER.read_text()
    source = SOURCE.read_text()
    bindings = BINDINGS.read_text()
    cmake = CMAKE.read_text()
    assert "class NativePackedMoEDecodePlan" in header
    assert "NativePackedMoEDecodePlan::execute" in source
    assert '"NativePackedMoEDecodePlan"' in bindings
    assert "native_packed_moe_plan.cpp" in cmake
    for counter in (
        "dynamic_allocation_count",
        "graph_node_count",
        "shape_discovery_count",
        "host_synchronization_count",
        "returned_intermediate_tensor_bytes",
    ):
        assert f"{counter}() const {{ return 0; }}" in header


def test_native_plan_contains_exact_qualified_moe_ordering():
    metal = METAL.read_text()
    for kernel in (
        "glm53_native_packed_selected8_gate_up_swiglu",
        "glm53_native_packed_selected8_down",
        "glm53_native_packed_selected8_weighted_reduction",
        "glm53_native_shared_gate_up_swiglu",
        "glm53_native_shared_down",
        "glm53_native_add_routed_shared",
    ):
        assert kernel in metal
    assert "constant float kNativeE4M3[256]" in metal
    assert "metal::exp(metal::abs(gate_activation))" in metal
    assert "bfloat16_t gate_t = bfloat16_t(gate_total)" in metal
    assert "float contribution" in metal
    assert "total += contribution" in metal


def test_native_plan_uses_fixed_scratch_and_one_command_topology():
    source = SOURCE.read_text()
    for scratch in (
        "routed_hidden_",
        "routed_down_",
        "routed_output_",
        "shared_hidden_",
        "shared_down_",
        "output_",
    ):
        assert f"owned_array(" in source
        assert scratch in source
    assert "mx::metal::get_command_encoder(stream_)" in source
    assert source.count("encoder.barrier();") == 5
    assert "buffer identity changed during execute" in source


def test_probe_compares_native_plan_to_exact_composition_at_fixed_gates():
    source = PROBE.read_text()
    ast.parse(source)
    assert "CONTEXTS = (2_048, 262_144)" in source
    assert "TARGET_2K_TPS = 15.0" in source
    assert "MIN_NATIVE_WALL_SAVING_MS = 0.50" in source
    assert "MIN_NATIVE_HOST_SAVING_MS = 0.50" in source
    assert "MIN_CONTEXT_RETENTION = 0.90" in source
    assert "residual.Arm(\"B1\", True)" in source
    assert '"all_full_vocab_logits_byte_exact"' in source
    assert '"all_generated_tokens_exact"' in source
    assert '"post_state_byte_exact"' in source
    assert '"official_oracle_exact_both_arms"' in source


def test_probe_keeps_router_outside_native_boundary_and_production_unchanged():
    source = PROBE.read_text()
    assert "indices, scores = moe.gate(x)" in source
    assert "The authoritative MLX Float32 router remains outside" in source
    assert '"probe_only": True' in source
    for component in ("runtime", "server", "apc", "cache_abi", "kernel_abi"):
        assert f'"{component}": False' in source


def test_native_packed_moe_qualification_is_an_exact_short_screen_but_stops():
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is False
    assert artifact["decision"] == "stop_or_redesign_native_packed_moe_execution_plan"
    assert artifact["artificial_native_contract"]["output_byte_exact"]
    assert artifact["process_peak_memory_bytes"] <= 340_000_000_000
    for context in ("2048", "262144"):
        row = artifact["contexts"][context]
        assert row["all_full_vocab_logits_byte_exact"]
        assert row["all_generated_tokens_exact"]
        assert row["post_state_byte_exact"]
        assert row["native_wall_saving_ms"] < 0.0
        assert row["native_host_saving_ms"] < 0.0
    oracle = artifact["official_oracle"]
    assert oracle["A_exact_composition"]["all_full_vocab_logits_hashes_match"]
    assert oracle["B_native_packed_moe_plan"]["first_16_match"]
    assert not oracle["B_native_packed_moe_plan"]["full_128_match"]
    assert "step 68 logits hash mismatch" in oracle[
        "B_native_packed_moe_plan"
    ]["failures_128"]
    assert artifact["acceptance"]["2k_decode_at_least_15_tps"]
    assert not artifact["acceptance"]["official_oracle_exact_both_arms"]
