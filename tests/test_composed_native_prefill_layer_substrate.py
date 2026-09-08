from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_composed_native_prefill_layer_substrate.py"
NATIVE = ROOT / "native_execution"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-composed-native-prefill-layer-substrate-20260908.json"
)


def test_substrate_reserves_composed_512k_prefill_geometry():
    header = (NATIVE / "native_prefill_layer_plan.h").read_text()
    assert "kQueryRows = 256" in header
    assert "kQueryBlockRows = 128" in header
    assert "kQueryBlockCount = 2" in header
    assert "kPhysicalPoolRows = 131072" in header
    assert "kSelectedWidth = 2051" in header
    assert "kRouteRows = kQueryRows * kTopK" in header


def test_execute_has_one_encoder_scope_and_no_allocation_or_wait():
    cpp = (NATIVE / "native_prefill_layer_plan.cpp").read_text()
    execute = cpp[cpp.index("NativePrefillLayerSubstrate::execute(") :]
    assert execute.count("get_command_encoder(stream_)") == 1
    execute_body = execute.split(
        "NativePrefillLayerSubstrate::dsa_score_scratch_bytes"
    )[0]
    assert "get_kernel(" not in execute_body
    assert "get_library(" not in execute_body
    assert "mx::allocator::malloc" not in execute.split(
        "NativePrefillLayerSubstrate::dsa_score_scratch_bytes"
    )[0]
    assert "mx::eval(" not in execute
    assert "mx::async_eval(" not in execute
    assert "mx::synchronize(" not in execute
    assert "return {hidden_ping_};" in execute


def test_arena_and_intermediates_are_plan_owned_and_do_not_escape():
    header = (NATIVE / "native_prefill_layer_plan.h").read_text()
    for name in (
        "hidden_ping_",
        "hidden_pong_",
        "dsa_score_scratch_",
        "selected_indices_",
        "route_experts_",
        "moe_hidden_scratch_",
        "moe_down_scratch_",
    ):
        assert name in header
    assert "returned_intermediate_tensor_bytes() const { return 0; }" in header
    assert "dynamic_allocation_count() const { return 0; }" in header


def test_probe_makes_no_arithmetic_or_performance_claim():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert '"pass_through_only": True' in source
    assert '"dsa": False' in source
    assert '"moe": False' in source
    assert "timing is not a prefill performance claim" in source
    assert "advance_substrate_to_exact_dsa_and_moe_arithmetic" in source


def test_native_build_and_package_export_the_substrate():
    cmake = (NATIVE / "CMakeLists.txt").read_text()
    bindings = (NATIVE / "bindings.cpp").read_text()
    package = (NATIVE / "glm53_native_execution" / "__init__.py").read_text()
    assert "native_prefill_layer_plan.cpp" in cmake
    assert "NativePrefillLayerSubstrate" in bindings
    assert "NativePrefillLayerSubstrate" in package
    assert "glm53_native_prefill_copy_bfloat16" in (
        NATIVE / "native_indexer_plan.metal"
    ).read_text()


def test_artifact_when_present_passes_every_substrate_gate():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert all(artifact["checks"].values())
    assert artifact["arithmetic_coverage"]["pass_through_only"] is True
    assert not any(artifact["runtime_changes"].values())
