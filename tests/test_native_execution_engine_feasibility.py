import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_native_execution_engine_feasibility.py"
BUILD = ROOT / "scripts" / "build_native_execution_engine.py"
NATIVE = ROOT / "native_execution"


def test_probe_is_tiered_across_prefill_and_decode_without_overclaiming():
    source = PROBE.read_text()
    ast.parse(source)
    assert "PREFILL_CONTEXT = 32_768" in source
    assert "PREFILL_ROWS = 256" in source
    assert "DECODE_CONTEXTS = (2_048, 262_144)" in source
    assert '"score_producer_native": False' in source
    assert '"kda_dsa_native": False' in source
    assert '"full_32k_prefill_gate_evaluated": False' in source
    assert '"final_256k_decode_3ms_gate_evaluated": False' in source
    assert '"native_binding_matches_runtime"' in source
    assert '"artificial_order_tie_sentinel_contract_exact"' in source
    assert '"kth_tie"' in source
    assert '"signed_zero"' in source
    assert '"minimum_256k_decode_saving_ms": FINAL_NATIVE_DECODE_SAVING_MS' in source
    assert '"minimum_32k_prefill_speedup": FINAL_NATIVE_PREFILL_SPEEDUP' in source


def test_native_submission_bridge_does_not_build_or_wait_inside_execute():
    source = PROBE.read_text()
    cpp = (NATIVE / "native_indexer_plan.cpp").read_text()
    assert "mx.async_eval(*dependencies)" in source
    assert "plan.execute(" in source
    execute = cpp[cpp.index("NativeIndexSelectionPlan::execute(") :]
    assert "mx::eval(" not in execute
    assert "mx::async_eval(" not in execute
    assert "mx::synchronize(" not in execute
    assert "mx::allocator::malloc" not in execute
    assert "encoder.barrier();" in execute
    assert "dynamic_allocation_count() const { return 0; }" in (
        NATIVE / "native_indexer_plan.h"
    ).read_text()


def test_native_island_has_two_fixed_pipelines_and_no_intermediate_escape():
    metal = (NATIVE / "native_indexer_plan.metal").read_text()
    cpp = (NATIVE / "native_indexer_plan.cpp").read_text()
    assert "native_exact_partial_topk_512" in metal
    assert "glm53_native_expand_selected_pools" in metal
    assert "uint row = group_position.y;" in metal
    assert "set_output_array(selected_pool_scratch_" in cpp
    returned = "return {selected_token_indices_, selected_token_valid_};"
    assert returned in cpp
    assert "return {selected_pool_scratch_" not in cpp


def test_native_kernel_preserves_exact_order_sentinel_and_range_contracts():
    metal = (NATIVE / "native_indexer_plan.metal").read_text()
    for fragment in (
        "(bits & 0x7fffffffu) == 0u",
        "(ulong(ordered) << 17) | ulong(0x1ffffu - index)",
        "pool < uint(logical_pool_rows)",
        "token >= 0 && token < kv_len",
        "valid ? int(token) : -1",
    ):
        assert fragment in metal


def test_build_is_isolated_from_the_production_python_package():
    source = BUILD.read_text()
    cmake = (NATIVE / "CMakeLists.txt").read_text()
    assert 'SOURCE = ROOT / "native_execution"' in source
    assert 'MLX_NATIVE_ABI_VERSION = "0.32.2"' in source
    assert 'NANOBIND_VERSION = "2.15.0"' in source
    assert 'f"nanobind=={NANOBIND_VERSION}"' in source
    assert '"build_ext",' in source
    assert '"--inplace",' in source
    assert "subprocess.run(command, cwd=SOURCE, check=True" in source
    assert "glm53_native_execution._ext" in (NATIVE / "setup.py").read_text()
    assert "find_package(nanobind 2.15.0 EXACT CONFIG REQUIRED)" in cmake
    assert "FetchContent" not in cmake


def test_probe_keeps_every_production_abi_unchanged():
    source = PROBE.read_text()
    assert '"probe_only": True' in source
    for component in ("runtime", "server", "apc", "cache_abi", "kernel_abi"):
        assert f'"{component}": False' in source
