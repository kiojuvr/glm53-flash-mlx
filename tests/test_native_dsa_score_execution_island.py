import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_native_dsa_score_execution_island.py"
NATIVE = ROOT / "native_execution"


def test_tier1_probe_is_exactness_first_and_stops_before_long_model_work():
    source = PROBE.read_text()
    ast.parse(source)
    assert "DECODE_CONTEXTS = (2_048, 262_144)" in source
    assert "PREFILL_CONTEXT = 32_768" in source
    assert "PREFILL_ROWS = 256" in source
    assert '"reject_native_score_numerical_order"' in source
    assert '"reject_native_score_prefill_exactness"' in source
    assert '"score_byte_exact"' in source
    assert '"all_logits_byte_exact"' in source
    assert '"post_state_byte_exact"' in source


def test_tier1_uses_exact_mlx_steel_gemm_and_simd_bf16_reduction_order():
    metal = (NATIVE / "native_indexer_plan.metal").read_text()
    cpp = (NATIVE / "native_dsa_score_plan.cpp").read_text()
    assert "steel/gemm/kernels/steel_gemm_fused.h" in metal
    assert "glm53_native_steel_gemm_nt_bfloat16_bfloat16" in metal
    assert "64," in metal and "16," in metal
    assert "bfloat16_t total = simd_sum(weighted);" in metal
    assert "glm53_native_finish_pooled_score_bfloat16" in cpp
    assert "encoder.barrier();" in cpp


def test_tier1_execute_has_fixed_buffers_and_no_graph_allocation_or_wait():
    cpp = (NATIVE / "native_dsa_score_plan.cpp").read_text()
    header = (NATIVE / "native_dsa_score_plan.h").read_text()
    execute = cpp[cpp.index("NativeDSAScoreSelectionPlan::execute(") :]
    assert "mx::eval(" not in execute
    assert "mx::async_eval(" not in execute
    assert "mx::synchronize(" not in execute
    assert "mx::allocator::malloc" not in execute
    assert "return {selected_token_indices_, selected_token_valid_};" in execute
    assert "return {index_scores_" not in execute
    assert "returned_score_tensor_bytes() const { return 0; }" in header


def test_tier1_scope_does_not_claim_projection_attention_or_runtime_promotion():
    source = PROBE.read_text()
    for fragment in (
        '"query_projection_native": False',
        '"pool_update_native": False',
        '"sparse_gather_attention_native": False',
        '"kda_native": False',
        '"moe_native": False',
        '"probe_only": True',
    ):
        assert fragment in source
    for component in ("runtime", "server", "apc", "cache_abi", "kernel_abi"):
        assert f'"{component}": False' in source
