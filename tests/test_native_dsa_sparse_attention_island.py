from pathlib import Path

from glm53_flash_mlx.native_execution import (
    NATIVE_DSA_SPARSE_ATTENTION_ISLAND_ABI,
    NativeBufferRole,
    plan_native_dsa_sparse_attention_island,
)


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native_execution"
PROBE = ROOT / "scripts" / "probe_native_dsa_sparse_attention_island.py"


def test_tier2_plan_keeps_selection_gather_scores_and_splitk_private():
    plan = plan_native_dsa_sparse_attention_island(
        logical_capacity_tokens=262_145
    )
    assert NATIVE_DSA_SPARSE_ATTENTION_ISLAND_ABI.startswith(
        "glm53-native-dsa-sparse-attention-island-v1"
    )
    assert plan.mode.value == "decode"
    assert plan.physical_pool_rows == 65_600
    by_name = {buffer.name: buffer for buffer in plan.buffers}
    for name in (
        "head_scores",
        "index_scores",
        "selected_indices",
        "selected_valid",
        "scaled_query",
        "gathered_latent",
        "attention_scores",
        "splitk_accum",
    ):
        assert by_name[name].role is NativeBufferRole.SCRATCH
        assert by_name[name].returned_to_mlx is False
        assert by_name[name].owned_by_plan is True
        assert by_name[name].stable_address is True
    assert by_name["attention_output"].returned_to_mlx is True


def test_tier2_uses_pinned_mlx_fallback_order_and_fixed_geometry():
    plan = plan_native_dsa_sparse_attention_island(
        logical_capacity_tokens=262_145
    )
    assert plan.fixed_command_topology[-5:] == (
        "glm53_native_prepare_sparse_attention_bfloat16",
        "mlx0322-steel-nt-bf16",
        "glm53_native_mask_sparse_attention_scores_bfloat16",
        "mlx0322-block-softmax-precise-bf16",
        "mlx0322-steel-splitk-nn-bf16-fp32",
    )
    by_name = {buffer.name: buffer for buffer in plan.buffers}
    assert by_name["gathered_latent"].shape == (2051, 512)
    assert by_name["attention_scores"].shape == (64, 2051)
    assert by_name["splitk_accum"].shape == (4, 64, 512)
    assert by_name["attention_output"].shape == (1, 64, 1, 512)


def test_tier2_execute_has_no_allocation_graph_or_wait_and_one_output():
    cpp = (NATIVE / "native_dsa_attention_plan.cpp").read_text()
    header = (NATIVE / "native_dsa_attention_plan.h").read_text()
    execute = cpp[cpp.index("NativeDSASparseAttentionPlan::execute(") :]
    assert "mx::allocator::malloc" not in execute
    assert "mx::eval(" not in execute
    assert "mx::async_eval(" not in execute
    assert "mx::synchronize(" not in execute
    assert "return attention_output_;" in execute
    assert "returned_intermediate_tensor_bytes() const { return 0; }" in header


def test_tier2_metal_matches_exact_mlx0322_d512_attention_primitives():
    metal = (NATIVE / "native_indexer_plan.metal").read_text()
    cpp = (NATIVE / "native_dsa_attention_plan.cpp").read_text()
    for fragment in (
        "glm53_native_prepare_sparse_attention_bfloat16",
        "glm53_native_mask_sparse_attention_scores_bfloat16",
        "glm53_native_attention_gemm_nt_bfloat16_bfloat16_bm64_bn32_bk32_wm2_wn2",
        "glm53_native_block_softmax_precise_bfloat16",
        "glm53_native_attention_gemm_splitk_nn_bfloat16_float32_bm32_bn32_bk16_wm2_wn2_MN_taligned_K_naligned",
        "glm53_native_attention_gemm_splitk_accum_bfloat16_float32",
    ):
        assert fragment in metal
    for fragment in (
        "glm53_native_prepare_sparse_attention_bfloat16",
        "glm53_native_mask_sparse_attention_scores_bfloat16",
        "glm53_native_attention_gemm_nt_bfloat16_bfloat16",
        "glm53_native_block_softmax_precise_bfloat16",
        "glm53_native_attention_gemm_splitk_nn_bfloat16_",
        "glm53_native_attention_gemm_splitk_accum_bfloat16_float32",
    ):
        assert fragment in cpp
    header = (NATIVE / "native_dsa_attention_plan.h").read_text()
    assert "kSplitKPartitions = 4" in header
    assert "kSelectedWidth = 2051" in header


def test_tier2_probe_is_exactness_first_and_compares_against_tier1():
    source = PROBE.read_text()
    assert "DECODE_CONTEXTS = (2_048, 262_144)" in source
    assert '"artificial_native_contract"' in source
    assert '"reject_native_sparse_attention_numerical_order"' in source
    assert '"A_tier1_mlx_attention"' in source
    assert '"B_tier2_native_attention"' in source
    assert '"all_logits_byte_exact"' in source
    assert '"post_state_byte_exact"' in source
    assert "MIN_256K_INCREMENTAL_SAVING_MS = 0.75" in source
    for component in ("runtime", "server", "apc", "cache_abi", "kernel_abi"):
        assert f'"{component}": False' in source
