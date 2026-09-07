import ast
from pathlib import Path

from glm53_flash_mlx.native_execution import (
    NATIVE_INDEXPOOL_UPDATE_ISLAND_ABI,
    NativeBufferRole,
    plan_native_indexpool_update_island,
)


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native_execution"
PROBE = ROOT / "scripts" / "probe_native_indexpool_update_submission_island.py"


def test_update_plan_has_explicit_external_cache_state_and_private_raw19():
    plan = plan_native_indexpool_update_island(
        logical_capacity_tokens=262_145
    )
    assert NATIVE_INDEXPOOL_UPDATE_ISLAND_ABI.startswith(
        "glm53-native-indexpool-update-island-v1"
    )
    assert plan.physical_pool_rows == 65_600
    by_name = {buffer.name: buffer for buffer in plan.buffers}
    for name in ("pool_keys", "pool_indices", "pool_valid"):
        buffer = by_name[name]
        assert buffer.role is NativeBufferRole.EXTERNAL_MUTABLE_STATE
        assert buffer.owned_by_plan is False
        assert buffer.stable_address is True
    for name in (
        "raw_keys_a",
        "raw_keys_b",
        "raw_gates_a",
        "raw_gates_b",
        "raw_valid_a",
        "raw_valid_b",
        "raw_positions_a",
        "raw_positions_b",
    ):
        assert by_name[name].role is NativeBufferRole.SCRATCH
        assert by_name[name].owned_by_plan is True
        assert by_name[name].returned_to_mlx is False


def test_update_plan_is_decode_only_and_connects_directly_to_tier1():
    plan = plan_native_indexpool_update_island(logical_capacity_tokens=2_049)
    assert plan.fixed_command_topology == (
        "glm53_native_advance_indexpool_raw19",
        "glm53_native_update_indexpool_row_bfloat16",
        "glm53-native-dsa-score-island-v2-bf16-eager-rounding-coalesced-pool32-head32-exact-topk-expand",
    )
    assert plan.dynamic_allocations_per_execute == 0
    assert plan.python_graph_nodes_per_execute == 0
    assert plan.shape_discovery_per_execute == 0
    assert plan.host_synchronizations_per_execute == 0


def test_update_kernel_preserves_eager_bfloat16_softmax_boundaries():
    metal = (NATIVE / "native_indexer_plan.metal").read_text()
    kernel = metal[metal.index("glm53_native_update_indexpool_row_bfloat16") :]
    assert "bfloat16_t delta = bfloat16_t(" in kernel
    assert "bfloat16_t(fast::exp(float(delta)))" in kernel
    assert "bfloat16_t(float(normalizer) + float(exponentials[lane]))" in kernel
    assert "float(exponentials[lane]) / float(normalizer)" in kernel
    assert "bfloat16_t(float(probability) * float(key))" in kernel
    assert "bfloat16_t(float(total) + float(product))" in kernel


def test_native_execute_has_fixed_arena_and_no_graph_allocation_or_wait():
    cpp = (NATIVE / "native_indexpool_update_plan.cpp").read_text()
    header = (NATIVE / "native_indexpool_update_plan.h").read_text()
    execute = cpp[cpp.index("NativeIndexPoolUpdateSelectionPlan::execute(") :]
    assert "mx::allocator::malloc" not in execute
    assert "mx::eval(" not in execute
    assert "mx::async_eval(" not in execute
    assert "mx::synchronize(" not in execute
    assert "score_plan_.execute(" in execute
    assert "returned_intermediate_tensor_bytes() const { return 0; }" in header


def test_probe_has_exactness_first_screen_and_fixed_full_model_gate():
    source = PROBE.read_text()
    ast.parse(source)
    assert "CONTEXTS = (2_048, 262_144)" in source
    assert "MIN_256K_SAVING_MS = 0.75" in source
    assert '"artificial_native_contract"' in source
    assert "previous_total_mod4" in source
    assert '"pool_state_byte_exact"' in source
    assert '"all_logits_byte_exact"' in source
    assert '"post_state_byte_exact"' in source
    assert "or mask is not None" in source
    for component in ("runtime", "server", "apc", "cache_abi", "kernel_abi"):
        assert f'"{component}": False' in source
