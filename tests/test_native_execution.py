from dataclasses import replace

import pytest

from glm53_flash_mlx.native_execution import (
    NATIVE_EXECUTION_ENGINE_ABI,
    NativeBufferRole,
    NativeExecutionContractError,
    NativeExecutionMode,
    plan_native_indexer_island,
)


def test_prefill_and_decode_share_one_fixed_native_execution_abi():
    decode = plan_native_indexer_island(
        NativeExecutionMode.DECODE,
        query_rows=1,
        logical_capacity_tokens=262_145,
    )
    prefill = plan_native_indexer_island(
        NativeExecutionMode.PREFILL,
        query_rows=256,
        logical_capacity_tokens=262_145,
    )
    assert decode.descriptor()["abi"] == NATIVE_EXECUTION_ENGINE_ABI
    assert prefill.descriptor()["abi"] == NATIVE_EXECUTION_ENGINE_ABI
    assert decode.physical_pool_rows == prefill.physical_pool_rows == 65_600
    assert decode.fixed_command_topology == prefill.fixed_command_topology
    assert decode.query_rows == 1
    assert prefill.query_rows == 256


def test_native_plan_has_no_per_execute_graph_allocation_or_sync():
    plan = plan_native_indexer_island(
        "decode", query_rows=1, logical_capacity_tokens=262_144
    )
    descriptor = plan.descriptor()
    assert descriptor["dynamic_allocations_per_execute"] == 0
    assert descriptor["python_graph_nodes_per_execute"] == 0
    assert descriptor["shape_discovery_per_execute"] == 0
    assert descriptor["host_synchronizations_per_execute"] == 0
    assert descriptor["fixed_command_topology"] == [
        "glm53_native_exact_partial_topk_512",
        "glm53_native_expand_selected_pools",
    ]


def test_scratch_is_owned_stable_and_never_returned_to_mlx():
    plan = plan_native_indexer_island(
        "prefill", query_rows=256, logical_capacity_tokens=32_768
    )
    scratch = next(
        buffer for buffer in plan.buffers if buffer.role is NativeBufferRole.SCRATCH
    )
    assert scratch.owned_by_plan is True
    assert scratch.stable_address is True
    assert scratch.row_major is True
    assert scratch.returned_to_mlx is False
    outputs = [
        buffer for buffer in plan.buffers if buffer.role is NativeBufferRole.OUTPUT
    ]
    assert all(buffer.owned_by_plan and buffer.stable_address for buffer in outputs)
    assert all(buffer.returned_to_mlx for buffer in outputs)


def test_logical_and_physical_capacity_remain_distinct():
    plan = plan_native_indexer_island(
        "decode", query_rows=1, logical_capacity_tokens=262_145
    )
    assert plan.logical_capacity_tokens == 262_145
    assert plan.physical_pool_rows == 65_600
    scores = next(buffer for buffer in plan.buffers if buffer.name == "scores")
    assert scores.shape == (1, 1, 65_600)


@pytest.mark.parametrize(
    "call",
    [
        lambda: plan_native_indexer_island(
            "decode", query_rows=2, logical_capacity_tokens=2048
        ),
        lambda: plan_native_indexer_island(
            "prefill", query_rows=0, logical_capacity_tokens=2048
        ),
        lambda: plan_native_indexer_island(
            "unknown", query_rows=1, logical_capacity_tokens=2048
        ),
        lambda: plan_native_indexer_island(
            "decode",
            query_rows=1,
            logical_capacity_tokens=2048,
            score_dtype="uint8",
        ),
    ],
)
def test_invalid_native_plan_fails_closed(call):
    with pytest.raises(NativeExecutionContractError):
        call()


def test_mutable_scratch_cannot_be_borrowed_or_escape():
    plan = plan_native_indexer_island(
        "decode", query_rows=1, logical_capacity_tokens=2048
    )
    scratch_index = next(
        i for i, buffer in enumerate(plan.buffers) if buffer.role is NativeBufferRole.SCRATCH
    )
    borrowed = replace(plan.buffers[scratch_index], owned_by_plan=False)
    buffers = list(plan.buffers)
    buffers[scratch_index] = borrowed
    with pytest.raises(NativeExecutionContractError, match="owned and stable"):
        replace(plan, buffers=tuple(buffers))

    escaped = replace(plan.buffers[scratch_index], returned_to_mlx=True)
    buffers[scratch_index] = escaped
    with pytest.raises(NativeExecutionContractError, match="cannot escape"):
        replace(plan, buffers=tuple(buffers))
