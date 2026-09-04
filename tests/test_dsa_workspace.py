from __future__ import annotations

import pytest

from glm53_flash_mlx.dsa_workspace import (
    DEFAULT_MAX_WORKSPACE_BYTES,
    DSA_INDEXER_WORKSPACE_CONTRACT,
    DSAWorkspaceGeometryError,
    account_dsa_indexer_memory,
    ceil_div,
    plan_dsa_indexer_workspace,
)


@pytest.mark.parametrize(
    ("context", "pools"),
    [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 2),
        (131_071, 32_768),
        (131_072, 32_768),
        (131_073, 32_769),
        (262_143, 65_536),
        (262_144, 65_536),
        (262_145, 65_537),
        (1_048_576, 262_144),
    ],
)
def test_pool_count_uses_tail_preserving_ceil_div(context, pools):
    geometry = plan_dsa_indexer_workspace(
        context_tokens=context,
        num_query_rows=256,
    )
    assert geometry.pool_count == pools
    assert geometry.pool_count == ceil_div(context, 4)


def test_128k_256k_and_1m_geometries_match_the_64mib_contract():
    at_128k = plan_dsa_indexer_workspace(
        context_tokens=131_072, num_query_rows=256
    )
    assert at_128k.pool_count == 32_768
    assert at_128k.query_block_rows == 256
    assert at_128k.query_block_count == 1
    assert at_128k.fp32_logits_workspace_bytes == 32 << 20

    at_256k = plan_dsa_indexer_workspace(
        context_tokens=262_144, num_query_rows=256
    )
    assert at_256k.pool_count == 65_536
    assert at_256k.query_block_rows == 256
    assert at_256k.query_block_count == 1
    assert at_256k.fp32_logits_workspace_bytes == 64 << 20

    at_1m = plan_dsa_indexer_workspace(
        context_tokens=1_048_576, num_query_rows=256
    )
    assert at_1m.pool_count == 262_144
    assert at_1m.query_block_rows == 64
    assert at_1m.query_block_count == 4
    assert at_1m.fp32_logits_workspace_bytes == 64 << 20


def test_partial_pool_above_256k_reduces_rows_instead_of_exceeding_budget():
    geometry = plan_dsa_indexer_workspace(
        context_tokens=262_145, num_query_rows=256
    )
    assert geometry.pool_count == 65_537
    assert geometry.query_block_rows == 255
    assert geometry.query_block_count == 2
    assert geometry.fp32_logits_workspace_bytes <= DEFAULT_MAX_WORKSPACE_BYTES


@pytest.mark.parametrize("queries", [1, 63, 64, 65, 127, 128, 255, 256])
def test_query_geometry_never_uses_logical_context_as_key_count(queries):
    geometry = plan_dsa_indexer_workspace(
        context_tokens=262_145,
        num_query_rows=queries,
    )
    assert geometry.pool_count == 65_537
    assert geometry.pool_count != geometry.context_tokens
    assert geometry.query_block_rows <= queries
    assert geometry.fp32_logits_workspace_bytes <= DEFAULT_MAX_WORKSPACE_BYTES


def test_selected_width_is_context_independent_and_includes_only_tail_slots():
    for context in (1, 2_047, 2_048, 2_049, 131_072, 262_144, 1_048_576):
        geometry = plan_dsa_indexer_workspace(
            context_tokens=context, num_query_rows=256
        )
        assert geometry.selected_token_width == 2_051
        assert geometry.selected_pool_count <= 512


def test_transient_and_persistent_accounting_are_explicit_and_disjoint():
    geometry = plan_dsa_indexer_workspace(
        context_tokens=262_144, num_query_rows=256
    )
    accounting = account_dsa_indexer_memory(geometry, index_head_dim=128)
    descriptor = accounting.descriptor()
    assert descriptor["transient"]["logits_workspace_bytes"] == 64 << 20
    assert descriptor["transient"]["topk_index_scratch_bytes"] == 64 << 20
    assert descriptor["transient"]["topk_score_scratch_bytes"] == 524_288
    assert descriptor["persistent_indexpool"] == {
        "persistent_pool_keys_bytes": 16_777_216,
        "persistent_pool_indices_bytes": 2_097_152,
        "persistent_pool_validity_bytes": 65_536,
    }
    assert descriptor["anonymous_allocation_bytes"] == 0
    assert not set(descriptor["transient"]) & set(descriptor["persistent_indexpool"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"context_tokens": -1, "num_query_rows": 1},
        {"context_tokens": 1, "num_query_rows": -1},
        {"context_tokens": 1, "num_query_rows": 1, "index_kpool": 0},
        {"context_tokens": 1, "num_query_rows": 1, "index_topk": 3},
        {"context_tokens": 1, "num_query_rows": 1, "max_workspace_bytes": 3},
        {"context_tokens": True, "num_query_rows": 1},
    ],
)
def test_invalid_geometry_fails_closed(kwargs):
    with pytest.raises(DSAWorkspaceGeometryError):
        plan_dsa_indexer_workspace(**kwargs)


def test_single_pool_row_larger_than_budget_is_rejected():
    with pytest.raises(DSAWorkspaceGeometryError, match="one pooled-logit"):
        plan_dsa_indexer_workspace(
            context_tokens=1_048_576,
            num_query_rows=1,
            max_workspace_bytes=(1 << 20) - 1,
        )


def test_contract_names_pooling_row_blocking_budget_and_lifecycle_split():
    assert "ceil-div-kpool4" in DSA_INDEXER_WORKSPACE_CONTRACT
    assert "query-row-blocked" in DSA_INDEXER_WORKSPACE_CONTRACT
    assert "fp32-logits-64mib" in DSA_INDEXER_WORKSPACE_CONTRACT
    assert "transient-separate-from-indexpool" in DSA_INDEXER_WORKSPACE_CONTRACT
