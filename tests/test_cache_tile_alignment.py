from __future__ import annotations

import pytest

from glm53_flash_mlx.cache_geometry import (
    DEFAULT_NOPE_CACHE_TILE_ALIGNMENT,
    NOPE_CACHE_TILE_ALIGNMENT_CONTRACT,
    CacheTileAlignmentError,
    NoPECacheTileAlignment,
    align_up,
    logical_token_to_pool_lane,
    plan_nope_cache_capacity,
    pool_lane_to_logical_token,
)


def test_default_alignment_derives_every_local_native_constraint_once():
    contract = DEFAULT_NOPE_CACHE_TILE_ALIGNMENT
    assert contract.logical_block_tokens == 256
    assert contract.index_kpool == 4
    assert contract.physical_indexer_tile_rows == 64
    assert contract.physical_indexer_tile_tokens == 256
    assert contract.selected_gather_token_granularity == 1
    assert contract.allocation_alignment_tokens == 256
    assert contract.allocation_alignment_pool_rows == 64
    assert contract.kda_state_slots == 2
    assert contract.custom_metal_indexer_kernel is False
    assert contract.virtual_tile_split_allowed is False
    assert "no-virtual-split" in NOPE_CACHE_TILE_ALIGNMENT_CONTRACT
    assert contract.descriptor()["kda_capacity_geometry"] == (
        "fixed-slots-orthogonal-to-token-capacity"
    )


@pytest.mark.parametrize(
    ("logical", "physical", "logical_rows", "physical_rows"),
    [
        (0, 0, 0, 0),
        (1, 256, 1, 64),
        (255, 256, 64, 64),
        (256, 256, 64, 64),
        (257, 512, 65, 128),
        (511, 512, 128, 128),
        (512, 512, 128, 128),
        (513, 768, 129, 192),
        (262_143, 262_144, 65_536, 65_536),
        (262_144, 262_144, 65_536, 65_536),
        (262_145, 262_400, 65_537, 65_600),
    ],
)
def test_alignment_minus_exact_plus_boundaries(
    logical, physical, logical_rows, physical_rows
):
    plan = plan_nope_cache_capacity(logical)
    assert plan.logical_capacity_tokens == logical
    assert plan.physical_capacity_tokens == physical
    assert plan.logical_pool_rows == logical_rows
    assert plan.physical_pool_rows == physical_rows
    assert physical >= logical
    assert physical % 256 == 0 if physical else True
    assert physical_rows % 64 == 0 if physical_rows else True


def test_logical_pool_physical_roundtrip_rejects_padding():
    logical = 257
    plan = plan_nope_cache_capacity(logical)
    for token in (0, 1, 255, 256):
        row, lane = logical_token_to_pool_lane(
            token, logical_extent_tokens=logical
        )
        assert pool_lane_to_logical_token(
            row, lane, logical_extent_tokens=logical
        ) == token

    assert plan.physical_capacity_tokens == 512
    assert plan.logical_pool_rows == 65
    assert plan.physical_pool_rows == 128
    with pytest.raises(CacheTileAlignmentError, match="logical cache extent"):
        logical_token_to_pool_lane(257, logical_extent_tokens=logical)
    with pytest.raises(CacheTileAlignmentError, match="logical cache extent"):
        pool_lane_to_logical_token(64, 1, logical_extent_tokens=logical)
    with pytest.raises(CacheTileAlignmentError, match="logical cache extent"):
        pool_lane_to_logical_token(127, 3, logical_extent_tokens=logical)


def test_partial_pool_lanes_beyond_tail_are_not_logical():
    logical = 259
    assert pool_lane_to_logical_token(
        64, 2, logical_extent_tokens=logical
    ) == 258
    for lane in (3,):
        with pytest.raises(CacheTileAlignmentError):
            pool_lane_to_logical_token(
                64, lane, logical_extent_tokens=logical
            )


@pytest.mark.parametrize(
    "call",
    [
        lambda: align_up(-1, 256),
        lambda: align_up(1, 0),
        lambda: plan_nope_cache_capacity(-1),
        lambda: logical_token_to_pool_lane(-1, logical_extent_tokens=1),
        lambda: pool_lane_to_logical_token(0, 4, logical_extent_tokens=8),
        lambda: NoPECacheTileAlignment(virtual_tile_split_allowed=True),
        lambda: NoPECacheTileAlignment(logical_block_tokens=0),
    ],
)
def test_invalid_or_virtual_split_geometry_fails_closed(call):
    with pytest.raises(CacheTileAlignmentError):
        call()
