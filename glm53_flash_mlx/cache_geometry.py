"""Authoritative physical geometry for GLM-5.3 NoPE DSA caches.

Logical token extent and physical allocation extent are deliberately separate.
The current MLX runtime has no custom paged Indexer Metal kernel: its native
constraint is the contiguous 256-token cache allocation quantum.  With
``index_kpool=4`` this is exactly 64 physical IndexPool rows.  KDA recurrent
state remains a fixed two-slot ``ArraysCache`` and never shares this geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import lcm

DEFAULT_INDEX_KPOOL = 4
MAX_SIGNED_64 = (1 << 63) - 1
NOPE_CACHE_TILE_ALIGNMENT_CONTRACT = (
    "glm53-nope-cache-tile-alignment-v1"
    "-logical-block256"
    "-kpool4"
    "-physical-pool64"
    "-no-virtual-split"
)


class CacheTileAlignmentError(ValueError):
    """Raised before an invalid logical/physical cache mapping is used."""


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CacheTileAlignmentError(f"{name} must be a Python int")
    if value < minimum:
        raise CacheTileAlignmentError(f"{name} must be >= {minimum}")
    if value > MAX_SIGNED_64:
        raise CacheTileAlignmentError(f"{name} exceeds signed 64-bit ABI")
    return value


def align_up(value: int, alignment: int) -> int:
    """Return the least aligned capacity without fixed-width overflow math."""

    value = _integer("value", value)
    alignment = _integer("alignment", alignment, minimum=1)
    quotient, remainder = divmod(value, alignment)
    result = (quotient + int(remainder != 0)) * alignment
    if result > MAX_SIGNED_64:
        raise CacheTileAlignmentError("aligned capacity exceeds signed 64-bit ABI")
    return result


@dataclass(frozen=True)
class NoPECacheTileAlignment:
    """Every native constraint participating in one allocation unit."""

    logical_block_tokens: int = 256
    index_kpool: int = DEFAULT_INDEX_KPOOL
    physical_indexer_tile_rows: int = 64
    selected_gather_token_granularity: int = 1
    kda_state_slots: int = 2
    custom_metal_indexer_kernel: bool = False
    virtual_tile_split_allowed: bool = False

    def __post_init__(self) -> None:
        for name in (
            "logical_block_tokens",
            "index_kpool",
            "physical_indexer_tile_rows",
            "selected_gather_token_granularity",
            "kda_state_slots",
        ):
            _integer(name, getattr(self, name), minimum=1)
        if self.virtual_tile_split_allowed:
            raise CacheTileAlignmentError(
                "virtual tile splitting cannot repair incompatible cache geometry"
            )

    @property
    def physical_indexer_tile_tokens(self) -> int:
        value = self.physical_indexer_tile_rows * self.index_kpool
        if value > MAX_SIGNED_64:
            raise CacheTileAlignmentError(
                "physical Indexer tile exceeds signed 64-bit ABI"
            )
        return value

    @property
    def allocation_alignment_tokens(self) -> int:
        return lcm(
            self.logical_block_tokens,
            self.index_kpool,
            self.physical_indexer_tile_tokens,
            self.selected_gather_token_granularity,
        )

    @property
    def allocation_alignment_pool_rows(self) -> int:
        alignment = self.allocation_alignment_tokens
        if alignment % self.index_kpool:
            raise CacheTileAlignmentError(
                "token allocation alignment is not divisible by index_kpool"
            )
        return alignment // self.index_kpool

    def descriptor(self) -> dict[str, object]:
        result = asdict(self)
        result.update(
            {
                "identity": NOPE_CACHE_TILE_ALIGNMENT_CONTRACT,
                "physical_indexer_tile_tokens": self.physical_indexer_tile_tokens,
                "allocation_alignment_tokens": self.allocation_alignment_tokens,
                "allocation_alignment_pool_rows": (
                    self.allocation_alignment_pool_rows
                ),
                "kda_capacity_geometry": "fixed-slots-orthogonal-to-token-capacity",
                "indexer_storage": "contiguous-pool-rows",
            }
        )
        return result


DEFAULT_NOPE_CACHE_TILE_ALIGNMENT = NoPECacheTileAlignment()


@dataclass(frozen=True)
class NoPECacheCapacityPlan:
    """One logical capacity mapped to aligned latent and IndexPool storage."""

    logical_capacity_tokens: int
    logical_pool_rows: int
    physical_capacity_tokens: int
    physical_pool_rows: int
    padding_tokens: int
    padding_pool_rows: int
    allocation_alignment_tokens: int
    allocation_alignment_pool_rows: int

    def descriptor(self) -> dict[str, int]:
        return asdict(self)


def plan_nope_cache_capacity(
    logical_capacity_tokens: int,
    *,
    alignment: NoPECacheTileAlignment = DEFAULT_NOPE_CACHE_TILE_ALIGNMENT,
) -> NoPECacheCapacityPlan:
    """Plan aligned physical storage without exposing padding logically."""

    if not isinstance(alignment, NoPECacheTileAlignment):
        raise TypeError("alignment must be a NoPECacheTileAlignment")
    logical = _integer("logical_capacity_tokens", logical_capacity_tokens)
    quotient, remainder = divmod(logical, alignment.index_kpool)
    logical_rows = quotient + int(remainder != 0)
    physical = align_up(logical, alignment.allocation_alignment_tokens)
    physical_rows = physical // alignment.index_kpool
    if physical_rows % alignment.physical_indexer_tile_rows:
        raise AssertionError("planned IndexPool capacity violates its native tile")
    return NoPECacheCapacityPlan(
        logical_capacity_tokens=logical,
        logical_pool_rows=logical_rows,
        physical_capacity_tokens=physical,
        physical_pool_rows=physical_rows,
        padding_tokens=physical - logical,
        padding_pool_rows=physical_rows - logical_rows,
        allocation_alignment_tokens=alignment.allocation_alignment_tokens,
        allocation_alignment_pool_rows=alignment.allocation_alignment_pool_rows,
    )


def logical_token_to_pool_lane(
    token_index: int,
    *,
    logical_extent_tokens: int,
    alignment: NoPECacheTileAlignment = DEFAULT_NOPE_CACHE_TILE_ALIGNMENT,
) -> tuple[int, int]:
    """Map only a logically visible token; physical padding fails closed."""

    token = _integer("token_index", token_index)
    logical = _integer("logical_extent_tokens", logical_extent_tokens)
    if token >= logical:
        raise CacheTileAlignmentError("token index is outside logical cache extent")
    return divmod(token, alignment.index_kpool)


def pool_lane_to_logical_token(
    pool_row: int,
    lane: int,
    *,
    logical_extent_tokens: int,
    alignment: NoPECacheTileAlignment = DEFAULT_NOPE_CACHE_TILE_ALIGNMENT,
) -> int:
    """Expand a pool lane while rejecting rows belonging only to padding."""

    row = _integer("pool_row", pool_row)
    lane = _integer("lane", lane)
    logical = _integer("logical_extent_tokens", logical_extent_tokens)
    if lane >= alignment.index_kpool:
        raise CacheTileAlignmentError("pool lane is outside index_kpool")
    token = row * alignment.index_kpool + lane
    if token > MAX_SIGNED_64 or token >= logical:
        raise CacheTileAlignmentError("pool lane maps outside logical cache extent")
    return token
