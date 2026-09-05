"""Bounded workspace geometry for GLM-5.3's pooled DSA Indexer.

The planner is intentionally independent of MLX allocation and the production
cache implementation.  It separates transient score/selection workspace from
persistent IndexPool state and uses Python's arbitrary-precision integers plus
explicit signed-64-bit ABI bounds for overflow-safe planning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .cache_geometry import DEFAULT_INDEX_KPOOL

DEFAULT_INDEX_TOPK = 2048
DEFAULT_MAX_WORKSPACE_BYTES = 64 << 20
FP32_BYTES = 4
INT32_BYTES = 4
INT64_BYTES = 8
BOOL_BYTES = 1
BF16_BYTES = 2
MAX_SIGNED_64 = (1 << 63) - 1
DSA_INDEXER_WORKSPACE_CONTRACT = (
    "glm53-dsa-indexer-workspace-v1"
    "-ceil-div-kpool4"
    "-query-row-blocked"
    "-fp32-logits-64mib"
    "-transient-separate-from-indexpool"
)


class DSAWorkspaceGeometryError(ValueError):
    """Raised before an unbounded or invalid workspace can be planned."""


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DSAWorkspaceGeometryError(f"{name} must be a Python int")
    if value < minimum:
        raise DSAWorkspaceGeometryError(f"{name} must be >= {minimum}")
    if value > MAX_SIGNED_64:
        raise DSAWorkspaceGeometryError(f"{name} exceeds signed 64-bit ABI")
    return value


def ceil_div(value: int, divisor: int) -> int:
    """Ceiling division without the fixed-width ``value + divisor - 1`` form."""

    value = _integer("value", value)
    divisor = _integer("divisor", divisor, minimum=1)
    quotient, remainder = divmod(value, divisor)
    return quotient + int(remainder != 0)


def _checked_product(name: str, *values: int) -> int:
    result = 1
    for value in values:
        value = _integer(name, value)
        result *= value
        if result > MAX_SIGNED_64:
            raise DSAWorkspaceGeometryError(f"{name} exceeds signed 64-bit ABI")
    return result


@dataclass(frozen=True)
class DSAIndexerWorkspaceGeometry:
    context_tokens: int
    index_kpool: int
    pool_count: int
    num_query_rows: int
    query_block_rows: int
    query_block_count: int
    fp32_logits_workspace_bytes: int
    max_workspace_bytes: int
    selected_pool_count: int
    selected_token_width: int

    def descriptor(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class DSAIndexerMemoryAccounting:
    """Explicit transient and persistent byte classes for one geometry."""

    logits_workspace_bytes: int
    topk_score_scratch_bytes: int
    topk_index_scratch_bytes: int
    indexpool_expansion_scratch_bytes: int
    selected_index_output_bytes: int
    persistent_pool_keys_bytes: int
    persistent_pool_indices_bytes: int
    persistent_pool_validity_bytes: int
    anonymous_allocation_bytes: int = 0

    def descriptor(self) -> dict[str, object]:
        values = asdict(self)
        return {
            "transient": {
                key: values[key]
                for key in (
                    "logits_workspace_bytes",
                    "topk_score_scratch_bytes",
                    "topk_index_scratch_bytes",
                    "indexpool_expansion_scratch_bytes",
                    "selected_index_output_bytes",
                )
            },
            "persistent_indexpool": {
                key: values[key]
                for key in (
                    "persistent_pool_keys_bytes",
                    "persistent_pool_indices_bytes",
                    "persistent_pool_validity_bytes",
                )
            },
            "anonymous_allocation_bytes": self.anonymous_allocation_bytes,
        }


def plan_dsa_indexer_workspace(
    *,
    context_tokens: int,
    num_query_rows: int,
    index_kpool: int = DEFAULT_INDEX_KPOOL,
    index_topk: int = DEFAULT_INDEX_TOPK,
    max_workspace_bytes: int = DEFAULT_MAX_WORKSPACE_BYTES,
) -> DSAIndexerWorkspaceGeometry:
    """Plan a score workspace from pooled key geometry, never logical tokens."""

    context_tokens = _integer("context_tokens", context_tokens)
    num_query_rows = _integer("num_query_rows", num_query_rows)
    index_kpool = _integer("index_kpool", index_kpool, minimum=1)
    index_topk = _integer("index_topk", index_topk, minimum=1)
    max_workspace_bytes = _integer(
        "max_workspace_bytes", max_workspace_bytes, minimum=FP32_BYTES
    )
    if index_topk % index_kpool:
        raise DSAWorkspaceGeometryError(
            "index_topk must be divisible by index_kpool"
        )

    pool_count = ceil_div(context_tokens, index_kpool)
    selected_pool_count = min(index_topk // index_kpool, pool_count)
    selected_token_width = index_topk + index_kpool - 1
    if num_query_rows == 0:
        query_block_rows = 0
        block_count = 0
        logits_bytes = 0
    elif pool_count == 0:
        query_block_rows = num_query_rows
        block_count = 1
        logits_bytes = 0
    else:
        bytes_per_query_row = _checked_product(
            "bytes_per_query_row", pool_count, FP32_BYTES
        )
        if bytes_per_query_row > max_workspace_bytes:
            raise DSAWorkspaceGeometryError(
                "one pooled-logit query row exceeds the configured workspace budget"
            )
        max_query_rows = max_workspace_bytes // bytes_per_query_row
        query_block_rows = min(num_query_rows, max_query_rows)
        block_count = ceil_div(num_query_rows, query_block_rows)
        logits_bytes = _checked_product(
            "fp32_logits_workspace_bytes",
            query_block_rows,
            pool_count,
            FP32_BYTES,
        )
    if logits_bytes > max_workspace_bytes:
        raise AssertionError("planned logits workspace exceeds configured budget")
    return DSAIndexerWorkspaceGeometry(
        context_tokens=context_tokens,
        index_kpool=index_kpool,
        pool_count=pool_count,
        num_query_rows=num_query_rows,
        query_block_rows=query_block_rows,
        query_block_count=block_count,
        fp32_logits_workspace_bytes=logits_bytes,
        max_workspace_bytes=max_workspace_bytes,
        selected_pool_count=selected_pool_count,
        selected_token_width=selected_token_width,
    )


def account_dsa_indexer_memory(
    geometry: DSAIndexerWorkspaceGeometry,
    *,
    index_head_dim: int,
) -> DSAIndexerMemoryAccounting:
    """Classify known planner bytes without charging scratch to IndexPool state."""

    if not isinstance(geometry, DSAIndexerWorkspaceGeometry):
        raise TypeError("memory accounting requires planned DSA geometry")
    index_head_dim = _integer("index_head_dim", index_head_dim, minimum=1)
    block_rows = geometry.query_block_rows
    selected_pools = geometry.selected_pool_count
    selected_width = geometry.selected_token_width
    return DSAIndexerMemoryAccounting(
        logits_workspace_bytes=geometry.fp32_logits_workspace_bytes,
        topk_score_scratch_bytes=_checked_product(
            "topk_score_scratch_bytes", block_rows, selected_pools, FP32_BYTES
        ),
        topk_index_scratch_bytes=_checked_product(
            # Exact argsort produces one index for every candidate pool.  Do
            # not under-account this as only the selected top-k suffix.
            "topk_index_scratch_bytes",
            block_rows,
            geometry.pool_count,
            INT32_BYTES,
        ),
        indexpool_expansion_scratch_bytes=_checked_product(
            "indexpool_expansion_scratch_bytes",
            block_rows,
            selected_width,
            INT32_BYTES + BOOL_BYTES,
        ),
        selected_index_output_bytes=_checked_product(
            "selected_index_output_bytes",
            geometry.num_query_rows,
            selected_width,
            INT32_BYTES + BOOL_BYTES,
        ),
        persistent_pool_keys_bytes=_checked_product(
            "persistent_pool_keys_bytes",
            geometry.pool_count,
            index_head_dim,
            BF16_BYTES,
        ),
        persistent_pool_indices_bytes=_checked_product(
            "persistent_pool_indices_bytes",
            geometry.pool_count,
            geometry.index_kpool,
            INT64_BYTES,
        ),
        persistent_pool_validity_bytes=_checked_product(
            "persistent_pool_validity_bytes", geometry.pool_count, BOOL_BYTES
        ),
    )
