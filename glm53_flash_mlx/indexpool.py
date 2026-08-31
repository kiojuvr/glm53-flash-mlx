"""Sentinel and range contracts for GLM-5.3-Flash NoPE IndexPool."""

from __future__ import annotations

import mlx.core as mx

INDEXPOOL_SENTINEL = -1


def indexpool_cache_kv_len(cache, current_tokens: int) -> int:
    """Return the physical KV column count without reading per-row offsets."""
    if cache is None:
        return current_tokens
    physical_length = getattr(cache, "_idx", None)
    if isinstance(physical_length, int):
        return physical_length
    logical_offset = getattr(cache, "offset", None)
    if isinstance(logical_offset, int):
        return logical_offset
    keys = getattr(cache, "keys", None)
    if keys is not None:
        return keys.shape[-2]
    return current_tokens


def sanitize_indexpool_indices(indices, kv_len: int):
    """Map every non-sentinel out-of-range index to the canonical sentinel."""
    if kv_len <= 0:
        raise ValueError(f"kv_len must be positive, got {kv_len}")
    valid = (indices >= 0) & (indices < kv_len)
    return mx.where(valid, indices, INDEXPOOL_SENTINEL)


def expand_selected_pools(
    selected_pool_rows: mx.array,
    pool_token_indices: mx.array,
    selected_valid: mx.array,
    *,
    kv_len: int,
    index_topk: int,
    index_kpool: int,
    tail_positions: mx.array,
    tail_valid: mx.array,
    always_select_tail: bool,
) -> tuple[mx.array, mx.array]:
    """Expand selected pool rows without observing KV storage or its dtype."""
    if kv_len <= 0:
        raise ValueError(f"kv_len must be positive, got {kv_len}")
    if index_topk <= 0 or index_kpool <= 0:
        raise ValueError("index_topk and index_kpool must be positive")
    if index_topk % index_kpool:
        raise ValueError("index_topk must be divisible by index_kpool")
    if selected_pool_rows.ndim != 3 or selected_valid.shape != selected_pool_rows.shape:
        raise ValueError("selected pool rows and validity must have shape [B, Q, K]")
    if pool_token_indices.ndim != 3:
        raise ValueError("pool token indices must have shape [B, P, index_kpool]")
    if pool_token_indices.shape[0] != selected_pool_rows.shape[0]:
        raise ValueError("selected rows and pool token indices must share batch size")
    if pool_token_indices.shape[-1] != index_kpool:
        raise ValueError("pool token index width does not match index_kpool")

    batch, queries, selected_count = selected_pool_rows.shape
    pool_count = int(pool_token_indices.shape[1])
    row_valid = (selected_pool_rows >= 0) & (selected_pool_rows < pool_count)
    safe_rows = mx.where(row_valid, selected_pool_rows, 0)
    source = mx.broadcast_to(
        pool_token_indices[:, None],
        (batch, queries, pool_count, index_kpool),
    )
    selected_expanded = mx.broadcast_to(
        safe_rows[..., None],
        (batch, queries, selected_count, index_kpool),
    )
    chosen = mx.take_along_axis(source, selected_expanded, axis=2).reshape(
        batch, queries, selected_count * index_kpool
    )
    pool_valid = mx.broadcast_to(
        (selected_valid & row_valid)[..., None],
        (batch, queries, selected_count, index_kpool),
    ).reshape(batch, queries, selected_count * index_kpool)
    pool_valid = pool_valid & (chosen >= 0) & (chosen < kv_len)
    indices = mx.where(pool_valid, chosen, INDEXPOOL_SENTINEL)
    valid = pool_valid

    tail_width = index_kpool - 1 if always_select_tail and index_kpool > 1 else 0
    if tail_width:
        if tail_positions.ndim == 2:
            tail_positions = mx.broadcast_to(
                tail_positions[:, None],
                (batch, queries, tail_positions.shape[-1]),
            )
        if tail_valid.ndim == 2:
            tail_valid = mx.broadcast_to(
                tail_valid[:, None],
                (batch, queries, tail_valid.shape[-1]),
            )
        if tail_positions.ndim != 3 or tail_valid.shape != tail_positions.shape:
            raise ValueError("tail positions and validity must have shape [B, Q, T]")
        if tail_positions.shape[0] != batch or tail_positions.shape[1] != queries:
            raise ValueError("tail positions must match selected batch and query axes")
        if tail_positions.shape[-1] > tail_width:
            raise ValueError("tail may contain only the incomplete pool")
        valid_tail = tail_valid & (tail_positions >= 0) & (tail_positions < kv_len)
        tail = mx.where(valid_tail, tail_positions, INDEXPOOL_SENTINEL)
        missing_tail = tail_width - int(tail.shape[-1])
        if missing_tail:
            tail = mx.concatenate(
                [
                    tail,
                    mx.full(
                        (batch, queries, missing_tail),
                        INDEXPOOL_SENTINEL,
                        dtype=tail.dtype,
                    ),
                ],
                axis=-1,
            )
            valid_tail = mx.concatenate(
                [valid_tail, mx.zeros((batch, queries, missing_tail), dtype=mx.bool_)],
                axis=-1,
            )
        indices = mx.concatenate([indices, tail], axis=-1)
        valid = mx.concatenate([valid, valid_tail], axis=-1)

    width = index_topk + tail_width
    if indices.shape[-1] < width:
        padding = width - int(indices.shape[-1])
        indices = mx.concatenate(
            [
                indices,
                mx.full(
                    (batch, queries, padding),
                    INDEXPOOL_SENTINEL,
                    dtype=indices.dtype,
                ),
            ],
            axis=-1,
        )
        valid = mx.concatenate(
            [valid, mx.zeros((batch, queries, padding), dtype=mx.bool_)], axis=-1
        )
    indices = sanitize_indexpool_indices(indices[..., :width], kv_len).astype(mx.int32)
    valid = valid[..., :width] & (indices >= 0) & (indices < kv_len)
    return mx.where(valid, indices, INDEXPOOL_SENTINEL), valid


def prepare_decode_indexpool_gather(topk_indices, kv_len: int):
    """Return gather-safe decode indices and an independent validity mask."""
    if kv_len <= 0:
        raise ValueError(f"kv_len must be positive, got {kv_len}")
    valid = (topk_indices >= 0) & (topk_indices < kv_len)
    safe = mx.where(valid, topk_indices, 0)
    return safe, valid


def build_prefill_indexpool_mask(topk_indices, kv_len: int):
    """Build a sparse mask through a temporary, discarded sentinel column."""
    if kv_len <= 0:
        raise ValueError(f"kv_len must be positive, got {kv_len}")
    valid = (topk_indices >= 0) & (topk_indices < kv_len)
    safe = mx.where(valid, topk_indices, kv_len)
    shape = list(topk_indices.shape)
    shape[-1] = kv_len + 1
    sparse = mx.zeros(shape, dtype=mx.bool_)
    sparse = mx.put_along_axis(sparse, safe, mx.array(True), axis=-1)
    return sparse[..., :kv_len], valid
