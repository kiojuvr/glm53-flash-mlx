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
