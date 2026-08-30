"""Production caches for GLM-5.3's NoPE DSA attention.

The compact backend is deliberately single-sequence and opt-in.  It keeps one
authoritative latent buffer plus contiguous IndexPool rows and a bounded raw
rollback suffix; the full packed Indexer token history is never retained.
"""

from __future__ import annotations

import mlx.core as mx

from .indexpool import INDEXPOOL_SENTINEL, sanitize_indexpool_indices

DEFAULT_CACHE_STEP = 256
DEFAULT_ROLLBACK_WINDOW = 16
DEFAULT_CAPACITY_TOKENS = 4352


def _round_up(value: int, step: int) -> int:
    return ((value + step - 1) // step) * step


def _concat(left: mx.array | None, right: mx.array) -> mx.array:
    if left is None or left.shape[1] == 0:
        return right
    if right.shape[1] == 0:
        return left
    return mx.concatenate([left, right], axis=1)


def pool_indexer_states(
    keys: mx.array,
    gate_scores: mx.array,
    valid: mx.array,
    *,
    index_kpool: int,
    compress_ape: mx.array,
):
    """Apply the audited Indexer pooling tree without a model reference."""
    batch, tokens, head_dim = keys.shape
    pools = (tokens + index_kpool - 1) // index_kpool
    any_valid = mx.any(valid, axis=-1)
    first_key = mx.where(
        any_valid,
        mx.argmax(valid.astype(mx.int32), axis=-1),
        mx.array(tokens),
    )
    pool_offsets = mx.arange(pools * index_kpool).reshape(
        1, pools, index_kpool
    )
    pool_indices = first_key[:, None, None] + pool_offsets
    safe = mx.clip(pool_indices, 0, tokens - 1)
    flat = safe.reshape(batch, pools * index_kpool)
    index_columns = mx.broadcast_to(
        flat[..., None], (batch, pools * index_kpool, head_dim)
    )
    grouped_keys = mx.take_along_axis(keys, index_columns, axis=1).reshape(
        batch, pools, index_kpool, head_dim
    )
    grouped_gate = mx.take_along_axis(
        gate_scores, index_columns, axis=1
    ).reshape(batch, pools, index_kpool, head_dim)
    grouped_valid = (
        mx.take_along_axis(valid.astype(mx.int32), flat, axis=1).reshape(
            batch, pools, index_kpool
        )
        > 0
    )
    grouped_valid = grouped_valid & (pool_indices < tokens)
    pool_valid = mx.all(grouped_valid, axis=-1)
    pool_indices = mx.where(grouped_valid, pool_indices, INDEXPOOL_SENTINEL)
    logits = grouped_gate + compress_ape[None, None]
    logits = mx.where(grouped_valid[..., None], logits, -1e30)
    probabilities = mx.softmax(logits, axis=2)
    probabilities = mx.where(mx.isnan(probabilities), 0.0, probabilities)
    pool_keys = mx.sum(probabilities * grouped_keys, axis=2)
    return pool_keys, pool_indices, pool_valid


class SingleNoPELatentCache:
    """KVCache-compatible storage which owns one latent K/V buffer."""

    step = DEFAULT_CACHE_STEP

    def __init__(
        self,
        *,
        capacity_tokens: int = DEFAULT_CAPACITY_TOKENS,
        rollback_window: int = DEFAULT_ROLLBACK_WINDOW,
    ):
        if capacity_tokens < 0:
            raise ValueError("latent capacity must be non-negative")
        self._latent = None
        self.offset = 0
        self.capacity_tokens = int(capacity_tokens)
        self.rollback_window = int(rollback_window)

    @property
    def keys(self):
        return self._latent

    @keys.setter
    def keys(self, value):
        self._latent = value

    @property
    def values(self):
        # GLM-5.3 NoPE MLA consumes the same latent representation as K and V.
        return self._latent

    @values.setter
    def values(self, value):
        self._latent = value

    def _ensure_capacity(self, required: int, template: mx.array | None = None) -> None:
        current = 0 if self._latent is None else int(self._latent.shape[2])
        if required <= current:
            return
        if template is None and self._latent is None:
            raise ValueError("a latent template is required for first allocation")
        source = template if self._latent is None else self._latent
        capacity = _round_up(required, self.step)
        shape = (int(source.shape[0]), int(source.shape[1]), capacity, int(source.shape[3]))
        grown = mx.zeros(shape, dtype=source.dtype)
        if self._latent is not None and self.offset:
            grown[..., : self.offset, :] = self._latent[..., : self.offset, :]
        self._latent = grown

    @property
    def physical_capacity_tokens(self) -> int:
        return 0 if self._latent is None else int(self._latent.shape[2])

    def reserve_until(self, absolute_token_capacity: int) -> None:
        if absolute_token_capacity < 0:
            raise ValueError("latent capacity must be non-negative")
        self.capacity_tokens = max(
            self.capacity_tokens, int(absolute_token_capacity)
        )
        if self._latent is not None:
            self._ensure_capacity(max(self.offset, self.capacity_tokens))

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        if int(keys.shape[0]) != 1:
            raise ValueError("compact NoPE DSA cache supports batch size 1 only")
        if keys.shape != values.shape:
            raise ValueError("NoPE latent K/V inputs must have identical shapes")
        count = int(keys.shape[2])
        previous = self.offset
        next_offset = previous + count
        self._ensure_capacity(max(next_offset, self.capacity_tokens), keys)
        self._latent[..., previous:next_offset, :] = keys
        self.offset = next_offset
        logical = self._latent[..., :next_offset, :]
        return logical, logical

    def size(self) -> int:
        return self.offset

    def is_trimmable(self) -> bool:
        return True

    def validate_trim(self, tokens: int) -> None:
        if tokens < 1 or tokens > self.rollback_window:
            raise ValueError(
                f"single NoPE latent trim must be within [1, {self.rollback_window}]"
            )
        if tokens > self.offset:
            raise ValueError("single NoPE latent trim exceeds cached token count")

    def trim(self, tokens: int) -> int:
        self.validate_trim(tokens)
        self.offset -= tokens
        return tokens

    def empty(self) -> bool:
        return self._latent is None or self.offset == 0

    def make_mask(self, *args, **kwargs):
        from mlx_vlm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)

    @property
    def state(self):
        if self._latent is None:
            return (None,)
        return (self._latent[..., : self.offset, :],)

    @state.setter
    def state(self, value):
        latent = value[0]
        self._latent = latent
        self.offset = 0 if latent is None else int(latent.shape[2])
        if not hasattr(self, "capacity_tokens"):
            self.capacity_tokens = 0
        if not hasattr(self, "rollback_window"):
            self.rollback_window = DEFAULT_ROLLBACK_WINDOW

    @property
    def meta_state(self):
        return (
            str(self.offset),
            str(self.capacity_tokens),
            str(self.rollback_window),
            str(self.step),
        )

    @meta_state.setter
    def meta_state(self, value):
        offset, capacity, rollback, step = map(int, value)
        self.offset = offset
        self.capacity_tokens = capacity
        self.rollback_window = rollback
        self.step = step
        if self._latent is not None:
            self._ensure_capacity(max(self.offset, self.capacity_tokens))

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls.__new__(cls)
        obj.state = state
        obj.meta_state = meta_state
        return obj

    def prefix_cache_snapshot(self):
        return {"state": self.state, "meta_state": self.meta_state}

    def prefix_cache_restore(self, snapshot) -> None:
        self.state = snapshot["state"]
        self.meta_state = snapshot["meta_state"]

    @property
    def nbytes(self) -> int:
        return 0 if self._latent is None else int(self._latent.nbytes)

    def dependency_arrays(self):
        return () if self._latent is None else (self._latent,)


class CompactIndexPoolCache:
    """Authoritative contiguous IndexPool plus a bounded raw rollback suffix."""

    step = DEFAULT_CACHE_STEP

    def __init__(
        self,
        indexer,
        *,
        capacity_tokens: int = DEFAULT_CAPACITY_TOKENS,
        rollback_window: int = DEFAULT_ROLLBACK_WINDOW,
    ):
        if capacity_tokens < 0:
            raise ValueError("IndexPool capacity must be non-negative")
        self.index_kpool = int(indexer.index_kpool)
        self.index_topk = int(indexer.index_topk)
        self.head_dim = int(indexer.head_dim)
        self.always_select_tail = bool(indexer.index_kpool_always_select_tail)
        self.rollback_window = int(rollback_window)
        self.raw_state_window = self.rollback_window + self.index_kpool - 1
        self.capacity_tokens = int(capacity_tokens)
        self.total_tokens = 0
        self.logical_pool_count = 0
        self.pool_capacity = 0
        self.pool_keys = None
        self.pool_indices = None
        self.pool_valid = None
        self.raw_keys = None
        self.raw_gates = None
        self.raw_valid = None
        self.raw_positions = None
        # This tiny tensor is sufficient to rebuild a partial pool after an APC
        # restore; no model or Indexer object is part of the cache state.
        self.compress_ape = indexer.index_kpool_compress_ape

    @property
    def offset(self) -> int:
        return self.total_tokens

    @property
    def active_tail_count(self) -> int:
        return self.total_tokens % self.index_kpool

    @property
    def raw_token_count(self) -> int:
        return 0 if self.raw_keys is None else int(self.raw_keys.shape[1])

    def logical_pool(self):
        if self.pool_keys is None:
            return None, None, None
        end = self.logical_pool_count
        return (
            self.pool_keys[:, :end],
            self.pool_indices[:, :end],
            self.pool_valid[:, :end],
        )

    def _validate_indexer(self, indexer) -> None:
        if (
            int(indexer.index_kpool) != self.index_kpool
            or int(indexer.index_topk) != self.index_topk
            or int(indexer.head_dim) != self.head_dim
        ):
            raise ValueError("restored compact IndexPool metadata does not match Indexer")
        if indexer.index_kpool_compress_ape.shape != self.compress_ape.shape:
            raise ValueError("restored compact IndexPool APE shape does not match Indexer")

    def validate_update(self, indexer, *, batch: int, length: int) -> bool:
        if batch != 1:
            raise ValueError("compact NoPE DSA cache supports batch size 1 only")
        self._validate_indexer(indexer)
        resulting_tokens = self.total_tokens + length
        short_bypass = (
            getattr(indexer, "bypass_short", True)
            and resulting_tokens <= self.index_topk
        )
        if length != 1 and not short_bypass:
            raise ValueError(
                "compact NoPE DSA sparse path supports incremental decode only; "
                "long sparse prefill is not admitted"
            )
        return short_bypass

    def _ensure_pool_capacity(self, rows: int, dtype) -> None:
        if rows <= self.pool_capacity:
            return
        capacity = _round_up(rows * self.index_kpool, self.step) // self.index_kpool
        keys = mx.zeros((1, capacity, self.head_dim), dtype=dtype)
        indices = mx.full(
            (1, capacity, self.index_kpool),
            INDEXPOOL_SENTINEL,
            dtype=mx.int64,
        )
        valid = mx.zeros((1, capacity), dtype=mx.bool_)
        if self.pool_keys is not None and self.logical_pool_count:
            end = self.logical_pool_count
            keys[:, :end] = self.pool_keys[:, :end]
            indices[:, :end] = self.pool_indices[:, :end]
            valid[:, :end] = self.pool_valid[:, :end]
        self.pool_keys, self.pool_indices, self.pool_valid = keys, indices, valid
        self.pool_capacity = capacity

    @property
    def physical_capacity_rows(self) -> int:
        return self.pool_capacity

    def _required_pool_rows(self, written_end: int) -> int:
        target_tokens = max(self.total_tokens, self.capacity_tokens)
        reserved_rows = (
            target_tokens + self.index_kpool - 1
        ) // self.index_kpool
        return max(written_end, reserved_rows)

    def reserve_until(self, absolute_token_capacity: int) -> None:
        if absolute_token_capacity < 0:
            raise ValueError("IndexPool capacity must be non-negative")
        self.capacity_tokens = max(
            self.capacity_tokens, int(absolute_token_capacity)
        )
        if self.pool_keys is not None:
            rows = self._required_pool_rows(self.logical_pool_count)
            self._ensure_pool_capacity(rows, self.pool_keys.dtype)

    def _set_raw(self, keys, gates, valid, positions) -> None:
        if keys.shape[1] > self.raw_state_window:
            keys = keys[:, -self.raw_state_window :]
            gates = gates[:, -self.raw_state_window :]
            valid = valid[:, -self.raw_state_window :]
            positions = positions[:, -self.raw_state_window :]
        self.raw_keys = keys
        self.raw_gates = gates
        self.raw_valid = valid
        self.raw_positions = positions

    def _pool_suffix(self, start: int, keys, gates, valid):
        pooled = pool_indexer_states(
            keys,
            gates,
            valid,
            index_kpool=self.index_kpool,
            compress_ape=self.compress_ape,
        )
        absolute_indices = mx.where(
            pooled[1] >= 0,
            pooled[1] + start,
            INDEXPOOL_SENTINEL,
        )
        return pooled[0], absolute_indices, pooled[2]

    def _write_pool_rows(self, start: int, pooled) -> None:
        end = start // self.index_kpool + int(pooled[0].shape[1])
        self._ensure_pool_capacity(
            self._required_pool_rows(end), pooled[0].dtype
        )
        row = start // self.index_kpool
        self.pool_keys[:, row:end] = pooled[0]
        self.pool_indices[:, row:end] = pooled[1]
        self.pool_valid[:, row:end] = pooled[2]
        self.logical_pool_count = (
            self.total_tokens + self.index_kpool - 1
        ) // self.index_kpool

    def _write_pool_suffix(self, start: int, keys, gates, valid) -> None:
        self._write_pool_rows(
            start,
            self._pool_suffix(start, keys, gates, valid),
        )

    def _append_projected(self, keys, gates, valid) -> None:
        previous = self.total_tokens
        count = int(keys.shape[1])
        stable = previous // self.index_kpool
        old_partial = previous % self.index_kpool
        if old_partial:
            suffix_keys = _concat(None, self.raw_keys[:, -old_partial:])
            suffix_gates = _concat(None, self.raw_gates[:, -old_partial:])
            suffix_valid = _concat(None, self.raw_valid[:, -old_partial:])
            suffix_keys = _concat(suffix_keys, keys)
            suffix_gates = _concat(suffix_gates, gates)
            suffix_valid = _concat(suffix_valid, valid)
        else:
            suffix_keys, suffix_gates, suffix_valid = keys, gates, valid
        positions = mx.arange(previous, previous + count, dtype=mx.int64)[None]
        raw_keys = _concat(self.raw_keys, keys)
        raw_gates = _concat(self.raw_gates, gates)
        raw_valid = _concat(self.raw_valid, valid)
        raw_positions = _concat(self.raw_positions, positions)
        self.total_tokens = previous + count
        self._write_pool_suffix(
            stable * self.index_kpool,
            suffix_keys,
            suffix_gates,
            suffix_valid,
        )
        self._set_raw(raw_keys, raw_gates, raw_valid, raw_positions)

    def _decode_selection(self, indexer, x, qr, valid_cur):
        pool_keys, pool_indices, pool_valid = self.logical_pool()
        pool_count = self.logical_pool_count
        query = indexer.wq_b(qr).reshape(1, 1, indexer.n_heads, indexer.head_dim)
        scores = query @ pool_keys[:, None].swapaxes(-1, -2)
        scores = mx.maximum(scores * indexer.softmax_scale, 0.0)
        weights = indexer.weights_proj(x) * (indexer.n_heads**-0.5)
        index_scores = mx.sum(weights[..., None] * scores, axis=2)
        pool_end = mx.clip(pool_indices[..., -1], 0, self.total_tokens - 1)
        valid_candidates = (
            (pool_end[:, None, :] < self.total_tokens) & pool_valid[:, None]
        )
        index_scores = mx.where(valid_candidates, index_scores, -1e30)
        select_k = min(self.index_topk // self.index_kpool, pool_count)
        order = mx.argsort(-index_scores, axis=-1)
        selected = order[..., :select_k]
        selected_valid = mx.take_along_axis(valid_candidates, selected, axis=-1)
        source = mx.broadcast_to(
            pool_indices[:, None],
            (1, 1, pool_count, self.index_kpool),
        )
        selected_expanded = mx.broadcast_to(
            selected[..., None],
            (1, 1, select_k, self.index_kpool),
        )
        chosen = mx.take_along_axis(source, selected_expanded, axis=2)
        topk = chosen.reshape(1, 1, select_k * self.index_kpool)
        chosen_valid = mx.broadcast_to(
            selected_valid[..., None],
            (1, 1, select_k, self.index_kpool),
        ).reshape(1, 1, select_k * self.index_kpool)
        topk = mx.where(chosen_valid, topk, INDEXPOOL_SENTINEL)
        if self.always_select_tail and self.index_kpool > 1:
            active = self.active_tail_count
            tail_positions = self.raw_positions[:, -active:] if active else mx.zeros((1, 0), dtype=mx.int64)
            tail_valid = self.raw_valid[:, -active:] if active else mx.zeros((1, 0), dtype=mx.bool_)
            tail = mx.where(tail_valid, tail_positions, INDEXPOOL_SENTINEL)
            missing = self.index_kpool - 1 - active
            if missing > 0:
                tail = mx.concatenate(
                    [
                        tail,
                        mx.full(
                            (1, missing),
                            INDEXPOOL_SENTINEL,
                            dtype=mx.int64,
                        ),
                    ],
                    axis=-1,
                )
            topk = mx.concatenate([topk, tail[:, None]], axis=-1)
        width = self.index_topk + (
            self.index_kpool - 1 if self.always_select_tail else 0
        )
        if topk.shape[-1] < width:
            topk = mx.concatenate(
                [
                    topk,
                    mx.full(
                        (1, 1, width - topk.shape[-1]),
                        INDEXPOOL_SENTINEL,
                        dtype=topk.dtype,
                    ),
                ],
                axis=-1,
            )
        topk = mx.where(valid_cur[..., None], topk, INDEXPOOL_SENTINEL)
        return sanitize_indexpool_indices(
            topk[..., :width][:, None].astype(mx.int32), self.total_tokens
        )

    def update(self, indexer, x: mx.array, qr: mx.array, mask=None):
        length = int(x.shape[1])
        short_bypass = self.validate_update(
            indexer, batch=int(x.shape[0]), length=length
        )
        keys = indexer.k_norm(indexer.wk(x)).reshape(1, length, self.head_dim)
        gates = x @ indexer.index_kpool_compress_gate.swapaxes(-1, -2)
        if mask is not None and mask.dtype == mx.bool_ and mask.shape == (1, length):
            valid = mask
        else:
            valid = mx.ones((1, length), dtype=mx.bool_)
        self._append_projected(keys, gates, valid)
        if short_bypass:
            return None
        return self._decode_selection(indexer, x, qr, valid)

    def is_trimmable(self) -> bool:
        return True

    def size(self) -> int:
        return self.total_tokens

    def validate_trim(self, tokens: int) -> None:
        if tokens < 1 or tokens > self.rollback_window:
            raise ValueError(
                f"compact IndexPool trim must be within [1, {self.rollback_window}]"
            )
        if tokens > self.total_tokens:
            raise ValueError("compact IndexPool trim exceeds cached token count")
        if self.raw_token_count - tokens < 0:
            raise ValueError("compact IndexPool raw state cannot reconstruct trim target")

    def trim(self, tokens: int) -> int:
        self.validate_trim(tokens)
        target = self.total_tokens - tokens
        keep = self.raw_token_count - tokens
        raw_keys = self.raw_keys[:, :keep]
        raw_gates = self.raw_gates[:, :keep]
        raw_valid = self.raw_valid[:, :keep]
        raw_positions = self.raw_positions[:, :keep]
        active = target % self.index_kpool
        pooled = None
        if active:
            start = target - active
            pooled = self._pool_suffix(
                start,
                raw_keys[:, -active:],
                raw_gates[:, -active:],
                raw_valid[:, -active:],
            )
        self.raw_keys = raw_keys
        self.raw_gates = raw_gates
        self.raw_valid = raw_valid
        self.raw_positions = raw_positions
        self.total_tokens = target
        self.logical_pool_count = (
            target + self.index_kpool - 1
        ) // self.index_kpool
        if pooled is not None:
            self._write_pool_rows(start, pooled)
        return tokens

    def empty(self) -> bool:
        return self.total_tokens == 0

    @property
    def state(self):
        pool = self.logical_pool()
        return (
            *pool,
            self.raw_keys,
            self.raw_gates,
            self.raw_valid,
            self.raw_positions,
            self.compress_ape,
        )

    @state.setter
    def state(self, value):
        (
            self.pool_keys,
            self.pool_indices,
            self.pool_valid,
            self.raw_keys,
            self.raw_gates,
            self.raw_valid,
            self.raw_positions,
            self.compress_ape,
        ) = value
        self.logical_pool_count = (
            0 if self.pool_keys is None else int(self.pool_keys.shape[1])
        )
        self.pool_capacity = self.logical_pool_count
        self.total_tokens = 0

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    self.total_tokens,
                    self.logical_pool_count,
                    self.capacity_tokens,
                    self.rollback_window,
                    self.index_kpool,
                    self.index_topk,
                    self.head_dim,
                    int(self.always_select_tail),
                    self.step,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, value):
        (
            total,
            logical,
            capacity,
            rollback,
            kpool,
            topk,
            head_dim,
            always_tail,
            step,
        ) = map(int, value)
        self.total_tokens = total
        self.logical_pool_count = logical
        self.capacity_tokens = capacity
        self.rollback_window = rollback
        self.index_kpool = kpool
        self.index_topk = topk
        self.head_dim = head_dim
        self.always_select_tail = bool(always_tail)
        self.step = step
        self.raw_state_window = rollback + kpool - 1
        if self.pool_keys is not None:
            wanted = self._required_pool_rows(self.logical_pool_count)
            self._ensure_pool_capacity(wanted, self.pool_keys.dtype)

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls.__new__(cls)
        obj.state = state
        obj.meta_state = meta_state
        return obj

    def prefix_cache_snapshot(self):
        return {"state": self.state, "meta_state": self.meta_state}

    def prefix_cache_restore(self, snapshot) -> None:
        self.state = snapshot["state"]
        self.meta_state = snapshot["meta_state"]

    @property
    def nbytes(self) -> int:
        values = (
            self.pool_keys,
            self.pool_indices,
            self.pool_valid,
            self.raw_keys,
            self.raw_gates,
            self.raw_valid,
            self.raw_positions,
            self.compress_ape,
        )
        return sum(int(value.nbytes) for value in values if value is not None)

    def dependency_arrays(self):
        return tuple(
            value
            for value in (
                self.pool_keys,
                self.pool_indices,
                self.pool_valid,
                self.raw_keys,
                self.raw_gates,
                self.raw_valid,
                self.raw_positions,
                self.compress_ape,
            )
            if value is not None
        )


def make_compact_nope_dsa_cache(indexer, *, capacity_tokens: int):
    from mlx_vlm.models.cache import CacheList

    return CacheList(
        SingleNoPELatentCache(capacity_tokens=capacity_tokens),
        CompactIndexPoolCache(indexer, capacity_tokens=capacity_tokens),
    )


def is_compact_nope_dsa_cache(cache) -> bool:
    return isinstance(cache, CompactIndexPoolCache)
