#!/usr/bin/env python3
"""Decompose incremental IndexPool update copies across all DSA layers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from glm53_flash_mlx.abi import MLX_VLM_REVISION, NOPE_DSA_CACHE_ABI
from glm53_flash_mlx.indexpool import sanitize_indexpool_indices
from glm53_flash_mlx.loader import load
from glm53_flash_mlx.manifest import EXPECTED_DSA, inspect_checkpoint
from glm53_flash_mlx.server import LEGACY_PROBE_MAX_PROMPT_TOKENS
from probe_long_context_dsa_decode_frontier import (
    _attention_phase,
    _gather_phase,
)
from probe_persistent_all_dsa_session_frontier import (
    _CapturingIndexer,
    _capacity,
    _combined_hash,
    _deterministic_rows,
    _expected_tail_sentinels,
    _hash_indices,
    _index_stats,
    _np,
    _percentile,
    _query,
)
from probe_single_buffer_nope_latent_cache_frontier import ProbeNoPELatentCache

CONTEXTS = (2049, 131072, 262144)
WARMUP_STEPS = 4
MEASURED_STEPS = 16
ARMS = ("reference_concat", "preallocated_pool_row", "segmented_pool")
UPDATE_PHASES = (
    "current_token_projection",
    "indexer_token_append",
    "partial_pool_recomputation",
    "complete_pool_prefix_carry",
    "pool_state_publication",
)
DOWNSTREAM_PHASES = (
    "pool_score",
    "selection",
    "pool_expansion",
    "latent_gather",
    "selected_attention",
)
ALL_PHASES = UPDATE_PHASES + DOWNSTREAM_PHASES
POOL_COMPONENTS = ("keys", "indices", "valid")


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), file=sys.stderr, flush=True)


def _eval(value) -> None:
    arrays = []

    def visit(item):
        if isinstance(item, mx.array):
            arrays.append(item)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    if arrays:
        mx.eval(*arrays)
    mx.synchronize()


def _time_phase(fn, sync_value=None):
    started = time.perf_counter()
    value = fn()
    _eval(sync_value(value) if sync_value is not None else value)
    return value, (time.perf_counter() - started) * 1000.0


def _zero_traffic() -> dict:
    return {
        "read_bytes": 0,
        "write_bytes": 0,
        "copy_bytes": 0,
        "pool_copy_bytes": {component: 0 for component in POOL_COMPONENTS},
    }


def _add_traffic(left: dict, right: dict) -> dict:
    output = {
        key: left[key] + right[key]
        for key in ("read_bytes", "write_bytes", "copy_bytes")
    }
    output["pool_copy_bytes"] = {
        component: left["pool_copy_bytes"][component]
        + right["pool_copy_bytes"][component]
        for component in POOL_COMPONENTS
    }
    return output


def _sum_traffic(rows) -> dict:
    total = _zero_traffic()
    for row in rows:
        total = _add_traffic(total, row)
    return total


@dataclass
class _ContiguousPool:
    keys: mx.array
    indices: mx.array
    valid: mx.array
    total_tokens: int
    logical_count: int
    capacity: int
    key_buffer: mx.array | None = None
    index_buffer: mx.array | None = None
    valid_buffer: mx.array | None = None

    def segments(self):
        return [(self.keys, self.indices, self.valid)]

    def gather_indices(self, selected):
        source = mx.broadcast_to(
            self.indices[:, None],
            (1, selected.shape[1], self.logical_count, self.indices.shape[-1]),
        )
        expanded = mx.broadcast_to(
            selected[..., None], selected.shape + (self.indices.shape[-1],)
        )
        return mx.take_along_axis(source, expanded, axis=2)


@dataclass
class _SegmentedPool:
    base_keys: mx.array
    base_indices: mx.array
    base_valid: mx.array
    total_tokens: int
    completed: list[tuple[mx.array, mx.array, mx.array]] = field(
        default_factory=list
    )
    tail: tuple[mx.array, mx.array, mx.array] | None = None

    @property
    def base_count(self) -> int:
        return int(self.base_keys.shape[1])

    @property
    def logical_count(self) -> int:
        return self.base_count + len(self.completed) + (self.tail is not None)

    @property
    def capacity(self) -> int:
        return self.logical_count

    def segments(self):
        rows = [(self.base_keys, self.base_indices, self.base_valid)]
        rows.extend(self.completed)
        if self.tail is not None:
            rows.append(self.tail)
        return [row for row in rows if row[0].shape[1]]

    def gather_indices(self, selected):
        base_count = self.base_count
        extra = self.completed + ([] if self.tail is None else [self.tail])
        extra_indices = mx.concatenate([row[1] for row in extra], axis=1)
        base_source = mx.broadcast_to(
            self.base_indices[:, None],
            (1, selected.shape[1], base_count, self.base_indices.shape[-1]),
        )
        base_ids = mx.clip(selected, 0, max(0, base_count - 1))
        base_expanded = mx.broadcast_to(
            base_ids[..., None], base_ids.shape + (self.base_indices.shape[-1],)
        )
        base_values = mx.take_along_axis(base_source, base_expanded, axis=2)
        extra_source = mx.broadcast_to(
            extra_indices[:, None],
            (
                1,
                selected.shape[1],
                extra_indices.shape[1],
                extra_indices.shape[-1],
            ),
        )
        extra_ids = mx.clip(
            selected - base_count, 0, max(0, extra_indices.shape[1] - 1)
        )
        extra_expanded = mx.broadcast_to(
            extra_ids[..., None],
            extra_ids.shape + (extra_indices.shape[-1],),
        )
        extra_values = mx.take_along_axis(
            extra_source, extra_expanded, axis=2
        )
        return mx.where(
            (selected < base_count)[..., None], base_values, extra_values
        )


@dataclass
class _Session:
    layer_id: int
    attention: object
    capture: _CapturingIndexer
    latent_cache: ProbeNoPELatentCache
    indexer_cache: object
    pool: _ContiguousPool | _SegmentedPool
    arm: str


def _new_latent_cache(attention, layer_id: int, context: int):
    history = context - 1
    capacity = _capacity(history + MEASURED_STEPS)
    latent = _deterministic_rows(
        capacity,
        attention.kv_lora_rank,
        1.125 + layer_id * 0.015625,
        mx.bfloat16,
    ).reshape(1, 1, capacity, attention.kv_lora_rank)
    cache = ProbeNoPELatentCache(latent, history)
    mx.eval(cache.keys)
    return cache


def _initial_indexer_state(
    attention,
    layer_id: int,
    context: int,
    *,
    token_headroom: bool,
):
    from mlx_vlm.models.cache import KVCache

    indexer = attention.indexer.delegate
    history = context - 1
    capacity = _capacity(
        history + MEASURED_STEPS if token_headroom else history
    )
    keys = _deterministic_rows(
        capacity, indexer.head_dim, 0.125 + layer_id * 0.015625, mx.bfloat16
    )[None]
    gates = _deterministic_rows(
        capacity, indexer.head_dim, 0.625 + layer_id * 0.015625, mx.bfloat16
    )[None]
    valid = (mx.arange(capacity) < history)[None, :, None]
    packed = mx.concatenate([keys, gates, valid.astype(mx.bfloat16)], axis=-1)
    logical = packed[:, :history]
    logical_keys, logical_gates, valid_channel = mx.split(
        logical, [indexer.head_dim, 2 * indexer.head_dim], axis=-1
    )
    pool = indexer._pooled_states(
        logical_keys, logical_gates, valid_channel[..., 0] > 0
    )
    cache = KVCache()
    cache.keys = packed[:, None]
    cache.values = mx.zeros((1, 1, capacity, 0), dtype=mx.bfloat16)
    cache.offset = history
    cache._no_pad = True
    mx.eval(packed, *pool)
    return cache, pool


def _make_pool(arm: str, initial, history: int, reserved_tokens: int):
    keys, indices, valid = initial
    logical_count = int(keys.shape[1])
    if arm == "reference_concat":
        return _ContiguousPool(
            keys, indices, valid, history, logical_count, logical_count
        )
    if arm == "preallocated_pool_row":
        capacity = (reserved_tokens + 3) // 4
        key_buffer = mx.zeros(
            (1, capacity, keys.shape[-1]), dtype=keys.dtype
        )
        index_buffer = mx.zeros(
            (1, capacity, indices.shape[-1]), dtype=indices.dtype
        )
        valid_buffer = mx.zeros((1, capacity), dtype=valid.dtype)
        key_buffer[:, :logical_count] = keys
        index_buffer[:, :logical_count] = indices
        valid_buffer[:, :logical_count] = valid
        mx.eval(key_buffer, index_buffer, valid_buffer)
        return _ContiguousPool(
            key_buffer[:, :logical_count],
            index_buffer[:, :logical_count],
            valid_buffer[:, :logical_count],
            history,
            logical_count,
            capacity,
            key_buffer,
            index_buffer,
            valid_buffer,
        )
    complete = history // 4
    tail = (
        (keys[:, complete:], indices[:, complete:], valid[:, complete:])
        if history % 4
        else None
    )
    return _SegmentedPool(
        keys[:, :complete],
        indices[:, :complete],
        valid[:, :complete],
        history,
        tail=tail,
    )


def _build_sessions(
    attentions,
    context: int,
    arm: str,
    *,
    token_headroom: bool = True,
):
    sessions = []
    reserved = context - 1 + (MEASURED_STEPS if token_headroom else 0)
    for layer_id in EXPECTED_DSA:
        attention = attentions[layer_id]
        indexer_cache, initial_pool = _initial_indexer_state(
            attention,
            layer_id,
            context,
            token_headroom=token_headroom,
        )
        pool = _make_pool(arm, initial_pool, context - 1, reserved)
        if isinstance(pool, _ContiguousPool):
            indexer_cache._pool = (
                pool.keys,
                pool.indices,
                pool.valid,
                pool.total_tokens,
            )
        else:
            indexer_cache._pool = None
        sessions.append(
            _Session(
                layer_id,
                attention,
                attention.indexer,
                _new_latent_cache(attention, layer_id, context),
                indexer_cache,
                pool,
                arm,
            )
        )
    mx.synchronize()
    return sessions


def _current_projection(session, x):
    indexer = session.capture.delegate
    key = indexer.k_norm(indexer.wk(x)).reshape(1, 1, indexer.head_dim)
    gate = x @ indexer.index_kpool_compress_gate.swapaxes(-1, -2)
    valid = mx.ones((1, 1), dtype=mx.bool_)
    packed = mx.concatenate(
        [key, gate, valid.astype(key.dtype)[..., None]], axis=-1
    )
    traffic = _zero_traffic()
    traffic["read_bytes"] = int(x.nbytes)
    traffic["write_bytes"] = int(key.nbytes + gate.nbytes)
    return {"key": key, "gate": gate, "valid": valid, "packed": packed}, traffic


def _append_token(session, projection):
    before_bytes = int(session.indexer_cache.keys.nbytes)
    packed_full, _ = session.indexer_cache.update_and_fetch(
        projection["packed"][:, None],
        mx.zeros((1, 1, 1, 0), dtype=projection["key"].dtype),
    )
    packed_full = packed_full[:, 0]
    after_bytes = int(session.indexer_cache.keys.nbytes)
    traffic = _zero_traffic()
    # MLX arrays are functional values.  Even with capacity headroom, slice
    # assignment is represented by a scatter producing a new buffer; it is not
    # evidence of an in-place write.  Count the carried prefix as copy traffic.
    traffic["read_bytes"] = before_bytes + int(projection["packed"].nbytes)
    traffic["write_bytes"] = after_bytes
    traffic["copy_bytes"] = before_bytes
    return packed_full, traffic


def _recompute_partial(session, packed_full):
    indexer = session.capture.delegate
    previous = session.pool.total_tokens
    stable = previous // indexer.index_kpool
    suffix_start = stable * indexer.index_kpool
    total = int(session.indexer_cache.offset)
    suffix = packed_full[:, suffix_start:total]
    keys, gates, valid_channel = mx.split(
        suffix, [indexer.head_dim, 2 * indexer.head_dim], axis=-1
    )
    pooled = indexer._pooled_states(keys, gates, valid_channel[..., 0] > 0)
    pool_keys, pool_indices, pool_valid = pooled
    pool_indices = mx.where(
        pool_indices >= 0, pool_indices + suffix_start, -1
    )
    traffic = _zero_traffic()
    traffic["read_bytes"] = int(suffix.nbytes)
    traffic["write_bytes"] = int(
        pool_keys.nbytes + pool_indices.nbytes + pool_valid.nbytes
    )
    return {
        "keys": pool_keys,
        "indices": pool_indices,
        "valid": pool_valid,
        "stable": stable,
        "suffix_start": suffix_start,
        "total": total,
        "updated_rows": int(pool_keys.shape[1]),
    }, traffic


def _carry_prefix(session, suffix):
    state = session.pool
    traffic = _zero_traffic()
    stable = suffix["stable"]
    new_total = suffix["total"]
    if session.arm == "reference_concat":
        prefix_keys = state.keys[:, :stable]
        prefix_indices = state.indices[:, :stable]
        prefix_valid = state.valid[:, :stable]
        keys = mx.concatenate([prefix_keys, suffix["keys"]], axis=1)
        indices = mx.concatenate([prefix_indices, suffix["indices"]], axis=1)
        valid = mx.concatenate([prefix_valid, suffix["valid"]], axis=1)
        component_bytes = {
            "keys": int(prefix_keys.nbytes),
            "indices": int(prefix_indices.nbytes),
            "valid": int(prefix_valid.nbytes),
        }
        traffic["pool_copy_bytes"] = component_bytes
        copied = sum(component_bytes.values())
        traffic["copy_bytes"] = copied
        traffic["read_bytes"] = copied
        traffic["write_bytes"] = copied
        session.pool = _ContiguousPool(
            keys, indices, valid, new_total, int(keys.shape[1]), int(keys.shape[1])
        )
        sync = (keys, indices, valid)
    elif session.arm == "preallocated_pool_row":
        state.key_buffer[:, stable : stable + 1] = suffix["keys"]
        state.index_buffer[:, stable : stable + 1] = suffix["indices"]
        state.valid_buffer[:, stable : stable + 1] = suffix["valid"]
        logical = stable + suffix["updated_rows"]
        state.keys = state.key_buffer[:, :logical]
        state.indices = state.index_buffer[:, :logical]
        state.valid = state.valid_buffer[:, :logical]
        state.logical_count = logical
        state.total_tokens = new_total
        row_component_bytes = {
            "keys": int(suffix["keys"].nbytes),
            "indices": int(suffix["indices"].nbytes),
            "valid": int(suffix["valid"].nbytes),
        }
        buffer_component_bytes = {
            "keys": int(state.key_buffer.nbytes),
            "indices": int(state.index_buffer.nbytes),
            "valid": int(state.valid_buffer.nbytes),
        }
        # As above, these Python slice assignments are functional scatters.
        # Preallocation removes capacity growth but does not prove physical
        # row-only mutation; the unchanged buffer portion is copy traffic.
        component_copies = {
            component: buffer_component_bytes[component]
            - row_component_bytes[component]
            for component in POOL_COMPONENTS
        }
        traffic["pool_copy_bytes"] = component_copies
        traffic["copy_bytes"] = sum(component_copies.values())
        traffic["read_bytes"] = sum(buffer_component_bytes.values())
        traffic["write_bytes"] = sum(buffer_component_bytes.values())
        sync = (
            state.keys[:, stable : stable + 1],
            state.indices[:, stable : stable + 1],
            state.valid[:, stable : stable + 1],
        )
    else:
        row = (suffix["keys"], suffix["indices"], suffix["valid"])
        if new_total % 4 == 0:
            state.completed.append(row)
            state.tail = None
        else:
            state.tail = row
        state.total_tokens = new_total
        row_bytes = int(sum(value.nbytes for value in row))
        traffic["read_bytes"] = row_bytes
        traffic["write_bytes"] = row_bytes
        sync = row
    return session.pool, traffic, sync


def _publish_pool(session):
    state = session.pool
    if isinstance(state, _ContiguousPool):
        session.indexer_cache._pool = (
            state.keys, state.indices, state.valid, state.total_tokens
        )
    else:
        session.indexer_cache._pool = None
    # Publishing Python-side state has no Metal consumer of its own.  In
    # particular, do not force the segmented immutable prefix here: the score
    # phase below is the first consumer and times that dependency explicitly.
    return (), _zero_traffic()


def _score_segments(session, x, qr):
    indexer = session.capture.delegate
    query = indexer.wq_b(qr).reshape(
        1, 1, indexer.n_heads, indexer.head_dim
    )
    weights = indexer.weights_proj(x) * (indexer.n_heads**-0.5)
    score_parts = []
    valid_parts = []
    total = session.pool.total_tokens
    for keys, indices, valid in session.pool.segments():
        scores = query @ keys[:, None].swapaxes(-1, -2)
        scores = mx.maximum(scores * indexer.softmax_scale, 0.0)
        index_scores = mx.sum(weights[..., None] * scores, axis=2)
        pool_end = mx.clip(indices[..., -1], 0, total - 1)
        visible = pool_end[:, None, :] < total
        valid_candidates = visible & valid[:, None]
        score_parts.append(mx.where(valid_candidates, index_scores, -1e30))
        valid_parts.append(valid_candidates)
    return {
        "scores": mx.concatenate(score_parts, axis=-1),
        "valid": mx.concatenate(valid_parts, axis=-1),
    }


def _select_pools(session, scored):
    indexer = session.capture.delegate
    select_k = min(
        indexer.index_topk // indexer.index_kpool,
        session.pool.logical_count,
    )
    order = mx.argsort(-scored["scores"], axis=-1)
    selected = order[..., :select_k]
    selected_valid = mx.take_along_axis(
        scored["valid"], selected, axis=-1
    )
    return {"selected": selected, "valid": selected_valid, "k": select_k}


def _expand_selection(session, selected):
    indexer = session.capture.delegate
    total = session.pool.total_tokens
    pool_indices = session.pool.gather_indices(selected["selected"])
    topk = pool_indices.reshape(1, 1, selected["k"] * indexer.index_kpool)
    selected_valid = mx.broadcast_to(
        selected["valid"][..., None],
        (1, 1, selected["k"], indexer.index_kpool),
    ).reshape(1, 1, selected["k"] * indexer.index_kpool)
    topk = mx.where(selected_valid, topk, -1)
    remainder = total % indexer.index_kpool
    offsets = mx.arange(indexer.index_kpool - 1)
    tail_start = total - remainder
    tail = mx.where(offsets < remainder, tail_start + offsets, -1)
    topk = mx.concatenate([topk, tail.reshape(1, 1, -1)], axis=-1)
    width = indexer.index_topk + indexer.index_kpool - 1
    if topk.shape[-1] < width:
        topk = mx.concatenate(
            [topk, mx.full((1, 1, width - topk.shape[-1]), -1)], axis=-1
        )
    topk = topk[..., :width]
    return sanitize_indexpool_indices(topk[:, None].astype(mx.int32), total)


def _append_latent(session, x):
    attention = session.attention
    current = attention.kv_a_layernorm(
        attention.kv_a_proj_with_mqa(x)
    )[:, None]
    latent, _ = session.latent_cache.update_and_fetch(current, current)
    return latent


def _attention_output(session, qr, gathered):
    attention = session.attention
    q = attention.q_b_proj(qr).reshape(
        1, 1, attention.num_heads, attention.q_head_dim
    ).transpose(0, 2, 1, 3)
    return _attention_phase(attention, q, gathered[0], gathered[1])


def _update_graph(session, x, qr):
    projection, projection_traffic = _current_projection(session, x)
    packed, append_traffic = _append_token(session, projection)
    suffix, partial_traffic = _recompute_partial(session, packed)
    _, carry_traffic, _ = _carry_prefix(session, suffix)
    _, publish_traffic = _publish_pool(session)
    traffic = {
        "current_token_projection": projection_traffic,
        "indexer_token_append": append_traffic,
        "partial_pool_recomputation": partial_traffic,
        "complete_pool_prefix_carry": carry_traffic,
        "pool_state_publication": publish_traffic,
    }
    return traffic


def _layer_result(session, output, indices, kv_len: int) -> dict:
    values = _np(output.astype(mx.float32), dtype=np.float32)
    return {
        "layer": session.layer_id,
        "index_hash": _hash_indices(indices),
        "output_hash": hashlib.sha256(values.tobytes()).hexdigest(),
        "nan_count": int(np.count_nonzero(np.isnan(values))),
        **_index_stats(indices, kv_len),
    }


def _actual_step(sessions, context: int, step: int) -> dict:
    outputs = []
    indices = []
    traffic_rows = []
    started = time.perf_counter()
    for session in sessions:
        x = _query(session.layer_id, step, session.attention.hidden_size)
        qr = session.attention.q_a_layernorm(session.attention.q_a_proj(x))
        latent = _append_latent(session, x)
        traffic_rows.append(_update_graph(session, x, qr))
        scored = _score_segments(session, x, qr)
        selected = _select_pools(session, scored)
        topk = _expand_selection(session, selected)
        gathered = _gather_phase(latent, topk)
        outputs.append(_attention_output(session, qr, gathered))
        indices.append(topk)
    mx.eval(*outputs, *indices)
    mx.synchronize()
    elapsed = (time.perf_counter() - started) * 1000.0
    kv_len = context + step - 1
    layers = [
        _layer_result(session, output, selected, kv_len)
        for session, output, selected in zip(
            sessions, outputs, indices, strict=True
        )
    ]
    traffic = {
        phase: _sum_traffic([row[phase] for row in traffic_rows])
        for phase in UPDATE_PHASES
    }
    return {
        "step": step,
        "context_tokens": kv_len,
        "context_mod_index_kpool": kv_len % 4,
        "aggregate_latency_ms": elapsed,
        "active_memory_bytes": int(mx.get_active_memory()),
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "logical_pool_count": sessions[0].pool.logical_count,
        "pool_capacity": sessions[0].pool.capacity,
        "updated_pool_rows": 1,
        "traffic": traffic,
        "combined_index_hash": _combined_hash(
            [layer["index_hash"] for layer in layers]
        ),
        "combined_output_hash": _combined_hash(
            [layer["output_hash"] for layer in layers]
        ),
        "layers": layers,
    }


def _phased_step(sessions, context: int, step: int) -> dict:
    phase_ms = {phase: 0.0 for phase in ALL_PHASES}
    traffic_rows = []
    state_rows = []
    for session in sessions:
        x = _query(session.layer_id, step, session.attention.hidden_size)
        qr = session.attention.q_a_layernorm(session.attention.q_a_proj(x))
        latent = _append_latent(session, x)
        state_rows.append({"session": session, "x": x, "qr": qr, "latent": latent})
    _eval([(row["qr"], row["latent"][..., -1:, :]) for row in state_rows])

    projections, phase_ms["current_token_projection"] = _time_phase(
        lambda: [_current_projection(row["session"], row["x"]) for row in state_rows],
        lambda rows: [row[0] for row in rows],
    )
    appended, phase_ms["indexer_token_append"] = _time_phase(
        lambda: [
            _append_token(row["session"], projection[0])
            for row, projection in zip(state_rows, projections, strict=True)
        ],
        lambda rows: [row[0][:, -1:, :] for row in rows],
    )
    partials, phase_ms["partial_pool_recomputation"] = _time_phase(
        lambda: [
            _recompute_partial(row["session"], appended_row[0])
            for row, appended_row in zip(state_rows, appended, strict=True)
        ],
        lambda rows: [
            (row[0]["keys"], row[0]["indices"], row[0]["valid"])
            for row in rows
        ],
    )
    carried, phase_ms["complete_pool_prefix_carry"] = _time_phase(
        lambda: [
            _carry_prefix(row["session"], partial[0])
            for row, partial in zip(state_rows, partials, strict=True)
        ],
        lambda rows: [row[2] for row in rows],
    )
    published, phase_ms["pool_state_publication"] = _time_phase(
        lambda: [_publish_pool(row["session"]) for row in state_rows],
        lambda rows: [row[0] for row in rows],
    )
    traffic_rows = [
        {
            "current_token_projection": projection[1],
            "indexer_token_append": append[1],
            "partial_pool_recomputation": partial[1],
            "complete_pool_prefix_carry": carry[1],
            "pool_state_publication": publication[1],
        }
        for projection, append, partial, carry, publication in zip(
            projections, appended, partials, carried, published, strict=True
        )
    ]

    scored, phase_ms["pool_score"] = _time_phase(
        lambda: [
            _score_segments(row["session"], row["x"], row["qr"])
            for row in state_rows
        ]
    )
    selected, phase_ms["selection"] = _time_phase(
        lambda: [
            _select_pools(row["session"], score)
            for row, score in zip(state_rows, scored, strict=True)
        ]
    )
    topk, phase_ms["pool_expansion"] = _time_phase(
        lambda: [
            _expand_selection(row["session"], selected_row)
            for row, selected_row in zip(state_rows, selected, strict=True)
        ]
    )
    gathered, phase_ms["latent_gather"] = _time_phase(
        lambda: [
            _gather_phase(row["latent"], indices)
            for row, indices in zip(state_rows, topk, strict=True)
        ]
    )
    outputs, phase_ms["selected_attention"] = _time_phase(
        lambda: [
            _attention_output(row["session"], row["qr"], gathered_row)
            for row, gathered_row in zip(state_rows, gathered, strict=True)
        ]
    )
    kv_len = context + step - 1
    layers = [
        _layer_result(row["session"], output, indices, kv_len)
        for row, output, indices in zip(state_rows, outputs, topk, strict=True)
    ]
    return {
        "step": step,
        "context_tokens": kv_len,
        "context_mod_index_kpool": kv_len % 4,
        "phase_ms": phase_ms,
        "update_phase_sum_ms": sum(phase_ms[phase] for phase in UPDATE_PHASES),
        "all_phase_sum_ms": sum(phase_ms.values()),
        "logical_pool_count": sessions[0].pool.logical_count,
        "pool_capacity": sessions[0].pool.capacity,
        "updated_pool_rows": 1,
        "traffic": {
            phase: _sum_traffic([row[phase] for row in traffic_rows])
            for phase in UPDATE_PHASES
        },
        "combined_index_hash": _combined_hash(
            [layer["index_hash"] for layer in layers]
        ),
        "combined_output_hash": _combined_hash(
            [layer["output_hash"] for layer in layers]
        ),
        "layers": layers,
    }


def _release(sessions) -> None:
    sessions.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def _warm(attentions, context: int, arm: str, steps: int) -> None:
    sessions = _build_sessions(attentions, context, arm)
    for step in range(1, steps + 1):
        _actual_step(sessions, context, step)
    _release(sessions)


def _summarize_actual(steps, baseline_active: int) -> dict:
    steady = [row["aggregate_latency_ms"] for row in steps]
    return {
        "median_ms": statistics.median(steady),
        "p95_ms": _percentile(steady, 95),
        "active_memory_before_bytes": baseline_active,
        "active_memory_after_bytes": steps[-1]["active_memory_bytes"],
        "peak_memory_bytes": max(row["peak_memory_bytes"] for row in steps),
        "working_peak_bytes": max(
            0, max(row["peak_memory_bytes"] for row in steps) - baseline_active
        ),
        "memory_drift_bytes": steps[-1]["active_memory_bytes"] - baseline_active,
        "steps": steps,
    }


def _summarize_phases(steps) -> dict:
    medians = {
        phase: statistics.median(row["phase_ms"][phase] for row in steps)
        for phase in ALL_PHASES
    }
    p95 = {
        phase: _percentile([row["phase_ms"][phase] for row in steps], 95)
        for phase in ALL_PHASES
    }
    traffic = {
        phase: {
            key: statistics.median(row["traffic"][phase][key] for row in steps)
            for key in ("read_bytes", "write_bytes", "copy_bytes")
        }
        for phase in UPDATE_PHASES
    }
    for phase in UPDATE_PHASES:
        traffic[phase]["pool_copy_bytes"] = {
            component: statistics.median(
                row["traffic"][phase]["pool_copy_bytes"][component]
                for row in steps
            )
            for component in POOL_COMPONENTS
        }
    return {
        "phase_median_ms": medians,
        "phase_p95_ms": p95,
        "update_total_median_ms": sum(medians[phase] for phase in UPDATE_PHASES),
        "traffic_median_bytes": traffic,
        "steps": steps,
    }


def _measure_arm(attentions, context: int, arm: str, warmup_steps: int):
    _progress("warm", context=context, arm=arm, steps=warmup_steps)
    _warm(attentions, context, arm, warmup_steps)
    actual_sessions = _build_sessions(attentions, context, arm)
    baseline = int(mx.get_active_memory())
    mx.reset_peak_memory()
    actual_steps = []
    for step in range(1, MEASURED_STEPS + 1):
        row = _actual_step(actual_sessions, context, step)
        actual_steps.append(row)
        _progress(
            "actual_step",
            context=context,
            arm=arm,
            step=step,
            milliseconds=row["aggregate_latency_ms"],
        )
    actual = _summarize_actual(actual_steps, baseline)
    _release(actual_sessions)

    phase_sessions = _build_sessions(attentions, context, arm)
    phase_steps = [
        _phased_step(phase_sessions, context, step)
        for step in range(1, MEASURED_STEPS + 1)
    ]
    phases = _summarize_phases(phase_steps)
    _release(phase_sessions)
    parity = [
        {
            "step": left["step"],
            "index_hash_match": left["combined_index_hash"]
            == right["combined_index_hash"],
            "output_hash_match": left["combined_output_hash"]
            == right["combined_output_hash"],
        }
        for left, right in zip(actual_steps, phase_steps, strict=True)
    ]
    return {"actual": actual, "phases": phases, "actual_phase_parity": parity}


def _measure_capacity_boundary(attentions, context: int, warmup_steps: int):
    boundary_step = 1 if context == 2049 else 2
    warm = _build_sessions(
        attentions, context, "reference_concat", token_headroom=False
    )
    for step in range(1, boundary_step + 1):
        _actual_step(warm, context, step)
    _release(warm)
    sessions = _build_sessions(
        attentions, context, "reference_concat", token_headroom=False
    )
    rows = [_actual_step(sessions, context, step) for step in range(1, boundary_step + 1)]
    _release(sessions)
    target = rows[-1]
    return {
        "step": boundary_step,
        "context_tokens": target["context_tokens"],
        "aggregate_latency_ms": target["aggregate_latency_ms"],
        "indexer_token_copy_bytes": target["traffic"]["indexer_token_append"][
            "copy_bytes"
        ],
        "excluded_from_steady_statistics": True,
    }


def _context_case(attentions, context: int, warmup_steps: int):
    arms = {
        arm: _measure_arm(attentions, context, arm, warmup_steps) for arm in ARMS
    }
    baseline = arms["reference_concat"]["actual"]["steps"]
    cross = {
        arm: [
            {
                "step": expected["step"],
                "index_hash_match": expected["combined_index_hash"]
                == actual["combined_index_hash"],
                "output_hash_match": expected["combined_output_hash"]
                == actual["combined_output_hash"],
            }
            for expected, actual in zip(
                baseline, row["actual"]["steps"], strict=True
            )
        ]
        for arm, row in arms.items()
    }
    return {
        "context_tokens": context,
        "arms": arms,
        "cross_arm_parity": cross,
        "capacity_boundary_reference": _measure_capacity_boundary(
            attentions, context, warmup_steps
        ),
    }


def _decision(cases):
    output = {}
    for arm in ARMS:
        update_base = cases["2049"]["arms"][arm]["phases"][
            "update_total_median_ms"
        ]
        update_large = cases["262144"]["arms"][arm]["phases"][
            "update_total_median_ms"
        ]
        aggregate_base = cases["2049"]["arms"][arm]["actual"]["median_ms"]
        aggregate_large = cases["262144"]["arms"][arm]["actual"]["median_ms"]
        parity_rows = [
            row
            for case in cases.values()
            for row in case["cross_arm_parity"][arm]
        ]
        phase_steps = [
            step
            for case in cases.values()
            for step in case["arms"][arm]["phases"]["steps"]
        ]
        byte_identical = all(
            row["index_hash_match"] and row["output_hash_match"]
            for row in parity_rows
        )
        no_prefix_copy = all(
            step["traffic"]["complete_pool_prefix_carry"]["copy_bytes"] == 0
            for step in phase_steps
        )
        output[arm] = {
            "pool_update_retention": update_base / update_large,
            "all_dsa_aggregate_retention": aggregate_base / aggregate_large,
            "byte_identical_to_reference": byte_identical,
            "complete_pool_prefix_copy_free": no_prefix_copy,
        }
    qualifies = {
        arm: row["byte_identical_to_reference"]
        and row["complete_pool_prefix_copy_free"]
        and row["pool_update_retention"] >= 0.8
        and row["all_dsa_aggregate_retention"] >= 0.8
        for arm, row in output.items()
    }
    return {
        "retentions": output,
        "retention_target": 0.8,
        "preallocated_pool_row_qualified": qualifies["preallocated_pool_row"],
        "segmented_pool_qualified": qualifies["segmented_pool"],
        "next_measurement": "probe copy-free packed Indexer token append and exact segmented scoring",
        "runtime_apc_abi_changed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--measured-steps", type=int, default=MEASURED_STEPS)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup_steps < 4:
        raise ValueError("warmup must cover all pool-tail shapes")
    if args.measured_steps != MEASURED_STEPS:
        raise ValueError(f"measured steps must remain {MEASURED_STEPS}")

    report = inspect_checkpoint(args.model, require_server_ready=True)
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    model, _ = load(args.model)
    attentions = {
        layer_id: model.language_model.model.layers[layer_id].self_attn
        for layer_id in EXPECTED_DSA
    }
    parameters = [
        value
        for attention in attentions.values()
        for _, value in tree_flatten(attention.parameters())
    ]
    mx.eval(*parameters)
    mx.synchronize()
    for attention in attentions.values():
        attention.indexer = _CapturingIndexer(attention.indexer)
    del model, parameters
    gc.collect()
    mx.clear_cache()
    mx.synchronize()

    cases = {
        str(context): _context_case(attentions, context, args.warmup_steps)
        for context in CONTEXTS
    }
    parity = [
        row
        for case in cases.values()
        for arm in case["cross_arm_parity"].values()
        for row in arm
    ]
    phase_parity = [
        row
        for case in cases.values()
        for arm in case["arms"].values()
        for row in arm["actual_phase_parity"]
    ]
    layer_rows = [
        layer
        for case in cases.values()
        for arm in case["arms"].values()
        for step in arm["actual"]["steps"]
        for layer in step["layers"]
    ]
    decision = _decision(cases)
    acceptance = {
        "all_11_layers_3_contexts_3_arms_16_steps_measured": all(
            len(case["arms"][arm]["actual"]["steps"]) == MEASURED_STEPS
            for case in cases.values()
            for arm in ARMS
        ),
        "reference_and_preallocated_match_all_hashes": all(
            row["index_hash_match"] and row["output_hash_match"]
            for case in cases.values()
            for arm in ("reference_concat", "preallocated_pool_row")
            for row in case["cross_arm_parity"][arm]
        ),
        "segmented_hash_parity_recorded": all(
            "index_hash_match" in row and "output_hash_match" in row
            for case in cases.values()
            for row in case["cross_arm_parity"]["segmented_pool"]
        ),
        "phase_sessions_match_actual_hashes": all(
            row["index_hash_match"] and row["output_hash_match"]
            for row in phase_parity
        ),
        "all_indices_sentinel_or_in_range": all(
            layer["non_sentinel_out_of_range"] == 0 for layer in layer_rows
        ),
        "all_unused_slots_are_minus1": all(
            layer["sentinel_count"]
            == _expected_tail_sentinels(step["context_tokens"], report.index_kpool)
            for case in cases.values()
            for arm in case["arms"].values()
            for step in arm["actual"]["steps"]
            for layer in step["layers"]
        ),
        "no_nan": all(layer["nan_count"] == 0 for layer in layer_rows),
        "segmented_arm_never_copies_complete_pool_prefix": all(
            step["traffic"]["complete_pool_prefix_carry"]["copy_bytes"] == 0
            for case in cases.values()
            for step in case["arms"]["segmented_pool"]["phases"]["steps"]
        ),
        "capacity_boundary_excluded_from_steady": all(
            case["capacity_boundary_reference"][
                "excluded_from_steady_statistics"
            ]
            for case in cases.values()
        ),
        "runtime_server_apc_admission_unchanged": LEGACY_PROBE_MAX_PROMPT_TOKENS == 256,
    }
    acceptance["accepted"] = all(acceptance.values())
    output = {
        "schema": "glm53-incremental-indexpool-update-copies-v1",
        "date": date.today().isoformat(),
        "machine": "Apple M3 Ultra, 80-core GPU, 512 GB unified memory",
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "nope_cache_abi": NOPE_DSA_CACHE_ABI,
        "dsa_layers": list(EXPECTED_DSA),
        "contexts": list(CONTEXTS),
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "arms": list(ARMS),
        "measurement_contract": {
            "steady_latent_capacity_reserved": "history + measured generation rounded to 256",
            "steady_indexer_token_capacity_reserved": "history + measured generation rounded to 256",
            "capacity_extension_measured_in_separate_reference_arm": True,
            "pool_indices_dtype_unchanged": "int64",
            "phase_timings_synchronize_consumer_boundaries": True,
            "copy_bytes_contract": "functional-buffer prefix carried into the new MLX value; zero means no complete-prefix materialization",
        },
        "cases": cases,
        "decision_gate": decision,
        "runtime_policy": {
            "probe_only": True,
            "default_backend": "direct",
            "prompt_limit": LEGACY_PROBE_MAX_PROMPT_TOKENS,
            "server_changed": False,
            "apc_abi_changed": False,
            "admission_changed": False,
        },
        "acceptance": acceptance,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "accepted": acceptance["accepted"],
                "decision_gate": decision,
            },
            indent=2,
        )
    )
    return 0 if acceptance["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
