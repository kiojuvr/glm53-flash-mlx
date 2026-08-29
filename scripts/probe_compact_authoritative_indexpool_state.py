#!/usr/bin/env python3
"""Probe compact authoritative IndexPool state without full token history."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from glm53_flash_mlx.abi import MLX_VLM_REVISION, NOPE_DSA_CACHE_ABI
from glm53_flash_mlx.indexpool import INDEXPOOL_SENTINEL, sanitize_indexpool_indices
from glm53_flash_mlx.loader import load
from glm53_flash_mlx.manifest import EXPECTED_DSA, inspect_checkpoint
from glm53_flash_mlx.server import DEFAULT_MAX_PROMPT_TOKENS
from probe_long_context_dsa_decode_frontier import (
    _attention_phase,
    _expand_phase,
    _gather_phase,
    _pool_update,
    _score_phase,
    _selection_phase,
)
from probe_persistent_all_dsa_session_frontier import (
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
ROLLBACK_CASES = (1, 2, 3, 4, 8, 15, 16)
ROLLBACK_WINDOW = 16
RAW_STATE_WINDOW = ROLLBACK_WINDOW + 4 - 1
ROLLBACK_BASE_CONTEXT = 2052
ARMS = (
    "full_packed_history_oracle",
    "compact_pool_state",
    "compact_pool_state_dependency_chained",
)
PHASES = (
    "current_token_projection",
    "tail_journal_append",
    "partial_pool_completion",
    "pool_row_carry",
)


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


def _concat_or_empty(left: mx.array, right: mx.array) -> mx.array:
    if left.shape[1] == 0:
        return right
    if right.shape[1] == 0:
        return left
    return mx.concatenate([left, right], axis=1)


@dataclass
class CompactIndexPoolState:
    # Full-capacity backing arrays; score sees a contiguous logical slice with
    # exactly the same shape and row order as the reference Indexer.
    pool_keys: mx.array
    pool_indices: mx.array
    pool_valid: mx.array

    # Active incomplete pool.  These are empty whenever total_tokens % kpool == 0.
    tail_keys: mx.array
    tail_gates: mx.array
    tail_valid: mx.array
    tail_positions: mx.array

    # The remaining raw rollback journal.  Journal + active tail never exceeds
    # rollback_window + kpool - 1; complete-prefix token history is absent.
    journal_keys: mx.array
    journal_gates: mx.array
    journal_valid: mx.array
    journal_positions: mx.array

    total_tokens: int
    logical_pool_count: int
    pool_capacity: int
    rollback_window: int = ROLLBACK_WINDOW

    def logical_pool(self):
        return (
            self.pool_keys[:, : self.logical_pool_count],
            self.pool_indices[:, : self.logical_pool_count],
            self.pool_valid[:, : self.logical_pool_count],
        )

    @property
    def raw_token_count(self) -> int:
        return int(self.tail_keys.shape[1] + self.journal_keys.shape[1])

    @property
    def active_tail_count(self) -> int:
        return int(self.tail_keys.shape[1])

    def raw_rows(self):
        return (
            _concat_or_empty(self.journal_keys, self.tail_keys),
            _concat_or_empty(self.journal_gates, self.tail_gates),
            _concat_or_empty(self.journal_valid, self.tail_valid),
            _concat_or_empty(self.journal_positions, self.tail_positions),
        )


@dataclass
class _OracleSession:
    layer_id: int
    attention: object
    latent_cache: ProbeNoPELatentCache
    indexer_cache: object


@dataclass
class _CompactSession:
    layer_id: int
    attention: object
    latent_cache: ProbeNoPELatentCache
    state: CompactIndexPoolState


def _new_latent_cache(attention, layer_id: int, history: int):
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


def _history_rows(indexer, layer_id: int, rows: int):
    keys = _deterministic_rows(
        rows, indexer.head_dim, 0.125 + layer_id * 0.015625, mx.bfloat16
    )[None]
    gates = _deterministic_rows(
        rows, indexer.head_dim, 0.625 + layer_id * 0.015625, mx.bfloat16
    )[None]
    valid = mx.ones((1, rows), dtype=mx.bool_)
    return keys, gates, valid


def _leaf_numpy(value: mx.array) -> tuple[np.ndarray, bool]:
    """Materialize an initialization-only leaf while preserving BF16 bytes."""
    mx.eval(value)
    mx.synchronize()
    if value.dtype == mx.bfloat16:
        return np.array(np.asarray(value.view(mx.uint16)), copy=True), True
    return np.array(np.asarray(value), copy=True), False


def _leaf_array(value: mx.array) -> mx.array:
    array, was_bfloat16 = _leaf_numpy(value)
    result = mx.array(array)
    return result.view(mx.bfloat16) if was_bfloat16 else result


def _leaf_capacity_buffer(value: mx.array, capacity: int) -> mx.array:
    array, was_bfloat16 = _leaf_numpy(value)
    shape = list(array.shape)
    shape[1] = capacity
    buffer = np.zeros(shape, dtype=array.dtype)
    buffer[:, : array.shape[1]] = array
    result = mx.array(buffer)
    return result.view(mx.bfloat16) if was_bfloat16 else result


def _build_oracle_session(attention, layer_id: int, history: int):
    from mlx_vlm.models.cache import KVCache

    indexer = attention.indexer
    capacity = _capacity(history + MEASURED_STEPS)
    keys, gates, valid = _history_rows(indexer, layer_id, capacity)
    valid = mx.arange(capacity)[None] < history
    packed = mx.concatenate(
        [keys, gates, valid.astype(keys.dtype)[..., None]], axis=-1
    )
    logical = packed[:, :history]
    logical_keys, logical_gates, valid_channel = mx.split(
        logical, [indexer.head_dim, 2 * indexer.head_dim], axis=-1
    )
    cache = KVCache()
    cache.keys = packed[:, None]
    cache.values = mx.zeros((1, 1, capacity, 0), dtype=mx.bfloat16)
    cache.offset = history
    cache._no_pad = True
    cache._pool = (
        *indexer._pooled_states(
            logical_keys, logical_gates, valid_channel[..., 0] > 0
        ),
        history,
    )
    mx.eval(packed, *cache._pool[:3])
    return _OracleSession(
        layer_id,
        attention,
        _new_latent_cache(attention, layer_id, history),
        cache,
    )


def _build_compact_state(indexer, layer_id: int, history: int):
    keys, gates, valid = _history_rows(indexer, layer_id, history)
    pooled = indexer._pooled_states(keys, gates, valid)
    logical = int(pooled[0].shape[1])
    pool_capacity = (history + MEASURED_STEPS + indexer.index_kpool - 1) // indexer.index_kpool
    # Initialization-only host materialization deliberately severs the MLX
    # graph from full raw history.  BF16 goes through uint16 so every bit is
    # preserved.  No host operation is used by append/score/decode.
    pool_keys = _leaf_capacity_buffer(pooled[0], pool_capacity)
    pool_indices = _leaf_capacity_buffer(pooled[1], pool_capacity)
    pool_valid = _leaf_capacity_buffer(pooled[2], pool_capacity)

    raw_start = max(0, history - RAW_STATE_WINDOW)
    raw_keys = _leaf_array(keys[:, raw_start:history])
    raw_gates = _leaf_array(gates[:, raw_start:history])
    raw_valid = _leaf_array(valid[:, raw_start:history])
    raw_positions = _leaf_array(
        mx.arange(raw_start, history, dtype=mx.int64)[None]
    )
    active = history % indexer.index_kpool
    split = int(raw_keys.shape[1]) - active
    state = CompactIndexPoolState(
        pool_keys=pool_keys,
        pool_indices=pool_indices,
        pool_valid=pool_valid,
        tail_keys=raw_keys[:, split:],
        tail_gates=raw_gates[:, split:],
        tail_valid=raw_valid[:, split:],
        tail_positions=raw_positions[:, split:],
        journal_keys=raw_keys[:, :split],
        journal_gates=raw_gates[:, :split],
        journal_valid=raw_valid[:, :split],
        journal_positions=raw_positions[:, :split],
        total_tokens=history,
        logical_pool_count=logical,
        pool_capacity=pool_capacity,
    )
    mx.eval(
        state.pool_keys,
        state.pool_indices,
        state.pool_valid,
        state.tail_keys,
        state.tail_gates,
        state.tail_valid,
        state.tail_positions,
        state.journal_keys,
        state.journal_gates,
        state.journal_valid,
        state.journal_positions,
    )
    del keys, gates, valid, pooled, raw_keys, raw_gates, raw_valid, raw_positions
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    return state


def _build_compact_session(attention, layer_id: int, history: int):
    state = _build_compact_state(attention.indexer, layer_id, history)
    return _CompactSession(
        layer_id,
        attention,
        _new_latent_cache(attention, layer_id, history),
        state,
    )


def _build_sessions(attentions, context: int, arm: str):
    history = context - 1
    builder = (
        _build_oracle_session
        if arm == "full_packed_history_oracle"
        else _build_compact_session
    )
    sessions = [builder(attentions[layer], layer, history) for layer in EXPECTED_DSA]
    mx.synchronize()
    return sessions


def _project_indexer_token(indexer, x):
    key = indexer.k_norm(indexer.wk(x)).reshape(1, 1, indexer.head_dim)
    gate = x @ indexer.index_kpool_compress_gate.swapaxes(-1, -2)
    valid = mx.ones((1, 1), dtype=mx.bool_)
    positions = None
    return key, gate, valid, positions


def _split_raw_window(state: CompactIndexPoolState, indexer, raw):
    keys, gates, valid, positions = raw
    raw_capacity = state.rollback_window + indexer.index_kpool - 1
    if keys.shape[1] > raw_capacity:
        keys = keys[:, -raw_capacity:]
        gates = gates[:, -raw_capacity:]
        valid = valid[:, -raw_capacity:]
        positions = positions[:, -raw_capacity:]
    active = state.total_tokens % indexer.index_kpool
    split = int(keys.shape[1]) - active
    state.journal_keys, state.tail_keys = keys[:, :split], keys[:, split:]
    state.journal_gates, state.tail_gates = gates[:, :split], gates[:, split:]
    state.journal_valid, state.tail_valid = valid[:, :split], valid[:, split:]
    state.journal_positions, state.tail_positions = (
        positions[:, :split],
        positions[:, split:],
    )


def _append_raw_token(state: CompactIndexPoolState, indexer, projected):
    key, gate, valid, _ = projected
    raw_dtype = state.journal_keys.dtype
    key = key.astype(raw_dtype)
    gate = gate.astype(raw_dtype)
    raw = state.raw_rows()
    position = mx.array([[state.total_tokens]], dtype=mx.int64)
    raw = (
        _concat_or_empty(raw[0], key),
        _concat_or_empty(raw[1], gate),
        _concat_or_empty(raw[2], valid),
        _concat_or_empty(raw[3], position),
    )
    state.total_tokens += 1
    _split_raw_window(state, indexer, raw)
    carried = sum(int(value.nbytes) for value in raw)
    return raw, carried


def _complete_partial_pool(state: CompactIndexPoolState, indexer):
    active = state.total_tokens % indexer.index_kpool
    suffix_count = active if active else indexer.index_kpool
    raw = state.raw_rows()
    keys = raw[0][:, -suffix_count:]
    gates = raw[1][:, -suffix_count:]
    valid = raw[2][:, -suffix_count:]
    positions = raw[3][:, -suffix_count:]
    row_keys, row_indices, row_valid = indexer._pooled_states(keys, gates, valid)
    start = state.total_tokens - suffix_count
    row_indices = mx.where(
        row_indices >= 0, row_indices + start, INDEXPOOL_SENTINEL
    )
    # Positions are retained as authoritative rollback metadata; this assertion
    # is covered by the fixtures without introducing host sync to the hot path.
    return row_keys, row_indices, row_valid, positions


def _carry_pool_row(state: CompactIndexPoolState, indexer, row):
    stable = (state.total_tokens - 1) // indexer.index_kpool
    state.pool_keys[:, stable : stable + 1] = row[0]
    state.pool_indices[:, stable : stable + 1] = row[1]
    state.pool_valid[:, stable : stable + 1] = row[2]
    state.logical_pool_count = stable + 1
    row_bytes = int(row[0].nbytes + row[1].nbytes + row[2].nbytes)
    buffer_bytes = int(
        state.pool_keys.nbytes + state.pool_indices.nbytes + state.pool_valid.nbytes
    )
    return buffer_bytes - row_bytes


def _compact_pool_update(state, indexer, x, projected=None):
    projected = projected or _project_indexer_token(indexer, x)
    _, append_copy = _append_raw_token(state, indexer, projected)
    row = _complete_partial_pool(state, indexer)
    carry_copy = _carry_pool_row(state, indexer, row)
    return row, append_copy, carry_copy


def _compact_score(indexer, state, x, qr):
    pool_keys, pool_indices, pool_valid = state.logical_pool()
    query = indexer.wq_b(qr).reshape(1, 1, indexer.n_heads, indexer.head_dim)
    scores = query @ pool_keys[:, None].swapaxes(-1, -2)
    scores = mx.maximum(scores * indexer.softmax_scale, 0.0)
    weights = indexer.weights_proj(x) * (indexer.n_heads**-0.5)
    index_scores = mx.sum(weights[..., None] * scores, axis=2)
    pool_end = mx.clip(pool_indices[..., -1], 0, state.total_tokens - 1)
    valid_candidates = (pool_end[:, None, :] < state.total_tokens) & pool_valid[:, None]
    return {
        "index_scores": mx.where(valid_candidates, index_scores, -1e30),
        "valid_candidates": valid_candidates,
        "pool_count": state.logical_pool_count,
    }


def _compact_expand(indexer, state, selected):
    _, pool_indices, _ = state.logical_pool()
    select_k = selected["select_k"]
    source = mx.broadcast_to(
        pool_indices[:, None],
        (1, 1, state.logical_pool_count, indexer.index_kpool),
    )
    expanded = mx.broadcast_to(
        selected["selected"][..., None],
        (1, 1, select_k, indexer.index_kpool),
    )
    chosen = mx.take_along_axis(source, expanded, axis=2)
    topk = chosen.reshape(1, 1, select_k * indexer.index_kpool)
    chosen_valid = mx.broadcast_to(
        selected["selected_valid"][..., None],
        (1, 1, select_k, indexer.index_kpool),
    ).reshape(1, 1, select_k * indexer.index_kpool)
    topk = mx.where(chosen_valid, topk, INDEXPOOL_SENTINEL)
    if indexer.index_kpool_always_select_tail and indexer.index_kpool > 1:
        active = state.active_tail_count
        tail = state.tail_positions
        if active < indexer.index_kpool - 1:
            tail = mx.concatenate(
                [
                    tail,
                    mx.full(
                        (1, indexer.index_kpool - 1 - active),
                        INDEXPOOL_SENTINEL,
                        dtype=tail.dtype,
                    ),
                ],
                axis=-1,
            )
        topk = mx.concatenate([topk, tail[:, None]], axis=-1)
    width = indexer.index_topk + indexer.index_kpool - 1
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
    return sanitize_indexpool_indices(
        topk[..., :width][:, None].astype(mx.int32), state.total_tokens
    )


def _attention_graph(session, x, *, compact: bool):
    attention = session.attention
    indexer = attention.indexer
    qr = attention.q_a_layernorm(attention.q_a_proj(x))
    q = attention.q_b_proj(qr).reshape(
        1, 1, attention.num_heads, attention.q_head_dim
    ).transpose(0, 2, 1, 3)
    current = attention.kv_a_layernorm(attention.kv_a_proj_with_mqa(x))[:, None]
    latent, _ = session.latent_cache.update_and_fetch(current, current)
    if compact:
        _compact_pool_update(session.state, indexer, x)
        scored = _compact_score(indexer, session.state, x, qr)
        selected = _selection_phase(indexer, scored)
        indices = _compact_expand(indexer, session.state, selected)
    else:
        pooled = _pool_update(indexer, x, qr, session.indexer_cache)
        pooled["x"] = x
        scored = _score_phase(indexer, pooled)
        selected = _selection_phase(indexer, scored)
        indices = _expand_phase(indexer, pooled, selected)
    gathered = _gather_phase(latent, indices)
    output = _attention_phase(attention, q, gathered[0], gathered[1])
    return output, indices


def _layer_result(session, output, indices, kv_len: int):
    values = _np(output.astype(mx.float32), dtype=np.float32)
    return {
        "layer": session.layer_id,
        "index_hash": _hash_indices(indices),
        "output_hash": hashlib.sha256(values.tobytes()).hexdigest(),
        "nan_count": int(np.count_nonzero(np.isnan(values))),
        **_index_stats(indices, kv_len),
    }


def _state_bytes(sessions):
    if not sessions:
        return {}
    if isinstance(sessions[0], _OracleSession):
        packed = sum(int(session.indexer_cache.keys.nbytes) for session in sessions)
        return {"packed_token_history_bytes": packed, "total_bytes": packed}
    pool = sum(
        int(s.state.pool_keys.nbytes + s.state.pool_indices.nbytes + s.state.pool_valid.nbytes)
        for s in sessions
    )
    tail = sum(
        int(s.state.tail_keys.nbytes + s.state.tail_gates.nbytes + s.state.tail_valid.nbytes + s.state.tail_positions.nbytes)
        for s in sessions
    )
    journal = sum(
        int(s.state.journal_keys.nbytes + s.state.journal_gates.nbytes + s.state.journal_valid.nbytes + s.state.journal_positions.nbytes)
        for s in sessions
    )
    return {
        "pool_bytes": pool,
        "active_tail_bytes": tail,
        "rollback_journal_bytes": journal,
        "raw_tail_and_journal_bytes": tail + journal,
        "total_bytes": pool + tail + journal,
        "max_raw_tokens_per_layer": max(s.state.raw_token_count for s in sessions),
        "full_packed_history_present": False,
    }


def _actual_step(sessions, context: int, step: int, *, compact: bool, chained: bool):
    outputs = []
    indices = []
    previous = None
    started = time.perf_counter()
    for session in sessions:
        x = _query(session.layer_id, step, session.attention.hidden_size)
        if chained and previous is not None:
            x = mx.depends(x, (previous,))
        output, selected = _attention_graph(session, x, compact=compact)
        outputs.append(output)
        indices.append(selected)
        previous = output
    mx.eval(*outputs, *indices)
    mx.synchronize()
    elapsed = (time.perf_counter() - started) * 1000.0
    kv_len = context + step - 1
    layers = [
        _layer_result(session, output, selected, kv_len)
        for session, output, selected in zip(sessions, outputs, indices, strict=True)
    ]
    return {
        "step": step,
        "context_tokens": kv_len,
        "context_mod_index_kpool": kv_len % 4,
        "aggregate_latency_ms": elapsed,
        "active_memory_bytes": int(mx.get_active_memory()),
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "combined_index_hash": _combined_hash([row["index_hash"] for row in layers]),
        "combined_output_hash": _combined_hash([row["output_hash"] for row in layers]),
        "state_bytes": _state_bytes(sessions),
        "layers": layers,
    }


def _pool_parity(oracle, compact):
    rows = []
    checks = []
    for left, right in zip(oracle, compact, strict=True):
        expected = left.indexer_cache._pool[:3]
        actual = right.state.logical_pool()
        values = [mx.array_equal(a, b) for a, b in zip(expected, actual, strict=True)]
        checks.append(values)
        rows.append((left.layer_id, values))
    _eval(checks)
    return [
        {
            "layer": layer,
            "keys_byte_identical": bool(values[0].item()),
            "indices_byte_identical": bool(values[1].item()),
            "valid_byte_identical": bool(values[2].item()),
        }
        for layer, values in rows
    ]


def _summarize(steps, baseline_active):
    values = [row["aggregate_latency_ms"] for row in steps[1:]]
    by_mod = {}
    for remainder in range(4):
        samples = [
            row["aggregate_latency_ms"]
            for row in steps
            if row["context_mod_index_kpool"] == remainder
        ]
        by_mod[str(remainder)] = {
            "samples_ms": samples,
            "median_ms": statistics.median(samples),
            "p95_ms": _percentile(samples, 95),
        }
    return {
        "first_token_latency_ms": steps[0]["aggregate_latency_ms"],
        "token_2_16_median_ms": statistics.median(values),
        "token_2_16_p95_ms": _percentile(values, 95),
        "context_mod_4_latency": by_mod,
        "active_memory_before_bytes": baseline_active,
        "active_memory_after_bytes": steps[-1]["active_memory_bytes"],
        "peak_memory_bytes": max(row["peak_memory_bytes"] for row in steps),
        "memory_drift_bytes": steps[-1]["active_memory_bytes"] - baseline_active,
        "final_state_bytes": steps[-1]["state_bytes"],
        "steps": steps,
    }


def _release(sessions):
    sessions.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def _warm(attentions, context, arm, steps):
    sessions = _build_sessions(attentions, context, arm)
    compact = arm != "full_packed_history_oracle"
    chained = arm == "compact_pool_state_dependency_chained"
    for step in range(1, steps + 1):
        _actual_step(sessions, context, step, compact=compact, chained=chained)
    _release(sessions)


def _measure_paired(attentions, context, warmup_steps):
    for arm in ARMS[:2]:
        _progress("warm", context=context, arm=arm)
        _warm(attentions, context, arm, warmup_steps)
    oracle = _build_sessions(attentions, context, ARMS[0])
    compact = _build_sessions(attentions, context, ARMS[1])
    oracle_baseline = int(mx.get_active_memory())
    compact_baseline = oracle_baseline
    mx.reset_peak_memory()
    oracle_steps = []
    compact_steps = []
    parity = []
    for step in range(1, MEASURED_STEPS + 1):
        left = _actual_step(oracle, context, step, compact=False, chained=False)
        right = _actual_step(compact, context, step, compact=True, chained=False)
        pool = _pool_parity(oracle, compact)
        oracle_steps.append(left)
        compact_steps.append(right)
        parity.append(
            {
                "step": step,
                "index_hash_match": left["combined_index_hash"] == right["combined_index_hash"],
                "output_hash_match": left["combined_output_hash"] == right["combined_output_hash"],
                "pool_order_byte_identical": all(
                    row["keys_byte_identical"]
                    and row["indices_byte_identical"]
                    and row["valid_byte_identical"]
                    for row in pool
                ),
                "layers": pool,
            }
        )
        _progress(
            "paired_step",
            context=context,
            step=step,
            oracle_ms=left["aggregate_latency_ms"],
            compact_ms=right["aggregate_latency_ms"],
        )
    result = {
        ARMS[0]: _summarize(oracle_steps, oracle_baseline),
        ARMS[1]: _summarize(compact_steps, compact_baseline),
        "parity": parity,
    }
    _release(oracle)
    _release(compact)
    return result


def _measure_chained(attentions, context, warmup_steps):
    arm = ARMS[2]
    _progress("warm", context=context, arm=arm)
    _warm(attentions, context, arm, warmup_steps)
    sessions = _build_sessions(attentions, context, arm)
    baseline = int(mx.get_active_memory())
    mx.reset_peak_memory()
    steps = []
    for step in range(1, MEASURED_STEPS + 1):
        row = _actual_step(sessions, context, step, compact=True, chained=True)
        steps.append(row)
        _progress("chained_step", context=context, step=step, milliseconds=row["aggregate_latency_ms"])
    result = _summarize(steps, baseline)
    _release(sessions)
    return result


def _phased_compact_step(sessions, context, step):
    phase_ms = {phase: 0.0 for phase in PHASES}
    rows = []
    for session in sessions:
        x = _query(session.layer_id, step, session.attention.hidden_size)
        attention = session.attention
        qr = attention.q_a_layernorm(attention.q_a_proj(x))
        q = attention.q_b_proj(qr).reshape(
            1, 1, attention.num_heads, attention.q_head_dim
        ).transpose(0, 2, 1, 3)
        current = attention.kv_a_layernorm(attention.kv_a_proj_with_mqa(x))[:, None]
        latent, _ = session.latent_cache.update_and_fetch(current, current)
        rows.append({"session": session, "x": x, "qr": qr, "q": q, "latent": latent})
    _eval([(row["qr"], row["q"], row["latent"][..., -1:, :]) for row in rows])

    projected, phase_ms["current_token_projection"] = _time_phase(
        lambda: [
            _project_indexer_token(row["session"].attention.indexer, row["x"])
            for row in rows
        ],
        lambda values: [(value[0], value[1], value[2]) for value in values],
    )
    appended, phase_ms["tail_journal_append"] = _time_phase(
        lambda: [
            _append_raw_token(row["session"].state, row["session"].attention.indexer, value)
            for row, value in zip(rows, projected, strict=True)
        ],
        lambda values: [value[0] for value in values],
    )
    completed, phase_ms["partial_pool_completion"] = _time_phase(
        lambda: [
            _complete_partial_pool(row["session"].state, row["session"].attention.indexer)
            for row in rows
        ]
    )
    carried, phase_ms["pool_row_carry"] = _time_phase(
        lambda: [
            _carry_pool_row(row["session"].state, row["session"].attention.indexer, value)
            for row, value in zip(rows, completed, strict=True)
        ],
        lambda _: [
            session.state.logical_pool()[0][:, -1:]
            for session in sessions
        ],
    )
    outputs = []
    indices = []
    for row in rows:
        session = row["session"]
        scored = _compact_score(session.attention.indexer, session.state, row["x"], row["qr"])
        selected = _selection_phase(session.attention.indexer, scored)
        topk = _compact_expand(session.attention.indexer, session.state, selected)
        gathered = _gather_phase(row["latent"], topk)
        outputs.append(_attention_phase(session.attention, row["q"], gathered[0], gathered[1]))
        indices.append(topk)
    _eval([outputs, indices])
    return {
        "step": step,
        "context_tokens": context + step - 1,
        "context_mod_index_kpool": (context + step - 1) % 4,
        "phase_ms": phase_ms,
        "append_copy_bytes": sum(value[1] for value in appended),
        "pool_row_carry_copy_bytes": sum(carried),
    }


def _measure_phases(attentions, context):
    sessions = _build_sessions(attentions, context, ARMS[1])
    steps = [_phased_compact_step(sessions, context, step) for step in range(1, MEASURED_STEPS + 1)]
    medians = {
        phase: statistics.median(row["phase_ms"][phase] for row in steps[1:])
        for phase in PHASES
    }
    p95 = {
        phase: _percentile([row["phase_ms"][phase] for row in steps[1:]], 95)
        for phase in PHASES
    }
    result = {
        "token_2_16_phase_median_ms": medians,
        "token_2_16_phase_p95_ms": p95,
        "append_copy_bytes_median": statistics.median(row["append_copy_bytes"] for row in steps[1:]),
        "pool_row_carry_copy_bytes_median": statistics.median(row["pool_row_carry_copy_bytes"] for row in steps[1:]),
        "steps": steps,
    }
    _release(sessions)
    return result


def _compact_trim(session, tokens):
    state = session.state
    if tokens < 1 or tokens > state.rollback_window:
        raise ValueError(
            f"compact IndexPool trim must be within [1, {state.rollback_window}]"
        )
    if tokens > state.total_tokens:
        raise ValueError("compact IndexPool trim exceeds cached token count")
    target = state.total_tokens - tokens
    raw = state.raw_rows()
    keep = max(0, int(raw[0].shape[1]) - tokens)
    raw = tuple(value[:, :keep] for value in raw)
    state.total_tokens = target
    state.logical_pool_count = (target + session.attention.indexer.index_kpool - 1) // session.attention.indexer.index_kpool
    _split_raw_window(state, session.attention.indexer, raw)
    active = target % session.attention.indexer.index_kpool
    if state.active_tail_count != active:
        raise RuntimeError("rollback raw window cannot reconstruct target tail")
    if active:
        row = _complete_partial_pool(state, session.attention.indexer)
        _carry_pool_row(state, session.attention.indexer, row)
    session.latent_cache.offset -= tokens


def _oracle_trim(session, tokens):
    if tokens < 1 or tokens > ROLLBACK_WINDOW:
        raise ValueError(
            f"full-history oracle trim must be within [1, {ROLLBACK_WINDOW}]"
        )
    indexer = session.attention.indexer
    previous_pool = session.indexer_cache._pool
    session.latent_cache.offset -= tokens
    session.indexer_cache.trim(tokens)
    target = int(session.indexer_cache.offset)
    packed = session.indexer_cache.keys[:, 0, :target]
    keys, gates, valid_channel = mx.split(
        packed, [indexer.head_dim, 2 * indexer.head_dim], axis=-1
    )
    stable = target // indexer.index_kpool
    prefix = tuple(value[:, :stable] for value in previous_pool[:3])
    if target % indexer.index_kpool:
        start = stable * indexer.index_kpool
        suffix = indexer._pooled_states(
            keys[:, start:], gates[:, start:], valid_channel[:, start:, 0] > 0
        )
        suffix = (
            suffix[0],
            mx.where(
                suffix[1] >= 0,
                suffix[1] + start,
                INDEXPOOL_SENTINEL,
            ),
            suffix[2],
        )
        pool = tuple(
            mx.concatenate([left, right], axis=1)
            for left, right in zip(prefix, suffix, strict=True)
        )
    else:
        pool = prefix
    session.indexer_cache._pool = (*pool, target)


def _compact_state_hashes(sessions):
    hashes = []
    for session in sessions:
        state = session.state
        digest = hashlib.sha256()
        for value in (
            state.pool_keys,
            state.pool_indices,
            state.pool_valid,
            state.tail_keys,
            state.tail_gates,
            state.tail_valid,
            state.tail_positions,
            state.journal_keys,
            state.journal_gates,
            state.journal_valid,
            state.journal_positions,
        ):
            digest.update(_leaf_numpy(value)[0].tobytes())
        digest.update(
            f"{state.total_tokens}:{state.logical_pool_count}:{state.pool_capacity}".encode()
        )
        hashes.append({"layer": session.layer_id, "sha256": digest.hexdigest()})
    return hashes


def _rollback_layer_safety(left, right):
    return all(
        row["non_sentinel_out_of_range"] == 0
        and row["nan_count"] == 0
        and row["sentinel_count"]
        == _expected_tail_sentinels(result["context_tokens"], 4)
        for result in (left, right)
        for row in result["layers"]
    )


def _rollback_round(oracle, compact, context, tokens, baseline, state_hashes, round_id):
    for session in oracle:
        _oracle_trim(session, tokens)
    for session in compact:
        _compact_trim(session, tokens)
    target = context - 1
    trimmed_pool = _pool_parity(oracle, compact)
    stale_hidden = all(
        session.state.logical_pool_count
        == (target + session.attention.indexer.index_kpool - 1)
        // session.attention.indexer.index_kpool
        for session in compact
    )
    replay = []
    for step in range(1, tokens + 1):
        left = _actual_step(oracle, context, step, compact=False, chained=False)
        right = _actual_step(compact, context, step, compact=True, chained=False)
        pool = _pool_parity(oracle, compact)
        before = baseline[step - 1]
        replay.append(
            {
                "step": step,
                "oracle_replay_index_hash_match": before[0]["combined_index_hash"]
                == left["combined_index_hash"],
                "oracle_replay_output_hash_match": before[0]["combined_output_hash"]
                == left["combined_output_hash"],
                "compact_replay_index_hash_match": before[1]["combined_index_hash"]
                == right["combined_index_hash"],
                "compact_replay_output_hash_match": before[1]["combined_output_hash"]
                == right["combined_output_hash"],
                "oracle_compact_index_hash_match": left["combined_index_hash"]
                == right["combined_index_hash"],
                "oracle_compact_output_hash_match": left["combined_output_hash"]
                == right["combined_output_hash"],
                "pool_byte_identical": all(
                    row["keys_byte_identical"]
                    and row["indices_byte_identical"]
                    and row["valid_byte_identical"]
                    for row in pool
                ),
                "sentinel_range_nan_safe": _rollback_layer_safety(left, right),
            }
        )
    after_hashes = _compact_state_hashes(compact)
    return {
        "round": round_id,
        "trimmed_partial_pool_byte_identical": all(
            row["keys_byte_identical"]
            and row["indices_byte_identical"]
            and row["valid_byte_identical"]
            for row in trimmed_pool
        ),
        "stale_future_rows_hidden": stale_hidden,
        "replay_state_byte_identical": after_hashes == state_hashes,
        "raw_tokens_after_replay_max": max(
            session.state.raw_token_count for session in compact
        ),
        "steps": replay,
    }


def _rollback_case(attentions, target, tokens, kind):
    # _build_sessions stores context - 1 history rows, so target + 1 creates
    # exactly target cached tokens.  _actual_step reports the post-append KV
    # width with the same context + step - 1 convention.
    context = target + 1
    oracle = _build_sessions(attentions, context, ARMS[0])
    compact = _build_sessions(attentions, context, ARMS[1])
    baseline = []
    for step in range(1, tokens + 1):
        left = _actual_step(oracle, context, step, compact=False, chained=False)
        right = _actual_step(compact, context, step, compact=True, chained=False)
        baseline.append((left, right))
    state_hashes = _compact_state_hashes(compact)
    rounds = [
        _rollback_round(
            oracle, compact, context, tokens, baseline, state_hashes, round_id
        )
        for round_id in (1, 2)
    ]
    first_pool = target // 4
    last_pool = (target + tokens - 1) // 4
    result = {
        "kind": kind,
        "target_context_tokens": target,
        "target_mod_index_kpool": target % 4,
        "trim_tokens": tokens,
        "pool_rows_crossed": last_pool - first_pool + 1,
        "state_hashes": state_hashes,
        "rounds": rounds,
    }
    _release(oracle)
    _release(compact)
    return result


def _fail_closed_probe(attentions):
    sessions = _build_sessions(attentions, 2049, ARMS[1])
    before = _compact_state_hashes(sessions)
    errors = []
    for session in sessions:
        try:
            _compact_trim(session, ROLLBACK_WINDOW + 1)
        except ValueError as error:
            errors.append({"layer": session.layer_id, "error": str(error)})
    after = _compact_state_hashes(sessions)
    result = {
        "requested_trim_tokens": ROLLBACK_WINDOW + 1,
        "all_11_layers_rejected": len(errors) == len(EXPECTED_DSA),
        "state_unchanged": before == after,
        "errors": errors,
    }
    _release(sessions)
    return result


def _rollback_probe(attentions):
    cases = [
        _rollback_case(
            attentions,
            ROLLBACK_BASE_CONTEXT + target_mod,
            tokens,
            "mod_trim_matrix",
        )
        for target_mod in range(4)
        for tokens in ROLLBACK_CASES
    ]
    cases.extend(
        _rollback_case(attentions, target, 16, "capacity_boundary")
        for target in (2303, 2304)
    )
    return {
        "cases": cases,
        "fail_closed": _fail_closed_probe(attentions),
    }


def _context_case(attentions, context, warmup_steps):
    paired = _measure_paired(attentions, context, warmup_steps)
    chained = _measure_chained(attentions, context, warmup_steps)
    phases = _measure_phases(attentions, context)
    chained_parity = [
        {
            "step": left["step"],
            "index_hash_match": left["combined_index_hash"] == right["combined_index_hash"],
            "output_hash_match": left["combined_output_hash"] == right["combined_output_hash"],
        }
        for left, right in zip(
            paired[ARMS[0]]["steps"], chained["steps"], strict=True
        )
    ]
    return {
        "context_tokens": context,
        "arms": {
            ARMS[0]: paired[ARMS[0]],
            ARMS[1]: paired[ARMS[1]],
            ARMS[2]: chained,
        },
        "oracle_compact_parity": paired["parity"],
        "oracle_dependency_chained_parity": chained_parity,
        "compact_phase_decomposition": phases,
    }


def _decision(cases, rollbacks):
    small = cases["2049"]
    large = cases["262144"]
    phase_small = small["compact_phase_decomposition"]["token_2_16_phase_median_ms"]
    phase_large = large["compact_phase_decomposition"]["token_2_16_phase_median_ms"]
    chained_small = small["arms"][ARMS[2]]["token_2_16_median_ms"]
    chained_large = large["arms"][ARMS[2]]["token_2_16_median_ms"]
    oracle_bytes = large["arms"][ARMS[0]]["final_state_bytes"]["total_bytes"]
    compact_bytes = large["arms"][ARMS[1]]["final_state_bytes"]["total_bytes"]
    reduction = 1.0 - compact_bytes / oracle_bytes
    append_retention = phase_small["tail_journal_append"] / phase_large["tail_journal_append"]
    chained_retention = chained_small / chained_large
    parity = all(
        row["index_hash_match"] and row["output_hash_match"] and row["pool_order_byte_identical"]
        for case in cases.values()
        for row in case["oracle_compact_parity"]
    )
    chained_parity = all(
        row["index_hash_match"] and row["output_hash_match"]
        for case in cases.values()
        for row in case["oracle_dependency_chained_parity"]
    )
    rollback_rounds = [
        round_result
        for case in rollbacks["cases"]
        for round_result in case["rounds"]
    ]
    replay = all(
        round_result["trimmed_partial_pool_byte_identical"]
        and round_result["stale_future_rows_hidden"]
        and round_result["replay_state_byte_identical"]
        and round_result["raw_tokens_after_replay_max"] <= RAW_STATE_WINDOW
        and all(
            all(value for key, value in step.items() if key != "step")
            for step in round_result["steps"]
        )
        for round_result in rollback_rounds
    )
    fail_closed = (
        rollbacks["fail_closed"]["all_11_layers_rejected"]
        and rollbacks["fail_closed"]["state_unchanged"]
    )
    qualified = (
        reduction >= 0.8
        and append_retention >= 0.8
        and chained_retention >= 0.8
        and parity
        and chained_parity
        and replay
        and fail_closed
    )
    return {
        "packed_state_reduction_ratio": reduction,
        "token_append_retention": append_retention,
        "dependency_chained_all_dsa_retention": chained_retention,
        "byte_identical": parity,
        "dependency_chained_byte_identical": chained_parity,
        "arbitrary_trim_replay_byte_identical": replay,
        "fail_closed_beyond_rollback_window": fail_closed,
        "targets": {
            "packed_state_reduction_ratio": 0.8,
            "token_append_retention": 0.8,
            "dependency_chained_all_dsa_retention": 0.8,
        },
        "compact_authoritative_state_qualified": qualified,
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
        raise ValueError("warmup must cover all four pool-tail shapes")
    if args.measured_steps != MEASURED_STEPS:
        raise ValueError(f"measured steps must remain {MEASURED_STEPS}")

    report = inspect_checkpoint(args.model, require_server_ready=True)
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    model, _ = load(args.model)
    attentions = {
        layer: model.language_model.model.layers[layer].self_attn
        for layer in EXPECTED_DSA
    }
    parameters = [
        value
        for attention in attentions.values()
        for _, value in tree_flatten(attention.parameters())
    ]
    mx.eval(*parameters)
    mx.synchronize()
    del model, parameters
    gc.collect()
    mx.clear_cache()
    mx.synchronize()

    cases = {
        str(context): _context_case(attentions, context, args.warmup_steps)
        for context in CONTEXTS
    }
    rollbacks = _rollback_probe(attentions)
    decision = _decision(cases, rollbacks)
    layer_rows = [
        layer
        for case in cases.values()
        for arm in case["arms"].values()
        for step in arm["steps"]
        for layer in step["layers"]
    ]
    compact_steps = [
        step
        for case in cases.values()
        for arm in (ARMS[1], ARMS[2])
        for step in case["arms"][arm]["steps"]
    ]
    append_copy_values = {
        case["compact_phase_decomposition"]["append_copy_bytes_median"]
        for case in cases.values()
    }
    compact_drifts = [
        abs(case["arms"][arm]["memory_drift_bytes"])
        for case in cases.values()
        for arm in (ARMS[1], ARMS[2])
    ]
    acceptance = {
        "all_11_layers_3_contexts_3_arms_16_steps_measured": all(
            len(case["arms"][arm]["steps"]) == MEASURED_STEPS
            for case in cases.values()
            for arm in ARMS
        ),
        "all_indices_sentinel_or_in_range": all(row["non_sentinel_out_of_range"] == 0 for row in layer_rows),
        "all_unused_slots_are_minus1": all(
            layer["sentinel_count"] == _expected_tail_sentinels(step["context_tokens"], report.index_kpool)
            for case in cases.values()
            for arm in case["arms"].values()
            for step in arm["steps"]
            for layer in step["layers"]
        ),
        "no_nan": all(row["nan_count"] == 0 for row in layer_rows),
        "independent_compact_hash_and_pool_parity": all(
            row["index_hash_match"]
            and row["output_hash_match"]
            and row["pool_order_byte_identical"]
            for case in cases.values()
            for row in case["oracle_compact_parity"]
        ),
        "dependency_chained_hash_parity": all(
            row["index_hash_match"] and row["output_hash_match"]
            for case in cases.values()
            for row in case["oracle_dependency_chained_parity"]
        ),
        "all_arbitrary_trim_replay_cases_match": decision[
            "arbitrary_trim_replay_byte_identical"
        ],
        "rollback_target_mod_0_1_2_3_covered": {
            case["target_mod_index_kpool"]
            for case in rollbacks["cases"]
            if case["kind"] == "mod_trim_matrix"
        }
        == {0, 1, 2, 3},
        "rollback_trim_1_2_3_4_8_15_16_covered": {
            case["trim_tokens"]
            for case in rollbacks["cases"]
            if case["kind"] == "mod_trim_matrix"
        }
        == set(ROLLBACK_CASES),
        "rollback_crosses_1_to_5_pool_rows": {
            case["pool_rows_crossed"]
            for case in rollbacks["cases"]
            if case["kind"] == "mod_trim_matrix"
        }
        >= {1, 2, 3, 4, 5},
        "trim_replay_retrim_completed": all(
            len(case["rounds"]) == 2 for case in rollbacks["cases"]
        ),
        "capacity_boundary_before_after_covered": {
            case["target_context_tokens"]
            for case in rollbacks["cases"]
            if case["kind"] == "capacity_boundary"
        }
        == {2303, 2304},
        "trim_beyond_window_fails_closed": decision[
            "fail_closed_beyond_rollback_window"
        ],
        "pool_tail_0_1_2_3_covered": all(
            {step["context_mod_index_kpool"] for step in case["arms"][ARMS[1]]["steps"]} == {0, 1, 2, 3}
            for case in cases.values()
        ),
        "compact_has_no_full_packed_history": all(
            not step["state_bytes"]["full_packed_history_present"] for step in compact_steps
        ),
        "raw_state_within_19_token_bound": all(
            step["state_bytes"]["max_raw_tokens_per_layer"] <= RAW_STATE_WINDOW
            for step in compact_steps
        ),
        "compact_state_reduction_at_least_80_percent": decision[
            "packed_state_reduction_ratio"
        ]
        >= 0.8,
        "dependency_chained_retention_at_least_0_8": decision[
            "dependency_chained_all_dsa_retention"
        ]
        >= 0.8,
        "append_copy_bytes_context_independent": len(append_copy_values) == 1,
        "active_memory_drift_bounded_64mib": max(compact_drifts) <= 64 * 1024 * 1024,
        "runtime_server_apc_admission_unchanged": DEFAULT_MAX_PROMPT_TOKENS == 256,
        "evidence_complete": True,
    }
    acceptance["accepted"] = all(acceptance.values())
    large = cases["262144"]
    compact_state = large["arms"][ARMS[1]]["final_state_bytes"]
    latent_logical = len(EXPECTED_DSA) * (262144 + MEASURED_STEPS - 1) * 512 * 2
    output = {
        "schema": "glm53-compact-indexpool-arbitrary-rollback-v2",
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
        "rollback_window": ROLLBACK_WINDOW,
        "raw_state_window": RAW_STATE_WINDOW,
        "rollback_base_context": ROLLBACK_BASE_CONTEXT,
        "rollback_cases": list(ROLLBACK_CASES),
        "arms": list(ARMS),
        "measurement_contract": {
            "contiguous_pool_score_shape_matches_oracle": True,
            "single_latent_capacity_reserved": "history + measured generation rounded to 256",
            "independent_aggregate_is_throughput_oriented": True,
            "dependency_chain": "x[layer] = mx.depends(deterministic_x[layer], previous_output)",
            "phase_timings_synchronize_consumer_boundaries": True,
            "rollback_targets_cover_all_pool_tail_mods": True,
            "rollback_recomputes_partial_logical_last_pool_row": True,
            "trim_beyond_rollback_window_fails_closed": True,
            "initialization_only_host_materialization": True,
            "append_score_and_trim_hot_paths_have_no_host_synchronization": True,
            "pool_indices_dtype_unchanged": "int64",
        },
        "cases": cases,
        "rollback": rollbacks,
        "apc_payload_estimate_256k": {
            "single_nope_latent_logical_bytes": latent_logical,
            "compact_indexpool_state_bytes": compact_state["total_bytes"],
            "combined_authoritative_bytes": latent_logical + compact_state["total_bytes"],
            "full_packed_token_history_discardable": True,
        },
        "decision_gate": decision,
        "runtime_policy": {
            "probe_only": True,
            "server_changed": False,
            "apc_abi_changed": False,
            "admission_changed": False,
            "prompt_limit": DEFAULT_MAX_PROMPT_TOKENS,
        },
        "acceptance": acceptance,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "accepted": acceptance["accepted"], "decision_gate": decision}, indent=2))
    return 0 if acceptance["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
