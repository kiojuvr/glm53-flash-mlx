#!/usr/bin/env python3
"""Characterize the layer-3 NoPE DSA S=1 frontier with synthetic caches."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from glm53_flash_mlx.abi import (
    GROUPED_MIN_ROUTES,
    MLX_VLM_REVISION,
    NOPE_DSA_CACHE_ABI,
)
from glm53_flash_mlx.indexpool import (
    INDEXPOOL_SENTINEL,
    prepare_decode_indexpool_gather,
    sanitize_indexpool_indices,
)
from glm53_flash_mlx.loader import load
from glm53_flash_mlx.manifest import inspect_checkpoint
from glm53_flash_mlx.server import DEFAULT_MAX_PROMPT_TOKENS

PRIMARY_CONTEXTS = (2048, 2049, 8192, 16384, 32768, 65536, 131072, 262144)
POOL_TAIL_CONTEXTS = (32768, 32769, 32770, 32771)
MODES = ("steady_incremental", "pool_rebuild")
PHASES = ("pool_update", "score", "selection", "expand", "gather", "attention")


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), file=sys.stderr, flush=True)


def _arrays(value) -> list[mx.array]:
    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, dict):
        return [array for item in value.values() for array in _arrays(item)]
    if isinstance(value, (tuple, list)):
        return [array for item in value for array in _arrays(item)]
    return []


def _eval(value) -> None:
    arrays = _arrays(value)
    if arrays:
        mx.eval(*arrays)
    mx.synchronize()


def _np(value, *, dtype=None) -> np.ndarray:
    mx.eval(value)
    array = np.ascontiguousarray(np.asarray(value))
    return array.astype(dtype, copy=False) if dtype is not None else array


def _hash_array(value) -> str:
    array = _np(value.astype(mx.float32), dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _hash_indices(value) -> str | None:
    if value is None:
        return None
    array = _np(value, dtype=np.int32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _deterministic_rows(rows: int, width: int, phase: float, dtype) -> mx.array:
    positions = mx.arange(rows, dtype=mx.float32)[:, None]
    columns = mx.arange(width, dtype=mx.float32)[None, :]
    values = mx.sin(positions * 0.0009765625 + columns * 0.0078125 + phase)
    return values.astype(dtype)


def _capacity(tokens: int, step: int = 256) -> int:
    if tokens <= 0:
        return 0
    return ((tokens + step - 1) // step) * step


def _build_base(attention, context_tokens: int) -> dict:
    indexer = attention.indexer
    history = context_tokens - 1
    capacity = _capacity(history)
    key_history = _deterministic_rows(
        capacity, indexer.head_dim, 0.125, mx.bfloat16
    )[None]
    gate_history = _deterministic_rows(
        capacity, indexer.head_dim, 0.625, mx.bfloat16
    )[None]
    valid_history = (mx.arange(capacity) < history)[None, :, None]
    packed_history = mx.concatenate(
        [
            key_history,
            gate_history,
            valid_history.astype(mx.bfloat16),
        ],
        axis=-1,
    )
    latent_history = _deterministic_rows(
        capacity, attention.kv_lora_rank, 1.125, mx.bfloat16
    ).reshape(1, 1, capacity, attention.kv_lora_rank)
    x = _deterministic_rows(1, attention.hidden_size, 1.625, mx.bfloat16)[
        None
    ]

    if history:
        logical = packed_history[:, :history]
        key_full, gate_full, valid_ch = mx.split(
            logical, [indexer.head_dim, 2 * indexer.head_dim], axis=-1
        )
        steady_pool = (*indexer._pooled_states(
            key_full, gate_full, valid_ch[..., 0] > 0
        ), history)
    else:
        steady_pool = None

    values = [packed_history, latent_history, x]
    if steady_pool is not None:
        values.extend(steady_pool[:3])
    mx.eval(*values)
    mx.synchronize()
    return {
        "context_tokens": context_tokens,
        "history": history,
        "capacity": capacity,
        "packed_history": packed_history,
        "latent_history": latent_history,
        "steady_pool": steady_pool,
        "x": x,
    }


def _new_indexer_cache(base: dict, mode: str):
    from mlx_vlm.models.cache import KVCache

    cache = KVCache()
    cache.keys = base["packed_history"][:, None]
    cache.values = mx.zeros(
        (1, 1, base["capacity"], 0), dtype=mx.bfloat16
    )
    cache.offset = base["history"]
    cache._no_pad = True
    cache._pool = base["steady_pool"] if mode == "steady_incremental" else None
    return cache


def _new_latent_cache(base: dict):
    from mlx_vlm.models.cache import KVCache

    cache = KVCache()
    cache.keys = base["latent_history"]
    cache.values = base["latent_history"]
    cache.offset = base["history"]
    return cache


def _current_projections(attention, x):
    qr = attention.q_a_layernorm(attention.q_a_proj(x))
    q = attention.q_b_proj(qr).reshape(
        1, 1, attention.num_heads, attention.q_head_dim
    ).transpose(0, 2, 1, 3)
    latent = attention.kv_a_layernorm(attention.kv_a_proj_with_mqa(x))[:, None]
    return qr, q, latent


def _pool_update(indexer, x, qr, cache) -> dict:
    key = indexer.k_norm(indexer.wk(x)).reshape(1, 1, indexer.head_dim)
    gate = x @ indexer.index_kpool_compress_gate.swapaxes(-1, -2)
    valid_cur = mx.ones((1, 1), dtype=mx.bool_)
    packed = mx.concatenate(
        [key, gate, valid_cur.astype(key.dtype)[..., None]], axis=-1
    )
    keys, _ = cache.update_and_fetch(
        packed[:, None], mx.zeros((1, 1, 1, 0), dtype=key.dtype)
    )
    packed_full = keys[:, 0]
    total = packed_full.shape[1]
    if total <= indexer.index_topk and getattr(indexer, "bypass_short", True):
        return {
            "path": "dense_bypass",
            "qr": qr,
            "valid_cur": valid_cur,
            "packed_full": packed_full,
            "pool_keys": None,
            "pool_indices": None,
            "pool_valid": None,
            "valid": mx.ones((1, total), dtype=mx.bool_),
        }

    key_full, gate_full, valid_ch = mx.split(
        packed_full, [indexer.head_dim, 2 * indexer.head_dim], axis=-1
    )
    valid = valid_ch[..., 0] > 0
    if (
        getattr(cache, "_pool", None) is not None
        and getattr(cache, "_no_pad", False)
        and cache._pool[0].shape[0] == 1
    ):
        cached_keys, cached_indices, cached_valid, previous = cache._pool
        stable = previous // indexer.index_kpool
        suffix_start = stable * indexer.index_kpool
        suffix_keys, suffix_indices, suffix_valid = indexer._pooled_states(
            key_full[:, suffix_start:],
            gate_full[:, suffix_start:],
            valid[:, suffix_start:],
        )
        suffix_indices = mx.where(
            suffix_indices >= 0, suffix_indices + suffix_start, INDEXPOOL_SENTINEL
        )
        pool_keys = mx.concatenate(
            [cached_keys[:, :stable], suffix_keys], axis=1
        )
        pool_indices = mx.concatenate(
            [cached_indices[:, :stable], suffix_indices], axis=1
        )
        pool_valid = mx.concatenate(
            [cached_valid[:, :stable], suffix_valid], axis=1
        )
    else:
        pool_keys, pool_indices, pool_valid = indexer._pooled_states(
            key_full, gate_full, valid
        )
        cache._no_pad = True
    cache._pool = (pool_keys, pool_indices, pool_valid, total)
    return {
        "path": "sparse_indexpool",
        "qr": qr,
        "valid_cur": valid_cur,
        "packed_full": packed_full,
        "pool_keys": pool_keys,
        "pool_indices": pool_indices,
        "pool_valid": pool_valid,
        "valid": valid,
    }


def _score_phase(indexer, pooled: dict) -> dict:
    pool_keys = pooled["pool_keys"]
    pool_indices = pooled["pool_indices"]
    pool_valid = pooled["pool_valid"]
    total = pooled["packed_full"].shape[1]
    pool_count = pool_keys.shape[1]
    pool_keys_t = pool_keys[:, None].swapaxes(-1, -2)
    query = indexer.wq_b(pooled["qr"]).reshape(
        1, 1, indexer.n_heads, indexer.head_dim
    )
    scores = query @ pool_keys_t
    scores = mx.maximum(scores * indexer.softmax_scale, 0.0)
    # The real weights projection is already represented in q construction;
    # retain the exact indexer mixture weights from the current token.
    weights = indexer.weights_proj(pooled["x"]) * (indexer.n_heads**-0.5)
    index_scores = mx.sum(weights[..., None] * scores, axis=2)
    pool_end = mx.clip(pool_indices[..., -1], 0, total - 1)
    visible = pool_end[:, None, :] < total
    valid_candidates = visible & pool_valid[:, None]
    masked_scores = mx.where(valid_candidates, index_scores, -1e30)
    return {
        "index_scores": masked_scores,
        "valid_candidates": valid_candidates,
        "pool_count": pool_count,
    }


def _selection_phase(indexer, scored: dict) -> dict:
    pool_count = scored["pool_count"]
    select_k = min(indexer.index_topk // indexer.index_kpool, pool_count)
    order = mx.argsort(-scored["index_scores"], axis=-1)
    selected = order[..., :select_k]
    selected_valid = mx.take_along_axis(
        scored["valid_candidates"], selected, axis=-1
    )
    return {
        "selected": selected,
        "selected_valid": selected_valid,
        "select_k": select_k,
    }


def _expand_phase(indexer, pooled: dict, selected: dict) -> mx.array:
    total = pooled["packed_full"].shape[1]
    pool_indices = pooled["pool_indices"]
    pool_count = pool_indices.shape[1]
    select_k = selected["select_k"]
    pi = mx.broadcast_to(
        pool_indices[:, None], (1, 1, pool_count, indexer.index_kpool)
    )
    selected_expanded = mx.broadcast_to(
        selected["selected"][..., None],
        (1, 1, select_k, indexer.index_kpool),
    )
    selected_indices = mx.take_along_axis(pi, selected_expanded, axis=2)
    topk = selected_indices.reshape(1, 1, select_k * indexer.index_kpool)
    selected_valid = mx.broadcast_to(
        selected["selected_valid"][..., None],
        (1, 1, select_k, indexer.index_kpool),
    ).reshape(1, 1, select_k * indexer.index_kpool)
    topk = mx.where(selected_valid, topk, INDEXPOOL_SENTINEL)
    if indexer.index_kpool_always_select_tail and indexer.index_kpool > 1:
        visible = mx.ones((1, 1, total), dtype=mx.bool_)
        topk = mx.concatenate(
            [topk, indexer._visible_tail(visible, pooled["valid"])], axis=-1
        )
    output_width = indexer.index_topk + (
        indexer.index_kpool - 1
        if indexer.index_kpool_always_select_tail and indexer.index_kpool > 1
        else 0
    )
    if topk.shape[-1] < output_width:
        topk = mx.concatenate(
            [
                topk,
                mx.full(
                    (1, 1, output_width - topk.shape[-1]),
                    INDEXPOOL_SENTINEL,
                    dtype=topk.dtype,
                ),
            ],
            axis=-1,
        )
    topk = topk[..., :output_width]
    topk = mx.where(pooled["valid_cur"][..., None], topk, INDEXPOOL_SENTINEL)
    topk = topk[:, None].astype(mx.int32)
    return sanitize_indexpool_indices(topk, total)


def _gather_phase(latent_full, indices):
    raw = indices[:, :, 0, :]
    safe, valid = prepare_decode_indexpool_gather(raw, latent_full.shape[2])
    expanded = mx.broadcast_to(
        safe[..., None], safe.shape + (latent_full.shape[-1],)
    )
    gathered = mx.take_along_axis(latent_full, expanded, axis=2)
    return gathered, valid[:, :, None, :]


def _attention_phase(attention, q, latent, mask):
    q = attention.embed_q(q)
    output = mx.fast.scaled_dot_product_attention(
        q, latent, latent, scale=attention.scale, mask=mask
    )
    output = attention.unembed_out(output)
    output = output.transpose(0, 2, 1, 3).reshape(1, 1, -1)
    return attention.o_proj(output)


def _manual_indexer(indexer, x, qr, cache):
    pooled = _pool_update(indexer, x, qr, cache)
    pooled["x"] = x
    if pooled["path"] == "dense_bypass":
        return None
    scored = _score_phase(indexer, pooled)
    selected = _selection_phase(indexer, scored)
    return _expand_phase(indexer, pooled, selected)


def _operator_graph(attention, base: dict, mode: str):
    x = base["x"]
    indexer_cache = _new_indexer_cache(base, mode)
    latent_cache = _new_latent_cache(base)
    qr, q, latent_current = _current_projections(attention, x)
    latent_full, _ = latent_cache.update_and_fetch(latent_current, latent_current)
    pooled = _pool_update(attention.indexer, x, qr, indexer_cache)
    pooled["x"] = x
    if pooled["path"] == "dense_bypass":
        output = _attention_phase(attention, q, latent_full, None)
        return output, None, pooled["path"]
    scored = _score_phase(attention.indexer, pooled)
    selected = _selection_phase(attention.indexer, scored)
    indices = _expand_phase(attention.indexer, pooled, selected)
    gathered, mask = _gather_phase(latent_full, indices)
    output = _attention_phase(attention, q, gathered, mask)
    return output, indices, pooled["path"]


class _CapturingIndexer:
    def __init__(self, delegate):
        self.delegate = delegate
        self.indices = None

    def __call__(self, *args, **kwargs):
        self.indices = self.delegate(*args, **kwargs)
        return self.indices


def _reference_graph(attention, base: dict, mode: str):
    original = attention.indexer
    capture = _CapturingIndexer(original)
    attention.indexer = capture
    try:
        output = attention(
            base["x"],
            mask=None,
            cache=[_new_latent_cache(base), _new_indexer_cache(base, mode)],
        )
        _eval((output, capture.indices))
        return output, capture.indices
    finally:
        attention.indexer = original


def _time_stage(fn):
    started = time.perf_counter()
    value = fn()
    _eval(value)
    return value, (time.perf_counter() - started) * 1000.0


def _phase_sample(attention, base: dict, mode: str) -> dict:
    indexer = attention.indexer
    indexer_cache = _new_indexer_cache(base, mode)
    latent_cache = _new_latent_cache(base)
    qr, q, latent_current = _current_projections(attention, base["x"])
    latent_full, _ = latent_cache.update_and_fetch(latent_current, latent_current)
    _eval((qr, q, latent_full))
    baseline = mx.get_active_memory()
    mx.reset_peak_memory()

    pooled, pool_ms = _time_stage(
        lambda: _pool_update(indexer, base["x"], qr, indexer_cache)
    )
    pooled["x"] = base["x"]
    if pooled["path"] == "dense_bypass":
        output, attention_ms = _time_stage(
            lambda: _attention_phase(attention, q, latent_full, None)
        )
        indices = None
        phase_ms = {
            "pool_update": pool_ms,
            "score": 0.0,
            "selection": 0.0,
            "expand": 0.0,
            "gather": 0.0,
            "attention": attention_ms,
        }
    else:
        scored, score_ms = _time_stage(lambda: _score_phase(indexer, pooled))
        selected, selection_ms = _time_stage(
            lambda: _selection_phase(indexer, scored)
        )
        indices, expand_ms = _time_stage(
            lambda: _expand_phase(indexer, pooled, selected)
        )
        gathered, gather_ms = _time_stage(
            lambda: _gather_phase(latent_full, indices)
        )
        output, attention_ms = _time_stage(
            lambda: _attention_phase(attention, q, gathered[0], gathered[1])
        )
        phase_ms = {
            "pool_update": pool_ms,
            "score": score_ms,
            "selection": selection_ms,
            "expand": expand_ms,
            "gather": gather_ms,
            "attention": attention_ms,
        }
    peak = mx.get_peak_memory()
    active = mx.get_active_memory()
    return {
        "phase_ms": phase_ms,
        "synchronized_phase_sum_ms": sum(phase_ms.values()),
        "active_memory_bytes": int(active),
        "peak_memory_bytes": int(peak),
        "working_peak_bytes": int(max(0, peak - baseline)),
        "index_hash": _hash_indices(indices),
        "output_hash": _hash_array(output),
    }


def _index_stats(indices, context_tokens: int) -> dict:
    if indices is None:
        return {
            "valid_index_min": None,
            "valid_index_max": None,
            "valid_index_count": 0,
            "sentinel_count": 0,
            "non_sentinel_out_of_range": 0,
            "selected_token_width": context_tokens,
        }
    values = _np(indices, dtype=np.int32)
    valid = values != INDEXPOOL_SENTINEL
    valid_values = values[valid]
    bad = valid & ((values < 0) | (values >= context_tokens))
    return {
        "valid_index_min": int(valid_values.min()) if valid_values.size else None,
        "valid_index_max": int(valid_values.max()) if valid_values.size else None,
        "valid_index_count": int(valid_values.size),
        "sentinel_count": int(np.count_nonzero(~valid)),
        "non_sentinel_out_of_range": int(np.count_nonzero(bad)),
        "selected_token_width": int(values.shape[-1]),
    }


def _operator_sample(attention, base: dict, mode: str) -> tuple[dict, object, object]:
    gc.collect()
    baseline = mx.get_active_memory()
    mx.reset_peak_memory()
    started = time.perf_counter()
    output, indices, path = _operator_graph(attention, base, mode)
    _eval((output, indices))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    peak = mx.get_peak_memory()
    active = mx.get_active_memory()
    output_values = _np(output.astype(mx.float32), dtype=np.float32)
    row = {
        "elapsed_ms": elapsed_ms,
        "path": path,
        "active_memory_bytes": int(active),
        "peak_memory_bytes": int(peak),
        "working_peak_bytes": int(max(0, peak - baseline)),
        "index_hash": _hash_indices(indices),
        "output_hash": hashlib.sha256(output_values.tobytes()).hexdigest(),
        "nan_count": int(np.count_nonzero(np.isnan(output_values))),
        **_index_stats(indices, base["context_tokens"]),
    }
    return row, output, indices


def _benchmark_mode(attention, base: dict, mode: str, warmups: int, repeats: int):
    first, _, _ = _operator_sample(attention, base, mode)
    warmup_rows = []
    for sample in range(warmups):
        row, _, _ = _operator_sample(attention, base, mode)
        warmup_rows.append(row)
        _progress(
            "warmup",
            context=base["context_tokens"],
            mode=mode,
            sample=sample + 1,
            milliseconds=row["elapsed_ms"],
        )
    measured = []
    last_output = last_indices = None
    for sample in range(repeats):
        row, last_output, last_indices = _operator_sample(attention, base, mode)
        measured.append(row)
        _progress(
            "measured",
            context=base["context_tokens"],
            mode=mode,
            sample=sample + 1,
            milliseconds=row["elapsed_ms"],
        )
    phase_rows = [_phase_sample(attention, base, mode) for _ in range(repeats)]
    phase_medians = {
        phase: statistics.median(row["phase_ms"][phase] for row in phase_rows)
        for phase in PHASES
    }
    hashes_repeat = (
        len({row["index_hash"] for row in measured}) == 1
        and len({row["output_hash"] for row in measured}) == 1
    )
    median_elapsed = statistics.median(row["elapsed_ms"] for row in measured)
    summary = {
        "compile_first_run_ms": first["elapsed_ms"],
        "warmup_ms": [row["elapsed_ms"] for row in warmup_rows],
        "measurement_ms": [row["elapsed_ms"] for row in measured],
        "unsynchronized_end_to_end_ms": median_elapsed,
        "phase_samples_ms": [row["phase_ms"] for row in phase_rows],
        "pool_update_ms": phase_medians["pool_update"],
        "score_ms": phase_medians["score"],
        "selection_ms": phase_medians["selection"],
        "expand_ms": phase_medians["expand"],
        "gather_ms": phase_medians["gather"],
        "attention_ms": phase_medians["attention"],
        "synchronized_phase_sum_ms": sum(phase_medians.values()),
        "active_memory_bytes": max(row["active_memory_bytes"] for row in measured),
        "peak_memory_bytes": max(row["peak_memory_bytes"] for row in measured),
        "working_peak_bytes": max(row["working_peak_bytes"] for row in measured),
        "index_hash": measured[0]["index_hash"],
        "output_hash": measured[0]["output_hash"],
        "repeated_hash_match": hashes_repeat,
        "nan_count": max(row["nan_count"] for row in measured),
        "path": measured[0]["path"],
        "valid_index_min": measured[0]["valid_index_min"],
        "valid_index_max": measured[0]["valid_index_max"],
        "valid_index_count": measured[0]["valid_index_count"],
        "sentinel_count": measured[0]["sentinel_count"],
        "non_sentinel_out_of_range": measured[0]["non_sentinel_out_of_range"],
        "selected_token_width": measured[0]["selected_token_width"],
    }
    return summary, last_output, last_indices


def _mode_case(attention, base: dict, mode: str, warmups: int, repeats: int) -> dict:
    measured, output, indices = _benchmark_mode(
        attention, base, mode, warmups, repeats
    )
    reference_output, reference_indices = _reference_graph(attention, base, mode)
    reference_output_hash = _hash_array(reference_output)
    reference_index_hash = _hash_indices(reference_indices)
    measured.update(
        {
            "reference_indexer_hash": reference_index_hash,
            "reference_attention_output_hash": reference_output_hash,
            "manual_reference_index_match": measured["index_hash"]
            == reference_index_hash,
            "manual_reference_output_match": measured["output_hash"]
            == reference_output_hash,
        }
    )
    del output, indices, reference_output, reference_indices
    gc.collect()
    return measured


def _context_case(attention, context: int, warmups: int, repeats: int) -> dict:
    _progress("build_context", context=context)
    base = _build_base(attention, context)
    indexer = attention.indexer
    modes = {
        mode: _mode_case(attention, base, mode, warmups, repeats)
        for mode in MODES
    }
    theoretical_pools = (context + indexer.index_kpool - 1) // indexer.index_kpool
    selected_pools = (
        0
        if context <= indexer.index_topk
        else min(indexer.index_topk // indexer.index_kpool, theoretical_pools)
    )
    comparison = {
        "index_hash_match": modes["steady_incremental"]["index_hash"]
        == modes["pool_rebuild"]["index_hash"],
        "output_hash_match": modes["steady_incremental"]["output_hash"]
        == modes["pool_rebuild"]["output_hash"],
    }
    result = {
        "context_tokens": context,
        "context_mod_index_kpool": context % indexer.index_kpool,
        "pool_count": theoretical_pools,
        "selected_pool_count": selected_pools,
        "expected_valid_selected_tokens": (
            context
            if context <= indexer.index_topk
            else min(
                context // indexer.index_kpool,
                indexer.index_topk // indexer.index_kpool,
            )
            * indexer.index_kpool
            + context % indexer.index_kpool
        ),
        "selected_token_width": modes["steady_incremental"][
            "selected_token_width"
        ],
        "modes": modes,
        "steady_rebuild_differential": comparison,
    }
    del base
    gc.collect()
    mx.clear_cache()
    return result


def _dominant_phase(case: dict, mode: str) -> str:
    values = case["modes"][mode]
    return max(PHASES, key=lambda phase: values[f"{phase}_ms"])


def _next_candidate(cases: dict) -> dict:
    largest = cases[str(max(PRIMARY_CONTEXTS))]
    steady = _dominant_phase(largest, "steady_incremental")
    rebuild = _dominant_phase(largest, "pool_rebuild")
    steady_retention = largest["modes"]["steady_incremental"][
        "sparse_path_retention"
    ]
    rebuild_retention = largest["modes"]["pool_rebuild"][
        "sparse_path_retention"
    ]
    mapping = {
        "score": "tiled/fused Metal pool-score kernel",
        "selection": "exact partial top-k Metal kernel",
        "expand": "sentinel-aware fused expand/gather",
        "gather": "sentinel-aware fused expand/gather",
        "attention": "indexed NoPE attention",
        "pool_update": (
            "steady incremental pool-update without full-history validity "
            "materialization"
        ),
    }
    rebuild_only = steady_retention >= 0.80 and rebuild_retention < 0.80
    selected = (
        "pool state as first-class session/APC state"
        if rebuild_only
        else mapping[steady]
    )
    return {
        "steady_256k_dominant_phase": steady,
        "pool_rebuild_256k_dominant_phase": rebuild,
        "steady_256k_retention": steady_retention,
        "pool_rebuild_256k_retention": rebuild_retention,
        "degradation_classification": (
            "pool_rebuild_only" if rebuild_only else "steady_hot_path"
        ),
        "single_next_candidate": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = inspect_checkpoint(args.model, require_server_ready=True)
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    model, _ = load(args.model)
    layer = model.language_model.model.layers[args.layer]
    attention = layer.self_attn
    if layer.is_linear:
        raise RuntimeError(f"layer {args.layer} is not a DSA layer")
    parameters = [value for _, value in tree_flatten(attention.parameters())]
    mx.eval(*parameters)
    mx.synchronize()
    del layer, model, parameters
    gc.collect()
    mx.clear_cache()

    contexts = sorted(set(PRIMARY_CONTEXTS) | set(POOL_TAIL_CONTEXTS))
    cases = {
        str(context): _context_case(
            attention, context, args.warmups, args.repeats
        )
        for context in contexts
    }

    for mode in MODES:
        baseline = cases["2049"]["modes"][mode]["unsynchronized_end_to_end_ms"]
        for context in contexts:
            value = cases[str(context)]["modes"][mode]
            value["sparse_path_retention"] = (
                None
                if context <= attention.indexer.index_topk
                else baseline / value["unsynchronized_end_to_end_ms"]
            )

    all_modes = [case["modes"][mode] for case in cases.values() for mode in MODES]
    sparse_widths = {
        case["selected_token_width"]
        for case in cases.values()
        if case["context_tokens"] > attention.indexer.index_topk
    }
    transition = {
        "2048_path": cases["2048"]["modes"]["steady_incremental"]["path"],
        "2049_path": cases["2049"]["modes"]["steady_incremental"]["path"],
    }
    acceptance = {
        "all_primary_and_pool_tail_contexts_measured": set(map(int, cases))
        == set(contexts),
        "dense_2048_sparse_2049_transition": transition
        == {"2048_path": "dense_bypass", "2049_path": "sparse_indexpool"},
        "all_indices_sentinel_or_in_range": all(
            row["non_sentinel_out_of_range"] == 0 for row in all_modes
        ),
        "all_unused_slots_minus1": all(
            case["modes"][mode]["sentinel_count"]
            == case["selected_token_width"]
            - case["expected_valid_selected_tokens"]
            for case in cases.values()
            if case["context_tokens"] > attention.indexer.index_topk
            for mode in MODES
        ),
        "no_nan": all(row["nan_count"] == 0 for row in all_modes),
        "repeated_execution_byte_identical": all(
            row["repeated_hash_match"] for row in all_modes
        ),
        "manual_decomposition_matches_fixed_mlx_vlm": all(
            row["manual_reference_index_match"]
            and row["manual_reference_output_match"]
            for row in all_modes
        ),
        "steady_incremental_pool_rebuild_parity": all(
            case["steady_rebuild_differential"]["index_hash_match"]
            and case["steady_rebuild_differential"]["output_hash_match"]
            for case in cases.values()
        ),
        "pool_tail_mod_0_1_2_3_measured": {
            cases[str(context)]["context_mod_index_kpool"]
            for context in POOL_TAIL_CONTEXTS
        }
        == {0, 1, 2, 3},
        "selected_sparse_attention_width_bounded": len(sparse_widths) == 1,
        "context_256k_completed_without_oom": "262144" in cases,
        "runtime_server_apc_admission_unchanged": (
            GROUPED_MIN_ROUTES == 256 and DEFAULT_MAX_PROMPT_TOKENS == 256
        ),
    }
    acceptance["accepted"] = all(acceptance.values())
    output = {
        "schema": "glm53-long-context-dsa-decode-frontier-v1",
        "date": date.today().isoformat(),
        "machine": "Apple M3 Ultra, 80-core GPU, 512 GB unified memory",
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "nope_cache_abi": NOPE_DSA_CACHE_ABI,
        "layer": args.layer,
        "qk_rope_head_dim": report.qk_rope_head_dim,
        "kv_lora_rank": report.kv_lora_rank,
        "index_topk": attention.indexer.index_topk,
        "index_kpool": attention.indexer.index_kpool,
        "primary_contexts": list(PRIMARY_CONTEXTS),
        "pool_tail_contexts": list(POOL_TAIL_CONTEXTS),
        "measurement_contract": {
            "cache_source": "deterministic latent/indexer state; no long prefill",
            "query_tokens": 1,
            "warmups": args.warmups,
            "measurements": args.repeats,
            "phase_timings_synchronize_each_boundary": True,
            "unsynchronized_end_to_end_syncs_only_final_output": True,
            "pool_update_includes_current_key_gate_projection_and_pooling": True,
            "score_includes_query projection and all-pool score": True,
            "attention_includes_embed_q, SDPA, unembed_out, o_proj": True,
        },
        "transition": transition,
        "cases": cases,
        "performance_kpi": {
            "operator_metric": "latency_ms_at_2049 / latency_ms_at_context",
            "informational_only": True,
            "future_full_model_decode_retention_target": 0.80,
            "future_minimum_decode_tps_at_256k": 15.0,
        },
        "next_implementation_branch": _next_candidate(cases),
        "roadmap": {
            "dsa_prefill_chunk_sizes": [512, 1024, 2048, 4096, 8192],
            "shared_metadata": "page table and row metadata may be shared; top-k is per-layer",
            "shared_row_plan_abi": "not implemented and not present in cache ABI",
            "state_safety": "idle/dummy forward must not mutate KDA/DSA state",
            "external_drafter": {
                "policy": "share target MLA/KV state; avoid an independent drafter KV pool",
                "benchmark": "acceptance_by_position[0..k-1]",
                "first_grouped_recheck_routes": 64,
            },
        },
        "runtime_policy": {
            "default_backend": "direct",
            "packed_grouped_prefill_candidate_stopped": True,
            "prompt_limit": DEFAULT_MAX_PROMPT_TOKENS,
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
                "steady_256k_ms": cases["262144"]["modes"][
                    "steady_incremental"
                ]["unsynchronized_end_to_end_ms"],
                "steady_256k_retention": cases["262144"]["modes"][
                    "steady_incremental"
                ]["sparse_path_retention"],
                "next_candidate": output["next_implementation_branch"],
            },
            indent=2,
        )
    )
    return 0 if acceptance["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
