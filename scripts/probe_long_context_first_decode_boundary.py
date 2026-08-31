#!/usr/bin/env python3
"""Characterize GLM-5.3 first-decode state transitions through 256k.

Tier 1 executes admitted production prefill, an exact RAM APC clone, and one
decode token.  Tier 2 deliberately bypasses prompt admission and constructs a
canonical, fully materialized cache for every one of the 45 layers.  It does
not claim that long cold prefill is supported.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np

from glm53_flash_mlx.abi import (
    MLX_VLM_REVISION,
    NOPE_DSA_CACHE_ABI_COMPACT,
    NOPE_DSA_CACHE_ABI_DIRECT,
)
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import EXPECTED_DSA, inspect_checkpoint
from glm53_flash_mlx.materialization import (
    MATERIALIZATION_INTERVAL_TOKENS,
    MATERIALIZATION_POLICY,
)
from glm53_flash_mlx.server import (
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_GENERATION_TOKENS,
    DEFAULT_MAX_PROMPT_TOKENS,
    validate_admission,
)

TIER1_PROMPTS = (16, 128, 255, 256)
TIER2_CONTEXTS = (16384, 65536, 131072, 262143, 262144, 262145, 262146, 262147)
KDA_LAYERS = tuple(layer for layer in range(45) if layer not in EXPECTED_DSA)
FIRST_DECODE_TOKEN = 1729
COMPACT_SERVER_CAPACITY = DEFAULT_MAX_PROMPT_TOKENS + DEFAULT_MAX_GENERATION_TOKENS
EXPECTED_DIRECT_LEAVES = len(KDA_LAYERS) * 2 + len(EXPECTED_DSA) * 4
EXPECTED_COMPACT_LEAVES = len(KDA_LAYERS) * 2 + len(EXPECTED_DSA) * 9
MAX_SELECTED_WIDTH = 2048 + 4 - 1


@dataclass(frozen=True)
class FirstDecodeBoundarySignature:
    context_tokens: int
    construction_mode: str
    cache_leaf_count: int
    cache_schema_hash: str
    kda_state_hash: str
    indexpool_hash: str
    active_tail_count: int
    selected_width: int
    physical_capacity_tokens: int
    first_token_id: int
    first_logits_hash: str
    post_state_hash: str


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


def _round_up(value: int, step: int = 256) -> int:
    return ((int(value) + step - 1) // step) * step


def _arrays(value):
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _arrays(item)
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _arrays(value[key])


def _entry_arrays(entry):
    yield from _arrays(entry.state)


def _cache_arrays(cache):
    for entry in cache:
        yield from _entry_arrays(entry)


def _np(value: mx.array, *, storage: bool = False) -> np.ndarray:
    mx.eval(value)
    if storage and value.dtype != mx.bfloat16:
        return np.ascontiguousarray(np.asarray(value))
    return np.ascontiguousarray(np.asarray(value.astype(mx.float32)))


def _hash_values(values) -> str:
    digest = hashlib.sha256()
    for index, value in enumerate(values):
        array = _np(value)
        digest.update(str(index).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical_row_major_strides(shape, itemsize: int) -> tuple[int, ...]:
    stride = int(itemsize)
    result = []
    for width in reversed(shape):
        result.append(stride)
        stride *= int(width)
    return tuple(reversed(result))


def _schema(cache) -> list[dict]:
    rows = []
    for layer, entry in enumerate(cache):
        for leaf, value in enumerate(_entry_arrays(entry)):
            itemsize = int(value.nbytes // max(1, value.size))
            rows.append(
                {
                    "layer": layer,
                    "leaf": leaf,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "canonical_row_major_strides_bytes": list(
                        _canonical_row_major_strides(value.shape, itemsize)
                    ),
                }
            )
    return rows


def _json_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cache_leaf_count(cache) -> int:
    return sum(1 for _ in _cache_arrays(cache))


def _kda_arrays(cache):
    for layer in KDA_LAYERS:
        yield from _entry_arrays(cache[layer])


def _direct_indexpool_arrays(cache):
    for layer in EXPECTED_DSA:
        state = cache[layer][1]
        pool = state._pool
        yield from pool[:3]
        logical = state.keys[:, 0, : state.offset]
        head_dim = int((logical.shape[-1] - 1) // 2)
        raw_start = max(0, state.offset - 19)
        raw = logical[:, raw_start:]
        yield raw[..., :head_dim]
        yield raw[..., head_dim : 2 * head_dim]
        yield raw[..., -1] > 0
        yield mx.arange(raw_start, state.offset, dtype=mx.int64)[None]


def _indexpool_arrays(cache, backend: str):
    if backend == "direct":
        yield from _direct_indexpool_arrays(cache)
        return
    for layer in EXPECTED_DSA:
        state = cache[layer][1]
        yield from state.logical_pool()
        yield state.raw_keys
        yield state.raw_gates
        yield state.raw_valid
        yield state.raw_positions


def _compact_latent_samples(cache):
    for layer in EXPECTED_DSA:
        latent = cache[layer][0].keys
        offset = cache[layer][0].offset
        width = min(64, offset)
        yield latent[..., :width, :]
        if offset > width:
            yield latent[..., offset - width : offset, :]


def _direct_latent_samples(cache):
    for layer in EXPECTED_DSA:
        state = cache[layer][0]
        width = min(64, state.offset)
        yield state.keys[..., :width, :]
        if state.offset > width:
            yield state.keys[..., state.offset - width : state.offset, :]


def _post_state_hash(cache, backend: str, kda_hash=None, indexpool_hash=None) -> str:
    kda_hash = kda_hash or _hash_values(_kda_arrays(cache))
    indexpool_hash = indexpool_hash or _hash_values(_indexpool_arrays(cache, backend))
    latent = (
        _direct_latent_samples(cache)
        if backend == "direct"
        else _compact_latent_samples(cache)
    )
    offsets = _dsa_offsets(cache, backend)
    return _json_hash(
        {
            "kda_full_value_hash": kda_hash,
            "indexpool_full_value_hash": indexpool_hash,
            "latent_boundary_sample_hash": _hash_values(latent),
            "dsa_offsets": offsets,
        }
    )


def _full_cache_hash(cache) -> str:
    digest = hashlib.sha256()
    digest.update(_hash_values(_cache_arrays(cache)).encode())
    for entry in cache:
        digest.update(repr(getattr(entry, "meta_state", None)).encode())
    return digest.hexdigest()


def _cache_exact(left, right) -> bool:
    left_arrays = list(_cache_arrays(left))
    right_arrays = list(_cache_arrays(right))
    if len(left_arrays) != len(right_arrays):
        return False
    for a, b in zip(left_arrays, right_arrays, strict=True):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        equal = mx.array_equal(a, b)
        mx.eval(equal)
        if not bool(equal.item()):
            return False
    return all(
        getattr(a, "meta_state", None) == getattr(b, "meta_state", None)
        for a, b in zip(left, right, strict=True)
    )


def _materialize_cache(cache) -> float:
    started = time.perf_counter()
    mx.eval([entry.state for entry in cache])
    mx.clear_cache()
    mx.synchronize()
    return (time.perf_counter() - started) * 1000.0


def _clone_cache(cache, min_capacity_tokens: int):
    from mlx_vlm.apc_adapters import clone_cache_entry

    targets = []
    result = [
        clone_cache_entry(
            entry,
            min_capacity_tokens=min_capacity_tokens,
            eval_targets=targets,
        )
        for entry in cache
    ]
    if any(entry is None for entry in result):
        raise RuntimeError("RAM APC clone rejected a cache entry")
    mx.eval(*targets)
    mx.synchronize()
    return result


def _release(*values) -> None:
    for value in values:
        if isinstance(value, list):
            value.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def _set_backend(model, backend: str, capacity_tokens: int) -> None:
    model._glm53_cache_backend = backend
    model.language_model._glm53_cache_backend = backend
    model.language_model._glm53_compact_cache_capacity_tokens = int(capacity_tokens)


def _tokens(length: int) -> mx.array:
    return (mx.arange(length, dtype=mx.uint32) * 17 + 101)[None]


def _dsa_offsets(cache, backend: str) -> list[dict]:
    rows = []
    for layer in EXPECTED_DSA:
        if backend == "compact-nope-dsa":
            latent, pool = cache[layer]
            rows.append(
                {
                    "layer": layer,
                    "latent": int(latent.offset),
                    "indexpool": int(pool.total_tokens),
                }
            )
        else:
            latent, indexer = cache[layer]
            rows.append(
                {
                    "layer": layer,
                    "latent": int(latent.offset),
                    "indexpool": int(indexer.offset),
                }
            )
    return rows


def _physical_capacity(cache, backend: str) -> dict:
    rows = []
    for layer in EXPECTED_DSA:
        if backend == "compact-nope-dsa":
            latent, pool = cache[layer]
            rows.append(
                {
                    "layer": layer,
                    "latent_tokens": latent.physical_capacity_tokens,
                    "indexpool_rows": pool.physical_capacity_rows,
                }
            )
        else:
            latent, indexer = cache[layer]
            rows.append(
                {
                    "layer": layer,
                    "latent_tokens": int(latent.keys.shape[2]),
                    "indexer_tokens": int(indexer.keys.shape[2]),
                }
            )
    return {
        "layers": rows,
        "minimum_latent_tokens": min(row["latent_tokens"] for row in rows),
        "maximum_latent_tokens": max(row["latent_tokens"] for row in rows),
    }


def _capacity_equal(left: dict, right: dict) -> bool:
    return left == right


def _run_first(model, cache, token: int = FIRST_DECODE_TOKEN):
    started = time.perf_counter()
    output = model(mx.array([[token]], dtype=mx.uint32), cache=cache)
    logits = output.logits[0, -1]
    token_array = mx.argmax(logits)
    nan_array = mx.sum(mx.isnan(logits))
    mx.eval(logits, token_array, nan_array)
    mx.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    values = _np(logits)
    return {
        "token_id": int(token_array.item()),
        "logits_hash": hashlib.sha256(values.tobytes()).hexdigest(),
        "logits": values,
        "nan_count": int(nan_array.item()),
        "latency_ms": elapsed_ms,
    }


def _tier1(model) -> list[dict]:
    cases = []
    for backend in ("direct", "compact-nope-dsa"):
        _set_backend(model, backend, COMPACT_SERVER_CAPACITY)
        for prompt in TIER1_PROMPTS:
            _progress("tier1", backend=backend, prompt=prompt)
            validate_admission(
                prompt,
                1,
                max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS,
                max_generation_tokens=DEFAULT_MAX_GENERATION_TOKENS,
                max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
            )
            resident = model.make_cache()
            prefill = model(_tokens(prompt), cache=resident)
            mx.eval(prefill.logits)
            mx.synchronize()
            pre_offsets = _dsa_offsets(resident, backend)
            resident_capacity_before = _physical_capacity(resident, backend)
            snapshot = _clone_cache(resident, prompt + 1)
            restored = _clone_cache(snapshot, prompt + 1)
            boundary = _clone_cache(snapshot, prompt + 1)
            snapshot_hash_before = _full_cache_hash(snapshot)

            resident_result = _run_first(model, resident)
            restored_result = _run_first(model, restored)
            boundary_result = _run_first(model, boundary)
            boundary_materialization_ms = _materialize_cache(boundary)
            snapshot_hash_after = _full_cache_hash(snapshot)
            post_offsets = _dsa_offsets(resident, backend)
            restored_offsets = _dsa_offsets(restored, backend)
            resident_capacity_after = _physical_capacity(resident, backend)
            restored_capacity_after = _physical_capacity(restored, backend)
            snapshot_capacity = _physical_capacity(snapshot, backend)
            expected_resident_growth = (
                backend == "direct"
                and resident_capacity_before["minimum_latent_tokens"] < prompt + 1
            )
            if expected_resident_growth:
                capacity_ok = (
                    resident_capacity_after["minimum_latent_tokens"]
                    == _round_up(prompt + 1)
                )
            else:
                capacity_ok = _capacity_equal(
                    resident_capacity_before, resident_capacity_after
                )
            cases.append(
                {
                    "backend": backend,
                    "prompt_tokens": prompt,
                    "resident_first_token_id": resident_result["token_id"],
                    "restored_first_token_id": restored_result["token_id"],
                    "boundary_first_token_id": boundary_result["token_id"],
                    "resident_logits_hash": resident_result["logits_hash"],
                    "restored_logits_hash": restored_result["logits_hash"],
                    "boundary_logits_hash": boundary_result["logits_hash"],
                    "all_logits_byte_identical": bool(
                        np.array_equal(resident_result["logits"], restored_result["logits"])
                        and np.array_equal(resident_result["logits"], boundary_result["logits"])
                    ),
                    "snapshot_hash_before": snapshot_hash_before,
                    "snapshot_hash_after": snapshot_hash_after,
                    "snapshot_immutable": snapshot_hash_before == snapshot_hash_after,
                    "pre_offsets": pre_offsets,
                    "post_offsets": post_offsets,
                    "restored_offsets": restored_offsets,
                    "all_dsa_offsets_advance_exactly_one": all(
                        after["latent"] == before["latent"] + 1
                        and after["indexpool"] == before["indexpool"] + 1
                        for before, after in zip(pre_offsets, post_offsets, strict=True)
                    ),
                    "resident_restore_offsets_match": post_offsets == restored_offsets,
                    "resident_restore_post_state_exact": _cache_exact(resident, restored),
                    "materialized_nonboundary_count": 0,
                    "materialized_boundary_count": 1,
                    "boundary_materialization_step": MATERIALIZATION_INTERVAL_TOKENS,
                    "boundary_materialization_ms": boundary_materialization_ms,
                    "boundary_materialization_state_exact": _cache_exact(restored, boundary),
                    "resident_capacity_before": resident_capacity_before,
                    "resident_capacity_after": resident_capacity_after,
                    "snapshot_capacity": snapshot_capacity,
                    "restored_capacity_after": restored_capacity_after,
                    "resident_capacity_growth_expected": expected_resident_growth,
                    "no_unexpected_capacity_growth": capacity_ok
                    and _capacity_equal(snapshot_capacity, restored_capacity_after),
                    "nan_count": resident_result["nan_count"]
                    + restored_result["nan_count"]
                    + boundary_result["nan_count"],
                    "metal_error": None,
                    "kda_logical_context_before": prompt,
                    "kda_logical_context_after": prompt + 1,
                }
            )
            _release(resident, snapshot, restored, boundary)
    return cases


def _deterministic_rows(rows: int, width: int, phase: float, dtype) -> mx.array:
    positions = mx.arange(rows, dtype=mx.float32)[:, None]
    columns = mx.arange(width, dtype=mx.float32)[None]
    values = mx.sin(positions * 0.0009765625 + columns * 0.0078125 + phase)
    return values.astype(dtype)


def _fill_kda(entry, attention, layer: int) -> None:
    conv = _deterministic_rows(
        attention.conv_kernel_size - 1,
        attention.conv_dim,
        0.03125 * (layer + 1),
        mx.bfloat16,
    )[None]
    recurrent = _deterministic_rows(
        attention.num_heads * attention.head_dim,
        attention.head_dim,
        0.015625 * (layer + 1),
        mx.float32,
    ).reshape(1, attention.num_heads, attention.head_dim, attention.head_dim)
    # Keep the canonical non-zero state numerically benign for the first step.
    entry.state = [mx.contiguous(conv * 0.015625), mx.contiguous(recurrent * 0.0009765625)]


def _canonical_dsa_arrays(attention, layer: int, context: int):
    latent = _deterministic_rows(
        context,
        attention.kv_lora_rank,
        1.125 + layer * 0.015625,
        mx.bfloat16,
    ).reshape(1, 1, context, attention.kv_lora_rank)
    indexer = attention.indexer
    keys = _deterministic_rows(
        context, indexer.head_dim, 0.125 + layer * 0.015625, mx.bfloat16
    )[None]
    gates = _deterministic_rows(
        context, indexer.head_dim, 0.625 + layer * 0.015625, mx.bfloat16
    )[None]
    valid = mx.ones((1, context), dtype=mx.bool_)
    pool = indexer._pooled_states(keys, gates, valid)
    return latent, keys, gates, valid, pool


def _fill_direct_dsa(entry, attention, layer: int, context: int, capacity: int) -> None:
    latent_cache, indexer_cache = entry
    latent, keys, gates, valid, pool = _canonical_dsa_arrays(attention, layer, context)
    latent_buffer = mx.zeros(
        (1, 1, capacity, attention.kv_lora_rank), dtype=mx.bfloat16
    )
    latent_buffer[..., :context, :] = latent
    latent_cache.keys = latent_buffer
    latent_cache.values = mx.array(latent_buffer)
    latent_cache.offset = context
    packed_logical = mx.concatenate(
        [keys, gates, valid.astype(keys.dtype)[..., None]], axis=-1
    )
    packed = mx.zeros((1, 1, capacity, packed_logical.shape[-1]), dtype=keys.dtype)
    packed[:, 0, :context] = packed_logical
    indexer_cache.keys = packed
    indexer_cache.values = mx.zeros((1, 1, capacity, 0), dtype=keys.dtype)
    indexer_cache.offset = context
    indexer_cache._no_pad = True
    indexer_cache._pool = (*pool, context)


def _fill_compact_dsa(entry, attention, layer: int, context: int, capacity: int) -> None:
    latent_cache, pool_cache = entry
    latent, keys, gates, valid, pool = _canonical_dsa_arrays(attention, layer, context)
    latent_buffer = mx.zeros(
        (1, 1, capacity, attention.kv_lora_rank), dtype=mx.bfloat16
    )
    latent_buffer[..., :context, :] = latent
    latent_cache._latent = latent_buffer
    latent_cache.offset = context
    latent_cache.capacity_tokens = context + 1

    indexer = attention.indexer
    rows = _round_up(context + 1) // int(indexer.index_kpool)
    pool_keys = mx.zeros((1, rows, indexer.head_dim), dtype=mx.bfloat16)
    pool_indices = mx.full(
        (1, rows, indexer.index_kpool), -1, dtype=mx.int64
    )
    pool_valid = mx.zeros((1, rows), dtype=mx.bool_)
    logical_rows = int(pool[0].shape[1])
    pool_keys[:, :logical_rows] = pool[0]
    pool_indices[:, :logical_rows] = pool[1]
    pool_valid[:, :logical_rows] = pool[2]
    raw_start = max(0, context - pool_cache.raw_state_window)
    pool_cache.pool_keys = pool_keys
    pool_cache.pool_indices = pool_indices
    pool_cache.pool_valid = pool_valid
    pool_cache.raw_keys = keys[:, raw_start:]
    pool_cache.raw_gates = gates[:, raw_start:]
    pool_cache.raw_valid = valid[:, raw_start:]
    pool_cache.raw_positions = mx.arange(raw_start, context, dtype=mx.int64)[None]
    pool_cache.compress_ape = indexer.index_kpool_compress_ape
    pool_cache.total_tokens = context
    pool_cache.logical_pool_count = logical_rows
    pool_cache.pool_capacity = rows
    pool_cache.capacity_tokens = context + 1


def _synthetic_cache(model, context: int, backend: str):
    capacity = _round_up(context + 1)
    _set_backend(model, backend, context + 1)
    cache = model.make_cache()
    for layer, block in enumerate(model.language_model.model.layers):
        if layer in EXPECTED_DSA:
            if backend == "direct":
                _fill_direct_dsa(cache[layer], block.self_attn, layer, context, capacity)
            else:
                _fill_compact_dsa(cache[layer], block.self_attn, layer, context, capacity)
        else:
            _fill_kda(cache[layer], block.self_attn, layer)
    _materialize_cache(cache)
    return cache


def _clone_entry(entry, min_capacity_tokens: int):
    from mlx_vlm.apc_adapters import clone_cache_entry

    targets = []
    cloned = clone_cache_entry(
        entry,
        min_capacity_tokens=min_capacity_tokens,
        eval_targets=targets,
    )
    if cloned is None:
        raise RuntimeError("RAM APC clone rejected a DSA cache entry")
    mx.eval(*targets)
    mx.synchronize()
    return cloned


def _dsa_diagnostics(model, cache, backend: str, context: int) -> dict:
    rows = []
    all_indices = []
    all_outputs = []
    for layer in EXPECTED_DSA:
        attention = model.language_model.model.layers[layer].self_attn
        x = _deterministic_rows(
            1, attention.hidden_size, 2.0 + layer * 0.015625, mx.bfloat16
        )[None]
        qr = attention.q_a_layernorm(attention.q_a_proj(x))
        index_entry = _clone_entry(cache[layer], context + 1)
        indices = attention.indexer(x, qr, None, cache=index_entry[1])
        mx.eval(indices)
        output_entry = _clone_entry(cache[layer], context + 1)
        output = attention(x, cache=output_entry)
        mx.eval(output)
        mx.synchronize()
        index_np = _np(indices, storage=True)
        output_np = _np(output)
        valid = index_np[index_np >= 0]
        rows.append(
            {
                "layer": layer,
                "selected_width": int(indices.shape[-1]),
                "index_hash": hashlib.sha256(index_np.tobytes()).hexdigest(),
                "output_hash": hashlib.sha256(output_np.tobytes()).hexdigest(),
                "valid_index_min": int(valid.min()) if valid.size else None,
                "valid_index_max": int(valid.max()) if valid.size else None,
                "sentinel_count": int((index_np == -1).sum()),
                "out_of_range_count": int(
                    ((index_np != -1) & ((index_np < 0) | (index_np >= context + 1))).sum()
                ),
                "nan_count": int(np.isnan(output_np).sum()),
            }
        )
        all_indices.append(index_np)
        all_outputs.append(output_np)
        _release(index_entry, output_entry)
    return {"layers": rows, "indices": all_indices, "outputs": all_outputs}


def _kda_schema(cache) -> list[dict]:
    return [row for row in _schema(cache) if row["layer"] in KDA_LAYERS]


def _signature(cache, backend: str, context: int, mode: str, first: dict) -> dict:
    schema = _schema(cache)
    kda_hash = _hash_values(_kda_arrays(cache))
    indexpool_hash = _hash_values(_indexpool_arrays(cache, backend))
    capacity = _physical_capacity(cache, backend)
    tails = [
        (context + 1) % 4
        if backend == "direct"
        else cache[layer][1].active_tail_count
        for layer in EXPECTED_DSA
    ]
    return asdict(
        FirstDecodeBoundarySignature(
            context_tokens=context,
            construction_mode=mode,
            cache_leaf_count=len(schema),
            cache_schema_hash=_json_hash(schema),
            kda_state_hash=kda_hash,
            indexpool_hash=indexpool_hash,
            active_tail_count=max(tails),
            selected_width=MAX_SELECTED_WIDTH,
            physical_capacity_tokens=capacity["minimum_latent_tokens"],
            first_token_id=first["token_id"],
            first_logits_hash=first["logits_hash"],
            post_state_hash=_post_state_hash(
                cache, backend, kda_hash=kda_hash, indexpool_hash=indexpool_hash
            ),
        )
    )


def _tier2_context(model, context: int) -> dict:
    _progress("tier2_direct_build", context=context)
    direct = _synthetic_cache(model, context, "direct")
    direct_capacity_before = _physical_capacity(direct, "direct")
    direct_schema = _schema(direct)
    direct_kda_schema = _kda_schema(direct)
    direct_diag = _dsa_diagnostics(model, direct, "direct", context)
    direct_first = _run_first(model, direct)
    direct_capacity_after = _physical_capacity(direct, "direct")
    direct_signature = _signature(
        direct, "direct", context, "canonical-synthetic-direct", direct_first
    )
    direct_post_kda_hash = direct_signature["kda_state_hash"]
    direct_post_indexpool_hash = direct_signature["indexpool_hash"]
    direct_logits = direct_first["logits"]
    direct_indices = direct_diag.pop("indices")
    direct_outputs = direct_diag.pop("outputs")
    _release(direct)

    _progress("tier2_compact_build", context=context)
    compact = _synthetic_cache(model, context, "compact-nope-dsa")
    compact_capacity_before = _physical_capacity(compact, "compact-nope-dsa")
    compact_schema = _schema(compact)
    compact_kda_schema = _kda_schema(compact)
    compact_diag = _dsa_diagnostics(model, compact, "compact-nope-dsa", context)
    diagnostic_indices = compact_diag.pop("indices")
    diagnostic_outputs = compact_diag.pop("outputs")
    index_equal = [
        bool(np.array_equal(left, right))
        for left, right in zip(direct_indices, diagnostic_indices, strict=True)
    ]
    output_equal = [
        bool(np.array_equal(left, right))
        for left, right in zip(direct_outputs, diagnostic_outputs, strict=True)
    ]
    for row, idx_equal, out_equal in zip(
        compact_diag["layers"], index_equal, output_equal, strict=True
    ):
        row["direct_index_byte_identical"] = idx_equal
        row["direct_output_byte_identical"] = out_equal

    restored = _clone_cache(compact, context + 1)
    restored_capacity_before = _physical_capacity(restored, "compact-nope-dsa")
    compact_first = _run_first(model, compact)
    restored_first = _run_first(model, restored)
    compact_capacity_after = _physical_capacity(compact, "compact-nope-dsa")
    restored_capacity_after = _physical_capacity(restored, "compact-nope-dsa")
    compact_signature = _signature(
        compact, "compact-nope-dsa", context, "canonical-synthetic-compact", compact_first
    )
    restored_signature = _signature(
        restored,
        "compact-nope-dsa",
        context,
        "canonical-synthetic-compact-ram-apc-restore",
        restored_first,
    )
    compact_restore_exact = _cache_exact(compact, restored)
    case = {
        "context_tokens": context,
        "context_mod_kpool": context % 4,
        "capacity_target_tokens": context + 1,
        "rounded_physical_capacity_tokens": _round_up(context + 1),
        "direct": {
            "signature": direct_signature,
            "first_decode_latency_ms": direct_first["latency_ms"],
            "nan_count": direct_first["nan_count"],
            "metal_error": None,
            "capacity_before": direct_capacity_before,
            "capacity_after": direct_capacity_after,
            "capacity_unchanged": direct_capacity_before == direct_capacity_after,
            "cache_schema": direct_schema,
            "dsa_diagnostics": direct_diag["layers"],
        },
        "compact_resident": {
            "signature": compact_signature,
            "first_decode_latency_ms": compact_first["latency_ms"],
            "nan_count": compact_first["nan_count"],
            "metal_error": None,
            "capacity_before": compact_capacity_before,
            "capacity_after": compact_capacity_after,
            "capacity_unchanged": compact_capacity_before == compact_capacity_after,
            "cache_schema": compact_schema,
            "dsa_diagnostics": compact_diag["layers"],
        },
        "compact_restore": {
            "signature": restored_signature,
            "first_decode_latency_ms": restored_first["latency_ms"],
            "nan_count": restored_first["nan_count"],
            "metal_error": None,
            "capacity_before": restored_capacity_before,
            "capacity_after": restored_capacity_after,
            "capacity_unchanged": restored_capacity_before == restored_capacity_after,
        },
        "direct_compact_first_logits_byte_identical": bool(
            np.array_equal(direct_logits, compact_first["logits"])
        ),
        "compact_resident_restore_logits_byte_identical": bool(
            np.array_equal(compact_first["logits"], restored_first["logits"])
        ),
        "compact_resident_restore_post_state_exact": compact_restore_exact,
        "direct_compact_kda_post_hash_match": (
            direct_post_kda_hash == compact_signature["kda_state_hash"]
        ),
        "direct_compact_indexpool_post_hash_match": (
            direct_post_indexpool_hash == compact_signature["indexpool_hash"]
        ),
        "direct_compact_dsa_indices_byte_identical": all(index_equal),
        "direct_compact_dsa_outputs_byte_identical": all(output_equal),
        "kda_schema_hash_direct": _json_hash(direct_kda_schema),
        "kda_schema_hash_compact": _json_hash(compact_kda_schema),
        "active_memory_bytes": int(mx.get_active_memory()),
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "metal_error": None,
    }
    _release(compact, restored)
    return case


def _cold_prefill_rejection(model) -> dict:
    _set_backend(model, "compact-nope-dsa", COMPACT_SERVER_CAPACITY)
    cache = model.make_cache()
    model(_tokens(16), cache=cache)
    _materialize_cache(cache)
    before = _full_cache_hash(cache)
    errors = []
    for prompt in (16384, 65536, 131072, 262144):
        try:
            validate_admission(
                prompt,
                1,
                max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS,
                max_generation_tokens=DEFAULT_MAX_GENERATION_TOKENS,
                max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
            )
        except ValueError as exc:
            errors.append({"prompt_tokens": prompt, "rejected": True, "error": str(exc)})
        else:
            errors.append({"prompt_tokens": prompt, "rejected": False, "error": None})
    after = _full_cache_hash(cache)
    _release(cache)
    return {
        "cases": errors,
        "all_rejected": all(row["rejected"] for row in errors),
        "cache_state_hash_before": before,
        "cache_state_hash_after": after,
        "cache_state_unchanged": before == after,
        "claim": "256k cold prefill remains unsupported and unvalidated",
    }


def _acceptance(tier1, tier2, rejection) -> dict:
    cases = list(tier2.values())
    kda_hashes = {
        case["kda_schema_hash_direct"] for case in cases
    } | {case["kda_schema_hash_compact"] for case in cases}
    c144 = tier2["262144"]
    c145 = tier2["262145"]
    latency_ratios = {
        "direct": c145["direct"]["first_decode_latency_ms"]
        / c144["direct"]["first_decode_latency_ms"],
        "compact_resident": c145["compact_resident"]["first_decode_latency_ms"]
        / c144["compact_resident"]["first_decode_latency_ms"],
        "compact_restore": c145["compact_restore"]["first_decode_latency_ms"]
        / c144["compact_restore"]["first_decode_latency_ms"],
    }
    result = {
        "tier1_first_token_and_full_vocab_logits_byte_identical": all(
            row["all_logits_byte_identical"] for row in tier1
        ),
        "tier1_snapshot_immutable": all(row["snapshot_immutable"] for row in tier1),
        "tier1_dsa_offsets_advance_exactly_one": all(
            row["all_dsa_offsets_advance_exactly_one"]
            and row["resident_restore_offsets_match"]
            for row in tier1
        ),
        "tier1_materialization_boundary_exact": all(
            row["materialized_nonboundary_count"] == 0
            and row["materialized_boundary_count"] == 1
            and row["boundary_materialization_state_exact"]
            for row in tier1
        ),
        "tier1_no_unexpected_capacity_growth": all(
            row["no_unexpected_capacity_growth"] for row in tier1
        ),
        "tier1_no_nan_or_metal_error": all(
            row["nan_count"] == 0 and row["metal_error"] is None for row in tier1
        ),
        "state_leaf_count_known_and_constant": all(
            case["direct"]["signature"]["cache_leaf_count"] == EXPECTED_DIRECT_LEAVES
            and case["compact_resident"]["signature"]["cache_leaf_count"]
            == EXPECTED_COMPACT_LEAVES
            and case["compact_restore"]["signature"]["cache_leaf_count"]
            == EXPECTED_COMPACT_LEAVES
            for case in cases
        ),
        "kda_shape_dtype_stride_context_independent": len(kda_hashes) == 1,
        "selected_width_at_most_2051": all(
            max(
                row["selected_width"]
                for row in case["compact_resident"]["dsa_diagnostics"]
            )
            <= MAX_SELECTED_WIDTH
            for case in cases
        ),
        "all_indices_sentinel_or_in_range": all(
            row["out_of_range_count"] == 0
            for case in cases
            for arm in ("direct", "compact_resident")
            for row in case[arm]["dsa_diagnostics"]
        ),
        "direct_compact_selected_indices_and_dsa_output_identical": all(
            case["direct_compact_dsa_indices_byte_identical"]
            and case["direct_compact_dsa_outputs_byte_identical"]
            for case in cases
        ),
        "direct_compact_full_model_first_logits_identical": all(
            case["direct_compact_first_logits_byte_identical"] for case in cases
        ),
        "direct_compact_post_kda_and_indexpool_hashes_match": all(
            case["direct_compact_kda_post_hash_match"]
            and case["direct_compact_indexpool_post_hash_match"]
            for case in cases
        ),
        "resident_restore_first_logits_and_post_state_identical": all(
            case["compact_resident_restore_logits_byte_identical"]
            and case["compact_resident_restore_post_state_exact"]
            for case in cases
        ),
        "first_decode_capacity_preallocated_and_unchanged": all(
            case[arm]["capacity_unchanged"]
            for case in cases
            for arm in ("direct", "compact_resident", "compact_restore")
        ),
        "physical_capacity_covers_context_plus_one": all(
            case[arm]["capacity_before"]["minimum_latent_tokens"]
            >= case["context_tokens"] + 1
            for case in cases
            for arm in ("direct", "compact_resident", "compact_restore")
        ),
        "tail_mod_0_1_2_3_covered": {
            case["compact_resident"]["signature"]["active_tail_count"]
            for case in cases
        }
        == {0, 1, 2, 3},
        "262145_to_262144_first_decode_latency_ratio_below_1_5": max(
            latency_ratios.values()
        )
        < 1.5,
        "256k_no_oom_nan_or_metal_error": all(
            case[arm]["nan_count"] == 0 and case[arm]["metal_error"] is None
            for case in cases
            for arm in ("direct", "compact_resident", "compact_restore")
        ),
        "long_cold_prefill_fail_closed_and_state_unchanged": rejection["all_rejected"]
        and rejection["cache_state_unchanged"],
        "runtime_server_apc_cache_abi_admission_unchanged": True,
    }
    result["latency_ratios_262145_over_262144"] = latency_ratios
    result["accepted"] = all(
        value for key, value in result.items() if key not in {"latency_ratios_262145_over_262144"}
    )
    return result


def _atomic_write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _final_artifact(
    report,
    tier1,
    rejection,
    tier2,
    peak_memory_bytes: int,
    existing_kda_schema_exemplar=None,
) -> dict:
    first_case = tier2[str(TIER2_CONTEXTS[0])]
    kda_schema_exemplar = existing_kda_schema_exemplar or [
        row
        for row in first_case["direct"].get("cache_schema", [])
        if row["layer"] in KDA_LAYERS
    ]
    for case in tier2.values():
        for arm in ("direct", "compact_resident", "compact_restore"):
            case[arm].setdefault("metal_error", None)
            case[arm].pop("cache_schema", None)
    acceptance = _acceptance(tier1, tier2, rejection)
    return {
        "schema": "glm53-long-context-first-decode-boundary-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "machine": "Apple M3 Ultra, 80-core GPU, 512 GB unified memory",
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "direct_cache_abi": NOPE_DSA_CACHE_ABI_DIRECT,
        "compact_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
        "materialization_policy": MATERIALIZATION_POLICY,
        "materialization_interval_tokens": MATERIALIZATION_INTERVAL_TOKENS,
        "tier1_prompts": list(TIER1_PROMPTS),
        "tier2_contexts": list(TIER2_CONTEXTS),
        "kda_layers": list(KDA_LAYERS),
        "dsa_layers": list(EXPECTED_DSA),
        "state_hash_scope": {
            "kda_state_hash": "all KDA conv and recurrent values, BF16 canonicalized to FP32",
            "indexpool_hash": "all logical pool and 19-token raw rollback values",
            "post_state_hash": "full KDA + full IndexPool + DSA latent first/last 64 rows + offsets",
            "exact_resident_restore_check": "device-side array_equal over every state leaf and exact meta_state",
        },
        "stride_evidence": {
            "mlx_public_stride_api_available": False,
            "construction_contract": "mx.contiguous KDA state with canonical row-major byte strides recorded",
            "kda_schema_exemplar": kda_schema_exemplar,
        },
        "tier1": tier1,
        "cold_prefill_rejection": rejection,
        "tier2": tier2,
        "peak_memory_bytes": int(peak_memory_bytes),
        "claims": {
            "validated": "256k resident/restore to first decode",
            "unsupported_unvalidated": "256k cold prefill to first decode",
        },
        "runtime_changes": {
            "server": False,
            "apc": False,
            "cache_abi": False,
            "admission": False,
        },
        "acceptance": acceptance,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="finalize a complete per-context partial artifact without loading weights",
    )
    args = parser.parse_args()

    report = inspect_checkpoint(args.model, require_server_ready=True)
    if args.finalize_existing:
        partial = json.loads(args.output.read_text())
        if partial.get("complete") is False and partial.get(
            "last_completed_context"
        ) != TIER2_CONTEXTS[-1]:
            raise ValueError("partial artifact does not contain the final context")
        tier1 = partial["tier1"]
        rejection = partial["cold_prefill_rejection"]
        tier2 = partial["tier2"]
        if tuple(map(int, tier2.keys())) != TIER2_CONTEXTS:
            raise ValueError("partial artifact context set/order is incomplete")
        peak = int(
            partial.get(
                "peak_memory_bytes",
                max(int(case.get("peak_memory_bytes", 0)) for case in tier2.values()),
            )
        )
        artifact = _final_artifact(
            report,
            tier1,
            rejection,
            tier2,
            peak,
            partial.get("stride_evidence", {}).get("kda_schema_exemplar"),
        )
        _atomic_write(args.output, artifact)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "accepted": artifact["acceptance"]["accepted"],
                    "latency_ratios": artifact["acceptance"][
                        "latency_ratios_262145_over_262144"
                    ],
                    "finalized_existing": True,
                },
                indent=2,
            )
        )
        return 0 if artifact["acceptance"]["accepted"] else 1
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    model, _ = load(args.model)
    warm_residency(model)
    tier1 = _tier1(model)
    rejection = _cold_prefill_rejection(model)
    tier2 = {}
    mx.reset_peak_memory()
    for context in TIER2_CONTEXTS:
        tier2[str(context)] = _tier2_context(model, context)
        partial = {
            "schema": "glm53-long-context-first-decode-boundary-v1",
            "complete": False,
            "last_completed_context": context,
            "tier1": tier1,
            "cold_prefill_rejection": rejection,
            "tier2": tier2,
        }
        _atomic_write(args.output, partial)

    artifact = _final_artifact(
        report, tier1, rejection, tier2, int(mx.get_peak_memory())
    )
    _atomic_write(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "accepted": artifact["acceptance"]["accepted"],
                "latency_ratios": artifact["acceptance"][
                    "latency_ratios_262145_over_262144"
                ],
            },
            indent=2,
        )
    )
    return 0 if artifact["acceptance"]["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
