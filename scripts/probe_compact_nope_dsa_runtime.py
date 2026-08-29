#!/usr/bin/env python3
"""Validate the opt-in production compact NoPE DSA cache on M3 Ultra."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import time
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
from glm53_flash_mlx.nope_cache import (
    CompactIndexPoolCache,
    SingleNoPELatentCache,
)
from glm53_flash_mlx.server import DEFAULT_MAX_PROMPT_TOKENS

PROMPTS = (1, 16, 128, 256)
CONTEXTS = (2049, 8192, 16384, 32768, 131072, 262144)
RESTORE_MODS = (0, 1, 2, 3)
DECODE_STEPS = 16
WARMUP_STEPS = 2
SAMPLES = 5
RESERVE_TOKENS = 4096


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


def _np(value: mx.array, dtype=None) -> np.ndarray:
    mx.eval(value)
    array = np.ascontiguousarray(np.asarray(value))
    return array.astype(dtype, copy=False) if dtype is not None else array


def _hash(value: mx.array) -> str:
    array = _np(value.astype(mx.float32), np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _tokens(length: int) -> mx.array:
    return (mx.arange(length, dtype=mx.uint32) * 17 + 101)[None]


def _set_backend(model, backend: str) -> None:
    model._glm53_cache_backend = backend
    model.language_model._glm53_cache_backend = backend
    model.language_model._glm53_compact_cache_reserve_tokens = RESERVE_TOKENS


def _release(value) -> None:
    if isinstance(value, list):
        value.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def _trace(model, prompt_tokens: int, teacher_tokens=None):
    cache = model.make_cache()
    output = model(_tokens(prompt_tokens), cache=cache)
    mx.eval(output.logits)
    mx.synchronize()
    active_before_decode = int(mx.get_active_memory())
    hashes = []
    generated = []
    arrays = []
    for step in range(DECODE_STEPS):
        logits = output.logits[0, -1]
        arrays.append(_np(logits.astype(mx.float32), np.float32))
        hashes.append(hashlib.sha256(arrays[-1].tobytes()).hexdigest())
        token = (
            int(np.argmax(arrays[-1]))
            if teacher_tokens is None
            else int(teacher_tokens[step])
        )
        generated.append(token)
        if step + 1 < DECODE_STEPS:
            output = model(mx.array([[token]], dtype=mx.uint32), cache=cache)
    compact_entries = [cache[layer] for layer in EXPECTED_DSA]
    compact_backend = all(
        isinstance(entry[1], CompactIndexPoolCache) for entry in compact_entries
    )
    cache_evidence = {
        "single_latent_all_layers": all(
            isinstance(entry[0], SingleNoPELatentCache) for entry in compact_entries
        ),
        "compact_pool_all_layers": all(
            isinstance(entry[1], CompactIndexPoolCache) for entry in compact_entries
        ),
        "max_raw_tokens": (
            max((entry[1].raw_token_count for entry in compact_entries), default=0)
            if compact_backend
            else None
        ),
        "full_packed_history_present": any(
            hasattr(entry[1], "keys") or hasattr(entry[1], "packed_token_history")
            for entry in compact_entries
        )
        if compact_backend
        else True,
        "cache_nbytes": sum(int(entry.nbytes) for entry in cache),
        "decode_active_memory_drift_bytes": (
            int(mx.get_active_memory()) - active_before_decode
        ),
    }
    return cache, generated, hashes, arrays, cache_evidence


def _prompt_correctness(model):
    cases = []
    for prompt_tokens in PROMPTS:
        _progress("correctness", prompt_tokens=prompt_tokens, backend="direct")
        _set_backend(model, "direct")
        direct_cache, generated, direct_hashes, direct_arrays, _ = _trace(
            model, prompt_tokens
        )
        _progress("correctness", prompt_tokens=prompt_tokens, backend="compact")
        _set_backend(model, "compact-nope-dsa")
        compact_cache, compact_tokens, compact_hashes, compact_arrays, evidence = _trace(
            model, prompt_tokens, generated
        )
        equality = [
            bool(np.array_equal(left, right))
            for left, right in zip(direct_arrays, compact_arrays, strict=True)
        ]
        cases.append(
            {
                "prompt_tokens": prompt_tokens,
                "generated_tokens_match": generated == compact_tokens,
                "all_logits_byte_identical": all(equality),
                "step_logits_byte_identical": equality,
                "direct_logits_hashes": direct_hashes,
                "compact_logits_hashes": compact_hashes,
                **evidence,
            }
        )
        _release(direct_cache)
        _release(compact_cache)
    return cases


def _clone_cache(cache):
    from mlx_vlm.apc_adapters import clone_cache_entry

    targets = []
    cloned = [
        clone_cache_entry(
            entry,
            min_capacity_tokens=entry.size() + DECODE_STEPS,
            eval_targets=targets,
        )
        for entry in cache
    ]
    if any(entry is None for entry in cloned):
        raise RuntimeError("RAM APC exact clone rejected a compact cache entry")
    mx.eval(*targets)
    mx.synchronize()
    return cloned


def _ram_apc_correctness(model):
    _set_backend(model, "compact-nope-dsa")
    cases = []
    for remainder in RESTORE_MODS:
        prompt_tokens = 16 + remainder
        original = model.make_cache()
        model(_tokens(prompt_tokens), cache=original)
        restored = _clone_cache(original)
        steps = []
        for step in range(DECODE_STEPS):
            token = 2000 + step
            left = model(mx.array([[token]], dtype=mx.uint32), cache=original)
            right = model(mx.array([[token]], dtype=mx.uint32), cache=restored)
            left_hash = _hash(left.logits[0, -1])
            right_hash = _hash(right.logits[0, -1])
            steps.append(
                {
                    "step": step,
                    "logits_hash_match": left_hash == right_hash,
                    "original_hash": left_hash,
                    "restored_hash": right_hash,
                }
            )
        cases.append(
            {
                "restore_position_mod_index_kpool": remainder,
                "prompt_tokens": prompt_tokens,
                "all_continuation_hashes_match": all(
                    row["logits_hash_match"] for row in steps
                ),
                "steps": steps,
            }
        )
        _release(original)
        _release(restored)
    return cases


def _deterministic_rows(rows: int, width: int, phase: float, dtype) -> mx.array:
    positions = mx.arange(rows, dtype=mx.float32)[:, None]
    columns = mx.arange(width, dtype=mx.float32)[None]
    return mx.sin(positions * 0.0009765625 + columns * 0.0078125 + phase).astype(
        dtype
    )


def _fill_compact_dsa(entry, attention, layer_id: int, context: int) -> None:
    latent_cache, pool_cache = entry
    latent = _deterministic_rows(
        context,
        attention.kv_lora_rank,
        1.125 + layer_id * 0.015625,
        mx.bfloat16,
    ).reshape(1, 1, context, attention.kv_lora_rank)
    latent_cache.state = (latent,)
    latent_cache.meta_state = (
        str(context),
        str(RESERVE_TOKENS),
        "16",
        "256",
    )

    indexer = attention.indexer
    keys = _deterministic_rows(
        context, indexer.head_dim, 0.125 + layer_id * 0.015625, mx.bfloat16
    )[None]
    gates = _deterministic_rows(
        context, indexer.head_dim, 0.625 + layer_id * 0.015625, mx.bfloat16
    )[None]
    valid = mx.ones((1, context), dtype=mx.bool_)
    pool = indexer._pooled_states(keys, gates, valid)
    raw_start = max(0, context - pool_cache.raw_state_window)
    raw = (
        keys[:, raw_start:],
        gates[:, raw_start:],
        valid[:, raw_start:],
        mx.arange(raw_start, context, dtype=mx.int64)[None],
    )
    pool_cache.state = (*pool, *raw)
    pool_cache.meta_state = (
        str(context),
        str(pool[0].shape[1]),
        str(RESERVE_TOKENS),
        "16",
        str(indexer.index_kpool),
        str(indexer.index_topk),
        str(indexer.head_dim),
        str(int(indexer.index_kpool_always_select_tail)),
        "256",
    )
    mx.eval(
        latent_cache.keys,
        *pool_cache.dependency_arrays(),
    )
    mx.synchronize()


def _fill_direct_dsa(entry, attention, layer_id: int, context: int) -> None:
    latent_cache, indexer_cache = entry
    capacity = ((context + DECODE_STEPS + 255) // 256) * 256
    latent = mx.zeros(
        (1, 1, capacity, attention.kv_lora_rank), dtype=mx.bfloat16
    )
    logical_latent = _deterministic_rows(
        context,
        attention.kv_lora_rank,
        1.125 + layer_id * 0.015625,
        mx.bfloat16,
    ).reshape(1, 1, context, attention.kv_lora_rank)
    latent[..., :context, :] = logical_latent
    latent_cache.keys = latent
    latent_cache.values = mx.array(latent)
    latent_cache.offset = context

    indexer = attention.indexer
    keys = _deterministic_rows(
        context, indexer.head_dim, 0.125 + layer_id * 0.015625, mx.bfloat16
    )[None]
    gates = _deterministic_rows(
        context, indexer.head_dim, 0.625 + layer_id * 0.015625, mx.bfloat16
    )[None]
    valid = mx.ones((1, context), dtype=mx.bool_)
    packed_logical = mx.concatenate(
        [keys, gates, valid.astype(keys.dtype)[..., None]], axis=-1
    )
    packed = mx.zeros((1, 1, capacity, packed_logical.shape[-1]), dtype=keys.dtype)
    packed[:, 0, :context] = packed_logical
    indexer_cache.keys = packed
    indexer_cache.values = mx.zeros((1, 1, capacity, 0), dtype=keys.dtype)
    indexer_cache.offset = context
    indexer_cache._no_pad = True
    indexer_cache._pool = (*indexer._pooled_states(keys, gates, valid), context)
    mx.eval(
        latent_cache.keys,
        latent_cache.values,
        indexer_cache.keys,
        *indexer_cache._pool[:3],
    )
    mx.synchronize()


def _synthetic_cache(model, context: int, backend: str):
    _set_backend(model, backend)
    cache = model.make_cache()
    for layer_id in EXPECTED_DSA:
        attention = model.language_model.model.layers[layer_id].self_attn
        if backend == "compact-nope-dsa":
            _fill_compact_dsa(cache[layer_id], attention, layer_id, context)
        else:
            _fill_direct_dsa(cache[layer_id], attention, layer_id, context)
    return cache


def _time_model(model, cache, token_base: int):
    for step in range(WARMUP_STEPS):
        output = model(
            mx.array([[token_base + step]], dtype=mx.uint32), cache=cache
        )
        mx.eval(output.logits)
        mx.synchronize()
    samples = []
    hashes = []
    for step in range(SAMPLES):
        started = time.perf_counter()
        output = model(
            mx.array([[token_base + WARMUP_STEPS + step]], dtype=mx.uint32),
            cache=cache,
        )
        mx.eval(output.logits)
        mx.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
        hashes.append(_hash(output.logits[0, -1]))
    median = statistics.median(samples)
    return {
        "samples_ms": samples,
        "median_ms": median,
        "decode_tps": 1000.0 / median,
        "output_hashes": hashes,
    }


def _time_dsa_and_pool_phases(model, cache, context: int):
    dsa_ms = 0.0
    update_ms = 0.0
    carry_ms = 0.0
    for layer_id in EXPECTED_DSA:
        attention = model.language_model.model.layers[layer_id].self_attn
        entry = cache[layer_id]
        x = _deterministic_rows(
            1, attention.hidden_size, 2.0 + layer_id * 0.015625, mx.bfloat16
        )[None]
        started = time.perf_counter()
        output = attention(x, cache=entry)
        mx.eval(output)
        mx.synchronize()
        dsa_ms += (time.perf_counter() - started) * 1000.0

        state = entry[1]
        qr = attention.q_a_layernorm(attention.q_a_proj(x))
        started = time.perf_counter()
        selected = state.update(attention.indexer, x, qr, None)
        mx.eval(selected)
        mx.synchronize()
        update_ms += (time.perf_counter() - started) * 1000.0

        active = state.active_tail_count
        count = active if active else state.index_kpool
        start = state.total_tokens - count
        pooled = state._pool_suffix(
            attention.indexer,
            start,
            state.raw_keys[:, -count:],
            state.raw_gates[:, -count:],
            state.raw_valid[:, -count:],
        )
        mx.eval(*pooled)
        mx.synchronize()
        started = time.perf_counter()
        state._write_pool_rows(start, pooled)
        mx.eval(*state.logical_pool())
        mx.synchronize()
        carry_ms += (time.perf_counter() - started) * 1000.0
    return {
        "dsa_total_ms": dsa_ms,
        "indexpool_update_ms": update_ms,
        "pool_carry_ms": carry_ms,
    }


def _performance_frontier(model):
    cases = {}
    direct_2k = None
    for context in CONTEXTS:
        _progress("frontier_build", context=context, backend="compact-nope-dsa")
        cache = _synthetic_cache(model, context, "compact-nope-dsa")
        active = int(mx.get_active_memory())
        mx.reset_peak_memory()
        timing = _time_model(model, cache, 3000)
        phases = _time_dsa_and_pool_phases(model, cache, context)
        cases[str(context)] = {
            "context_tokens": context,
            "full_model": timing,
            **phases,
            "active_memory_bytes": active,
            "peak_memory_bytes": int(mx.get_peak_memory()),
            "cache_nbytes": sum(int(entry.nbytes) for entry in cache),
            "max_raw_tokens": max(cache[layer][1].raw_token_count for layer in EXPECTED_DSA),
        }
        _release(cache)
        if context == CONTEXTS[0]:
            _progress("frontier_build", context=context, backend="direct")
            direct_cache = _synthetic_cache(model, context, "direct")
            direct_2k = _time_model(model, direct_cache, 3000)
            _release(direct_cache)
    compact_2k = cases[str(CONTEXTS[0])]["full_model"]
    compact_256k = cases[str(CONTEXTS[-1])]["full_model"]
    return {
        "cases": cases,
        "direct_2k": direct_2k,
        "decode_retention": compact_256k["decode_tps"] / compact_2k["decode_tps"],
        "compact_vs_direct_2k": compact_2k["decode_tps"] / direct_2k["decode_tps"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    args = parser.parse_args()

    report = inspect_checkpoint(args.model, require_server_ready=True)
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    model, _ = load(
        args.model,
        experimental_compact_nope_dsa_cache=True,
        compact_cache_reserve_tokens=RESERVE_TOKENS,
    )
    warm_residency(model)
    correctness = _prompt_correctness(model)
    ram_apc = _ram_apc_correctness(model)
    frontier = _performance_frontier(model)
    compact_cases = frontier["cases"]
    memory_growth = [
        max(0, row["decode_active_memory_drift_bytes"]) for row in correctness
    ]
    acceptance = {
        "prompt_1_16_128_256_all_logits_byte_identical": all(
            row["all_logits_byte_identical"] for row in correctness
        ),
        "all_generated_tokens_match": all(
            row["generated_tokens_match"] for row in correctness
        ),
        "single_latent_and_compact_pool_all_dsa_layers": all(
            row["single_latent_all_layers"] and row["compact_pool_all_layers"]
            for row in correctness
        ),
        "full_packed_indexer_history_absent": all(
            not row["full_packed_history_present"] for row in correctness
        ),
        "raw_state_at_most_19_tokens": all(
            row["max_raw_tokens"] <= 19 for row in correctness
        )
        and all(row["max_raw_tokens"] <= 19 for row in compact_cases.values()),
        "ram_apc_mod_0_1_2_3_continuation_matches": all(
            row["all_continuation_hashes_match"] for row in ram_apc
        ),
        "sparse_2k_full_model_logits_byte_identical": frontier[
            "direct_2k"
        ]["output_hashes"]
        == compact_cases[str(CONTEXTS[0])]["full_model"]["output_hashes"],
        "active_memory_growth_bounded_64mib": max(memory_growth)
        <= 64 * 1024 * 1024,
        "decode_retention_at_least_0_8": frontier["decode_retention"] >= 0.8,
        "compact_2k_no_more_than_5_percent_slower_than_direct": frontier[
            "compact_vs_direct_2k"
        ]
        >= 0.95,
        "compact_256k_no_oom": str(CONTEXTS[-1]) in compact_cases,
        "default_prompt_limit_unchanged": DEFAULT_MAX_PROMPT_TOKENS == 256,
        "non_speculative_kda_rollback_not_claimed": True,
    }
    acceptance["accepted"] = all(acceptance.values())
    output = {
        "schema": "glm53-compact-nope-dsa-production-runtime-v1",
        "date": date.today().isoformat(),
        "machine": "Apple M3 Ultra, 80-core GPU, 512 GB unified memory",
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "direct_cache_abi": NOPE_DSA_CACHE_ABI_DIRECT,
        "compact_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
        "contexts": list(CONTEXTS),
        "prompt_cases": list(PROMPTS),
        "decode_steps": DECODE_STEPS,
        "reserve_tokens": RESERVE_TOKENS,
        "correctness": correctness,
        "ram_apc": ram_apc,
        "performance_frontier": frontier,
        "runtime_policy": {
            "opt_in_flag": "--experimental-compact-nope-dsa-cache",
            "batch_size": 1,
            "disk_apc": "fail-closed",
            "ram_apc": "exact state/meta_state snapshot",
            "default_backend": "direct",
            "prompt_limit": DEFAULT_MAX_PROMPT_TOKENS,
            "mtp_or_dflash2_supported": False,
        },
        "acceptance": acceptance,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "accepted": acceptance["accepted"],
                "decode_retention": frontier["decode_retention"],
                "compact_vs_direct_2k": frontier["compact_vs_direct_2k"],
            },
            indent=2,
        )
    )
    return 0 if acceptance["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
