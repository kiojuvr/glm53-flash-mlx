#!/usr/bin/env python3
"""Probe row-blocked vector-gate KDA prefill without changing runtime policy."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import probe_long_context_first_decode_boundary as boundary
from glm53_flash_mlx.abi import KERNEL_ABI_VERSION, MLX_VLM_REVISION
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import EXPECTED_DSA, inspect_checkpoint

ROW_BLOCKS = (1, 2, 4, 8)
TOKENS = (128, 256, 512, 2048, 4096, 8192, 16384)
KDA_LAYERS = tuple(layer for layer in range(45) if layer not in EXPECTED_DSA)
REPRESENTATIVE_KDA_LAYERS = (0, 20, 44)
OPERATOR_WARMUPS = 2
OPERATOR_SAMPLES = 5
MODEL_WARMUPS = 1
MODEL_SAMPLES = 3
MODEL_PREFILL_TOKENS = (2048, 4096)
EXTENDED_MODEL_PREFILL_TOKENS = (8192, 16384)
MODEL_PREFILL_CHUNK = 2048
MAX_WORKING_PEAK_INCREASE = 256 * 2**20


_ROW_BLOCKED_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint n = thread_position_in_grid.z;
    uint b_idx = n / Hv;
    uint hv_idx = n % Hv;
    uint hk_idx = hv_idx / (Hv / Hk);
    uint dv_base = threadgroup_position_in_grid.y * ROW_BLOCK;
    constexpr uint N_PER_THREAD = Dk / 32;

    const device InT* q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    const device InT* k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
    const device InT* v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    device InT* y_ = y + b_idx * T * Hv * Dv + hv_idx * Dv;
    const device float* g_ = g + (b_idx * T * Hv + hv_idx) * Dk;
    auto beta_ = beta + b_idx * T * Hv;

    thread float state[ROW_BLOCK][N_PER_THREAD];
    for (uint r = 0; r < ROW_BLOCK; ++r) {
        uint dv = dv_base + r;
        uint safe_dv = dv < Dv ? dv : Dv - 1;
        const device StT* src = state_in + (n * Dv + safe_dv) * Dk;
        for (uint i = 0; i < N_PER_THREAD; ++i) {
            uint s_idx = N_PER_THREAD * lane + i;
            state[r][i] = dv < Dv ? float(src[s_idx]) : 0.0f;
        }
    }

    for (uint t = 0; t < T; ++t) {
        if (HAS_MASK == 0 || mask[b_idx * T + t]) {
            thread float kv_mem[ROW_BLOCK];
            for (uint r = 0; r < ROW_BLOCK; ++r) kv_mem[r] = 0.0f;
            for (uint i = 0; i < N_PER_THREAD; ++i) {
                uint s_idx = N_PER_THREAD * lane + i;
                float decay = g_[s_idx];
                float key = float(k_[s_idx]);
                for (uint r = 0; r < ROW_BLOCK; ++r) {
                    state[r][i] = state[r][i] * decay;
                    kv_mem[r] += state[r][i] * key;
                }
            }
            for (uint r = 0; r < ROW_BLOCK; ++r) {
                kv_mem[r] = simd_sum(kv_mem[r]);
            }

            thread float out[ROW_BLOCK];
            for (uint r = 0; r < ROW_BLOCK; ++r) {
                uint dv = dv_base + r;
                float value = dv < Dv ? float(v_[dv]) : 0.0f;
                float delta = (value - kv_mem[r]) * float(beta_[hv_idx]);
                out[r] = 0.0f;
                for (uint i = 0; i < N_PER_THREAD; ++i) {
                    uint s_idx = N_PER_THREAD * lane + i;
                    state[r][i] = state[r][i] + float(k_[s_idx]) * delta;
                    out[r] += state[r][i] * float(q_[s_idx]);
                }
                out[r] = simd_sum(out[r]);
                if (lane == 0 && dv < Dv) y_[dv] = InT(out[r]);
            }
        } else if (lane == 0) {
            for (uint r = 0; r < ROW_BLOCK; ++r) {
                uint dv = dv_base + r;
                if (dv < Dv) y_[dv] = InT(0.0f);
            }
        }
        q_ += Hk * Dk;
        k_ += Hk * Dk;
        v_ += Hv * Dv;
        y_ += Hv * Dv;
        g_ += Hv * Dk;
        beta_ += Hv;
    }

    for (uint r = 0; r < ROW_BLOCK; ++r) {
        uint dv = dv_base + r;
        if (dv >= Dv) continue;
        device StT* dst = state_out + (n * Dv + dv) * Dk;
        for (uint i = 0; i < N_PER_THREAD; ++i) {
            uint s_idx = N_PER_THREAD * lane + i;
            dst[s_idx] = StT(state[r][i]);
        }
    }
"""


_row_blocked_kernel = (
    mx.fast.metal_kernel(
        name="glm53_probe_row_blocked_vector_kda",
        input_names=["q", "k", "v", "g", "beta", "state_in", "mask"],
        output_names=["y", "state_out"],
        source=_ROW_BLOCKED_SOURCE,
    )
    if mx.metal.is_available()
    else None
)


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), file=sys.stderr, flush=True)


def _contiguous(value: mx.array) -> mx.array:
    return mx.contiguous(value, allow_col_major=False)


def row_blocked_vector_kda(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    *,
    row_block: int,
    mask: mx.array | None = None,
):
    """Run the diagnostic vector-gate kernel with Direct recurrence order."""
    if _row_blocked_kernel is None:
        raise RuntimeError("row-blocked KDA probe requires Metal")
    if row_block not in ROW_BLOCKS:
        raise ValueError(f"unsupported row block: {row_block}")
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if g.shape != (B, T, Hv, Dk):
        raise ValueError("row-blocked probe requires vector gate [B,T,Hv,Dk]")
    if Dk % 32:
        raise ValueError("Dk must be divisible by a SIMD width")
    if mask is None:
        mask_input = mx.ones((1,), dtype=mx.bool_)
        has_mask = 0
    else:
        if mask.shape != (B, T):
            raise ValueError("mask must have shape [B,T]")
        mask_input = _contiguous(mask)
        has_mask = 1
    inputs = [
        _contiguous(q),
        _contiguous(k),
        _contiguous(v),
        _contiguous(g),
        _contiguous(beta),
        _contiguous(state),
        mask_input,
    ]
    return _row_blocked_kernel(
        inputs=inputs,
        template=[
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("T", T),
            ("ROW_BLOCK", row_block),
            ("HAS_MASK", has_mask),
        ],
        grid=(32, math.ceil(Dv / row_block), B * Hv),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )


def _storage_np(value: mx.array) -> np.ndarray:
    mx.eval(value)
    if value.dtype == mx.bfloat16:
        value = value.view(mx.uint16)
    return np.ascontiguousarray(np.asarray(value))


def _hash(value: mx.array) -> str:
    return hashlib.sha256(_storage_np(value).tobytes()).hexdigest()


def _equal(left: mx.array, right: mx.array) -> bool:
    mx.eval(left, right)
    return bool(mx.array_equal(left, right).item())


def _memory() -> dict[str, int]:
    mx.synchronize()
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }


def _release(*values) -> None:
    del values
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def _deterministic(shape, phase: float, dtype) -> mx.array:
    total = math.prod(shape)
    values = mx.arange(total, dtype=mx.float32)
    values = mx.sin(values * 0.0009765625 + phase)
    return values.reshape(shape).astype(dtype)


def _operator_inputs(tokens: int, *, heads: int = 64, strided: bool = False):
    shape = (1, tokens, heads, 128)
    q = (_deterministic(shape, 0.125, mx.bfloat16) * 0.125).astype(mx.bfloat16)
    k = (_deterministic(shape, 0.625, mx.bfloat16) * 0.125).astype(mx.bfloat16)
    v = (_deterministic(shape, 1.125, mx.bfloat16) * 0.125).astype(mx.bfloat16)
    g = mx.exp(_deterministic(shape, 1.625, mx.float32) * -0.03125)
    beta = mx.sigmoid(_deterministic((1, tokens, heads), 2.125, mx.bfloat16))
    state = _deterministic((1, heads, 128, 128), 2.625, mx.float32) * 0.001
    if strided:
        def interleave(value):
            wide = mx.zeros((*value.shape[:-1], value.shape[-1] * 2), value.dtype)
            wide[..., 1::2] = value
            return wide[..., 1::2]

        q, k, v, g = map(interleave, (q, k, v, g))
        state = interleave(state)
    return q, k, v, g, beta, state


def _reference(q, k, v, g, beta, state, mask=None):
    from mlx_vlm.models import gated_delta as gd

    return gd.gated_delta_kernel(
        _contiguous(q),
        _contiguous(k),
        _contiguous(v),
        _contiguous(g),
        _contiguous(beta),
        _contiguous(state),
        None if mask is None else _contiguous(mask),
    )


def _fixture_cases() -> list[dict]:
    cases = []
    token_cases = sorted(
        {7, 8, 9, 10, 15, 16, 17, 18}
        | {31 + remainder for remainder in range(4)}
    )
    for initial in ("zero", "nonzero"):
        for mask_kind in ("none", "tail", "internal_gap"):
            for tokens in token_cases:
                q, k, v, g, beta, state = _operator_inputs(tokens, heads=2)
                if initial == "zero":
                    state = mx.zeros_like(state)
                mask = None
                if mask_kind != "none":
                    mask = mx.ones((1, tokens), dtype=mx.bool_)
                    if mask_kind == "tail":
                        mask[:, -1] = False
                    else:
                        mask[:, tokens // 2] = False
                expected_y, expected_state = _reference(
                    q, k, v, g, beta, state, mask
                )
                mx.eval(expected_y, expected_state)
                row = {
                    "tokens": tokens,
                    "initial_state": initial,
                    "mask": mask_kind,
                    "token_mod_4": tokens % 4,
                    "arms": {},
                }
                for block in ROW_BLOCKS:
                    actual_y, actual_state = row_blocked_vector_kda(
                        q, k, v, g, beta, state, row_block=block, mask=mask
                    )
                    mx.eval(actual_y, actual_state)
                    row["arms"][str(block)] = {
                        "output_byte_identical": _equal(expected_y, actual_y),
                        "state_byte_identical": _equal(expected_state, actual_state),
                    }
                cases.append(row)
                _release(q, k, v, g, beta, state, expected_y, expected_state)

    q, k, v, g, beta, state = _operator_inputs(17, heads=2, strided=True)
    expected_y, expected_state = _reference(q, k, v, g, beta, state)
    strided = {"input": "interleaved_last_axis", "arms": {}}
    for block in ROW_BLOCKS:
        actual_y, actual_state = row_blocked_vector_kda(
            q, k, v, g, beta, state, row_block=block
        )
        strided["arms"][str(block)] = {
            "output_byte_identical": _equal(expected_y, actual_y),
            "state_byte_identical": _equal(expected_state, actual_state),
        }
    _release(q, k, v, g, beta, state, expected_y, expected_state)
    return cases, strided


def _timed(callable_, warmups: int, samples: int) -> dict:
    rows = []
    for sample in range(warmups + samples):
        mx.reset_peak_memory()
        active = int(mx.get_active_memory())
        started = time.perf_counter()
        output, state = callable_()
        mx.eval(output, state)
        mx.synchronize()
        elapsed = (time.perf_counter() - started) * 1000.0
        if sample >= warmups:
            rows.append(
                {
                    "latency_ms": elapsed,
                    "working_peak_bytes": max(
                        0, int(mx.get_peak_memory()) - active
                    ),
                }
            )
        del output, state
    return {
        "warmups": warmups,
        "samples": samples,
        "samples_ms": [row["latency_ms"] for row in rows],
        "median_ms": statistics.median(row["latency_ms"] for row in rows),
        "p95_ms": sorted(row["latency_ms"] for row in rows)[
            max(0, math.ceil(0.95 * len(rows)) - 1)
        ],
        "working_peak_bytes": max(row["working_peak_bytes"] for row in rows),
    }


def _operator_sweep(warmups: int, samples: int) -> tuple[dict, int]:
    results = {}
    for tokens in TOKENS:
        _progress("operator", tokens=tokens)
        q, k, v, g, beta, state = _operator_inputs(tokens)
        expected_y, expected_state = _reference(q, k, v, g, beta, state)
        mx.eval(expected_y, expected_state)
        arms = {
            "current": _timed(
                lambda: _reference(q, k, v, g, beta, state), warmups, samples
            )
        }
        for block in ROW_BLOCKS:
            arms[str(block)] = _timed(
                lambda block=block: row_blocked_vector_kda(
                    q, k, v, g, beta, state, row_block=block
                ),
                warmups,
                samples,
            )
            actual_y, actual_state = row_blocked_vector_kda(
                q, k, v, g, beta, state, row_block=block
            )
            arms[str(block)].update(
                {
                    "output_byte_identical": _equal(expected_y, actual_y),
                    "state_byte_identical": _equal(expected_state, actual_state),
                    "speedup_vs_row_block_1": None,
                    "speedup_vs_current": (
                        arms["current"]["median_ms"]
                        / arms[str(block)]["median_ms"]
                    ),
                }
            )
        r1 = arms["1"]["median_ms"]
        for block in ROW_BLOCKS:
            arms[str(block)]["speedup_vs_row_block_1"] = (
                r1 / arms[str(block)]["median_ms"]
            )
        results[str(tokens)] = {
            "tokens": tokens,
            "shape": list(q.shape),
            "arms": arms,
        }
        _release(q, k, v, g, beta, state, expected_y, expected_state)

    long_tokens = [token for token in TOKENS if token >= 4096]
    scores = {}
    for block in ROW_BLOCKS:
        ratios = [
            results[str(token)]["arms"][str(block)]["speedup_vs_row_block_1"]
            for token in long_tokens
        ]
        scores[block] = math.prod(ratios) ** (1.0 / len(ratios))
    winner = max(scores, key=scores.get)
    for result in results.values():
        result["winner"] = winner
    return {
        "contexts": results,
        "winner_selection": {
            "metric": "geometric mean speedup vs R=1 over T>=4096",
            "scores": {str(key): value for key, value in scores.items()},
            "winner": winner,
        },
    }, winner


def _candidate_update(row_block: int):
    from mlx_vlm.models import gated_delta as gd

    def update(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        state=None,
        mask=None,
        use_kernel=True,
        lower_bound=None,
    ):
        beta = mx.sigmoid(b)
        g = (
            gd.compute_g(A_log, a, dt_bias)
            if lower_bound is None
            else gd.compute_g_safe(A_log, a, dt_bias, lower_bound)
        )
        if state is None:
            B, _, _, Dk = q.shape
            Hv, Dv = v.shape[-2:]
            state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        if g.ndim != 4:
            return gd.gated_delta_update(
                q,
                k,
                v,
                a,
                b,
                A_log,
                dt_bias,
                state=state,
                mask=mask,
                use_kernel=use_kernel,
                lower_bound=lower_bound,
            )
        return row_blocked_vector_kda(
            q, k, v, g, beta, state, row_block=row_block, mask=mask
        )

    return update


@contextmanager
def _kda_backend(row_block: int | None):
    from mlx_vlm.models.glm5_next import language as glm

    original = glm.gated_delta_update
    if row_block is not None:
        glm.gated_delta_update = _candidate_update(row_block)
    try:
        yield
    finally:
        glm.gated_delta_update = original


def _deterministic_hidden(tokens: int, phase: float = 0.0) -> mx.array:
    rows = mx.arange(tokens, dtype=mx.float32)[:, None]
    cols = mx.arange(4096, dtype=mx.float32)[None]
    return mx.sin(rows * 0.00390625 + cols * 0.0009765625 + phase)[None].astype(
        mx.bfloat16
    )


def _run_attention(attention, hidden, row_block: int | None):
    from mlx_vlm.models.cache import ArraysCache

    cache = ArraysCache(size=2)
    with _kda_backend(row_block):
        output = attention(hidden, cache=cache)
        mx.eval(output, cache.state)
        mx.synchronize()
    return output, cache.state


def _actual_layer_parity(model, winner: int) -> dict:
    rows = []
    for layer in KDA_LAYERS:
        tokens = 128 if layer in REPRESENTATIVE_KDA_LAYERS else 8
        attention = model.language_model.model.layers[layer].self_attn
        hidden = _deterministic_hidden(tokens, phase=layer * 0.015625)
        expected, expected_state = _run_attention(attention, hidden, None)
        actual, actual_state = _run_attention(attention, hidden, winner)
        row = {
            "layer": layer,
            "tokens": tokens,
            "representative": layer in REPRESENTATIVE_KDA_LAYERS,
            "output_byte_identical": _equal(expected, actual),
            "conv_state_byte_identical": _equal(
                expected_state[0], actual_state[0]
            ),
            "recurrent_state_byte_identical": _equal(
                expected_state[1], actual_state[1]
            ),
            "output_hash": _hash(actual),
            "state_hash": _hash(actual_state[1]),
        }
        rows.append(row)
        _progress("layer_parity", **row)
        _release(hidden, expected, actual, expected_state, actual_state)
    return {
        "winner": winner,
        "representative_layers": list(REPRESENTATIVE_KDA_LAYERS),
        "layers": rows,
        "all_34_layers_byte_identical": all(
            row["output_byte_identical"]
            and row["conv_state_byte_identical"]
            and row["recurrent_state_byte_identical"]
            for row in rows
        ),
    }


def _tokens(count: int, vocab: int) -> mx.array:
    values = ((np.arange(count, dtype=np.uint64) * 7919) % (vocab - 1024) + 100)
    return mx.array(values.astype(np.uint32)[None])


def _cache_kda_hash(cache) -> str:
    digest = hashlib.sha256()
    for layer in KDA_LAYERS:
        for value in cache[layer].state:
            digest.update(_storage_np(value).tobytes())
    return digest.hexdigest()


def _run_model_prefill(model, token_ids, row_block: int | None) -> dict:
    model._glm53_cache_backend = "direct"
    model.language_model._glm53_cache_backend = "direct"
    cache = model.make_cache()
    mx.reset_peak_memory()
    active = int(mx.get_active_memory())
    started = time.perf_counter()
    with _kda_backend(row_block):
        output = None
        chunks = 0
        for start in range(0, token_ids.shape[1], MODEL_PREFILL_CHUNK):
            output = model(
                token_ids[:, start : start + MODEL_PREFILL_CHUNK], cache=cache
            )
            chunks += 1
        final_logits = output.logits[:, -1]
        mx.eval(final_logits, [cache[layer].state for layer in KDA_LAYERS])
        mx.synchronize()
    elapsed = (time.perf_counter() - started) * 1000.0
    row = {
        "latency_ms": elapsed,
        "chunks": chunks,
        "chunk_tokens": MODEL_PREFILL_CHUNK,
        "final_logits_hash": _hash(final_logits),
        "kda_state_hash": _cache_kda_hash(cache),
        "working_peak_bytes": max(0, int(mx.get_peak_memory()) - active),
        "nan_count": int(np.isnan(np.asarray(final_logits.astype(mx.float32))).sum()),
    }
    del output, final_logits, cache
    _release()
    return row


def _model_prefill(
    model,
    vocab: int,
    winner: int,
    warmups: int,
    samples: int,
    *,
    token_counts=MODEL_PREFILL_TOKENS,
):
    results = {}
    for count in token_counts:
        token_ids = _tokens(count, vocab)
        arms = {"current": [], "winner": []}
        for sample in range(warmups + samples):
            order = (None, winner) if sample % 2 == 0 else (winner, None)
            for row_block in order:
                key = "current" if row_block is None else "winner"
                row = _run_model_prefill(model, token_ids, row_block)
                if sample >= warmups:
                    arms[key].append(row)
                _progress(
                    "model_prefill",
                    tokens=count,
                    arm=key,
                    sample=sample,
                    latency_ms=row["latency_ms"],
                )
        summarized = {}
        for key, rows in arms.items():
            summarized[key] = {
                "samples_ms": [row["latency_ms"] for row in rows],
                "median_ms": statistics.median(row["latency_ms"] for row in rows),
                "chunks": rows[0]["chunks"],
                "chunk_tokens": rows[0]["chunk_tokens"],
                "final_logits_hashes": [row["final_logits_hash"] for row in rows],
                "kda_state_hashes": [row["kda_state_hash"] for row in rows],
                "nan_count": sum(row["nan_count"] for row in rows),
                "working_peak_bytes": max(row["working_peak_bytes"] for row in rows),
            }
        summarized["speedup"] = (
            summarized["current"]["median_ms"]
            / summarized["winner"]["median_ms"]
        )
        summarized["exact"] = (
            len(
                set(
                    summarized["current"]["final_logits_hashes"]
                    + summarized["winner"]["final_logits_hashes"]
                )
            )
            == 1
            and len(
                set(
                    summarized["current"]["kda_state_hashes"]
                    + summarized["winner"]["kda_state_hashes"]
                )
            )
            == 1
        )
        results[str(count)] = summarized
        del token_ids
    return results


def _prefill_cache(model, tokens, vocab, row_block):
    model._glm53_cache_backend = "direct"
    model.language_model._glm53_cache_backend = "direct"
    cache = model.make_cache()
    with _kda_backend(row_block):
        output = model(_tokens(tokens, vocab), cache=cache)
        mx.eval(output.logits, [cache[layer].state for layer in KDA_LAYERS])
        mx.synchronize()
    return cache


def _decode_arm(model, cache, vocab: int, row_block: int | None) -> dict:
    samples = []
    hashes = []
    for step in range(32):
        token = mx.array([[100 + (step * 7919) % (vocab - 1024)]], dtype=mx.uint32)
        started = time.perf_counter()
        with _kda_backend(row_block):
            output = model(token, cache=cache)
            logits = output.logits[:, -1]
            mx.eval(logits)
            mx.synchronize()
        if step >= 4:
            samples.append((time.perf_counter() - started) * 1000.0)
            hashes.append(_hash(logits))
    return {
        "warmup_steps": 4,
        "measured_steps": 28,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "logits_hashes": hashes,
        "final_kda_state_hash": _cache_kda_hash(cache),
    }


def _decode_regression(model, vocab: int, winner: int) -> dict:
    current_cache = _prefill_cache(model, 256, vocab, None)
    winner_cache = _prefill_cache(model, 256, vocab, winner)
    initial_exact = _cache_kda_hash(current_cache) == _cache_kda_hash(winner_cache)
    current = _decode_arm(model, current_cache, vocab, None)
    candidate = _decode_arm(model, winner_cache, vocab, winner)
    exact = (
        current["logits_hashes"] == candidate["logits_hashes"]
        and current["final_kda_state_hash"] == candidate["final_kda_state_hash"]
    )
    result = {
        "current": current,
        "winner": candidate,
        "initial_kda_state_exact": initial_exact,
        "all_logits_and_final_state_exact": exact,
        "latency_ratio": candidate["median_ms"] / current["median_ms"],
        "regression": candidate["median_ms"] / current["median_ms"] - 1.0,
    }
    _release(current_cache, winner_cache)
    return result


def _official_oracle(model, processor, report, winner: int) -> dict:
    import oracle_trace

    expected_16 = json.loads(
        Path("oracles/glm53-official-greedy-16.json").read_text()
    )
    expected_128 = json.loads(
        Path("oracles/glm53-official-greedy-128.json").read_text()
    )
    prompt = oracle_trace.DEFAULT_PROMPT
    formatted = processor.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = processor(formatted, return_tensors="np", add_special_tokens=True)
    prompt_ids = np.asarray(encoded["input_ids"], dtype=np.int32).reshape(1, -1)
    model._glm53_cache_backend = "direct"
    model.language_model._glm53_cache_backend = "direct"
    cache = model.make_cache()
    generated = []
    steps = []
    with _kda_backend(winner):
        output = model(mx.array(prompt_ids), cache=cache)
        for step in range(128):
            logits = output.logits[0, -1].astype(mx.float32)
            mx.eval(logits)
            values = np.ascontiguousarray(np.asarray(logits), dtype=np.float32)
            top2 = np.argpartition(values, -2)[-2:]
            top2 = top2[np.argsort(values[top2])[::-1]]
            token = int(top2[0])
            generated.append(token)
            steps.append(
                {
                    "step": step,
                    "token": token,
                    "logits_f32_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                }
            )
            if step + 1 < 128:
                output = model(mx.array([[token]], dtype=mx.uint32), cache=cache)

    def actual(count):
        return {
                "schema": "glm53-greedy-oracle-v1",
                "official_hf_revision": report.official_revision,
                "checkpoint_fingerprint": report.fingerprint,
                "checkpoint_layout_digest": report.layout_digest,
                "kernel_abi": KERNEL_ABI_VERSION,
                "prompt": prompt,
                "formatted_prompt_sha256": hashlib.sha256(
                    formatted.encode()
                ).hexdigest(),
                "prompt_token_count": int(prompt_ids.size),
                "prompt_token_ids_sha256": hashlib.sha256(
                    prompt_ids.tobytes()
                ).hexdigest(),
                "generation_tokens": count,
                "generated_token_ids": generated[:count],
                "steps": steps[:count],
            }

    actual_16 = actual(16)
    actual_128 = actual(128)
    failures_16 = oracle_trace.compare_trace(actual_16, expected_16)
    failures_128 = oracle_trace.compare_trace(actual_128, expected_128)
    result = {
        "row_block": winner,
        "prompt_tokens": int(prompt_ids.size),
        "generated_token_sha256": hashlib.sha256(
            np.asarray(generated, dtype=np.uint32).tobytes()
        ).hexdigest(),
        "first_16_match": not failures_16,
        "full_128_match": not failures_128,
        "failures_16": failures_16,
        "failures_128": failures_128,
        "all_full_vocab_logits_hashes_match": not failures_128,
    }
    _release(cache, output)
    return result


@contextmanager
def _rope_counter():
    counts = {"nn_rope_calls": 0, "mx_fast_rope_calls": 0}
    original_nn = nn.RoPE.__call__
    original_fast = mx.fast.rope

    def nn_call(self, *args, **kwargs):
        counts["nn_rope_calls"] += 1
        return original_nn(self, *args, **kwargs)

    def fast_call(*args, **kwargs):
        counts["mx_fast_rope_calls"] += 1
        return original_fast(*args, **kwargs)

    nn.RoPE.__call__ = nn_call
    mx.fast.rope = fast_call
    try:
        yield counts
    finally:
        nn.RoPE.__call__ = original_nn
        mx.fast.rope = original_fast


def _nope_fixture(model, vocab: int, winner: int) -> dict:
    attentions = [
        model.language_model.model.layers[layer].self_attn for layer in EXPECTED_DSA
    ]
    assertions = {
        "all_attention_use_nope": all(attention.use_nope for attention in attentions),
        "all_qk_rope_head_dim_zero": all(
            attention.qk_rope_head_dim == 0 for attention in attentions
        ),
        "no_attention_has_rotary_emb": all(
            not hasattr(attention, "rotary_emb") for attention in attentions
        ),
    }
    if not all(assertions.values()):
        raise AssertionError(f"NoPE structural contract failed: {assertions}")

    with _rope_counter() as counts:
        prompt_rows = {}
        for backend in ("direct", "compact-nope-dsa"):
            model._glm53_cache_backend = backend
            model.language_model._glm53_cache_backend = backend
            model.language_model._glm53_compact_cache_capacity_tokens = 4352
            cache = model.make_cache()
            with _kda_backend(winner):
                output = model(_tokens(128, vocab), cache=cache)
                first = model(mx.array([[123]], dtype=mx.uint32), cache=cache)
                mx.eval(output.logits, first.logits)
                mx.synchronize()
            prompt_rows[backend] = {
                "prefill_logits_hash": _hash(output.logits),
                "decode_logits_hash": _hash(first.logits),
            }
            _release(cache, output, first)

        diagnostic_rows = {}
        for backend in ("direct", "compact-nope-dsa"):
            cache = boundary._synthetic_cache(model, 2049, backend)
            diagnostics = boundary._dsa_diagnostics(model, cache, backend, 2049)
            diagnostic_rows[backend] = diagnostics["layers"]
            _release(cache)

    direct_rows = diagnostic_rows["direct"]
    compact_rows = diagnostic_rows["compact-nope-dsa"]
    selected_equal = all(
        left["index_hash"] == right["index_hash"]
        for left, right in zip(direct_rows, compact_rows, strict=True)
    )
    output_equal = all(
        left["output_hash"] == right["output_hash"]
        for left, right in zip(direct_rows, compact_rows, strict=True)
    )
    return {
        "structural_assertions": assertions,
        "rotary_position_embedding_dispatch": counts,
        "prompt_direct_compact": prompt_rows,
        "sparse_context_tokens": 2049,
        "direct_diagnostics": direct_rows,
        "compact_diagnostics": compact_rows,
        "selected_index_hashes_unchanged": selected_equal,
        "selected_output_hashes_unchanged": output_equal,
        "indexpool_token_position_processing_preserved": selected_equal
        and all(row["selected_width"] == 2051 for row in direct_rows),
    }


def _write_atomic(path: Path, artifact: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator-warmups", type=int, default=OPERATOR_WARMUPS)
    parser.add_argument("--operator-samples", type=int, default=OPERATOR_SAMPLES)
    parser.add_argument("--model-warmups", type=int, default=MODEL_WARMUPS)
    parser.add_argument("--model-samples", type=int, default=MODEL_SAMPLES)
    parser.add_argument("--refresh-oracle-existing", action="store_true")
    args = parser.parse_args()

    mx.set_wired_limit(int(440e9))
    mx.set_cache_limit(int(32e9))
    report = inspect_checkpoint(args.model, require_server_ready=True)
    if args.refresh_oracle_existing:
        artifact = json.loads(args.output.read_text())
        winner = int(artifact["winner"])
        model, processor = load(args.model, experimental_packed_decode_moe=True)
        warm_residency(model)
        oracle = _official_oracle(model, processor, report, winner)
        artifact["official_oracle"] = oracle
        artifact["acceptance"]["official_16_128_token_oracle_exact"] = (
            oracle["first_16_match"] and oracle["full_128_match"]
        )
        artifact["acceptance"]["accepted"] = all(
            value
            for key, value in artifact["acceptance"].items()
            if key != "accepted"
        )
        artifact["complete"] = True
        _write_atomic(args.output, artifact)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "oracle": oracle,
                    "accepted": artifact["acceptance"]["accepted"],
                },
                indent=2,
            )
        )
        return 0 if oracle["first_16_match"] and oracle["full_128_match"] else 1
    fixtures, strided = _fixture_cases()
    operator, winner = _operator_sweep(
        args.operator_warmups, args.operator_samples
    )

    _progress("load_model")
    model, processor = load(args.model, experimental_packed_decode_moe=True)
    warm_residency(model)
    vocab = int(model.language_model.model.embed_tokens.weight.shape[0])
    layers = _actual_layer_parity(model, winner)
    nope = _nope_fixture(model, vocab, winner)
    model_prefill = _model_prefill(
        model, vocab, winner, args.model_warmups, args.model_samples
    )
    primary_prefill_gate = all(
        model_prefill[str(token)]["speedup"] >= 1.02
        and model_prefill[str(token)]["exact"]
        for token in MODEL_PREFILL_TOKENS
    )
    if layers["all_34_layers_byte_identical"] and primary_prefill_gate:
        extended_prefill = _model_prefill(
            model,
            vocab,
            winner,
            args.model_warmups,
            args.model_samples,
            token_counts=EXTENDED_MODEL_PREFILL_TOKENS,
        )
        extended_prefill_decision = "measured_after_layer_and_2k_4k_gates_passed"
    else:
        extended_prefill = {}
        extended_prefill_decision = (
            "skipped_because_layer_or_2k_4k_gate_did_not_pass"
        )
    decode = _decode_regression(model, vocab, winner)
    oracle = _official_oracle(model, processor, report, winner)

    long_contexts = [str(token) for token in TOKENS if token >= 4096]
    long_speedups = [
        operator["contexts"][token]["arms"][str(winner)][
            "speedup_vs_row_block_1"
        ]
        for token in long_contexts
    ]
    peak_increase = max(
        operator["contexts"][token]["arms"][str(winner)]["working_peak_bytes"]
        - operator["contexts"][token]["arms"]["1"]["working_peak_bytes"]
        for token in operator["contexts"]
    )
    fixture_exact = all(
        all(
            arm["output_byte_identical"] and arm["state_byte_identical"]
            for arm in row["arms"].values()
        )
        for row in fixtures
    ) and all(
        arm["output_byte_identical"] and arm["state_byte_identical"]
        for arm in strided["arms"].values()
    )
    acceptance = {
        "all_operator_outputs_and_final_states_byte_identical": all(
            row["arms"][str(winner)]["output_byte_identical"]
            and row["arms"][str(winner)]["state_byte_identical"]
            for row in operator["contexts"].values()
        ),
        "zero_nonzero_mask_mod_and_strided_fixtures_exact": fixture_exact,
        "early_middle_late_and_all_34_kda_layers_exact": layers[
            "all_34_layers_byte_identical"
        ],
        "true_nope_structure_and_zero_rotary_dispatch": (
            all(nope["structural_assertions"].values())
            and sum(nope["rotary_position_embedding_dispatch"].values()) == 0
        ),
        "direct_compact_indexpool_position_hashes_unchanged": (
            nope["selected_index_hashes_unchanged"]
            and nope["selected_output_hashes_unchanged"]
            and nope["indexpool_token_position_processing_preserved"]
            and len(
                {
                    row["prefill_logits_hash"]
                    for row in nope["prompt_direct_compact"].values()
                }
            )
            == 1
            and len(
                {
                    row["decode_logits_hash"]
                    for row in nope["prompt_direct_compact"].values()
                }
            )
            == 1
        ),
        "kda_aggregate_speedup_at_least_1_05_at_4k_and_above": min(
            long_speedups
        )
        >= 1.05,
        "whole_model_prefill_2k_and_4k_speedup_at_least_1_02": all(
            model_prefill[str(token)]["speedup"] >= 1.02
            for token in MODEL_PREFILL_TOKENS
        ),
        "whole_model_prefill_exact": all(
            model_prefill[str(token)]["exact"] for token in MODEL_PREFILL_TOKENS
        ),
        "decode_regression_at_most_1_percent_and_exact": (
            decode["regression"] <= 0.01
            and decode["all_logits_and_final_state_exact"]
        ),
        "working_peak_increase_at_most_256_mib": (
            peak_increase <= MAX_WORKING_PEAK_INCREASE
        ),
        "official_16_128_token_oracle_exact": (
            oracle["first_16_match"] and oracle["full_128_match"]
        ),
        "runtime_kernel_abi_server_and_admission_unchanged": True,
    }
    acceptance["accepted"] = all(acceptance.values())
    artifact = {
        "schema": "glm53-row-blocked-vector-kda-prefill-v1",
        "date": date.today().isoformat(),
        "machine": platform.machine(),
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "runtime_kernel_abi": KERNEL_ABI_VERSION,
        "probe_only": True,
        "method": {
            "operator": "official H64 Dk128 Dv128 vector-gate shape",
            "operator_warmups": args.operator_warmups,
            "operator_samples": args.operator_samples,
            "whole_model_backend": "packed-decode MoE + Direct cache",
            "whole_model_prefill_chunk_tokens": MODEL_PREFILL_CHUNK,
            "whole_model_warmups": args.model_warmups,
            "whole_model_samples": args.model_samples,
            "one_shot_4096_not_used": (
                "existing FP8 projection 1-D grid reaches 2^32 threads; "
                "production server chunks prefill at 2048"
            ),
        },
        "row_blocks": list(ROW_BLOCKS),
        "tokens": list(TOKENS),
        "kda_layers": list(KDA_LAYERS),
        "fixture_cases": fixtures,
        "strided_fixture": strided,
        "operator": operator,
        "winner": winner,
        "actual_layer_parity": layers,
        "true_nope_fixture": nope,
        "whole_model_prefill": model_prefill,
        "extended_whole_model_prefill": extended_prefill,
        "extended_whole_model_prefill_decision": extended_prefill_decision,
        "decode": decode,
        "official_oracle": oracle,
        "working_peak_increase_bytes": peak_increase,
        "memory_final": _memory(),
        "runtime_changes": {
            "kernel_abi": False,
            "server": False,
            "admission": False,
            "default_backend": False,
        },
        "acceptance": acceptance,
        "complete": True,
    }
    _write_atomic(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "winner": winner,
                "accepted": acceptance["accepted"],
                "long_speedups": long_speedups,
                "model_prefill_speedups": {
                    key: value["speedup"] for key, value in model_prefill.items()
                },
                "decode_regression": decode["regression"],
            },
            indent=2,
        )
    )
    return 0 if acceptance["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
