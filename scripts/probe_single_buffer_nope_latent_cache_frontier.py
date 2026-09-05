#!/usr/bin/env python3
"""Probe single-buffer NoPE latent caches across all 11 DSA layers."""

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

from glm53_flash_mlx.abi import MLX_VLM_REVISION, NOPE_DSA_CACHE_ABI
from glm53_flash_mlx.loader import load
from glm53_flash_mlx.manifest import EXPECTED_DSA, inspect_checkpoint
from glm53_flash_mlx.server import LEGACY_PROBE_MAX_PROMPT_TOKENS
from probe_long_context_dsa_decode_frontier import (
    _attention_phase,
    _expand_phase,
    _gather_phase,
    _pool_update,
    _score_phase,
    _selection_phase,
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

CONTEXTS = (2049, 131072, 262144)
WARMUP_STEPS = 4
MEASURED_STEPS = 16
PHASES = (
    "latent_projection_append",
    "indexer_pool_update",
    "pool_score",
    "selection",
    "pool_expansion",
    "latent_gather",
    "selected_attention",
)
ARM_CONFIG = {
    "dual_kv_step256": {"single": False, "preallocated": False},
    "dual_kv_preallocated": {"single": False, "preallocated": True},
    "single_latent_step256": {"single": True, "preallocated": False},
    "single_latent_preallocated": {"single": True, "preallocated": True},
}


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


def _separate_copy(value: mx.array) -> mx.array:
    selector = mx.ones(value.shape[:-1] + (1,), dtype=mx.bool_)
    return mx.contiguous(mx.where(selector, value, mx.zeros_like(value)))


class ProbeNoPELatentCache:
    """Probe-only KVCache-compatible storage with a single NoPE latent buffer."""

    step = 256

    def __init__(self, latent: mx.array, offset: int):
        self._latent = latent
        self.offset = offset
        self.last_copy_bytes = 0
        self.total_copy_bytes = 0
        self.extension_count = 0

    @property
    def keys(self):
        return self._latent

    @keys.setter
    def keys(self, value):
        self._latent = value

    @property
    def values(self):
        return self._latent

    @values.setter
    def values(self, value):
        # The patched NoPE attention only sets keys for dependency ordering.
        self._latent = value

    @property
    def nbytes(self) -> int:
        return int(self._latent.nbytes)

    def update_and_fetch(self, keys, values):
        if keys.shape != values.shape:
            raise ValueError("NoPE latent K/V shapes must match")
        previous = self.offset
        self.last_copy_bytes = 0
        required = previous + keys.shape[2]
        if required > self._latent.shape[2]:
            if previous % self.step != 0:
                self._latent = self._latent[..., :previous, :]
            old_bytes = int(self._latent.nbytes)
            steps = (self.step + keys.shape[2] - 1) // self.step
            extension = mx.zeros(
                (
                    keys.shape[0],
                    keys.shape[1],
                    steps * self.step,
                    keys.shape[3],
                ),
                dtype=keys.dtype,
            )
            self._latent = mx.concatenate([self._latent, extension], axis=2)
            self.last_copy_bytes = old_bytes
            self.total_copy_bytes += old_bytes
            self.extension_count += 1
        self.offset = required
        self._latent[..., previous:required, :] = keys
        current = self._latent[..., :required, :]
        return current, current


@dataclass
class _Session:
    layer_id: int
    attention: object
    capture: _CapturingIndexer
    latent_cache: object
    indexer_cache: object
    single: bool
    preallocated: bool
    latent_storage_buffers: int


def _new_indexer_cache(attention, layer_id: int, context: int):
    from mlx_vlm.models.cache import KVCache

    indexer = attention.indexer.delegate
    history = context - 1
    capacity = _capacity(history)
    key_history = _deterministic_rows(
        capacity, indexer.head_dim, 0.125 + layer_id * 0.015625, mx.bfloat16
    )[None]
    gate_history = _deterministic_rows(
        capacity, indexer.head_dim, 0.625 + layer_id * 0.015625, mx.bfloat16
    )[None]
    valid = (mx.arange(capacity) < history)[None, :, None]
    packed = mx.concatenate(
        [key_history, gate_history, valid.astype(mx.bfloat16)], axis=-1
    )
    logical = packed[:, :history]
    keys, gates, valid_channel = mx.split(
        logical, [indexer.head_dim, 2 * indexer.head_dim], axis=-1
    )
    cache = KVCache()
    cache.keys = packed[:, None]
    cache.values = mx.zeros((1, 1, capacity, 0), dtype=mx.bfloat16)
    cache.offset = history
    cache._no_pad = True
    cache._pool = (
        *indexer._pooled_states(keys, gates, valid_channel[..., 0] > 0),
        history,
    )
    mx.eval(packed, *cache._pool[:3])
    return cache


def _new_latent_cache(
    attention, layer_id: int, context: int, *, single: bool, preallocated: bool
):
    from mlx_vlm.models.cache import KVCache

    history = context - 1
    capacity = _capacity(
        history + MEASURED_STEPS if preallocated else history
    )
    latent = _deterministic_rows(
        capacity,
        attention.kv_lora_rank,
        1.125 + layer_id * 0.015625,
        mx.bfloat16,
    ).reshape(1, 1, capacity, attention.kv_lora_rank)
    if single:
        cache = ProbeNoPELatentCache(latent, history)
        mx.eval(cache.keys)
        return cache, 1
    cache = KVCache()
    cache.keys = latent
    if preallocated:
        cache.values = _separate_copy(latent)
        storage_buffers = 2
    else:
        # Reproduce the synthetic restore state used by the previous frontier:
        # K/V initially alias, then current KVCache splits them on extension.
        cache.values = latent
        storage_buffers = 1
    cache.offset = history
    mx.eval(cache.keys, cache.values)
    return cache, storage_buffers


def _build_sessions(attentions, context: int, arm: str):
    config = ARM_CONFIG[arm]
    sessions = []
    for layer_id in EXPECTED_DSA:
        attention = attentions[layer_id]
        latent_cache, buffers = _new_latent_cache(
            attention,
            layer_id,
            context,
            single=config["single"],
            preallocated=config["preallocated"],
        )
        sessions.append(
            _Session(
                layer_id=layer_id,
                attention=attention,
                capture=attention.indexer,
                latent_cache=latent_cache,
                indexer_cache=_new_indexer_cache(attention, layer_id, context),
                single=config["single"],
                preallocated=config["preallocated"],
                latent_storage_buffers=buffers,
            )
        )
    mx.synchronize()
    return sessions


def _latent_storage_bytes(sessions) -> int:
    total = 0
    for session in sessions:
        if session.single or session.latent_storage_buffers == 1:
            total += int(session.latent_cache.keys.nbytes)
        else:
            total += int(session.latent_cache.keys.nbytes)
            total += int(session.latent_cache.values.nbytes)
    return total


def _latent_capacity(session) -> int:
    return int(session.latent_cache.keys.shape[2])


def _latent_copy_bytes(before_capacity: int, session) -> int:
    after_capacity = _latent_capacity(session)
    if after_capacity == before_capacity:
        return 0
    bytes_per_buffer = (
        before_capacity
        * session.attention.kv_lora_rank
        * 2
    )
    if session.single:
        copied = int(session.latent_cache.last_copy_bytes)
        if copied != bytes_per_buffer:
            raise AssertionError((copied, bytes_per_buffer))
        return copied
    session.latent_storage_buffers = 2
    return bytes_per_buffer * 2


def _layer_result(session, output, indices, kv_len: int) -> dict:
    output_values = _np(output.astype(mx.float32), dtype=np.float32)
    return {
        "layer": session.layer_id,
        "index_hash": _hash_indices(indices),
        "output_hash": hashlib.sha256(output_values.tobytes()).hexdigest(),
        "nan_count": int(np.count_nonzero(np.isnan(output_values))),
        **_index_stats(indices, kv_len),
    }


def _actual_step(sessions, context: int, step: int) -> dict:
    kv_len = context + step - 1
    before_capacity = [_latent_capacity(session) for session in sessions]
    outputs = []
    indices = []
    started = time.perf_counter()
    for session in sessions:
        x = _query(session.layer_id, step, session.attention.hidden_size)
        output = session.attention(
            x,
            mask=None,
            cache=[session.latent_cache, session.indexer_cache],
        )
        outputs.append(output)
        indices.append(session.capture.indices)
    mx.eval(*outputs, *indices)
    mx.synchronize()
    elapsed = (time.perf_counter() - started) * 1000.0
    copied = sum(
        _latent_copy_bytes(capacity, session)
        for capacity, session in zip(before_capacity, sessions, strict=True)
    )
    layers = [
        _layer_result(session, output, selected, kv_len)
        for session, output, selected in zip(
            sessions, outputs, indices, strict=True
        )
    ]
    return {
        "step": step,
        "context_tokens": kv_len,
        "context_mod_index_kpool": kv_len % 4,
        "aggregate_latency_ms": elapsed,
        "active_memory_bytes": int(mx.get_active_memory()),
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "unique_latent_storage_bytes": _latent_storage_bytes(sessions),
        "latent_copy_bytes": copied,
        "combined_index_hash": _combined_hash(
            [layer["index_hash"] for layer in layers]
        ),
        "combined_output_hash": _combined_hash(
            [layer["output_hash"] for layer in layers]
        ),
        "layers": layers,
    }


def _phase_latent(sessions, step: int):
    values = []
    current_values = []
    copied = 0
    for session in sessions:
        before = _latent_capacity(session)
        attention = session.attention
        x = _query(session.layer_id, step, attention.hidden_size)
        current = attention.kv_a_layernorm(
            attention.kv_a_proj_with_mqa(x)
        )[:, None]
        latent, _ = session.latent_cache.update_and_fetch(current, current)
        copied += _latent_copy_bytes(before, session)
        values.append(latent)
        current_values.append(current)
    return values, current_values, copied


def _phase_pool_update(sessions, step: int):
    values = []
    for session in sessions:
        attention = session.attention
        x = _query(session.layer_id, step, attention.hidden_size)
        qr = attention.q_a_layernorm(attention.q_a_proj(x))
        pooled = _pool_update(
            session.capture.delegate, x, qr, session.indexer_cache
        )
        pooled["x"] = x
        values.append({"qr": qr, "pooled": pooled})
    return values


def _phase_score(sessions, pooled_rows):
    return [
        _score_phase(session.capture.delegate, row["pooled"])
        for session, row in zip(sessions, pooled_rows, strict=True)
    ]


def _phase_selection(sessions, score_rows):
    return [
        _selection_phase(session.capture.delegate, score)
        for session, score in zip(sessions, score_rows, strict=True)
    ]


def _phase_expansion(sessions, pooled_rows, selected_rows):
    return [
        _expand_phase(session.capture.delegate, pooled["pooled"], selected)
        for session, pooled, selected in zip(
            sessions, pooled_rows, selected_rows, strict=True
        )
    ]


def _phase_gather(latent_rows, index_rows):
    return [
        _gather_phase(latent, indices)
        for latent, indices in zip(latent_rows, index_rows, strict=True)
    ]


def _phase_attention(sessions, pooled_rows, gathered_rows):
    outputs = []
    for session, pooled, gathered in zip(
        sessions, pooled_rows, gathered_rows, strict=True
    ):
        attention = session.attention
        qr = pooled["qr"]
        q = attention.q_b_proj(qr).reshape(
            1, 1, attention.num_heads, attention.q_head_dim
        ).transpose(0, 2, 1, 3)
        outputs.append(
            _attention_phase(attention, q, gathered[0], gathered[1])
        )
    return outputs


def _phased_step(sessions, context: int, step: int) -> dict:
    phase_ms = {}
    latent_result, phase_ms["latent_projection_append"] = _time_phase(
        lambda: _phase_latent(sessions, step),
        lambda result: result[1],
    )
    latent_rows, _, copied = latent_result
    pooled_rows, phase_ms["indexer_pool_update"] = _time_phase(
        lambda: _phase_pool_update(sessions, step),
        lambda rows: [
            (
                row["qr"],
                row["pooled"]["pool_keys"],
                row["pooled"]["pool_indices"],
                row["pooled"]["pool_valid"],
            )
            for row in rows
        ],
    )
    score_rows, phase_ms["pool_score"] = _time_phase(
        lambda: _phase_score(sessions, pooled_rows),
        lambda rows: [
            (row["index_scores"], row["valid_candidates"]) for row in rows
        ],
    )
    selected_rows, phase_ms["selection"] = _time_phase(
        lambda: _phase_selection(sessions, score_rows)
    )
    index_rows, phase_ms["pool_expansion"] = _time_phase(
        lambda: _phase_expansion(sessions, pooled_rows, selected_rows)
    )
    gathered_rows, phase_ms["latent_gather"] = _time_phase(
        lambda: _phase_gather(latent_rows, index_rows)
    )
    outputs, phase_ms["selected_attention"] = _time_phase(
        lambda: _phase_attention(sessions, pooled_rows, gathered_rows)
    )
    kv_len = context + step - 1
    layers = [
        _layer_result(session, output, indices, kv_len)
        for session, output, indices in zip(
            sessions, outputs, index_rows, strict=True
        )
    ]
    return {
        "step": step,
        "context_tokens": kv_len,
        "context_mod_index_kpool": kv_len % 4,
        "phase_ms": phase_ms,
        "synchronized_phase_sum_ms": sum(phase_ms.values()),
        "latent_copy_bytes": copied,
        "combined_index_hash": _combined_hash(
            [layer["index_hash"] for layer in layers]
        ),
        "combined_output_hash": _combined_hash(
            [layer["output_hash"] for layer in layers]
        ),
        "layers": layers,
    }


def _release(sessions) -> None:
    for session in sessions:
        session.capture.indices = None
    sessions.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def _warm(attentions, context: int, arm: str, warmup_steps: int) -> None:
    sessions = _build_sessions(attentions, context, arm)
    for step in range(1, warmup_steps + 1):
        _actual_step(sessions, context, step)
    _release(sessions)


def _summarize_actual(steps, baseline_active: int) -> dict:
    steady = [step["aggregate_latency_ms"] for step in steps[1:]]
    by_mod = {}
    for remainder in range(4):
        values = [
            step["aggregate_latency_ms"]
            for step in steps
            if step["context_mod_index_kpool"] == remainder
        ]
        by_mod[str(remainder)] = {
            "samples_ms": values,
            "median_ms": statistics.median(values),
            "p95_ms": _percentile(values, 95),
        }
    return {
        "first_token_latency_ms": steps[0]["aggregate_latency_ms"],
        "token_2_16_median_ms": statistics.median(steady),
        "token_2_16_p95_ms": _percentile(steady, 95),
        "context_mod_4_latency": by_mod,
        "active_memory_before_decode_bytes": baseline_active,
        "active_memory_after_16_bytes": steps[-1]["active_memory_bytes"],
        "peak_memory_bytes": max(step["peak_memory_bytes"] for step in steps),
        "working_peak_bytes": max(
            0, max(step["peak_memory_bytes"] for step in steps) - baseline_active
        ),
        "memory_drift_16_tokens_bytes": (
            steps[-1]["active_memory_bytes"] - baseline_active
        ),
        "unique_latent_storage_after_16_bytes": steps[-1][
            "unique_latent_storage_bytes"
        ],
        "latent_copy_bytes": sum(step["latent_copy_bytes"] for step in steps),
        "steps": steps,
    }


def _summarize_phases(steps) -> dict:
    medians = {
        phase: statistics.median(step["phase_ms"][phase] for step in steps[1:])
        for phase in PHASES
    }
    p95 = {
        phase: _percentile(
            [step["phase_ms"][phase] for step in steps[1:]], 95
        )
        for phase in PHASES
    }
    return {
        "token_2_16_phase_median_ms": medians,
        "token_2_16_phase_p95_ms": p95,
        "steps": steps,
    }


def _measure_arm(attentions, context: int, arm: str, warmup_steps: int) -> dict:
    _progress("warm", context=context, arm=arm, steps=warmup_steps)
    _warm(attentions, context, arm, warmup_steps)

    _progress("actual_session", context=context, arm=arm)
    actual_sessions = _build_sessions(attentions, context, arm)
    baseline_active = int(mx.get_active_memory())
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
    actual_summary = _summarize_actual(actual_steps, baseline_active)
    _release(actual_sessions)

    _progress("phased_session", context=context, arm=arm)
    phase_sessions = _build_sessions(attentions, context, arm)
    phase_steps = [
        _phased_step(phase_sessions, context, step)
        for step in range(1, MEASURED_STEPS + 1)
    ]
    phase_summary = _summarize_phases(phase_steps)
    _release(phase_sessions)

    parity = []
    for actual, phased in zip(actual_steps, phase_steps, strict=True):
        parity.append(
            {
                "step": actual["step"],
                "index_hash_match": actual["combined_index_hash"]
                == phased["combined_index_hash"],
                "output_hash_match": actual["combined_output_hash"]
                == phased["combined_output_hash"],
                "layers": [
                    {
                        "layer": left["layer"],
                        "index_hash_match": left["index_hash"]
                        == right["index_hash"],
                        "output_hash_match": left["output_hash"]
                        == right["output_hash"],
                    }
                    for left, right in zip(
                        actual["layers"], phased["layers"], strict=True
                    )
                ],
            }
        )
    boundary_step = 1 if context == 2049 else 2
    return {
        "actual": actual_summary,
        "phases": phase_summary,
        "manual_phase_current_attention_parity": parity,
        "capacity_boundary": {
            "step": boundary_step,
            "context_tokens": context + boundary_step - 1,
            "aggregate_latency_ms": actual_steps[boundary_step - 1][
                "aggregate_latency_ms"
            ],
            "phase_ms": phase_steps[boundary_step - 1]["phase_ms"],
            "latent_copy_bytes": actual_steps[boundary_step - 1][
                "latent_copy_bytes"
            ],
        },
    }


def _cross_arm_parity(arms: dict) -> dict:
    baseline = arms["dual_kv_step256"]["actual"]["steps"]
    result = {}
    for arm, row in arms.items():
        comparisons = []
        for expected, actual in zip(
            baseline, row["actual"]["steps"], strict=True
        ):
            comparisons.append(
                {
                    "step": expected["step"],
                    "index_hash_match": expected["combined_index_hash"]
                    == actual["combined_index_hash"],
                    "output_hash_match": expected["combined_output_hash"]
                    == actual["combined_output_hash"],
                }
            )
        result[arm] = comparisons
    return result


def _context_case(attentions, context: int, warmup_steps: int) -> dict:
    arms = {
        arm: _measure_arm(attentions, context, arm, warmup_steps)
        for arm in ARM_CONFIG
    }
    return {
        "context_tokens": context,
        "arms": arms,
        "current_kvcache_cross_arm_parity": _cross_arm_parity(arms),
    }


def _all_layer_rows(cases):
    return [
        layer
        for case in cases.values()
        for arm in case["arms"].values()
        for step in arm["actual"]["steps"]
        for layer in step["layers"]
    ]


def _phase_retentions(cases) -> dict:
    output = {}
    for arm in ARM_CONFIG:
        baseline = cases["2049"]["arms"][arm]["phases"][
            "token_2_16_phase_median_ms"
        ]
        largest = cases["262144"]["arms"][arm]["phases"][
            "token_2_16_phase_median_ms"
        ]
        output[arm] = {
            phase: baseline[phase] / largest[phase] for phase in PHASES
        }
    return output


def _decision(cases, phase_retentions) -> dict:
    largest = cases["262144"]["arms"]
    dual_step = largest["dual_kv_step256"]["actual"]
    dual_pre = largest["dual_kv_preallocated"]["actual"]
    single_step = largest["single_latent_step256"]["actual"]
    single_pre = largest["single_latent_preallocated"]["actual"]
    storage_saved = (
        dual_step["unique_latent_storage_after_16_bytes"]
        - single_step["unique_latent_storage_after_16_bytes"]
    )
    step_steady_ratio = (
        single_step["token_2_16_median_ms"]
        / dual_step["token_2_16_median_ms"]
    )
    pre_steady_ratio = (
        single_pre["token_2_16_median_ms"]
        / dual_pre["token_2_16_median_ms"]
    )
    baseline_pre = cases["2049"]["arms"]["single_latent_preallocated"][
        "actual"
    ]["token_2_16_median_ms"]
    preallocated_retention = (
        baseline_pre / single_pre["token_2_16_median_ms"]
    )
    phase_2049 = cases["2049"]["arms"]["single_latent_preallocated"][
        "phases"
    ]["token_2_16_phase_median_ms"]
    phase_256k = largest["single_latent_preallocated"]["phases"][
        "token_2_16_phase_median_ms"
    ]
    dominant_delta = max(
        PHASES, key=lambda phase: phase_256k[phase] - phase_2049[phase]
    )
    current_boundary = largest["dual_kv_step256"]["capacity_boundary"]
    single_step_boundary = largest["single_latent_step256"]["capacity_boundary"]
    single_pre_boundary = largest["single_latent_preallocated"][
        "capacity_boundary"
    ]
    single_step_boundary_ratio = (
        single_step_boundary["aggregate_latency_ms"]
        / current_boundary["aggregate_latency_ms"]
    )
    single_pre_boundary_ratio = (
        single_pre_boundary["aggregate_latency_ms"]
        / current_boundary["aggregate_latency_ms"]
    )
    runtime_candidate = {
        "all_step_byte_identical": None,
        "storage_reduction_at_least_2_95gb": storage_saved >= 2.95e9,
        "single_step256_capacity_copy_bytes_reduced": single_step_boundary[
            "latent_copy_bytes"
        ]
        < current_boundary["latent_copy_bytes"],
        "preallocated_capacity_copy_bytes_eliminated": single_pre_boundary[
            "latent_copy_bytes"
        ]
        == 0,
        "preallocated_boundary_latency_reduced_at_least_5pct": (
            single_pre_boundary_ratio <= 0.95
        ),
        "preallocated_working_peak_reduced": single_pre["working_peak_bytes"]
        < dual_step["working_peak_bytes"],
        "steady_step256_not_over_5pct_slower": step_steady_ratio <= 1.05,
        "steady_preallocated_not_over_5pct_slower": pre_steady_ratio <= 1.05,
    }
    return {
        "single_buffer_storage_saved_256k_bytes": storage_saved,
        "single_to_dual_step256_steady_latency_ratio": step_steady_ratio,
        "single_to_dual_preallocated_steady_latency_ratio": pre_steady_ratio,
        "single_step256_to_dual_step256_boundary_latency_ratio": (
            single_step_boundary_ratio
        ),
        "single_preallocated_to_dual_step256_boundary_latency_ratio": (
            single_pre_boundary_ratio
        ),
        "single_step256_boundary_latency_materially_reduced": (
            single_step_boundary_ratio <= 0.95
        ),
        "single_preallocated_retention_2049_to_256k": preallocated_retention,
        "retention_target": 0.80,
        "dominant_phase_growth_if_retention_fails": dominant_delta,
        "phase_retentions": phase_retentions,
        "runtime_candidate_gate": runtime_candidate,
        "recommended_probe_candidate": (
            "single NoPE latent buffer with 16-token allocation headroom"
        ),
        "next_measurement": (
            f"localize {dominant_delta}"
            if preallocated_retention < 0.80
            else "full-model synthetic-cache decode frontier"
        ),
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
    if tuple(report.dsa_layers) != EXPECTED_DSA:
        raise RuntimeError(f"unexpected DSA layers: {report.dsa_layers}")
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
    phase_retentions = _phase_retentions(cases)
    decision = _decision(cases, phase_retentions)

    layer_rows = _all_layer_rows(cases)
    manual_parity = [
        row
        for case in cases.values()
        for arm in case["arms"].values()
        for row in arm["manual_phase_current_attention_parity"]
    ]
    cross_parity = [
        row
        for case in cases.values()
        for arm in case["current_kvcache_cross_arm_parity"].values()
        for row in arm
    ]
    all_step_identical = all(
        row["index_hash_match"] and row["output_hash_match"]
        for row in cross_parity
    )
    decision["runtime_candidate_gate"][
        "all_step_byte_identical"
    ] = all_step_identical
    decision["runtime_candidate_qualified"] = all(
        decision["runtime_candidate_gate"].values()
    )
    acceptance = {
        "all_11_dsa_layers_measured": all(
            layer_ids == list(EXPECTED_DSA)
            for case in cases.values()
            for arm in case["arms"].values()
            for layer_ids in (
                [layer["layer"] for layer in arm["actual"]["steps"][0]["layers"]],
            )
        ),
        "all_4_arms_3_contexts_16_steps_measured": set(map(int, cases))
        == set(CONTEXTS)
        and all(
            set(case["arms"]) == set(ARM_CONFIG)
            and all(
                len(arm["actual"]["steps"]) == MEASURED_STEPS
                and len(arm["phases"]["steps"]) == MEASURED_STEPS
                for arm in case["arms"].values()
            )
            for case in cases.values()
        ),
        "warmup_covers_pool_tail_mod_0_1_2_3": args.warmup_steps >= 4,
        "manual_phase_decomposition_matches_current_attention": all(
            row["index_hash_match"]
            and row["output_hash_match"]
            and all(
                layer["index_hash_match"] and layer["output_hash_match"]
                for layer in row["layers"]
            )
            for row in manual_parity
        ),
        "all_arms_match_current_dual_kvcache": all_step_identical,
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
        "runtime_server_apc_admission_unchanged": LEGACY_PROBE_MAX_PROMPT_TOKENS == 256,
    }
    acceptance["accepted"] = all(acceptance.values())
    output = {
        "schema": "glm53-single-buffer-nope-latent-cache-frontier-v1",
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
        "arms": ARM_CONFIG,
        "phase_contract": {
            "latent_projection_append": "kv_a projection + norm + lazy append registration; sync current token only",
            "indexer_pool_update": "q_a projection + norm + indexer token append + incremental pooling",
            "pool_score": "index query/weights projection + all-pool score",
            "selection": "full argsort + exact top-k pool selection",
            "pool_expansion": "selected pools to sentinel-safe token indices",
            "latent_gather": "sentinel-aware latent selection",
            "selected_attention": "q_b + embed_q + SDPA + unembed_out + o_proj",
            "phase_timings_synchronize_each_boundary": True,
            "capacity_copy_timed_by_actual_boundary_not_full_cache_sync": True,
            "actual_aggregate_syncs_all_11_outputs_once": True,
        },
        "cases": cases,
        "decision_gate": decision,
        "runtime_policy": {
            "probe_only_single_latent_cache": True,
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
                "runtime_candidate_qualified": decision[
                    "runtime_candidate_qualified"
                ],
                "storage_saved_256k_bytes": decision[
                    "single_buffer_storage_saved_256k_bytes"
                ],
                "single_preallocated_retention": decision[
                    "single_preallocated_retention_2049_to_256k"
                ],
                "next_measurement": decision["next_measurement"],
            },
            indent=2,
        )
    )
    for attention in attentions.values():
        attention.indexer = attention.indexer.delegate
    return 0 if acceptance["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
