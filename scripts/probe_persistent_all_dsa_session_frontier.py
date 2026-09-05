#!/usr/bin/env python3
"""Characterize persistent S=1 sessions across every GLM-5.3 DSA layer."""

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
from glm53_flash_mlx.indexpool import INDEXPOOL_SENTINEL
from glm53_flash_mlx.loader import load
from glm53_flash_mlx.manifest import EXPECTED_DSA, inspect_checkpoint
from glm53_flash_mlx.server import LEGACY_PROBE_MAX_PROMPT_TOKENS

CONTEXTS = (2049, 32768, 65536, 131072, 262144)
STEPS = 16
ARMS = ("resident_steady", "restored_session")


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), file=sys.stderr, flush=True)


def _deterministic_rows(rows: int, width: int, phase: float, dtype) -> mx.array:
    positions = mx.arange(rows, dtype=mx.float32)[:, None]
    columns = mx.arange(width, dtype=mx.float32)[None, :]
    values = mx.sin(positions * 0.0009765625 + columns * 0.0078125 + phase)
    return values.astype(dtype)


def _capacity(tokens: int, step: int = 256) -> int:
    return ((tokens + step - 1) // step) * step if tokens else 0


def _np(value, *, dtype=None) -> np.ndarray:
    if value.dtype == mx.bfloat16:
        value = value.astype(mx.float32)
    mx.eval(value)
    array = np.ascontiguousarray(np.asarray(value))
    return array.astype(dtype, copy=False) if dtype is not None else array


def _hash_indices(value) -> str:
    array = _np(value, dtype=np.int32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _combined_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _array_sample(value: mx.array, sequence_axis: int) -> dict:
    length = value.shape[sequence_axis]
    positions = sorted({0, max(0, length // 2), max(0, length - 1)})
    if not length or value.size == 0:
        payload = b""
    else:
        indices = mx.array(positions, dtype=mx.int32)
        payload = _np(mx.take(value, indices, axis=sequence_axis)).tobytes()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sample_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _cache_snapshot(latent_cache, indexer_cache) -> dict:
    snapshot = {
        "latent_offset": int(latent_cache.offset),
        "indexer_offset": int(indexer_cache.offset),
        "latent_keys": _array_sample(latent_cache.keys, 2),
        "latent_values": _array_sample(latent_cache.values, 2),
        "indexer_keys": _array_sample(indexer_cache.keys, 2),
        "indexer_values": _array_sample(indexer_cache.values, 2),
        "no_pad": bool(getattr(indexer_cache, "_no_pad", False)),
        "pool": None,
    }
    pool = getattr(indexer_cache, "_pool", None)
    if pool is not None:
        snapshot["pool"] = {
            "previous_tokens": int(pool[3]),
            "keys": _array_sample(pool[0], 1),
            "indices": _array_sample(pool[1], 1),
            "valid": _array_sample(pool[2], 1),
        }
    return snapshot


def _pool_bytes(sessions) -> dict[str, int]:
    by_dtype: dict[str, int] = {}
    for session in sessions:
        pool = getattr(session.indexer_cache, "_pool", None)
        if pool is None:
            continue
        for value in pool[:3]:
            dtype = str(value.dtype).removeprefix("mlx.core.")
            by_dtype[dtype] = by_dtype.get(dtype, 0) + int(value.nbytes)
    return dict(sorted(by_dtype.items()))


def _index_stats(indices, kv_len: int) -> dict:
    values = _np(indices, dtype=np.int32)
    valid = values != INDEXPOOL_SENTINEL
    invalid = valid & ((values < 0) | (values >= kv_len))
    valid_values = values[valid]
    return {
        "selected_token_width": int(values.shape[-1]),
        "valid_index_min": int(valid_values.min()) if valid_values.size else None,
        "valid_index_max": int(valid_values.max()) if valid_values.size else None,
        "valid_index_count": int(np.count_nonzero(valid)),
        "sentinel_count": int(np.count_nonzero(~valid)),
        "non_sentinel_out_of_range": int(np.count_nonzero(invalid)),
    }


def _expected_tail_sentinels(kv_len: int, index_kpool: int) -> int:
    return index_kpool - 1 - (kv_len % index_kpool)


def _array_equal(left: mx.array, right: mx.array) -> bool:
    if left.shape != right.shape:
        return False
    if left.size == 0:
        return True
    if left.dtype != right.dtype:
        return False
    result = mx.array_equal(left, right)
    mx.eval(result)
    return bool(result.item())


class _CapturingIndexer:
    def __init__(self, delegate):
        self.delegate = delegate
        self.indices = None

    def __call__(self, *args, **kwargs):
        self.indices = self.delegate(*args, **kwargs)
        return self.indices


@dataclass
class _LayerSession:
    layer_id: int
    attention: object
    capture: _CapturingIndexer
    latent_cache: object
    indexer_cache: object
    cache_identity: tuple[int, int]


def _new_cache(keys: mx.array, values: mx.array, offset: int):
    from mlx_vlm.models.cache import KVCache

    cache = KVCache()
    cache.keys = keys
    cache.values = values
    cache.offset = offset
    return cache


def _build_layer_session(attention, layer_id: int, context: int, arm: str):
    indexer = attention.indexer.delegate
    history = context - 1
    capacity = _capacity(history)
    key_history = _deterministic_rows(
        capacity, indexer.head_dim, 0.125 + layer_id * 0.015625, mx.bfloat16
    )[None]
    gate_history = _deterministic_rows(
        capacity, indexer.head_dim, 0.625 + layer_id * 0.015625, mx.bfloat16
    )[None]
    valid_history = (mx.arange(capacity) < history)[None, :, None]
    packed_history = mx.concatenate(
        [key_history, gate_history, valid_history.astype(mx.bfloat16)], axis=-1
    )
    latent_history = _deterministic_rows(
        capacity,
        attention.kv_lora_rank,
        1.125 + layer_id * 0.015625,
        mx.bfloat16,
    ).reshape(1, 1, capacity, attention.kv_lora_rank)
    indexer_cache = _new_cache(
        packed_history[:, None],
        mx.zeros((1, 1, capacity, 0), dtype=mx.bfloat16),
        history,
    )
    latent_cache = _new_cache(latent_history, latent_history, history)
    indexer_cache._no_pad = True
    if arm == "resident_steady":
        logical = packed_history[:, :history]
        keys, gates, valid = mx.split(
            logical, [indexer.head_dim, 2 * indexer.head_dim], axis=-1
        )
        indexer_cache._pool = (
            *indexer._pooled_states(keys, gates, valid[..., 0] > 0),
            history,
        )
        materialize = [
            packed_history,
            latent_history,
            *indexer_cache._pool[:3],
        ]
    else:
        indexer_cache._pool = None
        materialize = [packed_history, latent_history]
    mx.eval(*materialize)
    return _LayerSession(
        layer_id=layer_id,
        attention=attention,
        capture=attention.indexer,
        latent_cache=latent_cache,
        indexer_cache=indexer_cache,
        cache_identity=(id(latent_cache), id(indexer_cache)),
    )


def _build_sessions(attentions: dict[int, object], context: int, arm: str):
    sessions = [
        _build_layer_session(attentions[layer], layer, context, arm)
        for layer in EXPECTED_DSA
    ]
    mx.synchronize()
    return sessions


def _query(layer_id: int, step: int, hidden_size: int) -> mx.array:
    phase = 1.625 + layer_id * 0.03125 + step * 0.125
    return _deterministic_rows(1, hidden_size, phase, mx.bfloat16)[None]


def _prefix_state(sessions) -> list[dict]:
    rows = []
    for session in sessions:
        rows.append(
            {
                "layer": session.layer_id,
                "latent_offset": int(session.latent_cache.offset),
                "indexer_offset": int(session.indexer_cache.offset),
                "latent_capacity": int(session.latent_cache.keys.shape[2]),
                "indexer_capacity": int(session.indexer_cache.keys.shape[2]),
                "latent_keys": session.latent_cache.keys,
                "latent_values": session.latent_cache.values,
                "indexer_keys": session.indexer_cache.keys,
                "indexer_values": session.indexer_cache.values,
            }
        )
    return rows


def _capacity_extension_checks(before: list[dict], sessions) -> list[dict]:
    events = []
    for previous, session in zip(before, sessions, strict=True):
        checks = []
        for name, cache in (
            ("latent", session.latent_cache),
            ("indexer", session.indexer_cache),
        ):
            old_capacity = previous[f"{name}_capacity"]
            new_capacity = int(cache.keys.shape[2])
            if new_capacity == old_capacity:
                continue
            offset = previous[f"{name}_offset"]
            keys_unchanged = _array_equal(
                previous[f"{name}_keys"][..., :offset, :],
                cache.keys[..., :offset, :],
            )
            values_unchanged = _array_equal(
                previous[f"{name}_values"][..., :offset, :],
                cache.values[..., :offset, :],
            )
            checks.append(
                {
                    "cache": name,
                    "old_capacity": old_capacity,
                    "new_capacity": new_capacity,
                    "existing_keys_unchanged": keys_unchanged,
                    "existing_values_unchanged": values_unchanged,
                }
            )
        if checks:
            events.append({"layer": session.layer_id, "checks": checks})
    return events


def _idle_state_check(sessions) -> dict:
    before = [
        _cache_snapshot(session.latent_cache, session.indexer_cache)
        for session in sessions
    ]
    active_before = int(mx.get_active_memory())
    peak_before = int(mx.get_peak_memory())
    mx.synchronize()
    active_after = int(mx.get_active_memory())
    peak_after = int(mx.get_peak_memory())
    after = [
        _cache_snapshot(session.latent_cache, session.indexer_cache)
        for session in sessions
    ]
    return {
        "state_unchanged": before == after,
        "active_memory_before": active_before,
        "active_memory_after": active_after,
        "peak_memory_before": peak_before,
        "peak_memory_after": peak_after,
    }


def _run_step(sessions, context: int, step: int) -> dict:
    kv_len = context + step - 1
    prefix = _prefix_state(sessions)
    pool_present_before = [
        getattr(session.indexer_cache, "_pool", None) is not None
        for session in sessions
    ]
    previous_pool_tokens = [
        (
            int(session.indexer_cache._pool[3])
            if getattr(session.indexer_cache, "_pool", None) is not None
            else None
        )
        for session in sessions
    ]
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
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    extension_events = _capacity_extension_checks(prefix, sessions)

    layer_rows = []
    for session, output, selected, pool_before, pool_tokens in zip(
        sessions,
        outputs,
        indices,
        pool_present_before,
        previous_pool_tokens,
        strict=True,
    ):
        output_values = _np(output.astype(mx.float32), dtype=np.float32)
        layer_rows.append(
            {
                "layer": session.layer_id,
                "index_hash": _hash_indices(selected),
                "output_hash": hashlib.sha256(output_values.tobytes()).hexdigest(),
                "nan_count": int(np.count_nonzero(np.isnan(output_values))),
                "pool_present_before": pool_before,
                "previous_pool_tokens": pool_tokens,
                "pool_present_after": session.indexer_cache._pool is not None,
                "pool_tokens_after": int(session.indexer_cache._pool[3]),
                **_index_stats(selected, kv_len),
            }
        )
    index_hashes = [row["index_hash"] for row in layer_rows]
    output_hashes = [row["output_hash"] for row in layer_rows]
    return {
        "step": step,
        "context_tokens": kv_len,
        "context_mod_index_kpool": kv_len % 4,
        "aggregate_dsa_latency_ms": elapsed_ms,
        "active_memory_bytes": int(mx.get_active_memory()),
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "pool_state_bytes_by_dtype": _pool_bytes(sessions),
        "combined_index_hash": _combined_hash(index_hashes),
        "combined_output_hash": _combined_hash(output_hashes),
        "layers": layer_rows,
        "capacity_extension_events": extension_events,
        "cache_object_identity_preserved": all(
            session.cache_identity
            == (id(session.latent_cache), id(session.indexer_cache))
            for session in sessions
        ),
    }


def _summarize_arm(arm: str, context: int, sessions, idle: dict) -> dict:
    baseline_active = int(mx.get_active_memory())
    mx.reset_peak_memory()
    steps = []
    for step in range(1, STEPS + 1):
        row = _run_step(sessions, context, step)
        steps.append(row)
        _progress(
            "session_step",
            context=context,
            arm=arm,
            step=step,
            milliseconds=row["aggregate_dsa_latency_ms"],
        )
    steady = [row["aggregate_dsa_latency_ms"] for row in steps[1:]]
    by_mod = {}
    for remainder in range(4):
        values = [
            row["aggregate_dsa_latency_ms"]
            for row in steps
            if row["context_mod_index_kpool"] == remainder
        ]
        by_mod[str(remainder)] = {
            "samples_ms": values,
            "median_ms": statistics.median(values),
            "p95_ms": _percentile(values, 95),
        }
    final_active = int(mx.get_active_memory())
    return {
        "first_token_latency_ms": steps[0]["aggregate_dsa_latency_ms"],
        "token_2_16_median_ms": statistics.median(steady),
        "token_2_16_p95_ms": _percentile(steady, 95),
        "context_mod_4_latency": by_mod,
        "active_memory_before_decode_bytes": baseline_active,
        "active_memory_after_16_bytes": final_active,
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "memory_drift_16_tokens_bytes": final_active - baseline_active,
        "pool_state_bytes_after_16_by_dtype": _pool_bytes(sessions),
        "pool_state_bytes_after_16_total": sum(_pool_bytes(sessions).values()),
        "idle_measurement": idle,
        "steps": steps,
    }


def _release_sessions(sessions) -> None:
    for session in sessions:
        session.capture.indices = None
    sessions.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def _warm_arm(attentions, context: int, arm: str, warmup_steps: int) -> None:
    sessions = _build_sessions(attentions, context, arm)
    for step in range(1, warmup_steps + 1):
        _run_step(sessions, context, step)
    _release_sessions(sessions)


def _measure_arm(attentions, context: int, arm: str, warmup_steps: int) -> dict:
    _progress("warm_arm", context=context, arm=arm, steps=warmup_steps)
    _warm_arm(attentions, context, arm, warmup_steps)
    _progress("build_measured_session", context=context, arm=arm)
    sessions = _build_sessions(attentions, context, arm)
    idle = _idle_state_check(sessions)
    result = _summarize_arm(arm, context, sessions, idle)
    _release_sessions(sessions)
    return result


def _step_parity(resident: dict, restored: dict) -> list[dict]:
    rows = []
    for left, right in zip(resident["steps"], restored["steps"], strict=True):
        layer_parity = []
        for left_layer, right_layer in zip(
            left["layers"], right["layers"], strict=True
        ):
            layer_parity.append(
                {
                    "layer": left_layer["layer"],
                    "index_hash_match": left_layer["index_hash"]
                    == right_layer["index_hash"],
                    "output_hash_match": left_layer["output_hash"]
                    == right_layer["output_hash"],
                }
            )
        rows.append(
            {
                "step": left["step"],
                "context_tokens": left["context_tokens"],
                "context_mod_index_kpool": left["context_mod_index_kpool"],
                "combined_index_hash_match": left["combined_index_hash"]
                == right["combined_index_hash"],
                "combined_output_hash_match": left["combined_output_hash"]
                == right["combined_output_hash"],
                "layers": layer_parity,
            }
        )
    return rows


def _context_case(attentions, context: int, warmup_steps: int, disk_gbps: float):
    arms = {
        arm: _measure_arm(attentions, context, arm, warmup_steps) for arm in ARMS
    }
    parity = _step_parity(arms["resident_steady"], arms["restored_session"])
    resident = arms["resident_steady"]
    restored = arms["restored_session"]
    rebuild_extra = (
        restored["first_token_latency_ms"] - resident["first_token_latency_ms"]
    )
    pool_bytes = restored["pool_state_bytes_after_16_total"]
    ideal_disk_floor = pool_bytes / (disk_gbps * 1e9) * 1000.0
    steady_ratio = (
        restored["token_2_16_median_ms"]
        / resident["token_2_16_median_ms"]
    )
    extension_events = [
        event
        for arm in arms.values()
        for step in arm["steps"]
        for event in step["capacity_extension_events"]
    ]
    return {
        "initial_context_tokens": context,
        "steps": STEPS,
        "arms": arms,
        "resident_restored_step_parity": parity,
        "pool_rebuild_additional_ms": rebuild_extra,
        "restored_to_resident_steady_ratio": steady_ratio,
        "rebuild_is_first_token_only": steady_ratio <= 1.10,
        "pool_payload_bytes": pool_bytes,
        "optimistic_disk_bandwidth_gbps": disk_gbps,
        "pool_payload_ideal_io_floor_ms": ideal_disk_floor,
        "rebuild_cheaper_than_ideal_disk_io_floor": max(0.0, rebuild_extra)
        < ideal_disk_floor,
        "capacity_extension_event_count": len(extension_events),
        "capacity_extensions_preserved_existing_state": bool(extension_events)
        and all(
            check["existing_keys_unchanged"]
            and check["existing_values_unchanged"]
            for event in extension_events
            for check in event["checks"]
        ),
    }


def _all_layer_rows(cases: dict):
    return [
        layer
        for case in cases.values()
        for arm in case["arms"].values()
        for step in arm["steps"]
        for layer in step["layers"]
    ]


def _decision(cases: dict) -> dict:
    baseline = cases["2049"]["arms"]["resident_steady"][
        "token_2_16_median_ms"
    ]
    largest = cases["262144"]
    largest_steady = largest["arms"]["resident_steady"][
        "token_2_16_median_ms"
    ]
    retention = baseline / largest_steady
    disk_rebuild = (
        largest["rebuild_is_first_token_only"]
        and largest["rebuild_cheaper_than_ideal_disk_io_floor"]
    )
    disk_rebuild_contexts = [
        int(context)
        for context, case in cases.items()
        if case["rebuild_is_first_token_only"]
        and case["rebuild_cheaper_than_ideal_disk_io_floor"]
    ]
    return {
        "steady_aggregate_retention_2049_to_256k": retention,
        "steady_retention_target": 0.80,
        "next_if_steady_target_met": "full-model synthetic-cache decode frontier",
        "next_measurement": (
            "full-model synthetic-cache decode frontier"
            if retention >= 0.80
            else "localize all-DSA steady degradation"
        ),
        "disk_apc_pool_policy": (
            "at 256k rebuild pool from persisted indexer token state"
            if disk_rebuild
            else "inconclusive; do not persist pool or change APC ABI"
        ),
        "contexts_where_rebuild_beats_ideal_io_floor": disk_rebuild_contexts,
        "ram_apc_zero_copy_available": False,
        "ram_apc_pool_policy": "not considered without a zero-copy retention path",
        "apc_abi_changed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--optimistic-disk-gbps", type=float, default=7.0)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

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
        str(context): _context_case(
            attentions,
            context,
            args.warmup_steps,
            args.optimistic_disk_gbps,
        )
        for context in CONTEXTS
    }
    baseline = cases["2049"]["arms"]["resident_steady"][
        "token_2_16_median_ms"
    ]
    for case in cases.values():
        steady = case["arms"]["resident_steady"]
        steady["aggregate_retention_from_2049"] = (
            baseline / steady["token_2_16_median_ms"]
        )

    all_layers = _all_layer_rows(cases)
    all_steps = [
        step
        for case in cases.values()
        for arm in case["arms"].values()
        for step in arm["steps"]
    ]
    parity_rows = [
        row
        for case in cases.values()
        for row in case["resident_restored_step_parity"]
    ]
    acceptance = {
        "all_11_dsa_layers_measured": all(
            [layer["layer"] for layer in step["layers"]] == list(EXPECTED_DSA)
            for step in all_steps
        ),
        "all_contexts_16_tokens_measured": set(map(int, cases)) == set(CONTEXTS)
        and all(
            len(arm["steps"]) == STEPS
            for case in cases.values()
            for arm in case["arms"].values()
        ),
        "all_indices_sentinel_or_in_range": all(
            layer["non_sentinel_out_of_range"] == 0 for layer in all_layers
        ),
        "all_unused_slots_are_minus1": all(
            layer["sentinel_count"]
            == _expected_tail_sentinels(
                step["context_tokens"], report.index_kpool
            )
            for case in cases.values()
            for arm in case["arms"].values()
            for step in arm["steps"]
            for layer in step["layers"]
        ),
        "no_nan": all(layer["nan_count"] == 0 for layer in all_layers),
        "resident_restored_step_parity": all(
            row["combined_index_hash_match"]
            and row["combined_output_hash_match"]
            and all(
                layer["index_hash_match"] and layer["output_hash_match"]
                for layer in row["layers"]
            )
            for row in parity_rows
        ),
        "pool_boundary_mod_0_1_2_3_parity": {
            row["context_mod_index_kpool"] for row in parity_rows
        }
        == {0, 1, 2, 3}
        and all(
            row["combined_index_hash_match"]
            and row["combined_output_hash_match"]
            for row in parity_rows
        ),
        "cache_capacity_extension_preserves_existing_state": all(
            case["capacity_extensions_preserved_existing_state"]
            for case in cases.values()
        ),
        "idle_measurement_does_not_mutate_state": all(
            arm["idle_measurement"]["state_unchanged"]
            for case in cases.values()
            for arm in case["arms"].values()
        ),
        "cache_object_identity_preserved_across_tokens": all(
            step["cache_object_identity_preserved"] for step in all_steps
        ),
        "restored_pool_reused_after_first_token": all(
            all(
                not layer["pool_present_before"]
                for layer in case["arms"]["restored_session"]["steps"][0][
                    "layers"
                ]
            )
            and all(
                layer["pool_present_before"]
                for step in case["arms"]["restored_session"]["steps"][1:]
                for layer in step["layers"]
            )
            for case in cases.values()
        ),
        "runtime_server_apc_admission_unchanged": LEGACY_PROBE_MAX_PROMPT_TOKENS == 256,
    }
    acceptance["accepted"] = all(acceptance.values())
    decision = _decision(cases)
    output = {
        "schema": "glm53-persistent-all-dsa-session-frontier-v1",
        "date": date.today().isoformat(),
        "machine": "Apple M3 Ultra, 80-core GPU, 512 GB unified memory",
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "nope_cache_abi": NOPE_DSA_CACHE_ABI,
        "dsa_layers": list(EXPECTED_DSA),
        "contexts": list(CONTEXTS),
        "decode_tokens": STEPS,
        "index_topk": report.index_topk,
        "index_kpool": report.index_kpool,
        "measurement_contract": {
            "cache_source": "deterministic latent/indexer state; no long prefill",
            "cache_objects_persist_across_all_16_tokens": True,
            "resident_pool_present_before_token_1": True,
            "restored_pool_absent_only_before_token_1": True,
            "warmup_steps_on_disposable_sessions": args.warmup_steps,
            "aggregate_timing_syncs_all_11_layer_outputs_once_per_step": True,
            "pool_payload_io_floor_is_optimistic_not_observed_disk_io": True,
        },
        "cases": cases,
        "decision_gate": decision,
        "runtime_policy": {
            "default_backend": "direct",
            "prompt_limit": LEGACY_PROBE_MAX_PROMPT_TOKENS,
            "server_changed": False,
            "apc_abi_changed": False,
            "admission_changed": False,
            "pool_state_persisted": False,
        },
        "acceptance": acceptance,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "accepted": acceptance["accepted"],
                "steady_256k_retention": decision[
                    "steady_aggregate_retention_2049_to_256k"
                ],
                "next_measurement": decision["next_measurement"],
                "disk_apc_pool_policy": decision["disk_apc_pool_policy"],
            },
            indent=2,
        )
    )
    for attention in attentions.values():
        attention.indexer = attention.indexer.delegate
    return 0 if acceptance["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
