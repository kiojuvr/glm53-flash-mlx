#!/usr/bin/env python3
"""Profile Lightning Indexer decode without changing production execution.

The model phase separates synchronized A--G diagnostics from unsynchronized
full-model timing.  A counterfactual arm still performs the authoritative
compact-cache update but replays the exact selected indices, measuring the
maximum benefit available from removing query/score/top-k/expansion work.
Per-stage synchronization is diagnostic and is never summed as token wall.
"""

from __future__ import annotations

import argparse
import contextlib
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

from glm53_flash_mlx.abi import MLX_VLM_REVISION, NOPE_DSA_CACHE_ABI_COMPACT
from glm53_flash_mlx.indexpool import (
    INDEXPOOL_SENTINEL,
    expand_selected_pools,
    prepare_decode_indexpool_gather,
)
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import EXPECTED_DSA, inspect_checkpoint
from glm53_flash_mlx.nope_cache import CompactIndexPoolCache


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-lightning-indexer-decode-critical-path-20260905.json"
)
CONTEXTS = (2_048, 32_768, 131_072, 262_144)
PHASES = (
    "index_query_projection",
    "indexpool_update",
    "pooled_key_score",
    "topk_selection",
    "pool_token_expansion",
    "sanitize_gather_preparation",
    "sparse_attention",
)
INDEXER_PHASES = (
    "index_query_projection",
    "indexpool_update",
    "pooled_key_score",
    "topk_selection",
    "pool_token_expansion",
    "sanitize_gather_preparation",
)
PROFILE_TOKEN = 3000


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


def _load_probe_modules():
    scripts = str(REPOSITORY / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import probe_exact_sigmoid_gate_metal_barrier as oracle_probe
    import probe_long_context_dsa_decode_frontier as operator_probe
    import probe_long_context_first_decode_boundary as boundary_probe

    return oracle_probe, operator_probe, boundary_probe


def _arrays(value):
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _arrays(value[key])
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _arrays(item)


def _eval(value) -> None:
    arrays = list(_arrays(value))
    if arrays:
        mx.eval(*arrays)
    mx.synchronize()


def _hash(value: mx.array) -> str:
    mx.eval(value)
    materialized = np.ascontiguousarray(
        np.asarray(
            value.astype(mx.float32)
            if value.dtype == mx.bfloat16
            else value
        )
    )
    return hashlib.sha256(materialized.tobytes()).hexdigest()


def _release(*values) -> None:
    for value in values:
        if isinstance(value, list):
            value.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def _temporary_bytes(value) -> int:
    return sum(int(array.nbytes) for array in _arrays(value))


def _stage_sample(fn) -> tuple[object, dict]:
    baseline = int(mx.get_active_memory())
    mx.reset_peak_memory()
    started = time.perf_counter_ns()
    value = fn()
    built = time.perf_counter_ns()
    _eval(value)
    finished = time.perf_counter_ns()
    return value, {
        "cpu_graph_build_ms": (built - started) / 1e6,
        "synchronized_wall_ms": (finished - started) / 1e6,
        "output_temporary_bytes": _temporary_bytes(value),
        "working_peak_bytes": max(0, int(mx.get_peak_memory()) - baseline),
        "gpu_kernel_ms": None,
        "command_buffers": None,
        "submission_gap_ms": None,
        "timing_note": "boundary-synchronized diagnostic; not additive token wall",
    }


def _median_stage(samples: list[dict]) -> dict:
    result = {}
    for key in (
        "cpu_graph_build_ms",
        "synchronized_wall_ms",
        "output_temporary_bytes",
        "working_peak_bytes",
    ):
        result[key] = statistics.median(row[key] for row in samples)
    result.update(
        {
            "gpu_kernel_ms": None,
            "command_buffers": None,
            "submission_gap_ms": None,
            "system_trace_pending": True,
            "timing_note": samples[0]["timing_note"],
        }
    )
    return result


def _clone_entry(boundary_probe, entry, context: int):
    return boundary_probe._clone_entry(entry, context + 1)


def _projection(attention, x):
    qr = attention.q_a_layernorm(attention.q_a_proj(x))
    q = attention.q_b_proj(qr).reshape(
        1, 1, attention.num_heads, attention.q_head_dim
    ).transpose(0, 2, 1, 3)
    latent = attention.kv_a_layernorm(attention.kv_a_proj_with_mqa(x))[:, None]
    return qr, q, latent


def _pool_update(indexer, cache, x):
    key = indexer.k_norm(indexer.wk(x)).reshape(1, 1, indexer.head_dim)
    gate = x @ indexer.index_kpool_compress_gate.swapaxes(-1, -2)
    valid = mx.ones((1, 1), dtype=mx.bool_)
    cache._append_projected(key, gate, valid)
    return {
        "key": key,
        "gate": gate,
        "valid": valid,
    }


def _score(indexer, pool, x, query):
    pool_keys, pool_indices, pool_valid = pool.logical_pool()
    scores = query @ pool_keys[:, None].swapaxes(-1, -2)
    scores = mx.maximum(scores * indexer.softmax_scale, 0.0)
    weights = indexer.weights_proj(x) * (indexer.n_heads**-0.5)
    index_scores = mx.sum(weights[..., None] * scores, axis=2)
    pool_end = mx.clip(pool_indices[..., -1], 0, pool.total_tokens - 1)
    valid_candidates = (
        (pool_end[:, None, :] < pool.total_tokens) & pool_valid[:, None]
    )
    return {
        "index_scores": mx.where(valid_candidates, index_scores, -1e30),
        "valid_candidates": valid_candidates,
        "pool_count": pool.logical_pool_count,
    }


def _select(indexer, scored):
    select_k = min(
        indexer.index_topk // indexer.index_kpool, scored["pool_count"]
    )
    order = mx.argsort(-scored["index_scores"], axis=-1)
    selected = order[..., :select_k]
    selected_valid = mx.take_along_axis(
        scored["valid_candidates"], selected, axis=-1
    )
    return selected, selected_valid


def _expand(indexer, pool, selected, selected_valid, valid_current):
    active = pool.active_tail_count
    tail_positions = (
        pool.raw_positions[:, -active:]
        if active
        else mx.zeros((1, 0), dtype=mx.int64)
    )
    tail_valid = (
        pool.raw_valid[:, -active:]
        if active
        else mx.zeros((1, 0), dtype=mx.bool_)
    )
    indices, valid = expand_selected_pools(
        selected,
        pool.pool_indices,
        selected_valid,
        kv_len=pool.total_tokens,
        index_topk=indexer.index_topk,
        index_kpool=indexer.index_kpool,
        tail_positions=tail_positions,
        tail_valid=tail_valid,
        always_select_tail=indexer.index_kpool_always_select_tail,
    )
    valid = valid & valid_current[..., None]
    return mx.where(valid, indices, INDEXPOOL_SENTINEL)[:, None]


def _gather(latent_full, indices):
    raw = indices[:, :, 0, :]
    safe, valid = prepare_decode_indexpool_gather(raw, latent_full.shape[2])
    expanded = mx.broadcast_to(
        safe[..., None], safe.shape + (latent_full.shape[-1],)
    )
    gathered = mx.take_along_axis(latent_full, expanded, axis=2)
    return gathered, valid[:, :, None, :]


def _attention(operator_probe, attention, q, gathered, mask, pool):
    dependencies = pool.dependency_arrays()
    if dependencies:
        gathered = mx.depends(gathered, dependencies)
    return operator_probe._attention_phase(attention, q, gathered, mask)


@contextlib.contextmanager
def _capture_selection(fixtures: list[mx.array]):
    original = CompactIndexPoolCache._decode_selection

    def capture(self, indexer, x, qr, valid_cur):
        value = original(self, indexer, x, qr, valid_cur)
        fixtures.append(value)
        return value

    CompactIndexPoolCache._decode_selection = capture
    try:
        yield
    finally:
        CompactIndexPoolCache._decode_selection = original


@contextlib.contextmanager
def _replay_selection(fixtures: list[mx.array], observed: list[mx.array]):
    original = CompactIndexPoolCache._decode_selection
    position = 0

    def replay(self, indexer, x, qr, valid_cur):
        nonlocal position
        if position >= len(fixtures):
            raise RuntimeError("precomputed selection fixture exhausted")
        value = fixtures[position]
        position += 1
        observed.append(value)
        return value

    CompactIndexPoolCache._decode_selection = replay
    try:
        yield
        if position != len(fixtures):
            raise RuntimeError("precomputed selection fixture was not fully consumed")
    finally:
        CompactIndexPoolCache._decode_selection = original


def _single_layer_sample(
    boundary_probe,
    operator_probe,
    attention,
    source_entry,
    context: int,
    x,
) -> tuple[dict, object, object]:
    entry = _clone_entry(boundary_probe, source_entry, context)
    latent_cache, pool = entry
    stages = {}

    # Query/latent projection is outside the requested Indexer A--G split.
    # Materialize only the appended row, never the full logical latent view:
    # synchronizing that view creates a false O(context) diagnostic cost.
    qr, q, latent_current = _projection(attention, x)
    _eval((qr, q, latent_current))
    latent_full, _ = latent_cache.update_and_fetch(latent_current, latent_current)
    _eval(latent_full[..., -1:, :])
    query, stages[PHASES[0]] = _stage_sample(
        lambda: attention.indexer.wq_b(qr).reshape(
            1, 1, attention.indexer.n_heads, attention.indexer.head_dim
        )
    )
    updated, stages[PHASES[1]] = _stage_sample(
        lambda: _pool_update(attention.indexer, pool, x)
    )
    scored, stages[PHASES[2]] = _stage_sample(
        lambda: _score(attention.indexer, pool, x, query)
    )
    selected, stages[PHASES[3]] = _stage_sample(
        lambda: _select(attention.indexer, scored)
    )
    indices, stages[PHASES[4]] = _stage_sample(
        lambda: _expand(
            attention.indexer,
            pool,
            selected[0],
            selected[1],
            updated["valid"],
        )
    )
    gathered, stages[PHASES[5]] = _stage_sample(
        lambda: _gather(latent_full, indices)
    )
    output, stages[PHASES[6]] = _stage_sample(
        lambda: _attention(
            operator_probe,
            attention,
            q,
            gathered[0],
            gathered[1],
            pool,
        )
    )
    return stages, entry, {"indices": indices, "output": output}


def _layer_profile(
    boundary_probe,
    operator_probe,
    attention,
    source_entry,
    context: int,
    layer: int,
    warmups: int,
    samples: int,
) -> dict:
    x = boundary_probe._deterministic_rows(
        1, attention.hidden_size, 2.5 + layer * 0.015625, mx.bfloat16
    )[None]
    measured = []
    for iteration in range(warmups + samples):
        stages, entry, result = _single_layer_sample(
            boundary_probe,
            operator_probe,
            attention,
            source_entry,
            context,
            x,
        )
        if iteration >= warmups:
            measured.append(stages)
        _release(entry, result)

    manual_stages, manual_entry, manual = _single_layer_sample(
        boundary_probe,
        operator_probe,
        attention,
        source_entry,
        context,
        x,
    )
    reference_entry = _clone_entry(boundary_probe, source_entry, context)
    captured = []
    with _capture_selection(captured):
        reference_output = attention(x, cache=reference_entry)
        _eval((reference_output, captured))
    exact = {
        "selected_indices_byte_exact": bool(
            mx.array_equal(manual["indices"], captured[0]).item()
        ),
        "attention_output_byte_exact": bool(
            mx.array_equal(manual["output"], reference_output).item()
        ),
        "post_cache_state_byte_exact": boundary_probe._cache_exact(
            [manual_entry], [reference_entry]
        ),
    }
    valid = np.asarray(manual["indices"])
    out_of_range = int(
        np.count_nonzero(
            (valid != -1) & ((valid < 0) | (valid >= context + 1))
        )
    )
    result = {
        "layer": layer,
        "context_tokens_before_decode": context,
        "pool_count_after_decode": (context + 4) // 4,
        "phases": {
            phase: _median_stage([row[phase] for row in measured])
            for phase in PHASES
        },
        "phase_samples": measured,
        "read_write_byte_estimates": _phase_byte_estimates(
            attention, manual_entry, manual, x
        ),
        "exactness": exact,
        "non_sentinel_out_of_range": out_of_range,
        "selected_index_hash": _hash(manual["indices"]),
        "attention_output_hash": _hash(manual["output"]),
    }
    del manual_stages
    _release(manual_entry, reference_entry, manual, reference_output, captured)
    return result


def _phase_byte_estimates(attention, entry, manual, x) -> dict:
    latent, pool = entry
    pool_keys, pool_indices, pool_valid = pool.logical_pool()
    selected = manual["indices"]
    selected_width = int(selected.shape[-1])
    latent_row_bytes = int(latent.keys.shape[-1]) * 2
    return {
        "index_query_projection": {
            "read_bytes": int(attention.q_lora_rank) * 2,
            "write_bytes": int(attention.indexer.n_heads * attention.indexer.head_dim) * 2,
            "scope": "explicit activation payload only; weight traffic unavailable",
        },
        "indexpool_update": {
            "read_bytes": int(x.nbytes),
            "write_bytes": int(attention.indexer.head_dim) * 4 + 1,
            "functional_carry_bytes": int(pool.nbytes),
            "scope": "authoritative payload upper bound; allocator copy API unavailable",
        },
        "pooled_key_score": {
            "read_bytes": int(pool_keys.nbytes + pool_indices.nbytes + pool_valid.nbytes),
            "write_bytes": int(pool.logical_pool_count * 4),
            "scope": "pool payload plus FP32 scalar score per row",
        },
        "topk_selection": {
            "read_bytes": int(pool.logical_pool_count * 4),
            "write_bytes": int(pool.logical_pool_count * 4),
            "scope": "exact argsort order; internal scratch unavailable",
        },
        "pool_token_expansion": {
            "read_bytes": int(pool_indices.nbytes),
            "write_bytes": int(selected.nbytes),
            "scope": "logical pool indices and canonical selected output",
        },
        "sanitize_gather_preparation": {
            "read_bytes": int(selected.nbytes + selected_width * latent_row_bytes),
            "write_bytes": int(selected_width * latent_row_bytes + selected_width),
            "scope": "selected latent payload and bool validity",
        },
        "sparse_attention": {
            "read_bytes": int(selected_width * latent_row_bytes + selected_width),
            "write_bytes": int(manual["output"].nbytes),
            "scope": "selected latent/mask payload only; weight traffic unavailable",
        },
    }


def _aggregate_layers(rows: list[dict]) -> dict:
    phases = {}
    for phase in PHASES:
        phases[phase] = {
            "synchronized_wall_ms": sum(
                row["phases"][phase]["synchronized_wall_ms"] for row in rows
            ),
            "cpu_graph_build_ms": sum(
                row["phases"][phase]["cpu_graph_build_ms"] for row in rows
            ),
            "output_temporary_bytes": sum(
                row["phases"][phase]["output_temporary_bytes"] for row in rows
            ),
            "working_peak_bytes_max_layer": max(
                row["phases"][phase]["working_peak_bytes"] for row in rows
            ),
            "read_bytes_estimate": sum(
                row["read_write_byte_estimates"][phase]["read_bytes"]
                for row in rows
            ),
            "write_bytes_estimate": sum(
                row["read_write_byte_estimates"][phase]["write_bytes"]
                for row in rows
            ),
            "byte_estimate_scope": rows[0]["read_write_byte_estimates"][phase][
                "scope"
            ],
            "gpu_kernel_ms": None,
            "command_buffers": None,
            "submission_gap_ms": None,
        }
    return {
        "layers": len(rows),
        "phase_sums": phases,
        "synchronized_indexer_phase_sum_ms": sum(
            phases[phase]["synchronized_wall_ms"] for phase in INDEXER_PHASES
        ),
        "warning": (
            "phase boundaries synchronize independently; sums diagnose scaling "
            "but are not full-model critical-path wall time"
        ),
    }


def _set_layer_tags(cache) -> None:
    for layer in EXPECTED_DSA:
        cache[layer][1]._profile_layer = layer


def _run_model_once(model, cache, token, *, mode: str, fixtures=None):
    recorded = []
    started = time.perf_counter_ns()
    if mode == "off":
        output = model(token, cache=cache)
    elif mode == "capture":
        with _capture_selection(recorded):
            output = model(token, cache=cache)
    elif mode == "replay":
        with _replay_selection(fixtures, recorded):
            output = model(token, cache=cache)
    else:
        raise ValueError(f"unknown model profile mode: {mode}")
    built = time.perf_counter_ns()
    logits = output.logits[0, -1]
    _eval((logits, recorded))
    finished = time.perf_counter_ns()
    return {
        "logits": logits,
        "recorded": recorded,
        "cpu_graph_build_ms": (built - started) / 1e6,
        "wall_ms": (finished - started) / 1e6,
        "nan_count": int(np.count_nonzero(np.isnan(np.asarray(logits.astype(mx.float32))))),
    }


def _timed_model_arm(
    model,
    source,
    boundary_probe,
    context: int,
    token,
    *,
    mode: str,
    fixtures,
    warmups: int,
    samples: int,
) -> dict:
    rows = []
    captured_steps = []
    cache = boundary_probe._clone_cache(source, context + warmups + samples + 1)
    _set_layer_tags(cache)
    for iteration in range(warmups + samples):
        step_fixtures = (
            fixtures[iteration] if mode == "replay" else fixtures
        )
        row = _run_model_once(
            model, cache, token, mode=mode, fixtures=step_fixtures
        )
        if mode == "capture":
            captured_steps.append([mx.array(value) for value in row["recorded"]])
        if iteration >= warmups:
            rows.append(
                {
                    "wall_ms": row["wall_ms"],
                    "cpu_graph_build_ms": row["cpu_graph_build_ms"],
                    "nan_count": row["nan_count"],
                }
            )
    post_state_hash = boundary_probe._full_cache_hash(cache)
    final_logits_hash = _hash(row["logits"])
    _release(cache, row["recorded"], row["logits"])
    wall = statistics.median(row["wall_ms"] for row in rows)
    result = {
        "warmups": warmups,
        "samples": samples,
        "cache_policy": "one persistent cache; no clear between decode steps",
        "sample_rows": rows,
        "median_wall_ms": wall,
        "median_cpu_graph_build_ms": statistics.median(
            row["cpu_graph_build_ms"] for row in rows
        ),
        "decode_tokens_per_second": 1000.0 / wall,
        "nan_count": max(row["nan_count"] for row in rows),
        "post_cache_state_hash": post_state_hash,
        "final_logits_hash": final_logits_hash,
    }
    if mode == "capture":
        result["_selection_fixtures"] = captured_steps
    return result


def _full_model_context(
    model,
    source,
    boundary_probe,
    context: int,
    warmups: int,
    samples: int,
) -> dict:
    token = mx.array([[PROFILE_TOKEN]], dtype=mx.uint32)
    # Compare bit hashes while keeping only one full context clone resident.
    # At 256k, source plus three simultaneous clones can make the evidence
    # harness—not the decode path—breach the 340 GB peak gate.
    off_cache = boundary_probe._clone_cache(source, context + 1)
    _set_layer_tags(off_cache)
    off = _run_model_once(model, off_cache, token, mode="off")
    off_logits_hash = _hash(off["logits"])
    off_state_hash = boundary_probe._full_cache_hash(off_cache)
    _release(off_cache, off["recorded"], off["logits"])

    capture_cache = boundary_probe._clone_cache(source, context + 1)
    _set_layer_tags(capture_cache)
    captured = _run_model_once(model, capture_cache, token, mode="capture")
    fixtures = [mx.array(value) for value in captured["recorded"]]
    _eval(fixtures)
    capture_logits_hash = _hash(captured["logits"])
    capture_state_hash = boundary_probe._full_cache_hash(capture_cache)
    _release(capture_cache, captured["recorded"], captured["logits"])

    replay_cache = boundary_probe._clone_cache(source, context + 1)
    _set_layer_tags(replay_cache)
    replayed = _run_model_once(
        model, replay_cache, token, mode="replay", fixtures=fixtures
    )
    fixture_exact = all(
        bool(mx.array_equal(left, right).item())
        for left, right in zip(fixtures, replayed["recorded"], strict=True)
    )
    evidence = {
        "profile_off_capture_logits_byte_exact": (
            off_logits_hash == capture_logits_hash
        ),
        "profile_off_capture_state_byte_exact": (
            off_state_hash == capture_state_hash
        ),
        "capture_replay_selected_indices_byte_exact": fixture_exact,
        "capture_replay_logits_byte_exact": (
            capture_logits_hash == _hash(replayed["logits"])
        ),
        "capture_replay_state_byte_exact": (
            capture_state_hash == boundary_probe._full_cache_hash(replay_cache)
        ),
        "selected_layer_count": len(fixtures),
        "selected_hashes": [_hash(value) for value in fixtures],
        "logits_hash": off_logits_hash,
        "post_cache_state_hash": off_state_hash,
    }
    _release(replay_cache, replayed["recorded"], replayed["logits"])

    baseline = _timed_model_arm(
        model,
        source,
        boundary_probe,
        context,
        token,
        mode="off",
        fixtures=None,
        warmups=warmups,
        samples=samples,
    )
    profiled = _timed_model_arm(
        model,
        source,
        boundary_probe,
        context,
        token,
        mode="capture",
        fixtures=None,
        warmups=warmups,
        samples=samples,
    )
    timed_fixtures = profiled.pop("_selection_fixtures")
    precomputed = _timed_model_arm(
        model,
        source,
        boundary_probe,
        context,
        token,
        mode="replay",
        fixtures=timed_fixtures,
        warmups=warmups,
        samples=samples,
    )
    headroom_ms = baseline["median_wall_ms"] - precomputed["median_wall_ms"]
    return {
        "context_tokens_before_decode": context,
        "pool_count_after_decode": (context + 4) // 4,
        "profile_off": baseline,
        "profile_capture": profiled,
        "precomputed_selection": precomputed,
        "profile_overhead_ratio": (
            profiled["median_wall_ms"] / baseline["median_wall_ms"]
        ),
        "selection_free_headroom_ms": headroom_ms,
        "selection_free_speedup": (
            baseline["median_wall_ms"] / precomputed["median_wall_ms"]
        ),
        "selection_free_tokens_per_second": precomputed[
            "decode_tokens_per_second"
        ],
        "steady_trajectory_exactness": {
            "profile_off_capture_final_logits_exact": baseline[
                "final_logits_hash"
            ]
            == profiled["final_logits_hash"],
            "profile_off_capture_post_state_exact": baseline[
                "post_cache_state_hash"
            ]
            == profiled["post_cache_state_hash"],
            "capture_replay_final_logits_exact": profiled["final_logits_hash"]
            == precomputed["final_logits_hash"],
            "capture_replay_post_state_exact": profiled["post_cache_state_hash"]
            == precomputed["post_cache_state_hash"],
            "fixture_steps": len(timed_fixtures),
        },
        "evidence": evidence,
    }


def _context_case(
    model,
    boundary_probe,
    operator_probe,
    context: int,
    warmups: int,
    samples: int,
) -> dict:
    _progress("build_synthetic_context", context=context)
    source = boundary_probe._synthetic_cache(model, context, "compact-nope-dsa")
    layer_rows = []
    for layer in EXPECTED_DSA:
        _progress("profile_layer", context=context, layer=layer)
        attention = model.language_model.model.layers[layer].self_attn
        layer_rows.append(
            _layer_profile(
                boundary_probe,
                operator_probe,
                attention,
                source[layer],
                context,
                layer,
                warmups,
                samples,
            )
        )
    _progress("profile_full_model", context=context)
    full_model = _full_model_context(
        model, source, boundary_probe, context, warmups, samples
    )
    aggregate = _aggregate_layers(layer_rows)
    aggregate["indexer_share_of_token_wall_synchronized_diagnostic"] = (
        aggregate["synchronized_indexer_phase_sum_ms"]
        / full_model["profile_off"]["median_wall_ms"]
    )
    result = {
        "context_tokens": context,
        "pool_count": (context + 4) // 4,
        "layers": layer_rows,
        "all_dsa_aggregate": aggregate,
        "full_model": full_model,
        "system_trace": None,
    }
    _release(source)
    return result


def _atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _normalize_artifact(artifact: dict) -> None:
    legacy = "attention_query_and_latent_projection"
    removed = False
    for row in artifact.get("contexts", {}).values():
        for layer in row.get("layers", []):
            removed = layer.get("phases", {}).pop(legacy, None) is not None or removed
            layer.get("read_write_byte_estimates", {}).pop(legacy, None)
            for sample in layer.get("phase_samples", []):
                sample.pop(legacy, None)
        row.get("all_dsa_aggregate", {}).get("phase_sums", {}).pop(legacy, None)
    if removed:
        artifact["measurement_corrections"] = {
            "removed_phase": legacy,
            "reason": (
                "the legacy helper synchronized the full logical latent view and "
                "therefore measured an artificial O(context) read; requested "
                "Indexer A-G and full-model measurements are unchanged"
            ),
            "requires_model_rerun": False,
        }


def _merge_system_trace(artifact: dict, telemetry: dict) -> None:
    context = int(telemetry["context_tokens"])
    if context not in CONTEXTS:
        raise ValueError(f"unexpected telemetry context: {context}")
    if not telemetry.get("complete"):
        raise ValueError("refusing to merge incomplete System Trace telemetry")
    stages = telemetry.get("stages", {})
    required = set(PHASES) | {"full_model_decode"}
    if set(stages) != required:
        raise ValueError(
            f"telemetry stages differ: {sorted(set(stages) ^ required)}"
        )
    row = artifact["contexts"].get(str(context))
    if row is None:
        raise ValueError("model phase must complete before telemetry merge")
    for phase in PHASES:
        trace = stages[phase]
        repetitions = int(trace["repetitions"])
        measured = trace["telemetry"]
        aggregate = row["all_dsa_aggregate"]["phase_sums"][phase]
        aggregate.update(
            {
                "gpu_kernel_ms": measured["gpu_busy_ms"] / repetitions,
                "gpu_idle_gap_ms": measured["gpu_idle_gap_ms"] / repetitions,
                "command_buffers": measured[
                    "command_buffer_submission_rows"
                ]
                / repetitions,
                "submission_gap_ms": {
                    "p50": measured["cpu_inter_submission_p50_ms"],
                    "p95": measured["cpu_inter_submission_p95_ms"],
                },
                "bounded_trace_repetitions": repetitions,
                "system_trace_pending": False,
            }
        )
    full = stages["full_model_decode"]
    repetitions = int(full["repetitions"])
    measured = full["telemetry"]
    row["full_model"]["bounded_system_trace"] = {
        "gpu_busy_ms_per_token": measured["gpu_busy_ms"] / repetitions,
        "gpu_idle_ms_per_token": measured["gpu_idle_gap_ms"] / repetitions,
        "command_buffers_per_token": measured[
            "command_buffer_submission_rows"
        ]
        / repetitions,
        "submission_gap_ms": {
            "p50": measured["cpu_inter_submission_p50_ms"],
            "p95": measured["cpu_inter_submission_p95_ms"],
        },
    }
    indexer_gpu = sum(
        row["all_dsa_aggregate"]["phase_sums"][phase]["gpu_kernel_ms"]
        for phase in INDEXER_PHASES
    )
    baseline_wall = row["full_model"]["profile_off"]["median_wall_ms"]
    row["all_dsa_aggregate"].update(
        {
            "indexer_gpu_ms_per_token_isolated": indexer_gpu,
            "indexer_gpu_share_of_token_wall": indexer_gpu / baseline_wall,
            "selection_free_headroom_share_of_token_wall": row["full_model"]
            ["selection_free_headroom_ms"]
            / baseline_wall,
        }
    )
    row["system_trace"] = telemetry
    artifact["system_trace"][str(context)] = telemetry


def _optimization_decision(artifact: dict) -> dict:
    row = artifact["contexts"][str(max(CONTEXTS))]
    full = row["full_model"]
    headroom_ms = float(full["selection_free_headroom_ms"])
    headroom_ratio = headroom_ms / float(full["profile_off"]["median_wall_ms"])
    phases = row["all_dsa_aggregate"]["phase_sums"]
    dominant = max(INDEXER_PHASES, key=lambda name: phases[name]["gpu_kernel_ms"])
    if headroom_ms <= 0.5 or headroom_ratio <= 0.02:
        choice = "stop_lightning_indexer_optimization"
    elif dominant == "pooled_key_score":
        choice = "tiled_or_fused_metal_pool_score"
    elif dominant == "topk_selection":
        choice = "exact_partial_topk_metal_kernel"
    elif dominant in (
        "pool_token_expansion",
        "sanitize_gather_preparation",
    ):
        choice = "sentinel_aware_fused_expand_gather"
    elif dominant == "index_query_projection":
        choice = "dedicated_q1_index_query_projection"
    elif dominant == "indexpool_update":
        choice = "compact_indexpool_update_kernel"
    else:
        choice = "native_stateful_executor"
    return {
        "context_tokens": max(CONTEXTS),
        "selection_free_headroom_ms": headroom_ms,
        "selection_free_headroom_fraction": headroom_ratio,
        "selection_free_tokens_per_second": full[
            "selection_free_tokens_per_second"
        ],
        "dominant_measured_indexer_phase": dominant,
        "selected_single_next_candidate": choice,
        "policy": (
            "stop when exact selection removal saves <=0.5 ms/token or <=2%; "
            "otherwise select only the largest measured Indexer phase"
        ),
    }


def _acceptance(artifact: dict) -> dict:
    contexts = artifact.get("contexts", {})
    complete_contexts = set(map(int, contexts)) == set(CONTEXTS)
    rows = list(contexts.values())
    system_trace = artifact.get("system_trace", {})
    traces_complete = set(map(int, system_trace)) == set(CONTEXTS) and all(
        row.get("complete")
        and set(row.get("stages", {})) == set(PHASES) | {"full_model_decode"}
        for row in system_trace.values()
    )
    acceptance = {
        "all_2k_32k_128k_256k_contexts_measured": complete_contexts,
        "all_indexer_A_through_G_phases_measured": complete_contexts
        and all(
            set(layer["phases"]) == set(PHASES)
            for row in rows
            for layer in row["layers"]
        ),
        "all_11_dsa_layers_measured": complete_contexts
        and all(len(row["layers"]) == 11 for row in rows),
        "profile_instrumentation_logits_and_state_exact": complete_contexts
        and all(
            row["full_model"]["evidence"][
                "profile_off_capture_logits_byte_exact"
            ]
            and row["full_model"]["evidence"][
                "profile_off_capture_state_byte_exact"
            ]
            for row in rows
        ),
        "precomputed_selection_indices_logits_state_exact": complete_contexts
        and all(
            row["full_model"]["evidence"][
                "capture_replay_selected_indices_byte_exact"
            ]
            and row["full_model"]["evidence"][
                "capture_replay_logits_byte_exact"
            ]
            and row["full_model"]["evidence"][
                "capture_replay_state_byte_exact"
            ]
            for row in rows
        ),
        "steady_trajectory_profile_and_replay_exact": complete_contexts
        and all(
            all(
                row["full_model"]
                .get("steady_trajectory_exactness", {})
                .get(key, False)
                for key in (
                    "profile_off_capture_final_logits_exact",
                    "profile_off_capture_post_state_exact",
                    "capture_replay_final_logits_exact",
                    "capture_replay_post_state_exact",
                )
            )
            for row in rows
        ),
        "all_indices_sentinel_or_in_range": complete_contexts
        and all(
            layer["non_sentinel_out_of_range"] == 0
            for row in rows
            for layer in row["layers"]
        ),
        "profiling_overhead_explicit": complete_contexts
        and all("profile_overhead_ratio" in row["full_model"] for row in rows),
        "precomputed_selection_lower_bound_recorded": complete_contexts
        and all(
            row["full_model"]["selection_free_headroom_ms"] is not None
            for row in rows
        ),
        "context_scaling_recorded": complete_contexts,
        "bounded_system_trace_gpu_cb_submission_metrics_complete": traces_complete,
        "bounded_phase_working_peak_le_512mib": complete_contexts
        and all(
            layer["phases"][phase]["working_peak_bytes"] <= 512 << 20
            for row in rows
            for layer in row["layers"]
            for phase in PHASES
        ),
        "bounded_trace_each_under_2gib": traces_complete
        and all(
            stage["trace"]["bytes"] <= 2 << 30
            for trace in system_trace.values()
            for stage in trace["stages"].values()
        ),
        "official_16_128_oracle_exact": bool(
            artifact.get("official_oracle", {}).get("first_16_match")
            and artifact.get("official_oracle", {}).get("full_128_match")
        ),
        "peak_below_340gb": int(artifact.get("peak_memory_bytes", 1 << 62))
        <= 340_000_000_000,
        "runtime_unchanged": artifact.get("runtime_changes")
        == {
            "admission": False,
            "apc": False,
            "backend": False,
            "cache_abi": False,
            "kernel": False,
            "server": False,
        },
    }
    return acceptance


def _model_phase(args) -> int:
    report = inspect_checkpoint(args.model, require_server_ready=True)
    oracle_probe, operator_probe, boundary_probe = _load_probe_modules()
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    _progress("load_model")
    model, processor = load(
        args.model,
        experimental_packed_decode_moe=True,
        experimental_compact_nope_dsa_cache=True,
        compact_cache_capacity_tokens=max(CONTEXTS) + 1,
    )
    warm_residency(model)
    _progress("official_oracle")
    official_oracle = oracle_probe._official_oracle(model, processor, report)
    mx.reset_peak_memory()
    schema = "glm53-lightning-indexer-decode-critical-path-v1"
    requested_contexts = tuple(args.contexts or CONTEXTS)
    artifact = None
    if args.output.exists():
        candidate = json.loads(args.output.read_text())
        if candidate.get("schema") == schema:
            if candidate.get("checkpoint_fingerprint") != report.fingerprint:
                raise ValueError("existing profile artifact uses another checkpoint")
            artifact = candidate
    if artifact is None:
        artifact = {
            "schema": schema,
            "date": date.today().isoformat(),
            "complete": False,
            "accepted": False,
            "profiling_only": True,
            "official_hf_revision": report.official_revision,
            "checkpoint_fingerprint": report.fingerprint,
            "mlx_vlm_revision": MLX_VLM_REVISION,
            "compact_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
            "contexts_requested": list(CONTEXTS),
            "dsa_layers": list(EXPECTED_DSA),
            "measurement_contract": {
                "query_tokens": 1,
                "phase_sync": "each A-G boundary synchronized independently",
                "phase_sum_is_token_wall": False,
                "full_model_sync": "only final logits and recorded selection",
                "profile_off_on_differential": True,
                "counterfactual": (
                    "authoritative pool update retained; exact query/score/top-k/"
                    "expansion selection replayed; sanitize/gather/attention retained"
                ),
                "gpu_timing": "bounded non-replayable Metal System Trace",
                "static_kernel_labels_are_dispatch_evidence": False,
            },
            "official_oracle": official_oracle,
            "contexts": {},
            "system_trace": {},
            "peak_memory_bytes": 0,
            "runtime_changes": {
                "admission": False,
                "apc": False,
                "backend": False,
                "cache_abi": False,
                "kernel": False,
                "server": False,
            },
        }
    _normalize_artifact(artifact)
    artifact["official_oracle"] = official_oracle
    contexts = artifact["contexts"]
    for context in requested_contexts:
        if str(context) in contexts and not args.rerun:
            _progress("skip_completed_context", context=context)
            continue
        contexts[str(context)] = _context_case(
            model,
            boundary_probe,
            operator_probe,
            context,
            args.warmups,
            args.samples,
        )
        artifact["peak_memory_bytes"] = max(
            artifact["peak_memory_bytes"], int(mx.get_peak_memory())
        )
        artifact["last_completed_context"] = context
        artifact["acceptance"] = _acceptance(artifact)
        _atomic_write(args.output, artifact)
    artifact["model_phase_complete"] = set(map(int, contexts)) == set(CONTEXTS)
    artifact["acceptance"] = _acceptance(artifact)
    artifact["complete"] = all(artifact["acceptance"].values())
    artifact["accepted"] = artifact["complete"]
    artifact["decision"] = (
        "await_bounded_system_trace"
        if artifact["model_phase_complete"]
        else "await_remaining_model_contexts"
    )
    _atomic_write(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model_phase_complete": artifact["model_phase_complete"],
                "complete": artifact["complete"],
                "failed_gates": [
                    key for key, value in artifact["acceptance"].items() if not value
                ],
            },
            indent=2,
        )
    )
    return 0


def _retime_phase(args) -> int:
    artifact = json.loads(args.output.read_text())
    _normalize_artifact(artifact)
    report = inspect_checkpoint(args.model, require_server_ready=True)
    if artifact.get("checkpoint_fingerprint") != report.fingerprint:
        raise ValueError("profile artifact and retime checkpoint differ")
    _, _, boundary_probe = _load_probe_modules()
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    _progress("load_model_for_steady_retime")
    model, _ = load(
        args.model,
        experimental_packed_decode_moe=True,
        experimental_compact_nope_dsa_cache=True,
        compact_cache_capacity_tokens=max(CONTEXTS) + args.warmups + args.samples + 1,
    )
    warm_residency(model)
    requested_contexts = tuple(args.contexts or CONTEXTS)
    for context in requested_contexts:
        if str(context) not in artifact["contexts"]:
            raise ValueError(f"model-phase context is missing: {context}")
        _progress("steady_retime_full_model", context=context)
        source = boundary_probe._synthetic_cache(
            model, context, "compact-nope-dsa"
        )
        artifact["contexts"][str(context)]["full_model"] = _full_model_context(
            model,
            source,
            boundary_probe,
            context,
            args.warmups,
            args.samples,
        )
        _release(source)
        if str(context) in artifact.get("system_trace", {}):
            _merge_system_trace(
                artifact, artifact["system_trace"][str(context)]
            )
        artifact["last_retimed_context"] = context
        artifact["peak_memory_bytes"] = max(
            int(artifact.get("peak_memory_bytes", 0)), int(mx.get_peak_memory())
        )
        _atomic_write(args.output, artifact)
    artifact.setdefault("measurement_corrections", {}).update(
        {
            "full_model_timing": (
                "retimed with one persistent cache per arm and no mx.clear_cache "
                "between decode steps"
            ),
            "steady_retime_complete": set(requested_contexts) == set(CONTEXTS),
        }
    )
    artifact["acceptance"] = _acceptance(artifact)
    artifact["complete"] = all(artifact["acceptance"].values())
    artifact["accepted"] = artifact["complete"]
    if artifact["accepted"]:
        artifact["optimization_decision"] = _optimization_decision(artifact)
        artifact["decision"] = "single_dominant_candidate_selected"
    else:
        artifact["decision"] = "steady_retime_incomplete"
    _atomic_write(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": artifact["complete"],
                "decision": artifact["decision"],
            },
            indent=2,
        )
    )
    return 0 if artifact["accepted"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--phase",
        choices=("model", "retime", "merge-telemetry", "finalize"),
        default="model",
    )
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        choices=CONTEXTS,
        help="model-phase subset; an existing artifact is resumed by default",
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    args = parser.parse_args()
    if args.phase == "model":
        return _model_phase(args)
    if args.phase == "retime":
        return _retime_phase(args)
    artifact = json.loads(args.output.read_text())
    _normalize_artifact(artifact)
    if args.phase == "merge-telemetry":
        if args.telemetry is None:
            parser.error("--phase merge-telemetry requires --telemetry")
        _merge_system_trace(artifact, json.loads(args.telemetry.read_text()))
    artifact["acceptance"] = _acceptance(artifact)
    artifact["complete"] = all(artifact["acceptance"].values())
    artifact["accepted"] = artifact["complete"]
    if artifact["accepted"]:
        artifact["optimization_decision"] = _optimization_decision(artifact)
        artifact["decision"] = "single_dominant_candidate_selected"
    else:
        artifact["decision"] = "profiling_evidence_incomplete"
    _atomic_write(args.output, artifact)
    if args.phase == "merge-telemetry":
        return 0
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
