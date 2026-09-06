#!/usr/bin/env python3
"""Probe an exact, score-only partial top-k Metal operator for Lightning Indexer.

The candidate deliberately starts from the existing FP32 Indexer score tensor.
It does not change query projection, pooled score generation, IndexPool update,
pool expansion, gather, attention, or any production/runtime ABI.

The long model phase is resumable by context and is intended to be run by the
user.  Each completed context is atomically committed to the JSON artifact.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import statistics
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np

from glm53_flash_mlx.abi import MLX_VLM_REVISION, NOPE_DSA_CACHE_ABI_COMPACT
from glm53_flash_mlx.indexpool import INDEXPOOL_SENTINEL, expand_selected_pools
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import EXPECTED_DSA, inspect_checkpoint
from glm53_flash_mlx.nope_cache import CompactIndexPoolCache


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-exact-partial-topk-metal-20260906.json"
)
CONTEXTS = (2_048, 32_768, 131_072, 262_144)
SELECT_K = 512
THREADS = 512
PROFILE_TOKEN = 3000
MAX_POOL_COUNT = 65_600
TOPK_KEEP_MS = 1.15
TOPK_REJECT_MS = 1.50
MIN_TOPK_SAVING_MS = 0.75
MAX_FULL_MODEL_MS = 81.2
MAX_WORKING_PEAK_DELTA = 32 << 20


_PARTIAL_TOPK_SOURCE = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.y;
    uint count = uint(pool_count[0]);
    const device T* row_scores = scores + size_t(row) * count;
    device uint* row_output = indices + size_t(row) * K;

    threadgroup atomic_uint histogram[16];
    threadgroup atomic_uint candidate_count;
    threadgroup ulong prefix_shared;
    threadgroup uint rank_shared;
    threadgroup ulong candidates[K];

    if (lane == 0) {
        prefix_shared = 0;
        rank_shared = K;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Select the exact Kth composite key with thirteen 4-bit radix passes over
    // the 49 significant bits.  No score ordering or reduction is changed.
    for (int shift = 48; shift >= 0; shift -= 4) {
        if (lane < 16) {
            atomic_store_explicit(
                &histogram[lane], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ulong prefix = prefix_shared;
        for (uint index = lane; index < count; index += THREAD_COUNT) {
            // Numeric descending score, then original index ascending.  MLX
            // argsort is stable, so argsort(-score) preserves this exact tie
            // order. Collapse +0/-0 because they compare equal in the oracle.
            float score_value = float(row_scores[index]);
            uint bits = as_type<uint>(score_value);
            if ((bits & 0x7fffffffu) == 0u) bits = 0u;
            // Preserve the complete signed finite FP32 ordering. Indexer head
            // scores are nonnegative before weights_proj, but their weighted
            // head sum may be negative. Invalid candidates are also negative.
            uint ordered =
                (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
            ulong key =
                (ulong(ordered) << 17) | ulong(0x1ffffu - index);
            bool matches = shift == 48 || (key >> uint(shift + 4)) == prefix;
            if (matches) {
                uint digit = uint((key >> uint(shift)) & 0xful);
                atomic_fetch_add_explicit(
                    &histogram[digit], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            uint rank = rank_shared;
            uint skipped = 0;
            uint chosen = 0;
            for (int digit = 15; digit >= 0; --digit) {
                uint in_bin = atomic_load_explicit(
                    &histogram[digit], memory_order_relaxed);
                if (rank > skipped && rank <= skipped + in_bin) {
                    chosen = uint(digit);
                    rank_shared = rank - skipped;
                    break;
                }
                skipped += in_bin;
            }
            prefix_shared = (prefix_shared << 4) | ulong(chosen);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        atomic_store_explicit(&candidate_count, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong threshold = prefix_shared;
    for (uint index = lane; index < count; index += THREAD_COUNT) {
        float score_value = float(row_scores[index]);
        uint bits = as_type<uint>(score_value);
        if ((bits & 0x7fffffffu) == 0u) bits = 0u;
        uint ordered =
            (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
        ulong key = (ulong(ordered) << 17) | ulong(0x1ffffu - index);
        if (key >= threshold) {
            uint slot = atomic_fetch_add_explicit(
                &candidate_count, 1u, memory_order_relaxed);
            if (slot < K) candidates[slot] = key;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Candidate insertion order is intentionally irrelevant: sort all 512
    // unique composite keys in-threadgroup, then emit descending.
    for (uint span = 2; span <= K; span <<= 1) {
        for (uint stride = span >> 1; stride > 0; stride >>= 1) {
            uint peer = lane ^ stride;
            if (peer > lane) {
                ulong left = candidates[lane];
                ulong right = candidates[peer];
                bool ascending = (lane & span) == 0;
                bool swap = ascending ? (left > right) : (left < right);
                if (swap) {
                    candidates[lane] = right;
                    candidates[peer] = left;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    ulong key = candidates[K - 1 - lane];
    row_output[lane] = 0x1ffffu - uint(key & 0x1fffful);
"""


_partial_topk_kernel = (
    mx.fast.metal_kernel(
        name="glm53_probe_exact_partial_topk_512",
        input_names=["scores", "pool_count"],
        output_names=["indices"],
        source=_PARTIAL_TOPK_SOURCE,
    )
    if mx.metal.is_available()
    else None
)


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


def _atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _load_probe_modules():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import probe_exact_sigmoid_gate_metal_barrier as oracle_probe
    import probe_long_context_dsa_decode_frontier as operator_probe
    import probe_long_context_first_decode_boundary as boundary_probe
    import profile_lightning_indexer_decode_critical_path as profile

    return oracle_probe, operator_probe, boundary_probe, profile


def _eval(*values) -> None:
    arrays = []
    for value in values:
        if isinstance(value, mx.array):
            arrays.append(value)
        elif isinstance(value, dict):
            arrays.extend(item for item in value.values() if isinstance(item, mx.array))
        elif isinstance(value, (tuple, list)):
            arrays.extend(item for item in value if isinstance(item, mx.array))
    if arrays:
        mx.eval(*arrays)
    mx.synchronize()


def _hash(value: mx.array) -> str:
    mx.eval(value)
    array = np.asarray(value.astype(mx.float32) if value.dtype == mx.bfloat16 else value)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _exact(left: mx.array, right: mx.array) -> bool:
    return bool(mx.array_equal(left, right).item())


def _release(*values) -> None:
    for value in values:
        if isinstance(value, list):
            value.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def exact_partial_topk(
    scores: mx.array, *, k: int = SELECT_K
) -> tuple[mx.array, mx.array]:
    """Return exact values/indices for finite BF16/FP32 scores without sorting."""
    if _partial_topk_kernel is None:
        raise RuntimeError("exact partial top-k probe requires Metal")
    if scores.dtype not in (mx.bfloat16, mx.float32):
        raise TypeError("exact partial top-k requires BF16 or FP32 scores")
    if k != SELECT_K:
        raise ValueError(f"probe kernel supports exactly k={SELECT_K}")
    if scores.ndim < 1:
        raise ValueError("scores must have a selection axis")
    count = int(scores.shape[-1])
    if count < k or count > MAX_POOL_COUNT:
        raise ValueError(f"pool count must be in [512, {MAX_POOL_COUNT}]")
    rows = int(scores.size // count)
    contiguous = mx.contiguous(scores.reshape(rows, count), allow_col_major=False)
    count_value = mx.array([count], dtype=mx.uint32)
    indices = _partial_topk_kernel(
        inputs=[contiguous, count_value],
        template=[("T", scores.dtype), ("K", k), ("THREAD_COUNT", THREADS)],
        grid=(THREADS, rows, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(rows, k)],
        output_dtypes=[mx.uint32],
    )[0].reshape(scores.shape[:-1] + (k,))
    values = mx.take_along_axis(scores, indices, axis=-1)
    return values, indices


def _oracle_topk(scores: mx.array) -> tuple[mx.array, mx.array]:
    indices = mx.argsort(-scores, axis=-1)[..., :SELECT_K]
    return mx.take_along_axis(scores, indices, axis=-1), indices


def _artificial_fixtures() -> dict[str, mx.array]:
    n = 1024
    ascending = np.arange(n, dtype=np.float32)
    tiny = np.linspace(-2.0e-7, 2.0e-7, n, dtype=np.float32)
    bf16_source = np.linspace(-8.0, 8.0, n, dtype=np.float32)
    bf16_derived = np.asarray(mx.array(bf16_source).astype(mx.bfloat16).astype(mx.float32))
    kth_tie = np.arange(n, dtype=np.float32)
    kth_tie[500:530] = np.float32(700.0)
    signed_zero = np.arange(n, dtype=np.float32)
    signed_zero[480:544:2] = np.float32(0.0)
    signed_zero[481:544:2] = np.float32(-0.0)
    sentinel = np.full(n, np.float32(-1.0e30), dtype=np.float32)
    sentinel[:400] = np.arange(400, dtype=np.float32)
    fixtures = {
        "strictly_ascending": mx.array(ascending)[None],
        "strictly_descending": mx.array(ascending[::-1].copy())[None],
        "all_equal": mx.zeros((1, n), dtype=mx.float32),
        "large_tie_groups": mx.array((np.arange(n) // 64).astype(np.float32))[None],
        "tie_at_kth_boundary": mx.array(kth_tie)[None],
        "positive_negative_zero": mx.array(signed_zero)[None],
        "very_small_fp32_differences": mx.array(tiny)[None],
        "bf16_boundary_derived_fp32": mx.array(bf16_derived)[None],
        "production_bfloat16_ties": mx.array(
            (np.arange(n) // 32).astype(np.float32)
        )[None].astype(mx.bfloat16),
        "production_sentinel_ties": mx.array(sentinel)[None],
    }
    # Decode at exactly 256K appends a partial pool, so the first score shape is
    # 65,537 rather than 65,536. Also exercise the full 256-token-aligned
    # physical row capacity before loading the 320 GB model.
    fixtures["first_post_256k_partial_pool"] = mx.arange(
        65_537, dtype=mx.float32
    )[None]
    fixtures["aligned_256k_pool_capacity"] = mx.arange(
        MAX_POOL_COUNT, dtype=mx.float32
    )[None]
    return fixtures


def _run_artificial() -> dict:
    rows = {}
    for name, scores in _artificial_fixtures().items():
        oracle_values, oracle_indices = _oracle_topk(scores)
        candidate_values, candidate_indices = exact_partial_topk(scores)
        _eval(oracle_values, oracle_indices, candidate_values, candidate_indices)
        rows[name] = {
            "values_byte_exact": _exact(oracle_values, candidate_values),
            "indices_byte_exact": _exact(oracle_indices, candidate_indices),
            "oracle_index_hash": _hash(oracle_indices),
            "candidate_index_hash": _hash(candidate_indices),
        }
    return {
        "fixtures": rows,
        "all_values_and_indices_byte_exact": all(
            row["values_byte_exact"] and row["indices_byte_exact"]
            for row in rows.values()
        ),
        "tie_contract": "score descending; equal scores preserve ascending source index",
        "nan_policy": "excluded and rejected by probe contract",
    }


def _run_selection_path(
    profile,
    operator_probe,
    boundary_probe,
    attention,
    source,
    context,
    x,
    selector,
):
    entry = boundary_probe._clone_entry(source, context + 1)
    latent, pool = entry
    qr, q, current = profile._projection(attention, x)
    latent_full, _ = latent.update_and_fetch(current, current)
    query = attention.indexer.wq_b(qr).reshape(
        1, 1, attention.indexer.n_heads, attention.indexer.head_dim
    )
    updated = profile._pool_update(attention.indexer, pool, x)
    scored = profile._score(attention.indexer, pool, x, query)
    values, selected = selector(scored["index_scores"])
    selected_valid = mx.take_along_axis(
        scored["valid_candidates"], selected, axis=-1
    )
    expanded = profile._expand(
        attention.indexer, pool, selected, selected_valid, updated["valid"]
    )
    gathered = profile._gather(latent_full, expanded)
    output = profile._attention(
        operator_probe, attention, q, gathered[0], gathered[1], pool
    )
    _eval(values, selected, expanded, gathered, output)
    return {
        "entry": entry,
        "scores": scored["index_scores"],
        "values": values,
        "selected": selected,
        "expanded": expanded,
        "gathered": gathered[0],
        "gather_valid": gathered[1],
        "output": output,
    }


def _real_layer_case(profile, operator_probe, boundary_probe, model, source, context, layer):
    attention = model.language_model.model.layers[layer].self_attn
    x = boundary_probe._deterministic_rows(
        1, attention.hidden_size, 6.5 + layer * 0.015625, mx.bfloat16
    )[None]
    oracle = _run_selection_path(
        profile, operator_probe, boundary_probe, attention, source[layer], context, x, _oracle_topk
    )
    candidate = _run_selection_path(
        profile,
        operator_probe,
        boundary_probe,
        attention,
        source[layer],
        context,
        x,
        exact_partial_topk,
    )
    repeat_values, repeat_selected = exact_partial_topk(candidate["scores"])
    _eval(repeat_values, repeat_selected)
    result = {
        "layer": layer,
        "pool_count": int(candidate["scores"].shape[-1]),
        "score_dtype": str(candidate["scores"].dtype),
        "score_hash_exact": _hash(oracle["scores"]) == _hash(candidate["scores"]),
        "topk_values_byte_exact": _exact(oracle["values"], candidate["values"]),
        "topk_indices_byte_exact": _exact(oracle["selected"], candidate["selected"]),
        "expanded_indices_byte_exact": _exact(oracle["expanded"], candidate["expanded"]),
        "sentinel_positions_exact": _exact(
            oracle["expanded"] == INDEXPOOL_SENTINEL,
            candidate["expanded"] == INDEXPOOL_SENTINEL,
        ),
        "gather_byte_exact": _exact(oracle["gathered"], candidate["gathered"]),
        "gather_valid_byte_exact": _exact(oracle["gather_valid"], candidate["gather_valid"]),
        "attention_output_byte_exact": _exact(oracle["output"], candidate["output"]),
        "post_state_byte_exact": boundary_probe._cache_exact(
            [oracle["entry"]], [candidate["entry"]]
        ),
        "repeat_values_byte_exact": _exact(candidate["values"], repeat_values),
        "repeat_indices_byte_exact": _exact(candidate["selected"], repeat_selected),
        "score_nan_count": int(
            np.count_nonzero(
                np.isnan(np.asarray(candidate["scores"].astype(mx.float32)))
            )
        ),
        "selected_index_hash": _hash(candidate["selected"]),
        "attention_output_hash": _hash(candidate["output"]),
    }
    result["all_exact"] = all(
        value for key, value in result.items() if key.endswith("exact")
    )
    _release(oracle, candidate)
    return result


def _selector_timing(score_rows, selector, warmups: int, samples: int) -> dict:
    measured = []
    baseline_active = int(mx.get_active_memory())
    mx.reset_peak_memory()
    last = None
    for iteration in range(warmups + samples):
        started = time.perf_counter_ns()
        outputs = [selector(scores)[1] for scores in score_rows]
        built = time.perf_counter_ns()
        _eval(outputs)
        finished = time.perf_counter_ns()
        if iteration >= warmups:
            measured.append(
                {
                    "wall_ms": (finished - started) / 1e6,
                    "cpu_graph_build_ms": (built - started) / 1e6,
                }
            )
        last = outputs
    peak_delta = max(0, int(mx.get_peak_memory()) - baseline_active)
    _release(last)
    wall = statistics.median(row["wall_ms"] for row in measured)
    return {
        "warmups": warmups,
        "samples": samples,
        "sample_rows": measured,
        "median_wall_ms": wall,
        "median_cpu_graph_build_ms": statistics.median(
            row["cpu_graph_build_ms"] for row in measured
        ),
        "working_peak_delta_bytes": peak_delta,
    }


def _candidate_decode_selection(self, indexer, x, qr, valid_cur):
    pool_keys, pool_indices, pool_valid = self.logical_pool()
    pool_count = self.logical_pool_count
    query = indexer.wq_b(qr).reshape(1, 1, indexer.n_heads, indexer.head_dim)
    scores = query @ pool_keys[:, None].swapaxes(-1, -2)
    scores = mx.maximum(scores * indexer.softmax_scale, 0.0)
    weights = indexer.weights_proj(x) * (indexer.n_heads**-0.5)
    index_scores = mx.sum(weights[..., None] * scores, axis=2)
    pool_end = mx.clip(pool_indices[..., -1], 0, self.total_tokens - 1)
    valid_candidates = (pool_end[:, None, :] < self.total_tokens) & pool_valid[:, None]
    index_scores = mx.where(valid_candidates, index_scores, -1e30)
    select_k = min(self.index_topk // self.index_kpool, pool_count)
    if select_k != SELECT_K:
        order = mx.argsort(-index_scores, axis=-1)
        selected = order[..., :select_k]
    else:
        _, selected = exact_partial_topk(index_scores, k=select_k)
    selected_valid = mx.take_along_axis(valid_candidates, selected, axis=-1)
    active = self.active_tail_count
    tail_positions = (
        self.raw_positions[:, -active:]
        if active
        else mx.zeros((1, 0), dtype=mx.int64)
    )
    tail_valid = (
        self.raw_valid[:, -active:]
        if active
        else mx.zeros((1, 0), dtype=mx.bool_)
    )
    topk, valid = expand_selected_pools(
        selected,
        pool_indices,
        selected_valid,
        kv_len=self.total_tokens,
        index_topk=self.index_topk,
        index_kpool=self.index_kpool,
        tail_positions=tail_positions,
        tail_valid=tail_valid,
        always_select_tail=self.always_select_tail,
    )
    valid = valid & valid_cur[..., None]
    return mx.where(valid, topk, INDEXPOOL_SENTINEL)[:, None]


@contextlib.contextmanager
def _candidate_selection_installed():
    original = CompactIndexPoolCache._decode_selection
    CompactIndexPoolCache._decode_selection = _candidate_decode_selection
    try:
        yield
    finally:
        CompactIndexPoolCache._decode_selection = original


def _trajectory(model, source, boundary_probe, context: int, *, candidate: bool, steps: int):
    cache = boundary_probe._clone_cache(source, context + steps + 1)
    token = mx.array([[PROFILE_TOKEN]], dtype=mx.uint32)
    logits_hashes = []
    manager = _candidate_selection_installed() if candidate else contextlib.nullcontext()
    with manager:
        for _ in range(steps):
            output = model(token, cache=cache)
            logits = output.logits[0, -1]
            _eval(logits)
            logits_hashes.append(_hash(logits))
    result = {
        "logits_hashes": logits_hashes,
        "post_state_hash": boundary_probe._full_cache_hash(cache),
    }
    _release(cache, logits)
    return result


def _timed_model(model, source, boundary_probe, context, *, candidate, warmups, samples):
    cache = boundary_probe._clone_cache(source, context + warmups + samples + 1)
    token = mx.array([[PROFILE_TOKEN]], dtype=mx.uint32)
    rows = []
    manager = _candidate_selection_installed() if candidate else contextlib.nullcontext()
    baseline_active = int(mx.get_active_memory())
    mx.reset_peak_memory()
    with manager:
        for iteration in range(warmups + samples):
            started = time.perf_counter_ns()
            output = model(token, cache=cache)
            built = time.perf_counter_ns()
            logits = output.logits[0, -1]
            _eval(logits)
            finished = time.perf_counter_ns()
            if iteration >= warmups:
                rows.append(
                    {
                        "wall_ms": (finished - started) / 1e6,
                        "cpu_graph_build_ms": (built - started) / 1e6,
                    }
                )
    wall = statistics.median(row["wall_ms"] for row in rows)
    result = {
        "sample_rows": rows,
        "median_wall_ms": wall,
        "decode_tokens_per_second": 1000.0 / wall,
        "working_peak_delta_bytes": max(
            0, int(mx.get_peak_memory()) - baseline_active
        ),
        "final_logits_hash": _hash(logits),
        "post_state_hash": boundary_probe._full_cache_hash(cache),
    }
    _release(cache, logits)
    return result


def _context_case(model, boundary_probe, operator_probe, profile, context, warmups, samples):
    _progress("build_synthetic_context", context=context)
    source = boundary_probe._synthetic_cache(model, context, "compact-nope-dsa")
    layers = []
    score_rows = []
    for layer in EXPECTED_DSA:
        _progress("exact_layer", context=context, layer=layer)
        row = _real_layer_case(
            profile, operator_probe, boundary_probe, model, source, context, layer
        )
        layers.append(row)
        attention = model.language_model.model.layers[layer].self_attn
        x = boundary_probe._deterministic_rows(
            1, attention.hidden_size, 6.5 + layer * 0.015625, mx.bfloat16
        )[None]
        entry = boundary_probe._clone_entry(source[layer], context + 1)
        qr, _, _ = profile._projection(attention, x)
        query = attention.indexer.wq_b(qr).reshape(
            1, 1, attention.indexer.n_heads, attention.indexer.head_dim
        )
        profile._pool_update(attention.indexer, entry[1], x)
        scored = profile._score(attention.indexer, entry[1], x, query)
        _eval(scored)
        score_rows.append(scored["index_scores"])
        del entry
    _progress("time_topk", context=context)
    oracle_timing = _selector_timing(score_rows, _oracle_topk, warmups, samples)
    candidate_timing = _selector_timing(
        score_rows, exact_partial_topk, warmups, samples
    )
    _progress("full_model", context=context)
    oracle_trajectory = _trajectory(
        model, source, boundary_probe, context, candidate=False, steps=4
    )
    candidate_trajectory = _trajectory(
        model, source, boundary_probe, context, candidate=True, steps=4
    )
    baseline = _timed_model(
        model, source, boundary_probe, context,
        candidate=False, warmups=warmups, samples=samples,
    )
    candidate = _timed_model(
        model, source, boundary_probe, context,
        candidate=True, warmups=warmups, samples=samples,
    )
    result = {
        "context_tokens": context,
        "pool_count_after_decode": (context + 4) // 4,
        "layers": layers,
        "all_11_layers_exact": len(layers) == 11 and all(row["all_exact"] for row in layers),
        "topk": {
            "oracle": oracle_timing,
            "candidate": candidate_timing,
            "saving_ms": oracle_timing["median_wall_ms"] - candidate_timing["median_wall_ms"],
            "speedup": oracle_timing["median_wall_ms"] / candidate_timing["median_wall_ms"],
        },
        "trajectory": {
            "steps": 4,
            "all_full_vocab_logits_exact": oracle_trajectory["logits_hashes"]
            == candidate_trajectory["logits_hashes"],
            "post_state_exact": oracle_trajectory["post_state_hash"]
            == candidate_trajectory["post_state_hash"],
        },
        "full_model": {
            "baseline": baseline,
            "candidate": candidate,
            "saving_ms": baseline["median_wall_ms"] - candidate["median_wall_ms"],
            "speedup": baseline["median_wall_ms"] / candidate["median_wall_ms"],
        },
        "resource": {
            "threadgroup_scratch_bytes_per_dispatch": 16 * 4 + 4 + 8 + 4 + SELECT_K * 8,
            "persistent_allocation_bytes": 0,
            "anonymous_allocation_count": 0,
            "metal_buffer_count_api_available": False,
            "command_buffer_count_api_available": False,
            "command_buffer_count_inferred": False,
        },
    }
    _release(score_rows, source)
    return result


def _acceptance(artifact: dict) -> dict:
    contexts = artifact.get("contexts", {})
    complete = set(map(int, contexts)) == set(CONTEXTS)
    rows = list(contexts.values())
    long = contexts.get(str(max(CONTEXTS)), {})
    topk = long.get("topk", {})
    full = long.get("full_model", {})
    candidate_topk = topk.get("candidate", {})
    candidate_full = full.get("candidate", {})
    return {
        "artificial_values_indices_ordering_ties_exact": bool(
            artifact.get("artificial", {}).get("all_values_and_indices_byte_exact")
        ),
        "all_contexts_and_11_dsa_layers_exact": complete
        and all(row.get("all_11_layers_exact") for row in rows),
        "expanded_gather_attention_and_state_exact": complete
        and all(
            row["trajectory"]["all_full_vocab_logits_exact"]
            and row["trajectory"]["post_state_exact"]
            for row in rows
        ),
        "repeat_full_vocab_and_cache_state_exact": complete
        and all(
            row["full_model"]["baseline"]["final_logits_hash"]
            == row["full_model"]["candidate"]["final_logits_hash"]
            and row["full_model"]["baseline"]["post_state_hash"]
            == row["full_model"]["candidate"]["post_state_hash"]
            for row in rows
        ),
        "256k_topk_at_most_1_15ms": complete
        and candidate_topk.get("median_wall_ms", 1e9) <= TOPK_KEEP_MS,
        "256k_topk_saves_at_least_0_75ms": complete
        and topk.get("saving_ms", -1e9) >= MIN_TOPK_SAVING_MS,
        "256k_full_model_at_most_81_2ms": complete
        and candidate_full.get("median_wall_ms", 1e9) <= MAX_FULL_MODEL_MS,
        "256k_full_model_saves_at_least_0_7ms": complete
        and full.get("saving_ms", -1e9) >= 0.7,
        "additional_working_peak_at_most_32mib": complete
        and all(
            row["topk"]["candidate"]["working_peak_delta_bytes"]
            <= MAX_WORKING_PEAK_DELTA
            for row in rows
        ),
        "persistent_and_anonymous_allocation_zero": complete
        and all(
            row["resource"]["persistent_allocation_bytes"] == 0
            and row["resource"]["anonymous_allocation_count"] == 0
            for row in rows
        ),
        "nan_oob_and_metal_errors_zero": complete
        and all(
            layer["score_nan_count"] == 0
            for row in rows
            for layer in row["layers"]
        )
        and artifact.get("metal_error_count", 0) == 0,
        "official_16_128_oracle_exact": bool(
            artifact.get("official_oracle", {}).get("first_16_match")
            and artifact.get("official_oracle", {}).get("full_128_match")
        ),
        "runtime_server_apc_abi_unchanged": artifact.get("runtime_changes")
        == {
            "runtime": False,
            "server": False,
            "apc": False,
            "cache_abi": False,
            "kernel_abi": False,
        },
    }


def _rejection_screen(artifact: dict) -> dict:
    contexts = artifact.get("contexts", {})
    required = {2_048, 262_144}
    present = required.issubset(set(map(int, contexts)))
    rows = [contexts[str(context)] for context in sorted(required)] if present else []
    exact = present and all(
        row["all_11_layers_exact"]
        and row["trajectory"]["all_full_vocab_logits_exact"]
        and row["trajectory"]["post_state_exact"]
        and row["full_model"]["baseline"]["final_logits_hash"]
        == row["full_model"]["candidate"]["final_logits_hash"]
        and row["full_model"]["baseline"]["post_state_hash"]
        == row["full_model"]["candidate"]["post_state_hash"]
        for row in rows
    )
    long = contexts.get("262144", {})
    full = long.get("full_model", {})
    regression_ms = -float(full.get("saving_ms", 0.0)) if present else 0.0
    decisive_regression = present and regression_ms >= 0.7
    return {
        "contexts": [2_048, 262_144],
        "contexts_present": present,
        "all_operator_logits_and_state_exact": exact,
        "full_model_regression_ms_at_256k": regression_ms,
        "decisive_regression_threshold_ms": 0.7,
        "decisive_full_model_regression": decisive_regression,
        "complete": bool(
            artifact.get("artificial", {}).get(
                "all_values_and_indices_byte_exact"
            )
            and exact
            and decisive_regression
        ),
        "scope": (
            "early rejection only; no 32K/128K qualification claim"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contexts", type=int, nargs="+", choices=CONTEXTS)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    args = parser.parse_args()
    if not mx.metal.is_available():
        raise RuntimeError("partial top-k probe requires Metal")
    report = inspect_checkpoint(args.model, require_server_ready=True)
    oracle_probe, operator_probe, boundary_probe, profile = _load_probe_modules()
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    _progress("artificial_fixtures")
    artificial = _run_artificial()
    if not artificial["all_values_and_indices_byte_exact"]:
        raise RuntimeError("partial top-k failed artificial exactness gate")
    _progress("load_model")
    model, processor = load(
        args.model,
        experimental_packed_decode_moe=True,
        experimental_compact_nope_dsa_cache=True,
        compact_cache_capacity_tokens=max(CONTEXTS) + args.warmups + args.samples + 8,
    )
    warm_residency(model)
    official_oracle = oracle_probe._official_oracle(model, processor, report)
    schema = "glm53-exact-partial-topk-metal-v3"
    artifact = None
    if args.output.exists():
        candidate = json.loads(args.output.read_text())
        if candidate.get("schema") == schema:
            if candidate.get("checkpoint_fingerprint") != report.fingerprint:
                raise ValueError("existing artifact uses another checkpoint")
            artifact = candidate
    if artifact is None:
        artifact = {
            "schema": schema,
            "date": date.today().isoformat(),
            "complete": False,
            "accepted": False,
            "probe_only": True,
            "checkpoint_fingerprint": report.fingerprint,
            "official_hf_revision": report.official_revision,
            "mlx_vlm_revision": MLX_VLM_REVISION,
            "compact_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
            "contexts_requested": list(CONTEXTS),
            "artificial": artificial,
            "official_oracle": official_oracle,
            "contexts": {},
            "metal_error_count": 0,
            "kernel_contract": {
                "input": "existing finite BF16 or FP32 Indexer score tensor",
                "observed_production_dtype": "recorded per layer; expected bfloat16",
                "output_k": SELECT_K,
                "algorithm": "32-bit signed finite FP32 order plus 17-bit source index; radix select and in-threadgroup bitonic order",
                "score_domain": "all finite BF16/FP32 values; NaN rejected by probe contract",
                "tie_order": "stable ascending source index",
                "score_generation_changed": False,
                "score_topk_fused": False,
                "full_sort_materialized": False,
                "max_pool_count": MAX_POOL_COUNT,
            },
            "performance_gates": {
                "keep_topk_ms_at_256k": TOPK_KEEP_MS,
                "reject_topk_ms_above": TOPK_REJECT_MS,
                "minimum_topk_saving_ms": MIN_TOPK_SAVING_MS,
                "maximum_full_model_ms_at_256k": MAX_FULL_MODEL_MS,
                "maximum_working_peak_delta_bytes": MAX_WORKING_PEAK_DELTA,
            },
            "runtime_changes": {
                "runtime": False,
                "server": False,
                "apc": False,
                "cache_abi": False,
                "kernel_abi": False,
            },
        }
    artifact["artificial"] = artificial
    artifact["official_oracle"] = official_oracle
    requested = tuple(args.contexts or CONTEXTS)
    for context in requested:
        if str(context) in artifact["contexts"] and not args.rerun:
            _progress("skip_completed_context", context=context)
            continue
        artifact["contexts"][str(context)] = _context_case(
            model,
            boundary_probe,
            operator_probe,
            profile,
            context,
            args.warmups,
            args.samples,
        )
        artifact["last_completed_context"] = context
        artifact["peak_memory_bytes"] = int(mx.get_peak_memory())
        artifact["acceptance"] = _acceptance(artifact)
        _atomic_write(args.output, artifact)
    artifact["acceptance"] = _acceptance(artifact)
    artifact["qualification_complete"] = (
        set(map(int, artifact["contexts"])) == set(CONTEXTS)
    )
    artifact["rejection_screen"] = _rejection_screen(artifact)
    artifact["decision_complete"] = bool(
        artifact["qualification_complete"]
        or artifact["rejection_screen"]["complete"]
    )
    artifact["complete"] = artifact["decision_complete"]
    artifact["accepted"] = artifact["qualification_complete"] and all(
        artifact["acceptance"].values()
    )
    if artifact["rejection_screen"]["complete"]:
        artifact["decision"] = "reject_partial_topk_full_model_regression"
    elif not artifact["qualification_complete"]:
        artifact["decision"] = "await_remaining_contexts"
    elif artifact["accepted"]:
        artifact["decision"] = "keep_exact_partial_topk_candidate"
    elif (
        artifact["contexts"][str(max(CONTEXTS))]["topk"]["candidate"][
            "median_wall_ms"
        ]
        > TOPK_REJECT_MS
    ):
        artifact["decision"] = "reject_partial_topk_move_to_pooled_score"
    else:
        artifact["decision"] = "reject_partial_topk_fixed_gate_not_met"
    _atomic_write(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": artifact["complete"],
                "accepted": artifact["accepted"],
                "decision": artifact["decision"],
                "failed_gates": [
                    key for key, value in artifact["acceptance"].items() if not value
                ],
            },
            indent=2,
        )
    )
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
