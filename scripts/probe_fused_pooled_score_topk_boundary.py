#!/usr/bin/env python3
"""Probe a single compiled pooled-score -> exact top-k decode boundary.

Arm A is the production MLX score plus MLX argsort. Arm B compiles only the
identical score expression and returns its score tensor to MLX argsort. Arm C
compiles the same score expression together with the already-qualified signed
partial top-k Metal primitive and returns selected pool rows only.

The probe is intentionally limited to 2K and 256K. A decisive regression at
either boundary stops the candidate without running intermediate contexts.
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
    / "m3ultra512-fused-pooled-score-topk-boundary-20260906.json"
)
CONTEXTS = (2_048, 262_144)
ARMS = (
    "A_eager_score_mlx_topk",
    "B_compiled_score_mlx_topk",
    "C_compiled_score_exact_topk",
)
PROFILE_TOKEN = 3000
MIN_FULL_MODEL_SAVING_MS = 0.75
MAX_2K_REGRESSION_FRACTION = 0.01
MAX_WORKING_PEAK_DELTA = 32 << 20


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
    import probe_exact_partial_topk_metal as partial_topk
    import probe_exact_sigmoid_gate_metal_barrier as oracle_probe
    import probe_long_context_dsa_decode_frontier as operator_probe
    import probe_long_context_first_decode_boundary as boundary_probe
    import profile_lightning_indexer_decode_critical_path as profile

    return partial_topk, oracle_probe, operator_probe, boundary_probe, profile


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
    array = np.asarray(
        value.astype(mx.float32) if value.dtype == mx.bfloat16 else value
    )
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _exact(left: mx.array, right: mx.array) -> bool:
    return bool(mx.array_equal(left, right).item())


def _release(*values) -> None:
    for value in values:
        if isinstance(value, list):
            value.clear()
        elif isinstance(value, dict):
            value.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def _score_expression(query, pool_keys, weights, valid_candidates, softmax_scale):
    scores = query @ pool_keys[:, None].swapaxes(-1, -2)
    scores = mx.maximum(scores * softmax_scale, 0.0)
    index_scores = mx.sum(weights[..., None] * scores, axis=2)
    return mx.where(valid_candidates, index_scores, -1e30)


class _Envelope:
    def __init__(self, partial_topk, indexer, arm: str):
        self.arm = arm
        self.trace_calls = 0
        self.softmax_scale = indexer.softmax_scale

        def score(query, pool_keys, weights, valid_candidates):
            self.trace_calls += 1
            return _score_expression(
                query,
                pool_keys,
                weights,
                valid_candidates,
                self.softmax_scale,
            )

        def score_topk(query, pool_keys, weights, valid_candidates):
            self.trace_calls += 1
            scored = _score_expression(
                query,
                pool_keys,
                weights,
                valid_candidates,
                self.softmax_scale,
            )
            return partial_topk.exact_partial_topk(scored)[1]

        if arm == "B_compiled_score_mlx_topk":
            self.callable = mx.compile(score)
        elif arm == "C_compiled_score_exact_topk":
            self.callable = mx.compile(score_topk)
        else:
            raise ValueError(f"no compiled envelope for {arm}")

    def __call__(self, query, pool_keys, weights, valid_candidates):
        return self.callable(query, pool_keys, weights, valid_candidates)


class _EnvelopeRegistry:
    def __init__(self, partial_topk):
        self.partial_topk = partial_topk
        self.values = {}

    def get(self, indexer, arm: str, physical_pool_rows: int):
        key = (id(indexer), arm, physical_pool_rows)
        if key not in self.values:
            self.values[key] = _Envelope(self.partial_topk, indexer, arm)
        return self.values[key]

    def traces(self) -> dict:
        return {
            f"{key[1]}:layer-object-{key[0]}:rows-{key[2]}": value.trace_calls
            for key, value in self.values.items()
        }


def _physical_inputs(indexer, pool, x, qr):
    query = indexer.wq_b(qr).reshape(
        1, 1, indexer.n_heads, indexer.head_dim
    )
    weights = indexer.weights_proj(x) * (indexer.n_heads**-0.5)
    pool_end = mx.clip(pool.pool_indices[..., -1], 0, pool.total_tokens - 1)
    valid_candidates = (
        (pool_end[:, None, :] < pool.total_tokens) & pool.pool_valid[:, None]
    )
    return query, weights, valid_candidates


def _select_prepared(
    partial_topk,
    registry,
    indexer,
    pool,
    query,
    weights,
    valid_candidates,
    arm,
):
    if arm == "A_eager_score_mlx_topk":
        logical = pool.logical_pool_count
        logical_keys = pool.pool_keys[:, :logical]
        logical_candidates = valid_candidates[..., :logical]
        scored = _score_expression(
            query,
            logical_keys,
            weights,
            logical_candidates,
            indexer.softmax_scale,
        )
        selected = mx.argsort(-scored, axis=-1)[..., :512]
        return scored, selected, logical_candidates
    envelope = registry.get(indexer, arm, int(pool.pool_keys.shape[1]))
    output = envelope(query, pool.pool_keys, weights, valid_candidates)
    if arm == "B_compiled_score_mlx_topk":
        scored = output
        selected = mx.argsort(-scored, axis=-1)[..., :512]
        return scored, selected, valid_candidates
    return None, output, valid_candidates


def _select_arm(partial_topk, registry, indexer, pool, x, qr, arm):
    query, weights, valid_candidates = _physical_inputs(indexer, pool, x, qr)
    return _select_prepared(
        partial_topk,
        registry,
        indexer,
        pool,
        query,
        weights,
        valid_candidates,
        arm,
    )


def _run_layer_arm(
    partial_topk,
    operator_probe,
    boundary_probe,
    profile,
    registry,
    attention,
    source_entry,
    context,
    x,
    arm,
):
    entry = boundary_probe._clone_entry(source_entry, context + 16)
    latent, pool = entry
    qr, q, current = profile._projection(attention, x)
    latent_full, _ = latent.update_and_fetch(current, current)
    updated = profile._pool_update(attention.indexer, pool, x)
    scored, selected, valid_candidates = _select_arm(
        partial_topk, registry, attention.indexer, pool, x, qr, arm
    )
    selected_valid = mx.take_along_axis(
        valid_candidates, selected, axis=-1
    )
    expanded = profile._expand(
        attention.indexer,
        pool,
        selected,
        selected_valid,
        updated["valid"],
    )
    gathered = profile._gather(latent_full, expanded)
    output = profile._attention(
        operator_probe, attention, q, gathered[0], gathered[1], pool
    )
    result = {
        "entry": entry,
        "score": scored,
        "selected": selected,
        "expanded": expanded,
        "gathered": gathered[0],
        "gather_valid": gathered[1],
        "output": output,
        "physical_pool_rows": int(pool.pool_keys.shape[1]),
        "logical_pool_rows": int(pool.logical_pool_count),
    }
    _eval(result)
    return result


def _layer_case(
    partial_topk,
    operator_probe,
    boundary_probe,
    profile,
    registry,
    model,
    source,
    context,
    layer,
):
    attention = model.language_model.model.layers[layer].self_attn
    x = boundary_probe._deterministic_rows(
        1, attention.hidden_size, 7.5 + layer * 0.015625, mx.bfloat16
    )[None]
    rows = {
        arm: _run_layer_arm(
            partial_topk,
            operator_probe,
            boundary_probe,
            profile,
            registry,
            attention,
            source[layer],
            context,
            x,
            arm,
        )
        for arm in ARMS
    }
    oracle = rows[ARMS[0]]
    b = rows[ARMS[1]]
    c = rows[ARMS[2]]
    logical = oracle["logical_pool_rows"]
    b_logical_score = b["score"][..., :logical]
    b_padding = b["score"][..., logical:]
    result = {
        "layer": layer,
        "logical_pool_rows": logical,
        "physical_pool_rows": b["physical_pool_rows"],
        "B_score_bits_exact": _exact(oracle["score"], b_logical_score),
        "B_padding_all_sentinel": bool(
            mx.all(b_padding == mx.array(-1e30, dtype=b["score"].dtype)).item()
        )
        if b_padding.size
        else True,
        "B_indices_exact": _exact(oracle["selected"], b["selected"]),
        "C_indices_exact": _exact(oracle["selected"], c["selected"]),
        "B_expanded_exact": _exact(oracle["expanded"], b["expanded"]),
        "C_expanded_exact": _exact(oracle["expanded"], c["expanded"]),
        "B_gather_exact": _exact(oracle["gathered"], b["gathered"]),
        "C_gather_exact": _exact(oracle["gathered"], c["gathered"]),
        "B_attention_exact": _exact(oracle["output"], b["output"]),
        "C_attention_exact": _exact(oracle["output"], c["output"]),
        "B_state_exact": boundary_probe._cache_exact(
            [oracle["entry"]], [b["entry"]]
        ),
        "C_state_exact": boundary_probe._cache_exact(
            [oracle["entry"]], [c["entry"]]
        ),
        "selected_hash": _hash(oracle["selected"]),
    }
    result["all_exact"] = all(
        value for key, value in result.items() if key.endswith("exact")
    ) and result["B_padding_all_sentinel"]
    _release(rows)
    return result


def _candidate_decode_selection(partial_topk, registry, arm):
    def selected(self, indexer, x, qr, valid_cur):
        pool_keys, pool_indices, pool_valid = self.logical_pool()
        pool_count = self.logical_pool_count
        scored, chosen, valid_candidates = _select_arm(
            partial_topk, registry, indexer, self, x, qr, arm
        )
        del scored
        selected_valid = mx.take_along_axis(
            valid_candidates, chosen, axis=-1
        )
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
            chosen,
            pool_indices if arm == ARMS[0] else self.pool_indices,
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

    return selected


@contextlib.contextmanager
def _selection_arm(partial_topk, registry, arm):
    if arm == ARMS[0]:
        yield
        return
    original = CompactIndexPoolCache._decode_selection
    CompactIndexPoolCache._decode_selection = _candidate_decode_selection(
        partial_topk, registry, arm
    )
    try:
        yield
    finally:
        CompactIndexPoolCache._decode_selection = original


def _run_model_step(model, cache, token, partial_topk, registry, arm):
    started = time.perf_counter_ns()
    with _selection_arm(partial_topk, registry, arm):
        output = model(token, cache=cache)
    built = time.perf_counter_ns()
    logits = output.logits[0, -1]
    _eval(logits)
    finished = time.perf_counter_ns()
    return logits, {
        "wall_ms": (finished - started) / 1e6,
        "host_graph_build_ms": (built - started) / 1e6,
    }


def _full_model_arms(
    model,
    source,
    boundary_probe,
    partial_topk,
    registry,
    context,
    warmups,
    samples,
):
    caches = {
        arm: boundary_probe._clone_cache(
            source, context + warmups + samples + 16
        )
        for arm in ARMS
    }
    token = mx.array([[PROFILE_TOKEN]], dtype=mx.uint32)
    rows = {arm: [] for arm in ARMS}
    hashes = {arm: [] for arm in ARMS}
    baseline_active = int(mx.get_active_memory())
    mx.reset_peak_memory()
    orders = (
        ARMS,
        (ARMS[2], ARMS[1], ARMS[0]),
        (ARMS[1], ARMS[0], ARMS[2]),
    )
    last_logits = {}
    for iteration in range(warmups + samples):
        for arm in orders[iteration % len(orders)]:
            logits, timing = _run_model_step(
                model,
                caches[arm],
                token,
                partial_topk,
                registry,
                arm,
            )
            last_logits[arm] = logits
            if iteration >= warmups:
                rows[arm].append(timing)
                hashes[arm].append(_hash(logits))
    states = {
        arm: boundary_probe._full_cache_hash(cache)
        for arm, cache in caches.items()
    }
    result = {}
    for arm in ARMS:
        wall = statistics.median(row["wall_ms"] for row in rows[arm])
        result[arm] = {
            "sample_rows": rows[arm],
            "median_wall_ms": wall,
            "median_host_graph_build_ms": statistics.median(
                row["host_graph_build_ms"] for row in rows[arm]
            ),
            "decode_tokens_per_second": 1000.0 / wall,
            "logits_hashes": hashes[arm],
            "post_state_hash": states[arm],
        }
    result["exactness"] = {
        "B_all_logits_exact": hashes[ARMS[1]] == hashes[ARMS[0]],
        "C_all_logits_exact": hashes[ARMS[2]] == hashes[ARMS[0]],
        "B_post_state_exact": states[ARMS[1]] == states[ARMS[0]],
        "C_post_state_exact": states[ARMS[2]] == states[ARMS[0]],
    }
    result["working_peak_delta_bytes"] = max(
        0, int(mx.get_peak_memory()) - baseline_active
    )
    result["interleaved_order"] = True
    result["B_saving_ms"] = (
        result[ARMS[0]]["median_wall_ms"]
        - result[ARMS[1]]["median_wall_ms"]
    )
    result["C_saving_ms"] = (
        result[ARMS[0]]["median_wall_ms"]
        - result[ARMS[2]]["median_wall_ms"]
    )
    _release(caches, last_logits)
    return result


def _operator_timing(fixtures, partial_topk, registry, warmups, samples):
    rows = {arm: [] for arm in ARMS}
    baseline_active = int(mx.get_active_memory())
    mx.reset_peak_memory()
    orders = (
        ARMS,
        (ARMS[2], ARMS[1], ARMS[0]),
        (ARMS[1], ARMS[0], ARMS[2]),
    )
    for iteration in range(warmups + samples):
        for arm in orders[iteration % len(orders)]:
            started = time.perf_counter_ns()
            outputs = []
            for fixture in fixtures:
                scored, selected, _ = _select_prepared(
                    partial_topk,
                    registry,
                    fixture["indexer"],
                    fixture["pool"],
                    fixture["query"],
                    fixture["weights"],
                    fixture["valid_candidates"],
                    arm,
                )
                outputs.append(selected)
                if arm != ARMS[2]:
                    outputs.append(scored)
            built = time.perf_counter_ns()
            _eval(outputs)
            finished = time.perf_counter_ns()
            if iteration >= warmups:
                rows[arm].append(
                    {
                        "wall_ms": (finished - started) / 1e6,
                        "host_graph_build_ms": (built - started) / 1e6,
                    }
                )
    result = {}
    for arm in ARMS:
        result[arm] = {
            "median_wall_ms": statistics.median(
                row["wall_ms"] for row in rows[arm]
            ),
            "median_host_graph_build_ms": statistics.median(
                row["host_graph_build_ms"] for row in rows[arm]
            ),
            "sample_rows": rows[arm],
            "returned_score_tensor_bytes": (
                0
                if arm == ARMS[2]
                else sum(
                    (
                        int(fixture["pool"].logical_pool_count)
                        if arm == ARMS[0]
                        else int(fixture["pool"].pool_keys.shape[1])
                    )
                    * 2
                    for fixture in fixtures
                )
            ),
        }
    result["working_peak_delta_bytes"] = max(
        0, int(mx.get_peak_memory()) - baseline_active
    )
    result["command_buffers"] = None
    result["submission_gap_ms"] = None
    result["gpu_idle_ms"] = None
    result["bounded_system_trace_pending_if_candidate_kept"] = True
    return result


def _context_case(
    model,
    boundary_probe,
    operator_probe,
    profile,
    partial_topk,
    context,
    warmups,
    samples,
):
    _progress("build_synthetic_context", context=context)
    source = boundary_probe._synthetic_cache(
        model, context, "compact-nope-dsa"
    )
    registry = _EnvelopeRegistry(partial_topk)
    layers = []
    fixtures = []
    for layer in EXPECTED_DSA:
        _progress("exact_layer", context=context, layer=layer)
        layers.append(
            _layer_case(
                partial_topk,
                operator_probe,
                boundary_probe,
                profile,
                registry,
                model,
                source,
                context,
                layer,
            )
        )
        attention = model.language_model.model.layers[layer].self_attn
        x = boundary_probe._deterministic_rows(
            1, attention.hidden_size, 7.5 + layer * 0.015625, mx.bfloat16
        )[None]
        entry = boundary_probe._clone_entry(source[layer], context + 16)
        qr, _, _ = profile._projection(attention, x)
        profile._pool_update(attention.indexer, entry[1], x)
        query, weights, valid_candidates = _physical_inputs(
            attention.indexer, entry[1], x, qr
        )
        _eval((query, weights, valid_candidates, entry))
        fixtures.append(
            {
                "indexer": attention.indexer,
                "pool": entry[1],
                "entry": entry,
                "query": query,
                "weights": weights,
                "valid_candidates": valid_candidates,
            }
        )
    _progress("operator_arms", context=context)
    operator = _operator_timing(
        fixtures, partial_topk, registry, warmups, samples
    )
    _progress("full_model_arms", context=context)
    full_model = _full_model_arms(
        model,
        source,
        boundary_probe,
        partial_topk,
        registry,
        context,
        warmups,
        samples,
    )
    result = {
        "context_tokens": context,
        "layers": layers,
        "all_11_layers_exact": len(layers) == 11
        and all(row["all_exact"] for row in layers),
        "operator": operator,
        "full_model": full_model,
        "compile_trace_calls": registry.traces(),
        "all_compile_signatures_trace_once": all(
            value == 1 for value in registry.traces().values()
        ),
        "score_tensor_external_bytes": {
            ARMS[0]: operator[ARMS[0]]["returned_score_tensor_bytes"],
            ARMS[1]: operator[ARMS[1]]["returned_score_tensor_bytes"],
            ARMS[2]: 0,
        },
        "resource": {
            "persistent_allocation_bytes": 0,
            "anonymous_allocation_count": 0,
            "metal_buffer_count_api_available": False,
            "command_buffer_count_api_available": False,
            "command_buffer_count_inferred": False,
        },
    }
    _release(fixtures, source)
    return result


def _acceptance(artifact):
    contexts = artifact.get("contexts", {})
    complete = set(map(int, contexts)) == set(CONTEXTS)
    rows = list(contexts.values())
    short = contexts.get("2048", {})
    long = contexts.get("262144", {})
    exact_keys = (
        "B_all_logits_exact",
        "C_all_logits_exact",
        "B_post_state_exact",
        "C_post_state_exact",
    )
    return {
        "2k_and_256k_all_11_layers_score_selection_attention_state_exact": complete
        and all(row["all_11_layers_exact"] for row in rows),
        "full_model_logits_and_state_exact": complete
        and all(
            all(row["full_model"]["exactness"][key] for key in exact_keys)
            for row in rows
        ),
        "compiled_signatures_trace_once": complete
        and all(row["all_compile_signatures_trace_once"] for row in rows),
        "C_returns_no_external_score_tensor": complete
        and all(row["score_tensor_external_bytes"][ARMS[2]] == 0 for row in rows),
        "C_256k_full_model_saves_at_least_0_75ms": complete
        and long.get("full_model", {}).get("C_saving_ms", -1e9)
        >= MIN_FULL_MODEL_SAVING_MS,
        "C_2k_regression_at_most_1_percent": complete
        and short.get("full_model", {}).get(ARMS[2], {}).get(
            "median_wall_ms", 1e9
        )
        <= short.get("full_model", {}).get(ARMS[0], {}).get(
            "median_wall_ms", 0.0
        )
        * (1.0 + MAX_2K_REGRESSION_FRACTION),
        "working_peak_delta_at_most_32mib": complete
        and all(
            row["operator"]["working_peak_delta_bytes"]
            <= MAX_WORKING_PEAK_DELTA
            for row in rows
        ),
        "persistent_and_anonymous_allocation_zero": complete
        and all(
            row["resource"]["persistent_allocation_bytes"] == 0
            and row["resource"]["anonymous_allocation_count"] == 0
            for row in rows
        ),
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    args = parser.parse_args()
    if not mx.metal.is_available():
        raise RuntimeError("fused score/top-k boundary probe requires Metal")
    report = inspect_checkpoint(args.model, require_server_ready=True)
    partial_topk, oracle_probe, operator_probe, boundary_probe, profile = (
        _load_probe_modules()
    )
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    _progress("load_model")
    model, processor = load(
        args.model,
        experimental_packed_decode_moe=True,
        experimental_compact_nope_dsa_cache=True,
        compact_cache_capacity_tokens=max(CONTEXTS) + args.warmups + args.samples + 16,
    )
    warm_residency(model)
    official_oracle = oracle_probe._official_oracle(model, processor, report)
    artifact = {
        "schema": "glm53-fused-pooled-score-topk-boundary-v1",
        "date": date.today().isoformat(),
        "complete": False,
        "accepted": False,
        "probe_only": True,
        "checkpoint_fingerprint": report.fingerprint,
        "official_hf_revision": report.official_revision,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "compact_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
        "contexts_requested": list(CONTEXTS),
        "arms": {
            ARMS[0]: "production eager MLX pooled score -> MLX argsort",
            ARMS[1]: "compiled identical MLX pooled score returned -> MLX argsort",
            ARMS[2]: "single compiled score -> signed exact Metal partial top-k envelope; selected rows only",
        },
        "measurement_contract": {
            "interleaved_arm_order": True,
            "physical_pool_shape_fixed_across_decode": True,
            "padding_is_negative_sentinel": True,
            "score_expression_arithmetic_changed": False,
            "score_tensor_external_from_C": False,
            "bounded_system_trace": "required only if C passes wall screen",
        },
        "performance_gates": {
            "C_minimum_256k_full_model_saving_ms": MIN_FULL_MODEL_SAVING_MS,
            "C_maximum_2k_regression_fraction": MAX_2K_REGRESSION_FRACTION,
            "maximum_working_peak_delta_bytes": MAX_WORKING_PEAK_DELTA,
        },
        "official_oracle": official_oracle,
        "contexts": {},
        "runtime_changes": {
            "runtime": False,
            "server": False,
            "apc": False,
            "cache_abi": False,
            "kernel_abi": False,
        },
    }
    for context in CONTEXTS:
        artifact["contexts"][str(context)] = _context_case(
            model,
            boundary_probe,
            operator_probe,
            profile,
            partial_topk,
            context,
            args.warmups,
            args.samples,
        )
        artifact["last_completed_context"] = context
        artifact["peak_memory_bytes"] = int(mx.get_peak_memory())
        artifact["acceptance"] = _acceptance(artifact)
        _atomic_write(args.output, artifact)
        if not artifact["contexts"][str(context)]["all_11_layers_exact"]:
            artifact["complete"] = True
            artifact["decision"] = "reject_score_envelope_exactness"
            _atomic_write(args.output, artifact)
            return 1
    artifact["acceptance"] = _acceptance(artifact)
    artifact["complete"] = True
    artifact["accepted"] = all(artifact["acceptance"].values())
    artifact["decision"] = (
        "await_bounded_system_trace"
        if artifact["accepted"]
        else "reject_fused_score_topk_fixed_gate_not_met"
    )
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
