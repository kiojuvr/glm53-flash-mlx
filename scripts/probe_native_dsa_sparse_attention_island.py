#!/usr/bin/env python3
"""Probe a decode-only native DSA score -> sparse-attention island.

Tier 1 proved that pooled scoring, exact top-k, and token expansion benefit
from one persistent submission topology.  This probe keeps those selected
indices private and extends the same native command encoder through latent
gather and the exact MLX 0.32.2 D512 attention fallback.  Query/latent
projection and post-attention unembedding remain MLX-owned.

The artificial fixture runs before the 320 GB checkpoint is loaded.  Any bit
difference rejects the widening immediately; this remains probe-only and does
not change the runtime, cache, APC, server, or production kernel ABI.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.metadata
import json
import statistics
import sys
import tempfile
import time
import traceback
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from glm53_flash_mlx.abi import MLX_VLM_REVISION, NOPE_DSA_CACHE_ABI_COMPACT
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import EXPECTED_DSA, inspect_checkpoint
from glm53_flash_mlx.native_execution import (
    NATIVE_DSA_SPARSE_ATTENTION_ISLAND_ABI,
    NATIVE_EXECUTION_ENGINE_ABI,
    plan_native_dsa_sparse_attention_island,
)
from glm53_flash_mlx.nope_cache import CompactIndexPoolCache


ROOT = Path(__file__).resolve().parents[1]
NATIVE_PACKAGE = ROOT / "native_execution"
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-dsa-sparse-attention-island-20260907.json"
)
DECODE_CONTEXTS = (2_048, 262_144)
MIN_256K_INCREMENTAL_SAVING_MS = 0.75
MAX_2K_REGRESSION = 0.01
MAX_CONCURRENT_SCRATCH = 128 << 20
MAX_PROCESS_PEAK_BYTES = 340_000_000_000


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


def _load_helpers():
    for path in (str(ROOT / "scripts"), str(NATIVE_PACKAGE)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from glm53_native_execution import NativeDSASparseAttentionPlan
    import probe_exact_sigmoid_gate_metal_barrier as oracle_probe
    import probe_long_context_dsa_decode_frontier as frontier
    import probe_long_context_first_decode_boundary as boundary_probe
    import probe_native_dsa_score_execution_island as tier1
    import probe_native_execution_engine_feasibility as tier0

    return (
        NativeDSASparseAttentionPlan,
        oracle_probe,
        frontier,
        boundary_probe,
        tier1,
        tier0,
    )


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


def _exact(left: mx.array, right: mx.array) -> bool:
    mx.eval(left, right)
    return bool(mx.array_equal(left, right).item())


def _release(*values) -> None:
    for value in values:
        if isinstance(value, (list, dict)):
            value.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def _median(rows: list[dict]) -> dict:
    return {
        "median_wall_ms": statistics.median(row["wall_ms"] for row in rows),
        "median_host_submit_ms": statistics.median(
            row["host_submit_ms"] for row in rows
        ),
        "samples": rows,
    }


class _Registry:
    def __init__(self, plan_type):
        self.plan_type = plan_type
        self.plans = {}
        self.initial_identities = {}

    def get(self, pool, attention, latent):
        indexer = attention.indexer
        key = (
            id(pool),
            id(indexer),
            int(pool.pool_keys.shape[1]),
            int(latent.shape[2]),
        )
        if key not in self.plans:
            plan = self.plan_type(
                key[2],
                key[3],
                float(indexer.softmax_scale),
                float(attention.scale),
            )
            self.plans[key] = plan
            self.initial_identities[key] = list(plan.buffer_identities)
        return self.plans[key]

    def evidence(self) -> list[dict]:
        rows = []
        for key, plan in self.plans.items():
            rows.append(
                {
                    "key": list(map(str, key)),
                    "physical_pool_rows": plan.physical_pool_rows,
                    "physical_kv_rows": plan.physical_kv_rows,
                    "selected_width": plan.selected_width,
                    "execution_count": plan.execution_count,
                    "scratch_bytes": plan.scratch_bytes,
                    "returned_intermediate_tensor_bytes": (
                        plan.returned_intermediate_tensor_bytes
                    ),
                    "dynamic_allocation_count": plan.dynamic_allocation_count,
                    "graph_node_count": plan.graph_node_count,
                    "shape_discovery_count": plan.shape_discovery_count,
                    "host_synchronization_count": (
                        plan.host_synchronization_count
                    ),
                    "buffer_identities_stable": list(plan.buffer_identities)
                    == self.initial_identities[key],
                }
            )
        return rows


def _native_execute(plan, fixture):
    pool = fixture["pool"]
    dependencies = (
        fixture["index_query"],
        fixture["weights"],
        pool.pool_keys,
        pool.pool_indices,
        pool.pool_valid,
        pool.raw_positions,
        pool.raw_valid,
        fixture["current_valid"],
        fixture["attention_query"],
        fixture["latent_physical"],
    )
    mx.async_eval(*dependencies)
    return plan.execute(
        *dependencies,
        pool.logical_pool_count,
        pool.total_tokens,
        pool.active_tail_count,
    )


def _reference_attention(fixture, tier0):
    pool = fixture["pool"]
    indices, valid = tier0._reference_selection(
        fixture["attention"].indexer,
        pool,
        fixture["scores"],
        fixture["valid_candidates"],
        fixture["current_valid"],
    )
    safe = mx.where(valid, indices, 0)
    expanded = mx.broadcast_to(
        safe[..., None], safe.shape + (fixture["latent_physical"].shape[-1],)
    )
    gathered = mx.take_along_axis(
        fixture["latent_physical"], expanded, axis=2
    )
    output = mx.fast.scaled_dot_product_attention(
        fixture["attention_query"],
        gathered,
        gathered,
        scale=fixture["attention"].scale,
        mask=valid[:, :, None, :],
    )
    return output, indices, valid, gathered


def _artificial_contract(plan_type, tier0) -> dict:
    pools, logical, kv_rows, kv_len = 512, 512, 2_112, 2_051
    index_axis = mx.arange(32 * 128, dtype=mx.float32).reshape(1, 1, 32, 128)
    index_query = mx.sin(index_axis * 0.001).astype(mx.bfloat16)
    weights = mx.cos(mx.arange(32, dtype=mx.float32) * 0.03).reshape(
        1, 1, 32
    ).astype(mx.bfloat16)
    pool_keys = mx.cos(
        mx.arange(pools * 128, dtype=mx.float32).reshape(1, pools, 128)
        * 0.0001
    ).astype(mx.bfloat16)
    pool_indices = mx.arange(pools * 4, dtype=mx.int64).reshape(1, pools, 4)
    pool_valid = mx.arange(pools)[None] < (pools - 1)
    raw_positions = mx.array([[0, 2048, 2049, 2050]], dtype=mx.int64)
    raw_valid = mx.array([[False, True, True, True]], dtype=mx.bool_)
    current_valid = mx.ones((1,), dtype=mx.bool_)
    attention_query = mx.sin(
        mx.arange(64 * 512, dtype=mx.float32).reshape(1, 64, 1, 512)
        * 0.0003
    ).astype(mx.bfloat16)
    latent = mx.cos(
        mx.arange(kv_rows * 512, dtype=mx.float32).reshape(1, 1, kv_rows, 512)
        * 0.00007
    ).astype(mx.bfloat16)
    valid_candidates = mx.broadcast_to(pool_valid[:, None], (1, 1, pools))
    scores = tier0._score_expression(
        index_query, pool_keys, weights, valid_candidates, 128**-0.5
    )
    pool = SimpleNamespace(
        pool_keys=pool_keys,
        pool_indices=pool_indices,
        pool_valid=pool_valid,
        raw_positions=raw_positions,
        raw_valid=raw_valid,
        logical_pool_count=logical,
        total_tokens=kv_len,
        active_tail_count=3,
    )
    fixture = {
        "attention": SimpleNamespace(
            scale=512**-0.5,
            indexer=SimpleNamespace(
                softmax_scale=128**-0.5,
                index_topk=2048,
                index_kpool=4,
                index_kpool_always_select_tail=True,
            ),
        ),
        "pool": pool,
        "index_query": index_query,
        "weights": weights,
        "scores": scores,
        "valid_candidates": valid_candidates,
        "current_valid": current_valid,
        "attention_query": attention_query,
        "latent_physical": latent,
    }
    _eval(fixture)
    reference, indices, valid, gathered = _reference_attention(fixture, tier0)
    plan = plan_type(pools, kv_rows, 128**-0.5, 512**-0.5)
    before = list(plan.buffer_identities)
    candidate = _native_execute(plan, fixture)
    _eval((reference, candidate, gathered, plan.debug_gathered_latent))
    result = {
        "selected_indices_byte_exact": _exact(
            indices, plan.debug_selected_indices
        ),
        "selected_valid_byte_exact": _exact(valid, plan.debug_selected_valid),
        "gathered_latent_byte_exact": _exact(
            gathered.reshape(2051, 512), plan.debug_gathered_latent
        ),
        "attention_output_byte_exact": _exact(reference, candidate),
        "tail_width": 3,
        "invalid_slots": int(2051 - mx.sum(valid).item()),
        "buffer_identities_stable": before == list(plan.buffer_identities),
        "dynamic_allocation_count": plan.dynamic_allocation_count,
        "graph_node_count": plan.graph_node_count,
        "shape_discovery_count": plan.shape_discovery_count,
        "host_synchronization_count": plan.host_synchronization_count,
        "returned_intermediate_tensor_bytes": (
            plan.returned_intermediate_tensor_bytes
        ),
    }
    result["all_exact"] = all(
        result[name]
        for name in (
            "selected_indices_byte_exact",
            "selected_valid_byte_exact",
            "gathered_latent_byte_exact",
            "attention_output_byte_exact",
            "buffer_identities_stable",
        )
    )
    return result


def _prepared_fixture(model, boundary_probe, source, context, layer):
    attention = model.language_model.model.layers[layer].self_attn
    entry = boundary_probe._clone_entry(source[layer], context + 32)
    latent_cache, pool = entry
    x = boundary_probe._deterministic_rows(
        1, attention.hidden_size, 19.0 + layer * 0.015625, mx.bfloat16
    )[None]
    qr = attention.q_a_layernorm(attention.q_a_proj(x))
    q = attention.q_b_proj(qr).reshape(
        1, 1, attention.num_heads, attention.q_head_dim
    ).transpose(0, 2, 1, 3)
    latent_current = attention.kv_a_layernorm(
        attention.kv_a_proj_with_mqa(x)
    )[:, None]
    latent_cache.update_and_fetch(latent_current, latent_current)
    short_bypass = pool.validate_update(attention.indexer, batch=1, length=1)
    if short_bypass:
        raise AssertionError("sparse-attention fixture unexpectedly bypassed")
    keys = attention.indexer.k_norm(attention.indexer.wk(x)).reshape(
        1, 1, attention.indexer.head_dim
    )
    gates = x @ attention.indexer.index_kpool_compress_gate.swapaxes(-1, -2)
    current_valid = mx.ones((1, 1), dtype=mx.bool_)
    pool._append_projected(keys, gates, current_valid)
    index_query = attention.indexer.wq_b(qr).reshape(
        1, 1, attention.indexer.n_heads, attention.indexer.head_dim
    )
    weights = attention.indexer.weights_proj(x) * (
        attention.indexer.n_heads**-0.5
    )
    pool_end = mx.clip(pool.pool_indices[..., -1], 0, pool.total_tokens - 1)
    valid_candidates = (
        (pool_end[:, None, :] < pool.total_tokens) & pool.pool_valid[:, None]
    )
    valid_candidates = mx.broadcast_to(
        valid_candidates, (1, 1, int(pool.pool_keys.shape[1]))
    )
    scores = tier0_score = (
        index_query @ pool.pool_keys[:, None].swapaxes(-1, -2)
    )
    tier0_score = mx.maximum(
        tier0_score * attention.indexer.softmax_scale, 0.0
    )
    scores = mx.sum(weights[..., None] * tier0_score, axis=2)
    scores = mx.where(valid_candidates, scores, -1e30)
    fixture = {
        "layer": layer,
        "attention": attention,
        "entry": entry,
        "pool": pool,
        "index_query": index_query,
        "weights": weights,
        "scores": scores,
        "valid_candidates": valid_candidates,
        "current_valid": current_valid.reshape(1),
        "attention_query": attention.embed_q(q),
        "latent_physical": latent_cache.keys,
    }
    _eval(fixture)
    return fixture


def _exact_fixture(fixture, registry, tier0) -> dict:
    reference, indices, valid, gathered = _reference_attention(fixture, tier0)
    plan = registry.get(
        fixture["pool"], fixture["attention"], fixture["latent_physical"]
    )
    before = list(plan.buffer_identities)
    candidate = _native_execute(plan, fixture)
    _eval((reference, candidate, gathered, plan.debug_gathered_latent))
    return {
        "layer": fixture["layer"],
        "selected_indices_byte_exact": _exact(
            indices, plan.debug_selected_indices
        ),
        "selected_valid_byte_exact": _exact(valid, plan.debug_selected_valid),
        "gathered_latent_byte_exact": _exact(
            gathered.reshape(2051, 512), plan.debug_gathered_latent
        ),
        "attention_output_byte_exact": _exact(reference, candidate),
        "buffer_identities_stable": before == list(plan.buffer_identities),
    }


def _operator_timing(
    fixtures, tier1_registry, registry, tier1, tier0, warmups, samples
):
    measured = {"A_tier1_mlx_attention": [], "B_tier2_native_attention": []}
    orders = (
        ("A_tier1_mlx_attention", "B_tier2_native_attention"),
        ("B_tier2_native_attention", "A_tier1_mlx_attention"),
    )
    for iteration in range(warmups + samples):
        for arm in orders[iteration % 2]:
            started = time.perf_counter_ns()
            outputs = []
            for fixture in fixtures:
                if arm == "A_tier1_mlx_attention":
                    pool = fixture["pool"]
                    indexer = fixture["attention"].indexer
                    plan = tier1_registry.get(pool, indexer, 1, "decode")
                    indices, valid = tier1._native_execute(
                        plan,
                        pool,
                        fixture["index_query"],
                        fixture["weights"],
                        fixture["current_valid"],
                    )
                    safe = mx.where(valid, indices, 0)
                    gathered = mx.take_along_axis(
                        fixture["latent_physical"],
                        mx.broadcast_to(
                            safe[..., None], safe.shape + (512,)
                        ),
                        axis=2,
                    )
                    outputs.append(
                        mx.fast.scaled_dot_product_attention(
                            fixture["attention_query"],
                            gathered,
                            gathered,
                            scale=fixture["attention"].scale,
                            mask=valid[:, :, None, :],
                        )
                    )
                else:
                    plan = registry.get(
                        fixture["pool"],
                        fixture["attention"],
                        fixture["latent_physical"],
                    )
                    outputs.append(_native_execute(plan, fixture))
            submitted = time.perf_counter_ns()
            _eval(outputs)
            finished = time.perf_counter_ns()
            if iteration >= warmups:
                measured[arm].append(
                    {
                        "host_submit_ms": (submitted - started) / 1e6,
                        "wall_ms": (finished - started) / 1e6,
                    }
                )
    result = {arm: _median(rows) for arm, rows in measured.items()}
    result["incremental_native_saving_ms"] = (
        result["A_tier1_mlx_attention"]["median_wall_ms"]
        - result["B_tier2_native_attention"]["median_wall_ms"]
    )
    return result


def _candidate_sparse_call(original, registry):
    def sparse_call(self, x, mask=None, cache=None):
        if (
            cache is None
            or not isinstance(cache[1], CompactIndexPoolCache)
            or int(x.shape[1]) != 1
            or cache[1].total_tokens < cache[1].index_topk
        ):
            return original(self, x, mask=mask, cache=cache)

        batch, length, _ = x.shape
        pool = cache[1]
        short_bypass = pool.validate_update(
            self.indexer, batch=int(batch), length=int(length)
        )
        if short_bypass:
            raise AssertionError("native sparse path cannot execute dense bypass")
        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr).reshape(
            batch, length, self.num_heads, self.q_head_dim
        ).transpose(0, 2, 1, 3)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        latent_current = mx.expand_dims(
            self.kv_a_layernorm(compressed_kv), axis=1
        )
        cache[0].update_and_fetch(latent_current, latent_current)

        keys = self.indexer.k_norm(self.indexer.wk(x)).reshape(
            1, 1, self.indexer.head_dim
        )
        gates = x @ self.indexer.index_kpool_compress_gate.swapaxes(-1, -2)
        if mask is not None and mask.dtype == mx.bool_ and mask.shape == (1, 1):
            current_valid = mask
        else:
            current_valid = mx.ones((1, 1), dtype=mx.bool_)
        pool._append_projected(keys, gates, current_valid)

        dependencies = pool.dependency_arrays()
        if cache[0].keys is not None and dependencies:
            cache[0].keys = mx.depends(cache[0].keys, dependencies)
        latent_physical = cache[0].keys
        index_query = self.indexer.wq_b(qr).reshape(
            1, 1, self.indexer.n_heads, self.indexer.head_dim
        )
        weights = self.indexer.weights_proj(x) * (
            self.indexer.n_heads**-0.5
        )
        attention_query = self.embed_q(q)
        fixture = {
            "pool": pool,
            "index_query": index_query,
            "weights": weights,
            "current_valid": current_valid.reshape(1),
            "attention_query": attention_query,
            "latent_physical": latent_physical,
        }
        plan = registry.get(pool, self, latent_physical)
        output = _native_execute(plan, fixture)
        output = self.unembed_out(output)
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)

    return sparse_call


@contextlib.contextmanager
def _tier2_arm(attention_type, registry, enabled):
    if not enabled:
        yield
        return
    original = attention_type.__call__
    attention_type.__call__ = _candidate_sparse_call(original, registry)
    try:
        yield
    finally:
        attention_type.__call__ = original


def _full_model_case(
    model,
    boundary_probe,
    tier1,
    tier0,
    source,
    context,
    tier1_registry,
    registry,
    warmups,
    samples,
):
    arms = ("A_tier1_mlx_attention", "B_tier2_native_attention")
    caches = {
        arm: boundary_probe._clone_cache(source, context + warmups + samples + 32)
        for arm in arms
    }
    measured = {arm: [] for arm in arms}
    hashes = {arm: [] for arm in arms}
    token = mx.array([[3000]], dtype=mx.uint32)
    attention_type = type(model.language_model.model.layers[EXPECTED_DSA[0]].self_attn)
    orders = (arms, tuple(reversed(arms)))
    for iteration in range(warmups + samples):
        for arm in orders[iteration % 2]:
            started = time.perf_counter_ns()
            if arm == "A_tier1_mlx_attention":
                with tier1._native_arm(tier1_registry, True):
                    output = model(token, cache=caches[arm])
            else:
                with _tier2_arm(attention_type, registry, True):
                    output = model(token, cache=caches[arm])
            submitted = time.perf_counter_ns()
            logits = output.logits[0, -1]
            _eval(logits)
            finished = time.perf_counter_ns()
            if iteration >= warmups:
                measured[arm].append(
                    {
                        "host_submit_ms": (submitted - started) / 1e6,
                        "wall_ms": (finished - started) / 1e6,
                    }
                )
                hashes[arm].append(tier0._hash(logits))
    result = {arm: _median(rows) for arm, rows in measured.items()}
    result["incremental_native_saving_ms"] = (
        result[arms[0]]["median_wall_ms"] - result[arms[1]]["median_wall_ms"]
    )
    result["all_logits_byte_exact"] = hashes[arms[0]] == hashes[arms[1]]
    result["post_state_byte_exact"] = boundary_probe._cache_exact(
        caches[arms[0]], caches[arms[1]]
    )
    _release(caches)
    return result


def _acceptance(artifact) -> dict[str, bool]:
    short = artifact.get("contexts", {}).get("2048", {})
    long = artifact.get("contexts", {}).get("262144", {})
    evidence = artifact.get("native_plan_evidence", [])
    exact = bool(artifact.get("artificial_native_contract", {}).get("all_exact"))
    exact = exact and bool(short) and bool(long) and all(
        row.get("all_layers_exact")
        and row.get("full_model", {}).get("all_logits_byte_exact")
        and row.get("full_model", {}).get("post_state_byte_exact")
        for row in (short, long)
    )
    return {
        "artificial_and_all_dsa_layers_byte_exact": exact,
        "fixed_arena_and_zero_execute_allocation_graph_sync": bool(evidence)
        and all(
            row.get("buffer_identities_stable")
            and row.get("dynamic_allocation_count") == 0
            and row.get("graph_node_count") == 0
            and row.get("shape_discovery_count") == 0
            and row.get("host_synchronization_count") == 0
            and row.get("returned_intermediate_tensor_bytes") == 0
            for row in evidence
        ),
        "256k_incremental_full_model_saving_at_least_0_75ms": long.get(
            "full_model", {}
        ).get("incremental_native_saving_ms", -1e9)
        >= MIN_256K_INCREMENTAL_SAVING_MS,
        "2k_full_model_regression_at_most_1_percent": short.get(
            "full_model", {}
        ).get("B_tier2_native_attention", {}).get("median_wall_ms", 1e9)
        <= short.get("full_model", {}).get("A_tier1_mlx_attention", {}).get(
            "median_wall_ms", 0.0
        )
        * (1.0 + MAX_2K_REGRESSION),
        "native_plan_scratch_at_most_128MiB": artifact.get(
            "maximum_concurrent_native_plan_scratch_bytes", 1 << 60
        )
        <= MAX_CONCURRENT_SCRATCH,
        "process_peak_at_most_340GB": artifact.get(
            "process_peak_memory_bytes", 1 << 60
        )
        <= MAX_PROCESS_PEAK_BYTES,
        "official_oracle_exact": bool(
            artifact.get("official_oracle", {}).get("first_16_match")
            and artifact.get("official_oracle", {}).get("full_128_match")
            and artifact.get("official_oracle", {}).get(
                "all_full_vocab_logits_hashes_match"
            )
        ),
        "production_abi_unchanged": artifact.get("runtime_changes")
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
    parser.add_argument("--artificial-only", action="store_true")
    args = parser.parse_args()
    artifact = {
        "schema": "glm53-native-dsa-sparse-attention-island-v1",
        "date": date.today().isoformat(),
        "complete": False,
        "accepted": False,
        "probe_only": True,
        "engine_abi": NATIVE_EXECUTION_ENGINE_ABI,
        "island_abi": NATIVE_DSA_SPARSE_ATTENTION_ISLAND_ABI,
        "tier": "tier2-decode-score-selection-gather-d512-attention",
        "scope_limits": {
            "decode_only": True,
            "query_projection_native": False,
            "pool_update_native": False,
            "score_selection_native": True,
            "sparse_gather_attention_native": True,
            "unembed_output_projection_native": False,
            "kda_native": False,
            "moe_native": False,
        },
        "runtime_changes": {
            "runtime": False,
            "server": False,
            "apc": False,
            "cache_abi": False,
            "kernel_abi": False,
        },
        "mlx_version": importlib.metadata.version("mlx"),
    }
    baseline_peak = int(mx.get_peak_memory())
    try:
        if not mx.metal.is_available():
            raise RuntimeError("native sparse-attention probe requires Metal")
        Plan, oracle_probe, frontier, boundary_probe, tier1, tier0 = _load_helpers()
        _progress("artificial_native_contract")
        artifact["artificial_native_contract"] = _artificial_contract(
            Plan, tier0
        )
        _atomic_write(args.output, artifact)
        if not artifact["artificial_native_contract"]["all_exact"]:
            artifact["complete"] = True
            artifact["decision"] = "reject_native_sparse_attention_numerical_order"
            _atomic_write(args.output, artifact)
            return 1
        if args.artificial_only:
            artifact["complete"] = True
            artifact["decision"] = (
                "artificial_screen_passed_full_qualification_required"
            )
            _atomic_write(args.output, artifact)
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "complete": True,
                        "accepted": False,
                        "decision": artifact["decision"],
                    },
                    indent=2,
                )
            )
            return 0

        report = inspect_checkpoint(args.model, require_server_ready=True)
        artifact.update(
            {
                "checkpoint_fingerprint": report.fingerprint,
                "official_hf_revision": report.official_revision,
                "mlx_vlm_revision": MLX_VLM_REVISION,
                "compact_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
                "plan_contract": plan_native_dsa_sparse_attention_island(
                    logical_capacity_tokens=262_145
                ).descriptor(),
            }
        )
        mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
        mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
        _progress("load_model")
        model, processor = load(
            args.model,
            experimental_packed_decode_moe=True,
            experimental_compact_nope_dsa_cache=True,
            compact_cache_capacity_tokens=max(DECODE_CONTEXTS) + 64,
        )
        warm_residency(model)
        artifact["official_oracle"] = oracle_probe._official_oracle(
            model, processor, report
        )
        baseline_peak = int(mx.get_peak_memory())
        artifact["contexts"] = {}
        evidence = []
        for context in DECODE_CONTEXTS:
            _progress("decode_context", context=context)
            source = boundary_probe._synthetic_cache(
                model, context, "compact-nope-dsa"
            )
            registry = _Registry(Plan)
            tier1_registry = tier1._Registry(tier1._load_helpers()[0])
            fixtures = [
                _prepared_fixture(model, boundary_probe, source, context, layer)
                for layer in EXPECTED_DSA
            ]
            exact = [_exact_fixture(row, registry, tier0) for row in fixtures]
            all_exact = all(
                all(
                    row[name]
                    for name in (
                        "selected_indices_byte_exact",
                        "selected_valid_byte_exact",
                        "gathered_latent_byte_exact",
                        "attention_output_byte_exact",
                        "buffer_identities_stable",
                    )
                )
                for row in exact
            )
            operator_timing = _operator_timing(
                fixtures,
                tier1_registry,
                registry,
                tier1,
                tier0,
                args.warmups,
                args.samples,
            )
            operator_evidence = registry.evidence()
            operator_scratch = sum(
                item["scratch_bytes"] for item in operator_evidence
            )
            _release(fixtures)
            registry.plans.clear()
            tier1_registry.plans.clear()
            _release()
            full_registry = _Registry(Plan)
            full_tier1_registry = tier1._Registry(tier1._load_helpers()[0])
            row = {
                "context_tokens": context,
                "layers": exact,
                "all_layers_exact": all_exact,
                "operator_timing": operator_timing,
            }
            if all_exact:
                row["full_model"] = _full_model_case(
                    model,
                    boundary_probe,
                    tier1,
                    tier0,
                    source,
                    context,
                    full_tier1_registry,
                    full_registry,
                    args.warmups,
                    args.samples,
                )
            else:
                row["full_model"] = {"skipped": "operator exactness failed"}
            full_evidence = full_registry.evidence()
            row["concurrent_native_plan_scratch_bytes"] = max(
                operator_scratch,
                sum(item["scratch_bytes"] for item in full_evidence),
            )
            evidence.extend(operator_evidence)
            evidence.extend(full_evidence)
            artifact["contexts"][str(context)] = row
            artifact["native_plan_evidence"] = evidence
            _atomic_write(args.output, artifact)
            full_registry.plans.clear()
            full_tier1_registry.plans.clear()
            _release(source)
            if not all_exact:
                break

        artifact["process_peak_memory_bytes"] = int(mx.get_peak_memory())
        artifact["process_peak_delta_from_warm_model_bytes"] = max(
            0, artifact["process_peak_memory_bytes"] - baseline_peak
        )
        artifact["maximum_concurrent_native_plan_scratch_bytes"] = max(
            (
                row.get("concurrent_native_plan_scratch_bytes", 0)
                for row in artifact["contexts"].values()
            ),
            default=0,
        )
        artifact["acceptance"] = _acceptance(artifact)
        artifact["complete"] = True
        artifact["accepted"] = all(artifact["acceptance"].values())
        artifact["decision"] = (
            "advance_native_boundary_to_unembed_output_projection"
            if artifact["accepted"]
            else "stop_or_redesign_native_sparse_attention_island"
        )
    except Exception as error:
        artifact["complete"] = True
        artifact["accepted"] = False
        artifact["decision"] = "native_sparse_attention_feasibility_failed"
        artifact["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    _atomic_write(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": artifact["complete"],
                "accepted": artifact["accepted"],
                "decision": artifact["decision"],
                "failed_gates": [
                    key
                    for key, value in artifact.get("acceptance", {}).items()
                    if not value
                ],
            },
            indent=2,
        )
    )
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
