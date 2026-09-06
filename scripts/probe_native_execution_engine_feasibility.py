#!/usr/bin/env python3
"""Prove the Tier-0 native prefill/decode submission bridge.

This probe does not claim that KDA, DSA, or score production are native yet.
It tests the prerequisite architecture directly: an already-scheduled,
row-major Indexer score buffer flows through exact top-k and pool expansion in
one C++-owned Metal execution island with fixed scratch/output addresses, no
per-call MLX graph nodes, no executor allocation, and no host synchronization.

The same native ABI is exercised with decode Q=1 and prefill Q=256.  Only a
successful bridge may advance to a wider score -> KDA/DSA native island, where
the final 3 ms/token decode and 1.2x 32K prefill gates become applicable.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import statistics
import sys
import tempfile
import time
import traceback
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np

from glm53_flash_mlx.abi import MLX_VLM_REVISION, NOPE_DSA_CACHE_ABI_COMPACT
from glm53_flash_mlx.indexpool import INDEXPOOL_SENTINEL, expand_selected_pools
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import EXPECTED_DSA, inspect_checkpoint
from glm53_flash_mlx.native_execution import (
    NATIVE_EXECUTION_ENGINE_ABI,
    NativeExecutionMode,
    plan_native_indexer_island,
)
from glm53_flash_mlx.nope_cache import CompactIndexPoolCache


ROOT = Path(__file__).resolve().parents[1]
NATIVE_PACKAGE = ROOT / "native_execution"
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-execution-engine-feasibility-20260906.json"
)
DECODE_CONTEXTS = (2_048, 262_144)
PREFILL_CONTEXT = 32_768
PREFILL_ROWS = 256
SELECTED_POOLS = 512
BRIDGE_MIN_256K_WALL_SAVING_MS = 0.75
BRIDGE_MAX_2K_REGRESSION = 0.01
PREFILL_ISLAND_MIN_SPEEDUP = 1.20
FINAL_NATIVE_DECODE_SAVING_MS = 3.0
FINAL_NATIVE_PREFILL_SPEEDUP = 1.20


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
    scripts = str(ROOT / "scripts")
    native = str(NATIVE_PACKAGE)
    for path in (scripts, native):
        if path not in sys.path:
            sys.path.insert(0, path)
    from glm53_native_execution import NativeIndexSelectionPlan
    import probe_exact_sigmoid_gate_metal_barrier as oracle_probe
    import probe_long_context_first_decode_boundary as boundary_probe
    import profile_lightning_indexer_decode_critical_path as profile

    return NativeIndexSelectionPlan, oracle_probe, boundary_probe, profile


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
    if value.dtype == mx.bfloat16:
        value = value.astype(mx.float32)
        mx.eval(value)
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


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


def _score_expression(query, pool_keys, weights, valid_candidates, scale):
    scores = query @ pool_keys[:, None].swapaxes(-1, -2)
    scores = mx.maximum(scores * scale, 0.0)
    scores = mx.sum(weights[..., None] * scores, axis=2)
    scores = mx.where(valid_candidates, scores, -1e30)
    return mx.contiguous(scores, allow_col_major=False)


def _reference_selection(indexer, pool, scores, valid_candidates, current_valid):
    selected = mx.argsort(-scores, axis=-1)[..., :SELECTED_POOLS]
    selected_valid = mx.take_along_axis(valid_candidates, selected, axis=-1)
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
    valid = valid & current_valid[..., None]
    return mx.where(valid, indices, INDEXPOOL_SENTINEL), valid


class _NativePlanRegistry:
    def __init__(self, plan_type):
        self.plan_type = plan_type
        self.plans = {}
        self.initial_buffer_identities = {}

    def get(self, pool, rows: int, score_dtype, mode: str):
        dtype_name = "bfloat16" if score_dtype == mx.bfloat16 else "float32"
        key = (id(pool), rows, int(pool.pool_keys.shape[1]), dtype_name, mode)
        if key not in self.plans:
            self.plans[key] = self.plan_type(
                mode, rows, int(pool.pool_keys.shape[1]), dtype_name
            )
            self.initial_buffer_identities[key] = list(
                self.plans[key].buffer_identities
            )
        return self.plans[key]

    def evidence(self) -> list[dict]:
        rows = []
        for key, plan in self.plans.items():
            current_identities = list(plan.buffer_identities)
            rows.append(
                {
                    "key": list(map(str, key)),
                    "mode": plan.mode,
                    "query_rows": plan.query_rows,
                    "physical_pool_rows": plan.physical_pool_rows,
                    "selected_width": plan.selected_width,
                    "execution_count": plan.execution_count,
                    "dynamic_allocation_count": plan.dynamic_allocation_count,
                    "graph_node_count": plan.graph_node_count,
                    "shape_discovery_count": plan.shape_discovery_count,
                    "host_synchronization_count": plan.host_synchronization_count,
                    "buffer_identities": current_identities,
                    "buffer_identities_stable": current_identities
                    == self.initial_buffer_identities[key],
                }
            )
        return rows


def _native_selection(plan, pool, scores, current_valid):
    dependencies = (
        scores,
        pool.pool_indices,
        pool.pool_valid,
        pool.raw_positions,
        pool.raw_valid,
        current_valid,
    )
    # Scheduling is outside the native island in Tier 0.  The C++ executor
    # consumes these buffers without waiting or building another MLX graph.
    mx.async_eval(*dependencies)
    indices, valid = plan.execute(
        scores,
        pool.pool_indices,
        pool.pool_valid,
        pool.raw_positions,
        pool.raw_valid,
        current_valid,
        pool.logical_pool_count,
        pool.total_tokens,
        pool.active_tail_count,
    )
    return indices, valid


def _artificial_native_contract(plan_type) -> dict:
    pool_rows = 1_024
    ascending = np.arange(pool_rows, dtype=np.float32)
    kth_tie = np.arange(pool_rows, dtype=np.float32)
    kth_tie[508:516] = np.float32(700.0)
    signed_zero = np.zeros(pool_rows, dtype=np.float32)
    signed_zero[1::2] = np.float32(-0.0)
    patterns = (
        ("ascending", ascending, True),
        ("descending", -ascending, True),
        ("all_equal", np.zeros(pool_rows, dtype=np.float32), True),
        ("kth_tie", kth_tie, True),
        ("signed_zero", signed_zero, True),
        (
            "tiny_differences",
            np.linspace(-2.0e-7, 2.0e-7, pool_rows, dtype=np.float32),
            True,
        ),
        ("invalid_query", ascending, False),
    )
    results = []
    for dtype, dtype_name in ((mx.float32, "float32"), (mx.bfloat16, "bfloat16")):
        scores = mx.stack(
            [mx.array(values, dtype=dtype) for _, values, _ in patterns]
        )[None]
        current_valid = mx.array(
            [row_valid for _, _, row_valid in patterns], dtype=mx.bool_
        )
        pool_indices = mx.arange(
            pool_rows * 4, dtype=mx.int64
        ).reshape(1, pool_rows, 4)
        pool_valid = mx.ones((1, pool_rows), dtype=mx.bool_)
        selected = mx.argsort(-scores, axis=-1)[..., :SELECTED_POOLS]
        selected_valid = mx.ones(selected.shape, dtype=mx.bool_)
        reference, reference_valid = expand_selected_pools(
            selected,
            pool_indices,
            selected_valid,
            kv_len=pool_rows * 4,
            index_topk=2_048,
            index_kpool=4,
            tail_positions=mx.zeros((1, 0), dtype=mx.int64),
            tail_valid=mx.zeros((1, 0), dtype=mx.bool_),
            always_select_tail=True,
        )
        reference_valid = reference_valid & current_valid[..., None]
        reference = mx.where(reference_valid, reference, INDEXPOOL_SENTINEL)
        raw_positions = mx.zeros((1, 4), dtype=mx.int64)
        raw_valid = mx.zeros((1, 4), dtype=mx.bool_)
        dependencies = (
            scores,
            pool_indices,
            pool_valid,
            raw_positions,
            raw_valid,
            current_valid,
        )
        mx.async_eval(*dependencies)
        plan = plan_type("prefill", len(patterns), pool_rows, dtype_name)
        before = list(plan.buffer_identities)
        candidate, candidate_valid = plan.execute(
            scores,
            pool_indices,
            pool_valid,
            raw_positions,
            raw_valid,
            current_valid,
            pool_rows,
            pool_rows * 4,
            0,
        )
        _eval((reference, reference_valid, candidate, candidate_valid))
        reference_host = np.asarray(reference)
        candidate_host = np.asarray(candidate)
        reference_valid_host = np.asarray(reference_valid)
        candidate_valid_host = np.asarray(candidate_valid)
        results.append(
            {
                "score_dtype": dtype_name,
                "fixture_names": [name for name, _, _ in patterns],
                "fixture_exact": {
                    name: bool(
                        np.array_equal(reference_host[0, row], candidate_host[0, row])
                        and np.array_equal(
                            reference_valid_host[0, row],
                            candidate_valid_host[0, row],
                        )
                    )
                    for row, (name, _, _) in enumerate(patterns)
                },
                "indices_byte_exact": _exact(reference, candidate),
                "validity_byte_exact": _exact(reference_valid, candidate_valid),
                "reference_hash": _hash(reference),
                "candidate_hash": _hash(candidate),
                "buffer_identities_stable": before == list(plan.buffer_identities),
                "execution_count": plan.execution_count,
                "dynamic_allocation_count": plan.dynamic_allocation_count,
                "graph_node_count": plan.graph_node_count,
                "shape_discovery_count": plan.shape_discovery_count,
                "host_synchronization_count": plan.host_synchronization_count,
            }
        )
    return {
        "rows": len(patterns),
        "pool_rows": pool_rows,
        "dtypes": results,
        "all_exact": all(
            row["indices_byte_exact"]
            and row["validity_byte_exact"]
            and row["reference_hash"] == row["candidate_hash"]
            and row["buffer_identities_stable"]
            and row["execution_count"] == 1
            and row["dynamic_allocation_count"] == 0
            and row["graph_node_count"] == 0
            and row["shape_discovery_count"] == 0
            and row["host_synchronization_count"] == 0
            for row in results
        ),
    }


def _prepared_fixture(
    model,
    boundary_probe,
    profile,
    source,
    context: int,
    layer: int,
    rows: int,
    *,
    append_decode: bool,
):
    attention = model.language_model.model.layers[layer].self_attn
    entry = boundary_probe._clone_entry(source[layer], context + rows + 16)
    _, pool = entry
    x = boundary_probe._deterministic_rows(
        rows, attention.hidden_size, 11.0 + layer * 0.015625, mx.bfloat16
    )[None]
    qr = attention.q_a_layernorm(attention.q_a_proj(x))
    if append_decode:
        profile._pool_update(attention.indexer, pool, x)
    query = attention.indexer.wq_b(qr).reshape(
        1, rows, attention.indexer.n_heads, attention.indexer.head_dim
    )
    weights = attention.indexer.weights_proj(x) * (
        attention.indexer.n_heads**-0.5
    )
    pool_end = mx.clip(pool.pool_indices[..., -1], 0, pool.total_tokens - 1)
    valid_candidates = (
        (pool_end[:, None, :] < pool.total_tokens) & pool.pool_valid[:, None]
    )
    valid_candidates = mx.broadcast_to(
        valid_candidates, (1, rows, int(pool.pool_keys.shape[1]))
    )
    scores = _score_expression(
        query,
        pool.pool_keys,
        weights,
        valid_candidates,
        attention.indexer.softmax_scale,
    )
    current_valid = mx.ones((1, rows), dtype=mx.bool_)
    _eval(
        (
            scores,
            valid_candidates,
            current_valid,
            pool.pool_indices,
            pool.pool_valid,
            pool.raw_positions,
            pool.raw_valid,
        )
    )
    return {
        "layer": layer,
        "attention": attention,
        "entry": entry,
        "pool": pool,
        "scores": scores,
        "valid_candidates": valid_candidates,
        "current_valid": current_valid,
    }


def _exact_fixture(fixture, registry, mode: str) -> dict:
    pool = fixture["pool"]
    indexer = fixture["attention"].indexer
    reference, reference_valid = _reference_selection(
        indexer,
        pool,
        fixture["scores"],
        fixture["valid_candidates"],
        fixture["current_valid"],
    )
    _eval((reference, reference_valid))
    rows = int(fixture["scores"].shape[1])
    plan = registry.get(pool, rows, fixture["scores"].dtype, mode)
    before = list(plan.buffer_identities)
    candidate, candidate_valid = _native_selection(
        plan, pool, fixture["scores"], fixture["current_valid"]
    )
    _eval((candidate, candidate_valid))
    after = list(plan.buffer_identities)
    return {
        "layer": fixture["layer"],
        "mode": mode,
        "query_rows": rows,
        "logical_pool_rows": pool.logical_pool_count,
        "physical_pool_rows": int(pool.pool_keys.shape[1]),
        "indices_byte_exact": _exact(reference, candidate),
        "validity_byte_exact": _exact(reference_valid, candidate_valid),
        "reference_hash": _hash(reference),
        "candidate_hash": _hash(candidate),
        "buffer_identities_stable": before == after,
        "non_sentinel_out_of_range": int(
            np.count_nonzero(
                (np.asarray(candidate) != -1)
                & (
                    (np.asarray(candidate) < 0)
                    | (np.asarray(candidate) >= pool.total_tokens)
                )
            )
        ),
    }


def _median_rows(rows: list[dict]) -> dict:
    return {
        "median_wall_ms": statistics.median(row["wall_ms"] for row in rows),
        "median_host_submit_ms": statistics.median(
            row["host_submit_ms"] for row in rows
        ),
        "samples": rows,
    }


def _operator_timing(fixtures, registry, mode, warmups, samples):
    measured = {"A_mlx": [], "B_native": []}
    orders = (("A_mlx", "B_native"), ("B_native", "A_mlx"))
    for iteration in range(warmups + samples):
        for arm in orders[iteration % 2]:
            started = time.perf_counter_ns()
            outputs = []
            if arm == "A_mlx":
                for fixture in fixtures:
                    outputs.extend(
                        _reference_selection(
                            fixture["attention"].indexer,
                            fixture["pool"],
                            fixture["scores"],
                            fixture["valid_candidates"],
                            fixture["current_valid"],
                        )
                    )
            else:
                for fixture in fixtures:
                    rows = int(fixture["scores"].shape[1])
                    plan = registry.get(
                        fixture["pool"], rows, fixture["scores"].dtype, mode
                    )
                    outputs.extend(
                        _native_selection(
                            plan,
                            fixture["pool"],
                            fixture["scores"],
                            fixture["current_valid"],
                        )
                    )
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
    result = {arm: _median_rows(rows) for arm, rows in measured.items()}
    result["native_wall_saving_ms"] = (
        result["A_mlx"]["median_wall_ms"]
        - result["B_native"]["median_wall_ms"]
    )
    result["native_speedup"] = (
        result["A_mlx"]["median_wall_ms"]
        / result["B_native"]["median_wall_ms"]
    )
    result["interleaved_order"] = True
    return result


def _candidate_decode_selection(registry):
    def selected(self, indexer, x, qr, valid_cur):
        query = indexer.wq_b(qr).reshape(
            1, 1, indexer.n_heads, indexer.head_dim
        )
        weights = indexer.weights_proj(x) * (indexer.n_heads**-0.5)
        pool_end = mx.clip(
            self.pool_indices[..., -1], 0, self.total_tokens - 1
        )
        valid_candidates = (
            (pool_end[:, None, :] < self.total_tokens)
            & self.pool_valid[:, None]
        )
        scores = _score_expression(
            query,
            self.pool_keys,
            weights,
            valid_candidates,
            indexer.softmax_scale,
        )
        plan = registry.get(self, 1, scores.dtype, "decode")
        indices, _ = _native_selection(plan, self, scores, valid_cur)
        return indices[:, None]

    return selected


@contextlib.contextmanager
def _native_decode_arm(registry, enabled: bool):
    if not enabled:
        yield
        return
    original = CompactIndexPoolCache._decode_selection
    CompactIndexPoolCache._decode_selection = _candidate_decode_selection(registry)
    try:
        yield
    finally:
        CompactIndexPoolCache._decode_selection = original


def _full_model_case(model, boundary_probe, source, context, registry, warmups, samples):
    caches = {
        "A_mlx": boundary_probe._clone_cache(source, context + warmups + samples + 16),
        "B_native": boundary_probe._clone_cache(
            source, context + warmups + samples + 16
        ),
    }
    token = mx.array([[3000]], dtype=mx.uint32)
    measured = {arm: [] for arm in caches}
    hashes = {arm: [] for arm in caches}
    orders = (("A_mlx", "B_native"), ("B_native", "A_mlx"))
    for iteration in range(warmups + samples):
        for arm in orders[iteration % 2]:
            started = time.perf_counter_ns()
            with _native_decode_arm(registry, arm == "B_native"):
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
                hashes[arm].append(_hash(logits))
    result = {arm: _median_rows(rows) for arm, rows in measured.items()}
    result["native_wall_saving_ms"] = (
        result["A_mlx"]["median_wall_ms"]
        - result["B_native"]["median_wall_ms"]
    )
    result["native_speedup"] = (
        result["A_mlx"]["median_wall_ms"]
        / result["B_native"]["median_wall_ms"]
    )
    result["all_logits_byte_exact"] = hashes["A_mlx"] == hashes["B_native"]
    result["post_state_byte_exact"] = boundary_probe._cache_exact(
        caches["A_mlx"], caches["B_native"]
    )
    result["logits_hashes"] = hashes
    _release(caches)
    return result


def _acceptance(artifact):
    contexts = artifact.get("decode", {})
    short = contexts.get("2048", {})
    long = contexts.get("262144", {})
    plans = artifact.get("native_plan_evidence", [])
    return {
        "shared_prefill_decode_native_abi": artifact.get("native_abi")
        == NATIVE_EXECUTION_ENGINE_ABI,
        "native_binding_matches_runtime": artifact.get("mlx_version")
        == artifact.get("native_binding_abi", {}).get("mlx_version"),
        "artificial_order_tie_sentinel_contract_exact": artifact.get(
            "artificial_native_contract", {}
        ).get("all_exact", False),
        "prefill_and_decode_operator_exact": bool(
            artifact.get("prefill", {}).get("all_exact")
            and all(row.get("all_layers_exact") for row in contexts.values())
        ),
        "full_model_decode_logits_and_state_exact": bool(
            contexts
            and all(
                row.get("full_model", {}).get("all_logits_byte_exact")
                and row.get("full_model", {}).get("post_state_byte_exact")
                for row in contexts.values()
            )
        ),
        "native_buffers_stable": bool(plans)
        and all(row.get("buffer_identities_stable", False) for row in plans),
        "native_execute_dynamic_allocation_zero": bool(plans)
        and all(row.get("dynamic_allocation_count") == 0 for row in plans),
        "native_execute_graph_build_shape_discovery_sync_zero": bool(plans)
        and all(
            row.get("graph_node_count") == 0
            and row.get("shape_discovery_count") == 0
            and row.get("host_synchronization_count") == 0
            for row in plans
        ),
        "prefill_island_speedup_at_least_1_20x": artifact.get("prefill", {})
        .get("operator_timing", {})
        .get("native_speedup", 0.0)
        >= PREFILL_ISLAND_MIN_SPEEDUP,
        "decode_256k_bridge_saves_at_least_0_75ms": long.get(
            "full_model", {}
        ).get("native_wall_saving_ms", -1e9)
        >= BRIDGE_MIN_256K_WALL_SAVING_MS,
        "decode_2k_regression_at_most_1_percent": short.get(
            "full_model", {}
        ).get("B_native", {}).get("median_wall_ms", 1e9)
        <= short.get("full_model", {}).get("A_mlx", {}).get(
            "median_wall_ms", 0.0
        )
        * (1.0 + BRIDGE_MAX_2K_REGRESSION),
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
    artifact = {
        "schema": "glm53-native-execution-engine-feasibility-v1",
        "date": date.today().isoformat(),
        "complete": False,
        "accepted": False,
        "probe_only": True,
        "native_abi": NATIVE_EXECUTION_ENGINE_ABI,
        "mlx_version": importlib.metadata.version("mlx"),
        "native_binding_abi": {
            "mlx_version": "0.32.2",
            "nanobind_version": "2.15.0",
            "domain": "mlx",
        },
        "tier": "tier0-indexer-selection-expansion-submission-bridge",
        "scope_limits": {
            "score_producer_native": False,
            "kda_dsa_native": False,
            "moe_native": False,
            "full_32k_prefill_gate_evaluated": False,
            "final_256k_decode_3ms_gate_evaluated": False,
        },
        "future_engine_gates": {
            "minimum_256k_decode_saving_ms": FINAL_NATIVE_DECODE_SAVING_MS,
            "minimum_32k_prefill_speedup": FINAL_NATIVE_PREFILL_SPEEDUP,
        },
        "runtime_changes": {
            "runtime": False,
            "server": False,
            "apc": False,
            "cache_abi": False,
            "kernel_abi": False,
        },
    }
    try:
        if not mx.metal.is_available():
            raise RuntimeError("native execution feasibility requires Metal")
        NativePlan, oracle_probe, boundary_probe, profile = _load_helpers()
        _progress("artificial_native_contract")
        artifact["artificial_native_contract"] = _artificial_native_contract(
            NativePlan
        )
        _atomic_write(args.output, artifact)
        report = inspect_checkpoint(args.model, require_server_ready=True)
        artifact.update(
            {
                "checkpoint_fingerprint": report.fingerprint,
                "official_hf_revision": report.official_revision,
                "mlx_vlm_revision": MLX_VLM_REVISION,
                "compact_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
                "plan_contracts": {
                    "decode": plan_native_indexer_island(
                        NativeExecutionMode.DECODE,
                        query_rows=1,
                        logical_capacity_tokens=262_145,
                    ).descriptor(),
                    "prefill": plan_native_indexer_island(
                        NativeExecutionMode.PREFILL,
                        query_rows=PREFILL_ROWS,
                        logical_capacity_tokens=PREFILL_CONTEXT,
                    ).descriptor(),
                },
            }
        )
        mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
        mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
        _progress("load_model")
        model, processor = load(
            args.model,
            experimental_packed_decode_moe=True,
            experimental_compact_nope_dsa_cache=True,
            compact_cache_capacity_tokens=max(DECODE_CONTEXTS) + 32,
        )
        warm_residency(model)
        artifact["official_oracle"] = oracle_probe._official_oracle(
            model, processor, report
        )

        _progress("prefill_operator", context=PREFILL_CONTEXT, rows=PREFILL_ROWS)
        prefill_source = boundary_probe._synthetic_cache(
            model, PREFILL_CONTEXT, "compact-nope-dsa"
        )
        prefill_registry = _NativePlanRegistry(NativePlan)
        prefill_layers = (EXPECTED_DSA[0], EXPECTED_DSA[len(EXPECTED_DSA) // 2], EXPECTED_DSA[-1])
        prefill_fixtures = [
            _prepared_fixture(
                model,
                boundary_probe,
                profile,
                prefill_source,
                PREFILL_CONTEXT,
                layer,
                PREFILL_ROWS,
                append_decode=False,
            )
            for layer in prefill_layers
        ]
        prefill_exact = [
            _exact_fixture(fixture, prefill_registry, "prefill")
            for fixture in prefill_fixtures
        ]
        artifact["prefill"] = {
            "context_tokens": PREFILL_CONTEXT,
            "query_rows": PREFILL_ROWS,
            "layers": prefill_exact,
            "all_exact": all(
                row["indices_byte_exact"]
                and row["validity_byte_exact"]
                and row["buffer_identities_stable"]
                and row["non_sentinel_out_of_range"] == 0
                for row in prefill_exact
            ),
            "operator_timing": _operator_timing(
                prefill_fixtures,
                prefill_registry,
                "prefill",
                args.warmups,
                args.samples,
            ),
        }
        _release(prefill_fixtures, prefill_source)
        artifact["prefill_complete"] = True
        _atomic_write(args.output, artifact)

        artifact["decode"] = {}
        all_plan_evidence = prefill_registry.evidence()
        for context in DECODE_CONTEXTS:
            _progress("decode_context", context=context)
            source = boundary_probe._synthetic_cache(
                model, context, "compact-nope-dsa"
            )
            registry = _NativePlanRegistry(NativePlan)
            fixtures = [
                _prepared_fixture(
                    model,
                    boundary_probe,
                    profile,
                    source,
                    context,
                    layer,
                    1,
                    append_decode=True,
                )
                for layer in EXPECTED_DSA
            ]
            exact = [
                _exact_fixture(fixture, registry, "decode")
                for fixture in fixtures
            ]
            row = {
                "context_tokens": context,
                "layers": exact,
                "all_layers_exact": all(
                    item["indices_byte_exact"]
                    and item["validity_byte_exact"]
                    and item["buffer_identities_stable"]
                    and item["non_sentinel_out_of_range"] == 0
                    for item in exact
                ),
                "operator_timing": _operator_timing(
                    fixtures,
                    registry,
                    "decode",
                    args.warmups,
                    args.samples,
                ),
                "full_model": _full_model_case(
                    model,
                    boundary_probe,
                    source,
                    context,
                    registry,
                    args.warmups,
                    args.samples,
                ),
            }
            artifact["decode"][str(context)] = row
            artifact["last_completed_context"] = context
            all_plan_evidence.extend(registry.evidence())
            artifact["native_plan_evidence"] = all_plan_evidence
            artifact["acceptance"] = _acceptance(artifact)
            artifact["peak_memory_bytes"] = int(mx.get_peak_memory())
            _atomic_write(args.output, artifact)
            _release(fixtures, source)

        artifact["native_plan_evidence"] = all_plan_evidence
        artifact["acceptance"] = _acceptance(artifact)
        artifact["complete"] = True
        artifact["accepted"] = all(artifact["acceptance"].values())
        artifact["decision"] = (
            "advance_native_boundary_to_score_kda_dsa"
            if artifact["accepted"]
            else "stop_or_redesign_native_submission_bridge"
        )
    except Exception as error:
        artifact["complete"] = True
        artifact["accepted"] = False
        artifact["decision"] = "native_feasibility_failed"
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
