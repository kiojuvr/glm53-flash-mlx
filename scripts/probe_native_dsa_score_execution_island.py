#!/usr/bin/env python3
"""Probe the Tier-1 native pooled-score -> selected-token execution island.

The accepted Tier-0 plan starts from a materialized Indexer score tensor.  This
probe moves BF16 pooled scoring into the same persistent C++/Metal execution
plan as exact top-k and token expansion.  Query/weight projections, cache
updates, sparse gather, and attention remain MLX-owned and are stated as such.

This is deliberately an exactness-first feasibility gate.  A single score,
selection, logits, or state bit difference rejects the widening before any
production ABI is changed.
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
    NATIVE_DSA_SCORE_ISLAND_ABI,
    NATIVE_EXECUTION_ENGINE_ABI,
    NativeExecutionMode,
    plan_native_dsa_score_island,
)
from glm53_flash_mlx.nope_cache import CompactIndexPoolCache


ROOT = Path(__file__).resolve().parents[1]
NATIVE_PACKAGE = ROOT / "native_execution"
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-dsa-score-execution-island-v2-20260907.json"
)
DECODE_CONTEXTS = (2_048, 262_144)
PREFILL_CONTEXT = 32_768
PREFILL_ROWS = 256
MIN_256K_FULL_MODEL_SAVING_MS = 3.0
MIN_32K_PREFILL_SPEEDUP = 1.20
MAX_2K_REGRESSION = 0.01
MAX_WORKING_PEAK_DELTA = 512 << 20


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
    from glm53_native_execution import NativeDSAScoreSelectionPlan
    import probe_exact_sigmoid_gate_metal_barrier as oracle_probe
    import probe_long_context_first_decode_boundary as boundary_probe
    import probe_native_execution_engine_feasibility as tier0
    import profile_lightning_indexer_decode_critical_path as profile

    return NativeDSAScoreSelectionPlan, oracle_probe, boundary_probe, tier0, profile


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

    def get(self, pool, indexer, rows: int, mode: str):
        key = (id(pool), id(indexer), rows, int(pool.pool_keys.shape[1]), mode)
        if key not in self.plans:
            plan = self.plan_type(
                mode,
                rows,
                int(pool.pool_keys.shape[1]),
                float(indexer.softmax_scale),
            )
            self.plans[key] = plan
            self.initial_identities[key] = list(plan.buffer_identities)
        return self.plans[key]

    def evidence(self) -> list[dict]:
        result = []
        for key, plan in self.plans.items():
            identities = list(plan.buffer_identities)
            result.append(
                {
                    "key": list(map(str, key)),
                    "mode": plan.mode,
                    "query_rows": plan.query_rows,
                    "physical_pool_rows": plan.physical_pool_rows,
                    "execution_count": plan.execution_count,
                    "scratch_bytes": plan.scratch_bytes,
                    "returned_score_tensor_bytes": plan.returned_score_tensor_bytes,
                    "dynamic_allocation_count": plan.dynamic_allocation_count,
                    "graph_node_count": plan.graph_node_count,
                    "shape_discovery_count": plan.shape_discovery_count,
                    "host_synchronization_count": plan.host_synchronization_count,
                    "buffer_identities_stable": identities
                    == self.initial_identities[key],
                }
            )
        return result


def _native_execute(plan, pool, query, weights, current_valid):
    dependencies = (
        query,
        weights,
        pool.pool_keys,
        pool.pool_indices,
        pool.pool_valid,
        pool.raw_positions,
        pool.raw_valid,
        current_valid,
    )
    mx.async_eval(*dependencies)
    indices, valid = plan.execute(
        query,
        weights,
        pool.pool_keys,
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


def _artificial_contract(plan_type, tier0) -> dict:
    rows, pools, logical = 4, 1_024, 1_000
    query_values = np.zeros((1, rows, 32, 128), dtype=np.float32)
    key_values = np.zeros((1, pools, 128), dtype=np.float32)
    weight_values = np.zeros((1, rows, 32), dtype=np.float32)
    for row in range(1, rows):
        query_values[0, row] = np.sin(
            np.arange(32 * 128, dtype=np.float32).reshape(32, 128)
            * np.float32(0.001 * (row + 1))
        )
        weight_values[0, row] = np.cos(
            np.arange(32, dtype=np.float32) * np.float32(0.03 * row)
        )
    key_values[0] = np.cos(
        np.arange(pools * 128, dtype=np.float32).reshape(pools, 128)
        * np.float32(0.0001)
    )
    query = mx.array(query_values, dtype=mx.bfloat16)
    keys = mx.array(key_values, dtype=mx.bfloat16)
    weights = mx.array(weight_values, dtype=mx.bfloat16)
    pool_valid = mx.arange(pools)[None] < logical
    pool_indices = mx.arange(pools * 4, dtype=mx.int64).reshape(1, pools, 4)
    raw_positions = mx.zeros((1, 4), dtype=mx.int64)
    raw_valid = mx.zeros((1, 4), dtype=mx.bool_)
    current_valid = mx.array([True, True, True, False], dtype=mx.bool_)
    valid_candidates = mx.broadcast_to(pool_valid[:, None], (1, rows, pools))
    scale = 128**-0.5
    reference_scores = tier0._score_expression(
        query, keys, weights, valid_candidates, scale
    )
    selected = mx.argsort(-reference_scores, axis=-1)[..., :512]
    selected_valid = mx.take_along_axis(valid_candidates, selected, axis=-1)

    class Pool:
        pass

    pool = Pool()
    pool.pool_keys = keys
    pool.pool_indices = pool_indices
    pool.pool_valid = pool_valid
    pool.raw_positions = raw_positions
    pool.raw_valid = raw_valid
    pool.logical_pool_count = logical
    pool.total_tokens = logical * 4
    pool.active_tail_count = 0
    reference, reference_valid = tier0.expand_selected_pools(
        selected,
        pool_indices,
        selected_valid,
        kv_len=pool.total_tokens,
        index_topk=2_048,
        index_kpool=4,
        tail_positions=mx.zeros((1, 0), dtype=mx.int64),
        tail_valid=mx.zeros((1, 0), dtype=mx.bool_),
        always_select_tail=True,
    )
    reference_valid = reference_valid & current_valid[..., None]
    reference = mx.where(reference_valid, reference, tier0.INDEXPOOL_SENTINEL)
    plan = plan_type("prefill", rows, pools, float(scale))
    before = list(plan.buffer_identities)
    candidate, candidate_valid = _native_execute(
        plan, pool, query, weights, current_valid
    )
    reference_head_scores = (
        query @ keys[:, None].swapaxes(-1, -2)
    ).reshape(rows * 32, pools)
    candidate_head_scores = plan.debug_head_scores
    candidate_scores = plan.debug_index_scores
    _eval(
        (
            reference_scores,
            reference_head_scores,
            reference,
            reference_valid,
            candidate_scores,
            candidate_head_scores,
            candidate,
            candidate_valid,
        )
    )
    reference_score_host = np.asarray(reference_scores.astype(mx.float32))
    candidate_score_host = np.asarray(candidate_scores.astype(mx.float32))
    reference_head_host = np.asarray(reference_head_scores.astype(mx.float32))
    candidate_head_host = np.asarray(candidate_head_scores.astype(mx.float32))
    differing = np.argwhere(reference_score_host != candidate_score_host)
    first_score_difference = None
    if differing.size:
        coordinate = tuple(int(value) for value in differing[0])
        first_score_difference = {
            "coordinate": list(coordinate),
            "reference": float(reference_score_host[coordinate]),
            "candidate": float(candidate_score_host[coordinate]),
            "absolute_error": float(
                abs(reference_score_host[coordinate] - candidate_score_host[coordinate])
            ),
        }
    score_exact = _exact(reference_scores, candidate_scores)
    return {
        "steel_head_scores_byte_exact": _exact(
            reference_head_scores, candidate_head_scores
        ),
        "steel_head_scores_differing_elements": int(
            np.count_nonzero(reference_head_host != candidate_head_host)
        ),
        "steel_head_scores_max_absolute_error": float(
            np.max(np.abs(reference_head_host - candidate_head_host))
        ),
        "score_byte_exact": score_exact,
        "score_differing_elements": int(differing.shape[0]),
        "score_total_elements": int(reference_score_host.size),
        "score_max_absolute_error": float(
            np.max(np.abs(reference_score_host - candidate_score_host))
        ),
        "first_score_difference": first_score_difference,
        "selected_token_indices_byte_exact": _exact(reference, candidate),
        "selected_token_validity_byte_exact": _exact(
            reference_valid, candidate_valid
        ),
        "all_equal_score_row_included": True,
        "invalid_query_row_included": True,
        "padding_rows_included": pools - logical,
        "buffer_identities_stable": before == list(plan.buffer_identities),
        "returned_score_tensor_bytes": plan.returned_score_tensor_bytes,
        "dynamic_allocation_count": plan.dynamic_allocation_count,
        "graph_node_count": plan.graph_node_count,
        "shape_discovery_count": plan.shape_discovery_count,
        "host_synchronization_count": plan.host_synchronization_count,
        "all_exact": score_exact
        and _exact(reference, candidate)
        and _exact(reference_valid, candidate_valid),
    }


def _artificial_prefill_geometry(plan_type, tier0) -> dict:
    """Exercise the actual Q256/P8256 memory geometry before model loading."""

    rows, pools, logical = PREFILL_ROWS, 8_256, 8_192
    query_axis = np.arange(rows * 32 * 128, dtype=np.float32).reshape(
        1, rows, 32, 128
    )
    key_axis = np.arange(pools * 128, dtype=np.float32).reshape(1, pools, 128)
    weight_axis = np.arange(rows * 32, dtype=np.float32).reshape(1, rows, 32)
    query = mx.array(np.sin(query_axis * np.float32(0.0007)), dtype=mx.bfloat16)
    pool_keys = mx.array(
        np.cos(key_axis * np.float32(0.00011)), dtype=mx.bfloat16
    )
    weights = mx.array(
        np.cos(weight_axis * np.float32(0.013)), dtype=mx.bfloat16
    )
    pool_valid = mx.arange(pools)[None] < logical
    pool_indices = mx.arange(pools * 4, dtype=mx.int64).reshape(1, pools, 4)
    current_valid = mx.ones((1, rows), dtype=mx.bool_)
    valid_candidates = mx.broadcast_to(pool_valid[:, None], (1, rows, pools))
    scale = 128**-0.5
    scores = tier0._score_expression(
        query, pool_keys, weights, valid_candidates, scale
    )
    pool = SimpleNamespace(
        pool_keys=pool_keys,
        pool_indices=pool_indices,
        pool_valid=pool_valid,
        raw_positions=mx.zeros((1, 4), dtype=mx.int64),
        raw_valid=mx.zeros((1, 4), dtype=mx.bool_),
        logical_pool_count=logical,
        total_tokens=logical * 4,
        active_tail_count=0,
    )
    indexer = SimpleNamespace(
        softmax_scale=scale,
        index_topk=2_048,
        index_kpool=4,
        index_kpool_always_select_tail=True,
    )
    fixture = {
        "layer": "artificial-q256",
        "attention": SimpleNamespace(indexer=indexer),
        "pool": pool,
        "query": query,
        "weights": weights,
        "scores": scores,
        "valid_candidates": valid_candidates,
        "current_valid": current_valid,
    }
    _eval(fixture)
    registry = _Registry(plan_type)
    exact = _exact_fixture(fixture, registry, tier0, "prefill")
    timing = _operator_timing(
        [fixture], registry, tier0, "prefill", warmups=1, samples=3
    )
    evidence = registry.evidence()
    result = {
        "query_rows": rows,
        "physical_pool_rows": pools,
        "logical_pool_rows": logical,
        "exact": exact,
        "operator_timing": timing,
        "plan_evidence": evidence,
        "all_exact": exact["score_byte_exact"]
        and exact["indices_byte_exact"]
        and exact["validity_byte_exact"],
    }
    _release(fixture)
    return result


def _exact_fixture(fixture, registry, tier0, mode: str) -> dict:
    pool = fixture["pool"]
    indexer = fixture["attention"].indexer
    reference, reference_valid = tier0._reference_selection(
        indexer,
        pool,
        fixture["scores"],
        fixture["valid_candidates"],
        fixture["current_valid"],
    )
    plan = registry.get(pool, indexer, int(fixture["scores"].shape[1]), mode)
    before = list(plan.buffer_identities)
    candidate, candidate_valid = _native_execute(
        plan,
        pool,
        fixture["query"],
        fixture["weights"],
        fixture["current_valid"],
    )
    candidate_scores = plan.debug_index_scores
    _eval(
        (
            reference,
            reference_valid,
            fixture["scores"],
            candidate_scores,
            candidate,
            candidate_valid,
        )
    )
    host = np.asarray(candidate)
    return {
        "layer": fixture["layer"],
        "score_byte_exact": _exact(fixture["scores"], candidate_scores),
        "indices_byte_exact": _exact(reference, candidate),
        "validity_byte_exact": _exact(reference_valid, candidate_valid),
        "buffer_identities_stable": before == list(plan.buffer_identities),
        "non_sentinel_out_of_range": int(
            np.count_nonzero(
                (host != -1) & ((host < 0) | (host >= pool.total_tokens))
            )
        ),
    }


def _operator_timing(fixtures, registry, tier0, mode, warmups, samples):
    measured = {"A_mlx": [], "C_native_score_selection": []}
    orders = (
        ("A_mlx", "C_native_score_selection"),
        ("C_native_score_selection", "A_mlx"),
    )
    for iteration in range(warmups + samples):
        for arm in orders[iteration % 2]:
            started = time.perf_counter_ns()
            outputs = []
            for fixture in fixtures:
                if arm == "A_mlx":
                    scores = tier0._score_expression(
                        fixture["query"],
                        fixture["pool"].pool_keys,
                        fixture["weights"],
                        fixture["valid_candidates"],
                        fixture["attention"].indexer.softmax_scale,
                    )
                    outputs.extend(
                        tier0._reference_selection(
                            fixture["attention"].indexer,
                            fixture["pool"],
                            scores,
                            fixture["valid_candidates"],
                            fixture["current_valid"],
                        )
                    )
                else:
                    plan = registry.get(
                        fixture["pool"],
                        fixture["attention"].indexer,
                        int(fixture["scores"].shape[1]),
                        mode,
                    )
                    outputs.extend(
                        _native_execute(
                            plan,
                            fixture["pool"],
                            fixture["query"],
                            fixture["weights"],
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
    result = {arm: _median(rows) for arm, rows in measured.items()}
    result["native_wall_saving_ms"] = (
        result["A_mlx"]["median_wall_ms"]
        - result["C_native_score_selection"]["median_wall_ms"]
    )
    result["native_speedup"] = (
        result["A_mlx"]["median_wall_ms"]
        / result["C_native_score_selection"]["median_wall_ms"]
    )
    return result


def _candidate_decode_selection(registry):
    def selected(self, indexer, x, qr, valid_cur):
        query = indexer.wq_b(qr).reshape(
            1, 1, indexer.n_heads, indexer.head_dim
        )
        weights = indexer.weights_proj(x) * (indexer.n_heads**-0.5)
        plan = registry.get(self, indexer, 1, "decode")
        indices, _ = _native_execute(plan, self, query, weights, valid_cur)
        return indices[:, None]

    return selected


@contextlib.contextmanager
def _native_arm(registry, enabled: bool):
    if not enabled:
        yield
        return
    original = CompactIndexPoolCache._decode_selection
    CompactIndexPoolCache._decode_selection = _candidate_decode_selection(registry)
    try:
        yield
    finally:
        CompactIndexPoolCache._decode_selection = original


def _full_model_case(
    model, boundary_probe, tier0, source, context, registry, warmups, samples
):
    caches = {
        "A_mlx": boundary_probe._clone_cache(
            source, context + warmups + samples + 16
        ),
        "C_native_score_selection": boundary_probe._clone_cache(
            source, context + warmups + samples + 16
        ),
    }
    token = mx.array([[3000]], dtype=mx.uint32)
    measured = {arm: [] for arm in caches}
    logits_by_arm = {arm: [] for arm in caches}
    orders = (
        ("A_mlx", "C_native_score_selection"),
        ("C_native_score_selection", "A_mlx"),
    )
    for iteration in range(warmups + samples):
        for arm in orders[iteration % 2]:
            started = time.perf_counter_ns()
            with _native_arm(registry, arm == "C_native_score_selection"):
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
                logits_by_arm[arm].append(tier0._hash(logits))
    result = {arm: _median(rows) for arm, rows in measured.items()}
    result["native_wall_saving_ms"] = (
        result["A_mlx"]["median_wall_ms"]
        - result["C_native_score_selection"]["median_wall_ms"]
    )
    result["all_logits_byte_exact"] = (
        logits_by_arm["A_mlx"] == logits_by_arm["C_native_score_selection"]
    )
    result["post_state_byte_exact"] = boundary_probe._cache_exact(
        caches["A_mlx"], caches["C_native_score_selection"]
    )
    _release(caches)
    return result


def _acceptance(artifact) -> dict[str, bool]:
    prefill = artifact.get("prefill", {})
    decode = artifact.get("decode", {})
    short = decode.get("2048", {})
    long = decode.get("262144", {})
    evidence = artifact.get("native_plan_evidence", [])
    correctness = bool(prefill.get("all_exact")) and bool(decode) and all(
        row.get("all_layers_exact")
        and row.get("full_model", {}).get("all_logits_byte_exact")
        and row.get("full_model", {}).get("post_state_byte_exact")
        for row in decode.values()
    )
    return {
        "artificial_score_selection_exact": artifact.get(
            "artificial_native_contract", {}
        ).get("all_exact", False),
        "prefill_decode_score_selection_logits_state_byte_exact": correctness,
        "fixed_arena_and_zero_execute_allocation_graph_sync": bool(evidence)
        and all(
            row.get("buffer_identities_stable")
            and row.get("dynamic_allocation_count") == 0
            and row.get("graph_node_count") == 0
            and row.get("shape_discovery_count") == 0
            and row.get("host_synchronization_count") == 0
            and row.get("returned_score_tensor_bytes") == 0
            for row in evidence
        ),
        "32k_q256_operator_speedup_at_least_1_20x": prefill.get(
            "operator_timing", {}
        ).get("native_speedup", 0.0)
        >= MIN_32K_PREFILL_SPEEDUP,
        "256k_full_model_saving_at_least_3ms": long.get("full_model", {}).get(
            "native_wall_saving_ms", -1e9
        )
        >= MIN_256K_FULL_MODEL_SAVING_MS,
        "2k_full_model_regression_at_most_1_percent": short.get(
            "full_model", {}
        ).get("C_native_score_selection", {}).get("median_wall_ms", 1e9)
        <= short.get("full_model", {}).get("A_mlx", {}).get(
            "median_wall_ms", 0.0
        )
        * (1.0 + MAX_2K_REGRESSION),
        "native_plan_scratch_at_most_512MiB": artifact.get(
            "maximum_concurrent_native_plan_scratch_bytes", 1 << 60
        )
        <= MAX_WORKING_PEAK_DELTA,
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
    args = parser.parse_args()
    artifact = {
        "schema": "glm53-native-dsa-score-execution-island-v2",
        "date": date.today().isoformat(),
        "complete": False,
        "accepted": False,
        "probe_only": True,
        "engine_abi": NATIVE_EXECUTION_ENGINE_ABI,
        "island_abi": NATIVE_DSA_SCORE_ISLAND_ABI,
        "tier": "tier1-v2-pool32-tiled-score-exact-topk-token-expansion",
        "mlx_version": importlib.metadata.version("mlx"),
        "scope_limits": {
            "query_projection_native": False,
            "pool_update_native": False,
            "pooled_score_native": True,
            "topk_expansion_native": True,
            "sparse_gather_attention_native": False,
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
    }
    baseline_peak = int(mx.get_peak_memory())
    try:
        if not mx.metal.is_available():
            raise RuntimeError("native DSA score island requires Metal")
        Plan, oracle_probe, boundary_probe, tier0, profile = _load_helpers()
        _progress("artificial_native_contract")
        artifact["artificial_native_contract"] = _artificial_contract(Plan, tier0)
        _atomic_write(args.output, artifact)
        if not artifact["artificial_native_contract"]["all_exact"]:
            artifact["complete"] = True
            artifact["decision"] = "reject_native_score_numerical_order"
            _atomic_write(args.output, artifact)
            print(json.dumps({"output": str(args.output), **artifact}, indent=2))
            return 1

        _progress("artificial_prefill_geometry", rows=PREFILL_ROWS, pools=8_256)
        artifact["artificial_prefill_geometry"] = _artificial_prefill_geometry(
            Plan, tier0
        )
        _atomic_write(args.output, artifact)
        artificial_prefill = artifact["artificial_prefill_geometry"]
        if (
            not artificial_prefill["all_exact"]
            or artificial_prefill["operator_timing"]["native_speedup"]
            < MIN_32K_PREFILL_SPEEDUP
        ):
            artifact["complete"] = True
            artifact["decision"] = "reject_native_score_prefill_geometry_screen"
            _atomic_write(args.output, artifact)
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "complete": True,
                        "accepted": False,
                        "decision": artifact["decision"],
                        "speedup": artificial_prefill["operator_timing"][
                            "native_speedup"
                        ],
                    },
                    indent=2,
                )
            )
            return 1

        report = inspect_checkpoint(args.model, require_server_ready=True)
        artifact.update(
            {
                "checkpoint_fingerprint": report.fingerprint,
                "official_hf_revision": report.official_revision,
                "mlx_vlm_revision": MLX_VLM_REVISION,
                "compact_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
                "plan_contracts": {
                    "decode": plan_native_dsa_score_island(
                        NativeExecutionMode.DECODE,
                        query_rows=1,
                        logical_capacity_tokens=262_145,
                    ).descriptor(),
                    "prefill": plan_native_dsa_score_island(
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
        baseline_peak = int(mx.get_peak_memory())

        _progress("prefill_operator", context=PREFILL_CONTEXT, rows=PREFILL_ROWS)
        source = boundary_probe._synthetic_cache(
            model, PREFILL_CONTEXT, "compact-nope-dsa"
        )
        registry = _Registry(Plan)
        layers = (EXPECTED_DSA[0], EXPECTED_DSA[len(EXPECTED_DSA) // 2], EXPECTED_DSA[-1])
        fixtures = [
            tier0._prepared_fixture(
                model,
                boundary_probe,
                profile,
                source,
                PREFILL_CONTEXT,
                layer,
                PREFILL_ROWS,
                append_decode=False,
            )
            for layer in layers
        ]
        exact = [_exact_fixture(row, registry, tier0, "prefill") for row in fixtures]
        artifact["prefill"] = {
            "context_tokens": PREFILL_CONTEXT,
            "query_rows": PREFILL_ROWS,
            "layers": exact,
            "all_exact": all(
                row["score_byte_exact"]
                and row["indices_byte_exact"]
                and row["validity_byte_exact"]
                and row["buffer_identities_stable"]
                and row["non_sentinel_out_of_range"] == 0
                for row in exact
            ),
            "operator_timing": _operator_timing(
                fixtures, registry, tier0, "prefill", args.warmups, args.samples
            ),
        }
        artifact["prefill"]["concurrent_native_plan_scratch_bytes"] = sum(
            row["scratch_bytes"] for row in registry.evidence()
        )
        evidence = registry.evidence()
        _release(fixtures, source)
        _atomic_write(args.output, artifact)
        if not artifact["prefill"]["all_exact"]:
            artifact["complete"] = True
            artifact["decision"] = "reject_native_score_prefill_exactness"
            artifact["native_plan_evidence"] = evidence
            _atomic_write(args.output, artifact)
            return 1

        artifact["decode"] = {}
        for context in DECODE_CONTEXTS:
            _progress("decode_context", context=context)
            source = boundary_probe._synthetic_cache(
                model, context, "compact-nope-dsa"
            )
            registry = _Registry(Plan)
            fixtures = [
                tier0._prepared_fixture(
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
            exact = [_exact_fixture(row, registry, tier0, "decode") for row in fixtures]
            all_exact = all(
                row["score_byte_exact"]
                and row["indices_byte_exact"]
                and row["validity_byte_exact"]
                and row["buffer_identities_stable"]
                and row["non_sentinel_out_of_range"] == 0
                for row in exact
            )
            row = {
                "context_tokens": context,
                "layers": exact,
                "all_layers_exact": all_exact,
                "operator_timing": _operator_timing(
                    fixtures,
                    registry,
                    tier0,
                    "decode",
                    args.warmups,
                    args.samples,
                ),
            }
            if all_exact:
                row["full_model"] = _full_model_case(
                    model,
                    boundary_probe,
                    tier0,
                    source,
                    context,
                    registry,
                    args.warmups,
                    args.samples,
                )
            else:
                row["full_model"] = {"skipped": "operator exactness failed"}
            artifact["decode"][str(context)] = row
            row["concurrent_native_plan_scratch_bytes"] = sum(
                item["scratch_bytes"] for item in registry.evidence()
            )
            evidence.extend(registry.evidence())
            artifact["native_plan_evidence"] = evidence
            _atomic_write(args.output, artifact)
            _release(fixtures, source)
            if not all_exact:
                break

        artifact["process_peak_memory_bytes"] = int(mx.get_peak_memory())
        artifact["process_peak_delta_from_warm_model_bytes"] = max(
            0, artifact["process_peak_memory_bytes"] - baseline_peak
        )
        artifact["maximum_concurrent_native_plan_scratch_bytes"] = max(
            [artifact["prefill"]["concurrent_native_plan_scratch_bytes"]]
            + [
                row.get("concurrent_native_plan_scratch_bytes", 0)
                for row in artifact["decode"].values()
            ]
        )
        artifact["acceptance"] = _acceptance(artifact)
        artifact["complete"] = True
        artifact["accepted"] = all(artifact["acceptance"].values())
        artifact["decision"] = (
            "advance_native_boundary_to_sparse_gather_attention"
            if artifact["accepted"]
            else "stop_or_redesign_native_dsa_score_island"
        )
    except Exception as error:
        artifact["complete"] = True
        artifact["accepted"] = False
        artifact["decision"] = "native_dsa_score_feasibility_failed"
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
