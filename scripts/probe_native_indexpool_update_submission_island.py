#!/usr/bin/env python3
"""Probe native compact IndexPool update -> Tier-1 score/selection.

This decode-only, probe-only island receives already projected BF16 key/gate,
query, and mixture weights.  It advances the fixed raw19 rollback state,
recomputes and publishes exactly one kpool=4 row, then invokes the accepted
native pooled-score/top-k/token-expansion plan without returning the update
intermediates to MLX.
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
from glm53_flash_mlx.manifest import inspect_checkpoint
from glm53_flash_mlx.nope_cache import CompactIndexPoolCache


ROOT = Path(__file__).resolve().parents[1]
NATIVE_PACKAGE = ROOT / "native_execution"
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-indexpool-update-submission-island-20260907.json"
)
CONTEXTS = (2_048, 262_144)
MIN_256K_SAVING_MS = 0.75
MAX_2K_REGRESSION = 0.01
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
    from glm53_native_execution import NativeIndexPoolUpdateSelectionPlan
    import probe_exact_sigmoid_gate_metal_barrier as oracle_probe
    import probe_long_context_first_decode_boundary as boundary
    import probe_native_dsa_score_execution_island as tier1
    import probe_native_execution_engine_feasibility as tier0

    return NativeIndexPoolUpdateSelectionPlan, oracle_probe, boundary, tier1, tier0


def _arrays(value):
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _arrays(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _arrays(child)


def _eval(value) -> None:
    arrays = list(_arrays(value))
    if arrays:
        mx.eval(*arrays)
    mx.synchronize()


def _exact(left: mx.array, right: mx.array) -> bool:
    mx.eval(left, right)
    return bool(mx.array_equal(left, right).item())


def _owned(array: mx.array) -> mx.array:
    result = mx.array(array)
    mx.eval(result)
    return result


def _make_pool(indexer, total: int, physical_rows: int, seed: int):
    rng = np.random.default_rng(seed)
    pool = CompactIndexPoolCache(indexer, capacity_tokens=physical_rows * 4)
    logical = (total + 3) // 4
    pool.pool_keys = mx.array(
        rng.normal(size=(1, physical_rows, 128)).astype(np.float32),
        dtype=mx.bfloat16,
    )
    pool.pool_indices = mx.full(
        (1, physical_rows, 4), -1, dtype=mx.int64
    )
    complete = total // 4
    if complete:
        pool.pool_indices[:, :complete] = mx.arange(
            complete * 4, dtype=mx.int64
        ).reshape(1, complete, 4)
    pool.pool_valid = mx.zeros((1, physical_rows), dtype=mx.bool_)
    if complete:
        pool.pool_valid[:, :complete] = True
    start = total - 19
    pool.raw_keys = mx.array(
        rng.normal(size=(1, 19, 128)).astype(np.float32), dtype=mx.bfloat16
    )
    pool.raw_gates = mx.array(
        rng.normal(size=(1, 19, 128)).astype(np.float32), dtype=mx.bfloat16
    )
    pool.raw_valid = mx.ones((1, 19), dtype=mx.bool_)
    pool.raw_positions = mx.arange(start, total, dtype=mx.int64)[None]
    pool.total_tokens = total
    pool.logical_pool_count = logical
    pool.pool_capacity = physical_rows
    _eval(pool.state)
    return pool


def _clone_pool(source, indexer):
    pool = CompactIndexPoolCache(indexer, capacity_tokens=source.capacity_tokens)
    for name in (
        "pool_keys",
        "pool_indices",
        "pool_valid",
        "raw_keys",
        "raw_gates",
        "raw_valid",
        "raw_positions",
        "compress_ape",
    ):
        setattr(pool, name, _owned(getattr(source, name)))
    pool.total_tokens = source.total_tokens
    pool.logical_pool_count = source.logical_pool_count
    pool.pool_capacity = source.pool_capacity
    return pool


def _pool_state_exact(left, right) -> bool:
    return (
        left.total_tokens == right.total_tokens
        and left.logical_pool_count == right.logical_pool_count
        and all(
            _exact(getattr(left, name), getattr(right, name))
            for name in (
                "pool_keys",
                "pool_indices",
                "pool_valid",
                "raw_keys",
                "raw_gates",
                "raw_valid",
                "raw_positions",
            )
        )
    )


def _first_difference(left: mx.array, right: mx.array):
    left_host = np.asarray(left.astype(mx.float32))
    right_host = np.asarray(right.astype(mx.float32))
    differing = np.argwhere(left_host != right_host)
    if not differing.size:
        return None
    coordinate = tuple(int(value) for value in differing[0])
    return {
        "coordinate": list(coordinate),
        "reference": float(left_host[coordinate]),
        "candidate": float(right_host[coordinate]),
        "differing_elements": int(differing.shape[0]),
        "max_absolute_error": float(np.max(np.abs(left_host - right_host))),
    }


def _native_update(plan, pool, key, gate, valid, query, weights):
    dependencies = (
        key,
        gate,
        valid,
        query,
        weights,
        pool.pool_keys,
        pool.pool_indices,
        pool.pool_valid,
        pool.raw_keys,
        pool.raw_gates,
        pool.raw_valid,
        pool.raw_positions,
        pool.compress_ape,
    )
    mx.async_eval(*dependencies)
    previous = pool.total_tokens
    selected, selected_valid, raw_keys, raw_gates, raw_valid, raw_positions = (
        plan.execute(
            key,
            gate,
            valid,
            query,
            weights,
            pool.pool_keys,
            pool.pool_indices,
            pool.pool_valid,
            pool.raw_keys,
            pool.raw_gates,
            pool.raw_valid,
            pool.raw_positions,
            pool.compress_ape,
            previous,
        )
    )
    pool.raw_keys = raw_keys
    pool.raw_gates = raw_gates
    pool.raw_valid = raw_valid
    pool.raw_positions = raw_positions
    pool.total_tokens = previous + 1
    pool.logical_pool_count = (pool.total_tokens + 3) // 4
    return selected, selected_valid


def _artificial_contract(plan_type, tier0) -> dict:
    rng = np.random.default_rng(5307)
    physical_rows = 576
    indexer = SimpleNamespace(
        index_kpool=4,
        index_topk=2_048,
        head_dim=128,
        index_kpool_always_select_tail=True,
        index_kpool_compress_ape=mx.array(
            rng.normal(size=(4, 128)).astype(np.float32), dtype=mx.bfloat16
        ),
        softmax_scale=128**-0.5,
        bypass_short=True,
    )
    source = _make_pool(indexer, 2_048, physical_rows, 5308)
    reference = _clone_pool(source, indexer)
    candidate = _clone_pool(source, indexer)
    plan = plan_type(physical_rows, float(indexer.softmax_scale))
    before = list(plan.buffer_identities)
    steps = []
    all_exact = True
    for step in range(4):
        key = mx.array(
            rng.normal(size=(1, 1, 128)).astype(np.float32), dtype=mx.bfloat16
        )
        gate = mx.array(
            rng.normal(size=(1, 1, 128)).astype(np.float32), dtype=mx.bfloat16
        )
        valid = mx.array([[True]], dtype=mx.bool_)
        query = mx.array(
            rng.normal(size=(1, 1, 32, 128)).astype(np.float32),
            dtype=mx.bfloat16,
        )
        weights = mx.array(
            rng.normal(size=(1, 1, 32)).astype(np.float32), dtype=mx.bfloat16
        )
        reference._append_projected(key, gate, valid)
        active = reference.active_tail_count or 4
        suffix_gates = reference.raw_gates[:, -active:]
        reference_logits_grouped = (
            suffix_gates + reference.compress_ape[None, :active]
        )
        if active < 4:
            reference_logits_grouped = mx.concatenate(
                [
                    reference_logits_grouped,
                    mx.full((1, 4 - active, 128), -1e30, dtype=mx.bfloat16),
                ],
                axis=1,
            )
        reference_probabilities = mx.softmax(
            reference_logits_grouped[:, None], axis=2
        )[0, 0].swapaxes(0, 1)
        reference_logits = reference_logits_grouped[0].swapaxes(0, 1)
        valid_candidates = reference.pool_valid[:, None, : reference.logical_pool_count]
        scores = tier0._score_expression(
            query,
            reference.pool_keys[:, : reference.logical_pool_count],
            weights,
            valid_candidates,
            indexer.softmax_scale,
        )
        expected, expected_valid = tier0._reference_selection(
            indexer, reference, scores, valid_candidates, valid
        )
        selected, selected_valid = _native_update(
            plan, candidate, key, gate, valid, query, weights
        )
        _eval((expected, expected_valid, selected, selected_valid, reference.state, candidate.state, reference_logits, reference_probabilities, plan.debug_pool_logits, plan.debug_pool_probabilities))
        state_exact = _pool_state_exact(reference, candidate)
        selection_exact = _exact(expected, selected) and _exact(
            expected_valid, selected_valid
        )
        row = {
            "previous_total_mod4": step,
            "active_tail_count": candidate.active_tail_count,
            "pool_state_byte_exact": state_exact,
            "selected_indices_byte_exact": _exact(expected, selected),
            "selected_validity_byte_exact": _exact(expected_valid, selected_valid),
            "pool_logits_byte_exact": _exact(reference_logits, plan.debug_pool_logits),
            "pool_probabilities_byte_exact": _exact(reference_probabilities, plan.debug_pool_probabilities),
            "pool_probability_first_difference": _first_difference(
                reference_probabilities, plan.debug_pool_probabilities
            ),
            "pool_key_first_difference": _first_difference(
                reference.pool_keys[:, reference.logical_pool_count - 1],
                candidate.pool_keys[:, candidate.logical_pool_count - 1],
            ),
        }
        row["all_exact"] = state_exact and selection_exact
        steps.append(row)
        all_exact = all_exact and row["all_exact"]
    evidence = {
        "execution_count": plan.execution_count,
        "scratch_bytes": plan.scratch_bytes,
        "dynamic_allocation_count": plan.dynamic_allocation_count,
        "graph_node_count": plan.graph_node_count,
        "shape_discovery_count": plan.shape_discovery_count,
        "host_synchronization_count": plan.host_synchronization_count,
        "returned_intermediate_tensor_bytes": plan.returned_intermediate_tensor_bytes,
        "buffer_identities_stable": before == list(plan.buffer_identities),
    }
    return {"steps": steps, "plan": evidence, "all_exact": all_exact}


class _Registry:
    def __init__(self, plan_type):
        self.plan_type = plan_type
        self.plans = {}
        self.initial = {}

    def get(self, pool, indexer):
        key = (id(pool), id(indexer), int(pool.pool_keys.shape[1]))
        if key not in self.plans:
            plan = self.plan_type(
                int(pool.pool_keys.shape[1]), float(indexer.softmax_scale)
            )
            self.plans[key] = plan
            self.initial[key] = list(plan.buffer_identities)
        return self.plans[key]

    def evidence(self):
        return [
            {
                "key": list(map(str, key)),
                "execution_count": plan.execution_count,
                "scratch_bytes": plan.scratch_bytes,
                "dynamic_allocation_count": plan.dynamic_allocation_count,
                "graph_node_count": plan.graph_node_count,
                "shape_discovery_count": plan.shape_discovery_count,
                "host_synchronization_count": plan.host_synchronization_count,
                "returned_intermediate_tensor_bytes": plan.returned_intermediate_tensor_bytes,
                "buffer_identities_stable": self.initial[key]
                == list(plan.buffer_identities),
            }
            for key, plan in self.plans.items()
        ]


def _candidate_update(registry, original):
    def update(self, indexer, x, qr, mask=None):
        length = int(x.shape[1])
        short_bypass = self.validate_update(
            indexer, batch=int(x.shape[0]), length=length
        )
        if (
            short_bypass
            or length != 1
            or self.raw_token_count != 19
            or mask is not None
        ):
            return original(self, indexer, x, qr, mask=mask)
        key = indexer.k_norm(indexer.wk(x)).reshape(1, 1, self.head_dim)
        gate = x @ indexer.index_kpool_compress_gate.swapaxes(-1, -2)
        valid = mx.ones((1, 1), dtype=mx.bool_)
        query = indexer.wq_b(qr).reshape(1, 1, indexer.n_heads, indexer.head_dim)
        weights = indexer.weights_proj(x) * (indexer.n_heads**-0.5)
        selected, _ = _native_update(
            registry.get(self, indexer), self, key, gate, valid, query, weights
        )
        return selected[:, None]

    return update


@contextlib.contextmanager
def _native_arm(registry, enabled: bool):
    if not enabled:
        yield
        return
    original = CompactIndexPoolCache.update
    CompactIndexPoolCache.update = _candidate_update(registry, original)
    try:
        yield
    finally:
        CompactIndexPoolCache.update = original


def _median(rows):
    return {
        "median_wall_ms": statistics.median(row["wall_ms"] for row in rows),
        "median_host_submit_ms": statistics.median(
            row["host_submit_ms"] for row in rows
        ),
        "samples": rows,
    }


def _full_model_case(model, boundary, tier1, tier0, source, plan_type, context, warmups, samples):
    arms = ("A_tier1_update", "B_native_update_tier1_selection")
    caches = {
        arm: boundary._clone_cache(source, context + warmups + samples + 16)
        for arm in arms
    }
    tier1_registries = {arm: tier1._Registry(tier1._load_helpers()[0]) for arm in arms}
    native_registry = _Registry(plan_type)
    measured = {arm: [] for arm in arms}
    hashes = {arm: [] for arm in arms}
    token = mx.array([[3000]], dtype=mx.uint32)
    for iteration in range(warmups + samples):
        order = arms if iteration % 2 == 0 else tuple(reversed(arms))
        for arm in order:
            started = time.perf_counter_ns()
            if arm == arms[0]:
                with tier1._native_arm(tier1_registries[arm], True):
                    output = model(token, cache=caches[arm])
            else:
                with _native_arm(native_registry, True):
                    output = model(token, cache=caches[arm])
            submitted = time.perf_counter_ns()
            _eval(output.logits)
            finished = time.perf_counter_ns()
            hashes[arm].append(tier0._hash(output.logits[0, -1]))
            if iteration >= warmups:
                measured[arm].append(
                    {
                        "host_submit_ms": (submitted - started) / 1e6,
                        "wall_ms": (finished - started) / 1e6,
                    }
                )
    timing = {arm: _median(rows) for arm, rows in measured.items()}
    saving = timing[arms[0]]["median_wall_ms"] - timing[arms[1]]["median_wall_ms"]
    result = {
        "context_tokens": context,
        "timing": timing,
        "native_wall_saving_ms": saving,
        "all_logits_byte_exact": hashes[arms[0]] == hashes[arms[1]],
        "post_state_byte_exact": boundary._cache_exact(caches[arms[0]], caches[arms[1]]),
        "native_plan_evidence": native_registry.evidence(),
    }
    caches.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    return result


def _acceptance(artifact):
    short = artifact.get("contexts", {}).get("2048", {})
    long = artifact.get("contexts", {}).get("262144", {})
    evidence = [row for case in artifact.get("contexts", {}).values() for row in case.get("native_plan_evidence", [])]
    exact = bool(short) and bool(long) and all(
        case.get("all_logits_byte_exact") and case.get("post_state_byte_exact")
        for case in (short, long)
    )
    return {
        "artificial_update_pool_selection_byte_exact": artifact.get("artificial_native_contract", {}).get("all_exact", False),
        "full_model_logits_and_state_byte_exact": exact,
        "256k_incremental_full_model_saving_at_least_0_75ms": long.get("native_wall_saving_ms", -1e9) >= MIN_256K_SAVING_MS,
        "2k_regression_at_most_1_percent": short.get("timing", {}).get("B_native_update_tier1_selection", {}).get("median_wall_ms", 1e9) <= short.get("timing", {}).get("A_tier1_update", {}).get("median_wall_ms", 0.0) * (1 + MAX_2K_REGRESSION),
        "fixed_arena_zero_execute_allocation_graph_shape_sync": bool(evidence) and all(
            row.get("buffer_identities_stable")
            and row.get("dynamic_allocation_count") == 0
            and row.get("graph_node_count") == 0
            and row.get("shape_discovery_count") == 0
            and row.get("host_synchronization_count") == 0
            and row.get("returned_intermediate_tensor_bytes") == 0
            for row in evidence
        ),
        "official_oracle_exact": bool(artifact.get("official_oracle", {}).get("all_full_vocab_logits_hashes_match")),
        "process_peak_at_most_340GB": artifact.get("process_peak_memory_bytes", 1 << 60) <= MAX_PROCESS_PEAK_BYTES,
        "production_abi_unchanged": artifact.get("runtime_changes") == {"runtime": False, "server": False, "apc": False, "cache_abi": False, "kernel_abi": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artificial-only", action="store_true")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    args = parser.parse_args()
    artifact = {
        "schema": "glm53-native-indexpool-update-submission-island-v1",
        "date": date.today().isoformat(),
        "complete": False,
        "accepted": False,
        "probe_only": True,
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "compact_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
        "runtime_changes": {"runtime": False, "server": False, "apc": False, "cache_abi": False, "kernel_abi": False},
    }
    try:
        plan_type, oracle_probe, boundary, tier1, tier0 = _load_helpers()
        _progress("artificial_native_contract")
        artifact["artificial_native_contract"] = _artificial_contract(plan_type, tier0)
        if not artifact["artificial_native_contract"]["all_exact"]:
            raise RuntimeError("artificial native IndexPool update is not byte-exact")
        if args.artificial_only:
            artifact.update(complete=True, accepted=True, decision="advance_to_full_model")
        else:
            report = inspect_checkpoint(args.model, require_server_ready=True)
            artifact["checkpoint_fingerprint"] = report.fingerprint
            artifact["official_hf_revision"] = report.official_revision
            mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
            mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
            _progress("load_model")
            model, processor = load(
                args.model,
                experimental_packed_decode_moe=True,
                experimental_compact_nope_dsa_cache=True,
                compact_cache_capacity_tokens=max(CONTEXTS) + 64,
            )
            warm_residency(model)
            artifact["official_oracle"] = oracle_probe._official_oracle(model, processor, report)
            artifact["contexts"] = {}
            for context in CONTEXTS:
                _progress("decode_context", context=context)
                source = boundary._synthetic_cache(model, context, "compact-nope-dsa")
                artifact["contexts"][str(context)] = _full_model_case(
                    model, boundary, tier1, tier0, source, plan_type, context, args.warmups, args.samples
                )
                _atomic_write(args.output, artifact)
                source.clear()
                gc.collect()
                mx.clear_cache()
                mx.synchronize()
            artifact["process_peak_memory_bytes"] = int(mx.get_peak_memory())
            artifact["acceptance"] = _acceptance(artifact)
            artifact["complete"] = True
            artifact["accepted"] = all(artifact["acceptance"].values())
            artifact["decision"] = "keep_native_indexpool_update_submission_island" if artifact["accepted"] else "stop_or_redesign_native_indexpool_update_submission_island"
    except Exception as error:
        artifact.update(
            complete=True,
            accepted=False,
            decision="native_indexpool_update_submission_island_probe_failed",
            error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
        )
    _atomic_write(args.output, artifact)
    print(json.dumps({
        "output": str(args.output),
        "complete": artifact["complete"],
        "accepted": artifact["accepted"],
        "decision": artifact["decision"],
        "failed_gates": [key for key, value in artifact.get("acceptance", {}).items() if not value],
    }, indent=2))
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
