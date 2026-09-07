#!/usr/bin/env python3
"""Qualify an exact persistent native packed-MoE decode plan.

The authoritative MLX Float32 router remains outside the native boundary.
For L=1, its selected expert IDs and scores feed one fixed-address native plan
covering routed gate+up+SwiGLU, routed down, exact B1 weighting/reduction,
shared gate+up+SwiGLU, shared down, and the final routed/shared addition.
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
from contextlib import ExitStack
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from glm53_flash_mlx.abi import MLX_VLM_REVISION, NOPE_DSA_CACHE_ABI_COMPACT
from glm53_flash_mlx.fp8 import block_fp8_linear
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint
from glm53_flash_mlx.packed import PackedFP8MoE


ROOT = Path(__file__).resolve().parents[1]
NATIVE_PACKAGE = ROOT / "native_execution"
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-packed-moe-execution-plan-20260907.json"
)
COMPOSITION_ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-fused-decode-composition-20260907.json"
)
CONTEXTS = (2_048, 262_144)
TARGET_2K_TPS = 15.0
MIN_NATIVE_WALL_SAVING_MS = 0.50
MIN_NATIVE_HOST_SAVING_MS = 0.50
MIN_CONTEXT_RETENTION = 0.90
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
    from glm53_native_execution import (
        NativeIndexPoolUpdateSelectionPlan,
        NativePackedMoEDecodePlan,
    )
    import probe_exact_sigmoid_gate_metal_barrier as oracle_probe
    import probe_fused_packed_gate_up_swiglu_decode as d99f
    import probe_long_context_first_decode_boundary as boundary
    import probe_native_execution_engine_feasibility as tier0
    import probe_native_indexpool_update_submission_island as update_island
    import probe_residual_packed_decode_moe_fusion as residual

    return (
        NativeIndexPoolUpdateSelectionPlan,
        NativePackedMoEDecodePlan,
        oracle_probe,
        d99f,
        boundary,
        tier0,
        update_island,
        residual,
    )


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


def _qualified_reference_arm(residual):
    source = json.loads(COMPOSITION_ARTIFACT.read_text())
    if not source["accepted"] or not all(source["acceptance"].values()):
        raise RuntimeError("native/fused composition source is not qualified")
    if source["fused_moe_source"]["selected_aggregation"] != "B1":
        raise RuntimeError("native plan requires the exact B1 reduction oracle")
    return residual.Arm("B1", True), {
        "artifact": str(COMPOSITION_ARTIFACT.relative_to(ROOT)),
        "accepted": True,
        "2k_tokens_per_second": source["contexts"]["2048"]["timing"][
            "B_native_indexpool_exact_fused_packed"
        ]["tokens_per_second"],
        "256k_tokens_per_second": source["contexts"]["262144"]["timing"][
            "B_native_indexpool_exact_fused_packed"
        ]["tokens_per_second"],
    }


def _random_array(rng, shape, dtype, *, scale=False):
    if dtype == mx.uint8:
        host = rng.integers(0, 256, size=shape, dtype=np.uint8)
    elif scale:
        host = rng.uniform(0.0005, 0.025, size=shape).astype(np.float32)
    else:
        host = rng.normal(0.0, 0.25, size=shape).astype(np.float32)
    result = mx.array(host, dtype=dtype)
    mx.eval(result)
    return result


def _artificial_contract(plan_type, d99f, residual, tier0) -> dict:
    rng = np.random.default_rng(0x53A11)
    hidden_size = 256
    intermediate_size = 128
    shared_intermediate = 128
    expert_count = 16
    hidden_scale_rows = hidden_size // 128
    intermediate_scale_rows = intermediate_size // 128
    x = _random_array(rng, (1, 1, hidden_size), mx.bfloat16)
    expert_ids = mx.array([0, 3, 5, 7, 9, 11, 13, 15], dtype=mx.uint32)
    scores = mx.array(
        np.asarray([0.19, 0.17, 0.15, 0.13, 0.12, 0.10, 0.08, 0.06], np.float32)
    )
    bank = SimpleNamespace(
        intermediate_size=intermediate_size,
        intermediate_scale_rows=intermediate_scale_rows,
        gate_up_weight=_random_array(
            rng,
            (expert_count, 2 * intermediate_size, hidden_size),
            mx.uint8,
        ),
        gate_up_scale_inv=_random_array(
            rng,
            (
                expert_count,
                2 * intermediate_scale_rows,
                hidden_scale_rows,
            ),
            mx.float32,
            scale=True,
        ),
        down_weight=_random_array(
            rng, (expert_count, hidden_size, intermediate_size), mx.uint8
        ),
        down_scale_inv=_random_array(
            rng,
            (expert_count, hidden_scale_rows, intermediate_scale_rows),
            mx.float32,
            scale=True,
        ),
    )
    shared = SimpleNamespace(
        gate_proj=SimpleNamespace(
            weight=_random_array(
                rng, (shared_intermediate, hidden_size), mx.uint8
            ),
            weight_scale_inv=_random_array(
                rng,
                (shared_intermediate // 128, hidden_scale_rows),
                mx.float32,
                scale=True,
            ),
        ),
        up_proj=SimpleNamespace(
            weight=_random_array(
                rng, (shared_intermediate, hidden_size), mx.uint8
            ),
            weight_scale_inv=_random_array(
                rng,
                (shared_intermediate // 128, hidden_scale_rows),
                mx.float32,
                scale=True,
            ),
        ),
        down_proj=SimpleNamespace(
            weight=_random_array(
                rng, (hidden_size, shared_intermediate), mx.uint8
            ),
            weight_scale_inv=_random_array(
                rng,
                (hidden_scale_rows, shared_intermediate // 128),
                mx.float32,
                scale=True,
            ),
        ),
    )

    flat = x.reshape(-1, hidden_size)[0]
    routed_hidden = d99f.fused_packed_gate_up_swiglu(
        flat, expert_ids, bank, limit=7.0
    )
    routed_down = d99f._packed_down_raw(routed_hidden, expert_ids, bank)
    routed = residual.aggregate_b1(routed_down, scores)
    shared_hidden = residual.fused_shared_gate_up_swiglu(
        flat, shared, limit=7.0
    )
    shared_down = block_fp8_linear(
        shared_hidden,
        shared.down_proj.weight,
        shared.down_proj.weight_scale_inv,
    )
    reference = (routed + shared_down).reshape(1, 1, hidden_size)

    plan = plan_type(
        hidden_size,
        intermediate_size,
        shared_intermediate,
        expert_count,
        7,
    )
    initial_identities = list(plan.buffer_identities)
    dependencies = (
        x,
        expert_ids,
        scores,
        bank.gate_up_weight,
        bank.gate_up_scale_inv,
        bank.down_weight,
        bank.down_scale_inv,
        shared.gate_proj.weight,
        shared.gate_proj.weight_scale_inv,
        shared.up_proj.weight,
        shared.up_proj.weight_scale_inv,
        shared.down_proj.weight,
        shared.down_proj.weight_scale_inv,
    )
    mx.async_eval(*dependencies)
    candidate = plan.execute(*dependencies)
    _eval((reference, candidate))
    return {
        "output_byte_exact": _exact(reference, candidate),
        "reference_hash": tier0._hash(reference),
        "candidate_hash": tier0._hash(candidate),
        "execution_count": int(plan.execution_count),
        "dynamic_allocation_count": int(plan.dynamic_allocation_count),
        "graph_node_count": int(plan.graph_node_count),
        "shape_discovery_count": int(plan.shape_discovery_count),
        "host_synchronization_count": int(plan.host_synchronization_count),
        "returned_intermediate_tensor_bytes": int(
            plan.returned_intermediate_tensor_bytes
        ),
        "scratch_bytes": int(plan.scratch_bytes),
        "buffer_identities_stable": list(plan.buffer_identities)
        == initial_identities,
    }


class _Registry:
    def __init__(self, plan_type):
        self.plan_type = plan_type
        self.plans = {}
        self.initial_identities = {}

    def get(self, moe):
        key = id(moe)
        plan = self.plans.get(key)
        if plan is None:
            shared = moe.shared_experts
            if shared is None:
                raise RuntimeError("native packed MoE requires one shared expert")
            limit = float(moe.config.swiglu_limit)
            if limit != int(limit):
                raise RuntimeError("native packed MoE requires integral SwiGLU limit")
            plan = self.plan_type(
                int(moe.bank.down_weight.shape[1]),
                int(moe.bank.intermediate_size),
                int(shared.gate_proj.weight.shape[0]),
                int(moe.bank.expert_count),
                int(limit),
            )
            self.plans[key] = plan
            self.initial_identities[key] = list(plan.buffer_identities)
        return plan

    def evidence(self):
        return [
            {
                "execution_count": int(plan.execution_count),
                "dynamic_allocation_count": int(plan.dynamic_allocation_count),
                "graph_node_count": int(plan.graph_node_count),
                "shape_discovery_count": int(plan.shape_discovery_count),
                "host_synchronization_count": int(
                    plan.host_synchronization_count
                ),
                "returned_intermediate_tensor_bytes": int(
                    plan.returned_intermediate_tensor_bytes
                ),
                "scratch_bytes": int(plan.scratch_bytes),
                "buffer_identities": list(plan.buffer_identities),
                "buffer_identities_stable": list(plan.buffer_identities)
                == self.initial_identities[key],
            }
            for key, plan in self.plans.items()
        ]


_ORIGINAL_PACKED_CALL = PackedFP8MoE.__call__


def _native_call(registry, moe, x):
    flat = mx.contiguous(x.reshape(-1, x.shape[-1]), allow_col_major=False)
    if int(flat.shape[0]) != 1 or moe.shared_experts is None:
        return _ORIGINAL_PACKED_CALL(moe, x)
    indices, scores = moe.gate(x)
    if int(indices.shape[-1]) != 8:
        return _ORIGINAL_PACKED_CALL(moe, x)
    expert_ids = mx.contiguous(
        indices.reshape(-1).astype(mx.uint32), allow_col_major=False
    )
    flat_scores = mx.contiguous(scores.reshape(-1), allow_col_major=False)
    shared = moe.shared_experts
    dependencies = (
        flat[0],
        expert_ids,
        flat_scores,
        moe.bank.gate_up_weight,
        moe.bank.gate_up_scale_inv,
        moe.bank.down_weight,
        moe.bank.down_scale_inv,
        shared.gate_proj.weight,
        shared.gate_proj.weight_scale_inv,
        shared.up_proj.weight,
        shared.up_proj.weight_scale_inv,
        shared.down_proj.weight,
        shared.down_proj.weight_scale_inv,
    )
    mx.async_eval(*dependencies)
    return registry.get(moe).execute(*dependencies).reshape(x.shape)


@contextlib.contextmanager
def _native_moe_arm(registry):
    previous = PackedFP8MoE.__call__

    def wrapped(moe, x):
        return _native_call(registry, moe, x)

    PackedFP8MoE.__call__ = wrapped
    try:
        yield
    finally:
        PackedFP8MoE.__call__ = previous


def _median(rows: list[dict]) -> dict:
    wall = statistics.median(row["wall_ms"] for row in rows)
    return {
        "median_wall_ms": wall,
        "median_host_submit_ms": statistics.median(
            row["host_submit_ms"] for row in rows
        ),
        "tokens_per_second": 1_000.0 / wall,
        "samples": rows,
    }


def _context_case(
    model,
    source,
    context,
    *,
    index_plan_type,
    moe_plan_type,
    boundary,
    tier0,
    update_island,
    residual,
    reference_arm,
    warmups,
    samples,
):
    arms = ("A_exact_composition", "B_native_packed_moe_plan")
    caches = {
        arm: boundary._clone_cache(source, context + warmups + samples + 16)
        for arm in arms
    }
    index_registries = {
        arm: update_island._Registry(index_plan_type) for arm in arms
    }
    moe_registry = _Registry(moe_plan_type)
    measured = {arm: [] for arm in arms}
    hashes = {arm: [] for arm in arms}
    token_ids = {arm: [] for arm in arms}
    reference_fused_calls = 0
    token = mx.array([[3_000]], dtype=mx.uint32)

    for iteration in range(warmups + samples):
        order = arms if iteration % 2 == 0 else tuple(reversed(arms))
        for arm in order:
            started = time.perf_counter_ns()
            with ExitStack() as stack:
                stack.enter_context(
                    update_island._native_arm(index_registries[arm], True)
                )
                if arm == arms[0]:
                    stack.enter_context(residual._runtime(reference_arm))
                else:
                    stack.enter_context(_native_moe_arm(moe_registry))
                output = model(token, cache=caches[arm])
                if arm == arms[0]:
                    reference_fused_calls += residual._FUSED_CALL_COUNT
            submitted = time.perf_counter_ns()
            _eval(output.logits)
            finished = time.perf_counter_ns()
            hashes[arm].append(tier0._hash(output.logits[0, -1]))
            token_ids[arm].append(int(mx.argmax(output.logits[0, -1]).item()))
            if iteration >= warmups:
                measured[arm].append(
                    {
                        "host_submit_ms": (submitted - started) / 1e6,
                        "wall_ms": (finished - started) / 1e6,
                    }
                )

    timing = {arm: _median(rows) for arm, rows in measured.items()}
    reference = timing[arms[0]]
    candidate = timing[arms[1]]
    evidence = moe_registry.evidence()
    result = {
        "context_tokens": context,
        "timing": timing,
        "native_wall_saving_ms": reference["median_wall_ms"]
        - candidate["median_wall_ms"],
        "native_host_saving_ms": reference["median_host_submit_ms"]
        - candidate["median_host_submit_ms"],
        "native_speedup": reference["median_wall_ms"]
        / candidate["median_wall_ms"],
        "all_full_vocab_logits_byte_exact": hashes[arms[0]] == hashes[arms[1]],
        "all_generated_tokens_exact": token_ids[arms[0]] == token_ids[arms[1]],
        "post_state_byte_exact": boundary._cache_exact(
            caches[arms[0]], caches[arms[1]]
        ),
        "reference_fused_sparse_moe_calls": reference_fused_calls,
        "expected_sparse_moe_calls": 42 * (warmups + samples),
        "native_moe_plan_count": len(evidence),
        "native_moe_execution_count": sum(
            row["execution_count"] for row in evidence
        ),
        "native_moe_scratch_bytes": sum(row["scratch_bytes"] for row in evidence),
        "native_moe_plan_evidence": evidence,
        "native_indexpool_plan_evidence": {
            arm: index_registries[arm].evidence() for arm in arms
        },
    }
    caches.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    return result


def _acceptance(artifact):
    short = artifact.get("contexts", {}).get("2048", {})
    long = artifact.get("contexts", {}).get("262144", {})
    cases = (short, long)
    candidate_short = short.get("timing", {}).get("B_native_packed_moe_plan", {})
    candidate_long = long.get("timing", {}).get("B_native_packed_moe_plan", {})
    short_tps = candidate_short.get("tokens_per_second", 0.0)
    long_tps = candidate_long.get("tokens_per_second", 0.0)
    moe_evidence = [
        row for case in cases for row in case.get("native_moe_plan_evidence", [])
    ]
    return {
        "artificial_native_output_byte_exact": artifact.get(
            "artificial_native_contract", {}
        ).get("output_byte_exact", False),
        "all_full_model_logits_tokens_and_state_byte_exact": bool(short)
        and bool(long)
        and all(
            case.get("all_full_vocab_logits_byte_exact")
            and case.get("all_generated_tokens_exact")
            and case.get("post_state_byte_exact")
            for case in cases
        ),
        "all_42_sparse_layers_use_native_plan": all(
            case.get("native_moe_plan_count") == 42
            and case.get("native_moe_execution_count")
            == case.get("expected_sparse_moe_calls")
            for case in cases
        ),
        "2k_decode_at_least_15_tps": short_tps >= TARGET_2K_TPS,
        "2k_native_wall_saving_at_least_0_50ms": short.get(
            "native_wall_saving_ms", -1e9
        )
        >= MIN_NATIVE_WALL_SAVING_MS,
        "256k_native_wall_saving_at_least_0_50ms": long.get(
            "native_wall_saving_ms", -1e9
        )
        >= MIN_NATIVE_WALL_SAVING_MS,
        "2k_native_host_saving_at_least_0_50ms": short.get(
            "native_host_saving_ms", -1e9
        )
        >= MIN_NATIVE_HOST_SAVING_MS,
        "256k_native_host_saving_at_least_0_50ms": long.get(
            "native_host_saving_ms", -1e9
        )
        >= MIN_NATIVE_HOST_SAVING_MS,
        "2k_to_256k_retention_at_least_0_90": bool(short_tps)
        and long_tps / short_tps >= MIN_CONTEXT_RETENTION,
        "fixed_arena_zero_execute_allocation_graph_shape_sync": bool(moe_evidence)
        and all(
            row["buffer_identities_stable"]
            and row["dynamic_allocation_count"] == 0
            and row["graph_node_count"] == 0
            and row["shape_discovery_count"] == 0
            and row["host_synchronization_count"] == 0
            and row["returned_intermediate_tensor_bytes"] == 0
            for row in moe_evidence
        ),
        "official_oracle_exact_both_arms": all(
            row.get("all_full_vocab_logits_hashes_match")
            for row in artifact.get("official_oracle", {}).values()
        ),
        "process_peak_at_most_340GB": artifact.get(
            "process_peak_memory_bytes", 1 << 60
        )
        <= MAX_PROCESS_PEAK_BYTES,
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
        "schema": "glm53-native-packed-moe-execution-plan-v1",
        "date": date.today().isoformat(),
        "complete": False,
        "accepted": False,
        "probe_only": True,
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "compact_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
        "runtime_changes": {
            "runtime": False,
            "server": False,
            "apc": False,
            "cache_abi": False,
            "kernel_abi": False,
        },
    }
    try:
        (
            index_plan_type,
            moe_plan_type,
            oracle_probe,
            d99f,
            boundary,
            tier0,
            update_island,
            residual,
        ) = _load_helpers()
        reference_arm, artifact["reference_composition"] = (
            _qualified_reference_arm(residual)
        )
        _progress("artificial_native_contract")
        artifact["artificial_native_contract"] = _artificial_contract(
            moe_plan_type, d99f, residual, tier0
        )
        if not artifact["artificial_native_contract"]["output_byte_exact"]:
            raise RuntimeError("native packed MoE artificial output is not exact")
        if args.artificial_only:
            artifact.update(
                complete=True,
                accepted=True,
                decision="advance_to_full_model_native_packed_moe",
            )
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

            _progress("official_oracle", arm="exact-composition")
            with residual._runtime(reference_arm):
                reference_oracle = oracle_probe._official_oracle(
                    model, processor, report
                )
            _progress("official_oracle", arm="native-packed-moe")
            oracle_registry = _Registry(moe_plan_type)
            with _native_moe_arm(oracle_registry):
                native_oracle = oracle_probe._official_oracle(
                    model, processor, report
                )
            artifact["official_oracle"] = {
                "A_exact_composition": reference_oracle,
                "B_native_packed_moe_plan": native_oracle,
            }

            artifact["contexts"] = {}
            for context in CONTEXTS:
                _progress("decode_context", context=context)
                source = boundary._synthetic_cache(
                    model, context, "compact-nope-dsa"
                )
                artifact["contexts"][str(context)] = _context_case(
                    model,
                    source,
                    context,
                    index_plan_type=index_plan_type,
                    moe_plan_type=moe_plan_type,
                    boundary=boundary,
                    tier0=tier0,
                    update_island=update_island,
                    residual=residual,
                    reference_arm=reference_arm,
                    warmups=args.warmups,
                    samples=args.samples,
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
            artifact["decision"] = (
                "keep_native_packed_moe_execution_plan"
                if artifact["accepted"]
                else "stop_or_redesign_native_packed_moe_execution_plan"
            )
    except Exception as error:
        artifact.update(
            complete=True,
            accepted=False,
            decision="native_packed_moe_execution_plan_probe_failed",
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
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
