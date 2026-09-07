#!/usr/bin/env python3
"""Probe a prebound exact native packed-MoE execution plan."""

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

import mlx.core as mx

from glm53_flash_mlx.abi import MLX_VLM_REVISION, NOPE_DSA_CACHE_ABI_COMPACT
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint
from glm53_flash_mlx.packed import PackedFP8MoE


ROOT = Path(__file__).resolve().parents[1]
NATIVE_PACKAGE = ROOT / "native_execution"
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-prebound-native-packed-moe-execution-plan-20260907.json"
)
EXACT_REPAIR_ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-exact-native-routed-fast-sigmoid-repair-20260907.json"
)
REJECTED_PERFORMANCE_ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-exact-native-packed-moe-performance-requalification-20260907.json"
)
CONTEXTS = (2_048, 262_144)
TARGET_2K_TPS = 15.0
MIN_WALL_SAVING_VS_EXACT_MS = 0.50
MIN_HOST_SAVING_VS_EXACT_MS = 0.50
MIN_HOST_SAVING_VS_UNBOUND_MS = 1.50
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
    import probe_fused_packed_gate_up_swiglu_decode as d99f
    import probe_long_context_first_decode_boundary as boundary
    import probe_native_execution_engine_feasibility as tier0
    import probe_native_indexpool_update_submission_island as update_island
    import probe_native_packed_moe_execution_plan as native_moe
    import probe_residual_packed_decode_moe_fusion as residual

    return (
        NativeIndexPoolUpdateSelectionPlan,
        NativePackedMoEDecodePlan,
        d99f,
        boundary,
        tier0,
        update_island,
        native_moe,
        residual,
    )


def _validate_sources() -> dict:
    exact = json.loads(EXACT_REPAIR_ARTIFACT.read_text())
    performance = json.loads(REJECTED_PERFORMANCE_ARTIFACT.read_text())
    if not exact["complete"] or not exact["accepted"]:
        raise RuntimeError("fast-BF16 native repair is not accepted")
    if not all(exact["acceptance"].values()):
        raise RuntimeError("fast-BF16 native repair has a failed exactness gate")
    if performance["decision"] != (
        "stop_exact_native_packed_moe_execution_plan_on_performance"
    ):
        raise RuntimeError("prebinding requires the recorded performance stop")
    expected_failures = {
        "2k_native_wall_saving_at_least_0_50ms",
        "256k_native_wall_saving_at_least_0_50ms",
        "2k_native_host_saving_at_least_0_50ms",
        "256k_native_host_saving_at_least_0_50ms",
    }
    if set(performance["failed_gates"]) != expected_failures:
        raise RuntimeError("native plan failed outside the expected boundary tax")
    return {
        "exact_repair": {
            "artifact": str(EXACT_REPAIR_ARTIFACT.relative_to(ROOT)),
            "accepted": True,
            "checkpoint_fingerprint": exact["checkpoint_fingerprint"],
            "formula": exact["repair"]["formula"],
        },
        "unbound_performance": {
            "artifact": str(REJECTED_PERFORMANCE_ARTIFACT.relative_to(ROOT)),
            "accepted": False,
            "failed_gates": performance["failed_gates"],
            "contexts": {
                context: {
                    "native_wall_saving_ms": row["native_wall_saving_ms"],
                    "native_host_saving_ms": row["native_host_saving_ms"],
                }
                for context, row in performance["contexts"].items()
            },
        },
    }


def _weights(moe) -> tuple[mx.array, ...]:
    shared = moe.shared_experts
    return (
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


class _BoundRegistry:
    def __init__(self, plan_type):
        self.plan_type = plan_type
        self.plans = {}
        self.initial_scratch = {}
        self.initial_weights = {}

    def get(self, moe):
        key = id(moe)
        plan = self.plans.get(key)
        if plan is None:
            shared = moe.shared_experts
            if shared is None:
                raise RuntimeError("prebound native plan requires one shared expert")
            limit = float(moe.config.swiglu_limit)
            if limit != int(limit):
                raise RuntimeError("prebound native plan requires integral SwiGLU limit")
            plan = self.plan_type(
                int(moe.bank.down_weight.shape[1]),
                int(moe.bank.intermediate_size),
                int(shared.gate_proj.weight.shape[0]),
                int(moe.bank.expert_count),
                int(limit),
            )
            weights = _weights(moe)
            mx.async_eval(*weights)
            plan.bind_weights(*weights)
            self.plans[key] = plan
            self.initial_scratch[key] = list(plan.buffer_identities)
            self.initial_weights[key] = list(plan.bound_weight_identities)
        return plan

    def evidence(self) -> list[dict]:
        return [
            {
                "execution_count": int(plan.execution_count),
                "dynamic_allocation_count": int(plan.dynamic_allocation_count),
                "graph_node_count": int(plan.graph_node_count),
                "shape_discovery_count": int(plan.shape_discovery_count),
                "host_synchronization_count": int(plan.host_synchronization_count),
                "returned_intermediate_tensor_bytes": int(
                    plan.returned_intermediate_tensor_bytes
                ),
                "static_input_validation_count": int(
                    plan.static_input_validation_count
                ),
                "dynamic_input_validation_count": int(
                    plan.dynamic_input_validation_count
                ),
                "pipeline_lookup_count": int(plan.pipeline_lookup_count),
                "weights_bound": bool(plan.weights_bound),
                "bound_weight_identities_stable": bool(
                    plan.bound_weight_identities_stable
                ),
                "uses_shape_specialized_routed_gate_up": bool(
                    plan.uses_shape_specialized_routed_gate_up
                ),
                "uses_fast_bf16_routed_sigmoid": bool(
                    plan.uses_fast_bf16_routed_sigmoid
                ),
                "scratch_buffer_identities_stable": list(plan.buffer_identities)
                == self.initial_scratch[key],
                "bound_weight_handles_stable": list(plan.bound_weight_identities)
                == self.initial_weights[key],
            }
            for key, plan in self.plans.items()
        ]


_ORIGINAL_PACKED_CALL = PackedFP8MoE.__call__


def _bound_call(registry: _BoundRegistry, moe, x):
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
    mx.async_eval(flat[0], expert_ids, flat_scores)
    return registry.get(moe).execute_bound(
        flat[0], expert_ids, flat_scores
    ).reshape(x.shape)


@contextlib.contextmanager
def _prebound_native_arm(registry):
    previous = PackedFP8MoE.__call__

    def wrapped(moe, x):
        return _bound_call(registry, moe, x)

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
    native_moe,
    residual,
    reference_arm,
    warmups,
    samples,
) -> dict:
    arms = (
        "A_exact_composition",
        "B_unbound_native_plan",
        "C_prebound_native_plan",
    )
    caches = {
        arm: boundary._clone_cache(source, context + warmups + samples + 16)
        for arm in arms
    }
    index_registries = {
        arm: update_island._Registry(index_plan_type) for arm in arms
    }
    unbound_registry = native_moe._Registry(moe_plan_type)
    prebound_registry = _BoundRegistry(moe_plan_type)
    measured = {arm: [] for arm in arms}
    hashes = {arm: [] for arm in arms}
    token_ids = {arm: [] for arm in arms}
    token = mx.array([[3_000]], dtype=mx.uint32)

    for iteration in range(warmups + samples):
        offset = iteration % len(arms)
        order = arms[offset:] + arms[:offset]
        for arm in order:
            started = time.perf_counter_ns()
            with ExitStack() as stack:
                stack.enter_context(
                    update_island._native_arm(index_registries[arm], True)
                )
                if arm == arms[0]:
                    stack.enter_context(residual._runtime(reference_arm))
                elif arm == arms[1]:
                    stack.enter_context(native_moe._native_moe_arm(unbound_registry))
                else:
                    stack.enter_context(_prebound_native_arm(prebound_registry))
                output = model(token, cache=caches[arm])
            submitted = time.perf_counter_ns()
            native_moe._eval(output.logits)
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
    exact = timing[arms[0]]
    unbound = timing[arms[1]]
    prebound = timing[arms[2]]
    unbound_evidence = unbound_registry.evidence()
    prebound_evidence = prebound_registry.evidence()
    result = {
        "context_tokens": context,
        "timing": timing,
        "prebound_vs_exact_wall_saving_ms": exact["median_wall_ms"]
        - prebound["median_wall_ms"],
        "prebound_vs_exact_host_saving_ms": exact["median_host_submit_ms"]
        - prebound["median_host_submit_ms"],
        "prebound_vs_unbound_wall_saving_ms": unbound["median_wall_ms"]
        - prebound["median_wall_ms"],
        "prebound_vs_unbound_host_saving_ms": unbound["median_host_submit_ms"]
        - prebound["median_host_submit_ms"],
        "prebound_speedup_vs_exact": exact["median_wall_ms"]
        / prebound["median_wall_ms"],
        "all_full_vocab_logits_byte_exact": len(set(map(tuple, hashes.values())))
        == 1,
        "all_generated_tokens_exact": len(set(map(tuple, token_ids.values())))
        == 1,
        "all_post_states_byte_exact": boundary._cache_exact(
            caches[arms[0]], caches[arms[1]]
        )
        and boundary._cache_exact(caches[arms[0]], caches[arms[2]]),
        "expected_sparse_moe_calls_per_arm": 42 * (warmups + samples),
        "unbound_plan_count": len(unbound_evidence),
        "prebound_plan_count": len(prebound_evidence),
        "unbound_plan_evidence": unbound_evidence,
        "prebound_plan_evidence": prebound_evidence,
    }
    caches.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    return result


def _acceptance(artifact: dict) -> dict:
    short = artifact.get("contexts", {}).get("2048", {})
    long = artifact.get("contexts", {}).get("262144", {})
    cases = (short, long)
    short_tps = short.get("timing", {}).get("C_prebound_native_plan", {}).get(
        "tokens_per_second", 0.0
    )
    long_tps = long.get("timing", {}).get("C_prebound_native_plan", {}).get(
        "tokens_per_second", 0.0
    )
    prebound_rows = [
        row for case in cases for row in case.get("prebound_plan_evidence", [])
    ]
    return {
        "source_exactness_and_failure_localization_accepted": bool(
            artifact.get("source_evidence")
        ),
        "all_three_arms_logits_tokens_and_state_byte_exact": bool(short)
        and bool(long)
        and all(
            case.get("all_full_vocab_logits_byte_exact")
            and case.get("all_generated_tokens_exact")
            and case.get("all_post_states_byte_exact")
            for case in cases
        ),
        "all_42_layers_prebound_once_and_execute_dynamic_only": bool(
            prebound_rows
        )
        and all(
            case.get("prebound_plan_count") == 42
            and sum(row["execution_count"] for row in case["prebound_plan_evidence"])
            == case["expected_sparse_moe_calls_per_arm"]
            for case in cases
        )
        and all(
            row["weights_bound"]
            and row["bound_weight_identities_stable"]
            and row["bound_weight_handles_stable"]
            and row["scratch_buffer_identities_stable"]
            and row["static_input_validation_count"] == 10
            and row["dynamic_input_validation_count"]
            == 3 * row["execution_count"]
            and row["pipeline_lookup_count"] == 6
            for row in prebound_rows
        ),
        "2k_prebound_decode_at_least_15_tps": short_tps >= TARGET_2K_TPS,
        "2k_prebound_wall_saving_vs_exact_at_least_0_50ms": short.get(
            "prebound_vs_exact_wall_saving_ms", -1e9
        )
        >= MIN_WALL_SAVING_VS_EXACT_MS,
        "256k_prebound_wall_saving_vs_exact_at_least_0_50ms": long.get(
            "prebound_vs_exact_wall_saving_ms", -1e9
        )
        >= MIN_WALL_SAVING_VS_EXACT_MS,
        "2k_prebound_host_saving_vs_exact_at_least_0_50ms": short.get(
            "prebound_vs_exact_host_saving_ms", -1e9
        )
        >= MIN_HOST_SAVING_VS_EXACT_MS,
        "256k_prebound_host_saving_vs_exact_at_least_0_50ms": long.get(
            "prebound_vs_exact_host_saving_ms", -1e9
        )
        >= MIN_HOST_SAVING_VS_EXACT_MS,
        "2k_prebinding_recovers_at_least_1_50ms_host_tax": short.get(
            "prebound_vs_unbound_host_saving_ms", -1e9
        )
        >= MIN_HOST_SAVING_VS_UNBOUND_MS,
        "256k_prebinding_recovers_at_least_1_50ms_host_tax": long.get(
            "prebound_vs_unbound_host_saving_ms", -1e9
        )
        >= MIN_HOST_SAVING_VS_UNBOUND_MS,
        "2k_to_256k_retention_at_least_0_90": bool(short_tps)
        and long_tps / short_tps >= MIN_CONTEXT_RETENTION,
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
            "production_kernel_abi": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    args = parser.parse_args()

    artifact = {
        "schema": "glm53-prebound-native-packed-moe-execution-plan-v1",
        "date": date.today().isoformat(),
        "complete": False,
        "accepted": False,
        "probe_only": True,
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "compact_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
        "contexts": {},
        "runtime_changes": {
            "runtime": False,
            "server": False,
            "apc": False,
            "cache_abi": False,
            "production_kernel_abi": False,
        },
        "boundary_change": {
            "kernels": False,
            "immutable_weight_binding": "once per layer",
            "pipeline_lookup": "once per layer",
            "steady_execute_inputs": ["x", "expert_ids", "scores"],
        },
    }
    try:
        artifact["source_evidence"] = _validate_sources()
        (
            index_plan_type,
            moe_plan_type,
            d99f,
            boundary,
            tier0,
            update_island,
            native_moe,
            residual,
        ) = _load_helpers()
        reference_arm, artifact["reference_composition"] = (
            native_moe._qualified_reference_arm(residual)
        )
        _progress("artificial_prebound_contract")
        artifact["artificial_prebound_contract"] = native_moe._artificial_contract(
            moe_plan_type, d99f, residual, tier0
        )
        if not artifact["artificial_prebound_contract"]["output_byte_exact"]:
            raise RuntimeError("prebound native artificial output is not exact")
        report = inspect_checkpoint(args.model, require_server_ready=True)
        artifact["checkpoint_fingerprint"] = report.fingerprint
        artifact["official_hf_revision"] = report.official_revision
        if report.fingerprint != artifact["source_evidence"]["exact_repair"][
            "checkpoint_fingerprint"
        ]:
            raise RuntimeError("probe checkpoint differs from exact repair")

        mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
        mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
        _progress("load_model")
        model, _ = load(
            args.model,
            experimental_packed_decode_moe=True,
            experimental_compact_nope_dsa_cache=True,
            compact_cache_capacity_tokens=max(CONTEXTS) + 64,
        )
        warm_residency(model)

        for context in CONTEXTS:
            _progress("decode_context", context=context)
            source = boundary._synthetic_cache(model, context, "compact-nope-dsa")
            artifact["contexts"][str(context)] = _context_case(
                model,
                source,
                context,
                index_plan_type=index_plan_type,
                moe_plan_type=moe_plan_type,
                boundary=boundary,
                tier0=tier0,
                update_island=update_island,
                native_moe=native_moe,
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
        artifact["failed_gates"] = [
            name for name, passed in artifact["acceptance"].items() if not passed
        ]
        artifact["decision"] = (
            "advance_prebound_exact_native_moe_to_production_design"
            if artifact["accepted"]
            else "stop_prebound_native_packed_moe_execution_plan"
        )
    except Exception as error:
        artifact.update(
            complete=True,
            accepted=False,
            decision="abort_prebound_native_packed_moe_execution_plan",
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
                "failed_gates": artifact.get("failed_gates", []),
            },
            indent=2,
        )
    )
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
