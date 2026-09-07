#!/usr/bin/env python3
"""Probe an exact lazy MLX primitive around the native packed-MoE plan."""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.metadata
import json
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
    / "m3ultra512-lazy-native-packed-moe-primitive-20260907.json"
)
PREBOUND_ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-prebound-native-packed-moe-execution-plan-20260907.json"
)
CONTEXTS = (2_048, 262_144)
TARGET_2K_TPS = 15.0
MIN_WALL_SAVING_VS_EXACT_MS = 0.50
MIN_HOST_SAVING_VS_EXACT_MS = 0.50
MIN_WALL_RECOVERY_VS_EAGER_MS = 1.00
MIN_HOST_RECOVERY_VS_EAGER_MS = 1.50
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
    import probe_prebound_native_packed_moe_execution_plan as prebound
    import probe_residual_packed_decode_moe_fusion as residual

    return (
        NativeIndexPoolUpdateSelectionPlan,
        NativePackedMoEDecodePlan,
        d99f,
        boundary,
        tier0,
        update_island,
        native_moe,
        prebound,
        residual,
    )


def _validate_source() -> dict:
    source = json.loads(PREBOUND_ARTIFACT.read_text())
    if source["decision"] != "stop_prebound_native_packed_moe_execution_plan":
        raise RuntimeError("lazy primitive requires the recorded prebound stop")
    if not source["acceptance"][
        "all_three_arms_logits_tokens_and_state_byte_exact"
    ]:
        raise RuntimeError("prebound source is not exact")
    if not source["acceptance"][
        "all_42_layers_prebound_once_and_execute_dynamic_only"
    ]:
        raise RuntimeError("prebound source did not close static binding")
    expected = {
        "2k_prebound_wall_saving_vs_exact_at_least_0_50ms",
        "256k_prebound_wall_saving_vs_exact_at_least_0_50ms",
        "2k_prebound_host_saving_vs_exact_at_least_0_50ms",
        "256k_prebound_host_saving_vs_exact_at_least_0_50ms",
        "2k_prebinding_recovers_at_least_1_50ms_host_tax",
        "256k_prebinding_recovers_at_least_1_50ms_host_tax",
    }
    if set(source["failed_gates"]) != expected:
        raise RuntimeError("prebound source failed outside the eager boundary")
    return {
        "artifact": str(PREBOUND_ARTIFACT.relative_to(ROOT)),
        "accepted": False,
        "checkpoint_fingerprint": source["checkpoint_fingerprint"],
        "failed_gates": source["failed_gates"],
        "eager_prebound_boundary": {
            context: {
                "wall_saving_vs_exact_ms": row[
                    "prebound_vs_exact_wall_saving_ms"
                ],
                "host_saving_vs_exact_ms": row[
                    "prebound_vs_exact_host_saving_ms"
                ],
            }
            for context, row in source["contexts"].items()
        },
    }


_ORIGINAL_PACKED_CALL = PackedFP8MoE.__call__


def _lazy_call(registry, moe, x):
    flat = mx.contiguous(x.reshape(-1, x.shape[-1]), allow_col_major=False)
    if int(flat.shape[0]) != 1 or moe.shared_experts is None:
        return _ORIGINAL_PACKED_CALL(moe, x)
    indices, scores = moe.gate(x)
    if int(indices.shape[-1]) != 8:
        return _ORIGINAL_PACKED_CALL(moe, x)
    input_row = flat[0]
    expert_ids = mx.contiguous(
        indices.reshape(-1).astype(mx.uint32), allow_col_major=False
    )
    flat_scores = mx.contiguous(scores.reshape(-1), allow_col_major=False)
    return registry.get(moe).execute_lazy(
        input_row, expert_ids, flat_scores
    ).reshape(x.shape)


@contextlib.contextmanager
def _lazy_native_arm(registry):
    previous = PackedFP8MoE.__call__

    def wrapped(moe, x):
        return _lazy_call(registry, moe, x)

    PackedFP8MoE.__call__ = wrapped
    try:
        yield
    finally:
        PackedFP8MoE.__call__ = previous


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
    prebound,
    residual,
    reference_arm,
    warmups,
    samples,
) -> dict:
    arms = (
        "A_exact_composition",
        "B_eager_prebound_native",
        "C_lazy_prebound_native",
    )
    caches = {
        arm: boundary._clone_cache(source, context + warmups + samples + 16)
        for arm in arms
    }
    index_registries = {
        arm: update_island._Registry(index_plan_type) for arm in arms
    }
    eager_registry = prebound._BoundRegistry(moe_plan_type)
    lazy_registry = prebound._BoundRegistry(moe_plan_type)
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
                    stack.enter_context(prebound._prebound_native_arm(eager_registry))
                else:
                    stack.enter_context(_lazy_native_arm(lazy_registry))
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

    timing = {arm: prebound._median(rows) for arm, rows in measured.items()}
    exact = timing[arms[0]]
    eager = timing[arms[1]]
    lazy = timing[arms[2]]
    eager_evidence = eager_registry.evidence()
    lazy_evidence = lazy_registry.evidence()
    result = {
        "context_tokens": context,
        "timing": timing,
        "lazy_vs_exact_wall_saving_ms": exact["median_wall_ms"]
        - lazy["median_wall_ms"],
        "lazy_vs_exact_host_saving_ms": exact["median_host_submit_ms"]
        - lazy["median_host_submit_ms"],
        "lazy_vs_eager_wall_saving_ms": eager["median_wall_ms"]
        - lazy["median_wall_ms"],
        "lazy_vs_eager_host_saving_ms": eager["median_host_submit_ms"]
        - lazy["median_host_submit_ms"],
        "lazy_speedup_vs_exact": exact["median_wall_ms"] / lazy["median_wall_ms"],
        "all_full_vocab_logits_byte_exact": len(set(map(tuple, hashes.values())))
        == 1,
        "all_generated_tokens_exact": len(set(map(tuple, token_ids.values())))
        == 1,
        "all_post_states_byte_exact": boundary._cache_exact(
            caches[arms[0]], caches[arms[1]]
        )
        and boundary._cache_exact(caches[arms[0]], caches[arms[2]]),
        "expected_sparse_moe_calls_per_arm": 42 * (warmups + samples),
        "eager_plan_count": len(eager_evidence),
        "lazy_plan_count": len(lazy_evidence),
        "eager_plan_evidence": eager_evidence,
        "lazy_plan_evidence": lazy_evidence,
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
    short_tps = short.get("timing", {}).get("C_lazy_prebound_native", {}).get(
        "tokens_per_second", 0.0
    )
    long_tps = long.get("timing", {}).get("C_lazy_prebound_native", {}).get(
        "tokens_per_second", 0.0
    )
    lazy_rows = [
        row for case in cases for row in case.get("lazy_plan_evidence", [])
    ]
    return {
        "artificial_lazy_native_output_byte_exact": artifact.get(
            "artificial_lazy_contract", {}
        ).get("output_byte_exact", False),
        "all_three_arms_logits_tokens_and_state_byte_exact": bool(short)
        and bool(long)
        and all(
            case.get("all_full_vocab_logits_byte_exact")
            and case.get("all_generated_tokens_exact")
            and case.get("all_post_states_byte_exact")
            for case in cases
        ),
        "all_42_layers_use_one_lazy_graph_node_per_execution": bool(lazy_rows)
        and all(
            case.get("lazy_plan_count") == 42
            and sum(row["execution_count"] for row in case["lazy_plan_evidence"])
            == case["expected_sparse_moe_calls_per_arm"]
            and sum(row["lazy_graph_count"] for row in case["lazy_plan_evidence"])
            == case["expected_sparse_moe_calls_per_arm"]
            for case in cases
        ),
        "2k_lazy_decode_at_least_15_tps": short_tps >= TARGET_2K_TPS,
        "2k_lazy_wall_saving_vs_exact_at_least_0_50ms": short.get(
            "lazy_vs_exact_wall_saving_ms", -1e9
        )
        >= MIN_WALL_SAVING_VS_EXACT_MS,
        "256k_lazy_wall_saving_vs_exact_at_least_0_50ms": long.get(
            "lazy_vs_exact_wall_saving_ms", -1e9
        )
        >= MIN_WALL_SAVING_VS_EXACT_MS,
        "2k_lazy_host_saving_vs_exact_at_least_0_50ms": short.get(
            "lazy_vs_exact_host_saving_ms", -1e9
        )
        >= MIN_HOST_SAVING_VS_EXACT_MS,
        "256k_lazy_host_saving_vs_exact_at_least_0_50ms": long.get(
            "lazy_vs_exact_host_saving_ms", -1e9
        )
        >= MIN_HOST_SAVING_VS_EXACT_MS,
        "2k_lazy_recovers_at_least_1ms_wall_from_eager": short.get(
            "lazy_vs_eager_wall_saving_ms", -1e9
        )
        >= MIN_WALL_RECOVERY_VS_EAGER_MS,
        "256k_lazy_recovers_at_least_1ms_wall_from_eager": long.get(
            "lazy_vs_eager_wall_saving_ms", -1e9
        )
        >= MIN_WALL_RECOVERY_VS_EAGER_MS,
        "2k_lazy_recovers_at_least_1_50ms_host_from_eager": short.get(
            "lazy_vs_eager_host_saving_ms", -1e9
        )
        >= MIN_HOST_RECOVERY_VS_EAGER_MS,
        "256k_lazy_recovers_at_least_1_50ms_host_from_eager": long.get(
            "lazy_vs_eager_host_saving_ms", -1e9
        )
        >= MIN_HOST_RECOVERY_VS_EAGER_MS,
        "2k_to_256k_retention_at_least_0_90": bool(short_tps)
        and long_tps / short_tps >= MIN_CONTEXT_RETENTION,
        "lazy_plans_keep_bound_resources_and_fixed_scratch": bool(lazy_rows)
        and all(
            row["weights_bound"]
            and row["bound_weight_identities_stable"]
            and row["bound_weight_handles_stable"]
            and row["scratch_buffer_identities_stable"]
            and row["static_input_validation_count"] == 10
            and row["pipeline_lookup_count"] == 6
            for row in lazy_rows
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
        "schema": "glm53-lazy-native-packed-moe-primitive-v1",
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
            "forced_async_eval_per_layer": False,
            "native_encode_phase": "MLX primitive eval_gpu",
            "lazy_inputs": ["x", "expert_ids", "scores"],
        },
    }
    try:
        artifact["source_evidence"] = _validate_source()
        (
            index_plan_type,
            moe_plan_type,
            d99f,
            boundary,
            tier0,
            update_island,
            native_moe,
            prebound,
            residual,
        ) = _load_helpers()
        reference_arm, artifact["reference_composition"] = (
            native_moe._qualified_reference_arm(residual)
        )
        _progress("artificial_lazy_contract")
        artifact["artificial_lazy_contract"] = native_moe._artificial_contract(
            moe_plan_type, d99f, residual, tier0
        )
        if not artifact["artificial_lazy_contract"]["output_byte_exact"]:
            raise RuntimeError("lazy native artificial output is not exact")

        report = inspect_checkpoint(args.model, require_server_ready=True)
        artifact["checkpoint_fingerprint"] = report.fingerprint
        artifact["official_hf_revision"] = report.official_revision
        if report.fingerprint != artifact["source_evidence"][
            "checkpoint_fingerprint"
        ]:
            raise RuntimeError("probe checkpoint differs from prebound source")

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
                prebound=prebound,
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
            "advance_lazy_exact_native_moe_primitive_to_production_design"
            if artifact["accepted"]
            else "stop_lazy_native_packed_moe_primitive"
        )
    except Exception as error:
        artifact.update(
            complete=True,
            accepted=False,
            decision="abort_lazy_native_packed_moe_primitive",
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
