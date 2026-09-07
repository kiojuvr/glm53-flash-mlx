#!/usr/bin/env python3
"""Requalify the exact repaired native packed-MoE plan on performance only."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import sys
import tempfile
import traceback
from datetime import date
from pathlib import Path

import mlx.core as mx

from glm53_flash_mlx.abi import MLX_VLM_REVISION, NOPE_DSA_CACHE_ABI_COMPACT
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint


ROOT = Path(__file__).resolve().parents[1]
NATIVE_PACKAGE = ROOT / "native_execution"
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-exact-native-packed-moe-performance-requalification-20260907.json"
)
EXACT_REPAIR_ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-exact-native-routed-fast-sigmoid-repair-20260907.json"
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
    import probe_long_context_first_decode_boundary as boundary
    import probe_native_execution_engine_feasibility as tier0
    import probe_native_indexpool_update_submission_island as update_island
    import probe_native_packed_moe_execution_plan as native_moe
    import probe_residual_packed_decode_moe_fusion as residual

    return (
        NativeIndexPoolUpdateSelectionPlan,
        NativePackedMoEDecodePlan,
        boundary,
        tier0,
        update_island,
        native_moe,
        residual,
    )


def _qualified_exact_repair() -> dict:
    source = json.loads(EXACT_REPAIR_ARTIFACT.read_text())
    if not source["complete"] or not source["accepted"]:
        raise RuntimeError("native routed fast-BF16 repair is not accepted")
    if source["decision"] != "advance_exact_native_moe_to_performance_requalification":
        raise RuntimeError("native routed repair did not authorize performance requalification")
    if not all(source["acceptance"].values()):
        raise RuntimeError("native routed repair has a failed correctness gate")
    replay = source["owned_activation_replay"]
    if replay["exact_layer_step_comparisons"] != 2_856:
        raise RuntimeError("native routed repair did not close all 2,856 layer-steps")
    if replay["first_divergence"] is not None:
        raise RuntimeError("native routed repair still has a divergence")
    return {
        "artifact": str(EXACT_REPAIR_ARTIFACT.relative_to(ROOT)),
        "accepted": True,
        "checkpoint_fingerprint": source["checkpoint_fingerprint"],
        "formula": source["repair"]["formula"],
        "exact_layer_step_comparisons": replay["exact_layer_step_comparisons"],
        "official_16_128_oracle_exact_both_arms": source["acceptance"][
            "official_16_128_oracle_exact_both_arms"
        ],
    }


def _acceptance(artifact: dict) -> dict:
    short = artifact.get("contexts", {}).get("2048", {})
    long = artifact.get("contexts", {}).get("262144", {})
    cases = (short, long)
    short_timing = short.get("timing", {}).get("B_native_packed_moe_plan", {})
    long_timing = long.get("timing", {}).get("B_native_packed_moe_plan", {})
    short_tps = short_timing.get("tokens_per_second", 0.0)
    long_tps = long_timing.get("tokens_per_second", 0.0)
    plan_rows = [
        row for case in cases for row in case.get("native_moe_plan_evidence", [])
    ]
    return {
        "exact_fast_bf16_repair_source_accepted": artifact.get(
            "exact_repair_source", {}
        ).get("accepted", False),
        "official_16_128_oracle_exact_from_source": artifact.get(
            "exact_repair_source", {}
        ).get("official_16_128_oracle_exact_both_arms", False),
        "all_full_model_logits_tokens_and_state_byte_exact": bool(short)
        and bool(long)
        and all(
            case.get("all_full_vocab_logits_byte_exact")
            and case.get("all_generated_tokens_exact")
            and case.get("post_state_byte_exact")
            for case in cases
        ),
        "all_42_sparse_layers_use_fast_bf16_native_plan": bool(plan_rows)
        and all(
            case.get("native_moe_plan_count") == 42
            and case.get("native_moe_execution_count")
            == case.get("expected_sparse_moe_calls")
            for case in cases
        )
        and all(
            row["uses_shape_specialized_routed_gate_up"]
            and row["uses_fast_bf16_routed_sigmoid"]
            for row in plan_rows
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
        "fixed_arena_zero_execute_allocation_graph_shape_sync": bool(plan_rows)
        and all(
            row["buffer_identities_stable"]
            and row["dynamic_allocation_count"] == 0
            and row["graph_node_count"] == 0
            and row["shape_discovery_count"] == 0
            and row["host_synchronization_count"] == 0
            and row["returned_intermediate_tensor_bytes"] == 0
            for row in plan_rows
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
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    args = parser.parse_args()

    artifact = {
        "schema": "glm53-exact-native-packed-moe-performance-requalification-v1",
        "date": date.today().isoformat(),
        "complete": False,
        "accepted": False,
        "probe_only": True,
        "performance_only": True,
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "compact_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
        "contexts": {},
        "runtime_changes": {
            "runtime": False,
            "server": False,
            "apc": False,
            "cache_abi": False,
            "kernel_abi": False,
        },
        "gates": {
            "target_2k_tokens_per_second": TARGET_2K_TPS,
            "minimum_native_wall_saving_ms": MIN_NATIVE_WALL_SAVING_MS,
            "minimum_native_host_saving_ms": MIN_NATIVE_HOST_SAVING_MS,
            "minimum_2k_to_256k_retention": MIN_CONTEXT_RETENTION,
            "maximum_process_peak_bytes": MAX_PROCESS_PEAK_BYTES,
        },
    }
    try:
        artifact["exact_repair_source"] = _qualified_exact_repair()
        (
            index_plan_type,
            moe_plan_type,
            boundary,
            tier0,
            update_island,
            native_moe,
            residual,
        ) = _load_helpers()
        reference_arm, artifact["reference_composition"] = (
            native_moe._qualified_reference_arm(residual)
        )

        report = inspect_checkpoint(args.model, require_server_ready=True)
        artifact["checkpoint_fingerprint"] = report.fingerprint
        artifact["official_hf_revision"] = report.official_revision
        if report.fingerprint != artifact["exact_repair_source"][
            "checkpoint_fingerprint"
        ]:
            raise RuntimeError("performance model differs from exact repair checkpoint")

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
            artifact["contexts"][str(context)] = native_moe._context_case(
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
        artifact["failed_gates"] = [
            name for name, passed in artifact["acceptance"].items() if not passed
        ]
        artifact["decision"] = (
            "keep_exact_native_packed_moe_execution_plan"
            if artifact["accepted"]
            else "stop_exact_native_packed_moe_execution_plan_on_performance"
        )
    except Exception as error:
        artifact.update(
            complete=True,
            accepted=False,
            decision="abort_exact_native_packed_moe_performance_requalification",
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
