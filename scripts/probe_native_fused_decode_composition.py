#!/usr/bin/env python3
"""Compose the exact native IndexPool island with exact fused packed MoE.

This probe does not add another native boundary.  It combines the accepted
native IndexPool-update/Tier-1 selection island with the previously qualified
nonproduction routed gate+up+SwiGLU, exact B1 weighted reduction, and shared
gate+up+SwiGLU kernels.  The production packed-decode path is the oracle.
"""

from __future__ import annotations

import argparse
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


ROOT = Path(__file__).resolve().parents[1]
NATIVE_PACKAGE = ROOT / "native_execution"
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-fused-decode-composition-20260907.json"
)
RESIDUAL_ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-residual-packed-decode-moe-fusion-20260901.json"
)
CONTEXTS = (2_048, 262_144)
TARGET_2K_TPS = 15.0
TARGET_2K_MS = 1_000.0 / TARGET_2K_TPS
MIN_256K_SAVING_MS = 5.0
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
    from glm53_native_execution import NativeIndexPoolUpdateSelectionPlan
    import probe_exact_sigmoid_gate_metal_barrier as oracle_probe
    import probe_long_context_first_decode_boundary as boundary
    import probe_native_execution_engine_feasibility as tier0
    import probe_native_indexpool_update_submission_island as update_island
    import probe_residual_packed_decode_moe_fusion as residual

    return (
        NativeIndexPoolUpdateSelectionPlan,
        oracle_probe,
        boundary,
        tier0,
        update_island,
        residual,
    )


def _qualified_fused_arm(residual):
    source = json.loads(RESIDUAL_ARTIFACT.read_text())
    aggregation = source["selected_aggregation"]
    if aggregation != "B1":
        raise RuntimeError(f"unexpected exact aggregation oracle: {aggregation}")
    if not source["aggregation_exact"].get(aggregation) or not source["shared_exact"]:
        raise RuntimeError("residual packed MoE source artifact is not exact")
    if not all(source["correctness"].values()):
        raise RuntimeError("residual packed MoE qualification is incomplete")
    return residual.Arm(aggregation, True), {
        "artifact": str(RESIDUAL_ARTIFACT.relative_to(ROOT)),
        "selected_aggregation": aggregation,
        "all_recorded_correctness_gates": True,
        "decode_steps": 4_096,
    }


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


def _run_model(
    model,
    cache,
    token,
    *,
    update_island,
    registry,
    residual,
    fused_arm,
    fused: bool,
):
    with ExitStack() as stack:
        if fused:
            stack.enter_context(residual._runtime(fused_arm))
        stack.enter_context(update_island._native_arm(registry, True))
        output = model(token, cache=cache)
        fused_calls = residual._FUSED_CALL_COUNT if fused else 0
    return output, fused_calls


def _context_case(
    model,
    source,
    context: int,
    *,
    plan_type,
    boundary,
    tier0,
    update_island,
    residual,
    fused_arm,
    warmups: int,
    samples: int,
) -> dict:
    arms = (
        "A_native_indexpool_production_packed",
        "B_native_indexpool_exact_fused_packed",
    )
    caches = {
        arm: boundary._clone_cache(source, context + warmups + samples + 16)
        for arm in arms
    }
    registries = {arm: update_island._Registry(plan_type) for arm in arms}
    measured = {arm: [] for arm in arms}
    hashes = {arm: [] for arm in arms}
    token_ids = {arm: [] for arm in arms}
    fused_call_count = 0
    token = mx.array([[3_000]], dtype=mx.uint32)

    for iteration in range(warmups + samples):
        order = arms if iteration % 2 == 0 else tuple(reversed(arms))
        for arm in order:
            started = time.perf_counter_ns()
            output, calls = _run_model(
                model,
                caches[arm],
                token,
                update_island=update_island,
                registry=registries[arm],
                residual=residual,
                fused_arm=fused_arm,
                fused=arm == arms[1],
            )
            submitted = time.perf_counter_ns()
            _eval(output.logits)
            finished = time.perf_counter_ns()
            hashes[arm].append(tier0._hash(output.logits[0, -1]))
            token_ids[arm].append(int(mx.argmax(output.logits[0, -1]).item()))
            fused_call_count += calls
            if iteration >= warmups:
                measured[arm].append(
                    {
                        "host_submit_ms": (submitted - started) / 1e6,
                        "wall_ms": (finished - started) / 1e6,
                    }
                )

    timing = {arm: _median(rows) for arm, rows in measured.items()}
    baseline_ms = timing[arms[0]]["median_wall_ms"]
    candidate_ms = timing[arms[1]]["median_wall_ms"]
    result = {
        "context_tokens": context,
        "timing": timing,
        "candidate_wall_saving_ms": baseline_ms - candidate_ms,
        "candidate_speedup": baseline_ms / candidate_ms,
        "all_full_vocab_logits_byte_exact": hashes[arms[0]] == hashes[arms[1]],
        "all_generated_tokens_exact": token_ids[arms[0]] == token_ids[arms[1]],
        "post_state_byte_exact": boundary._cache_exact(
            caches[arms[0]], caches[arms[1]]
        ),
        "fused_sparse_moe_calls": fused_call_count,
        "expected_fused_sparse_moe_calls": 42 * (warmups + samples),
        "native_plan_evidence": {
            arm: registries[arm].evidence() for arm in arms
        },
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
    evidence = [
        plan
        for case in cases
        for plans in case.get("native_plan_evidence", {}).values()
        for plan in plans
    ]
    short_candidate = short.get("timing", {}).get(
        "B_native_indexpool_exact_fused_packed", {}
    )
    long_candidate = long.get("timing", {}).get(
        "B_native_indexpool_exact_fused_packed", {}
    )
    short_tps = short_candidate.get("tokens_per_second", 0.0)
    long_tps = long_candidate.get("tokens_per_second", 0.0)
    return {
        "2k_decode_at_least_15_tps": short_tps >= TARGET_2K_TPS,
        "2k_median_at_most_66_667ms": short_candidate.get(
            "median_wall_ms", 1e9
        )
        <= TARGET_2K_MS,
        "256k_incremental_saving_at_least_5ms": long.get(
            "candidate_wall_saving_ms", -1e9
        )
        >= MIN_256K_SAVING_MS,
        "2k_to_256k_retention_at_least_0_90": bool(short_tps)
        and long_tps / short_tps >= MIN_CONTEXT_RETENTION,
        "all_logits_tokens_and_state_byte_exact": bool(short)
        and bool(long)
        and all(
            case.get("all_full_vocab_logits_byte_exact")
            and case.get("all_generated_tokens_exact")
            and case.get("post_state_byte_exact")
            for case in cases
        ),
        "all_42_sparse_layers_use_fused_path": all(
            case.get("fused_sparse_moe_calls")
            == case.get("expected_fused_sparse_moe_calls")
            for case in cases
        ),
        "fixed_arena_zero_execute_allocation_graph_shape_sync": bool(evidence)
        and all(
            row.get("buffer_identities_stable")
            and row.get("dynamic_allocation_count") == 0
            and row.get("graph_node_count") == 0
            and row.get("shape_discovery_count") == 0
            and row.get("host_synchronization_count") == 0
            and row.get("returned_intermediate_tensor_bytes") == 0
            for row in evidence
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
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    args = parser.parse_args()

    artifact = {
        "schema": "glm53-native-fused-decode-composition-v1",
        "date": date.today().isoformat(),
        "complete": False,
        "accepted": False,
        "probe_only": True,
        "exact_nonproduction_composition": True,
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
            plan_type,
            oracle_probe,
            boundary,
            tier0,
            update_island,
            residual,
        ) = _load_helpers()
        fused_arm, artifact["fused_moe_source"] = _qualified_fused_arm(residual)
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

        _progress("official_oracle", arm="production-packed")
        artifact["official_oracle"] = {
            "A_production_packed": oracle_probe._official_oracle(
                model, processor, report
            )
        }
        _progress("official_oracle", arm="exact-fused-packed")
        with residual._runtime(fused_arm):
            artifact["official_oracle"]["B_exact_fused_packed"] = (
                oracle_probe._official_oracle(model, processor, report)
            )

        artifact["contexts"] = {}
        for context in CONTEXTS:
            _progress("decode_context", context=context)
            source = boundary._synthetic_cache(model, context, "compact-nope-dsa")
            artifact["contexts"][str(context)] = _context_case(
                model,
                source,
                context,
                plan_type=plan_type,
                boundary=boundary,
                tier0=tier0,
                update_island=update_island,
                residual=residual,
                fused_arm=fused_arm,
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
            "advance_fused_moe_topology_into_native_executor"
            if artifact["accepted"]
            else "stop_or_redesign_native_fused_decode_composition"
        )
    except Exception as error:
        artifact.update(
            complete=True,
            accepted=False,
            decision="native_fused_decode_composition_probe_failed",
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
