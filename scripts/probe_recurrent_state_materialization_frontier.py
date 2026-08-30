#!/usr/bin/env python3
"""Characterize recurrent-state materialization intervals on M3 Ultra."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
import time
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from glm53_flash_mlx.abi import (
    KERNEL_ABI_VERSION,
    MLX_VLM_REVISION,
    NOPE_DSA_CACHE_ABI_COMPACT,
)
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint

INTERVALS = (50, 0, 128, 256, 512)
DEFAULT_STEPS = 8192
RESERVE_TAIL = 16
TELEMETRY_INTERVAL = 256
MEMORY_BASELINE_STEP = 19
MAX_ACTIVE_DRIFT = 64 * 2**20
MAX_INTERVAL_256_REGRESSION = 0.02


def materialize_cache(cache) -> None:
    mx.eval([entry.state for entry in cache])
    mx.clear_cache()


def hash_steps(steps: int) -> tuple[int, ...]:
    boundaries = {
        1,
        49,
        50,
        51,
        127,
        128,
        129,
        255,
        256,
        257,
        511,
        512,
        513,
        steps - 1,
        steps,
    }
    boundaries.update(range(TELEMETRY_INTERVAL, steps + 1, TELEMETRY_INTERVAL))
    return tuple(sorted(step for step in boundaries if 1 <= step <= steps))


def expected_materialization_count(steps: int, interval: int) -> int:
    return 0 if interval == 0 else steps // interval


def _state_leaves(cache):
    flattened = tree_flatten([entry.state for entry in cache])
    arrays = [value for _, value in flattened if isinstance(value, mx.array)]
    return len(flattened), len(arrays)


def _cache_nbytes(cache) -> int:
    return sum(int(entry.nbytes) for entry in cache)


def _logits_evidence(logits) -> tuple[str, int]:
    values = np.ascontiguousarray(np.asarray(logits.astype(mx.float32)))
    return hashlib.sha256(values.tobytes()).hexdigest(), int(np.isnan(values).sum())


def _percentile(values, percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _write_artifact(path: Path, artifact: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n")


def _release(cache) -> None:
    if isinstance(cache, list):
        cache.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def _warm_model(model) -> None:
    cache = model.make_cache()
    output = model(mx.array([[1]], dtype=mx.uint32), cache=cache)
    mx.eval(output.logits)
    mx.synchronize()
    del output
    _release(cache)


def run_arm(model, *, interval: int, steps: int, teacher_tokens=None) -> dict:
    cache = model.make_cache()
    checkpoints = set(hash_steps(steps))
    latencies = []
    materializations = []
    telemetry = []
    logits_hashes = {}
    nan_count = 0
    token_trace = []
    input_token = 1
    baseline_active = None
    baseline_nbytes = None
    baseline_leaf_counts = None
    max_active_after_baseline = 0
    min_active_after_baseline = None
    mx.reset_peak_memory()

    for step in range(1, steps + 1):
        started = time.perf_counter()
        output = model(mx.array([[input_token]], dtype=mx.uint32), cache=cache)
        logits = output.logits[0, -1]
        predicted = int(mx.argmax(logits).item())
        mx.synchronize()
        latencies.append(time.perf_counter() - started)

        if teacher_tokens is None:
            next_token = predicted
            token_trace.append(predicted)
        else:
            next_token = int(teacher_tokens[step - 1])
            token_trace.append(next_token)

        if step in checkpoints:
            logits_hash, nans = _logits_evidence(logits)
            logits_hashes[str(step)] = logits_hash
            nan_count += nans

        materialization_seconds = None
        if interval > 0 and step % interval == 0:
            materialization_started = time.perf_counter()
            materialize_cache(cache)
            mx.synchronize()
            materialization_seconds = time.perf_counter() - materialization_started
            materializations.append(
                {"step": step, "seconds": materialization_seconds}
            )

        if step == MEMORY_BASELINE_STEP:
            mx.synchronize()
            baseline_active = int(mx.get_active_memory())
            baseline_nbytes = _cache_nbytes(cache)
            baseline_leaf_counts = _state_leaves(cache)
            max_active_after_baseline = baseline_active
            min_active_after_baseline = baseline_active
            mx.reset_peak_memory()

        if step % TELEMETRY_INTERVAL == 0:
            mx.synchronize()
            active = int(mx.get_active_memory())
            cache_memory = int(mx.get_cache_memory())
            peak = int(mx.get_peak_memory())
            nbytes = _cache_nbytes(cache)
            total_leaves, array_leaves = _state_leaves(cache)
            window = latencies[-TELEMETRY_INTERVAL:]
            max_active_after_baseline = max(max_active_after_baseline, active)
            min_active_after_baseline = min(min_active_after_baseline, active)
            row = {
                "step": step,
                "active_memory_bytes": active,
                "cache_memory_bytes": cache_memory,
                "peak_memory_bytes": peak,
                "authoritative_cache_bytes": nbytes,
                "cache_state_leaf_count": total_leaves,
                "cache_state_array_leaf_count": array_leaves,
                "decode_window_p50_ms": statistics.median(window) * 1000.0,
                "decode_window_p95_ms": _percentile(window, 95) * 1000.0,
                "last_materialization_ms": (
                    None
                    if materialization_seconds is None
                    else materialization_seconds * 1000.0
                ),
                "materialization_count": len(materializations),
                "tokens_since_materialization": (
                    step if interval == 0 else step % interval
                ),
                "logits_hash": logits_hashes.get(str(step)),
                "nan_count": nan_count,
            }
            telemetry.append(row)
            print(json.dumps({"interval": interval, **row}), flush=True)

        input_token = next_token
        del output, logits

    mx.synchronize()
    final_active = int(mx.get_active_memory())
    final_nbytes = _cache_nbytes(cache)
    warm_latencies = latencies[TELEMETRY_INTERVAL:]
    leaf_counts = {
        (row["cache_state_leaf_count"], row["cache_state_array_leaf_count"])
        for row in telemetry
    }
    arm = {
        "interval": interval,
        "steps": steps,
        "completed": True,
        "token_trace": token_trace if interval == 50 else None,
        "logits_hashes": logits_hashes,
        "nan_count": nan_count,
        "telemetry": telemetry,
        "decode_latency": {
            "warm_excludes_first_tokens": TELEMETRY_INTERVAL,
            "warm_median_ms": statistics.median(warm_latencies) * 1000.0,
            "warm_p95_ms": _percentile(warm_latencies, 95) * 1000.0,
        },
        "materializations": materializations,
        "materialization_count": len(materializations),
        "expected_materialization_count": expected_materialization_count(
            steps, interval
        ),
        "materialization_total_seconds": sum(
            row["seconds"] for row in materializations
        ),
        "materialization_median_ms": (
            statistics.median(row["seconds"] for row in materializations) * 1000.0
            if materializations
            else 0.0
        ),
        "memory_baseline_step": MEMORY_BASELINE_STEP,
        "baseline_active_memory_bytes": baseline_active,
        "final_active_memory_bytes": final_active,
        "active_memory_positive_drift_bytes": max_active_after_baseline
        - baseline_active,
        "active_memory_range_bytes": max_active_after_baseline
        - min_active_after_baseline,
        "baseline_authoritative_cache_bytes": baseline_nbytes,
        "final_authoritative_cache_bytes": final_nbytes,
        "authoritative_cache_drift_bytes": final_nbytes - baseline_nbytes,
        "baseline_cache_state_leaf_count": baseline_leaf_counts[0],
        "baseline_cache_state_array_leaf_count": baseline_leaf_counts[1],
        "telemetry_leaf_counts": [list(values) for values in sorted(leaf_counts)],
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "metal_buffer_count_api_available": False,
        "metal_error": None,
    }
    _release(cache)
    return arm


def _finalize(artifact: dict) -> None:
    steps = int(artifact["steps"])
    arms = artifact["arms"]
    stable = [str(interval) for interval in (50, 128, 256, 512)]
    reference_hashes = arms["50"]["logits_hashes"]
    hash_parity = all(arms[key]["logits_hashes"] == reference_hashes for key in stable)
    all_completed = all(arms[str(interval)]["completed"] for interval in INTERVALS)
    leaf_counts_constant = all(
        len(arms[str(interval)]["telemetry_leaf_counts"]) == 1
        and arms[str(interval)]["telemetry_leaf_counts"][0]
        == [
            arms[str(interval)]["baseline_cache_state_leaf_count"],
            arms[str(interval)]["baseline_cache_state_array_leaf_count"],
        ]
        for interval in INTERVALS
    )
    active_drift_ok = all(
        arms[str(interval)]["active_memory_positive_drift_bytes"]
        <= MAX_ACTIVE_DRIFT
        for interval in INTERVALS
    )
    count_ok = all(
        arms[str(interval)]["materialization_count"]
        == expected_materialization_count(steps, interval)
        for interval in INTERVALS
    )
    no_nan = all(arms[str(interval)]["nan_count"] == 0 for interval in INTERVALS)
    latency_50 = arms["50"]["decode_latency"]["warm_median_ms"]
    latency_256 = arms["256"]["decode_latency"]["warm_median_ms"]
    regression = latency_256 / latency_50 - 1.0
    acceptance = {
        "all_arms_completed_8192_steps": all_completed and steps == DEFAULT_STEPS,
        "stable_interval_logits_hashes_identical": hash_parity,
        "cache_state_leaf_count_constant": leaf_counts_constant,
        "all_arm_active_memory_drift_at_most_64_mib": active_drift_ok,
        "interval_256_warm_decode_regression_at_most_2_percent": (
            regression <= MAX_INTERVAL_256_REGRESSION
        ),
        "materialization_counts_exact": count_ok,
        "no_nan_or_metal_error": no_nan
        and all(arms[str(interval)]["metal_error"] is None for interval in INTERVALS),
    }
    interval_256_candidate = all(
        acceptance[key]
        for key in (
            "all_arms_completed_8192_steps",
            "stable_interval_logits_hashes_identical",
            "cache_state_leaf_count_constant",
            "all_arm_active_memory_drift_at_most_64_mib",
            "interval_256_warm_decode_regression_at_most_2_percent",
            "materialization_counts_exact",
            "no_nan_or_metal_error",
        )
    )
    artifact["comparison"] = {
        "interval_256_vs_50_warm_decode_regression": regression,
        "interval_0_is_comparison_only": True,
        "interval_512_not_eligible_from_this_screen": True,
    }
    artifact["acceptance"] = {
        **acceptance,
        "interval_256_selected_for_100k_soak": interval_256_candidate,
        "accepted": interval_256_candidate,
    }
    artifact["complete"] = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.steps < 513:
        raise ValueError("materialization frontier requires at least 513 steps")

    mx.set_wired_limit(int(440e9))
    mx.set_cache_limit(int(32e9))
    report = inspect_checkpoint(args.model, require_server_ready=True)
    artifact = {
        "schema": "glm53-recurrent-state-materialization-frontier-v1",
        "date": date.today().isoformat(),
        "machine": platform.machine(),
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "kernel_abi": KERNEL_ABI_VERSION,
        "cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
        "backend": "compact-nope-dsa+direct-moe",
        "steps": args.steps,
        "reserve_tokens": args.steps + RESERVE_TAIL,
        "telemetry_interval": TELEMETRY_INTERVAL,
        "hash_steps": list(hash_steps(args.steps)),
        "materialization_operation": (
            "mx.eval([entry.state for entry in cache]); mx.clear_cache()"
        ),
        "metal_buffer_count_api_available": False,
        "arms": {},
        "complete": False,
    }
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if (
            previous.get("schema") == artifact["schema"]
            and previous.get("checkpoint_fingerprint") == report.fingerprint
            and previous.get("steps") == args.steps
        ):
            artifact = previous

    model, _ = load(
        args.model,
        experimental_compact_nope_dsa_cache=True,
        compact_cache_reserve_tokens=args.steps + RESERVE_TAIL,
    )
    warm_residency(model)
    _warm_model(model)

    teacher_tokens = artifact.get("arms", {}).get("50", {}).get("token_trace")
    for interval in INTERVALS:
        key = str(interval)
        if artifact.get("arms", {}).get(key, {}).get("completed"):
            if interval == 50:
                teacher_tokens = artifact["arms"][key]["token_trace"]
            continue
        if interval != 50 and teacher_tokens is None:
            raise RuntimeError("interval 50 greedy trace must complete first")
        print(json.dumps({"phase": "arm_start", "interval": interval}), flush=True)
        try:
            arm = run_arm(
                model,
                interval=interval,
                steps=args.steps,
                teacher_tokens=None if interval == 50 else teacher_tokens,
            )
        except Exception as exc:
            artifact["arms"][key] = {
                "interval": interval,
                "steps": args.steps,
                "completed": False,
                "metal_error": f"{type(exc).__name__}: {exc}",
            }
            _write_artifact(args.output, artifact)
            raise
        artifact["arms"][key] = arm
        if interval == 50:
            teacher_tokens = arm["token_trace"]
        _write_artifact(args.output, artifact)
        print(
            json.dumps(
                {
                    "phase": "arm_complete",
                    "interval": interval,
                    "warm_median_ms": arm["decode_latency"]["warm_median_ms"],
                    "active_drift_bytes": arm["active_memory_positive_drift_bytes"],
                }
            ),
            flush=True,
        )

    _finalize(artifact)
    _write_artifact(args.output, artifact)
    print(json.dumps({"phase": "result", **artifact["acceptance"]}), flush=True)
    return 0 if artifact["acceptance"]["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
