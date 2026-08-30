#!/usr/bin/env python3
"""Soak the compact GLM-5.3 recurrent cache for 100,000 decode steps."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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

DEFAULT_STEPS = 100_000
RESERVE_TAIL = 16
MATERIALIZATION_INTERVAL = 256
TELEMETRY_INTERVAL = 256
MILESTONES = (25_000, 50_000, 75_000, 100_000)
REFERENCE_STEPS = 8_192
EARLY_WARM_START = 257
EARLY_WARM_END = 10_256
LATE_WINDOW_START = 90_001
MAX_ACTIVE_DRIFT = 64 * 2**20
MAX_MATERIALIZATION_P95_MS = 10.0
MAX_PEAK_MEMORY = 340_000_000_000
MIN_LATE_RETENTION = 0.90
EXPECTED_STATE_LEAVES = 167


class MilestoneGateError(RuntimeError):
    """Raised after atomically recording a failed 25k gate."""


def materialize_cache(cache) -> None:
    mx.eval([entry.state for entry in cache])
    mx.clear_cache()
    mx.synchronize()


def expected_materialization_count(steps: int) -> int:
    return steps // MATERIALIZATION_INTERVAL


def milestone_steps(steps: int) -> tuple[int, ...]:
    values = {step for step in MILESTONES if step <= steps}
    values.add(steps)
    return tuple(sorted(values))


def evidence_steps(steps: int, reference_hashes: dict[str, str]) -> tuple[int, ...]:
    values = {int(step) for step in reference_hashes if int(step) <= steps}
    values.update(range(4_096, steps + 1, 4_096))
    values.update(milestone_steps(steps))
    return tuple(sorted(values))


def token_digest(tokens) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(int(token).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def _state_leaves(cache) -> tuple[int, int]:
    flattened = tree_flatten([entry.state for entry in cache])
    arrays = [value for _, value in flattened if isinstance(value, mx.array)]
    return len(flattened), len(arrays)


def _cache_nbytes(cache) -> int:
    return sum(int(entry.nbytes) for entry in cache)


def _memory_snapshot() -> dict[str, int]:
    mx.synchronize()
    return {
        "active_memory_bytes": int(mx.get_active_memory()),
        "cache_memory_bytes": int(mx.get_cache_memory()),
        "peak_memory_bytes": int(mx.get_peak_memory()),
    }


def _logits_evidence(logits) -> tuple[str, int]:
    values = np.ascontiguousarray(np.asarray(logits.astype(mx.float32)))
    return hashlib.sha256(values.tobytes()).hexdigest(), int(np.isnan(values).sum())


def _percentile_ms(values, percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) * 1000.0


def _latency_summary(values) -> dict[str, float]:
    return {
        "p50_ms": statistics.median(values) * 1000.0,
        "p95_ms": _percentile_ms(values, 95),
    }


def _atomic_write_artifact(path: Path, artifact: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(artifact, indent=2) + "\n")
    temporary.replace(path)


def _release(cache) -> None:
    if isinstance(cache, list):
        cache.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def _warm_model(model) -> None:
    cache = model.make_cache()
    for token in (1, 2, 3, 4):
        output = model(mx.array([[token]], dtype=mx.uint32), cache=cache)
        mx.eval(output.logits)
        mx.synchronize()
    del output
    _release(cache)


def _load_reference(path: Path) -> dict:
    reference = json.loads(path.read_text())
    if reference.get("schema") != "glm53-recurrent-state-materialization-frontier-v1":
        raise ValueError("unexpected recurrent-state frontier reference schema")
    arm = reference.get("arms", {}).get("256", {})
    tokens = reference.get("arms", {}).get("50", {}).get("token_trace")
    if not arm.get("completed") or not isinstance(tokens, list):
        raise ValueError("reference interval-256 arm or greedy token trace is incomplete")
    if len(tokens) != REFERENCE_STEPS:
        raise ValueError("reference greedy token trace must contain exactly 8,192 tokens")
    return reference


def _reference_descriptor(reference: dict, path: Path) -> dict:
    tokens = reference["arms"]["50"]["token_trace"]
    return {
        "path": str(path),
        "commit": "4006e9d",
        "checkpoint_fingerprint": reference["checkpoint_fingerprint"],
        "interval": 256,
        "steps": REFERENCE_STEPS,
        "token_sha256": token_digest(tokens),
        "logits_hashes": reference["arms"]["256"]["logits_hashes"],
    }


def _post_materialization_drift(boundaries: list[dict]) -> int:
    if not boundaries:
        return 0
    baseline = boundaries[0]["after"]["active_memory_bytes"]
    return max(
        0,
        max(row["after"]["active_memory_bytes"] for row in boundaries) - baseline,
    )


def _cache_memory_strictly_increasing(boundaries: list[dict]) -> bool:
    values = [row["after"]["cache_memory_bytes"] for row in boundaries]
    return len(values) > 1 and all(right > left for left, right in zip(values, values[1:]))


def _reference_prefix_matches(artifact: dict) -> bool:
    reference = artifact["reference_8192"]
    try:
        observed = {
            step: artifact["checkpoint_hashes"][step]["logits_sha256"]
            for step in reference["logits_hashes"]
        }
        token_hash = artifact["checkpoint_hashes"][str(REFERENCE_STEPS)][
            "token_sha256"
        ]
    except KeyError:
        return False
    return observed == reference["logits_hashes"] and token_hash == reference[
        "token_sha256"
    ]


def _milestone_stop_failures(artifact: dict) -> list[str]:
    boundaries = artifact["boundary_telemetry"]
    failures = []
    if artifact.get("metal_error") is not None:
        failures.append("metal_resource_error")
    if artifact["nan_count"] != 0:
        failures.append("nan_detected")
    if any(row["state_leaf_count"] != EXPECTED_STATE_LEAVES for row in boundaries):
        failures.append("state_leaf_count_changed")
    if artifact["scheduled_materialization_count"] != expected_materialization_count(
        artifact["last_completed_step"]
    ):
        failures.append("materialization_count_mismatch")
    if _post_materialization_drift(boundaries) > MAX_ACTIVE_DRIFT:
        failures.append("post_materialization_active_drift_over_64_mib")
    if _cache_memory_strictly_increasing(boundaries):
        failures.append("post_materialization_cache_memory_monotonic_increase")
    if max(row["after"]["peak_memory_bytes"] for row in boundaries) > MAX_PEAK_MEMORY:
        failures.append("peak_memory_over_340_gb")
    if artifact["last_completed_step"] >= REFERENCE_STEPS and not _reference_prefix_matches(
        artifact
    ):
        failures.append("first_8192_reference_mismatch")
    return failures


def _record_milestone(
    artifact: dict,
    *,
    step: int,
    cache,
    rolling_digest,
    latencies: list[float],
) -> dict:
    snapshot = _memory_snapshot()
    leaves, array_leaves = _state_leaves(cache)
    row = {
        "step": step,
        "token_sha256": rolling_digest.copy().hexdigest(),
        "logits_sha256": artifact["checkpoint_hashes"][str(step)]["logits_sha256"],
        "scheduled_materialization_count": artifact["scheduled_materialization_count"],
        "expected_materialization_count": expected_materialization_count(step),
        "tokens_since_materialization": step % MATERIALIZATION_INTERVAL,
        "state_leaf_count": leaves,
        "state_array_leaf_count": array_leaves,
        "authoritative_cache_bytes": _cache_nbytes(cache),
        "memory": snapshot,
        "decode_last_256": _latency_summary(latencies[-TELEMETRY_INTERVAL:]),
    }
    artifact["milestones"][str(step)] = row
    return row


def _finalize(artifact: dict, latencies: list[float]) -> None:
    boundaries = artifact["boundary_telemetry"]
    materialization_ms = [row["materialization_ms"] for row in boundaries]
    early = latencies[EARLY_WARM_START - 1 : EARLY_WARM_END]
    late = latencies[LATE_WINDOW_START - 1 : DEFAULT_STEPS]
    early_median = statistics.median(early) * 1000.0
    late_median = statistics.median(late) * 1000.0
    retention = early_median / late_median
    final_evidence = artifact["final_evidence_materialization"]
    last_boundary = boundaries[-1]["after"]
    final_after = final_evidence["after"]
    final_returns_to_band = (
        abs(final_after["active_memory_bytes"] - last_boundary["active_memory_bytes"])
        <= MAX_ACTIVE_DRIFT
        and abs(final_after["cache_memory_bytes"] - last_boundary["cache_memory_bytes"])
        <= MAX_ACTIVE_DRIFT
    )
    reference_match = _reference_prefix_matches(artifact)
    peak = max(
        row[phase]["peak_memory_bytes"]
        for row in boundaries
        for phase in ("before", "after")
    )
    acceptance = {
        "completed_100000_tokens": artifact["last_completed_step"] == DEFAULT_STEPS,
        "scheduled_materialization_count_390": (
            artifact["scheduled_materialization_count"]
            == expected_materialization_count(DEFAULT_STEPS)
            == 390
        ),
        "state_leaf_count_167_constant": all(
            row["state_leaf_count"] == EXPECTED_STATE_LEAVES
            and row["state_array_leaf_count"] == EXPECTED_STATE_LEAVES
            for row in boundaries
        ),
        "no_nan_or_metal_error": (
            artifact["nan_count"] == 0 and artifact["metal_error"] is None
        ),
        "post_materialization_active_drift_at_most_64_mib": (
            _post_materialization_drift(boundaries) <= MAX_ACTIVE_DRIFT
        ),
        "final_evidence_materialization_returns_to_boundary_band": final_returns_to_band,
        "late_decode_retention_at_least_0_90": retention >= MIN_LATE_RETENTION,
        "materialization_p95_at_most_10_ms": (
            float(np.percentile(materialization_ms, 95))
            <= MAX_MATERIALIZATION_P95_MS
        ),
        "peak_memory_at_most_340_gb": peak <= MAX_PEAK_MEMORY,
        "first_8192_matches_4006e9d": reference_match,
    }
    artifact["summary"] = {
        "scheduled_materialization_count": artifact["scheduled_materialization_count"],
        "scheduled_materialization_total_ms": sum(materialization_ms),
        "scheduled_materialization_median_ms": statistics.median(materialization_ms),
        "scheduled_materialization_p95_ms": float(np.percentile(materialization_ms, 95)),
        "amortized_materialization_ms_per_token": sum(materialization_ms)
        / DEFAULT_STEPS,
        "post_materialization_active_drift_bytes": _post_materialization_drift(
            boundaries
        ),
        "post_materialization_cache_memory_strictly_increasing": (
            _cache_memory_strictly_increasing(boundaries)
        ),
        "state_leaf_counts": sorted({row["state_leaf_count"] for row in boundaries}),
        "state_array_leaf_counts": sorted(
            {row["state_array_leaf_count"] for row in boundaries}
        ),
        "early_warm_window": {
            "start": EARLY_WARM_START,
            "end": EARLY_WARM_END,
            **_latency_summary(early),
        },
        "late_window": {
            "start": LATE_WINDOW_START,
            "end": DEFAULT_STEPS,
            **_latency_summary(late),
        },
        "late_decode_retention": retention,
        "peak_memory_bytes": peak,
        "final_token_sha256": artifact["rolling_token_sha256"],
    }
    artifact["acceptance"] = {**acceptance, "accepted": all(acceptance.values())}
    artifact["complete"] = True


def run_soak(model, artifact: dict, output_path: Path) -> None:
    steps = int(artifact["steps"])
    reference_hashes = artifact["reference_8192"]["logits_hashes"]
    checkpoints = set(evidence_steps(steps, reference_hashes))
    milestones = set(milestone_steps(steps))
    cache = model.make_cache()
    latencies: list[float] = []
    rolling_digest = hashlib.sha256()
    input_token = 1
    mx.reset_peak_memory()
    artifact["reserved_cache_initial_memory"] = _memory_snapshot()
    artifact["reserved_cache_initial_authoritative_bytes"] = _cache_nbytes(cache)

    try:
        for step in range(1, steps + 1):
            started = time.perf_counter()
            output = model(mx.array([[input_token]], dtype=mx.uint32), cache=cache)
            logits = output.logits[0, -1]
            predicted_array = mx.argmax(logits)
            nan_array = mx.sum(mx.isnan(logits))
            mx.eval(predicted_array, nan_array)
            predicted = int(predicted_array.item())
            step_nans = int(nan_array.item())
            mx.synchronize()
            latencies.append(time.perf_counter() - started)
            artifact["nan_count"] += step_nans
            rolling_digest.update(predicted.to_bytes(4, "little", signed=False))

            if step in checkpoints:
                logits_hash, evidence_nans = _logits_evidence(logits)
                artifact["checkpoint_hashes"][str(step)] = {
                    "token_sha256": rolling_digest.copy().hexdigest(),
                    "logits_sha256": logits_hash,
                    "nan_count": evidence_nans,
                }

            if step % MATERIALIZATION_INTERVAL == 0:
                before = _memory_snapshot()
                materialization_started = time.perf_counter()
                materialize_cache(cache)
                materialization_ms = (time.perf_counter() - materialization_started) * 1000.0
                after = _memory_snapshot()
                state_leaves, array_leaves = _state_leaves(cache)
                artifact["scheduled_materialization_count"] += 1
                row = {
                    "step": step,
                    "before": before,
                    "after": after,
                    "materialization_ms": materialization_ms,
                    "materialization_count": artifact["scheduled_materialization_count"],
                    "state_leaf_count": state_leaves,
                    "state_array_leaf_count": array_leaves,
                    "authoritative_cache_bytes": _cache_nbytes(cache),
                    "decode_window_p50_ms": statistics.median(
                        latencies[-TELEMETRY_INTERVAL:]
                    )
                    * 1000.0,
                    "decode_window_p95_ms": _percentile_ms(
                        latencies[-TELEMETRY_INTERVAL:], 95
                    ),
                    "nan_count": artifact["nan_count"],
                    "tokens_since_materialization_before": MATERIALIZATION_INTERVAL,
                    "tokens_since_materialization_after": 0,
                }
                artifact["boundary_telemetry"].append(row)
                print(
                    json.dumps(
                        {
                            "phase": "boundary",
                            **row,
                            "before": before,
                            "after": after,
                        }
                    ),
                    flush=True,
                )

            artifact["last_completed_step"] = step
            artifact["rolling_token_sha256"] = rolling_digest.copy().hexdigest()

            if step in milestones:
                row = _record_milestone(
                    artifact,
                    step=step,
                    cache=cache,
                    rolling_digest=rolling_digest,
                    latencies=latencies,
                )
                if step == MILESTONES[0]:
                    failures = _milestone_stop_failures(artifact)
                    artifact["milestone_25000_stop_failures"] = failures
                    artifact["milestone_25000_passed"] = not failures
                _atomic_write_artifact(output_path, artifact)
                print(json.dumps({"phase": "milestone", **row}), flush=True)
                if step == MILESTONES[0] and artifact["milestone_25000_stop_failures"]:
                    raise MilestoneGateError(
                        ", ".join(artifact["milestone_25000_stop_failures"])
                    )

            input_token = predicted
            del output, logits, predicted_array, nan_array

        final_before = _memory_snapshot()
        started = time.perf_counter()
        materialize_cache(cache)
        final_ms = (time.perf_counter() - started) * 1000.0
        final_after = _memory_snapshot()
        leaves, array_leaves = _state_leaves(cache)
        artifact["final_evidence_materialization"] = {
            "scheduled": False,
            "excluded_from_scheduled_count": True,
            "tokens_since_last_scheduled_materialization_before": (
                steps % MATERIALIZATION_INTERVAL
            ),
            "before": final_before,
            "after": final_after,
            "materialization_ms": final_ms,
            "scheduled_materialization_count_unchanged": artifact[
                "scheduled_materialization_count"
            ],
            "state_leaf_count": leaves,
            "state_array_leaf_count": array_leaves,
            "authoritative_cache_bytes": _cache_nbytes(cache),
        }
        if steps == DEFAULT_STEPS:
            _finalize(artifact, latencies)
        _atomic_write_artifact(output_path, artifact)
    finally:
        _release(cache)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(
            "bench-results/"
            "m3ultra512-recurrent-state-materialization-frontier-20260830.json"
        ),
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    args = parser.parse_args()
    if args.steps < TELEMETRY_INTERVAL:
        raise ValueError("100k soak probe requires at least 256 steps")

    reference = _load_reference(args.reference)
    report = inspect_checkpoint(args.model, require_server_ready=True)
    if report.fingerprint != reference["checkpoint_fingerprint"]:
        raise ValueError("checkpoint fingerprint does not match the 4006e9d reference")
    artifact = {
        "schema": "glm53-recurrent-state-100k-soak-v1",
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
        "materialization_interval": MATERIALIZATION_INTERVAL,
        "telemetry_interval": TELEMETRY_INTERVAL,
        "milestone_steps": list(milestone_steps(args.steps)),
        "server_admission_bypassed_inside_probe_only": True,
        "disk_resume_supported": False,
        "process_resume_supported": False,
        "metal_buffer_count_api_available": False,
        "materialization_operation": (
            "mx.eval([entry.state for entry in cache]); "
            "mx.clear_cache(); mx.synchronize()"
        ),
        "reference_8192": _reference_descriptor(reference, args.reference),
        "checkpoint_hashes": {},
        "boundary_telemetry": [],
        "milestones": {},
        "scheduled_materialization_count": 0,
        "final_evidence_materialization": None,
        "nan_count": 0,
        "metal_error": None,
        "last_completed_step": 0,
        "rolling_token_sha256": hashlib.sha256().hexdigest(),
        "milestone_25000_passed": None,
        "milestone_25000_stop_failures": [],
        "complete": False,
    }
    _atomic_write_artifact(args.output, artifact)

    mx.set_wired_limit(int(440e9))
    mx.set_cache_limit(int(32e9))
    try:
        model, _ = load(
            args.model,
            experimental_compact_nope_dsa_cache=True,
            compact_cache_reserve_tokens=args.steps + RESERVE_TAIL,
        )
        warm_residency(model)
        _warm_model(model)
        run_soak(model, artifact, args.output)
    except BaseException as exc:
        if not isinstance(exc, MilestoneGateError):
            artifact["metal_error"] = f"{type(exc).__name__}: {exc}"
        artifact["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "last_completed_step": artifact["last_completed_step"],
        }
        artifact["complete"] = False
        _atomic_write_artifact(args.output, artifact)
        print(json.dumps({"phase": "failure", **artifact["failure"]}), flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1

    print(
        json.dumps(
            {
                "phase": "result",
                "last_completed_step": artifact["last_completed_step"],
                "acceptance": artifact.get("acceptance"),
            }
        ),
        flush=True,
    )
    return 0 if artifact.get("acceptance", {}).get("accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
