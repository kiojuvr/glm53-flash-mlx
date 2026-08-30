#!/usr/bin/env python3
"""Validate the production recurrent-state materialization policy on Direct."""

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

from glm53_flash_mlx.abi import KERNEL_ABI_VERSION, MLX_VLM_REVISION
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint
from glm53_flash_mlx.materialization import (
    MATERIALIZATION_INTERVAL_TOKENS,
    MATERIALIZATION_POLICY,
    RecurrentMaterializationTelemetry,
)

REFERENCE_INTERVAL = 50
DEFAULT_STEPS = 4096
MAX_REGRESSION = 0.02


def evidence_steps(steps: int) -> tuple[int, ...]:
    values = {1, 49, 50, 51, 255, 256, 257, 511, 512, 4095, 4096}
    values.update(range(MATERIALIZATION_INTERVAL_TOKENS, steps + 1, 256))
    return tuple(sorted(step for step in values if step <= steps))


def materialize_cache(cache) -> None:
    mx.eval([entry.state for entry in cache])
    mx.clear_cache()
    mx.synchronize()


def _logits_evidence(logits) -> tuple[str, int]:
    values = np.ascontiguousarray(np.asarray(logits.astype(mx.float32)))
    return hashlib.sha256(values.tobytes()).hexdigest(), int(np.isnan(values).sum())


def _token_digest(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(int(token).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def _memory_snapshot() -> dict[str, int]:
    return {
        "active_memory_bytes": int(mx.get_active_memory()),
        "cache_memory_bytes": int(mx.get_cache_memory()),
        "peak_memory_bytes": int(mx.get_peak_memory()),
    }


def _release(cache) -> None:
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
    telemetry = RecurrentMaterializationTelemetry(interval)
    telemetry.start_generator(interval)
    checkpoints = set(evidence_steps(steps))
    predicted_trace = []
    replay_input_trace = []
    logits_hashes = {}
    materialization_steps = []
    materialization_ms = []
    forward_ms = []
    total_ms = []
    nan_count = 0
    input_token = 1
    mx.reset_peak_memory()
    arm_started = time.perf_counter()

    for step in range(1, steps + 1):
        step_started = time.perf_counter()
        output = model(mx.array([[input_token]], dtype=mx.uint32), cache=cache)
        logits = output.logits[0, -1]
        predicted_array = mx.argmax(logits)
        nan_array = mx.sum(mx.isnan(logits))
        mx.eval(predicted_array, nan_array)
        predicted = int(predicted_array.item())
        nan_count += int(nan_array.item())
        forward_done = time.perf_counter()

        predicted_trace.append(predicted)
        token = predicted if teacher_tokens is None else int(teacher_tokens[step - 1])
        replay_input_trace.append(token)
        if step in checkpoints:
            digest, evidence_nans = _logits_evidence(logits)
            logits_hashes[str(step)] = digest
            if evidence_nans:
                raise RuntimeError(f"NaN found in logits evidence at step {step}")

        materialized = interval > 0 and step % interval == 0
        if materialized:
            boundary_started = time.perf_counter()
            materialize_cache(cache)
            materialization_ms.append(
                (time.perf_counter() - boundary_started) * 1000.0
            )
            materialization_steps.append(step)
        telemetry.observe_decode_step(
            step=step,
            materialized=materialized,
            memory=_memory_snapshot() if materialized else None,
        )
        step_done = time.perf_counter()
        forward_ms.append((forward_done - step_started) * 1000.0)
        total_ms.append((step_done - step_started) * 1000.0)
        input_token = token
        del output, logits, predicted_array, nan_array

        if step % 256 == 0:
            print(
                json.dumps(
                    {
                        "interval": interval,
                        "step": step,
                        "completed_materializations": telemetry.snapshot()[
                            "completed_materializations"
                        ],
                        "window_p50_ms": statistics.median(total_ms[-256:]),
                    }
                ),
                flush=True,
            )

    elapsed = time.perf_counter() - arm_started
    snapshot = telemetry.snapshot()
    result = {
        "interval_tokens": interval,
        "steps": steps,
        "complete": True,
        "token_trace": predicted_trace,
        "token_sha256": _token_digest(predicted_trace),
        "replay_input_sha256": _token_digest(replay_input_trace),
        "logits_hashes": logits_hashes,
        "nan_count": nan_count,
        "metal_error": None,
        "materialization_steps": materialization_steps,
        "materialization_count": len(materialization_steps),
        "expected_materialization_count": steps // interval,
        "materialization_total_ms": sum(materialization_ms),
        "materialization_median_ms": (
            statistics.median(materialization_ms) if materialization_ms else 0.0
        ),
        "forward_warm_median_ms": statistics.median(forward_ms[256:]),
        "end_to_end_elapsed_seconds": elapsed,
        "end_to_end_tokens_per_second": steps / elapsed,
        "telemetry": snapshot,
        "peak_memory_bytes": int(mx.get_peak_memory()),
    }
    _release(cache)
    return result


def _compact_policy_evidence(path: Path) -> dict:
    artifact = json.loads(path.read_text())
    return {
        "path": str(path),
        "schema": artifact["schema"],
        "accepted": artifact["acceptance"]["accepted"],
        "interval_tokens": artifact["materialization_interval"],
        "scheduled_materialization_count": artifact["summary"][
            "scheduled_materialization_count"
        ],
        "complete": artifact["complete"],
    }


def _finalize(artifact: dict) -> None:
    reference = artifact["arms"][str(REFERENCE_INTERVAL)]
    production = artifact["arms"][str(MATERIALIZATION_INTERVAL_TOKENS)]
    regression = (
        reference["end_to_end_tokens_per_second"]
        / production["end_to_end_tokens_per_second"]
        - 1.0
    )
    expected_boundaries = list(
        range(MATERIALIZATION_INTERVAL_TOKENS, artifact["steps"] + 1, 256)
    )
    compact = artifact["compact_100k_policy_evidence"]
    acceptance = {
        "direct_completed_4096_steps": all(
            arm["complete"] and arm["steps"] == DEFAULT_STEPS
            for arm in artifact["arms"].values()
        ),
        "interval_50_and_256_token_trace_identical": (
            reference["token_sha256"] == production["token_sha256"]
        ),
        "interval_50_and_256_logits_hashes_identical": (
            reference["logits_hashes"] == production["logits_hashes"]
        ),
        "boundary_hashes_cover_255_256_257_511_512_4095_4096": all(
            str(step) in production["logits_hashes"]
            for step in (255, 256, 257, 511, 512, 4095, 4096)
        ),
        "production_materialization_count_16": (
            production["materialization_count"] == 16
            and production["telemetry"]["completed_materializations"] == 16
        ),
        "production_boundary_steps_exact": (
            production["materialization_steps"] == expected_boundaries
            and production["telemetry"]["last_materialization_step"] == 4096
        ),
        "no_nan_or_metal_error": all(
            arm["nan_count"] == 0 and arm["metal_error"] is None
            for arm in artifact["arms"].values()
        ),
        "interval_256_end_to_end_regression_at_most_2_percent": (
            regression <= MAX_REGRESSION
        ),
        "compact_100k_uses_same_production_policy": (
            compact["complete"]
            and compact["accepted"]
            and compact["interval_tokens"] == MATERIALIZATION_INTERVAL_TOKENS
        ),
    }
    acceptance["accepted"] = all(acceptance.values())
    artifact["comparison"] = {
        "interval_256_vs_50_end_to_end_regression": regression,
        "interval_50_tokens_per_second": reference["end_to_end_tokens_per_second"],
        "interval_256_tokens_per_second": production[
            "end_to_end_tokens_per_second"
        ],
    }
    artifact["acceptance"] = acceptance
    artifact["complete"] = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument(
        "--compact-100k-artifact",
        type=Path,
        default=Path(
            "bench-results/"
            "m3ultra512-recurrent-state-100k-fixed-capacity-20260830.json"
        ),
    )
    args = parser.parse_args()
    if args.steps != DEFAULT_STEPS:
        raise ValueError("production gate requires exactly 4,096 decode steps")

    mx.set_wired_limit(int(440e9))
    mx.set_cache_limit(int(32e9))
    report = inspect_checkpoint(args.model, require_server_ready=True)
    artifact = {
        "schema": "glm53-bounded-recurrent-materialization-v1",
        "date": date.today().isoformat(),
        "machine": platform.machine(),
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "kernel_abi": KERNEL_ABI_VERSION,
        "cache_backend": "direct",
        "moe_backend": "direct",
        "policy": MATERIALIZATION_POLICY,
        "production_interval_tokens": MATERIALIZATION_INTERVAL_TOKENS,
        "steps": args.steps,
        "evidence_steps": list(evidence_steps(args.steps)),
        "compact_100k_policy_evidence": _compact_policy_evidence(
            args.compact_100k_artifact
        ),
        "arms": {},
        "complete": False,
    }

    model, _ = load(args.model)
    warm_residency(model)
    _warm_model(model)
    reference = run_arm(model, interval=REFERENCE_INTERVAL, steps=args.steps)
    artifact["arms"][str(REFERENCE_INTERVAL)] = reference
    production = run_arm(
        model,
        interval=MATERIALIZATION_INTERVAL_TOKENS,
        steps=args.steps,
        teacher_tokens=reference["token_trace"],
    )
    artifact["arms"][str(MATERIALIZATION_INTERVAL_TOKENS)] = production
    _finalize(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact["acceptance"], indent=2), flush=True)
    return 0 if artifact["acceptance"]["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
