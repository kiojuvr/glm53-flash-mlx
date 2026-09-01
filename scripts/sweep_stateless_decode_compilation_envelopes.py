#!/usr/bin/env python3
"""Sweep cache-free decode compilation envelopes with bounded Metal telemetry."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import characterize_packed_decode_bounded_telemetry as bounded
from capture_budget import (
    atomic_write,
    path_bytes,
    remove_partial_trace,
    terminate_process_group,
)
from stateless_decode_envelopes import ARMS, StatelessDecodeEnvelopeRunner

from glm53_flash_mlx.abi import MLX_VLM_REVISION, PACKED_DECODE_KERNEL_ABI
from glm53_flash_mlx.manifest import inspect_checkpoint


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-stateless-decode-compilation-envelopes-20260901.json"
)
INITIAL_CONTEXT = 2049
WARMUPS = 2
TRACED_TOKENS = 16
MEASURED_TOKENS = 272
MATERIALIZATION_STEP = 256
EVIDENCE_STEPS = (1, 16, 255, 256, 257, 272)
TRACE_SECONDS = 8
TARGET_MS = 1000.0 / 15.0


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(q * len(ordered)) - 1)])


def _memory(mx) -> dict[str, int]:
    mx.synchronize()
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }


def _release(mx, *values) -> None:
    for value in values:
        if isinstance(value, list):
            value.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def _warm_runner(model, runner, mx, boundary) -> dict:
    mx.reset_peak_memory()
    active_before = int(mx.get_active_memory())
    cache = model.make_cache()
    started = time.perf_counter()
    output = runner(mx.array([[101]], dtype=mx.uint32), cache=cache)
    mx.eval(output.logits)
    mx.synchronize()
    first_seconds = time.perf_counter() - started
    first_state_hash = boundary._full_cache_hash(cache)
    _release(mx, cache, output)

    cache = model.make_cache()
    started = time.perf_counter()
    output = runner(mx.array([[101]], dtype=mx.uint32), cache=cache)
    mx.eval(output.logits)
    mx.synchronize()
    second_seconds = time.perf_counter() - started
    second_state_hash = boundary._full_cache_hash(cache)
    active_after = int(mx.get_active_memory())
    peak = int(mx.get_peak_memory())
    _release(mx, cache, output)
    return {
        "first_warmup_seconds": first_seconds,
        "second_warmup_seconds": second_seconds,
        "active_before_bytes": active_before,
        "active_after_bytes": active_after,
        "active_delta_bytes": active_after - active_before,
        "peak_bytes": peak,
        "working_peak_bytes": peak - active_before,
        "first_warmup_post_state_hash": first_state_hash,
        "second_warmup_post_state_hash": second_state_hash,
        "fresh_cache_warmup_state_exact": first_state_hash == second_state_hash,
        "compile_cache_counter_api_available": False,
        "observed_input_shape_signatures": [[1, 1]],
        "warmup_retrace_claim": False,
    }


def _fixed_backend_parity(model, runner, mx, boundary, packed_probe) -> dict:
    rows = {}
    fixed_tokens = (3000, 3001, 3002)
    for backend in ("direct", "compact-nope-dsa"):
        cache = boundary._synthetic_cache(model, INITIAL_CONTEXT, backend)
        hashes = []
        for token_id in fixed_tokens:
            output = runner(mx.array([[token_id]], dtype=mx.uint32), cache=cache)
            mx.eval(output.logits)
            mx.synchronize()
            hashes.append(packed_probe._hash(output.logits[0, -1]))
        rows[backend] = {
            "full_vocab_hashes": hashes,
            "cache_leaf_count": boundary._cache_leaf_count(cache),
            "post_state_hash": boundary._full_cache_hash(cache),
        }
        _release(mx, cache, output)
    rows["full_vocab_exact"] = (
        rows["direct"]["full_vocab_hashes"]
        == rows["compact-nope-dsa"]["full_vocab_hashes"]
    )
    return rows


def _child(args) -> int:
    import mlx.core as mx

    sys.path.insert(0, str(Path(__file__).parent))
    import probe_long_context_first_decode_boundary as boundary
    import probe_packed_decode_runtime as packed_probe
    import probe_residual_packed_decode_moe_fusion as residual

    from glm53_flash_mlx.loader import load, warm_residency

    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    mx.reset_peak_memory()
    started = time.perf_counter()
    model, _ = load(args.model, experimental_packed_decode_moe=True)
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    warm_residency(model)
    residency_seconds = time.perf_counter() - started

    try:
        with residual._runtime(residual.Arm("B1", True)):
            runner = StatelessDecodeEnvelopeRunner(model.language_model, args.arm)
            warmup = _warm_runner(model, runner, mx, boundary)
            cache = boundary._synthetic_cache(model, INITIAL_CONTEXT, "direct")
            configured_capacity = bounded._reserve_direct_dsa(cache, boundary, mx)
            token = mx.array([[3000]], dtype=mx.uint32)
            warmup_tokens = []
            pretrace_warmup_latencies_ms = []
            for _ in range(WARMUPS):
                warmup_started = time.perf_counter()
                output = runner(token, cache=cache)
                predicted = mx.argmax(output.logits[0, -1])
                mx.eval(output.logits, predicted)
                mx.synchronize()
                pretrace_warmup_latencies_ms.append(
                    (time.perf_counter() - warmup_started) * 1000.0
                )
                value = int(predicted.item())
                warmup_tokens.append(value)
                token = mx.array([[value]], dtype=mx.uint32)

            physical_before = boundary._physical_capacity(cache, "direct")
            leaf_count_before = boundary._cache_leaf_count(cache)
            active_before = _memory(mx)
            atomic_write(
                args.ready,
                {
                    "pid": os.getpid(),
                    "arm": args.arm,
                    "context": INITIAL_CONTEXT + WARMUPS,
                    "policy": asdict(runner.policy),
                },
            )
            bounded._wait_for(args.go, 120.0)

            latencies = []
            generated = []
            evidence_logits = {}
            evaluated_logits = []
            materialization_ms = []
            for step in range(1, TRACED_TOKENS + 1):
                started = time.perf_counter()
                output = runner(token, cache=cache)
                logits = output.logits[0, -1]
                predicted = mx.argmax(logits)
                mx.eval(output.logits, predicted)
                mx.synchronize()
                value = int(predicted.item())
                latencies.append((time.perf_counter() - started) * 1000.0)
                generated.append(value)
                evaluated_logits.append(logits)
                if step in EVIDENCE_STEPS:
                    evidence_logits[str(step)] = logits
                token = mx.array([[value]], dtype=mx.uint32)
            atomic_write(args.decode_done, {"pid": os.getpid(), "steps": 16})
            bounded._wait_for(args.trace_done, 300.0)

            for step in range(TRACED_TOKENS + 1, MEASURED_TOKENS + 1):
                started = time.perf_counter()
                output = runner(token, cache=cache)
                logits = output.logits[0, -1]
                predicted = mx.argmax(logits)
                mx.eval(output.logits, predicted)
                mx.synchronize()
                value = int(predicted.item())
                latencies.append((time.perf_counter() - started) * 1000.0)
                generated.append(value)
                evaluated_logits.append(logits)
                if step in EVIDENCE_STEPS:
                    evidence_logits[str(step)] = logits
                if step == MATERIALIZATION_STEP:
                    materialized = time.perf_counter()
                    mx.eval([entry.state for entry in cache])
                    mx.clear_cache()
                    mx.synchronize()
                    materialization_ms.append(
                        (time.perf_counter() - materialized) * 1000.0
                    )
                token = mx.array([[value]], dtype=mx.uint32)

            evidence_hashes = {
                step: packed_probe._hash(logits)
                for step, logits in evidence_logits.items()
            }
            logits_stack = mx.stack(evaluated_logits)
            nan_array = mx.sum(mx.isnan(logits_stack))
            mx.eval(nan_array)
            nan_count = int(nan_array.item())
            del logits_stack, evaluated_logits, evidence_logits
            gc.collect()
            mx.clear_cache()
            mx.synchronize()
            physical_after = boundary._physical_capacity(cache, "direct")
            leaf_count_after = boundary._cache_leaf_count(cache)
            post_state_hash = boundary._full_cache_hash(cache)
            idle_hash_before = boundary._full_cache_hash(cache)
            mx.synchronize()
            idle_hash_after = boundary._full_cache_hash(cache)
            active_after = _memory(mx)
            backend_parity = _fixed_backend_parity(
                model, runner, mx, boundary, packed_probe
            )
            child = {
                "schema": "glm53-stateless-decode-envelope-child-v1",
                "complete": True,
                "arm": args.arm,
                "policy": asdict(runner.policy),
                "load_seconds": load_seconds,
                "residency_seconds": residency_seconds,
                "compile_warmup": warmup,
                "generated_token_sha256": packed_probe._token_digest(generated),
                "evidence_full_vocab_hashes": evidence_hashes,
                "post_cache_state_hash": post_state_hash,
                "warmup_generated_tokens": warmup_tokens,
                "pretrace_warmup_latencies_ms": pretrace_warmup_latencies_ms,
                "first_decode_after_synthetic_restore_ms": (
                    pretrace_warmup_latencies_ms[0]
                ),
                "latency_samples_ms": latencies,
                "capture_attached_first_token_latency_ms": latencies[0],
                "capture_attached_first_token_is_steady_evidence": False,
                "latency_p50_ms": statistics.median(latencies),
                "latency_p95_ms": _percentile(latencies, 0.95),
                "decode_tokens_per_second": 1000.0 / statistics.median(latencies),
                "materialization_step": MATERIALIZATION_STEP,
                "materialization_count": len(materialization_ms),
                "materialization_ms": materialization_ms,
                "physical_capacity_before": physical_before,
                "physical_capacity_after": physical_after,
                "physical_capacity_unchanged": physical_before == physical_after,
                "configured_capacity_tokens": configured_capacity,
                "cache_leaf_count_before": leaf_count_before,
                "cache_leaf_count_after": leaf_count_after,
                "cache_leaf_count_constant": leaf_count_before == leaf_count_after,
                "active_memory_before": active_before,
                "active_memory_after": active_after,
                "active_memory_drift_bytes": (
                    active_after["active_bytes"] - active_before["active_bytes"]
                ),
                "idle_without_forward_state_unchanged": idle_hash_before
                == idle_hash_after,
                "direct_compact_parity": backend_parity,
                "nan_count": nan_count,
                "metal_error": None,
            }
            atomic_write(args.child_result, child)
    except Exception as exc:
        atomic_write(
            args.child_result,
            {
                "schema": "glm53-stateless-decode-envelope-child-v1",
                "complete": False,
                "arm": args.arm,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    return 0


def _merge(
    output: Path, arm: str, row: dict, checkpoint_identity: dict | None = None
) -> dict:
    artifact = {
        "schema": "glm53-stateless-decode-compilation-envelope-sweep-v1",
        "date": date.today().isoformat(),
        "complete": False,
        "probe_only": True,
        "runtime_changes": False,
        "cache_or_apc_abi_changes": False,
        "arms": {},
        "target_tokens_per_second": 15.0,
        "target_ms_per_token": TARGET_MS,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "packed_decode_kernel_abi": PACKED_DECODE_KERNEL_ABI,
    }
    if output.exists():
        artifact = json.loads(output.read_text())
    artifact["mlx_vlm_revision"] = MLX_VLM_REVISION
    artifact["packed_decode_kernel_abi"] = PACKED_DECODE_KERNEL_ABI
    if checkpoint_identity is not None:
        artifact.update(checkpoint_identity)
    artifact["arms"][arm] = row
    if set(artifact["arms"]) == set(ARMS):
        baseline = artifact["arms"]["A"]
        comparisons = {}
        for name, candidate in artifact["arms"].items():
            exact = {
                "generated_tokens": candidate["child"]["generated_token_sha256"]
                == baseline["child"]["generated_token_sha256"],
                "full_vocab_logits": candidate["child"][
                    "evidence_full_vocab_hashes"
                ]
                == baseline["child"]["evidence_full_vocab_hashes"],
                "cache_state": candidate["child"]["post_cache_state_hash"]
                == baseline["child"]["post_cache_state_hash"],
                "direct_compact_logits": candidate["child"][
                    "direct_compact_parity"
                ]["full_vocab_exact"],
                "capacity": candidate["child"]["physical_capacity_unchanged"],
                "state_leaf_count": candidate["child"]["cache_leaf_count_constant"],
                "idle_state": candidate["child"][
                    "idle_without_forward_state_unchanged"
                ],
                "nan_zero": candidate["child"]["nan_count"] == 0,
                "metal_error_zero": candidate["child"]["metal_error"] is None,
                "materialization_boundary": candidate["child"][
                    "materialization_count"
                ]
                == 1,
                "active_memory_drift_le_64mib": abs(
                    candidate["child"]["active_memory_drift_bytes"]
                )
                <= 64 << 20,
            }
            busy_a = baseline["telemetry"]["gpu_busy_ms"] / TRACED_TOKENS
            busy_c = candidate["telemetry"]["gpu_busy_ms"] / TRACED_TOKENS
            idle_a = baseline["telemetry"]["gpu_idle_gap_ms"] / TRACED_TOKENS
            idle_c = candidate["telemetry"]["gpu_idle_gap_ms"] / TRACED_TOKENS
            submissions_a = (
                baseline["telemetry"]["command_buffer_submission_rows"]
                / TRACED_TOKENS
            )
            submissions_c = (
                candidate["telemetry"]["command_buffer_submission_rows"]
                / TRACED_TOKENS
            )
            performance = {
                "median_ms": candidate["child"]["latency_p50_ms"],
                "tokens_per_second": candidate["child"][
                    "decode_tokens_per_second"
                ],
                "gpu_busy_ms_per_token": busy_c,
                "gpu_idle_ms_per_token": idle_c,
                "gpu_busy_regression_fraction": (busy_c / busy_a - 1.0)
                if busy_a
                else None,
                "idle_reduction_ms_per_token": idle_a - idle_c,
                "command_buffer_reduction_per_token": submissions_a
                - submissions_c,
            }
            screening = {
                "all_exact": all(exact.values()),
                "gpu_busy_regression_le_0_5pct": (
                    performance["gpu_busy_regression_fraction"] <= 0.005
                ),
                "idle_reduction_ge_0_75ms": (
                    name != "A"
                    and performance["idle_reduction_ms_per_token"] >= 0.75
                ),
                "command_buffer_reduction_ge_2": (
                    name != "A"
                    and performance["command_buffer_reduction_per_token"] >= 2.0
                ),
                "working_peak_le_512mib": candidate["child"]["compile_warmup"].get(
                    "working_peak_bytes",
                    candidate["child"]["compile_warmup"]["peak_bytes"]
                    - candidate["child"]["compile_warmup"]["active_before_bytes"],
                )
                <= 512 << 20,
            }
            comparisons[name] = {
                "correctness": exact,
                "performance": performance,
                "screening": screening,
                "short_screen_passed": all(screening.values())
                if name != "A"
                else all(exact.values()),
            }
        artifact["comparisons_to_A"] = comparisons
        artifact["correctness_complete"] = all(
            row["screening"]["all_exact"] for row in comparisons.values()
        )
        artifact["complete"] = artifact["correctness_complete"]
        artifact["screening_passed_arms"] = [
            name
            for name in ("B", "C", "D")
            if comparisons[name]["short_screen_passed"]
        ]
        artifact["production_candidate_arms"] = [
            name
            for name in ("B", "C", "D")
            if comparisons[name]["correctness"]["generated_tokens"]
            and comparisons[name]["performance"]["tokens_per_second"] >= 15.0
        ]
    atomic_write(output, artifact)
    return artifact


def _parent(args) -> int:
    report = inspect_checkpoint(args.model, require_server_ready=True)
    checkpoint_identity = {
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "checkpoint_layout_digest": report.layout_digest,
    }
    trace = bounded._validate_trace(args.trace)
    with tempfile.TemporaryDirectory(
        prefix=f"glm53-stateless-envelope-{args.arm}-"
    ) as temporary:
        temporary = Path(temporary)
        ready = temporary / "ready.json"
        go = temporary / "go"
        decode_done = temporary / "decode-done.json"
        trace_done = temporary / "trace-done"
        child_result = temporary / "child.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            str(args.model),
            "--arm",
            args.arm,
            "--trace",
            str(trace),
            "--output",
            str(args.output),
            "--ready",
            str(ready),
            "--go",
            str(go),
            "--decode-done",
            str(decode_done),
            "--trace-done",
            str(trace_done),
            "--child-result",
            str(child_result),
            "--wired-limit-gb",
            str(args.wired_limit_gb),
            "--cache-limit-gb",
            str(args.cache_limit_gb),
            "--child",
        ]
        child = subprocess.Popen(command, start_new_session=True)
        tracer = None
        log_handle = None
        try:
            bounded._wait_for(ready, 600.0, child)
            pid = int(json.loads(ready.read_text())["pid"])
            log = temporary / "xctrace.log"
            log_handle = log.open("w")
            tracer = subprocess.Popen(
                [
                    "xcrun",
                    "xctrace",
                    "record",
                    "--template",
                    "Metal System Trace",
                    "--time-limit",
                    f"{TRACE_SECONDS}s",
                    "--output",
                    str(trace),
                    "--no-prompt",
                    "--attach",
                    str(pid),
                ],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                current = log.read_text() if log.exists() else ""
                if "Ctrl-C to stop the recording" in current:
                    break
                if tracer.poll() is not None:
                    raise RuntimeError(f"xctrace exited early: {current}")
                time.sleep(0.2)
            else:
                raise TimeoutError("xctrace did not report recording readiness")
            go.touch()
            bounded._wait_for(decode_done, 120.0, child)
            if tracer.wait(timeout=300.0) != 0:
                raise RuntimeError(f"xctrace failed: {log.read_text()}")
            log_handle.close()
            log_handle = None
            trace_done.touch()
            if child.wait(timeout=300.0) != 0:
                raise RuntimeError("stateless envelope child failed")
            if not child_result.exists():
                raise RuntimeError("child result is missing")
            child_value = json.loads(child_result.read_text())
            if not child_value.get("complete"):
                raise RuntimeError(child_value.get("error", "child incomplete"))
            if path_bytes(trace) > bounded.MAX_TRACE_BYTES:
                raise RuntimeError("bounded System Trace exceeded 4 GiB")
            telemetry = bounded._export_telemetry(trace, pid)
            identity = bounded._trace_identity(trace)
            identity.update(path=str(trace), stored_in_repository=False)
            artifact = _merge(
                args.output,
                args.arm,
                {
                    "pid": pid,
                    "child": child_value,
                    "telemetry": telemetry,
                    "trace": identity,
                    "trace_time_limit_seconds": TRACE_SECONDS,
                },
                checkpoint_identity,
            )
            print(
                json.dumps(
                    {
                        "arm": args.arm,
                        "artifact_complete": artifact["complete"],
                        "trace": identity,
                    },
                    indent=2,
                )
            )
            return 0
        except Exception:
            if tracer is not None:
                terminate_process_group(tracer)
            terminate_process_group(child)
            remove_partial_trace(trace)
            raise
        finally:
            if log_handle is not None:
                log_handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ready", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--go", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--decode-done", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--trace-done", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--child-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    return _child(args) if args.child else _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
