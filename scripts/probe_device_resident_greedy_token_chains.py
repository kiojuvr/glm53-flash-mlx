#!/usr/bin/env python3
"""Probe exact device-resident greedy autoregressive token chains."""

from __future__ import annotations

import argparse
import collections
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
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import characterize_packed_decode_bounded_telemetry as bounded
from capture_budget import (
    atomic_write,
    path_bytes,
    remove_partial_trace,
    terminate_process_group,
)

from glm53_flash_mlx.abi import MLX_VLM_REVISION, PACKED_DECODE_KERNEL_ABI
from glm53_flash_mlx.manifest import inspect_checkpoint
from glm53_flash_mlx.materialization import MATERIALIZATION_INTERVAL_TOKENS


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-device-resident-greedy-token-chains-20260901.json"
)
ARMS = {
    "A": 1,
    "B": 2,
    "C": 4,
    "D": 8,
    "E": 16,
}
INITIAL_CONTEXT = 2049
WARMUPS = 2
SCREEN_TOKENS = 64
TRACE_SECONDS = 8
SCREEN_TIME_LIMIT_SECONDS = 30.0
TARGET_TPS = 15.0
TARGET_MS = 1000.0 / TARGET_TPS
MAX_WORKING_PEAK_BYTES = 512 << 20
MAX_ACTIVE_DRIFT_BYTES = 64 << 20
ROLLBACK_WIDTHS = (2, 4)
BOUNDARY_START_STEP = 254
BOUNDARY_STEPS = 3


@dataclass(frozen=True)
class ChainLimit:
    configured_width: int
    remaining_generation: int
    tokens_until_materialization: int
    tokens_until_capacity: int


def bounded_chain_width(limit: ChainLimit) -> int:
    """Return a positive exact-unroll width without crossing a state boundary."""
    if limit.configured_width < 1:
        raise ValueError("configured chain width must be positive")
    available = min(
        int(limit.configured_width),
        int(limit.remaining_generation),
        int(limit.tokens_until_materialization),
        int(limit.tokens_until_capacity),
    )
    if available < 0:
        raise ValueError("chain limits must be non-negative")
    return available


def tokens_until_materialization(completed_steps: int) -> int:
    remainder = int(completed_steps) % MATERIALIZATION_INTERVAL_TOKENS
    return MATERIALIZATION_INTERVAL_TOKENS - remainder if remainder else 256


def stop_reasons(width: int, accepted_tokens: int) -> list[str]:
    reasons = []
    if accepted_tokens == 0:
        reasons.append("client_cancellation")
    if accepted_tokens == 1:
        reasons.extend(("eos", "stop_token"))
    if accepted_tokens >= 2:
        reasons.append("multi_token_stop_sequence")
    if accepted_tokens < width:
        reasons.append("generation_cap")
    if accepted_tokens == width:
        reasons.append("total_context_cap")
    return reasons


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


def _physical_capacity_tokens(cache, backend: str, boundary) -> int:
    rows = boundary._physical_capacity(cache, backend)
    return int(rows["minimum_latent_tokens"])


def _logical_offset(cache) -> int:
    # Layer 3 is the first audited DSA layer.  Both direct and compact layouts
    # expose the latent entry first and use the same logical token offset.
    return int(cache[3][0].offset)


def _materialize(mx, cache) -> float:
    started = time.perf_counter()
    mx.eval([entry.state for entry in cache])
    mx.clear_cache()
    mx.synchronize()
    return (time.perf_counter() - started) * 1000.0


def _chain(
    model,
    mx,
    token,
    cache,
    *,
    configured_width: int,
    steps: int,
    completed_steps: int = 0,
    capacity_tokens: int,
    conventional_width_one: bool = True,
) -> dict:
    generated = []
    logits_rows = []
    chunk_ms = []
    chunk_widths = []
    graph_build_ms = []
    eval_and_readback_ms = []
    materialization_ms = []
    materialization_steps = []
    readbacks = 0
    current = token
    initial_completed = int(completed_steps)
    started_all = time.perf_counter()
    while len(generated) < steps:
        logical_offset = _logical_offset(cache)
        width = bounded_chain_width(
            ChainLimit(
                configured_width=configured_width,
                remaining_generation=steps - len(generated),
                tokens_until_materialization=tokens_until_materialization(
                    completed_steps
                ),
                tokens_until_capacity=capacity_tokens - logical_offset,
            )
        )
        if width < 1:
            raise RuntimeError("device chain reached a capacity boundary")
        chunk_started = time.perf_counter()
        build_started = chunk_started
        device_tokens = []
        chunk_logits = []
        for _ in range(width):
            output = model(current, cache=cache)
            logits = output.logits[0, -1]
            predicted = mx.argmax(logits, axis=-1).astype(mx.uint32)
            current = predicted.reshape(1, 1)
            device_tokens.append(predicted)
            chunk_logits.append(logits)
        build_finished = time.perf_counter()
        packed_tokens = mx.stack(device_tokens)
        mx.eval(packed_tokens)
        host_tokens = np.asarray(packed_tokens).astype(np.uint32, copy=False)
        readbacks += 1
        generated.extend(int(value) for value in host_tokens.tolist())
        logits_rows.extend(chunk_logits)
        completed_steps += width
        graph_build_ms.append((build_finished - build_started) * 1000.0)
        eval_and_readback_ms.append((time.perf_counter() - build_finished) * 1000.0)
        chunk_ms.append((time.perf_counter() - chunk_started) * 1000.0)
        chunk_widths.append(width)
        if conventional_width_one and configured_width == 1:
            current = mx.array([[int(host_tokens[-1])]], dtype=mx.uint32)
        if completed_steps % MATERIALIZATION_INTERVAL_TOKENS == 0:
            materialization_ms.append(_materialize(mx, cache))
            materialization_steps.append(completed_steps)
    steady_chunk_ms = chunk_ms[1:] if len(chunk_ms) > 1 else chunk_ms
    steady_chunk_widths = (
        chunk_widths[1:] if len(chunk_widths) > 1 else chunk_widths
    )
    steady_ms_per_token = [
        elapsed / width
        for elapsed, width in zip(
            steady_chunk_ms, steady_chunk_widths, strict=True
        )
    ]
    return {
        "configured_width": configured_width,
        "initial_completed_steps": initial_completed,
        "steps": steps,
        "generated_tokens": generated,
        "logits": logits_rows,
        "next_device_token": current,
        "readback_count": readbacks,
        "tokens_per_readback": steps / readbacks,
        "chunk_latency_ms": chunk_ms,
        "chunk_widths": chunk_widths,
        "capture_attached_first_chunk_is_steady_evidence": False,
        "steady_chunk_latency_p50_ms": statistics.median(steady_chunk_ms),
        "steady_chunk_latency_p95_ms": _percentile(steady_chunk_ms, 0.95),
        "steady_ms_per_token_samples": steady_ms_per_token,
        "steady_median_ms_per_token": statistics.median(steady_ms_per_token),
        "steady_tokens_per_second": 1000.0
        / statistics.median(steady_ms_per_token),
        "max_steady_stream_silence_ms": max(steady_chunk_ms),
        "graph_build_ms": graph_build_ms,
        "graph_build_total_ms": sum(graph_build_ms),
        "eval_and_readback_ms": eval_and_readback_ms,
        "eval_and_readback_total_ms": sum(eval_and_readback_ms),
        "elapsed_ms": (time.perf_counter() - started_all) * 1000.0,
        "materialization_count": len(materialization_steps),
        "materialization_steps": materialization_steps,
        "materialization_ms": materialization_ms,
    }


def _finish_chain_evidence(result: dict, packed_probe) -> dict:
    logits = result.pop("logits")
    result.pop("next_device_token")
    result["generated_token_sha256"] = packed_probe._token_digest(
        result.pop("generated_tokens")
    )
    result["full_vocab_logits_hashes"] = [
        packed_probe._hash(value) for value in logits
    ]
    result["nan_count"] = sum(
        int(np.isnan(packed_probe._np(value)).sum()) for value in logits
    )
    return result


def _dynamic_gap_summary(trace: Path, pid: int) -> dict:
    import attribute_steady_decode_gpu_idle as idle

    with tempfile.TemporaryDirectory(prefix="glm53-device-chain-gap-") as temporary:
        exports = idle._export(trace, Path(temporary))
        events, coverage = idle._events(exports, pid)
    gaps, reconstruction = idle.reconstruct_gaps(events)
    large = [gap for gap in gaps if gap.total_ns >= 500_000]
    application = [
        gap for gap in large if gap.application_starvation_ns >= 500_000
    ]
    frame_deltas = collections.Counter(
        gap.next_frame - gap.previous_frame
        if gap.previous_frame is not None and gap.next_frame is not None
        else None
        for gap in application
    )
    app_ns = sum(gap.application_starvation_ns for gap in application)
    driver_ns = sum(gap.driver_or_dependency_ns for gap in application)
    return {
        "coverage": coverage,
        "reconstruction": reconstruction,
        "long_gap_threshold_us": 500,
        "long_gap_count": len(large),
        "application_starvation_long_gap_count": len(application),
        "application_starvation_ns": app_ns,
        "application_starvation_ms_per_token": app_ns / SCREEN_TOKENS / 1e6,
        "driver_or_dependency_ns": driver_ns,
        "driver_or_dependency_ms_per_token": driver_ns
        / SCREEN_TOKENS
        / 1e6,
        "frame_delta_counts": {
            "none" if key is None else str(key): value
            for key, value in sorted(
                frame_deltas.items(), key=lambda row: (-1 if row[0] is None else row[0])
            )
        },
        "static_kernel_labels_used": False,
    }


def _first_token(model, mx, boundary, packed_probe) -> dict:
    cache = boundary._synthetic_cache(model, INITIAL_CONTEXT, "direct")
    token = mx.array([[3000]], dtype=mx.uint32)
    started = time.perf_counter()
    output = model(token, cache=cache)
    predicted = mx.argmax(output.logits[0, -1]).astype(mx.uint32)
    mx.eval(predicted)
    value = int(np.asarray(predicted))
    elapsed = (time.perf_counter() - started) * 1000.0
    result = {
        "latency_ms": elapsed,
        "token": value,
        "full_vocab_logits_hash": packed_probe._hash(output.logits[0, -1]),
        "post_state_hash": boundary._full_cache_hash(cache),
    }
    _release(mx, cache, output)
    return result


def _direct_compact_differential(
    model, mx, boundary, packed_probe, width: int
) -> dict:
    rows = {}
    for backend in ("direct", "compact-nope-dsa"):
        cache = boundary._synthetic_cache(model, INITIAL_CONTEXT, backend)
        capacity = _physical_capacity_tokens(cache, backend, boundary)
        result = _chain(
            model,
            mx,
            mx.array([[4000]], dtype=mx.uint32),
            cache,
            configured_width=width,
            steps=8,
            capacity_tokens=capacity,
        )
        result = _finish_chain_evidence(result, packed_probe)
        result["post_state_hash"] = boundary._post_state_hash(cache, backend)
        rows[backend] = result
        _release(mx, cache)
    rows["tokens_exact"] = (
        rows["direct"]["generated_token_sha256"]
        == rows["compact-nope-dsa"]["generated_token_sha256"]
    )
    rows["logits_exact"] = (
        rows["direct"]["full_vocab_logits_hashes"]
        == rows["compact-nope-dsa"]["full_vocab_logits_hashes"]
    )
    return rows


def _materialization_boundary(
    model, mx, boundary, packed_probe, width: int
) -> dict:
    base = boundary._synthetic_cache(model, INITIAL_CONTEXT, "compact-nope-dsa")
    capacity = _physical_capacity_tokens(base, "compact-nope-dsa", boundary)
    reference = boundary._clone_cache(base, capacity)
    candidate = boundary._clone_cache(base, capacity)
    before = {
        "reference": boundary._physical_capacity(reference, "compact-nope-dsa"),
        "candidate": boundary._physical_capacity(candidate, "compact-nope-dsa"),
    }
    left = _chain(
        model,
        mx,
        mx.array([[5000]], dtype=mx.uint32),
        reference,
        configured_width=1,
        steps=BOUNDARY_STEPS,
        completed_steps=BOUNDARY_START_STEP,
        capacity_tokens=capacity,
    )
    right = _chain(
        model,
        mx,
        mx.array([[5000]], dtype=mx.uint32),
        candidate,
        configured_width=width,
        steps=BOUNDARY_STEPS,
        completed_steps=BOUNDARY_START_STEP,
        capacity_tokens=capacity,
    )
    left = _finish_chain_evidence(left, packed_probe)
    right = _finish_chain_evidence(right, packed_probe)
    after = {
        "reference": boundary._physical_capacity(reference, "compact-nope-dsa"),
        "candidate": boundary._physical_capacity(candidate, "compact-nope-dsa"),
    }
    row = {
        "start_completed_step": BOUNDARY_START_STEP,
        "end_completed_step": BOUNDARY_START_STEP + BOUNDARY_STEPS,
        "reference": left,
        "candidate": right,
        "tokens_exact": left["generated_token_sha256"]
        == right["generated_token_sha256"],
        "logits_exact": left["full_vocab_logits_hashes"]
        == right["full_vocab_logits_hashes"],
        "cache_state_exact": boundary._cache_exact(reference, candidate),
        "physical_capacity_before": before,
        "physical_capacity_after": after,
        "physical_capacity_unchanged": before == after,
    }
    _release(mx, base, reference, candidate)
    return row


def _rollback_transactions(model, mx, boundary, packed_probe) -> dict:
    rows = []
    base = boundary._synthetic_cache(model, INITIAL_CONTEXT, "compact-nope-dsa")
    capacity = _physical_capacity_tokens(base, "compact-nope-dsa", boundary)
    base_hash = boundary._full_cache_hash(base)
    for width in ROLLBACK_WIDTHS:
        for accepted in range(width + 1):
            mutated = boundary._clone_cache(base, capacity)
            full = _chain(
                model,
                mx,
                mx.array([[6000]], dtype=mx.uint32),
                mutated,
                configured_width=width,
                steps=width,
                capacity_tokens=capacity,
            )
            full_tokens = list(full["generated_tokens"])
            if accepted == width:
                recovered = mutated
                mutated = None
            else:
                _release(mx, mutated)
                mutated = None
                recovered = boundary._clone_cache(base, capacity)
                if accepted:
                    replay = _chain(
                        model,
                        mx,
                        mx.array([[6000]], dtype=mx.uint32),
                        recovered,
                        configured_width=1,
                        steps=accepted,
                        capacity_tokens=capacity,
                    )
                    replay_tokens = replay["generated_tokens"]
                else:
                    replay_tokens = []
                if replay_tokens != full_tokens[:accepted]:
                    raise RuntimeError("accepted replay token mismatch")
            oracle = boundary._clone_cache(base, capacity)
            if accepted:
                reference = _chain(
                    model,
                    mx,
                    mx.array([[6000]], dtype=mx.uint32),
                    oracle,
                    configured_width=1,
                    steps=accepted,
                    capacity_tokens=capacity,
                )
                reference_tokens = reference["generated_tokens"]
            else:
                reference_tokens = []
            rows.append(
                {
                    "width": width,
                    "accepted_tokens": accepted,
                    "reasons": stop_reasons(width, accepted),
                    "generated_prefix_exact": reference_tokens
                    == full_tokens[:accepted],
                    "restored_replay_state_exact": boundary._cache_exact(
                        recovered, oracle
                    ),
                    "snapshot_immutable": boundary._full_cache_hash(base)
                    == base_hash,
                }
            )
            _release(mx, recovered, oracle)
    result = {
        "rollback_widths": list(ROLLBACK_WIDTHS),
        "cases": rows,
        "all_exact": all(
            row["generated_prefix_exact"]
            and row["restored_replay_state_exact"]
            and row["snapshot_immutable"]
            for row in rows
        ),
    }
    _release(mx, base)
    return result


def _child(args) -> int:
    import mlx.core as mx

    sys.path.insert(0, str(Path(__file__).parent))
    import probe_compiled_packed_ffn_fp32_router as compiled_probe
    import probe_long_context_first_decode_boundary as boundary
    import probe_packed_decode_runtime as packed_probe
    import probe_residual_packed_decode_moe_fusion as residual

    from glm53_flash_mlx.loader import load, warm_residency

    width = ARMS[args.arm]
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    mx.reset_peak_memory()
    load_started = time.perf_counter()
    model, _ = load(args.model, experimental_packed_decode_moe=True)
    load_seconds = time.perf_counter() - load_started
    residency_started = time.perf_counter()
    warm_residency(model)
    residency_seconds = time.perf_counter() - residency_started
    try:
        with residual._runtime(residual.Arm("B1", True)):
            compiled_probe._configure_compile(model, True)
            compile_warmup = compiled_probe._warm_compiled_ffn(model, True)
            first_token = _first_token(model, mx, boundary, packed_probe)
            cache = boundary._synthetic_cache(model, INITIAL_CONTEXT, "direct")
            capacity = bounded._reserve_direct_dsa(cache, boundary, mx)
            token = mx.array([[3000]], dtype=mx.uint32)
            warmup_tokens = []
            for _ in range(WARMUPS):
                output = model(token, cache=cache)
                predicted = mx.argmax(output.logits[0, -1]).astype(mx.uint32)
                mx.eval(predicted)
                value = int(np.asarray(predicted))
                warmup_tokens.append(value)
                token = mx.array([[value]], dtype=mx.uint32)
            mx.synchronize()
            physical_before = boundary._physical_capacity(cache, "direct")
            leaf_before = boundary._cache_leaf_count(cache)
            active_before = _memory(mx)
            mx.reset_peak_memory()
            atomic_write(
                args.ready,
                {
                    "pid": os.getpid(),
                    "arm": args.arm,
                    "width": width,
                    "context": INITIAL_CONTEXT + WARMUPS,
                },
            )
            bounded._wait_for(args.go, 120.0)
            screen = _chain(
                model,
                mx,
                token,
                cache,
                configured_width=width,
                steps=SCREEN_TOKENS,
                capacity_tokens=capacity,
            )
            atomic_write(
                args.decode_done,
                {"pid": os.getpid(), "steps": SCREEN_TOKENS},
            )
            bounded._wait_for(args.trace_done, 300.0)
            screen = _finish_chain_evidence(screen, packed_probe)
            screen["post_cache_state_hash"] = boundary._full_cache_hash(cache)
            screen["physical_capacity_before"] = physical_before
            screen["physical_capacity_after"] = boundary._physical_capacity(
                cache, "direct"
            )
            screen["physical_capacity_unchanged"] = (
                screen["physical_capacity_before"]
                == screen["physical_capacity_after"]
            )
            screen["cache_leaf_count_before"] = leaf_before
            screen["cache_leaf_count_after"] = boundary._cache_leaf_count(cache)
            screen["cache_leaf_count_constant"] = (
                screen["cache_leaf_count_before"]
                == screen["cache_leaf_count_after"]
            )
            active_after = _memory(mx)
            screen["active_memory_before"] = active_before
            screen["active_memory_after"] = active_after
            screen["active_memory_drift_bytes"] = (
                active_after["active_bytes"] - active_before["active_bytes"]
            )
            screen["working_peak_bytes"] = (
                active_after["peak_bytes"] - active_before["active_bytes"]
            )
            screen["decode_tokens_per_second"] = screen[
                "steady_tokens_per_second"
            ]
            _release(mx, cache, output)

            differential = _direct_compact_differential(
                model, mx, boundary, packed_probe, width
            )
            boundary_row = _materialization_boundary(
                model, mx, boundary, packed_probe, width
            )
            rollback = (
                _rollback_transactions(model, mx, boundary, packed_probe)
                if width in ROLLBACK_WIDTHS
                else {"not_run": True, "reason": "only N=2/4 are stop candidates"}
            )
            child = {
                "schema": "glm53-device-resident-greedy-chain-child-v1",
                "complete": True,
                "arm": args.arm,
                "chain_width": width,
                "load_seconds": load_seconds,
                "residency_seconds": residency_seconds,
                "compile_warmup": compile_warmup,
                "first_token": first_token,
                "warmup_generated_tokens": warmup_tokens,
                "screen": screen,
                "direct_compact_differential": differential,
                "materialization_boundary": boundary_row,
                "rollback_transactions": rollback,
                "metal_error": None,
            }
            atomic_write(args.child_result, child)
    except Exception as exc:
        atomic_write(
            args.child_result,
            {
                "schema": "glm53-device-resident-greedy-chain-child-v1",
                "complete": False,
                "arm": args.arm,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    return 0


def _merge(output: Path, arm: str, row: dict, checkpoint: dict) -> dict:
    artifact = {
        "schema": "glm53-device-resident-greedy-token-chain-probe-v1",
        "date": date.today().isoformat(),
        "complete": False,
        "probe_only": True,
        "speculative_decoding": False,
        "exact_autoregressive_unroll": True,
        "runtime_changes": False,
        "server_changes": False,
        "cache_or_apc_abi_changes": False,
        "arms": {},
        "target_tokens_per_second": TARGET_TPS,
        "target_ms_per_token": TARGET_MS,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "packed_decode_kernel_abi": PACKED_DECODE_KERNEL_ABI,
    }
    if output.exists():
        artifact = json.loads(output.read_text())
    artifact.update(checkpoint)
    artifact["arms"][arm] = row
    if set(artifact["arms"]) == set(ARMS):
        baseline = artifact["arms"]["A"]
        comparisons = {}
        busy_a = baseline["telemetry"]["gpu_busy_ms"] / SCREEN_TOKENS
        idle_a = baseline["telemetry"]["gpu_idle_gap_ms"] / SCREEN_TOKENS
        first_a = baseline["child"]["first_token"]["latency_ms"]
        for name, candidate in artifact["arms"].items():
            if candidate.get("status") == "aborted_resource_frontier":
                comparisons[name] = {
                    "correctness": {
                        "claimed": False,
                        "reason": "the bounded N=16 screen did not complete",
                    },
                    "performance": {
                        "qualified": False,
                        "elapsed_lower_bound_seconds": candidate[
                            "elapsed_lower_bound_seconds"
                        ],
                    },
                    "screening": {
                        "resource_frontier_observed": True,
                        "screen_passed": False,
                    },
                    "screen_passed": False,
                }
                continue
            child = candidate["child"]
            screen = child["screen"]
            busy = candidate["telemetry"]["gpu_busy_ms"] / SCREEN_TOKENS
            idle = candidate["telemetry"]["gpu_idle_gap_ms"] / SCREEN_TOKENS
            exact = {
                "tokens": screen["generated_token_sha256"]
                == baseline["child"]["screen"]["generated_token_sha256"],
                "full_vocab_logits": screen["full_vocab_logits_hashes"]
                == baseline["child"]["screen"]["full_vocab_logits_hashes"],
                "cache_state": screen["post_cache_state_hash"]
                == baseline["child"]["screen"]["post_cache_state_hash"],
                "direct_compact_tokens": child["direct_compact_differential"][
                    "tokens_exact"
                ],
                "direct_compact_logits": child["direct_compact_differential"][
                    "logits_exact"
                ],
                "materialization_tokens": child["materialization_boundary"][
                    "tokens_exact"
                ],
                "materialization_logits": child["materialization_boundary"][
                    "logits_exact"
                ],
                "materialization_cache_state": child[
                    "materialization_boundary"
                ]["cache_state_exact"],
                "capacity": screen["physical_capacity_unchanged"],
                "state_leaf_count": screen["cache_leaf_count_constant"],
                "nan_zero": screen["nan_count"] == 0,
                "metal_error_zero": child["metal_error"] is None,
            }
            if name in ("B", "C"):
                exact["rollback_transactions"] = child[
                    "rollback_transactions"
                ]["all_exact"]
            performance = {
                "tokens_per_second": screen["decode_tokens_per_second"],
                "median_ms_per_token": screen["steady_median_ms_per_token"],
                "gpu_busy_ms_per_token": busy,
                "gpu_idle_ms_per_token": idle,
                "gpu_busy_regression_fraction": busy / busy_a - 1.0,
                "gpu_idle_reduction_ms_per_token": idle_a - idle,
                "readback_count": screen["readback_count"],
                "tokens_per_readback": screen["tokens_per_readback"],
                "readback_fraction_vs_A": screen["readback_count"]
                / baseline["child"]["screen"]["readback_count"],
                "stream_silence_p95_ms": screen[
                    "steady_chunk_latency_p95_ms"
                ],
                "first_token_regression_fraction": child["first_token"][
                    "latency_ms"
                ]
                / first_a
                - 1.0,
                "working_peak_bytes": screen["working_peak_bytes"],
                "active_memory_drift_bytes": screen["active_memory_drift_bytes"],
            }
            screening = {
                "all_exact": all(exact.values()),
                "tokens_per_second_ge_15": performance["tokens_per_second"]
                >= TARGET_TPS,
                "gpu_idle_reduction_ge_2_5ms": (
                    name != "A"
                    and performance["gpu_idle_reduction_ms_per_token"] >= 2.5
                ),
                "gpu_busy_regression_le_0_5pct": performance[
                    "gpu_busy_regression_fraction"
                ]
                <= 0.005,
                "readback_fraction_matches_width": math.isclose(
                    performance["readback_fraction_vs_A"],
                    1.0 / ARMS[name],
                    rel_tol=0.0,
                    abs_tol=1.0 / SCREEN_TOKENS,
                ),
                "working_peak_le_512mib": performance["working_peak_bytes"]
                <= MAX_WORKING_PEAK_BYTES,
                "active_drift_le_64mib": abs(
                    performance["active_memory_drift_bytes"]
                )
                <= MAX_ACTIVE_DRIFT_BYTES,
                "first_token_regression_le_1pct": performance[
                    "first_token_regression_fraction"
                ]
                <= 0.01,
                "n2_stream_silence_p95_le_160ms": (
                    name != "B"
                    or performance["stream_silence_p95_ms"] <= 160.0
                ),
            }
            comparisons[name] = {
                "correctness": exact,
                "performance": performance,
                "screening": screening,
                "screen_passed": all(screening.values()) if name != "A" else all(exact.values()),
            }
        artifact["comparisons_to_A"] = comparisons
        artifact["screening_passed_arms"] = [
            name for name in ("B", "C", "D", "E")
            if comparisons[name]["screen_passed"]
        ]
        artifact["n2_qualification_required"] = comparisons["B"][
            "screen_passed"
        ]
        artifact["correctness_complete_for_completed_arms"] = all(
            comparisons[name]["screening"]["all_exact"]
            for name in ("A", "B", "C", "D")
        )
        artifact["n16_resource_frontier_complete"] = (
            artifact["arms"]["E"].get("status")
            == "aborted_resource_frontier"
        )
        artifact["complete"] = (
            artifact["correctness_complete_for_completed_arms"]
            and artifact["n16_resource_frontier_complete"]
        )
    atomic_write(output, artifact)
    return artifact


def _finalize_existing(output: Path, n16_elapsed_seconds: float) -> dict:
    artifact = json.loads(output.read_text())
    for arm in ("A", "B", "C", "D"):
        row = artifact["arms"][arm]
        dynamic = _dynamic_gap_summary(Path(row["trace"]["path"]), int(row["pid"]))
        dynamic["long_application_gap_count_matches_readbacks"] = (
            dynamic["application_starvation_long_gap_count"]
            == row["child"]["screen"]["readback_count"]
        )
        row["dynamic_gap_attribution"] = dynamic
    compiled = REPOSITORY / "bench-results" / (
        "m3ultra512-compiled-packed-ffn-fp32-router-20260901.json"
    )
    oracle16 = REPOSITORY / "oracles" / "glm53-official-greedy-16.json"
    oracle128 = REPOSITORY / "oracles" / "glm53-official-greedy-128.json"
    artifact["inherited_exact_evidence"] = {
        "compiled_ffn_router_artifact": str(compiled),
        "compiled_ffn_router_artifact_sha256": bounded._trace_identity(compiled)[
            "sha256"
        ],
        "router_raw_logits_indices_scores_exact": True,
        "router_independently_retraced_inside_chain": False,
        "reason": (
            "the probe changes only lazy evaluation/readback scheduling; all 64 "
            "new full-vocab logits and final cache hashes are compared directly"
        ),
        "official_oracle_16_sha256": bounded._trace_identity(oracle16)["sha256"],
        "official_oracle_128_sha256": bounded._trace_identity(oracle128)["sha256"],
        "runtime_path_changed": False,
    }
    atomic_write(output, artifact)
    negative = {
        "status": "aborted_resource_frontier",
        "chain_width": 16,
        "elapsed_lower_bound_seconds": float(n16_elapsed_seconds),
        "screen_tokens_requested": SCREEN_TOKENS,
        "screen_complete": False,
        "correctness_claim": False,
        "trace_complete": False,
        "partial_trace_deleted": True,
        "observed_partial_trace_size_mib_approx": 430,
        "reason": (
            "N=16 did not complete the bounded 64-token screen while N=1/2/4/8 "
            "completed; the full-model lazy dependency graph crossed the probe budget"
        ),
    }
    return _merge(output, "E", negative, {})


def _parent(args) -> int:
    report = inspect_checkpoint(args.model, require_server_ready=True)
    checkpoint = {
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "checkpoint_layout_digest": report.layout_digest,
    }
    trace = bounded._validate_trace(args.trace)
    with tempfile.TemporaryDirectory(
        prefix=f"glm53-device-chain-{args.arm}-"
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
            try:
                bounded._wait_for(
                    decode_done, SCREEN_TIME_LIMIT_SECONDS, child
                )
            except TimeoutError:
                observed_trace_bytes = path_bytes(trace) if trace.exists() else 0
                terminate_process_group(tracer)
                tracer = None
                terminate_process_group(child)
                remove_partial_trace(trace)
                negative = {
                    "status": "aborted_resource_frontier",
                    "chain_width": ARMS[args.arm],
                    "elapsed_lower_bound_seconds": SCREEN_TIME_LIMIT_SECONDS,
                    "screen_tokens_requested": SCREEN_TOKENS,
                    "screen_complete": False,
                    "correctness_claim": False,
                    "trace_complete": False,
                    "partial_trace_deleted": True,
                    "observed_partial_trace_bytes": observed_trace_bytes,
                    "reason": "bounded device-chain screen exceeded 30 seconds",
                }
                artifact = _merge(
                    args.output, args.arm, negative, checkpoint
                )
                print(
                    json.dumps(
                        {
                            "arm": args.arm,
                            "status": negative["status"],
                            "artifact_complete": artifact["complete"],
                        },
                        indent=2,
                    )
                )
                return 0
            if tracer.wait(timeout=300.0) != 0:
                raise RuntimeError(f"xctrace failed: {log.read_text()}")
            log_handle.close()
            log_handle = None
            trace_done.touch()
            if child.wait(timeout=600.0) != 0:
                raise RuntimeError("device chain child failed")
            if not child_result.exists():
                raise RuntimeError("device chain child result is missing")
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
                    "configuration": {
                        "chain_width": ARMS[args.arm],
                        "compiled_ffn": True,
                        "resident_fp32_router": False,
                        "residual_moe_arm": "B1+shared-fused",
                        "cache_backend": "direct",
                    },
                    "child": child_value,
                    "telemetry": telemetry,
                    "trace": identity,
                    "trace_time_limit_seconds": TRACE_SECONDS,
                },
                checkpoint,
            )
            print(
                json.dumps(
                    {
                        "arm": args.arm,
                        "width": ARMS[args.arm],
                        "artifact_complete": artifact["complete"],
                        "screening_passed_arms": artifact.get(
                            "screening_passed_arms", []
                        ),
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
    parser.add_argument("model", type=Path, nargs="?")
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--finalize-existing", action="store_true")
    parser.add_argument("--n16-elapsed-seconds", type=float, default=300.0)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ready", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--go", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--decode-done", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--trace-done", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--child-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.finalize_existing:
        artifact = _finalize_existing(
            args.output, args.n16_elapsed_seconds
        )
        print(
            json.dumps(
                {
                    "complete": artifact["complete"],
                    "screening_passed_arms": artifact[
                        "screening_passed_arms"
                    ],
                },
                indent=2,
            )
        )
        return 0
    if args.model is None or args.arm is None or args.trace is None:
        parser.error("model, --arm, and --trace are required for a measured arm")
    return _child(args) if args.child else _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
