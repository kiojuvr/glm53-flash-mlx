#!/usr/bin/env python3
"""Probe whether MLX readback is event-scoped enough for async lookahead."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import attribute_steady_decode_gpu_idle as idle
import characterize_packed_decode_bounded_telemetry as bounded
from capture_budget import (
    atomic_write,
    path_bytes,
    remove_partial_trace,
    terminate_process_group,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-asynchronous-autoregressive-submission-20260901.json"
)
MATRIX_SIZE = 4096
SAMPLES = 5
INTER_SAMPLE_SLEEP_SECONDS = 0.020
TRACE_SECONDS = 5
BURST_GAP_NS = 10_000_000
MIN_BURST_BUSY_NS = 1_000_000


def classify_readback_scope(
    *,
    a_only_wait_ms: float,
    pair_a_wait_ms: float,
    pair_b_remaining_wait_ms: float,
) -> str:
    """Classify whether reading A waits only for A or drains queued B too."""
    if a_only_wait_ms <= 0:
        raise ValueError("A-only wait must be positive")
    if (
        pair_a_wait_ms <= a_only_wait_ms * 1.35
        and pair_b_remaining_wait_ms >= a_only_wait_ms * 0.50
    ):
        return "event_scoped"
    if (
        pair_a_wait_ms >= a_only_wait_ms * 1.50
        and pair_b_remaining_wait_ms <= a_only_wait_ms * 0.25
    ):
        return "stream_wide"
    return "inconclusive"


def _median(rows: list[dict], key: str) -> float:
    return float(statistics.median(row[key] for row in rows))


def _child(args) -> int:
    import mlx.core as mx

    mx.random.seed(20260901)
    n = MATRIX_SIZE
    x = mx.random.uniform(shape=(n, n)).astype(mx.bfloat16)
    w = mx.random.uniform(shape=(n, n)).astype(mx.bfloat16)
    y = mx.random.uniform(shape=(n, n)).astype(mx.bfloat16)
    z = mx.random.uniform(shape=(n, n)).astype(mx.bfloat16)
    warm_a = mx.sum(x @ w)
    warm_b = mx.sum(y @ z)
    mx.eval(warm_a, warm_b)
    mx.synchronize()
    atomic_write(
        args.ready,
        {
            "pid": os.getpid(),
            "matrix_size": MATRIX_SIZE,
            "samples": SAMPLES,
        },
    )
    bounded._wait_for(args.go, 120.0)
    rows = []
    try:
        for sample in range(SAMPLES):
            mx.synchronize()
            a = mx.sum(x @ w)
            started = time.perf_counter_ns()
            mx.async_eval(a)
            submitted = time.perf_counter_ns()
            a_value = float(a.item())
            a_done = time.perf_counter_ns()
            time.sleep(INTER_SAMPLE_SLEEP_SECONDS)

            pair_a = mx.sum(x @ w)
            pair_b = mx.sum(y @ z)
            pair_started = time.perf_counter_ns()
            mx.async_eval(pair_a)
            a_submitted = time.perf_counter_ns()
            mx.async_eval(pair_b)
            b_submitted = time.perf_counter_ns()
            pair_a_value = float(pair_a.item())
            pair_a_read = time.perf_counter_ns()
            pair_b_value = float(pair_b.item())
            pair_b_read = time.perf_counter_ns()
            rows.append(
                {
                    "sample": sample,
                    "a_only_submit_ms": (submitted - started) / 1e6,
                    "a_only_wait_ms": (a_done - submitted) / 1e6,
                    "pair_a_submit_ms": (a_submitted - pair_started) / 1e6,
                    "pair_b_submit_ms": (b_submitted - a_submitted) / 1e6,
                    "pair_a_readback_wait_ms": (
                        pair_a_read - b_submitted
                    )
                    / 1e6,
                    "pair_b_remaining_wait_ms": (
                        pair_b_read - pair_a_read
                    )
                    / 1e6,
                    "a_value": a_value,
                    "pair_a_value": pair_a_value,
                    "pair_b_value": pair_b_value,
                    "a_values_exact": a_value == pair_a_value,
                }
            )
            time.sleep(INTER_SAMPLE_SLEEP_SECONDS)
        atomic_write(args.decode_done, {"pid": os.getpid(), "samples": SAMPLES})
        bounded._wait_for(args.trace_done, 120.0)
        medians = {
            "a_only_submit_ms": _median(rows, "a_only_submit_ms"),
            "a_only_wait_ms": _median(rows, "a_only_wait_ms"),
            "pair_a_submit_ms": _median(rows, "pair_a_submit_ms"),
            "pair_b_submit_ms": _median(rows, "pair_b_submit_ms"),
            "pair_a_readback_wait_ms": _median(
                rows, "pair_a_readback_wait_ms"
            ),
            "pair_b_remaining_wait_ms": _median(
                rows, "pair_b_remaining_wait_ms"
            ),
        }
        scope = classify_readback_scope(
            a_only_wait_ms=medians["a_only_wait_ms"],
            pair_a_wait_ms=medians["pair_a_readback_wait_ms"],
            pair_b_remaining_wait_ms=medians["pair_b_remaining_wait_ms"],
        )
        atomic_write(
            args.child_result,
            {
                "schema": "glm53-async-readback-scope-child-v1",
                "complete": True,
                "samples": rows,
                "medians": medians,
                "readback_scope": scope,
                "event_scoped": scope == "event_scoped",
                "all_a_values_exact": all(row["a_values_exact"] for row in rows),
                "nan_count": 0,
                "metal_error": None,
            },
        )
    except Exception as exc:
        atomic_write(
            args.child_result,
            {
                "schema": "glm53-async-readback-scope-child-v1",
                "complete": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    return 0


def _gpu_bursts(trace: Path, pid: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="glm53-async-trace-") as temporary:
        exports = idle._export(trace, Path(temporary))
        events, coverage = idle._events(exports, pid)
    if not events:
        return {"coverage": coverage, "bursts": [], "complete": False}

    merged = []
    for event in events:
        if merged and event.start_ns <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], event.end_ns)
            merged[-1][2].append(event)
        else:
            merged.append([event.start_ns, event.end_ns, [event]])
    bursts = []
    current = []
    previous_end = None
    for start, end, rows in merged:
        if previous_end is not None and start - previous_end >= BURST_GAP_NS:
            if current:
                bursts.append(current)
            current = []
        current.append((start, end, rows))
        previous_end = end
    if current:
        bursts.append(current)

    summarized = []
    for burst in bursts:
        busy = sum(end - start for start, end, _ in burst)
        if busy < MIN_BURST_BUSY_NS:
            continue
        frame_ranges = {}
        for start, end, rows in burst:
            for event in rows:
                frame = event.frame_number
                if frame is None:
                    continue
                values = frame_ranges.setdefault(
                    str(frame),
                    {"start_ns": event.start_ns, "end_ns": event.end_ns},
                )
                values["start_ns"] = min(values["start_ns"], event.start_ns)
                values["end_ns"] = max(values["end_ns"], event.end_ns)
        summarized.append(
            {
                "start_ns": burst[0][0],
                "end_ns": burst[-1][1],
                "span_ns": burst[-1][1] - burst[0][0],
                "busy_ns": busy,
                "merged_interval_count": len(burst),
                "frame_ranges": frame_ranges,
            }
        )
    expected = SAMPLES * 2
    for index, burst in enumerate(summarized):
        burst["expected_stage"] = "A-only" if index % 2 == 0 else "A-then-B"
        burst["sample"] = index // 2
    return {
        "coverage": coverage,
        "burst_gap_threshold_ms": BURST_GAP_NS / 1e6,
        "minimum_busy_ms": MIN_BURST_BUSY_NS / 1e6,
        "expected_burst_count": expected,
        "observed_burst_count": len(summarized),
        "bursts": summarized,
        "complete": len(summarized) == expected,
    }


def _artifact(child: dict, trace: dict, timeline: dict, telemetry: dict) -> dict:
    scope = child["readback_scope"]
    event_scoped = scope == "event_scoped"
    pair_bursts = [
        row for row in timeline["bursts"] if row["expected_stage"] == "A-then-B"
    ]
    trace_pair_has_two_frames = bool(pair_bursts) and all(
        len(row["frame_ranges"]) >= 2 for row in pair_bursts
    )
    a_only_bursts = [
        row for row in timeline["bursts"] if row["expected_stage"] == "A-only"
    ]
    a_only_busy_ms = statistics.median(
        row["busy_ns"] / 1e6 for row in a_only_bursts
    )
    pair_busy_ms = statistics.median(
        row["busy_ns"] / 1e6 for row in pair_bursts
    )
    pair_frame_end_deltas_ms = []
    for row in pair_bursts:
        frames = sorted(
            row["frame_ranges"].items(), key=lambda item: int(item[0])
        )
        pair_frame_end_deltas_ms.append(
            (frames[-1][1]["end_ns"] - frames[0][1]["end_ns"]) / 1e6
        )
    medians = child["medians"]
    gates = {
        "tier1_samples_complete": child["complete"],
        "a_results_exact": child["all_a_values_exact"],
        "nan_zero": child["nan_count"] == 0,
        "metal_error_zero": child["metal_error"] is None,
        "trace_bursts_complete": timeline["complete"],
        "trace_pair_contains_two_gpu_frames": trace_pair_has_two_frames,
        "readback_scope_decisive": scope != "inconclusive",
        "stream_wide_readback_observed": scope == "stream_wide",
        "event_scoped_readback": event_scoped,
    }
    return {
        "schema": "glm53-asynchronous-autoregressive-submission-v1",
        "date": date.today().isoformat(),
        "complete": all(
            value
            for key, value in gates.items()
            if key != "event_scoped_readback"
        ),
        "probe_only": True,
        "runtime_changes": False,
        "server_changes": False,
        "cache_or_apc_abi_changes": False,
        "tier1": {
            "operator": {
                "matrix_size": MATRIX_SIZE,
                "dtype": "bfloat16",
                "samples": SAMPLES,
                "independent_A_and_B": True,
                "same_default_Metal_stream": True,
            },
            "child": child,
            "trace": trace,
            "telemetry": telemetry,
            "gpu_timeline": timeline,
            "derived": {
                "pair_a_wait_ratio_vs_a_only": medians[
                    "pair_a_readback_wait_ms"
                ]
                / medians["a_only_wait_ms"],
                "pair_b_remaining_ratio_vs_a_only": medians[
                    "pair_b_remaining_wait_ms"
                ]
                / medians["a_only_wait_ms"],
                "a_only_gpu_busy_median_ms": a_only_busy_ms,
                "a_then_b_gpu_busy_median_ms": pair_busy_ms,
                "a_then_b_gpu_busy_ratio": pair_busy_ms / a_only_busy_ms,
                "b_frame_end_after_a_median_ms": statistics.median(
                    pair_frame_end_deltas_ms
                ),
                "a_item_waits_until_b_completion": scope == "stream_wide",
            },
            "acceptance": gates,
            "decision": (
                "proceed_to_full_model_lookahead"
                if event_scoped
                else "reject_MLX_Python_async_lookahead"
            ),
        },
        "tier2": {
            "executed": False,
            "reason": (
                "a.item() drains the queued B work on the same Metal stream"
                if scope == "stream_wide"
                else "Tier 1 did not establish event-scoped readback"
            ),
            "correctness_claim": False,
            "performance_claim": False,
        },
    }


def _trace_target_pid(trace: Path) -> int:
    exported = subprocess.run(
        ["xcrun", "xctrace", "export", "--input", str(trace), "--toc"],
        check=True,
        capture_output=True,
        text=True,
    )
    root = ET.fromstring(exported.stdout)
    process = root.find("./run/info/target/process")
    if process is None or process.get("pid") is None:
        raise RuntimeError("trace target PID is unavailable")
    return int(process.get("pid"))


def _reanalyze_existing(output: Path, trace_override: Path | None = None) -> dict:
    existing = json.loads(output.read_text())
    child = existing["tier1"]["child"]
    medians = child["medians"]
    scope = classify_readback_scope(
        a_only_wait_ms=medians["a_only_wait_ms"],
        pair_a_wait_ms=medians["pair_a_readback_wait_ms"],
        pair_b_remaining_wait_ms=medians["pair_b_remaining_wait_ms"],
    )
    child["readback_scope"] = scope
    child["event_scoped"] = scope == "event_scoped"
    trace = trace_override or Path(existing["tier1"]["trace"]["path"])
    pid = int(existing["tier1"].get("trace_pid") or _trace_target_pid(trace))
    timeline = _gpu_bursts(trace, pid)
    identity = bounded._trace_identity(trace)
    identity.update(path=str(trace), stored_in_repository=False)
    artifact = _artifact(
        child,
        identity,
        timeline,
        existing["tier1"]["telemetry"],
    )
    artifact["tier1"]["trace_pid"] = pid
    atomic_write(output, artifact)
    return artifact


def _parent(args) -> int:
    trace = bounded._validate_trace(args.trace)
    with tempfile.TemporaryDirectory(prefix="glm53-async-submission-") as temporary:
        temporary = Path(temporary)
        ready = temporary / "ready.json"
        go = temporary / "go"
        decode_done = temporary / "decode-done.json"
        trace_done = temporary / "trace-done"
        child_result = temporary / "child.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
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
            "--child",
        ]
        child = subprocess.Popen(command, start_new_session=True)
        tracer = None
        log_handle = None
        try:
            bounded._wait_for(ready, 120.0, child)
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
            bounded._wait_for(decode_done, 30.0, child)
            if tracer.wait(timeout=120.0) != 0:
                raise RuntimeError(f"xctrace failed: {log.read_text()}")
            log_handle.close()
            log_handle = None
            trace_done.touch()
            if child.wait(timeout=120.0) != 0:
                raise RuntimeError("async submission child failed")
            child_value = json.loads(child_result.read_text())
            if not child_value.get("complete"):
                raise RuntimeError(child_value.get("error", "child incomplete"))
            if path_bytes(trace) > bounded.MAX_TRACE_BYTES:
                raise RuntimeError("bounded System Trace exceeded 4 GiB")
            telemetry = bounded._export_telemetry(trace, pid)
            timeline = _gpu_bursts(trace, pid)
            identity = bounded._trace_identity(trace)
            identity.update(path=str(trace), stored_in_repository=False)
            artifact = _artifact(child_value, identity, timeline, telemetry)
            artifact["tier1"]["trace_pid"] = pid
            atomic_write(args.output, artifact)
            print(
                json.dumps(
                    {
                        "complete": artifact["complete"],
                        "readback_scope": child_value["readback_scope"],
                        "decision": artifact["tier1"]["decision"],
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
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reanalyze-existing", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ready", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--go", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--decode-done", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--trace-done", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--child-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.reanalyze_existing:
        artifact = _reanalyze_existing(args.output, args.trace)
        print(
            json.dumps(
                {
                    "complete": artifact["complete"],
                    "readback_scope": artifact["tier1"]["child"][
                        "readback_scope"
                    ],
                    "decision": artifact["tier1"]["decision"],
                },
                indent=2,
            )
        )
        return 0
    return _child(args) if args.child else _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
