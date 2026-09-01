#!/usr/bin/env python3
"""Record bounded, non-replayable Metal telemetry for exact packed decode."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from capture_budget import atomic_write, path_bytes, remove_partial_trace, terminate_process_group
from capture_steady_packed_decode_critical_path import _trace_identity


ARMS = {"A": False, "B": True}
INITIAL_CONTEXT = 2049
WARMUPS = 2
TELEMETRY_TOKENS = 272
TRACED_TOKENS = 16
MATERIALIZATION_STEP = 256
TRACE_TIME_LIMIT_SECONDS = 8
MAX_TRACE_BYTES = 4 << 30
MIN_FREE_BYTES = 64 << 30
REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-packed-decode-bounded-telemetry-20260901.json"
)
EXPORT_SCHEMAS = (
    "metal-gpu-intervals",
    "metal-application-command-buffer-submissions",
    "metal-application-intervals",
    "metal-driver-intervals",
    "metal-resource-allocations",
)


def _round_up(value: int, step: int = 256) -> int:
    return ((int(value) + step - 1) // step) * step


def _reserve_direct_dsa(cache, boundary, mx) -> int:
    capacity = _round_up(INITIAL_CONTEXT + WARMUPS + TELEMETRY_TOKENS + 1)
    targets = []
    for layer in boundary.EXPECTED_DSA:
        latent, indexer = cache[layer]
        for entry in (latent, indexer):
            for name in ("keys", "values"):
                current = getattr(entry, name)
                if current is None or int(current.shape[2]) >= capacity:
                    continue
                shape = list(current.shape)
                shape[2] = capacity
                reserved = mx.zeros(shape, dtype=current.dtype)
                reserved[..., : int(entry.offset), :] = current[
                    ..., : int(entry.offset), :
                ]
                setattr(entry, name, reserved)
                targets.append(reserved)
    mx.eval(*targets)
    mx.clear_cache()
    mx.synchronize()
    return capacity


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(q * len(ordered)) - 1)])


def _validate_trace(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.suffix != ".trace":
        raise ValueError("--trace must end in .trace")
    if path.is_relative_to(REPOSITORY):
        raise ValueError("Metal System Trace must live outside the repository")
    if path.exists():
        raise FileExistsError(f"trace path already exists: {path}")
    if shutil.disk_usage(path.parent).free < MIN_FREE_BYTES:
        raise RuntimeError("less than 64 GiB is free at the trace destination")
    return path


def _wait_for(path: Path, timeout_s: float, process=None) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"process exited before creating {path}")
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for {path}")


def _child(args) -> int:
    import mlx.core as mx

    sys.path.insert(0, str(Path(__file__).parent))
    import probe_compiled_packed_ffn_fp32_router as compiled_probe
    import probe_long_context_first_decode_boundary as boundary
    import probe_packed_decode_runtime as packed_probe
    import probe_residual_packed_decode_moe_fusion as residual

    from glm53_flash_mlx.loader import load, warm_residency

    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    model, _ = load(args.model, experimental_packed_decode_moe=True)
    warm_residency(model)
    with residual._runtime(residual.Arm("B1", True)):
        compiled_probe._configure_compile(model, ARMS[args.arm])
        compile_warmup = compiled_probe._warm_compiled_ffn(model, ARMS[args.arm])
        cache = boundary._synthetic_cache(model, INITIAL_CONTEXT, "direct")
        configured_capacity = _reserve_direct_dsa(cache, boundary, mx)
        token = mx.array([[3000]], dtype=mx.uint32)
        warmup_tokens = []
        for _ in range(WARMUPS):
            output = model(token, cache=cache)
            predicted = mx.argmax(output.logits[0, -1])
            mx.eval(output.logits, predicted)
            mx.synchronize()
            value = int(predicted.item())
            warmup_tokens.append(value)
            token = mx.array([[value]], dtype=mx.uint32)
        physical_before = boundary._physical_capacity(cache, "direct")
        atomic_write(
            args.ready,
            {
                "pid": os.getpid(),
                "arm": args.arm,
                "context": INITIAL_CONTEXT + WARMUPS,
                "compile_warmup": compile_warmup,
                "configured_capacity_tokens": configured_capacity,
            },
        )
        _wait_for(args.go, 120.0)
        latencies = []
        generated = []
        evidence_logits = {}
        materialization_ms = None
        started_all = time.perf_counter()
        for step in range(1, TRACED_TOKENS + 1):
            started = time.perf_counter()
            output = model(token, cache=cache)
            logits = output.logits[0, -1]
            predicted = mx.argmax(logits)
            mx.eval(output.logits, predicted)
            mx.synchronize()
            value = int(predicted.item())
            latencies.append((time.perf_counter() - started) * 1000.0)
            generated.append(value)
            if step in (1, 16):
                evidence_logits[str(step)] = logits
            token = mx.array([[value]], dtype=mx.uint32)
        atomic_write(
            args.decode_done,
            {
                "pid": os.getpid(),
                "steps": TRACED_TOKENS,
                "elapsed_seconds": time.perf_counter() - started_all,
            },
        )
        _wait_for(args.trace_done, 300.0)
        for step in range(TRACED_TOKENS + 1, TELEMETRY_TOKENS + 1):
            started = time.perf_counter()
            output = model(token, cache=cache)
            logits = output.logits[0, -1]
            predicted = mx.argmax(logits)
            mx.eval(output.logits, predicted)
            mx.synchronize()
            value = int(predicted.item())
            latencies.append((time.perf_counter() - started) * 1000.0)
            generated.append(value)
            if step in (MATERIALIZATION_STEP, TELEMETRY_TOKENS):
                evidence_logits[str(step)] = logits
            if step == MATERIALIZATION_STEP:
                materialized = time.perf_counter()
                mx.eval([entry.state for entry in cache])
                mx.clear_cache()
                mx.synchronize()
                materialization_ms = (time.perf_counter() - materialized) * 1000.0
            token = mx.array([[value]], dtype=mx.uint32)
        elapsed = sum(latencies) / 1000.0
        physical_after = boundary._physical_capacity(cache, "direct")
        evidence_hashes = {
            step: packed_probe._hash(logits)
            for step, logits in evidence_logits.items()
        }
        result = {
            "schema": "glm53-bounded-metal-telemetry-child-v1",
            "arm": args.arm,
            "pid": os.getpid(),
            "initial_context": INITIAL_CONTEXT,
            "warmups": WARMUPS,
            "telemetry_tokens": TELEMETRY_TOKENS,
            "system_trace_tokens": TRACED_TOKENS,
            "materialization_step": MATERIALIZATION_STEP,
            "materialization_count": 1,
            "materialization_ms": materialization_ms,
            "warmup_generated_tokens": warmup_tokens,
            "generated_token_sha256": packed_probe._token_digest(generated),
            "evidence_full_vocab_hashes": evidence_hashes,
            "post_cache_state_hash": boundary._full_cache_hash(cache),
            "latency_samples_ms": latencies,
            "latency_p50_ms": statistics.median(latencies),
            "latency_p95_ms": _percentile(latencies, 0.95),
            "decode_tokens_per_second": TELEMETRY_TOKENS / elapsed,
            "physical_capacity_before": physical_before,
            "physical_capacity_after": physical_after,
            "physical_capacity_unchanged": physical_before == physical_after,
            "configured_capacity_tokens": configured_capacity,
        }
        atomic_write(args.child_result, result)
    return 0


def _value(element, ids: dict[str, ET.Element]) -> tuple[str | None, str | None]:
    if ref := element.get("ref"):
        element = ids.get(ref, element)
    raw = (element.text or "").strip() or None
    return raw, element.get("fmt")


def _parse_export(path: Path, pid: int) -> list[dict]:
    root = ET.parse(path).getroot()
    node = root.find(".//node")
    if node is None:
        return []
    schema = node.find("schema")
    if schema is None:
        return []
    mnemonics = [
        column.findtext("mnemonic", default=f"column_{index}")
        for index, column in enumerate(schema.findall("col"))
    ]
    ids = {
        element.get("id"): element
        for element in node.iter()
        if element.get("id") is not None
    }
    rows = []
    needles = (f"({pid})", f"pid: {pid}")
    for element in node.findall("row"):
        values = {}
        formats = []
        for index, child in enumerate(list(element)):
            name = mnemonics[index] if index < len(mnemonics) else f"column_{index}"
            raw, formatted = _value(child, ids)
            values[name] = {"raw": raw, "fmt": formatted}
            if formatted:
                formats.append(formatted)
        if any(needle in formatted for formatted in formats for needle in needles):
            rows.append(values)
    return rows


def _integer(row: dict, name: str) -> int | None:
    raw = row.get(name, {}).get("raw")
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


def _merged_interval_stats(rows: list[dict]) -> dict:
    intervals = []
    for row in rows:
        start = _integer(row, "start")
        duration = _integer(row, "duration")
        if start is not None and duration is not None and duration >= 0:
            intervals.append((start, start + duration))
    if not intervals:
        return {
            "gpu_interval_count": 0,
            "gpu_busy_ms": 0.0,
            "gpu_idle_gap_ms": 0.0,
            "gpu_busy_ratio": 0.0,
            "gpu_gap_p50_ms": 0.0,
            "gpu_gap_p95_ms": 0.0,
        }
    intervals.sort()
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    busy = sum(end - start for start, end in merged)
    span = merged[-1][1] - merged[0][0]
    gaps = [merged[index + 1][0] - merged[index][1] for index in range(len(merged) - 1)]
    return {
        "gpu_interval_count": len(intervals),
        "gpu_busy_ms": busy / 1e6,
        "gpu_idle_gap_ms": (span - busy) / 1e6,
        "gpu_busy_ratio": busy / span if span else 0.0,
        "gpu_gap_p50_ms": statistics.median(gaps) / 1e6 if gaps else 0.0,
        "gpu_gap_p95_ms": _percentile(gaps, 0.95) / 1e6 if gaps else 0.0,
    }


def _submission_stats(rows: list[dict]) -> dict:
    starts = sorted(value for row in rows if (value := _integer(row, "start")) is not None)
    gaps = [right - left for left, right in zip(starts, starts[1:])]
    return {
        "command_buffer_submission_rows": len(rows),
        "cpu_inter_submission_p50_ms": statistics.median(gaps) / 1e6 if gaps else 0.0,
        "cpu_inter_submission_p95_ms": _percentile(gaps, 0.95) / 1e6 if gaps else 0.0,
    }


def _export_telemetry(trace: Path, pid: int) -> dict:
    tables = {}
    with tempfile.TemporaryDirectory(prefix="glm53-xctrace-export-") as temporary:
        temporary = Path(temporary)
        for schema in EXPORT_SCHEMAS:
            output = temporary / f"{schema}.xml"
            subprocess.run(
                [
                    "xcrun",
                    "xctrace",
                    "export",
                    "--input",
                    str(trace),
                    "--xpath",
                    f"/trace-toc/run[@number='1']/data/table[@schema='{schema}']",
                    "--output",
                    str(output),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            tables[schema] = _parse_export(output, pid)
    gpu = _merged_interval_stats(tables["metal-gpu-intervals"])
    submissions = _submission_stats(
        tables["metal-application-command-buffer-submissions"]
    )
    labels = []
    for row in tables["metal-application-intervals"]:
        for value in row.values():
            formatted = value.get("fmt")
            if formatted and "Command" in formatted:
                labels.append(formatted)
                break
    return {
        **gpu,
        **submissions,
        "metal_application_interval_rows": len(
            tables["metal-application-intervals"]
        ),
        "metal_driver_interval_rows": len(tables["metal-driver-intervals"]),
        "resource_allocation_rows": len(tables["metal-resource-allocations"]),
        "compute_or_command_interval_count": len(labels),
        "application_label_examples": sorted(set(labels))[:16],
    }


def _merge(output: Path, arm: str, row: dict) -> dict:
    artifact = {
        "schema": "glm53-packed-decode-bounded-telemetry-v1",
        "complete": False,
        "probe_only": True,
        "replayable": False,
        "arms": {},
        "runtime_changes": False,
    }
    if output.exists():
        artifact = json.loads(output.read_text())
    artifact["arms"][arm] = row
    if set(artifact["arms"]) == set(ARMS):
        a, b = artifact["arms"]["A"], artifact["arms"]["B"]
        exact = {
            "generated_tokens_exact": a["child"]["generated_token_sha256"]
            == b["child"]["generated_token_sha256"],
            "evidence_logits_exact": a["child"]["evidence_full_vocab_hashes"]
            == b["child"]["evidence_full_vocab_hashes"],
            "post_cache_state_exact": a["child"]["post_cache_state_hash"]
            == b["child"]["post_cache_state_hash"],
            "capacity_unchanged": a["child"]["physical_capacity_unchanged"]
            and b["child"]["physical_capacity_unchanged"],
        }
        artifact["correctness"] = exact
        artifact["complete"] = all(exact.values())
        artifact["compiled_vs_eager"] = {
            "decode_tps_speedup": b["child"]["decode_tokens_per_second"]
            / a["child"]["decode_tokens_per_second"],
            "gpu_busy_ms_ratio": b["telemetry"]["gpu_busy_ms"]
            / a["telemetry"]["gpu_busy_ms"]
            if a["telemetry"]["gpu_busy_ms"]
            else None,
            "gpu_interval_delta": b["telemetry"]["gpu_interval_count"]
            - a["telemetry"]["gpu_interval_count"],
            "command_buffer_submission_delta": b["telemetry"][
                "command_buffer_submission_rows"
            ]
            - a["telemetry"]["command_buffer_submission_rows"],
            "gpu_busy_ms_per_token_A": a["telemetry"]["gpu_busy_ms"]
            / TRACED_TOKENS,
            "gpu_busy_ms_per_token_B": b["telemetry"]["gpu_busy_ms"]
            / TRACED_TOKENS,
            "gpu_idle_ms_per_token_A": a["telemetry"]["gpu_idle_gap_ms"]
            / TRACED_TOKENS,
            "gpu_idle_ms_per_token_B": b["telemetry"]["gpu_idle_gap_ms"]
            / TRACED_TOKENS,
            "gpu_idle_reduction_ms_per_token": (
                a["telemetry"]["gpu_idle_gap_ms"]
                - b["telemetry"]["gpu_idle_gap_ms"]
            )
            / TRACED_TOKENS,
        }
    atomic_write(output, artifact)
    return artifact


def _parent(args) -> int:
    trace = _validate_trace(args.trace)
    with tempfile.TemporaryDirectory(prefix=f"glm53-telemetry-{args.arm}-") as temporary:
        temporary = Path(temporary)
        ready = temporary / "ready.json"
        go = temporary / "go"
        child_result = temporary / "result.json"
        decode_done = temporary / "decode-done.json"
        trace_done = temporary / "trace-done"
        child_command = [
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
            "--child-result",
            str(child_result),
            "--decode-done",
            str(decode_done),
            "--trace-done",
            str(trace_done),
            "--wired-limit-gb",
            str(args.wired_limit_gb),
            "--cache-limit-gb",
            str(args.cache_limit_gb),
            "--child",
        ]
        child = subprocess.Popen(child_command, start_new_session=True)
        tracer = None
        try:
            _wait_for(ready, 600.0, child)
            ready_value = json.loads(ready.read_text())
            pid = int(ready_value["pid"])
            xctrace_log = temporary / "xctrace.log"
            log_handle = xctrace_log.open("w")
            tracer = subprocess.Popen(
                [
                    "xcrun",
                    "xctrace",
                    "record",
                    "--template",
                    "Metal System Trace",
                    "--time-limit",
                    f"{TRACE_TIME_LIMIT_SECONDS}s",
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
                text = xctrace_log.read_text() if xctrace_log.exists() else ""
                if "Ctrl-C to stop the recording" in text:
                    break
                if tracer.poll() is not None:
                    raise RuntimeError(f"xctrace exited early: {text}")
                time.sleep(0.2)
            else:
                raise TimeoutError("xctrace did not report recording readiness")
            go.touch()
            _wait_for(decode_done, 120.0, child)
            # In attach mode xctrace does not reliably finalize on a programmatic
            # SIGINT.  Let the bounded 45-second recording end naturally while
            # the target waits without issuing additional Metal work.
            if tracer.wait(timeout=300.0) != 0:
                raise RuntimeError(f"xctrace failed: {xctrace_log.read_text()}")
            log_handle.close()
            trace_done.touch()
            if child.wait(timeout=120.0) != 0:
                raise RuntimeError("telemetry child failed")
            if not child_result.exists() or not trace.exists():
                raise RuntimeError("telemetry child or xctrace produced no result")
            size = path_bytes(trace)
            if size > MAX_TRACE_BYTES:
                raise RuntimeError("non-replayable trace exceeded 4 GiB")
            child_value = json.loads(child_result.read_text())
            telemetry = _export_telemetry(trace, pid)
            identity = _trace_identity(trace)
            identity.update(path=str(trace), stored_in_repository=False)
            artifact = _merge(
                args.output,
                args.arm,
                {
                    "pid": pid,
                    "configuration": {
                        "compiled_ffn": ARMS[args.arm],
                        "resident_fp32_router": False,
                        "residual_moe_arm": "B1+shared-fused",
                    },
                    "child": child_value,
                    "telemetry": telemetry,
                    "trace": identity,
                    "trace_time_limit_seconds": TRACE_TIME_LIMIT_SECONDS,
                },
            )
            print(json.dumps({"arm": args.arm, "complete": artifact["complete"], "trace": identity}, indent=2))
            return 0
        except Exception:
            if tracer is not None:
                terminate_process_group(tracer)
            terminate_process_group(child)
            remove_partial_trace(trace)
            raise


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
    parser.add_argument("--child-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--decode-done", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--trace-done", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    return _child(args) if args.child else _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
