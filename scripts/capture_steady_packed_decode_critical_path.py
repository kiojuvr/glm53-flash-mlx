#!/usr/bin/env python3
"""Capture the exact packed-decode critical path for Xcode Metal analysis.

Each invocation captures one arm in a fresh process.  The binary ``.gputrace``
bundle must live outside the repository; only its canonical identity and the
decode evidence are merged into the JSON artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from capture_budget import CaptureBudget, supervise_capture


ARMS = {
    "A": {"compile_ffn": False, "router_weight_dtype": "bfloat16"},
    "B": {"compile_ffn": True, "router_weight_dtype": "bfloat16"},
}
INITIAL_CONTEXT_TOKENS = 2049
WARMUP_TOKENS = 2
CAPTURE_TOKENS = 8
CACHE_BACKEND = "direct"
INITIAL_TOKEN = 3000
REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-steady-packed-decode-critical-path-20260901.json"
)
DEFAULT_NEGATIVE_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-full-model-gputrace-negative-evidence-20260901.json"
)


def _atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return float(ordered[max(0, math.ceil(q * len(ordered)) - 1)])


def _trace_identity(path: Path) -> dict:
    """Hash a gputrace file or bundle without depending on Xcode internals."""

    digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    if path.is_file():
        files = [(Path(path.name), path)]
        kind = "file"
    elif path.is_dir():
        files = [
            (item.relative_to(path), item)
            for item in sorted(path.rglob("*"))
            if item.is_file()
        ]
        kind = "bundle-directory"
    else:
        raise RuntimeError(f"Metal capture did not create a trace: {path}")
    for relative, item in files:
        encoded = relative.as_posix().encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        size = item.stat().st_size
        digest.update(size.to_bytes(8, "little"))
        with item.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        total_bytes += size
        file_count += 1
    if file_count == 0:
        raise RuntimeError(f"Metal capture trace is empty: {path}")
    return {
        "kind": kind,
        "sha256": digest.hexdigest(),
        "bytes": total_bytes,
        "file_count": file_count,
    }


def _command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or completed.stderr.strip() or None


def _git_value(*arguments: str) -> str | None:
    return _command_output(["git", *arguments])


def _base_artifact(checkpoint: dict) -> dict:
    reference_path = (
        REPOSITORY
        / "bench-results"
        / "m3ultra512-compiled-packed-ffn-fp32-router-20260901.json"
    )
    reference = None
    if reference_path.exists():
        value = json.loads(reference_path.read_text())
        reference = {
            "artifact": str(reference_path.relative_to(REPOSITORY)),
            "schema": value.get("schema"),
            "screen_2k_tokens_per_second": value.get(
                "screen_2k_tokens_per_second"
            ),
            "screen_2k_speedup_vs_A": value.get("screen_2k_speedup_vs_A"),
        }
    return {
        "schema": "glm53-steady-packed-decode-critical-path-v1",
        "date": date.today().isoformat(),
        "complete": False,
        "probe_only": True,
        "process_isolation": True,
        "checkpoint_revision": checkpoint["revision"],
        "checkpoint_fingerprint": checkpoint["fingerprint"],
        "execution_geometry": {
            "initial_context_tokens": INITIAL_CONTEXT_TOKENS,
            "warmup_tokens": WARMUP_TOKENS,
            "capture_start_context_tokens": INITIAL_CONTEXT_TOKENS
            + WARMUP_TOKENS,
            "capture_tokens": CAPTURE_TOKENS,
            "capture_end_context_tokens": INITIAL_CONTEXT_TOKENS
            + WARMUP_TOKENS
            + CAPTURE_TOKENS,
            "cache_backend": CACHE_BACKEND,
            "batch_size": 1,
            "sequence_length": 1,
            "greedy_argmax_readback_in_capture": True,
        },
        "arms": {},
        "reference_screen": reference,
        "capture_overhead_notice": (
            "capture-process wall latency includes GPUTools resource download and "
            "is not a decode performance measurement; fill manual_trace_summary "
            "from Xcode GPU event durations"
        ),
        "manual_trace_summary": {
            arm: {
                "analysis_status": "pending_xcode_dependencies_view",
                "token_wall_ms": None,
                "gpu_busy_ms": None,
                "gpu_idle_ms": None,
                "routed_moe_ms": None,
                "shared_expert_ms": None,
                "ffn_shell_ms": None,
                "kda_ms": None,
                "dsa_mla_ms": None,
                "dense_layers_ms": None,
                "router_ms": None,
                "lm_head_ms": None,
                "sampling_readback_ms": None,
            }
            for arm in ARMS
        },
        "runtime_changes": {
            "admission": False,
            "apc": False,
            "cache_abi": False,
            "kernel_abi": False,
            "packed_runtime": False,
            "server": False,
        },
    }


def _merge_result(output: Path, checkpoint: dict, arm: str, result: dict) -> dict:
    artifact = _base_artifact(checkpoint)
    if output.exists():
        existing = json.loads(output.read_text())
        if existing.get("schema") != artifact["schema"]:
            raise RuntimeError(f"refusing to merge incompatible artifact: {output}")
        artifact = existing
    artifact["capture_overhead_notice"] = (
        "capture-process wall latency includes GPUTools resource download and "
        "is not a decode performance measurement; fill manual_trace_summary "
        "from Xcode GPU event durations"
    )
    artifact["arms"][arm] = result
    complete = set(artifact["arms"]) == set(ARMS) and all(
        row.get("complete") for row in artifact["arms"].values()
    )
    artifact["complete"] = complete
    if complete:
        a = artifact["arms"]["A"]
        b = artifact["arms"]["B"]
        distinct = a["pid"] != b["pid"]
        exact = {
            "distinct_processes": distinct,
            "generated_tokens_exact": a["generated_tokens"]
            == b["generated_tokens"],
            "full_vocab_logits_hashes_exact": a["full_vocab_logits_hashes"]
            == b["full_vocab_logits_hashes"],
            "post_cache_state_exact": a["post_cache_state_hash"]
            == b["post_cache_state_hash"],
            "physical_capacity_unchanged": bool(
                a["physical_capacity_unchanged"]
                and b["physical_capacity_unchanged"]
            ),
            "nan_and_metal_error_zero": bool(
                a["nan_count"] == 0
                and b["nan_count"] == 0
                and a["metal_error"] is None
                and b["metal_error"] is None
            ),
            "trace_identity_complete": bool(
                a["trace"]["sha256"] and b["trace"]["sha256"]
            ),
        }
        artifact["capture_correctness"] = exact
        artifact["capture_correctness_passed"] = all(exact.values())
        artifact.pop("compiled_speedup", None)
        artifact["decision"] = (
            "captures are correctness-complete; inspect both traces in the "
            "Xcode Dependencies view before selecting the next optimization"
        )
    _atomic_write(output, artifact)
    return artifact


def _validate_trace_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.suffix != ".gputrace":
        raise ValueError("--trace must end in .gputrace")
    if path.is_relative_to(REPOSITORY):
        raise ValueError(".gputrace must be written outside the repository")
    if path.exists():
        raise FileExistsError(f"capture path already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _memory(mx) -> dict[str, int]:
    mx.synchronize()
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }


def _run_capture(args) -> dict:
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        raise RuntimeError("set MTL_CAPTURE_ENABLED=1 before importing MLX")
    trace_path = _validate_trace_path(args.trace)

    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError("MLX/Metal is unavailable")
    sys.path.insert(0, str(Path(__file__).parent))
    import probe_compiled_packed_ffn_fp32_router as compiled_probe
    import probe_long_context_first_decode_boundary as boundary
    import probe_packed_decode_runtime as packed_probe
    import probe_residual_packed_decode_moe_fusion as residual

    from glm53_flash_mlx.abi import MLX_VLM_REVISION, PACKED_DECODE_KERNEL_ABI
    from glm53_flash_mlx.loader import load, warm_residency
    from glm53_flash_mlx.manifest import inspect_checkpoint

    arm = ARMS[args.arm]
    report = inspect_checkpoint(args.model, require_server_ready=True)
    checkpoint = {
        "revision": report.official_revision,
        "fingerprint": report.fingerprint,
    }
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    mx.reset_peak_memory()
    print(json.dumps({"phase": "load", "arm": args.arm, "pid": os.getpid()}), flush=True)
    load_started = time.perf_counter()
    model, _ = load(args.model, experimental_packed_decode_moe=True)
    load_seconds = time.perf_counter() - load_started
    residency_started = time.perf_counter()
    warm_residency(model)
    residency_seconds = time.perf_counter() - residency_started

    with residual._runtime(residual.Arm("B1", True)):
        compiled_probe._configure_compile(model, arm["compile_ffn"])
        compile_warmup = compiled_probe._warm_compiled_ffn(
            model, arm["compile_ffn"]
        )
        print(
            json.dumps(
                {
                    "phase": "build_synthetic_cache",
                    "arm": args.arm,
                    "context": INITIAL_CONTEXT_TOKENS,
                }
            ),
            flush=True,
        )
        cache = boundary._synthetic_cache(
            model, INITIAL_CONTEXT_TOKENS, CACHE_BACKEND
        )
        physical_before_warmup = boundary._physical_capacity(cache, CACHE_BACKEND)
        token = mx.array([[INITIAL_TOKEN]], dtype=mx.uint32)
        warmup_tokens = []
        for _ in range(WARMUP_TOKENS):
            output_value = model(token, cache=cache)
            predicted_value = mx.argmax(output_value.logits[0, -1])
            mx.eval(output_value.logits, predicted_value)
            mx.synchronize()
            predicted_token = int(predicted_value.item())
            warmup_tokens.append(predicted_token)
            token = mx.array([[predicted_token]], dtype=mx.uint32)
        physical_capture_start = boundary._physical_capacity(cache, CACHE_BACKEND)
        memory_capture_start = _memory(mx)
        mx.synchronize()

        print(
            json.dumps(
                {
                    "phase": "capture",
                    "arm": args.arm,
                    "trace": str(trace_path),
                    "tokens": CAPTURE_TOKENS,
                }
            ),
            flush=True,
        )
        capture_started = False
        captured_logits = []
        generated_tokens = []
        latencies_ms = []
        metal_error = None
        try:
            mx.metal.start_capture(str(trace_path))
            capture_started = True
            for _ in range(CAPTURE_TOKENS):
                started = time.perf_counter()
                output_value = model(token, cache=cache)
                logits_value = output_value.logits[0, -1]
                predicted_value = mx.argmax(logits_value)
                mx.eval(output_value.logits, predicted_value)
                mx.synchronize()
                predicted_token = int(predicted_value.item())
                latencies_ms.append((time.perf_counter() - started) * 1000.0)
                captured_logits.append(logits_value)
                generated_tokens.append(predicted_token)
                token = mx.array([[predicted_token]], dtype=mx.uint32)
        except Exception as exc:
            metal_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if capture_started:
                mx.metal.stop_capture()

        # Evidence collection is intentionally outside the captured interval.
        full_vocab_hashes = [packed_probe._hash(value) for value in captured_logits]
        nan_count = sum(
            int(packed_probe._np(mx.sum(mx.isnan(value))).reshape(-1)[0])
            for value in captured_logits
        )
        post_cache_state_hash = boundary._full_cache_hash(cache)
        physical_after = boundary._physical_capacity(cache, CACHE_BACKEND)
        memory_after = _memory(mx)

    trace = _trace_identity(trace_path)
    trace["path"] = str(trace_path)
    trace["stored_in_repository"] = False
    result = {
        "schema": "glm53-steady-packed-decode-critical-path-arm-v1",
        "complete": True,
        "arm": args.arm,
        "configuration": arm,
        "pid": os.getpid(),
        "host": platform.node(),
        "git_head_at_capture": _git_value("rev-parse", "HEAD"),
        "checkpoint_revision": checkpoint["revision"],
        "checkpoint_fingerprint": checkpoint["fingerprint"],
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "packed_decode_kernel_abi": PACKED_DECODE_KERNEL_ABI,
        "residual_moe_baseline_commit": "aad32b1",
        "compiled_probe_commit": "0450bf2",
        "xcode_version": _command_output(["xcodebuild", "-version"]),
        "macos_version": _command_output(["sw_vers"]),
        "load_seconds": load_seconds,
        "warm_residency_seconds": residency_seconds,
        "compile_warmup": compile_warmup,
        "initial_context_tokens": INITIAL_CONTEXT_TOKENS,
        "capture_start_context_tokens": INITIAL_CONTEXT_TOKENS + WARMUP_TOKENS,
        "capture_end_context_tokens": (
            INITIAL_CONTEXT_TOKENS + WARMUP_TOKENS + CAPTURE_TOKENS
        ),
        "cache_backend": CACHE_BACKEND,
        "warmup_generated_tokens": warmup_tokens,
        "generated_tokens": generated_tokens,
        "full_vocab_logits_hashes": full_vocab_hashes,
        "post_cache_state_hash": post_cache_state_hash,
        "capture_process_wall_samples_ms": latencies_ms,
        "capture_process_wall_median_ms": statistics.median(latencies_ms),
        "capture_process_wall_p95_ms": _percentile(latencies_ms, 0.95),
        "xcode_gpu_event_measurement_required": True,
        "nan_count": nan_count,
        "metal_error": metal_error,
        "physical_capacity_before_warmup": physical_before_warmup,
        "physical_capacity_at_capture_start": physical_capture_start,
        "physical_capacity_after_capture": physical_after,
        "physical_capacity_unchanged": (
            physical_before_warmup == physical_capture_start == physical_after
        ),
        "memory_at_capture_start": memory_capture_start,
        "memory_after_evidence": memory_after,
        "trace": trace,
        "capture_exclusions": [
            "mx.clear_cache",
            "256-token materialization",
            "hash calculation",
            "memory probes",
            "logging",
        ],
    }
    return checkpoint, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    parser.add_argument("--capture-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--negative-output", type=Path, default=DEFAULT_NEGATIVE_OUTPUT
    )
    parser.add_argument("--max-elapsed-s", type=float, default=900.0)
    parser.add_argument("--max-trace-gib", type=float, default=32.0)
    parser.add_argument("--min-free-gib", type=float, default=64.0)
    args = parser.parse_args()
    if not args.capture_child:
        if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
            raise RuntimeError("set MTL_CAPTURE_ENABLED=1 before starting capture")
        trace_path = _validate_trace_path(args.trace)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            str(args.model),
            "--arm",
            args.arm,
            "--trace",
            str(trace_path),
            "--output",
            str(args.output),
            "--wired-limit-gb",
            str(args.wired_limit_gb),
            "--cache-limit-gb",
            str(args.cache_limit_gb),
            "--capture-child",
        ]
        return supervise_capture(
            command,
            trace_path=trace_path,
            evidence_path=args.negative_output,
            budget=CaptureBudget(
                max_elapsed_s=args.max_elapsed_s,
                max_trace_bytes=int(args.max_trace_gib * (1 << 30)),
                min_free_bytes=int(args.min_free_gib * (1 << 30)),
            ),
            metadata={
                "arm": args.arm,
                "capture_kind": "full-model-replayable-gputrace",
                "trace_path": str(trace_path),
            },
        )
    try:
        checkpoint, result = _run_capture(args)
    except Exception as exc:
        failure = {
            "schema": "glm53-steady-packed-decode-critical-path-arm-v1",
            "complete": False,
            "arm": args.arm,
            "pid": os.getpid(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1
    artifact = _merge_result(args.output, checkpoint, args.arm, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "arm": args.arm,
                "trace_sha256": result["trace"]["sha256"],
                "capture_observed_wall_median_ms": result[
                    "capture_process_wall_median_ms"
                ],
                "complete": artifact["complete"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
