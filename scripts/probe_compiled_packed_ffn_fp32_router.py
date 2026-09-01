#!/usr/bin/env python3
"""Probe decode-only compiled packed FFN shells and resident FP32 routers."""

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
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import probe_long_context_first_decode_boundary as boundary
import probe_packed_decode_runtime as packed_probe
import probe_residual_packed_decode_moe_fusion as residual

from glm53_flash_mlx.abi import MLX_VLM_REVISION, PACKED_DECODE_KERNEL_ABI
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint
from glm53_flash_mlx.packed import PackedFP8MoE

ARMS = {
    "A": {"compile_ffn": False, "resident_fp32_router": False},
    "B": {"compile_ffn": True, "resident_fp32_router": False},
    "C": {"compile_ffn": False, "resident_fp32_router": True},
    "D": {"compile_ffn": True, "resident_fp32_router": True},
}
REPRESENTATIVE_LAYERS = (3, 5)
SPARSE_LAYERS = tuple(range(3, 45))
FRONTIER_CONTEXT = 2049
QUALIFICATION_CONTEXT = 262144
TARGET_TPS = 15.0
TARGET_MS = 1000.0 / TARGET_TPS
DECODE_STEPS = 4096
MATERIALIZATION_INTERVAL = 256
EVIDENCE_STEPS = set(range(1, 17)) | set(range(256, DECODE_STEPS + 1, 256))


def _atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _memory() -> dict[str, int]:
    mx.synchronize()
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }


def _release(*values) -> None:
    for value in values:
        if isinstance(value, list):
            value.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(q * len(ordered)) - 1)
    return float(ordered[rank])


class _CaptureHC:
    def __init__(self, inner, sink: dict[int, mx.array], layer_id: int):
        self.inner = inner
        self.sink = sink
        self.layer_id = layer_id

    def __call__(self, x):
        self.sink[self.layer_id] = x
        return self.inner(x)


def _capture_ffn_inputs(model) -> dict[int, mx.array]:
    layers = model.language_model.model.layers
    captured = {}
    originals = {layer_id: layers[layer_id].ffn_hc for layer_id in REPRESENTATIVE_LAYERS}
    cache = model.make_cache()
    try:
        for layer_id, original in originals.items():
            layers[layer_id].ffn_hc = _CaptureHC(original, captured, layer_id)
        output = model(mx.array([[101]], dtype=mx.uint32), cache=cache)
        mx.eval(output.logits, *captured.values())
        mx.synchronize()
    finally:
        for layer_id, original in originals.items():
            layers[layer_id].ffn_hc = original
    result = {layer_id: mx.array(value) for layer_id, value in captured.items()}
    mx.eval(*result.values())
    _release(cache)
    return result


def _router_outputs(gate, x):
    logits = x.astype(mx.float32) @ gate.weight.astype(mx.float32).T
    indices, scores = gate(x)
    mx.eval(logits, indices, scores)
    return logits, indices, scores


def _router_parity_before_after(model, inputs, resident: bool) -> tuple[dict, dict]:
    layers = model.language_model.model.layers
    before = {}
    before_dtype = {}
    before_bytes = 0
    for layer_id in SPARSE_LAYERS:
        weight = layers[layer_id].mlp.gate.weight
        before_dtype[str(layer_id)] = str(weight.dtype)
        before_bytes += int(weight.nbytes)
    for layer_id in REPRESENTATIVE_LAYERS:
        gate = layers[layer_id].mlp.gate
        before[str(layer_id)] = _router_outputs(gate, inputs[layer_id])

    conversion_started = time.perf_counter()
    active_before = int(mx.get_active_memory())
    if resident:
        for layer_id in SPARSE_LAYERS:
            gate = layers[layer_id].mlp.gate
            converted = mx.contiguous(gate.weight.astype(mx.float32))
            mx.eval(converted)
            gate.weight = converted
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
    conversion_seconds = time.perf_counter() - conversion_started
    active_after = int(mx.get_active_memory())

    after_dtype = {}
    after_bytes = 0
    parity = {}
    for layer_id in SPARSE_LAYERS:
        weight = layers[layer_id].mlp.gate.weight
        after_dtype[str(layer_id)] = str(weight.dtype)
        after_bytes += int(weight.nbytes)
    for layer_id in REPRESENTATIVE_LAYERS:
        gate = layers[layer_id].mlp.gate
        after = _router_outputs(gate, inputs[layer_id])
        reference = before[str(layer_id)]
        parity[str(layer_id)] = {
            "raw_logits": residual.d99f._metrics(reference[0], after[0]),
            "selected_indices": residual.d99f._metrics(reference[1], after[1]),
            "routing_scores": residual.d99f._metrics(reference[2], after[2]),
        }
    storage = {
        "resident_fp32_requested": resident,
        "conversion_seconds": conversion_seconds,
        "active_before_bytes": active_before,
        "active_after_bytes": active_after,
        "active_delta_bytes": active_after - active_before,
        "weight_bytes_before": before_bytes,
        "weight_bytes_after": after_bytes,
        "weight_bytes_delta": after_bytes - before_bytes,
        "all_before_dtypes": sorted(set(before_dtype.values())),
        "all_after_dtypes": sorted(set(after_dtype.values())),
    }
    return parity, storage


def _ffn_parts(layer, x):
    residual_x = x
    xc, post, comb = layer.ffn_hc(x)
    normalized = layer.post_attention_layernorm(xc)
    moe_output = layer.mlp(normalized)
    from mlx_vlm.models.glm5_next.language import hc_expand

    hc_output = hc_expand(moe_output, residual_x, post, comb)
    return moe_output, hc_output


def _compiled_stage_parity(model, inputs) -> dict:
    layers = model.language_model.model.layers
    rows = {}
    for layer_id in REPRESENTATIVE_LAYERS:
        layer = layers[layer_id]
        x = inputs[layer_id]
        eager_moe, eager_hc = _ffn_parts(layer, x)
        compiled = mx.compile(lambda value, layer=layer: _ffn_parts(layer, value))
        compiled_moe, compiled_hc = compiled(x)
        mx.eval(eager_moe, eager_hc, compiled_moe, compiled_hc)
        rows[str(layer_id)] = {
            "moe_output": residual.d99f._metrics(eager_moe, compiled_moe),
            "hc_output": residual.d99f._metrics(eager_hc, compiled_hc),
        }
        del compiled
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    return rows


def _configure_compile(model, enabled: bool) -> None:
    for layer_id, layer in enumerate(model.language_model.model.layers):
        layer.compile_ffn = bool(enabled and layer_id in SPARSE_LAYERS)
        layer._ffn_c = None


def _warm_compiled_ffn(model, enabled: bool) -> dict:
    if not enabled:
        return {
            "enabled": False,
            "compiled_layers": 0,
            "warmup_seconds": 0.0,
            "active_before_bytes": int(mx.get_active_memory()),
            "active_after_bytes": int(mx.get_active_memory()),
            "peak_bytes": int(mx.get_peak_memory()),
        }
    mx.reset_peak_memory()
    active_before = int(mx.get_active_memory())
    cache = model.make_cache()
    started = time.perf_counter()
    output = model(mx.array([[101]], dtype=mx.uint32), cache=cache)
    mx.eval(output.logits)
    mx.synchronize()
    elapsed = time.perf_counter() - started
    active_after = int(mx.get_active_memory())
    peak = int(mx.get_peak_memory())
    compiled_layers = sum(
        layer._ffn_c is not None
        for layer in model.language_model.model.layers
        if isinstance(layer.mlp, PackedFP8MoE)
    )
    _release(cache)
    return {
        "enabled": True,
        "compiled_layers": compiled_layers,
        "warmup_seconds": elapsed,
        "active_before_bytes": active_before,
        "active_after_bytes": active_after,
        "active_delta_bytes": active_after - active_before,
        "peak_bytes": peak,
    }


def _materialize(cache) -> float:
    started = time.perf_counter()
    mx.eval([entry.state for entry in cache])
    mx.clear_cache()
    mx.synchronize()
    return (time.perf_counter() - started) * 1000.0


def _qualification_4096(model) -> tuple[object, dict]:
    packed_probe._set_cache_backend(model, "compact-nope-dsa", 4352)
    cache = model.make_cache()
    started = time.perf_counter()
    output = model(mx.array([[1]], dtype=mx.uint32), cache=cache)
    mx.eval(output.logits)
    mx.synchronize()
    latencies = [(time.perf_counter() - started) * 1000.0]
    baseline_memory = _memory()
    generated = []
    evidence = {}
    materialization_ms = []
    nan_count = 0
    for step in range(1, DECODE_STEPS + 1):
        logits = output.logits[0, -1]
        predicted_array = mx.argmax(logits)
        nan_array = mx.sum(mx.isnan(logits))
        mx.eval(predicted_array, nan_array)
        predicted = int(predicted_array.item())
        nan_count += int(nan_array.item())
        generated.append(predicted)
        if step in EVIDENCE_STEPS:
            evidence[str(step)] = packed_probe._hash(logits)
        if step % MATERIALIZATION_INTERVAL == 0:
            materialization_ms.append(_materialize(cache))
        if step < DECODE_STEPS:
            started = time.perf_counter()
            output = model(mx.array([[predicted]], dtype=mx.uint32), cache=cache)
            mx.eval(output.logits)
            mx.synchronize()
            latencies.append((time.perf_counter() - started) * 1000.0)
    windows = {
        "1_256": latencies[0:256],
        "1793_2048": latencies[1792:2048],
        "3841_4096": latencies[3840:4096],
    }
    steady = latencies[255:]
    final_memory = _memory()
    return cache, {
        "steps": DECODE_STEPS,
        "token_sha256": packed_probe._token_digest(generated),
        "evidence_logits_hashes": evidence,
        "cache_state_hash": boundary._full_cache_hash(cache),
        "nan_count": nan_count,
        "metal_error": None,
        "materialization_count": len(materialization_ms),
        "materialization_ms": materialization_ms,
        "decode_median_ms": statistics.median(steady),
        "decode_tokens_per_second": 1000.0 / statistics.median(steady),
        "latency_windows": {
            name: {
                "count": len(values),
                "median_ms": statistics.median(values),
                "p95_ms": _percentile(values, 0.95),
            }
            for name, values in windows.items()
        },
        "late_early_retention": (
            statistics.median(windows["1_256"])
            / statistics.median(windows["3841_4096"])
        ),
        "active_memory_baseline": baseline_memory,
        "active_memory_final": final_memory,
        "active_memory_drift_bytes": (
            final_memory["active_bytes"] - baseline_memory["active_bytes"]
        ),
    }


def _all_metrics_exact(value) -> bool:
    if isinstance(value, dict) and "byte_identical" in value:
        return bool(value["byte_identical"])
    if isinstance(value, dict):
        return all(_all_metrics_exact(item) for item in value.values())
    return True


def _run_child(args) -> int:
    arm = ARMS[args.arm]
    report = inspect_checkpoint(args.model, require_server_ready=True)
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    mx.reset_peak_memory()
    started = time.perf_counter()
    _progress("child_load", arm=args.arm, mode=args.mode, pid=os.getpid())
    model, _ = load(args.model, experimental_packed_decode_moe=True)
    load_seconds = time.perf_counter() - started
    warm_started = time.perf_counter()
    warm_residency(model)
    warm_seconds = time.perf_counter() - warm_started
    steady_before = _memory()

    try:
        with residual._runtime(residual.Arm("B1", True)):
            ffn_inputs = _capture_ffn_inputs(model)
            moe_inputs = residual.d99f._capture_layer_inputs(model)
            router_parity, router_storage = _router_parity_before_after(
                model, moe_inputs, arm["resident_fp32_router"]
            )
            compile_stage_parity = _compiled_stage_parity(model, ffn_inputs)
            _configure_compile(model, arm["compile_ffn"])
            compile_warmup = _warm_compiled_ffn(model, arm["compile_ffn"])
            steady_after_setup = _memory()
            _progress("child_screen_2k", arm=args.arm, mode=args.mode)
            screen, screen_hashes = packed_probe._frontier_arm(
                model, context=FRONTIER_CONTEXT, cache_backend="direct"
            )
            qualification = None
            frontier_256k = None
            if args.mode == "qualify":
                _progress("child_qualify_4096", arm=args.arm)
                cache, qualification = _qualification_4096(model)
                _progress("child_qualify_256k", arm=args.arm)
                frontier_256k, frontier_hashes = packed_probe._frontier_arm(
                    model,
                    context=QUALIFICATION_CONTEXT,
                    cache_backend="compact-nope-dsa",
                )
                frontier_256k["logits_hashes"] = frontier_hashes
                _release(cache)
    except Exception as exc:
        failure = {
            "schema": "glm53-compiled-packed-ffn-fp32-router-child-v1",
            "complete": False,
            "arm": args.arm,
            "mode": args.mode,
            "pid": os.getpid(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        _atomic_write(args.output, failure)
        raise

    correctness = {
        "router_raw_logits_indices_scores_exact": _all_metrics_exact(router_parity),
        "layer_3_5_compiled_moe_and_hc_output_exact": _all_metrics_exact(
            compile_stage_parity
        ),
        "screen_2k_nan_zero": screen["nan_count"] == 0,
    }
    child = {
        "schema": "glm53-compiled-packed-ffn-fp32-router-child-v1",
        "complete": True,
        "arm": args.arm,
        "mode": args.mode,
        "pid": os.getpid(),
        "configuration": arm,
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "packed_decode_kernel_abi": PACKED_DECODE_KERNEL_ABI,
        "residual_moe_baseline_commit": "aad32b1",
        "load_seconds": load_seconds,
        "warm_residency_seconds": warm_seconds,
        "steady_before_setup": steady_before,
        "router_parity": router_parity,
        "router_storage": router_storage,
        "compile_stage_parity": compile_stage_parity,
        "compile_warmup": compile_warmup,
        "steady_after_setup": steady_after_setup,
        "screen_2k": screen,
        "screen_2k_logits_hashes": screen_hashes,
        "screen_2k_tokens_per_second": 1000.0 / screen["median_ms"],
        "qualification_4096": qualification,
        "synthetic_256k": frontier_256k,
        "correctness": correctness,
        "complete_memory": _memory(),
    }
    _atomic_write(args.output, child)
    return 0 if all(correctness.values()) else 1


def _child_command(args, arm: str, mode: str, output: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        str(args.model),
        "--arm",
        arm,
        "--mode",
        mode,
        "--output",
        str(output),
        "--wired-limit-gb",
        str(args.wired_limit_gb),
        "--cache-limit-gb",
        str(args.cache_limit_gb),
    ]


def _run_orchestrator(args) -> int:
    report = inspect_checkpoint(args.model, require_server_ready=True)
    artifact = {
        "schema": "glm53-compiled-packed-ffn-fp32-router-v1",
        "date": date.today().isoformat(),
        "complete": False,
        "probe_only": True,
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "packed_decode_kernel_abi": PACKED_DECODE_KERNEL_ABI,
        "residual_moe_baseline_commit": "aad32b1",
        "process_isolation": True,
        "screens": {},
        "qualifications": {},
    }
    _atomic_write(args.output, artifact)
    with tempfile.TemporaryDirectory(prefix="glm53-compiled-ffn-") as temporary:
        temporary = Path(temporary)
        for arm in ARMS:
            child_output = temporary / f"screen-{arm}.json"
            _progress("orchestrator_screen", arm=arm)
            completed = subprocess.run(
                _child_command(args, arm, "screen", child_output), check=False
            )
            if child_output.exists():
                artifact["screens"][arm] = json.loads(child_output.read_text())
            else:
                artifact["screens"][arm] = {
                    "complete": False,
                    "returncode": completed.returncode,
                    "error": "child produced no artifact",
                }
            artifact["screens"][arm]["returncode"] = completed.returncode
            _atomic_write(args.output, artifact)

        complete_screens = all(
            row.get("complete") and row.get("returncode") == 0
            for row in artifact["screens"].values()
        )
        if not complete_screens:
            artifact.update(
                complete=True,
                runtime_candidate_accepted=False,
                decision="reject: one or more isolated screen processes failed",
            )
            _atomic_write(args.output, artifact)
            return 1

        baseline_hashes = artifact["screens"]["A"]["screen_2k_logits_hashes"]
        for row in artifact["screens"].values():
            row["screen_2k_hashes_exact_vs_A"] = (
                row["screen_2k_logits_hashes"] == baseline_hashes
            )
        candidates = [
            arm
            for arm, row in artifact["screens"].items()
            if row["screen_2k_tokens_per_second"] >= TARGET_TPS
            and all(row["correctness"].values())
        ]
        artifact["qualification_candidates"] = candidates
        if candidates:
            qualification_arms = ["A"] + [arm for arm in candidates if arm != "A"]
            for arm in qualification_arms:
                child_output = temporary / f"qualify-{arm}.json"
                _progress("orchestrator_qualify", arm=arm)
                completed = subprocess.run(
                    _child_command(args, arm, "qualify", child_output), check=False
                )
                artifact["qualifications"][arm] = (
                    json.loads(child_output.read_text())
                    if child_output.exists()
                    else {
                        "complete": False,
                        "error": "qualification child produced no artifact",
                    }
                )
                artifact["qualifications"][arm]["returncode"] = completed.returncode
                _atomic_write(args.output, artifact)

    screen_tps = {
        arm: row["screen_2k_tokens_per_second"]
        for arm, row in artifact["screens"].items()
    }
    screen_speedup = {
        arm: value / screen_tps["A"] for arm, value in screen_tps.items()
    }
    retained_screen_candidates = [
        arm for arm in ("B", "C") if screen_tps[arm] >= 14.4
    ]
    screen_performance = {
        "B_at_least_14_4_tps": screen_tps["B"] >= 14.4,
        "C_at_least_14_4_tps": screen_tps["C"] >= 14.4,
        "D_at_least_15_tps": screen_tps["D"] >= TARGET_TPS,
        "D_median_at_most_66_667_ms": (
            artifact["screens"]["D"]["screen_2k"]["median_ms"] <= TARGET_MS
        ),
    }
    screen_correctness = {
        "all_child_correctness_gates": all(
            all(row["correctness"].values()) for row in artifact["screens"].values()
        ),
        "all_2k_full_vocab_hashes_exact": all(
            row["screen_2k_hashes_exact_vs_A"]
            for row in artifact["screens"].values()
        ),
        "four_distinct_processes": len(
            {row["pid"] for row in artifact["screens"].values()}
        )
        == 4,
    }
    qualification_correctness = {}
    qualification_performance = {}
    if artifact["qualifications"]:
        baseline = artifact["qualifications"]["A"]
        for arm, row in artifact["qualifications"].items():
            q = row["qualification_4096"]
            f = row["synthetic_256k"]
            qualification_correctness[arm] = {
                "generated_tokens_exact": q["token_sha256"]
                == baseline["qualification_4096"]["token_sha256"],
                "evidence_full_vocab_hashes_exact": q["evidence_logits_hashes"]
                == baseline["qualification_4096"]["evidence_logits_hashes"],
                "final_kda_dsa_state_exact": q["cache_state_hash"]
                == baseline["qualification_4096"]["cache_state_hash"],
                "materialization_count_16": q["materialization_count"] == 16,
                "synthetic_256k_logits_exact": f["logits_hashes"]
                == baseline["synthetic_256k"]["logits_hashes"],
                "nan_and_metal_error_zero": q["nan_count"] == 0
                and q["metal_error"] is None,
            }
            qualification_performance[arm] = {
                "at_least_15_tps": q["decode_tokens_per_second"] >= TARGET_TPS,
                "median_at_most_66_667_ms": q["decode_median_ms"] <= TARGET_MS,
                "late_early_retention_at_least_0_95": q["late_early_retention"]
                >= 0.95,
                "active_drift_at_most_64_mib": q["active_memory_drift_bytes"]
                <= 64 * 1024 * 1024,
                "peak_at_most_340_gb": q["active_memory_final"]["peak_bytes"]
                <= 340e9,
            }

    accepted_arms = [
        arm
        for arm in candidates
        if arm in qualification_correctness
        and all(qualification_correctness[arm].values())
        and all(qualification_performance[arm].values())
    ]
    artifact.update(
        complete=True,
        screen_2k_tokens_per_second=screen_tps,
        screen_2k_speedup_vs_A=screen_speedup,
        retained_screen_candidates=retained_screen_candidates,
        screen_performance=screen_performance,
        screen_correctness=screen_correctness,
        qualification_correctness=qualification_correctness,
        qualification_performance=qualification_performance,
        accepted_arms=accepted_arms,
        runtime_candidate_accepted=bool(accepted_arms),
        decision=(
            f"promote exact compiled/router arm(s) in separate commit: {accepted_arms}"
            if accepted_arms
            else "retain B as a profiling candidate and aad32b1-D as the exact "
            "nonproduction baseline; no isolated arm qualified at 15 tok/s"
        ),
        runtime_changes={
            "packed_runtime": False,
            "kernel_abi": False,
            "server": False,
            "apc": False,
            "admission": False,
        },
    )
    _atomic_write(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "screen_2k_tokens_per_second": screen_tps,
                "qualification_candidates": candidates,
                "accepted_arms": accepted_arms,
            },
            indent=2,
        )
    )
    return (
        0
        if all(screen_correctness.values())
        and (not candidates or bool(accepted_arms))
        else 1
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--mode", choices=("screen", "qualify"), default="screen")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "bench-results/m3ultra512-compiled-packed-ffn-fp32-router-20260901.json"
        ),
    )
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    args = parser.parse_args()
    if args.arm:
        return _run_child(args)
    return _run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
