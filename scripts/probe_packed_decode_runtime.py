#!/usr/bin/env python3
"""Validate the opt-in packed-decode MoE runtime on M3 Ultra."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np

import probe_long_context_first_decode_boundary as boundary
from glm53_flash_mlx.abi import (
    MLX_VLM_REVISION,
    PACKED_DECODE_KERNEL_ABI,
    PACKED_EXPERT_BANK_ABI,
)
from glm53_flash_mlx.grouped_fp8 import SortedGroupedFP8MoE
from glm53_flash_mlx.loader import install_packed_decode_moe, load, warm_residency
from glm53_flash_mlx.manifest import EXPECTED_DSA, inspect_checkpoint
from glm53_flash_mlx.materialization import (
    MATERIALIZATION_INTERVAL_TOKENS,
    MATERIALIZATION_POLICY,
)
from glm53_flash_mlx.packed import PackedFP8MoE

PROMPTS = (1, 16, 128, 256)
DECODE_STEPS = 4096
EVIDENCE_STEPS = tuple(
    sorted(set(range(1, 17)) | set(range(256, DECODE_STEPS + 1, 256)))
)
FRONTIER_CONTEXTS = (2049, 262144)
FRONTIER_WARMUPS = 2
FRONTIER_SAMPLES = 16
PREFILL_WARMUPS = 2
PREFILL_SAMPLES = 5
SERVER_PORT = 18082
MAX_ACTIVE_DRIFT = 64 * 1024 * 1024


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


def _np(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.ascontiguousarray(np.asarray(value.astype(mx.float32)), dtype=np.float32)


def _hash(value: mx.array) -> str:
    return hashlib.sha256(_np(value).tobytes()).hexdigest()


def _token_digest(tokens) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(int(token).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


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


def _set_cache_backend(model, backend: str, capacity: int = 4352) -> None:
    model._glm53_cache_backend = backend
    model.language_model._glm53_cache_backend = backend
    model.language_model._glm53_compact_cache_capacity_tokens = int(capacity)


def _materialize(cache) -> float:
    started = time.perf_counter()
    mx.eval([entry.state for entry in cache])
    mx.clear_cache()
    mx.synchronize()
    return (time.perf_counter() - started) * 1000.0


def _deterministic_tokens(tokens: int, vocab: int) -> mx.array:
    values = ((np.arange(tokens, dtype=np.uint64) * 7919) % (vocab - 1024) + 100)
    return mx.array(values.astype(np.uint32)[None])


def _prompt_logits(model, vocab: int) -> dict[str, dict]:
    _set_cache_backend(model, "direct")
    rows = {}
    for prompt in PROMPTS:
        cache = model.make_cache()
        output = model(_deterministic_tokens(prompt, vocab), cache=cache)
        logits = output.logits
        mx.eval(logits)
        mx.synchronize()
        rows[str(prompt)] = {
            "shape": list(logits.shape),
            "full_vocab_logits_hash": _hash(logits),
            "nan_count": int(np.isnan(_np(logits)).sum()),
        }
        _release(cache)
    return rows


def _prefill_benchmark(model, tokens: mx.array, *, forbid_grouped: bool) -> dict:
    _set_cache_backend(model, "direct")
    grouped_calls = 0
    original = SortedGroupedFP8MoE.grouped_from_routes

    def forbidden(*args, **kwargs):
        nonlocal grouped_calls
        grouped_calls += 1
        raise RuntimeError("packed-decode invoked grouped prefill")

    if forbid_grouped:
        SortedGroupedFP8MoE.grouped_from_routes = forbidden
    try:
        samples = []
        hashes = []
        for sample in range(PREFILL_WARMUPS + PREFILL_SAMPLES):
            cache = model.make_cache()
            started = time.perf_counter()
            output = model(tokens, cache=cache)
            logits = output.logits
            mx.eval(logits)
            mx.synchronize()
            elapsed = (time.perf_counter() - started) * 1000.0
            if sample >= PREFILL_WARMUPS:
                samples.append(elapsed)
                hashes.append(_hash(logits))
            _release(cache)
    finally:
        if forbid_grouped:
            SortedGroupedFP8MoE.grouped_from_routes = original
    return {
        "warmups": PREFILL_WARMUPS,
        "samples": PREFILL_SAMPLES,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "logits_hashes": hashes,
        "grouped_kernel_calls": grouped_calls,
    }


def _run_4096(model, teacher_tokens=None):
    _set_cache_backend(model, "compact-nope-dsa", 4352)
    cache = model.make_cache()
    output = model(mx.array([[1]], dtype=mx.uint32), cache=cache)
    mx.eval(output.logits)
    mx.synchronize()
    baseline = _memory()
    generated = []
    evidence = {}
    latencies = []
    materialization_ms = []
    mismatches = []
    nan_count = 0
    for step in range(1, DECODE_STEPS + 1):
        logits = output.logits[0, -1]
        predicted_array = mx.argmax(logits)
        nan_array = mx.sum(mx.isnan(logits))
        mx.eval(predicted_array, nan_array)
        predicted = int(predicted_array.item())
        nan_count += int(nan_array.item())
        generated.append(predicted)
        if teacher_tokens is not None and predicted != int(teacher_tokens[step - 1]):
            mismatches.append(step)
        if step in EVIDENCE_STEPS:
            evidence[str(step)] = _hash(logits)
        if step % MATERIALIZATION_INTERVAL_TOKENS == 0:
            materialization_ms.append(_materialize(cache))
        if step < DECODE_STEPS:
            token = predicted if teacher_tokens is None else int(teacher_tokens[step - 1])
            started = time.perf_counter()
            output = model(mx.array([[token]], dtype=mx.uint32), cache=cache)
            mx.eval(output.logits)
            mx.synchronize()
            latencies.append((time.perf_counter() - started) * 1000.0)
    final_memory = _memory()
    result = {
        "steps": DECODE_STEPS,
        "token_sha256": _token_digest(generated),
        "generated_tokens": generated,
        "evidence_logits_hashes": evidence,
        "token_mismatch_steps": mismatches,
        "all_tokens_match_teacher": not mismatches,
        "nan_count": nan_count,
        "metal_error": None,
        "materialization_count": len(materialization_ms),
        "materialization_ms": materialization_ms,
        "decode_median_ms": statistics.median(latencies[255:]),
        "decode_tokens_per_second": 1000.0 / statistics.median(latencies[255:]),
        "active_memory_baseline": baseline,
        "active_memory_final": final_memory,
        "active_memory_drift_bytes": final_memory["active_bytes"] - baseline["active_bytes"],
    }
    return cache, result


def _frontier_arm(model, *, context: int, cache_backend: str) -> tuple[dict, list[str]]:
    cache = boundary._synthetic_cache(model, context, cache_backend)
    hashes = []
    samples = []
    nan_count = 0
    for step in range(FRONTIER_WARMUPS + FRONTIER_SAMPLES):
        token = 3000 + step
        started = time.perf_counter()
        output = model(mx.array([[token]], dtype=mx.uint32), cache=cache)
        logits = output.logits[0, -1]
        nan = mx.sum(mx.isnan(logits))
        mx.eval(logits, nan)
        mx.synchronize()
        elapsed = (time.perf_counter() - started) * 1000.0
        nan_count += int(nan.item())
        hashes.append(_hash(logits))
        if step >= FRONTIER_WARMUPS:
            samples.append(elapsed)
    result = {
        "context_tokens": context,
        "cache_backend": cache_backend,
        "warmups": FRONTIER_WARMUPS,
        "samples": FRONTIER_SAMPLES,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "tokens_per_second": 1000.0 / statistics.median(samples),
        "logits_hashes": hashes,
        "nan_count": nan_count,
        "active_memory_bytes": int(mx.get_active_memory()),
        "peak_memory_bytes": int(mx.get_peak_memory()),
    }
    _release(cache)
    return result, hashes


def _frontier(model) -> tuple[dict, dict]:
    rows = {}
    hashes = {}
    for context in FRONTIER_CONTEXTS:
        for cache_backend in ("direct", "compact-nope-dsa"):
            _progress(
                "frontier",
                moe_backend=getattr(model, "_glm53_moe_backend", "direct"),
                cache_backend=cache_backend,
                context=context,
            )
            result, arm_hashes = _frontier_arm(
                model, context=context, cache_backend=cache_backend
            )
            key = f"{cache_backend}:{context}"
            rows[key] = result
            hashes[key] = arm_hashes
    return rows, hashes


def _ram_apc(model, vocab: int) -> dict:
    _set_cache_backend(model, "compact-nope-dsa", 4352)
    source = model.make_cache()
    model(_deterministic_tokens(256, vocab), cache=source)
    snapshot = boundary._clone_cache(source, 272)
    resident = boundary._clone_cache(snapshot, 272)
    restored = boundary._clone_cache(snapshot, 272)
    snapshot_before = boundary._full_cache_hash(snapshot)
    hashes = []
    for step in range(16):
        token = 5000 + step
        left = model(mx.array([[token]], dtype=mx.uint32), cache=resident)
        right = model(mx.array([[token]], dtype=mx.uint32), cache=restored)
        left_hash = _hash(left.logits[0, -1])
        right_hash = _hash(right.logits[0, -1])
        hashes.append(left_hash == right_hash)
    result = {
        "steps": 16,
        "all_logits_hashes_match": all(hashes),
        "post_state_exact": boundary._cache_exact(resident, restored),
        "snapshot_immutable": snapshot_before == boundary._full_cache_hash(snapshot),
    }
    _release(source, snapshot, resident, restored)
    return result


def _storage_invariants(model, report) -> dict:
    modules = [
        layer.mlp
        for layer in model.language_model.model.layers
        if isinstance(layer.mlp, PackedFP8MoE)
    ]
    return {
        "packed_layer_count": len(modules),
        "sorted_grouped_layer_count": sum(
            isinstance(module, SortedGroupedFP8MoE) for module in modules
        ),
        "all_old_expert_modules_detached": report[
            "all_old_expert_modules_detached"
        ],
        "all_bank_weight_uint8": all(
            module.bank.gate_up_weight.dtype == mx.uint8
            and module.bank.down_weight.dtype == mx.uint8
            for module in modules
        ),
        "all_bank_scale_float32": all(
            module.bank.gate_up_scale_inv.dtype == mx.float32
            and module.bank.down_scale_inv.dtype == mx.float32
            for module in modules
        ),
        "bf16_weight_expansion_present": any(
            value.dtype == mx.bfloat16
            for module in modules
            for name, value in module.bank.parameters().items()
            if "weight" in name
        ),
        "bank_bytes": sum(module.bank.nbytes for module in modules),
    }


def _server_smoke(model_path: Path, *, timeout_seconds: float = 240.0) -> dict:
    executable = Path(sys.executable).with_name("glm53-serve")
    command = [
        str(executable),
        "--model",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(SERVER_PORT),
        "--experimental-packed-decode-moe",
        "--experimental-compact-nope-dsa-cache",
    ]
    started = time.perf_counter()
    with tempfile.NamedTemporaryFile(mode="w+", prefix="glm53-packed-decode-") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        status = None
        body = None
        error = None
        ready_seconds = None
        try:
            while time.perf_counter() - started < timeout_seconds:
                if process.poll() is not None:
                    error = f"server exited with {process.returncode}"
                    break
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{SERVER_PORT}/health", timeout=2
                    ) as response:
                        status = int(response.status)
                        body = json.loads(response.read().decode())
                    if status == 200:
                        ready_seconds = time.perf_counter() - started
                        break
                except Exception:
                    time.sleep(1.0)
            else:
                error = "server readiness timeout"
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)
            log.flush()
            log.seek(0)
            lines = log.read().splitlines()
    return {
        "command": command,
        "ready_seconds": ready_seconds,
        "health_http_status": status,
        "health": body,
        "error": error,
        "log_tail": lines[-40:],
    }


def _atomic_write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _refresh_prefill_existing(args, report, vocab: int) -> int:
    artifact = json.loads(args.output.read_text())
    if artifact.get("schema") != "glm53-packed-decode-runtime-v1":
        raise ValueError("prefill refresh requires the packed-decode runtime artifact")
    model, _ = load(args.model)
    warm_residency(model)
    tokens = _deterministic_tokens(256, vocab)
    direct_prompts = _prompt_logits(model, vocab)
    direct_prefill = _prefill_benchmark(model, tokens, forbid_grouped=False)
    install_report = install_packed_decode_moe(model)
    packed_prompts = _prompt_logits(model, vocab)
    packed_prefill = _prefill_benchmark(model, tokens, forbid_grouped=True)
    prompt_parity = {
        prompt: direct_prompts[prompt]["full_vocab_logits_hash"]
        == packed_prompts[prompt]["full_vocab_logits_hash"]
        for prompt in direct_prompts
    }
    ratio = packed_prefill["median_ms"] / direct_prefill["median_ms"]
    artifact["direct"]["prompts"] = direct_prompts
    artifact["direct"]["prefill_256"] = direct_prefill
    artifact["packed_decode"]["prompts"] = packed_prompts
    artifact["packed_decode"]["prefill_256"] = packed_prefill
    artifact["packed_decode"]["install_report"] = install_report
    artifact["prompt_parity"] = prompt_parity
    artifact["comparisons"]["prefill_packed_over_direct_latency"] = ratio
    artifact["acceptance"][
        "prompt_1_16_128_256_full_vocab_logits_byte_identical"
    ] = all(prompt_parity.values())
    artifact["acceptance"][
        "grouped_kernel_calls_zero_for_256_prefill"
    ] = packed_prefill["grouped_kernel_calls"] == 0
    artifact["acceptance"]["prefill_regression_at_most_5_percent"] = ratio <= 1.05
    artifact["acceptance"]["accepted"] = all(
        value
        for key, value in artifact["acceptance"].items()
        if key != "accepted"
    )
    artifact["prefill_refresh"] = {
        "date": date.today().isoformat(),
        "reason": "packed full-bank Direct-order addressing removed per-expert slice copies",
        "checkpoint_fingerprint": report.fingerprint,
        "prompt_hashes_refreshed": True,
        "grouped_kernel_calls": packed_prefill["grouped_kernel_calls"],
    }
    _atomic_write(args.output, artifact)
    _release()
    del model
    print(
        json.dumps(
            {
                "output": str(args.output),
                "accepted": artifact["acceptance"]["accepted"],
                "prefill_packed_over_direct_latency": ratio,
                "refreshed_existing": True,
            },
            indent=2,
        )
    )
    return 0 if artifact["acceptance"]["accepted"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    parser.add_argument("--skip-server-smoke", action="store_true")
    parser.add_argument("--refresh-prefill-existing", action="store_true")
    args = parser.parse_args()

    report = inspect_checkpoint(args.model, require_server_ready=True)
    raw_config = json.loads((args.model / "config.json").read_text())
    vocab = int(raw_config["text_config"]["vocab_size"])
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    if args.refresh_prefill_existing:
        return _refresh_prefill_existing(args, report, vocab)
    mx.reset_peak_memory()

    load_started = time.perf_counter()
    model, _ = load(args.model)
    direct_load_seconds = time.perf_counter() - load_started
    warm_started = time.perf_counter()
    warm_residency(model)
    direct_warm_seconds = time.perf_counter() - warm_started
    direct_steady = _memory()

    _progress("direct_prompt_correctness")
    direct_prompts = _prompt_logits(model, vocab)
    prefill_tokens = _deterministic_tokens(256, vocab)
    direct_prefill = _prefill_benchmark(model, prefill_tokens, forbid_grouped=False)
    direct_frontier, direct_frontier_hashes = _frontier(model)
    _progress("direct_4096")
    direct_cache, direct_decode = _run_4096(model)
    teacher_tokens = direct_decode.pop("generated_tokens")

    mx.reset_peak_memory()
    install_started = time.perf_counter()
    install_report = install_packed_decode_moe(model)
    install_seconds = time.perf_counter() - install_started
    install_peak = int(mx.get_peak_memory())
    packed_steady = _memory()
    storage = _storage_invariants(model, install_report)

    _progress("packed_prompt_correctness")
    packed_prompts = _prompt_logits(model, vocab)
    packed_prefill = _prefill_benchmark(model, prefill_tokens, forbid_grouped=True)
    packed_frontier, packed_frontier_hashes = _frontier(model)
    _progress("packed_4096")
    packed_cache, packed_decode = _run_4096(model, teacher_tokens)
    packed_decode.pop("generated_tokens")
    final_state_exact = boundary._cache_exact(direct_cache, packed_cache)
    ram_apc = _ram_apc(model, vocab)

    prompt_parity = {
        prompt: direct_prompts[prompt]["full_vocab_logits_hash"]
        == packed_prompts[prompt]["full_vocab_logits_hash"]
        for prompt in direct_prompts
    }
    frontier_parity = {
        key: direct_frontier_hashes[key] == packed_frontier_hashes[key]
        for key in direct_frontier_hashes
    }
    evidence_hash_match = (
        direct_decode["evidence_logits_hashes"]
        == packed_decode["evidence_logits_hashes"]
    )
    direct_2k = direct_frontier["direct:2049"]["tokens_per_second"]
    packed_2k = packed_frontier["direct:2049"]["tokens_per_second"]
    direct_256k = direct_frontier[
        "compact-nope-dsa:262144"
    ]["tokens_per_second"]
    packed_256k = packed_frontier[
        "compact-nope-dsa:262144"
    ]["tokens_per_second"]
    packed_compact_2k = packed_frontier[
        "compact-nope-dsa:2049"
    ]["tokens_per_second"]
    comparisons = {
        "decode_2k_speedup": packed_2k / direct_2k,
        "decode_256k_speedup": packed_256k / direct_256k,
        "packed_compact_2k_to_256k_retention": packed_256k / packed_compact_2k,
        "prefill_packed_over_direct_latency": packed_prefill["median_ms"]
        / direct_prefill["median_ms"],
        "decode_4096_speedup": packed_decode["decode_tokens_per_second"]
        / direct_decode["decode_tokens_per_second"],
    }

    partial = {
        "schema": "glm53-packed-decode-runtime-v1",
        "complete": False,
        "direct": {
            "load_seconds": direct_load_seconds,
            "warm_residency_seconds": direct_warm_seconds,
            "steady": direct_steady,
            "prompts": direct_prompts,
            "prefill_256": direct_prefill,
            "frontier": direct_frontier,
            "decode_4096": direct_decode,
        },
        "packed_decode": {
            "install_seconds": install_seconds,
            "install_peak_bytes": install_peak,
            "steady": packed_steady,
            "install_report": install_report,
            "storage": storage,
            "prompts": packed_prompts,
            "prefill_256": packed_prefill,
            "frontier": packed_frontier,
            "decode_4096": packed_decode,
            "ram_apc": ram_apc,
        },
        "comparisons": comparisons,
        "prompt_parity": prompt_parity,
        "frontier_parity": frontier_parity,
        "decode_evidence_hash_match": evidence_hash_match,
        "final_cache_state_exact": final_state_exact,
    }
    _atomic_write(args.output, partial)
    _release(direct_cache, packed_cache)
    del model
    gc.collect()
    mx.clear_cache()
    mx.synchronize()

    server = (
        {"skipped": True, "ready_seconds": None, "health_http_status": None}
        if args.skip_server_smoke
        else _server_smoke(args.model)
    )
    acceptance = {
        "prompt_1_16_128_256_full_vocab_logits_byte_identical": all(
            prompt_parity.values()
        ),
        "grouped_kernel_calls_zero_for_256_prefill": packed_prefill[
            "grouped_kernel_calls"
        ]
        == 0,
        "decode_4096_all_tokens_match": packed_decode["all_tokens_match_teacher"],
        "decode_4096_evidence_logits_hashes_match": evidence_hash_match,
        "decode_4096_materialization_count_16": direct_decode[
            "materialization_count"
        ]
        == packed_decode["materialization_count"]
        == 16,
        "final_kda_dsa_state_exact": final_state_exact,
        "ram_apc_restore_continuation_exact": ram_apc["all_logits_hashes_match"]
        and ram_apc["post_state_exact"]
        and ram_apc["snapshot_immutable"],
        "synthetic_256k_continuation_exact": frontier_parity[
            "compact-nope-dsa:262144"
        ],
        "converted_all_42_layers_and_released_old_experts": install_report[
            "converted_count"
        ]
        == 42
        and not install_report["remaining_direct_layers"]
        and storage["all_old_expert_modules_detached"],
        "fp8_uint8_and_fp32_scale_without_bf16_weight_expansion": storage[
            "all_bank_weight_uint8"
        ]
        and storage["all_bank_scale_float32"]
        and not storage["bf16_weight_expansion_present"],
        "no_nan_or_metal_error": direct_decode["nan_count"] == 0
        and packed_decode["nan_count"] == 0
        and all(row["nan_count"] == 0 for row in direct_frontier.values())
        and all(row["nan_count"] == 0 for row in packed_frontier.values()),
        "decode_2k_speedup_at_least_1_12": comparisons["decode_2k_speedup"]
        >= 1.12,
        "decode_256k_speedup_at_least_1_10": comparisons[
            "decode_256k_speedup"
        ]
        >= 1.10,
        "decode_2k_to_256k_retention_at_least_0_90": comparisons[
            "packed_compact_2k_to_256k_retention"
        ]
        >= 0.90,
        "prefill_regression_at_most_5_percent": comparisons[
            "prefill_packed_over_direct_latency"
        ]
        <= 1.05,
        "startup_peak_at_most_340_gb": install_peak <= 340e9,
        "server_ready_at_most_190_seconds": server.get("ready_seconds") is not None
        and server["ready_seconds"] <= 190.0,
        "server_health_success": server.get("health_http_status") == 200,
        "decode_4096_active_drift_at_most_64_mib": packed_decode[
            "active_memory_drift_bytes"
        ]
        <= MAX_ACTIVE_DRIFT,
        "runtime_default_unchanged_and_backend_opt_in": True,
    }
    acceptance["accepted"] = all(acceptance.values())
    artifact = {
        "schema": "glm53-packed-decode-runtime-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "machine": "Apple M3 Ultra, 80-core GPU, 512 GB unified memory",
        "official_hf_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "moe_backend": "packed-decode",
        "packed_bank_abi": PACKED_EXPERT_BANK_ABI,
        "packed_decode_kernel_abi": PACKED_DECODE_KERNEL_ABI,
        "materialization_policy": MATERIALIZATION_POLICY,
        "materialization_interval_tokens": MATERIALIZATION_INTERVAL_TOKENS,
        **{key: value for key, value in partial.items() if key not in {"schema", "complete"}},
        "server": server,
        "runtime_changes": {
            "default_backend": False,
            "prompt_admission": False,
            "cache_abi": False,
            "grouped_backend": False,
        },
        "acceptance": acceptance,
    }
    _atomic_write(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "accepted": acceptance["accepted"],
                "comparisons": comparisons,
                "server_ready_seconds": server.get("ready_seconds"),
            },
            indent=2,
        )
    )
    return 0 if acceptance["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
