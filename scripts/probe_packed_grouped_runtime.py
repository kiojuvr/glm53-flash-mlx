#!/usr/bin/env python3
"""Full-model opt-in packed/grouped MoE acceptance gate for M3 Ultra."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from glm53_flash_mlx.grouped_fp8 import SortedGroupedFP8MoE
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint
from glm53_flash_mlx.packed import PackedFP8MoE

DEFAULT_PROMPT = "Reply with exactly: OK"


def _snapshot() -> dict[str, int]:
    mx.synchronize()
    return {
        "active_bytes": mx.get_active_memory(),
        "cache_bytes": mx.get_cache_memory(),
        "peak_bytes": mx.get_peak_memory(),
    }


def _deterministic_tokens(tokens: int, vocab: int) -> np.ndarray:
    return (
        (np.arange(tokens, dtype=np.uint64) * 7919) % (vocab - 1024) + 100
    ).astype(np.uint32)[None, :]


def _prefill(model, token_ids: np.ndarray) -> tuple[dict, np.ndarray]:
    mx.clear_cache()
    mx.reset_peak_memory()
    started = time.perf_counter()
    output = model(mx.array(token_ids), cache=model.make_cache())
    logits = output.logits[0, -1].astype(mx.float32)
    mx.eval(logits)
    mx.synchronize()
    elapsed = time.perf_counter() - started
    array = np.ascontiguousarray(np.asarray(logits), dtype=np.float32)
    return (
        {
            "elapsed_seconds": elapsed,
            "tokens_per_second": token_ids.size / elapsed,
            "last_logits_f32_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "active_bytes": mx.get_active_memory(),
            "peak_bytes": mx.get_peak_memory(),
        },
        array,
    )


def _decode(model, *, measured_steps: int = 16) -> dict:
    cache = model.make_cache()
    token = 1
    samples = []
    generated = []
    for step in range(measured_steps + 2):
        started = time.perf_counter()
        output = model(mx.array([[token]], dtype=mx.uint32), cache=cache)
        token = int(mx.argmax(output.logits[0, -1]).item())
        elapsed = time.perf_counter() - started
        generated.append(token)
        if step >= 2:
            samples.append(elapsed)
    median = statistics.median(samples)
    return {
        "warmup_steps": 2,
        "measured_steps": measured_steps,
        "median_seconds": median,
        "median_tokens_per_second": 1.0 / median,
        "samples_seconds": samples,
        "generated_token_ids": generated,
    }


def _error_metrics(expected, actual) -> dict:
    diff = actual.astype(mx.float32) - expected.astype(mx.float32)
    absolute = mx.abs(diff)
    values = (mx.max(absolute), mx.mean(absolute), mx.sqrt(mx.mean(diff * diff)))
    mx.eval(*values)
    return {
        "max_abs": float(values[0].item()),
        "mean_abs": float(values[1].item()),
        "rms": float(values[2].item()),
        "allclose_rtol_0_02_atol_0_02": bool(
            mx.allclose(actual, expected, rtol=0.02, atol=0.02).item()
        ),
    }


def _transition_parity(model, *, tokens: int = 32) -> list[dict]:
    layers = model.language_model.model.layers
    grouped_ids = [
        layer_id
        for layer_id, layer in enumerate(layers)
        if isinstance(layer.mlp, SortedGroupedFP8MoE)
    ]
    selected = [grouped_ids[0], grouped_ids[len(grouped_ids) // 2], grouped_ids[-1]]
    results = []
    for layer_id in selected:
        grouped = layers[layer_id].mlp
        fallback = PackedFP8MoE(
            grouped.bank,
            grouped.config,
            grouped.gate,
            grouped.shared_experts,
        )
        size = tokens * grouped.config.hidden_size
        x = mx.sin(mx.arange(size, dtype=mx.float32) * 0.0009765625)
        x = x.reshape(1, tokens, grouped.config.hidden_size).astype(mx.bfloat16)
        expected = fallback(x)
        actual = grouped(x)
        mx.eval(expected, actual)
        results.append({"layer": layer_id, **_error_metrics(expected, actual)})
        del expected, actual, fallback, x
        mx.clear_cache()
    return results


def _oracle(model, processor, expected_path: Path, *, tokens: int = 16) -> dict:
    expected = json.loads(expected_path.read_text())
    formatted = processor.apply_chat_template(
        [{"role": "user", "content": DEFAULT_PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = processor(formatted, return_tensors="np", add_special_tokens=True)
    prompt_ids = np.asarray(encoded["input_ids"], dtype=np.int32).reshape(1, -1)
    cache = model.make_cache()
    output = model(mx.array(prompt_ids), cache=cache)
    hashes = []
    generated = []
    for step in range(tokens):
        logits = output.logits[0, -1].astype(mx.float32)
        mx.eval(logits)
        array = np.ascontiguousarray(np.asarray(logits), dtype=np.float32)
        hashes.append(hashlib.sha256(array.tobytes()).hexdigest())
        top2 = np.argpartition(array, -2)[-2:]
        top2 = top2[np.argsort(array[top2])[::-1]]
        token = int(top2[0])
        generated.append(token)
        if step + 1 < tokens:
            output = model(mx.array([[token]], dtype=mx.uint32), cache=cache)
    expected_steps = expected["steps"][:tokens]
    expected_hashes = [row["logits_f32_sha256"] for row in expected_steps]
    expected_tokens = [row["token"] for row in expected_steps]
    return {
        "prompt_tokens": int(prompt_ids.size),
        "generation_tokens": tokens,
        "generated_token_ids": generated,
        "expected_token_ids": expected_tokens,
        "all_token_ids_match": generated == expected_tokens,
        "all_step_logits_hashes_match": hashes == expected_hashes,
        "actual_logits_hashes": hashes,
    }


def _logits_comparison(reference: np.ndarray, actual: np.ndarray, *, top_k: int = 10):
    reference64 = reference.astype(np.float64)
    actual64 = actual.astype(np.float64)
    diff = actual64 - reference64
    reference_top = np.argsort(reference)[-top_k:][::-1]
    actual_top = np.argsort(actual)[-top_k:][::-1]
    return {
        "relative_l2": float(np.linalg.norm(diff) / np.linalg.norm(reference64)),
        "max_abs": float(np.max(np.abs(diff))),
        "argmax_match": int(reference_top[0]) == int(actual_top[0]),
        "reference_top_k": reference_top.tolist(),
        "actual_top_k": actual_top.tolist(),
        "top_k_order_match": np.array_equal(reference_top, actual_top),
        "top_k_set_match": set(reference_top.tolist()) == set(actual_top.tolist()),
        "top_k_overlap": len(
            set(reference_top.tolist()) & set(actual_top.tolist())
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--oracle", type=Path, default=Path("oracles/glm53-official-greedy-16.json"))
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    args = parser.parse_args()

    report = inspect_checkpoint(args.model, require_server_ready=True)
    raw_config = json.loads((Path(args.model) / "config.json").read_text())
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    vocab = int(raw_config["text_config"]["vocab_size"])
    token_ids = _deterministic_tokens(args.tokens, vocab)

    mx.reset_peak_memory()
    direct_started = time.perf_counter()
    direct_model, processor = load(args.model)
    direct_load_seconds = time.perf_counter() - direct_started
    warm_started = time.perf_counter()
    warm_residency(direct_model)
    direct_warm_seconds = time.perf_counter() - warm_started
    mx.clear_cache()
    direct_steady = _snapshot()
    direct_prefill, reference_logits = _prefill(direct_model, token_ids)
    direct_decode = _decode(direct_model)

    del direct_model
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    after_direct_release = _snapshot()

    mx.reset_peak_memory()
    grouped_started = time.perf_counter()
    grouped_model, _ = load(
        args.model, experimental_packed_grouped_moe=True
    )
    grouped_load_seconds = time.perf_counter() - grouped_started
    install_report = grouped_model._glm53_packed_grouped_report
    warm_started = time.perf_counter()
    warm_residency(grouped_model)
    grouped_warm_seconds = time.perf_counter() - warm_started
    mx.clear_cache()
    grouped_steady = _snapshot()
    startup_peak_bytes = mx.get_peak_memory()

    transition_parity = _transition_parity(grouped_model)
    grouped_prefill, grouped_logits = _prefill(grouped_model, token_ids)
    grouped_decode = _decode(grouped_model)
    oracle = _oracle(grouped_model, processor, args.oracle)
    logits = _logits_comparison(reference_logits, grouped_logits)

    prefill_speedup = (
        direct_prefill["elapsed_seconds"] / grouped_prefill["elapsed_seconds"]
    )
    decode_speed_ratio = (
        grouped_decode["median_tokens_per_second"]
        / direct_decode["median_tokens_per_second"]
    )
    converted_ok = (
        install_report["converted_count"] == 42
        and not install_report["remaining_direct_layers"]
        and install_report["all_old_expert_modules_detached"]
    )
    parity_ok = all(
        row["allclose_rtol_0_02_atol_0_02"] for row in transition_parity
    )
    runtime_acceptance = {
        "converted_42_layers": converted_ok,
        "startup_peak_at_most_340_gb": startup_peak_bytes <= 340e9,
        "steady_at_most_321_gb": grouped_steady["active_bytes"] <= 321e9,
        "transition_parity_early_middle_late": parity_ok,
        "prefill_speedup_at_least_1_20": prefill_speedup >= 1.20,
        "decode_slowdown_at_most_3_percent": decode_speed_ratio >= 0.97,
        "oracle_all_hashes_match": oracle["all_step_logits_hashes_match"],
    }
    runtime_accepted = all(runtime_acceptance.values())
    acceptance = {
        **runtime_acceptance,
        "runtime_integration_and_performance_accepted": runtime_accepted,
        "full_model_grouped_correctness_accepted": False,
        "eligible_for_default_or_prompt_limit_increase": False,
        "accepted": False,
        "reason": (
            "the short oracle exercises packed fallback, not grouped prefill; "
            "full-model grouped quality impact remains unassessed"
        ),
    }
    output = {
        "schema": "glm53-packed-grouped-runtime-gate-v1",
        "official_hf_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "prompt_tokens": args.tokens,
        "direct": {
            "load_seconds": direct_load_seconds,
            "warm_residency_seconds": direct_warm_seconds,
            "steady": direct_steady,
            "prefill": direct_prefill,
            "decode": direct_decode,
        },
        "after_direct_release": after_direct_release,
        "packed_grouped": {
            "load_and_pack_seconds": grouped_load_seconds,
            "warm_residency_seconds": grouped_warm_seconds,
            "startup_peak_bytes": startup_peak_bytes,
            "steady": grouped_steady,
            "converted_layers": install_report["converted_layers"],
            "converted_count": install_report["converted_count"],
            "remaining_direct_layers": install_report["remaining_direct_layers"],
            "all_old_expert_modules_detached": install_report[
                "all_old_expert_modules_detached"
            ],
            "layer_conversion_memory": install_report["layers"],
            "transition_parity": transition_parity,
            "prefill": grouped_prefill,
            "decode": grouped_decode,
            "oracle_16": oracle,
        },
        "comparison": {
            "prefill_speedup": prefill_speedup,
            "decode_speed_ratio": decode_speed_ratio,
            "final_logits": logits,
        },
        "acceptance": acceptance,
        "runtime_default_changed": False,
        "prompt_limit_changed": False,
    }
    print(json.dumps(output, indent=2), flush=True)
    return 0 if runtime_accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
