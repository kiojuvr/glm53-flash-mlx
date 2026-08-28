#!/usr/bin/env python3
"""Localize full-model divergence introduced by grouped FP8 MoE prefill."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np

from glm53_flash_mlx.grouped_fp8 import SortedGroupedFP8MoE
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint

DISABLED_MIN_ROUTES = 1 << 30


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), file=sys.stderr, flush=True)


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


def _benchmark_prefill(
    model,
    token_ids: np.ndarray,
    *,
    warmups: int,
    repeats: int,
    phase: str,
) -> tuple[dict, np.ndarray]:
    cold, logits = _prefill(model, token_ids)
    _progress(f"{phase}_cold", elapsed_seconds=cold["elapsed_seconds"])
    warmup_rows = []
    for sample in range(warmups):
        row, logits = _prefill(model, token_ids)
        warmup_rows.append(row)
        _progress(
            f"{phase}_warmup",
            sample=sample + 1,
            elapsed_seconds=row["elapsed_seconds"],
        )
    measured_rows = []
    for sample in range(repeats):
        row, logits = _prefill(model, token_ids)
        measured_rows.append(row)
        _progress(
            f"{phase}_measured",
            sample=sample + 1,
            elapsed_seconds=row["elapsed_seconds"],
        )
    seconds = [row["elapsed_seconds"] for row in measured_rows]
    median = statistics.median(seconds)
    return (
        {
            "cold_first_pass": cold,
            "warmups": warmup_rows,
            "warmup_count": warmups,
            "measured": measured_rows,
            "measured_count": repeats,
            "median_seconds": median,
            "median_tokens_per_second": token_ids.size / median,
        },
        logits,
    )


def _top_k(values: np.ndarray, k: int) -> np.ndarray:
    candidates = np.argpartition(values, -k)[-k:]
    return candidates[np.argsort(values[candidates], kind="stable")][::-1]


def _logits_metrics(reference: np.ndarray, actual: np.ndarray, top_k: int = 10) -> dict:
    reference64 = reference.astype(np.float64)
    actual64 = actual.astype(np.float64)
    diff = actual64 - reference64
    reference_top = _top_k(reference, top_k)
    actual_top = _top_k(actual, top_k)

    reference_shifted = reference64 - np.max(reference64)
    actual_shifted = actual64 - np.max(actual64)
    reference_log_z = np.log(np.exp(reference_shifted).sum())
    actual_log_z = np.log(np.exp(actual_shifted).sum())
    reference_log_p = reference_shifted - reference_log_z
    actual_log_p = actual_shifted - actual_log_z
    reference_p = np.exp(reference_log_p)

    return {
        "array_equal": bool(np.array_equal(reference, actual)),
        "relative_l2": float(np.linalg.norm(diff) / np.linalg.norm(reference64)),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rms": float(np.sqrt(np.mean(diff * diff))),
        "kl_reference_to_actual": float(
            np.sum(reference_p * (reference_log_p - actual_log_p))
        ),
        "argmax_match": int(reference_top[0]) == int(actual_top[0]),
        "reference_argmax": int(reference_top[0]),
        "actual_argmax": int(actual_top[0]),
        "reference_top_k": reference_top.tolist(),
        "actual_top_k": actual_top.tolist(),
        "top_k_order_match": bool(np.array_equal(reference_top, actual_top)),
        "top_k_set_match": set(reference_top.tolist()) == set(actual_top.tolist()),
        "top_k_overlap": len(set(reference_top.tolist()) & set(actual_top.tolist())),
    }


def _grouped_layers(model) -> dict[int, SortedGroupedFP8MoE]:
    return {
        layer_id: layer.mlp
        for layer_id, layer in enumerate(model.language_model.model.layers)
        if isinstance(layer.mlp, SortedGroupedFP8MoE)
    }


def _set_grouped_layers(
    grouped: dict[int, SortedGroupedFP8MoE],
    enabled: set[int],
    *,
    grouped_min_routes: int,
) -> None:
    unknown = enabled.difference(grouped)
    if unknown:
        raise ValueError(f"requested non-grouped layers: {sorted(unknown)}")
    for layer_id, moe in grouped.items():
        moe.min_routes = (
            grouped_min_routes if layer_id in enabled else DISABLED_MIN_ROUTES
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--grouped-min-routes", type=int, default=256)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = inspect_checkpoint(args.model, require_server_ready=True)
    raw_config = json.loads((Path(args.model) / "config.json").read_text())
    vocab = int(raw_config["text_config"]["vocab_size"])
    token_ids = _deterministic_tokens(args.tokens, vocab)
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))

    _progress("load_direct")
    direct_started = time.perf_counter()
    direct_model, _ = load(args.model)
    direct_load_seconds = time.perf_counter() - direct_started
    warm_started = time.perf_counter()
    warm_residency(direct_model)
    direct_residency_seconds = time.perf_counter() - warm_started
    direct_benchmark, reference_logits = _benchmark_prefill(
        direct_model,
        token_ids,
        warmups=args.warmups,
        repeats=args.repeats,
        phase="direct",
    )

    del direct_model
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    _progress("direct_released", active_bytes=mx.get_active_memory())

    _progress("load_packed_grouped")
    grouped_started = time.perf_counter()
    grouped_model, _ = load(args.model, experimental_packed_grouped_moe=True)
    grouped_load_seconds = time.perf_counter() - grouped_started
    warm_started = time.perf_counter()
    warm_residency(grouped_model)
    grouped_residency_seconds = time.perf_counter() - warm_started
    grouped = _grouped_layers(grouped_model)
    layer_ids = sorted(grouped)
    if layer_ids != list(range(3, 45)):
        raise RuntimeError(f"expected grouped layers 3..44, got {layer_ids}")

    all_layers = set(layer_ids)
    _set_grouped_layers(
        grouped, all_layers, grouped_min_routes=args.grouped_min_routes
    )
    grouped_benchmark, grouped_logits = _benchmark_prefill(
        grouped_model,
        token_ids,
        warmups=args.warmups,
        repeats=args.repeats,
        phase="grouped",
    )
    all_grouped_metrics = _logits_metrics(reference_logits, grouped_logits)

    _set_grouped_layers(grouped, set(), grouped_min_routes=args.grouped_min_routes)
    fallback_timing, fallback_logits = _prefill(grouped_model, token_ids)
    fallback_metrics = _logits_metrics(reference_logits, fallback_logits)
    _progress(
        "packed_fallback",
        elapsed_seconds=fallback_timing["elapsed_seconds"],
        array_equal=fallback_metrics["array_equal"],
    )
    if not fallback_metrics["array_equal"]:
        raise RuntimeError("packed fallback logits are not byte-identical to Direct")

    one_layer = []
    for layer_id in layer_ids:
        _set_grouped_layers(
            grouped, {layer_id}, grouped_min_routes=args.grouped_min_routes
        )
        timing, logits = _prefill(grouped_model, token_ids)
        metrics = _logits_metrics(reference_logits, logits)
        row = {"layer": layer_id, "timing": timing, "logits": metrics}
        one_layer.append(row)
        _progress(
            "one_layer_grouped",
            layer=layer_id,
            elapsed_seconds=timing["elapsed_seconds"],
            relative_l2=metrics["relative_l2"],
            max_abs=metrics["max_abs"],
        )

    cumulative = []
    enabled: set[int] = set()
    for layer_id in layer_ids:
        enabled.add(layer_id)
        _set_grouped_layers(
            grouped, enabled, grouped_min_routes=args.grouped_min_routes
        )
        timing, logits = _prefill(grouped_model, token_ids)
        metrics = _logits_metrics(reference_logits, logits)
        row = {
            "through_layer": layer_id,
            "enabled_count": len(enabled),
            "timing": timing,
            "logits": metrics,
        }
        cumulative.append(row)
        _progress(
            "cumulative_grouped",
            through_layer=layer_id,
            enabled_count=len(enabled),
            elapsed_seconds=timing["elapsed_seconds"],
            relative_l2=metrics["relative_l2"],
            max_abs=metrics["max_abs"],
        )

    direct_median = direct_benchmark["median_seconds"]
    grouped_median = grouped_benchmark["median_seconds"]
    worst_one_layer = max(one_layer, key=lambda row: row["logits"]["relative_l2"])
    worst_cumulative = max(
        cumulative, key=lambda row: row["logits"]["relative_l2"]
    )
    one_layer_argmax_mismatches = [
        row["layer"] for row in one_layer if not row["logits"]["argmax_match"]
    ]
    cumulative_argmax_mismatches = [
        row["through_layer"]
        for row in cumulative
        if not row["logits"]["argmax_match"]
    ]
    output = {
        "schema": "glm53-grouped-fp8-divergence-localization-v1",
        "date": date.today().isoformat(),
        "official_hf_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "prompt": {
            "kind": "deterministic-token-sequence",
            "tokens": args.tokens,
            "token_formula": "(arange(tokens) * 7919) % (vocab_size - 1024) + 100",
        },
        "dispatch": {
            "grouped_min_routes_when_enabled": args.grouped_min_routes,
            "disabled_min_routes": DISABLED_MIN_ROUTES,
            "routes_per_layer": args.tokens * 8,
        },
        "direct": {
            "load_seconds": direct_load_seconds,
            "warm_residency_seconds": direct_residency_seconds,
            "prefill": direct_benchmark,
        },
        "packed_grouped": {
            "load_and_pack_seconds": grouped_load_seconds,
            "warm_residency_seconds": grouped_residency_seconds,
            "prefill_all_layers_grouped": grouped_benchmark,
        },
        "performance": {
            "method": (
                "first pass recorded separately; "
                f"{args.warmups} warmups; median of {args.repeats} measured passes"
            ),
            "cold_scope": (
                "first evaluated pass after each backend load in one process; "
                "the grouped pass can reuse compiled non-MoE kernels"
            ),
            "direct_median_seconds": direct_median,
            "grouped_median_seconds": grouped_median,
            "warm_median_speedup": direct_median / grouped_median,
        },
        "correctness": {
            "packed_fallback": {
                "timing": fallback_timing,
                "logits": fallback_metrics,
                "required_array_equal": True,
                "accepted": fallback_metrics["array_equal"],
            },
            "all_layers_grouped": all_grouped_metrics,
            "localization_summary": {
                "one_layer_worst_relative_l2": worst_one_layer["logits"][
                    "relative_l2"
                ],
                "one_layer_worst_layer": worst_one_layer["layer"],
                "one_layer_argmax_mismatch_layers": one_layer_argmax_mismatches,
                "cumulative_worst_relative_l2": worst_cumulative["logits"][
                    "relative_l2"
                ],
                "cumulative_worst_through_layer": worst_cumulative[
                    "through_layer"
                ],
                "cumulative_argmax_mismatch_through_layers": (
                    cumulative_argmax_mismatches
                ),
                "observed_pattern": (
                    "broad early-layer amplification followed by a non-monotonic "
                    "plateau; not a single-layer spike or smooth accumulation"
                ),
            },
            "one_layer_grouped": one_layer,
            "cumulative_grouped": cumulative,
            "full_model_grouped_correctness_accepted": False,
            "reason": "localization evidence only; real-prompt quality suite remains required",
        },
        "runtime_default_changed": False,
        "prompt_limit_changed": False,
    }
    serialized = json.dumps(output, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(serialized)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "packed_fallback_array_equal": fallback_metrics["array_equal"],
                    "all_grouped_relative_l2": all_grouped_metrics["relative_l2"],
                    "warm_median_speedup": direct_median / grouped_median,
                },
                indent=2,
            ),
            flush=True,
        )
    else:
        print(serialized, end="", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
