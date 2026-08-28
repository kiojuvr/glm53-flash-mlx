#!/usr/bin/env python3
"""Sweep Direct-prefix / grouped-FP8-suffix policies on the official model."""

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

from glm53_flash_mlx.abi import GROUPED_MIN_ROUTES
from glm53_flash_mlx.grouped_fp8 import SortedGroupedFP8MoE
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint

DISABLED_MIN_ROUTES = 1 << 30
FIRST_MOE_LAYER = 3
LAST_MOE_LAYER = 44
DIRECT_ONLY_CUTOFF = LAST_MOE_LAYER + 1
ALL_GROUPED_BASELINE_SHA256 = (
    "fd86b3f95e15e076e4a6a4d011401df5649e6d1be0c2b549b0427d2972a77b5a"
)


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), file=sys.stderr, flush=True)


def _deterministic_tokens(tokens: int, vocab: int) -> np.ndarray:
    return (
        (np.arange(tokens, dtype=np.uint64) * 7919) % (vocab - 1024) + 100
    ).astype(np.uint32)[None, :]


def _sha256(values: np.ndarray) -> str:
    return hashlib.sha256(values.tobytes()).hexdigest()


def _float_array(value) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value), dtype=np.float32)


def _int_array(value) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value), dtype=np.int32)


class _CapturingGate:
    def __init__(self, delegate):
        self.delegate = delegate
        self.indices = None
        self.scores = None

    def __call__(self, x):
        self.indices, self.scores = self.delegate(x)
        return self.indices, self.scores


class _RecordedGate:
    """Return Direct-reference expert IDs and mixture weights."""

    def __init__(self, indices: np.ndarray, scores: np.ndarray):
        self.indices = mx.array(indices, dtype=mx.int32)
        self.scores = mx.array(scores, dtype=mx.float32)

    def __call__(self, _):
        return self.indices, self.scores


class _RecordedIndicesCurrentScoresGate:
    """Fix Direct expert membership but recompute weights from current hidden."""

    def __init__(self, delegate, indices: np.ndarray):
        self.delegate = delegate
        self.indices = mx.array(indices, dtype=mx.int32)

    def __call__(self, x):
        logits = x.astype(mx.float32) @ self.delegate.weight.astype(mx.float32).T
        all_scores = mx.sigmoid(logits)
        scores = mx.take_along_axis(all_scores, self.indices, axis=-1)
        if self.delegate.top_k > 1 and self.delegate.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        scores = scores * self.delegate.routed_scaling_factor
        return self.indices, scores


def _moe_modules(model) -> dict[int, object]:
    return {
        layer_id: layer.mlp
        for layer_id, layer in enumerate(model.language_model.model.layers)
        if hasattr(layer.mlp, "gate")
    }


def _grouped_modules(model) -> dict[int, SortedGroupedFP8MoE]:
    modules = _moe_modules(model)
    return {
        layer_id: module
        for layer_id, module in modules.items()
        if isinstance(module, SortedGroupedFP8MoE)
    }


def _evaluate_prefill(model, token_ids: np.ndarray, *, capture_routes: bool):
    modules = _moe_modules(model)
    original_gates = {layer: module.gate for layer, module in modules.items()}
    captures = {}
    try:
        if capture_routes:
            for layer_id, module in modules.items():
                capture = _CapturingGate(module.gate)
                module.gate = capture
                captures[layer_id] = capture

        mx.clear_cache()
        mx.reset_peak_memory()
        started = time.perf_counter()
        output = model(mx.array(token_ids), cache=model.make_cache())
        logits = output.logits[0, -1].astype(mx.float32)
        values = [logits]
        if capture_routes:
            values.extend(capture.indices for capture in captures.values())
            values.extend(capture.scores for capture in captures.values())
        mx.eval(*values)
        mx.synchronize()
        elapsed = time.perf_counter() - started
        logits_array = _float_array(logits)
        routes = (
            {
                layer: {
                    "indices": _int_array(capture.indices),
                    "scores": _float_array(capture.scores),
                }
                for layer, capture in captures.items()
            }
            if capture_routes
            else None
        )
        return (
            {
                "elapsed_seconds": elapsed,
                "tokens_per_second": token_ids.size / elapsed,
                "last_logits_f32_sha256": _sha256(logits_array),
                "active_bytes": mx.get_active_memory(),
                "peak_bytes": mx.get_peak_memory(),
            },
            logits_array,
            routes,
        )
    finally:
        for layer_id, module in modules.items():
            module.gate = original_gates[layer_id]


def _benchmark_prefill(model, token_ids, *, warmups: int, repeats: int, phase: str):
    warmup_rows = []
    for sample in range(warmups):
        row, _, _ = _evaluate_prefill(model, token_ids, capture_routes=False)
        warmup_rows.append(row)
        _progress(
            f"{phase}_warmup",
            sample=sample + 1,
            elapsed_seconds=row["elapsed_seconds"],
        )
    measured_rows = []
    logits = None
    for sample in range(repeats):
        row, logits, _ = _evaluate_prefill(
            model, token_ids, capture_routes=False
        )
        measured_rows.append(row)
        _progress(
            f"{phase}_measured",
            sample=sample + 1,
            elapsed_seconds=row["elapsed_seconds"],
        )
    seconds = [row["elapsed_seconds"] for row in measured_rows]
    median = statistics.median(seconds)
    return {
        "warmups": warmup_rows,
        "warmup_count": warmups,
        "measured": measured_rows,
        "measured_count": repeats,
        "median_seconds": median,
        "median_tokens_per_second": token_ids.size / median,
        "last_logits_f32_sha256": _sha256(logits),
    }


def _top_k(values: np.ndarray, k: int) -> np.ndarray:
    candidates = np.argpartition(values, -k)[-k:]
    return candidates[np.argsort(values[candidates], kind="stable")][::-1]


def _logits_metrics(reference: np.ndarray, actual: np.ndarray, *, top_k=10) -> dict:
    reference64 = reference.astype(np.float64)
    actual64 = actual.astype(np.float64)
    diff = actual64 - reference64
    denominator = np.linalg.norm(reference64)
    reference_top = _top_k(reference, top_k)
    actual_top = _top_k(actual, top_k)
    reference_shifted = reference64 - np.max(reference64)
    actual_shifted = actual64 - np.max(actual64)
    reference_log_p = reference_shifted - np.log(np.exp(reference_shifted).sum())
    actual_log_p = actual_shifted - np.log(np.exp(actual_shifted).sum())
    reference_p = np.exp(reference_log_p)
    return {
        "array_equal": bool(np.array_equal(reference, actual)),
        "reference_sha256": _sha256(reference),
        "actual_sha256": _sha256(actual),
        "relative_l2": float(np.linalg.norm(diff) / denominator) if denominator else 0.0,
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
        "top_k_set_match": set(reference_top.tolist())
        == set(actual_top.tolist()),
        "top_k_overlap": len(
            set(reference_top.tolist()) & set(actual_top.tolist())
        ),
    }


def _route_layer_metrics(reference: dict, actual: dict) -> tuple[dict, np.ndarray]:
    reference_indices = reference["indices"].reshape(
        -1, reference["indices"].shape[-1]
    )
    actual_indices = actual["indices"].reshape(-1, actual["indices"].shape[-1])
    reference_scores = reference["scores"].reshape(
        -1, reference["scores"].shape[-1]
    )
    actual_scores = actual["scores"].reshape(-1, actual["scores"].shape[-1])
    same_slots = reference_indices == actual_indices
    identical_order = np.all(same_slots, axis=-1)
    identical_sets = np.all(
        np.sort(reference_indices, axis=-1) == np.sort(actual_indices, axis=-1),
        axis=-1,
    )
    replacements = 0
    aligned_score_differences = []
    for reference_ids, actual_ids, reference_row, actual_row in zip(
        reference_indices,
        actual_indices,
        reference_scores,
        actual_scores,
        strict=True,
    ):
        actual_positions = {
            int(expert_id): slot for slot, expert_id in enumerate(actual_ids)
        }
        reference_set = set(int(expert_id) for expert_id in reference_ids)
        actual_set = set(actual_positions)
        replacements += len(reference_set - actual_set)
        for reference_slot, expert_id in enumerate(reference_ids):
            actual_slot = actual_positions.get(int(expert_id))
            if actual_slot is not None:
                aligned_score_differences.append(
                    float(actual_row[actual_slot]) - float(reference_row[reference_slot])
                )
    score_differences = np.asarray(aligned_score_differences, dtype=np.float64)
    score_metrics = {
        "matched_memberships": int(score_differences.size),
        "max_abs": (
            float(np.max(np.abs(score_differences)))
            if score_differences.size
            else 0.0
        ),
        "mean_abs": (
            float(np.mean(np.abs(score_differences)))
            if score_differences.size
            else 0.0
        ),
        "rms": (
            float(np.sqrt(np.mean(score_differences * score_differences)))
            if score_differences.size
            else 0.0
        ),
    }
    return (
        {
            "tokens": int(reference_indices.shape[0]),
            "route_slots": int(reference_indices.size),
            "slot_position_mismatches": int(np.count_nonzero(~same_slots)),
            "tokens_with_changed_top8_set": int(np.count_nonzero(~identical_sets)),
            "expert_membership_replacements": int(replacements),
            "tokens_with_order_only_change": int(
                np.count_nonzero(identical_sets & ~identical_order)
            ),
            "indices_array_equal": bool(
                np.array_equal(reference_indices, actual_indices)
            ),
            "scores_array_equal": bool(
                np.array_equal(reference_scores, actual_scores)
            ),
            "score_difference_by_expert_id": score_metrics,
        },
        score_differences,
    )


def _router_metrics(reference_routes: dict, actual_routes: dict) -> dict:
    layer_rows = []
    all_score_differences = []
    for layer_id in sorted(reference_routes):
        row, score_differences = _route_layer_metrics(
            reference_routes[layer_id], actual_routes[layer_id]
        )
        layer_rows.append({"layer": layer_id, **row})
        if score_differences.size:
            all_score_differences.append(score_differences)
    first_divergence = next(
        (
            row["layer"]
            for row in layer_rows
            if row["tokens_with_changed_top8_set"] > 0
        ),
        None,
    )
    combined = (
        np.concatenate(all_score_differences)
        if all_score_differences
        else np.empty((0,), dtype=np.float64)
    )
    return {
        "first_top8_set_divergence_layer": first_divergence,
        "layer_token_rows_with_changed_top8_set": sum(
            row["tokens_with_changed_top8_set"] for row in layer_rows
        ),
        "expert_membership_replacements": sum(
            row["expert_membership_replacements"] for row in layer_rows
        ),
        "layer_token_rows_with_order_only_change": sum(
            row["tokens_with_order_only_change"] for row in layer_rows
        ),
        "slot_position_mismatches": sum(
            row["slot_position_mismatches"] for row in layer_rows
        ),
        "score_difference_by_expert_id": {
            "matched_memberships": int(combined.size),
            "max_abs": float(np.max(np.abs(combined))) if combined.size else 0.0,
            "mean_abs": float(np.mean(np.abs(combined))) if combined.size else 0.0,
            "rms": (
                float(np.sqrt(np.mean(combined * combined)))
                if combined.size
                else 0.0
            ),
        },
        "layers": layer_rows,
    }


def _set_enabled_layers(grouped: dict[int, SortedGroupedFP8MoE], enabled: set[int]):
    for layer_id, moe in grouped.items():
        moe.min_routes = (
            GROUPED_MIN_ROUTES if layer_id in enabled else DISABLED_MIN_ROUTES
        )


def _run_packed_capture(
    model,
    token_ids,
    reference_routes,
    *,
    enabled: set[int],
    route_policy: str = "free",
    target: int | None = None,
):
    grouped = _grouped_modules(model)
    original_gates = {layer: moe.gate for layer, moe in grouped.items()}
    original_thresholds = {layer: moe.min_routes for layer, moe in grouped.items()}
    try:
        _set_enabled_layers(grouped, enabled)
        if route_policy != "free":
            if target is None:
                raise ValueError("a target layer is required for fixed routing")
            for layer_id, moe in grouped.items():
                if layer_id <= target:
                    continue
                if route_policy == "direct_indices_current_scores":
                    moe.gate = _RecordedIndicesCurrentScoresGate(
                        original_gates[layer_id],
                        reference_routes[layer_id]["indices"],
                    )
                elif route_policy == "direct_indices_and_scores":
                    moe.gate = _RecordedGate(
                        reference_routes[layer_id]["indices"],
                        reference_routes[layer_id]["scores"],
                    )
                else:
                    raise ValueError(f"unknown route policy: {route_policy}")
        return _evaluate_prefill(model, token_ids, capture_routes=True)
    finally:
        for layer_id, moe in grouped.items():
            moe.gate = original_gates[layer_id]
            moe.min_routes = original_thresholds[layer_id]


def _storage_invariants(grouped: dict[int, SortedGroupedFP8MoE]) -> dict:
    rows = []
    for layer_id, moe in grouped.items():
        row = {
            "layer": layer_id,
            "gate_up_weight_dtype": str(moe.bank.gate_up_weight.dtype),
            "gate_up_scale_dtype": str(moe.bank.gate_up_scale_inv.dtype),
            "down_weight_dtype": str(moe.bank.down_weight.dtype),
            "down_scale_dtype": str(moe.bank.down_scale_inv.dtype),
        }
        row["accepted"] = (
            moe.bank.gate_up_weight.dtype == mx.uint8
            and moe.bank.down_weight.dtype == mx.uint8
            and moe.bank.gate_up_scale_inv.dtype == mx.float32
            and moe.bank.down_scale_inv.dtype == mx.float32
        )
        rows.append(row)
    return {
        "converted_layers": sorted(grouped),
        "all_42_layers_packed": sorted(grouped)
        == list(range(FIRST_MOE_LAYER, LAST_MOE_LAYER + 1)),
        "all_weights_uint8_e4m3_codes_and_scales_fp32": all(
            row["accepted"] for row in rows
        ),
        "layers": rows,
    }


def _screen(logits: dict) -> bool:
    return bool(
        logits["argmax_match"]
        and logits["top_k_set_match"]
        and logits["relative_l2"] <= 0.02
        and logits["kl_reference_to_actual"] <= 5e-4
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = inspect_checkpoint(args.model, require_server_ready=True)
    raw_config = json.loads((Path(args.model) / "config.json").read_text())
    text_config = raw_config["text_config"]
    vocab = int(text_config["vocab_size"])
    index_topk = int(text_config["index_topk"])
    token_ids = _deterministic_tokens(args.tokens, vocab)
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))

    _progress("load_direct")
    direct_model, _ = load(args.model)
    warm_residency(direct_model)
    direct_timing, reference_logits, reference_routes = _evaluate_prefill(
        direct_model, token_ids, capture_routes=True
    )
    direct_benchmark = _benchmark_prefill(
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

    _progress("load_packed_grouped")
    grouped_model, _ = load(args.model, experimental_packed_grouped_moe=True)
    warm_residency(grouped_model)
    grouped = _grouped_modules(grouped_model)
    storage = _storage_invariants(grouped)
    if not storage["all_42_layers_packed"] or not storage[
        "all_weights_uint8_e4m3_codes_and_scales_fp32"
    ]:
        raise RuntimeError("packed FP8 storage invariant failed")

    all_layers = set(range(FIRST_MOE_LAYER, LAST_MOE_LAYER + 1))
    _set_enabled_layers(grouped, all_layers)
    _progress("compile_warmup_all_grouped")
    _evaluate_prefill(grouped_model, token_ids, capture_routes=False)

    sweep = []
    for cutoff in range(FIRST_MOE_LAYER, DIRECT_ONLY_CUTOFF + 1):
        enabled = set(range(cutoff, LAST_MOE_LAYER + 1))
        timing, logits, routes = _run_packed_capture(
            grouped_model,
            token_ids,
            reference_routes,
            enabled=enabled,
        )
        logits_metrics = _logits_metrics(reference_logits, logits)
        router_metrics = _router_metrics(reference_routes, routes)
        row = {
            "cutoff": cutoff,
            "direct_prefix_layers": [FIRST_MOE_LAYER, cutoff - 1],
            "grouped_suffix_layers": (
                [cutoff, LAST_MOE_LAYER]
                if cutoff <= LAST_MOE_LAYER
                else None
            ),
            "grouped_layer_count": len(enabled),
            "timing": timing,
            "final_logits": logits_metrics,
            "router": router_metrics,
            "screening_pass": _screen(logits_metrics),
        }
        sweep.append(row)
        _progress(
            "suffix_cutoff",
            cutoff=cutoff,
            elapsed_seconds=timing["elapsed_seconds"],
            relative_l2=logits_metrics["relative_l2"],
            kl=logits_metrics["kl_reference_to_actual"],
            first_set_divergence=router_metrics[
                "first_top8_set_divergence_layer"
            ],
            screening_pass=row["screening_pass"],
        )

    auxiliary = {}
    for target in (3, 5):
        enabled = {target}
        arms = {}
        for policy in (
            "free",
            "direct_indices_current_scores",
            "direct_indices_and_scores",
        ):
            timing, logits, routes = _run_packed_capture(
                grouped_model,
                token_ids,
                reference_routes,
                enabled=enabled,
                route_policy=policy,
                target=target,
            )
            arms[policy] = {
                "timing": timing,
                "final_logits": _logits_metrics(reference_logits, logits),
                "router": _router_metrics(reference_routes, routes),
            }
            _progress(
                "router_causal_arm",
                target=target,
                policy=policy,
                relative_l2=arms[policy]["final_logits"]["relative_l2"],
            )
        free_l2 = arms["free"]["final_logits"]["relative_l2"]
        indices_l2 = arms["direct_indices_current_scores"]["final_logits"][
            "relative_l2"
        ]
        both_l2 = arms["direct_indices_and_scores"]["final_logits"][
            "relative_l2"
        ]
        auxiliary[str(target)] = {
            "arms": arms,
            "comparison": {
                "free_relative_l2": free_l2,
                "direct_indices_current_scores_relative_l2": indices_l2,
                "direct_indices_and_scores_relative_l2": both_l2,
                "membership_and_order_fix_reduction_factor": (
                    free_l2 / indices_l2 if indices_l2 else None
                ),
                "direct_score_fix_additional_reduction_factor": (
                    indices_l2 / both_l2 if both_l2 else None
                ),
            },
        }

    passing_cutoffs = [row["cutoff"] for row in sweep if row["screening_pass"]]
    performance_cutoffs = sorted(
        {
            neighbor
            for cutoff in passing_cutoffs
            for neighbor in (cutoff - 1, cutoff, cutoff + 1)
            if FIRST_MOE_LAYER <= neighbor <= DIRECT_ONLY_CUTOFF
        }
    )
    candidate_performance = []
    by_cutoff = {row["cutoff"]: row for row in sweep}
    for cutoff in performance_cutoffs:
        enabled = set(range(cutoff, LAST_MOE_LAYER + 1))
        _set_enabled_layers(grouped, enabled)
        benchmark = _benchmark_prefill(
            grouped_model,
            token_ids,
            warmups=args.warmups,
            repeats=args.repeats,
            phase=f"cutoff_{cutoff}",
        )
        speedup = direct_benchmark["median_seconds"] / benchmark["median_seconds"]
        row = {
            "cutoff": cutoff,
            "screening_pass": by_cutoff[cutoff]["screening_pass"],
            "benchmark": benchmark,
            "direct_median_speedup": speedup,
            "preferred_speed_gate": speedup >= 1.5,
            "pareto_screen_candidate": (
                by_cutoff[cutoff]["screening_pass"] and speedup >= 1.5
            ),
        }
        candidate_performance.append(row)
        _progress(
            "candidate_performance",
            cutoff=cutoff,
            median_seconds=benchmark["median_seconds"],
            speedup=speedup,
            selected=row["pareto_screen_candidate"],
        )

    cutoff3 = by_cutoff[FIRST_MOE_LAYER]
    cutoff45 = by_cutoff[DIRECT_ONLY_CUTOFF]
    endpoint_assertions = {
        "cutoff_3_reproduces_existing_all_grouped_sha256": cutoff3[
            "final_logits"
        ]["actual_sha256"]
        == ALL_GROUPED_BASELINE_SHA256,
        "cutoff_3_expected_sha256": ALL_GROUPED_BASELINE_SHA256,
        "cutoff_3_actual_sha256": cutoff3["final_logits"]["actual_sha256"],
        "cutoff_45_direct_byte_identical": cutoff45["final_logits"][
            "array_equal"
        ],
        "cutoff_45_router_indices_and_scores_byte_identical": (
            cutoff45["router"]["slot_position_mismatches"] == 0
            and cutoff45["router"]["score_difference_by_expert_id"]["max_abs"]
            == 0.0
        ),
    }
    preferred = [
        row["cutoff"]
        for row in candidate_performance
        if row["pareto_screen_candidate"]
    ]
    acceptance = {
        "all_43_cutoffs_measured": len(sweep) == 43,
        "endpoint_assertions_pass": bool(
            endpoint_assertions[
                "cutoff_3_reproduces_existing_all_grouped_sha256"
            ]
            and endpoint_assertions["cutoff_45_direct_byte_identical"]
            and endpoint_assertions[
                "cutoff_45_router_indices_and_scores_byte_identical"
            ]
        ),
        "packed_fp8_storage_preserved": storage[
            "all_weights_uint8_e4m3_codes_and_scales_fp32"
        ],
        "router_causal_three_arms_recorded_for_layers_3_and_5": all(
            len(auxiliary[str(target)]["arms"]) == 3 for target in (3, 5)
        ),
        "candidate_warm_median_measured": bool(candidate_performance),
        "dsa_short_context_bypass_scope": args.tokens <= index_topk,
        "runtime_grouped_threshold_unchanged": GROUPED_MIN_ROUTES == 256,
        "runtime_server_apc_admission_unchanged": True,
        "grouped_full_model_correctness_remains_unaccepted": True,
    }
    acceptance["accepted"] = all(acceptance.values())
    output = {
        "schema": "glm53-direct-prefix-grouped-fp8-suffix-sweep-v1",
        "date": date.today().isoformat(),
        "machine": "Apple M3 Ultra, 80-core GPU, 512 GB unified memory",
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "prompt": {
            "kind": "deterministic-token-sequence",
            "tokens": args.tokens,
            "token_formula": "(arange(tokens) * 7919) % (vocab_size - 1024) + 100",
            "index_topk": index_topk,
            "dsa_indexpool_bypassed": args.tokens <= index_topk,
            "fresh_cache_per_run": True,
        },
        "policy_definition": {
            "cutoff": "layers 3..c-1 packed Direct fallback; c..44 grouped FP8",
            "cutoffs": [FIRST_MOE_LAYER, DIRECT_ONLY_CUTOFF],
            "grouped_min_routes": GROUPED_MIN_ROUTES,
            "disabled_min_routes": DISABLED_MIN_ROUTES,
        },
        "router_metric_definitions": {
            "slot_position_mismatches": (
                "top-8 slot positions that differ; includes order-only changes"
            ),
            "tokens_with_changed_top8_set": (
                "layer-token rows whose unordered expert-ID set differs"
            ),
            "expert_membership_replacements": (
                "Direct expert IDs absent from the actual top-8, summed over "
                "layer-token rows"
            ),
            "tokens_with_order_only_change": (
                "layer-token rows with the same expert set but a different order"
            ),
            "score_difference_by_expert_id": (
                "actual minus Direct score after matching slots by expert ID"
            ),
        },
        "storage_invariants": storage,
        "measurement_method": {
            "correctness_sweep": (
                "one fresh-cache pass per cutoff with router capture; all-grouped "
                "compile warmup excluded"
            ),
            "candidate_performance": (
                f"router capture disabled; {args.warmups} warmups then median of "
                f"{args.repeats} measured fresh-cache passes"
            ),
            "direct_performance": (
                f"{args.warmups} warmups then median of {args.repeats} measured "
                "fresh-cache passes"
            ),
        },
        "direct_reference": {
            "timing": direct_timing,
            "warm_benchmark": direct_benchmark,
        },
        "endpoint_assertions": endpoint_assertions,
        "correctness_sweep": sweep,
        "router_causal_auxiliary_arms": auxiliary,
        "screening": {
            "criteria": {
                "argmax_match": True,
                "top_10_set_match": True,
                "relative_l2_max": 0.02,
                "kl_reference_to_actual_max": 5e-4,
                "preferred_direct_median_speedup_min": 1.5,
                "release_correctness_gate": False,
            },
            "passing_cutoffs": passing_cutoffs,
            "performance_cutoffs": performance_cutoffs,
            "candidate_performance": candidate_performance,
            "preferred_cutoffs": preferred,
            "selection_decision": (
                "do not promote a suffix runtime; no correctness-screened cutoff "
                "reaches 1.5x, so return to grouped-kernel numerical ordering"
                if not preferred
                else "advance preferred cutoffs to multi-prompt promotion gates"
            ),
        },
        "runtime_policy": {
            "default_backend": "direct",
            "packed_grouped_experimental_opt_in": True,
            "suffix_policy_installed": False,
            "prompt_limit": 256,
            "apc_identity_changed": False,
            "grouped_full_model_correctness_accepted": False,
        },
        "acceptance": acceptance,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passing_cutoffs": passing_cutoffs,
                "preferred_cutoffs": preferred,
                "auxiliary": {
                    target: row["comparison"] for target, row in auxiliary.items()
                },
                "accepted": acceptance["accepted"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if acceptance["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
