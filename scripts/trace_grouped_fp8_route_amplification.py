#!/usr/bin/env python3
"""Trace whether grouped-FP8 error is amplified by downstream MoE routing."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np

from glm53_flash_mlx.abi import GROUPED_MIN_ROUTES
from glm53_flash_mlx.grouped_fp8 import SortedGroupedFP8MoE
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint

DISABLED_MIN_ROUTES = 1 << 30
BOUNDARY_ORDER = (
    "layer_input",
    "attention_hc_collapse",
    "attention_output",
    "post_attention_hc_expand",
    "ffn_hc_collapse",
    "normalized_router_input",
    "routed_moe_output",
    "shared_expert_output",
    "moe_total_output",
    "post_ffn_hc_expand",
)


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), file=sys.stderr, flush=True)


def _deterministic_tokens(tokens: int, vocab: int) -> np.ndarray:
    return (
        (np.arange(tokens, dtype=np.uint64) * 7919) % (vocab - 1024) + 100
    ).astype(np.uint32)[None, :]


def _float_array(value) -> np.ndarray:
    value = value.astype(mx.float32)
    mx.eval(value)
    return np.ascontiguousarray(np.asarray(value), dtype=np.float32)


def _int_array(value) -> np.ndarray:
    mx.eval(value)
    return np.ascontiguousarray(np.asarray(value), dtype=np.int32)


def _sha256(values: np.ndarray) -> str:
    return hashlib.sha256(values.tobytes()).hexdigest()


class RecordedGate:
    """Return Direct-reference routes without inspecting the current input."""

    def __init__(self, indices: np.ndarray, scores: np.ndarray):
        self.indices = mx.array(indices, dtype=mx.int32)
        self.scores = mx.array(scores, dtype=mx.float32)

    def __call__(self, _):
        return self.indices, self.scores


class _CapturingGate:
    def __init__(self, delegate):
        self.delegate = delegate
        self.indices = None
        self.scores = None

    def __call__(self, x):
        self.indices, self.scores = self.delegate(x)
        return self.indices, self.scores


class _CapturingIndexer:
    def __init__(self, delegate):
        self.delegate = delegate
        self.output = "not-called"

    def __call__(self, *args, **kwargs):
        self.output = self.delegate(*args, **kwargs)
        return self.output


@dataclass
class TraceRun:
    final_logits: np.ndarray
    boundaries: dict[int, dict[str, np.ndarray]]
    routes: dict[int, dict[str, np.ndarray]]
    dsa_indexpool: dict[int, dict]
    elapsed_seconds: float


def _normal_prefill(model, token_ids: np.ndarray) -> np.ndarray:
    output = model(mx.array(token_ids), cache=model.make_cache())
    return _float_array(output.logits[0, -1])


def _run_traced_prefill(
    model,
    token_ids: np.ndarray,
    *,
    capture_layers: set[int],
) -> TraceRun:
    from mlx_vlm.models.glm5_next import language as glm
    from mlx_vlm.models.deepseek_v4.hyper_connection import hc_expand

    started = time.perf_counter()
    language = model.language_model
    stack = language.model
    cache = model.make_cache()
    h = stack.embed_tokens(mx.array(token_ids))
    fa_cache = cache[stack.fa_idx]
    fa_mask = glm.create_attention_mask(
        h, fa_cache[0] if fa_cache else None, return_array=True
    )
    ssm_mask = glm.create_ssm_mask(h, cache[stack.ssm_idx])
    h = mx.broadcast_to(
        h[:, :, None, :],
        (h.shape[0], h.shape[1], stack.hc_mult, h.shape[2]),
    )
    h = mx.contiguous(h)

    boundaries: dict[int, dict[str, np.ndarray]] = {}
    routes: dict[int, dict[str, np.ndarray]] = {}
    dsa_indexpool: dict[int, dict] = {}

    for layer_id, (layer, layer_cache) in enumerate(zip(stack.layers, cache)):
        capture = layer_id in capture_layers
        row = boundaries.setdefault(layer_id, {}) if capture else None
        if capture:
            row["layer_input"] = _float_array(h)

        residual = h
        attention_x, attention_post, attention_comb = layer.attn_hc(h)
        if capture:
            row["attention_hc_collapse"] = _float_array(attention_x)
        attention_input = layer.input_layernorm(attention_x)
        mask = ssm_mask if layer.is_linear else fa_mask

        indexer_capture = None
        if not layer.is_linear:
            original_indexer = layer.self_attn.indexer
            indexer_capture = _CapturingIndexer(original_indexer)
            layer.self_attn.indexer = indexer_capture
        try:
            attention_output = layer.self_attn(
                attention_input, mask=mask, cache=layer_cache
            )
        finally:
            if indexer_capture is not None:
                layer.self_attn.indexer = original_indexer
        if indexer_capture is not None:
            dsa_indexpool[layer_id] = {
                "bypassed": indexer_capture.output is None,
                "query_tokens": int(token_ids.size),
                "index_topk": int(original_indexer.index_topk),
            }
        if capture:
            row["attention_output"] = _float_array(attention_output)
        h = hc_expand(
            attention_output, residual, attention_post, attention_comb
        )
        if capture:
            row["post_attention_hc_expand"] = _float_array(h)

        residual = h
        ffn_x, ffn_post, ffn_comb = layer.ffn_hc(h)
        if capture:
            row["ffn_hc_collapse"] = _float_array(ffn_x)
        router_input = layer.post_attention_layernorm(ffn_x)
        if capture:
            row["normalized_router_input"] = _float_array(router_input)

        mlp = layer.mlp
        if hasattr(mlp, "gate"):
            original_gate = mlp.gate
            capturing_gate = _CapturingGate(original_gate)
            original_shared = mlp.shared_experts
            mlp.gate = capturing_gate
            mlp.shared_experts = None
            try:
                routed_output = mlp(router_input)
            finally:
                mlp.gate = original_gate
                mlp.shared_experts = original_shared
            shared_output = original_shared(router_input)
            moe_output = routed_output + shared_output
            routes[layer_id] = {
                "indices": _int_array(capturing_gate.indices),
                "scores": _float_array(capturing_gate.scores),
            }
            if capture:
                row["routed_moe_output"] = _float_array(routed_output)
                row["shared_expert_output"] = _float_array(shared_output)
                row["moe_total_output"] = _float_array(moe_output)
        else:
            moe_output = mlp(router_input)

        h = hc_expand(moe_output, residual, ffn_post, ffn_comb)
        if capture:
            row["post_ffn_hc_expand"] = _float_array(h)
        _progress("trace_layer", layer=layer_id)

    h = stack.norm(h.mean(axis=2))
    logits = (
        language.model.embed_tokens.as_linear(h)
        if language.args.tie_word_embeddings
        else language.lm_head(h)
    )
    final_logits = _float_array(logits[0, -1])
    return TraceRun(
        final_logits=final_logits,
        boundaries=boundaries,
        routes=routes,
        dsa_indexpool=dsa_indexpool,
        elapsed_seconds=time.perf_counter() - started,
    )


def _tensor_metrics(reference: np.ndarray, actual: np.ndarray) -> dict:
    reference64 = reference.astype(np.float64)
    actual64 = actual.astype(np.float64)
    diff = actual64 - reference64
    denominator = np.linalg.norm(reference64)
    return {
        "array_equal": bool(np.array_equal(reference, actual)),
        "reference_sha256": _sha256(reference),
        "actual_sha256": _sha256(actual),
        "relative_l2": float(np.linalg.norm(diff) / denominator) if denominator else 0.0,
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rms": float(np.sqrt(np.mean(diff * diff))),
    }


def _top_k(values: np.ndarray, k: int) -> np.ndarray:
    candidates = np.argpartition(values, -k)[-k:]
    return candidates[np.argsort(values[candidates], kind="stable")][::-1]


def _logits_metrics(reference: np.ndarray, actual: np.ndarray, *, top_k=10) -> dict:
    metrics = _tensor_metrics(reference, actual)
    reference64 = reference.astype(np.float64)
    actual64 = actual.astype(np.float64)
    reference_shifted = reference64 - np.max(reference64)
    actual_shifted = actual64 - np.max(actual64)
    reference_log_p = reference_shifted - np.log(np.exp(reference_shifted).sum())
    actual_log_p = actual_shifted - np.log(np.exp(actual_shifted).sum())
    reference_p = np.exp(reference_log_p)
    reference_top = _top_k(reference, top_k)
    actual_top = _top_k(actual, top_k)
    metrics.update(
        {
            "kl_reference_to_actual": float(
                np.sum(reference_p * (reference_log_p - actual_log_p))
            ),
            "argmax_match": int(reference_top[0]) == int(actual_top[0]),
            "reference_top_k": reference_top.tolist(),
            "actual_top_k": actual_top.tolist(),
            "top_k_order_match": bool(np.array_equal(reference_top, actual_top)),
            "top_k_set_match": set(reference_top.tolist())
            == set(actual_top.tolist()),
            "top_k_overlap": len(
                set(reference_top.tolist()) & set(actual_top.tolist())
            ),
        }
    )
    return metrics


def _route_metrics(reference: dict, actual: dict) -> dict:
    ref_indices = reference["indices"]
    actual_indices = actual["indices"]
    ref_scores = reference["scores"]
    actual_scores = actual["scores"]
    same_slots = ref_indices == actual_indices
    ref_sets = np.sort(ref_indices, axis=-1)
    actual_sets = np.sort(actual_indices, axis=-1)
    identical_sets = np.all(ref_sets == actual_sets, axis=-1)
    identical_order = np.all(same_slots, axis=-1)
    flat_reference = ref_indices.reshape(-1, ref_indices.shape[-1])
    flat_actual = actual_indices.reshape(-1, actual_indices.shape[-1])
    membership_replacements = sum(
        len(set(reference_row.tolist()) - set(actual_row.tolist()))
        for reference_row, actual_row in zip(
            flat_reference, flat_actual, strict=True
        )
    )
    score_diff = actual_scores.astype(np.float64) - ref_scores.astype(np.float64)
    return {
        "slot_agreement": float(np.mean(same_slots)),
        "tokens_with_identical_top8_set": int(np.count_nonzero(identical_sets)),
        "tokens_with_changed_top8_set": int(np.count_nonzero(~identical_sets)),
        "tokens_with_order_only_change": int(
            np.count_nonzero(identical_sets & ~identical_order)
        ),
        "expert_membership_replacements": int(membership_replacements),
        "total_tokens": int(identical_sets.size),
        "changed_route_slots": int(np.count_nonzero(~same_slots)),
        "indices_array_equal": bool(np.array_equal(ref_indices, actual_indices)),
        "scores_array_equal": bool(np.array_equal(ref_scores, actual_scores)),
        "score_max_abs": float(np.max(np.abs(score_diff))),
        "score_rms": float(np.sqrt(np.mean(score_diff * score_diff))),
    }


def _first_route_divergence(route_rows: dict[int, dict], target: int):
    return next(
        (
            layer_id
            for layer_id in sorted(route_rows)
            if layer_id > target and route_rows[layer_id]["changed_route_slots"] > 0
        ),
        None,
    )


def _first_route_divergence_context(
    boundary_rows: dict[str, dict],
    route_rows: dict[int, dict],
    target: int,
):
    layer = _first_route_divergence(route_rows, target)
    if layer is None:
        return None
    return {
        "layer": layer,
        "precursor_boundaries": {
            name: boundary_rows[str(layer)][name]
            for name in (
                "layer_input",
                "attention_hc_collapse",
                "attention_output",
                "post_attention_hc_expand",
                "ffn_hc_collapse",
                "normalized_router_input",
            )
        },
        "router": route_rows[layer],
    }


def _first_amplification_boundary(boundaries: dict, target: int):
    local = boundaries[target]["routed_moe_output"]["relative_l2"]
    threshold = max(local * 2.0, local + 1e-6)
    passed_local = False
    for layer_id in sorted(boundaries):
        for name in BOUNDARY_ORDER:
            if name not in boundaries[layer_id]:
                continue
            if layer_id == target and name == "routed_moe_output":
                passed_local = True
                continue
            if not passed_local:
                continue
            value = boundaries[layer_id][name]["relative_l2"]
            if value > threshold:
                return {
                    "layer": layer_id,
                    "boundary": name,
                    "relative_l2": value,
                    "local_routed_relative_l2": local,
                    "amplification_factor": value / local if local else None,
                    "criterion": "relative_l2 > 2x target routed-MoE local error",
                }
    return None


def _compare_trace(
    reference: TraceRun,
    actual: TraceRun,
    *,
    target: int,
) -> dict:
    boundary_rows = {
        str(layer_id): {
            name: _tensor_metrics(reference.boundaries[layer_id][name], value)
            for name, value in actual.boundaries[layer_id].items()
        }
        for layer_id in sorted(actual.boundaries)
    }
    route_rows = {
        layer_id: _route_metrics(reference.routes[layer_id], actual.routes[layer_id])
        for layer_id in sorted(actual.routes)
        if layer_id >= target
    }
    route_rows_json = {str(layer): row for layer, row in route_rows.items()}
    return {
        "elapsed_seconds": actual.elapsed_seconds,
        "boundary_metrics": boundary_rows,
        "router_metrics": route_rows_json,
        "target_layer_input_byte_identical": boundary_rows[str(target)][
            "layer_input"
        ]["array_equal"],
        "target_router_indices_equal": route_rows[target]["indices_array_equal"],
        "target_router_scores_equal": route_rows[target]["scores_array_equal"],
        "target_grouped_routed_moe_local_error": boundary_rows[str(target)][
            "routed_moe_output"
        ],
        "first_hidden_state_amplification": _first_amplification_boundary(
            {int(layer): row for layer, row in boundary_rows.items()}, target
        ),
        "first_route_divergence_layer": _first_route_divergence(
            route_rows, target
        ),
        "first_route_divergence_context": _first_route_divergence_context(
            boundary_rows, route_rows, target
        ),
        "final_logits": _logits_metrics(
            reference.final_logits, actual.final_logits
        ),
        "dsa_indexpool": {
            str(layer): row
            for layer, row in actual.dsa_indexpool.items()
            if layer in actual.boundaries
        },
    }


def _grouped_modules(model) -> dict[int, SortedGroupedFP8MoE]:
    return {
        layer_id: layer.mlp
        for layer_id, layer in enumerate(model.language_model.model.layers)
        if isinstance(layer.mlp, SortedGroupedFP8MoE)
    }


def _run_grouped_variant(
    model,
    token_ids: np.ndarray,
    reference: TraceRun,
    *,
    target: int,
    trace_end: int,
    fixed_routes: bool,
) -> TraceRun:
    grouped = _grouped_modules(model)
    original_gates = {layer: moe.gate for layer, moe in grouped.items()}
    original_thresholds = {layer: moe.min_routes for layer, moe in grouped.items()}
    try:
        for layer_id, moe in grouped.items():
            moe.min_routes = (
                GROUPED_MIN_ROUTES if layer_id == target else DISABLED_MIN_ROUTES
            )
            if fixed_routes and layer_id > target:
                moe.gate = RecordedGate(
                    reference.routes[layer_id]["indices"],
                    reference.routes[layer_id]["scores"],
                )
        return _run_traced_prefill(
            model,
            token_ids,
            capture_layers=set(range(target, trace_end + 1)),
        )
    finally:
        for layer_id, moe in grouped.items():
            moe.gate = original_gates[layer_id]
            moe.min_routes = original_thresholds[layer_id]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    args = parser.parse_args()

    report = inspect_checkpoint(args.model, require_server_ready=True)
    raw_config = json.loads((Path(args.model) / "config.json").read_text())
    vocab = int(raw_config["text_config"]["vocab_size"])
    token_ids = _deterministic_tokens(args.tokens, vocab)
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))

    _progress("load_direct")
    direct_model, _ = load(args.model)
    warm_residency(direct_model)
    normal_logits = _normal_prefill(direct_model, token_ids)
    reference = _run_traced_prefill(
        direct_model, token_ids, capture_layers=set(range(3, 13))
    )
    direct_manual_parity = _tensor_metrics(normal_logits, reference.final_logits)
    if not direct_manual_parity["array_equal"]:
        raise RuntimeError("manual traced Direct path differs from normal model forward")
    del direct_model, normal_logits
    gc.collect()
    mx.clear_cache()
    mx.synchronize()

    _progress("load_packed_grouped")
    grouped_model, _ = load(args.model, experimental_packed_grouped_moe=True)
    warm_residency(grouped_model)

    targets = {}
    specifications = ((3, 8), (5, 12))
    for target, trace_end in specifications:
        _progress("target_free", target=target)
        free_run = _run_grouped_variant(
            grouped_model,
            token_ids,
            reference,
            target=target,
            trace_end=trace_end,
            fixed_routes=False,
        )
        free = _compare_trace(reference, free_run, target=target)
        del free_run
        gc.collect()
        mx.clear_cache()

        _progress("target_fixed", target=target)
        fixed_run = _run_grouped_variant(
            grouped_model,
            token_ids,
            reference,
            target=target,
            trace_end=trace_end,
            fixed_routes=True,
        )
        fixed = _compare_trace(reference, fixed_run, target=target)
        del fixed_run
        gc.collect()
        mx.clear_cache()

        free_l2 = free["final_logits"]["relative_l2"]
        fixed_l2 = fixed["final_logits"]["relative_l2"]
        targets[str(target)] = {
            "trace_layers": [target, trace_end],
            "free_routing": free,
            "direct_indices_and_scores_fixed": fixed,
            "causal_comparison": {
                "free_final_relative_l2": free_l2,
                "fixed_final_relative_l2": fixed_l2,
                "relative_l2_reduction_factor": (
                    free_l2 / fixed_l2 if fixed_l2 else None
                ),
                "fixed_indices_and_scores_substantially_reduce_error": (
                    fixed_l2 <= free_l2 * 0.5
                ),
                "interpretation": (
                    "downstream router selection and mixture weighting are a "
                    "major combined amplifier"
                    if fixed_l2 <= free_l2 * 0.5
                    else "continuous hidden-state amplification remains material"
                ),
            },
        }

    patterns = [
        targets[str(target)]["causal_comparison"][
            "fixed_indices_and_scores_substantially_reduce_error"
        ]
        for target, _ in specifications
    ]
    same_pattern = patterns[0] == patterns[1]
    acceptance = {
        "direct_manual_trace_matches_normal_forward": direct_manual_parity[
            "array_equal"
        ],
        "target_layer_inputs_byte_identical": all(
            targets[str(target)][mode]["target_layer_input_byte_identical"]
            for target, _ in specifications
            for mode in ("free_routing", "direct_indices_and_scores_fixed")
        ),
        "target_router_indices_and_scores_identical": all(
            targets[str(target)][mode]["target_router_indices_equal"]
            and targets[str(target)][mode]["target_router_scores_equal"]
            for target, _ in specifications
            for mode in ("free_routing", "direct_indices_and_scores_fixed")
        ),
        "grouped_local_error_recorded": all(
            targets[str(target)]["free_routing"][
                "target_grouped_routed_moe_local_error"
            ]["relative_l2"]
            > 0
            for target, _ in specifications
        ),
        "first_hidden_amplification_boundary_identified": all(
            targets[str(target)]["free_routing"][
                "first_hidden_state_amplification"
            ]
            is not None
            for target, _ in specifications
        ),
        "first_free_router_divergence_identified": all(
            targets[str(target)]["free_routing"]["first_route_divergence_layer"]
            is not None
            for target, _ in specifications
        ),
        "fixed_route_final_metrics_recorded": all(
            "kl_reference_to_actual"
            in targets[str(target)]["direct_indices_and_scores_fixed"]["final_logits"]
            for target, _ in specifications
        ),
        "layer_3_and_5_same_causal_pattern": same_pattern,
        "fixed_indices_and_scores_substantially_reduce_error_both": all(patterns),
        "all_traced_dsa_layers_bypass_indexpool": all(
            row["bypassed"]
            for target, _ in specifications
            for mode in ("free_routing", "direct_indices_and_scores_fixed")
            for row in targets[str(target)][mode]["dsa_indexpool"].values()
        ),
        "runtime_policy_unchanged": GROUPED_MIN_ROUTES == 256,
        "grouped_full_model_correctness_remains_unaccepted": True,
    }
    acceptance["accepted"] = all(acceptance.values())
    output = {
        "schema": "glm53-grouped-fp8-route-amplification-v1",
        "date": date.today().isoformat(),
        "machine": "Apple M3 Ultra, 80-core GPU, 512 GB unified memory",
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "prompt": {
            "kind": "deterministic-token-sequence",
            "tokens": args.tokens,
            "token_formula": "(arange(tokens) * 7919) % (vocab_size - 1024) + 100",
        },
        "direct_reference": {
            "manual_trace_normal_forward_parity": direct_manual_parity,
            "elapsed_seconds": reference.elapsed_seconds,
            "final_logits_sha256": _sha256(reference.final_logits),
        },
        "targets": targets,
        "causal_pattern": {
            "layer_3_and_5_same_pattern": same_pattern,
            "layer_3_router_selection_and_weighting_major": patterns[0],
            "layer_5_router_selection_and_weighting_major": patterns[1],
        },
        "runtime_policy": {
            "default_backend": "direct",
            "packed_grouped_experimental_opt_in": True,
            "grouped_min_routes": GROUPED_MIN_ROUTES,
            "prompt_limit": 256,
            "grouped_full_model_correctness_accepted": False,
            "suffix_backend_policy_changed": False,
            "shared_expert_fusion_changed": False,
        },
        "acceptance": acceptance,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "layer_3": targets["3"]["causal_comparison"],
                "layer_5": targets["5"]["causal_comparison"],
                "accepted": acceptance["accepted"],
            },
            indent=2,
        )
    )
    return 0 if acceptance["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
