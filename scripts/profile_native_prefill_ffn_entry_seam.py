#!/usr/bin/env python3
"""Attribute the Q256 post-attention state-to-MoE-route prefill seam."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "native_execution", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from glm53_flash_mlx.loader import load
from mlx_vlm.models.deepseek_v32 import language as dsv32


DEFAULT_OUTPUT = ROOT / "bench-results" / (
    "m3ultra512-native-prefill-ffn-entry-seam-profile-20260909.json"
)
MIN_STANDALONE_WALL_HEADROOM_MS = 0.75


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _state() -> mx.array:
    value = mx.cos(
        mx.arange(256 * 4 * 4096, dtype=mx.float32).reshape(1, 256, 4, 4096)
        * 0.000031
    ).astype(mx.bfloat16)
    value = mx.contiguous(value)
    mx.eval(value)
    return value


def _select(gate, logits):
    return dsv32.group_expert_select(
        logits,
        gate.e_score_correction_bias,
        gate.top_k,
        gate.n_group,
        gate.topk_group,
        gate.routed_scaling_factor,
        gate.norm_topk_prob,
    )


def _eval(*values):
    mx.eval(*values)
    mx.synchronize()
    return values


def _timed(operation, samples):
    operation()
    values = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        operation()
        values.append((time.perf_counter_ns() - started) / 1e6)
    return {"median_wall_ms": statistics.median(values), "samples_ms": values}


def _full(layer, state):
    collapsed, post, comb = layer.ffn_hc(state)
    normalized = layer.post_attention_layernorm(collapsed)
    gate = layer.mlp.gate
    logits = normalized.astype(mx.float32) @ gate.weight.astype(mx.float32).T
    indices, scores = _select(gate, logits)
    _eval(collapsed, post, comb, normalized, logits, indices, scores)
    return collapsed, post, comb, normalized, logits, indices, scores


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    print(json.dumps({"phase": "load_model"}), flush=True)
    model, _ = load(args.model, experimental_packed_grouped_moe=True)
    layer = model.language_model.model.layers[args.layer]
    state = _state()
    gate = layer.mlp.gate
    mx.eval(
        layer.ffn_hc.fn,
        layer.ffn_hc.scale,
        layer.ffn_hc.base,
        layer.post_attention_layernorm.weight,
        gate.weight,
        gate.e_score_correction_bias,
    )

    print(json.dumps({"phase": "materialize_stage_anchors"}), flush=True)
    collapsed, post, comb, normalized, logits, indices, scores = _full(layer, state)
    anchor_bits = {
        "collapsed": np.asarray(collapsed.view(mx.uint16), dtype=np.uint16).copy(),
        "normalized": np.asarray(normalized.view(mx.uint16), dtype=np.uint16).copy(),
        "logits": np.asarray(logits.view(mx.uint32), dtype=np.uint32).copy(),
        "indices": np.asarray(indices, dtype=np.uint32).copy(),
        "scores": np.asarray(scores.view(mx.uint32), dtype=np.uint32).copy(),
    }

    print(json.dumps({"phase": "time_full_boundary"}), flush=True)
    full = _timed(lambda: _full(layer, state), args.samples)
    print(json.dumps({"phase": "time_ffn_hc"}), flush=True)
    hc = _timed(lambda: _eval(*layer.ffn_hc(state)), args.samples)
    print(json.dumps({"phase": "time_post_attention_norm"}), flush=True)
    norm = _timed(
        lambda: _eval(layer.post_attention_layernorm(collapsed)), args.samples
    )
    print(json.dumps({"phase": "time_router_logits"}), flush=True)
    router_logits = _timed(
        lambda: _eval(
            normalized.astype(mx.float32) @ gate.weight.astype(mx.float32).T
        ),
        args.samples,
    )
    print(json.dumps({"phase": "time_router_selection"}), flush=True)
    selection = _timed(lambda: _eval(*_select(gate, logits)), args.samples)

    print(json.dumps({"phase": "repeat_exact"}), flush=True)
    repeat = _full(layer, state)
    repeat_exact = {
        "collapsed": np.array_equal(
            anchor_bits["collapsed"],
            np.asarray(repeat[0].view(mx.uint16), dtype=np.uint16),
        ),
        "normalized": np.array_equal(
            anchor_bits["normalized"],
            np.asarray(repeat[3].view(mx.uint16), dtype=np.uint16),
        ),
        "logits": np.array_equal(
            anchor_bits["logits"],
            np.asarray(repeat[4].view(mx.uint32), dtype=np.uint32),
        ),
        "indices": np.array_equal(
            anchor_bits["indices"], np.asarray(repeat[5], dtype=np.uint32)
        ),
        "scores": np.array_equal(
            anchor_bits["scores"],
            np.asarray(repeat[6].view(mx.uint32), dtype=np.uint32),
        ),
    }
    component_sum = sum(
        row["median_wall_ms"] for row in (hc, norm, router_logits, selection)
    )
    checks = {
        "all_stage_anchors_repeat_byte_exact": all(repeat_exact.values()),
        "official_geometry_is_q256_hc4_hidden4096_experts288_top8": (
            tuple(state.shape) == (1, 256, 4, 4096)
            and tuple(gate.weight.shape) == (288, 4096)
            and gate.top_k == 8
        ),
        "profile_has_no_nan": (
            bool(mx.all(mx.isfinite(collapsed)).item())
            and bool(mx.all(mx.isfinite(normalized)).item())
            and bool(mx.all(mx.isfinite(logits)).item())
            and bool(mx.all(mx.isfinite(scores)).item())
        ),
    }
    accepted = all(checks.values())
    stages = {
        "ffn_hc_collapse": hc,
        "post_attention_norm": norm,
        "router_logits": router_logits,
        "router_selection": selection,
    }
    largest = max(stages, key=lambda name: stages[name]["median_wall_ms"])
    artifact = {
        "schema": "glm53-native-prefill-ffn-entry-seam-profile-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "stop_standalone_ffn_entry_native_island_and_move_to_cross_layer_plan"
            if accepted
            else "stop_invalid_ffn_entry_profile"
        ),
        "layer": args.layer,
        "geometry": {
            "query_rows": 256,
            "hc_branches": 4,
            "hidden_size": 4096,
            "experts": 288,
            "top_k": 8,
        },
        "full_boundary": full,
        "maximum_standalone_wall_headroom_ms": full["median_wall_ms"],
        "standalone_native_implementation_warranted": (
            full["median_wall_ms"] >= MIN_STANDALONE_WALL_HEADROOM_MS
        ),
        "synchronized_stages": stages,
        "synchronized_component_sum_ms": component_sum,
        "largest_synchronized_stage": largest,
        "repeat_exact": repeat_exact,
        "checks": checks,
        "profiling_only": True,
        "runtime_changes": False,
    }
    _write(args.output, artifact)
    print(json.dumps({
        "output": str(args.output),
        "complete": True,
        "accepted": accepted,
        "decision": artifact["decision"],
        "full_boundary_ms": full["median_wall_ms"],
        "largest_synchronized_stage": largest,
        "failed_gates": [name for name, passed in checks.items() if not passed],
    }))
    del model, layer, state, collapsed, post, comb, normalized, logits
    del indices, scores, repeat
    mx.clear_cache()
    gc.collect()
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
