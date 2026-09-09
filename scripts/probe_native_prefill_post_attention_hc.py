#!/usr/bin/env python3
"""Qualify the actual-geometry native DSA output through post-attention HC."""

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

from glm53_native_execution import NativePrefillDSAOutputPlan
from glm53_flash_mlx.fp8 import BlockFP8Linear
from glm53_flash_mlx.loader import load
from mlx_vlm.models.deepseek_v4.hyper_connection import hc_expand


DEFAULT_OUTPUT = ROOT / "bench-results" / (
    "m3ultra512-native-prefill-post-attention-hc-20260909.json"
)
MIN_SPEEDUP = 1.20


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _head_major() -> mx.array:
    value = mx.sin(
        mx.arange(64 * 256 * 256, dtype=mx.float32).reshape(64, 256, 256)
        * 0.000019
    ).astype(mx.bfloat16)
    value = mx.contiguous(value)
    mx.eval(value)
    return value


def _residual() -> mx.array:
    value = mx.cos(
        mx.arange(256 * 4 * 4096, dtype=mx.float32).reshape(1, 256, 4, 4096)
        * 0.000031
    ).astype(mx.bfloat16)
    value = mx.contiguous(value)
    mx.eval(value)
    return value


def _reference(projection, head_major, residual, post, comb):
    row_major = mx.contiguous(
        head_major.transpose(1, 0, 2).reshape(1, 256, 16_384),
        allow_col_major=False,
    )
    projected = projection(row_major)
    output = hc_expand(projected, residual, post, comb)
    mx.eval(row_major, projected, output)
    mx.synchronize()
    return row_major, projected, output


def _native(plan, projection, head_major, residual, post, comb):
    output = plan.execute_hc(
        head_major,
        projection.weight,
        projection.weight_scale_inv,
        residual,
        post,
        comb,
    )
    mx.eval(output)
    mx.synchronize()
    return output


def _bits(value):
    return np.asarray(value.view(mx.uint16), dtype=np.uint16)


def _timed(operation, samples):
    operation()
    values = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        operation()
        values.append((time.perf_counter_ns() - started) / 1e6)
    return {"median_wall_ms": statistics.median(values), "samples_ms": values}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    print(json.dumps({"phase": "load_model"}), flush=True)
    model, _ = load(args.model, experimental_packed_grouped_moe=True)
    layer = model.language_model.model.layers[args.layer]
    projection = layer.self_attn.o_proj
    if not isinstance(projection, BlockFP8Linear):
        raise RuntimeError("official DSA o_proj is not block-FP8")
    if tuple(projection.weight.shape) != (4096, 16_384):
        raise RuntimeError("unexpected official DSA o_proj geometry")

    head_major = _head_major()
    residual = _residual()
    _, post, comb = layer.attn_hc(residual)
    post = mx.contiguous(post)
    comb = mx.contiguous(comb)
    mx.eval(projection.weight, projection.weight_scale_inv, post, comb)
    plan = NativePrefillDSAOutputPlan()
    identities = list(plan.buffer_identities)

    print(json.dumps({"phase": "exact_reference"}), flush=True)
    expected_row_major, expected_projection, expected = _reference(
        projection, head_major, residual, post, comb
    )
    expected_row_bits = _bits(expected_row_major).copy()
    expected_projection_bits = _bits(expected_projection).copy()
    expected_bits = _bits(expected).copy()

    print(json.dumps({"phase": "projection_anchor"}), flush=True)
    projected = plan.execute(
        head_major, projection.weight, projection.weight_scale_inv
    )
    mx.eval(projected)
    mx.synchronize()
    projected_bits = _bits(projected).copy()
    row_bits = _bits(plan.debug_row_major).copy()

    print(json.dumps({"phase": "exact_native_hc"}), flush=True)
    actual = _native(plan, projection, head_major, residual, post, comb)
    actual_bits = _bits(actual).copy()
    repeat = _native(plan, projection, head_major, residual, post, comb)
    repeat_bits = _bits(repeat).copy()

    row_exact = np.array_equal(expected_row_bits.reshape(row_bits.shape), row_bits)
    projection_exact = np.array_equal(
        expected_projection_bits.reshape(projected_bits.shape), projected_bits
    )
    hc_exact = np.array_equal(expected_bits.reshape(actual_bits.shape), actual_bits)
    repeat_exact = np.array_equal(actual_bits, repeat_bits)

    print(json.dumps({"phase": "timing"}), flush=True)
    reference_timing = _timed(
        lambda: _reference(projection, head_major, residual, post, comb)[2],
        args.samples,
    )
    native_timing = _timed(
        lambda: _native(plan, projection, head_major, residual, post, comb),
        args.samples,
    )
    speedup = reference_timing["median_wall_ms"] / native_timing["median_wall_ms"]
    checks = {
        "head_to_row_layout_is_byte_exact": row_exact,
        "official_fp8_o_projection_anchor_is_byte_exact": projection_exact,
        "post_attention_hc_output_is_byte_exact": hc_exact,
        "repeat_is_byte_exact": repeat_exact,
        "single_fixed_native_execution_scope": (
            identities == list(plan.buffer_identities)
            and plan.dynamic_allocation_count == 0
            and plan.graph_node_count == 0
            and plan.shape_discovery_count == 0
            and plan.host_synchronization_count == 0
            and plan.returned_intermediate_tensor_bytes == 0
        ),
        "post_attention_hc_speedup_at_least_1_20x": speedup >= MIN_SPEEDUP,
    }
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-native-prefill-post-attention-hc-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "advance_post_attention_hc_into_ffn_collapse_norm_router"
            if accepted
            else "localize_or_stop_native_post_attention_hc"
        ),
        "layer": args.layer,
        "geometry": {
            "query_rows": 256,
            "heads": 64,
            "value_dim": 256,
            "hidden_size": 4096,
            "hc_branches": 4,
        },
        "reference_timing": reference_timing,
        "native_timing": native_timing,
        "speedup": speedup,
        "scratch_bytes": plan.hc_scratch_bytes,
        "checks": checks,
        "probe_only": True,
        "runtime_changes": False,
    }
    _write(args.output, artifact)
    print(json.dumps({
        "output": str(args.output),
        "complete": True,
        "accepted": accepted,
        "decision": artifact["decision"],
        "speedup": speedup,
        "failed_gates": [name for name, passed in checks.items() if not passed],
    }))
    del model, layer, projection, plan, expected_row_major, expected_projection
    del expected, projected, actual, repeat, residual, post, comb, head_major
    mx.clear_cache()
    gc.collect()
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
