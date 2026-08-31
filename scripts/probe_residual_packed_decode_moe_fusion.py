#!/usr/bin/env python3
"""Probe exact residual packed-decode MoE fusion opportunities."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import probe_fused_packed_gate_up_swiglu_decode as d99f
import probe_long_context_first_decode_boundary as boundary
import probe_packed_decode_runtime as packed_probe

from glm53_flash_mlx.abi import MLX_VLM_REVISION, PACKED_DECODE_KERNEL_ABI
from glm53_flash_mlx.fp8 import (
    BLOCK_SIZE,
    DECODE_TOP_K,
    THREADS,
    _FP8_LUT_HEADER,
    _metal_input,
)
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint
from glm53_flash_mlx.packed import PackedFP8MoE

REPRESENTATIVE_LAYERS = (3, 5)
OPERATOR_WARMUPS = 4
OPERATOR_SAMPLES = 20
FRONTIER_CONTEXT = 2049
DECODE_STEPS = 4096
TARGET_DECODE_TPS = 15.0
TARGET_DECODE_MS = 1000.0 / TARGET_DECODE_TPS


def _stable_silu_source(gate_name: str, up_name: str) -> str:
    return f"""
            T gate_activation = T({gate_name});
            T up_activation = T({up_name});
            auto sigmoid_tail = 1 / (
                1 + metal::exp(metal::abs(gate_activation))
            );
            T sigmoid_value = (gate_activation < 0)
                ? sigmoid_tail
                : 1 - sigmoid_tail;
            T silu_value = gate_activation * sigmoid_value;
            T activated = silu_value * up_activation;
    """


def _shared_source(*, diagnostics: bool) -> str:
    diagnostic_write = (
        "gate_output[out_row] = gate_t;\n"
        "            up_output[out_row] = up_t;"
        if diagnostics
        else ""
    )
    return rf"""
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd_id = simdgroup_index_in_threadgroup;
    uint out_row = threadgroup_position_in_grid.x;
    if (out_row >= OUT_FEATURES) return;

    const device uint8_t* gate_wr = gate_weight
        + size_t(out_row) * IN_FEATURES;
    const device uint8_t* up_wr = up_weight
        + size_t(out_row) * IN_FEATURES;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint k = tid; k < IN_FEATURES; k += THREADS) {{
        size_t scale_offset = size_t(out_row / BLOCK_SIZE) * SCALE_COLS
            + k / BLOCK_SIZE;
        gate_acc += float(x[k]) * glm53_fp8_lut[gate_wr[k]]
            * gate_scale_inv[scale_offset];
        up_acc += float(x[k]) * glm53_fp8_lut[up_wr[k]]
            * up_scale_inv[scale_offset];
    }}
    gate_acc = simd_sum(gate_acc);
    up_acc = simd_sum(up_acc);
    constexpr uint NSIMD = THREADS / 32;
    threadgroup float gate_partial[NSIMD];
    threadgroup float up_partial[NSIMD];
    if (lane == 0) {{
        gate_partial[simd_id] = gate_acc;
        up_partial[simd_id] = up_acc;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {{
        float gate_total = lane < NSIMD ? gate_partial[lane] : 0.0f;
        float up_total = lane < NSIMD ? up_partial[lane] : 0.0f;
        gate_total = simd_sum(gate_total);
        up_total = simd_sum(up_total);
        if (lane == 0) {{
            T gate_t = T(gate_total);
            T up_t = T(up_total);
            constexpr float LIMIT_F = float(LIMIT);
            float gate_value = min(float(gate_t), LIMIT_F);
            float up_value = clamp(float(up_t), -LIMIT_F, LIMIT_F);
            {_stable_silu_source("gate_value", "up_value")}
            hidden[out_row] = activated;
            {diagnostic_write}
        }}
    }}
    """


def _aggregation_source(*, diagnostics: bool) -> str:
    diagnostic_write = (
        "weighted[size_t(selected) * OUT_FEATURES + out_row] = contribution;"
        if diagnostics
        else ""
    )
    reduced_write = "reduced[out_row] = total;" if diagnostics else ""
    return rf"""
    uint out_row = thread_position_in_grid.x;
    if (out_row >= OUT_FEATURES) return;
    float total = 0.0f;
    for (uint selected = 0; selected < TOP_K; ++selected) {{
        float contribution = float(
            down[size_t(selected) * OUT_FEATURES + out_row]
        ) * float(scores[selected]);
        {diagnostic_write}
        total += contribution;
    }}
    {reduced_write}
    output[out_row] = T(total);
    """


_WEIGHTED_DOWN_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd_id = simdgroup_index_in_threadgroup;
    uint group_id = threadgroup_position_in_grid.x;
    uint selected = group_id / OUT_FEATURES;
    uint out_row = group_id % OUT_FEATURES;
    if (selected >= TOP_K) return;

    uint expert = expert_ids[selected];
    const device uint8_t* wr = weight
        + (size_t(expert) * OUT_FEATURES + out_row) * IN_FEATURES;
    const device T* xr = hidden + size_t(selected) * IN_FEATURES;
    float acc = 0.0f;
    for (uint k = tid; k < IN_FEATURES; k += THREADS) {
        size_t scale_offset =
            (size_t(expert) * SCALE_ROWS + out_row / BLOCK_SIZE)
            * SCALE_COLS + k / BLOCK_SIZE;
        acc += float(xr[k]) * glm53_fp8_lut[wr[k]]
            * scale_inv[scale_offset];
    }
    acc = simd_sum(acc);
    constexpr uint NSIMD = THREADS / 32;
    threadgroup float partial[NSIMD];
    if (lane == 0) partial[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        float reduced = lane < NSIMD ? partial[lane] : 0.0f;
        reduced = simd_sum(reduced);
        if (lane == 0) {
            T down_value = T(reduced);
            weighted[size_t(selected) * OUT_FEATURES + out_row]
                = float(down_value) * float(scores[selected]);
        }
    }
"""


def _weighted_reduction_source(*, diagnostics: bool) -> str:
    reduced_write = "reduced[out_row] = total;" if diagnostics else ""
    return rf"""
    uint out_row = thread_position_in_grid.x;
    if (out_row >= OUT_FEATURES) return;
    float total = 0.0f;
    for (uint selected = 0; selected < TOP_K; ++selected) {{
        total += weighted[size_t(selected) * OUT_FEATURES + out_row];
    }}
    {reduced_write}
    output[out_row] = T(total);
    """


def _kernel(name, inputs, outputs, source):
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name=name,
        input_names=inputs,
        output_names=outputs,
        source=source,
        header=_FP8_LUT_HEADER,
    )


_shared_kernel = _kernel(
    "glm53_probe_shared_gate_up_swiglu",
    ["x", "gate_weight", "gate_scale_inv", "up_weight", "up_scale_inv"],
    ["hidden"],
    _shared_source(diagnostics=False),
)
_shared_diagnostic_kernel = _kernel(
    "glm53_probe_shared_gate_up_swiglu_diagnostic",
    ["x", "gate_weight", "gate_scale_inv", "up_weight", "up_scale_inv"],
    ["hidden", "gate_output", "up_output"],
    _shared_source(diagnostics=True),
)
_aggregation_kernel = _kernel(
    "glm53_probe_selected8_weighted_reduction",
    ["down", "scores"],
    ["output"],
    _aggregation_source(diagnostics=False),
)
_aggregation_diagnostic_kernel = _kernel(
    "glm53_probe_selected8_weighted_reduction_diagnostic",
    ["down", "scores"],
    ["output", "weighted", "reduced"],
    _aggregation_source(diagnostics=True),
)
_weighted_down_kernel = _kernel(
    "glm53_probe_packed_selected8_weighted_down",
    ["hidden", "scores", "expert_ids", "weight", "scale_inv"],
    ["weighted"],
    _WEIGHTED_DOWN_SOURCE,
)
_weighted_reduction_kernel = _kernel(
    "glm53_probe_selected8_reduce_weighted",
    ["weighted"],
    ["output"],
    _weighted_reduction_source(diagnostics=False),
)
_weighted_reduction_diagnostic_kernel = _kernel(
    "glm53_probe_selected8_reduce_weighted_diagnostic",
    ["weighted"],
    ["output", "reduced"],
    _weighted_reduction_source(diagnostics=True),
)


def fused_shared_gate_up_swiglu(x, shared, *, limit: float, diagnostics=False):
    kernel = _shared_diagnostic_kernel if diagnostics else _shared_kernel
    if kernel is None:
        raise RuntimeError("shared fusion probe requires Metal")
    gate = shared.gate_proj
    up = shared.up_proj
    if float(limit) != int(limit):
        raise ValueError("shared fusion probe requires an integral clamp limit")
    x = _metal_input(x)
    inputs = [
        x,
        _metal_input(gate.weight),
        _metal_input(gate.weight_scale_inv),
        _metal_input(up.weight),
        _metal_input(up.weight_scale_inv),
    ]
    intermediate = int(gate.weight.shape[0])
    output_count = 3 if diagnostics else 1
    outputs = kernel(
        inputs=inputs,
        template=[
            ("T", x.dtype),
            ("IN_FEATURES", int(x.shape[-1])),
            ("OUT_FEATURES", intermediate),
            ("SCALE_COLS", int(gate.weight_scale_inv.shape[1])),
            ("BLOCK_SIZE", BLOCK_SIZE),
            ("THREADS", THREADS),
            ("LIMIT", int(limit)),
        ],
        grid=(intermediate * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(intermediate,)] * output_count,
        output_dtypes=[x.dtype] * output_count,
    )
    if diagnostics:
        hidden, gate_output, up_output = outputs
        return gate_output, up_output, hidden
    return outputs[0]


def aggregate_b1(down, scores, *, diagnostics=False):
    kernel = _aggregation_diagnostic_kernel if diagnostics else _aggregation_kernel
    if kernel is None:
        raise RuntimeError("aggregation probe requires Metal")
    down = _metal_input(down)
    scores = _metal_input(scores)
    out_features = int(down.shape[1])
    outputs = kernel(
        inputs=[down, scores],
        template=[
            ("T", down.dtype),
            ("S", scores.dtype),
            ("OUT_FEATURES", out_features),
            ("TOP_K", DECODE_TOP_K),
        ],
        grid=(out_features, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=(
            [(out_features,), (DECODE_TOP_K, out_features), (out_features,)]
            if diagnostics
            else [(out_features,)]
        ),
        output_dtypes=(
            [down.dtype, mx.float32, mx.float32]
            if diagnostics
            else [down.dtype]
        ),
    )
    return tuple(outputs) if diagnostics else outputs[0]


def weighted_down_b2(hidden, scores, expert_ids, bank):
    if _weighted_down_kernel is None:
        raise RuntimeError("weighted-down probe requires Metal")
    hidden = _metal_input(hidden)
    scores = _metal_input(scores)
    expert_ids = _metal_input(expert_ids)
    out_features = int(bank.down_weight.shape[1])
    in_features = int(bank.down_weight.shape[2])
    return _weighted_down_kernel(
        inputs=[
            hidden,
            scores,
            expert_ids,
            _metal_input(bank.down_weight),
            _metal_input(bank.down_scale_inv),
        ],
        template=[
            ("T", hidden.dtype),
            ("S", scores.dtype),
            ("IN_FEATURES", in_features),
            ("OUT_FEATURES", out_features),
            ("TOP_K", DECODE_TOP_K),
            ("SCALE_ROWS", int(bank.down_scale_inv.shape[1])),
            ("SCALE_COLS", int(bank.down_scale_inv.shape[2])),
            ("BLOCK_SIZE", BLOCK_SIZE),
            ("THREADS", THREADS),
        ],
        grid=(DECODE_TOP_K * out_features * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(DECODE_TOP_K, out_features)],
        output_dtypes=[mx.float32],
    )[0]


def reduce_b2(weighted, *, output_dtype, diagnostics=False):
    kernel = (
        _weighted_reduction_diagnostic_kernel
        if diagnostics
        else _weighted_reduction_kernel
    )
    if kernel is None:
        raise RuntimeError("weighted reduction probe requires Metal")
    weighted = _metal_input(weighted)
    out_features = int(weighted.shape[1])
    outputs = kernel(
        inputs=[weighted],
        template=[
            ("T", output_dtype),
            ("OUT_FEATURES", out_features),
            ("TOP_K", DECODE_TOP_K),
        ],
        grid=(out_features, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=(
            [(out_features,), (out_features,)]
            if diagnostics
            else [(out_features,)]
        ),
        output_dtypes=(
            [output_dtype, mx.float32] if diagnostics else [output_dtype]
        ),
    )
    return tuple(outputs) if diagnostics else outputs[0]


def _existing_shared(shared, x, limit):
    gate = shared.gate_proj(x)
    up = shared.up_proj(x)
    hidden = nn.silu(mx.minimum(gate, limit)) * mx.clip(up, -limit, limit)
    return gate, up, hidden, shared.down_proj(hidden)


def _fused_shared(shared, x, limit, *, diagnostics=False):
    if diagnostics:
        gate, up, hidden = fused_shared_gate_up_swiglu(
            x, shared, limit=limit, diagnostics=True
        )
        return gate, up, hidden, shared.down_proj(hidden)
    hidden = fused_shared_gate_up_swiglu(x, shared, limit=limit)
    return shared.down_proj(hidden)


def _routed_stages(moe, x):
    flat = x.reshape(-1, x.shape[-1])
    indices, scores = moe.gate(x)
    expert_ids = indices.reshape(-1).astype(mx.uint32)
    flat_scores = scores.reshape(-1)
    hidden = d99f.fused_packed_gate_up_swiglu(
        flat[0], expert_ids, moe.bank, limit=moe.config.swiglu_limit
    )
    raw_down = d99f._packed_down_raw(hidden, expert_ids, moe.bank)
    weighted = raw_down.astype(mx.float32) * flat_scores[:, None]
    reduced = mx.sum(weighted, axis=0)
    routed = reduced.astype(hidden.dtype)
    b1_routed, b1_weighted, b1_reduced = aggregate_b1(
        raw_down, flat_scores, diagnostics=True
    )
    b2_weighted = weighted_down_b2(hidden, flat_scores, expert_ids, moe.bank)
    b2_routed, b2_reduced = reduce_b2(
        b2_weighted, output_dtype=hidden.dtype, diagnostics=True
    )
    values = (
        indices,
        scores,
        hidden,
        raw_down,
        weighted,
        reduced,
        routed,
        b1_routed,
        b1_weighted,
        b1_reduced,
        b2_weighted,
        b2_routed,
        b2_reduced,
    )
    mx.eval(*values)
    return {
        "indices": indices,
        "scores": scores,
        "hidden": hidden,
        "raw_down": raw_down,
        "weighted": weighted,
        "reduced": reduced,
        "routed": routed,
        "b1_weighted": b1_weighted,
        "b1_reduced": b1_reduced,
        "b1_routed": b1_routed,
        "b2_weighted": b2_weighted,
        "b2_reduced": b2_reduced,
        "b2_routed": b2_routed,
        "expert_ids": expert_ids,
        "flat_scores": flat_scores,
    }


def _stage_parity(moe, x):
    routed = _routed_stages(moe, x)
    shared = moe.shared_experts
    flat = x.reshape(-1, x.shape[-1])[0]
    gate, up, hidden, down = _existing_shared(
        shared, flat, moe.config.swiglu_limit
    )
    fused_gate, fused_up, fused_hidden, fused_down = _fused_shared(
        shared, flat, moe.config.swiglu_limit, diagnostics=True
    )
    mx.eval(gate, up, hidden, down, fused_gate, fused_up, fused_hidden, fused_down)
    existing_final = routed["routed"].reshape(x.shape) + down.reshape(x.shape)
    b1_final = routed["b1_routed"].reshape(x.shape) + down.reshape(x.shape)
    b2_final = routed["b2_routed"].reshape(x.shape) + down.reshape(x.shape)
    c_final = routed["routed"].reshape(x.shape) + fused_down.reshape(x.shape)
    b1d_final = routed["b1_routed"].reshape(x.shape) + fused_down.reshape(x.shape)
    b2d_final = routed["b2_routed"].reshape(x.shape) + fused_down.reshape(x.shape)
    mx.eval(existing_final, b1_final, b2_final, c_final, b1d_final, b2d_final)
    return {
        "routing": {
            "indices_hash": d99f._hash(routed["indices"]),
            "scores_hash": d99f._hash(routed["scores"]),
        },
        "down_aggregation": {
            "raw_down_bf16": d99f._metrics(routed["raw_down"], routed["raw_down"]),
            "b1_weighted_fp32": d99f._metrics(
                routed["weighted"], routed["b1_weighted"]
            ),
            "b1_reduced_fp32": d99f._metrics(
                routed["reduced"], routed["b1_reduced"]
            ),
            "b1_final_bf16": d99f._metrics(
                routed["routed"], routed["b1_routed"]
            ),
            "b2_weighted_fp32": d99f._metrics(
                routed["weighted"], routed["b2_weighted"]
            ),
            "b2_reduced_fp32": d99f._metrics(
                routed["reduced"], routed["b2_reduced"]
            ),
            "b2_final_bf16": d99f._metrics(
                routed["routed"], routed["b2_routed"]
            ),
        },
        "shared_expert": {
            "gate_output": d99f._metrics(gate, fused_gate),
            "up_output": d99f._metrics(up, fused_up),
            "activated_hidden": d99f._metrics(hidden, fused_hidden),
            "down_output": d99f._metrics(down, fused_down),
        },
        "final_moe": {
            "B1": d99f._metrics(existing_final, b1_final),
            "B2": d99f._metrics(existing_final, b2_final),
            "C": d99f._metrics(existing_final, c_final),
            "B1_plus_C": d99f._metrics(existing_final, b1d_final),
            "B2_plus_C": d99f._metrics(existing_final, b2d_final),
        },
    }


def _routed_variant(moe, x, aggregation):
    flat = x.reshape(-1, x.shape[-1])
    indices, scores = moe.gate(x)
    expert_ids = indices.reshape(-1).astype(mx.uint32)
    flat_scores = scores.reshape(-1)
    hidden = d99f.fused_packed_gate_up_swiglu(
        flat[0], expert_ids, moe.bank, limit=moe.config.swiglu_limit
    )
    if aggregation == "existing":
        return d99f._packed_selected_down(
            hidden, flat_scores, expert_ids, moe.bank
        ).reshape(x.shape)
    if aggregation == "B1":
        raw_down = d99f._packed_down_raw(hidden, expert_ids, moe.bank)
        return aggregate_b1(raw_down, flat_scores).reshape(x.shape)
    if aggregation == "B2":
        weighted = weighted_down_b2(hidden, flat_scores, expert_ids, moe.bank)
        return reduce_b2(weighted, output_dtype=hidden.dtype).reshape(x.shape)
    raise ValueError(aggregation)


def _moe_variant(moe, x, aggregation, shared_fused):
    result = _routed_variant(moe, x, aggregation)
    if moe.shared_experts is not None:
        if shared_fused:
            shared = _fused_shared(
                moe.shared_experts,
                x.reshape(-1, x.shape[-1])[0],
                moe.config.swiglu_limit,
            ).reshape(x.shape)
        else:
            shared = moe.shared_experts(x)
        result = result + shared
    return result


def _benchmark(callable_):
    samples = []
    hashes = []
    for sample in range(OPERATOR_WARMUPS + OPERATOR_SAMPLES):
        started = time.perf_counter()
        output = callable_()
        mx.eval(output)
        mx.synchronize()
        elapsed = (time.perf_counter() - started) * 1000.0
        if sample >= OPERATOR_WARMUPS:
            samples.append(elapsed)
            hashes.append(d99f._hash(output))
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "hashes": hashes,
        "repeat_hash_exact": len(set(hashes)) == 1,
    }


def _operator_benchmarks(moe, x):
    routed = {
        variant: _benchmark(lambda variant=variant: _routed_variant(moe, x, variant))
        for variant in ("existing", "B1", "B2")
    }
    baseline = routed["existing"]["median_ms"]
    for row in routed.values():
        row["speedup_vs_existing"] = baseline / row["median_ms"]
    shared = {
        "existing": _benchmark(lambda: moe.shared_experts(x)),
        "fused": _benchmark(
            lambda: _fused_shared(
                moe.shared_experts,
                x.reshape(-1, x.shape[-1])[0],
                moe.config.swiglu_limit,
            )
        ),
    }
    shared["fused"]["speedup_vs_existing"] = (
        shared["existing"]["median_ms"] / shared["fused"]["median_ms"]
    )
    return {"routed": routed, "shared": shared}


def _all_metrics_exact(node):
    if isinstance(node, dict) and "byte_identical" in node:
        return bool(node["byte_identical"])
    if isinstance(node, dict):
        return all(_all_metrics_exact(value) for value in node.values())
    return True


def _choose_aggregation(stage_parity, operator):
    exact = {}
    for variant in ("B1", "B2"):
        exact[variant] = all(
            layer["down_aggregation"][f"{variant.lower()}_weighted_fp32"][
                "byte_identical"
            ]
            and layer["down_aggregation"][f"{variant.lower()}_reduced_fp32"][
                "byte_identical"
            ]
            and layer["down_aggregation"][f"{variant.lower()}_final_bf16"][
                "byte_identical"
            ]
            for layer in stage_parity.values()
        )
    candidates = [variant for variant, is_exact in exact.items() if is_exact]
    if not candidates:
        return None, exact
    winner = min(
        candidates,
        key=lambda variant: sum(
            rows["routed"][variant]["median_ms"] for rows in operator.values()
        ),
    )
    return winner, exact


@dataclass(frozen=True)
class Arm:
    aggregation: str
    shared_fused: bool


_ORIGINAL_PACKED_CALL = PackedFP8MoE.__call__
_ACTIVE_ARM = Arm("existing", False)
_FUSED_CALL_COUNT = 0


def _patched_call(self, x):
    global _FUSED_CALL_COUNT
    flat = x.reshape(-1, x.shape[-1])
    if flat.shape[0] != 1:
        return _ORIGINAL_PACKED_CALL(self, x)
    _FUSED_CALL_COUNT += 1
    return _moe_variant(
        self, x, _ACTIVE_ARM.aggregation, _ACTIVE_ARM.shared_fused
    )


@contextmanager
def _runtime(arm):
    global _ACTIVE_ARM, _FUSED_CALL_COUNT
    _ACTIVE_ARM = arm
    _FUSED_CALL_COUNT = 0
    PackedFP8MoE.__call__ = _patched_call
    try:
        yield
    finally:
        PackedFP8MoE.__call__ = _ORIGINAL_PACKED_CALL


def _memory():
    mx.synchronize()
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }


def _release(*values):
    for value in values:
        if isinstance(value, list):
            value.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def _atomic_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _progress(phase, **values):
    print(json.dumps({"phase": phase, **values}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "bench-results/m3ultra512-residual-packed-decode-moe-fusion-20260901.json"
        ),
    )
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    args = parser.parse_args()

    report = inspect_checkpoint(args.model, require_server_ready=True)
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    mx.reset_peak_memory()
    _progress("load")
    model, _ = load(args.model, experimental_packed_decode_moe=True)
    warm_residency(model)
    steady_memory = _memory()
    inputs = d99f._capture_layer_inputs(model)

    stage_parity = {}
    operator = {}
    for layer_id in REPRESENTATIVE_LAYERS:
        _progress("layer", layer=layer_id)
        moe = model.language_model.model.layers[layer_id].mlp
        stage_parity[str(layer_id)] = _stage_parity(moe, inputs[layer_id])
        operator[str(layer_id)] = _operator_benchmarks(moe, inputs[layer_id])

    aggregation, aggregation_exact = _choose_aggregation(stage_parity, operator)
    shared_exact = all(
        _all_metrics_exact(layer["shared_expert"])
        for layer in stage_parity.values()
    )
    partial = {
        "schema": "glm53-residual-packed-decode-moe-fusion-v1",
        "date": date.today().isoformat(),
        "complete": False,
        "probe_only": True,
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "packed_decode_kernel_abi": PACKED_DECODE_KERNEL_ABI,
        "d99f_baseline_commit": "d99f26b",
        "stage_parity": stage_parity,
        "operator": operator,
        "aggregation_exact": aggregation_exact,
        "selected_aggregation": aggregation,
        "shared_exact": shared_exact,
        "steady_memory": steady_memory,
    }
    _atomic_write(args.output, partial)
    if aggregation is None or not shared_exact:
        partial.update(
            complete=True,
            runtime_candidate_accepted=False,
            decision="reject: no exact residual fusion combination",
        )
        _atomic_write(args.output, partial)
        return 1

    arms = {
        "A": Arm("existing", False),
        "B": Arm(aggregation, False),
        "C": Arm("existing", True),
        "D": Arm(aggregation, True),
    }
    full_model_2k = {}
    baseline_hashes = None
    for name, arm in arms.items():
        _progress("full_model_2k", arm=name)
        with _runtime(arm):
            row, hashes = packed_probe._frontier_arm(
                model, context=FRONTIER_CONTEXT, cache_backend="direct"
            )
            calls = _FUSED_CALL_COUNT
        if baseline_hashes is None:
            baseline_hashes = hashes
        full_model_2k[name] = {
            "result": row,
            "logits_hashes": hashes,
            "hashes_exact_vs_A": hashes == baseline_hashes,
            "fused_kernel_calls": calls,
        }
        partial["full_model_2k"] = full_model_2k
        _atomic_write(args.output, partial)

    decode_4096 = {}
    baseline_cache = None
    teacher_tokens = None
    baseline_evidence = None
    for name, arm in arms.items():
        _progress("decode_4096", arm=name)
        with _runtime(arm):
            cache, row = packed_probe._run_4096(model, teacher_tokens)
            calls = _FUSED_CALL_COUNT
        generated_tokens = row.pop("generated_tokens")
        if name == "A":
            baseline_cache = cache
            teacher_tokens = generated_tokens
            baseline_evidence = row["evidence_logits_hashes"]
            state_exact = True
        else:
            state_exact = boundary._cache_exact(baseline_cache, cache)
        decode_4096[name] = {
            "result": row,
            "generated_tokens_exact_vs_A": generated_tokens == teacher_tokens,
            "evidence_logits_hashes_exact_vs_A": (
                row["evidence_logits_hashes"] == baseline_evidence
            ),
            "final_kda_dsa_state_exact_vs_A": state_exact,
            "fused_kernel_calls": calls,
        }
        partial["decode_4096"] = decode_4096
        _atomic_write(args.output, partial)
        if name != "A":
            _release(cache)

    comparisons = {
        "operator_selected_aggregation": aggregation,
        "operator_routed_speedup_min": min(
            rows["routed"][aggregation]["speedup_vs_existing"]
            for rows in operator.values()
        ),
        "operator_shared_speedup_min": min(
            rows["shared"]["fused"]["speedup_vs_existing"]
            for rows in operator.values()
        ),
        "full_model_2k_speedup_vs_A": {
            name: full_model_2k["A"]["result"]["median_ms"]
            / row["result"]["median_ms"]
            for name, row in full_model_2k.items()
        },
        "decode_4096_tokens_per_second": {
            name: row["result"]["decode_tokens_per_second"]
            for name, row in decode_4096.items()
        },
        "decode_4096_median_ms": {
            name: row["result"]["decode_median_ms"]
            for name, row in decode_4096.items()
        },
        "decode_4096_speedup_vs_A": {
            name: row["result"]["decode_tokens_per_second"]
            / decode_4096["A"]["result"]["decode_tokens_per_second"]
            for name, row in decode_4096.items()
        },
    }
    correctness = {
        "layer_3_5_all_relevant_stages_exact": all(
            _all_metrics_exact(layer) for layer in stage_parity.values()
        ),
        "full_model_2k_logits_hash_exact_all_arms": all(
            row["hashes_exact_vs_A"] for row in full_model_2k.values()
        ),
        "decode_4096_generated_tokens_exact_all_arms": all(
            row["generated_tokens_exact_vs_A"] for row in decode_4096.values()
        ),
        "decode_4096_evidence_hashes_exact_all_arms": all(
            row["evidence_logits_hashes_exact_vs_A"]
            for row in decode_4096.values()
        ),
        "decode_4096_final_kda_dsa_state_exact_all_arms": all(
            row["final_kda_dsa_state_exact_vs_A"]
            for row in decode_4096.values()
        ),
        "decode_4096_materialization_count_16_all_arms": all(
            row["result"]["materialization_count"] == 16
            for row in decode_4096.values()
        ),
        "nan_and_metal_error_zero_all_arms": all(
            row["result"]["nan_count"] == 0
            and row["result"]["metal_error"] is None
            for row in decode_4096.values()
        ),
    }
    performance = {
        "B_or_C_at_least_14_tps": max(
            comparisons["decode_4096_tokens_per_second"]["B"],
            comparisons["decode_4096_tokens_per_second"]["C"],
        )
        >= 14.0,
        "D_at_least_15_tps": comparisons["decode_4096_tokens_per_second"]["D"]
        >= TARGET_DECODE_TPS,
        "D_median_at_most_66_67_ms": comparisons["decode_4096_median_ms"]["D"]
        <= TARGET_DECODE_MS,
    }
    runtime_candidate_accepted = all(correctness.values()) and all(
        performance.values()
    )
    artifact = {
        **partial,
        "complete": True,
        "arms": {
            name: {
                "gate_up": "d99f-exact-fused",
                "down_aggregation": arm.aggregation,
                "shared_expert": "fused-gate-up-swiglu"
                if arm.shared_fused
                else "existing",
            }
            for name, arm in arms.items()
        },
        "full_model_2k": full_model_2k,
        "decode_4096": decode_4096,
        "comparisons": comparisons,
        "correctness": correctness,
        "performance": performance,
        "runtime_candidate_accepted": runtime_candidate_accepted,
        "decision": (
            "promote exact residual packed decode fusion in a separate commit"
            if runtime_candidate_accepted
            else "retain as probe baseline; 15 tok/s production gate not met"
        ),
        "memory_final": _memory(),
        "runtime_changes": {
            "packed_runtime": False,
            "kernel_abi": False,
            "server": False,
            "apc": False,
            "admission": False,
        },
    }
    _atomic_write(args.output, artifact)
    _release(baseline_cache)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "runtime_candidate_accepted": runtime_candidate_accepted,
                "selected_aggregation": aggregation,
                "comparisons": comparisons,
            },
            indent=2,
        )
    )
    return 0 if runtime_candidate_accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
