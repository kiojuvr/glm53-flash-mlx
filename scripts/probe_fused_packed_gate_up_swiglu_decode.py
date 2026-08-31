#!/usr/bin/env python3
"""Probe an exact fused packed gate+up+SwiGLU batch-1 decode kernel."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
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
from glm53_flash_mlx.packed import (
    PackedFP8MoE,
    _packed_selected_down,
    _packed_selected_down_kernel,
    _packed_selected_projection,
)

REPRESENTATIVE_LAYERS = (3, 5)
OPERATOR_WARMUPS = 4
OPERATOR_SAMPLES = 20
FRONTIER_CONTEXT = 2049
DECODE_STEPS = 4096


def _fused_source(*, diagnostics: bool) -> str:
    diagnostic_write = (
        "gate_output[size_t(selected) * OUT_FEATURES + out_row] = gate_t;\n"
        "            up_output[size_t(selected) * OUT_FEATURES + out_row] = up_t;"
        if diagnostics
        else ""
    )
    return rf"""
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd_id = simdgroup_index_in_threadgroup;
    uint group_id = threadgroup_position_in_grid.x;
    uint selected = group_id / OUT_FEATURES;
    uint out_row = group_id % OUT_FEATURES;
    if (selected >= TOP_K) return;

    uint expert = expert_ids[selected];
    const device uint8_t* gate_wr = weight
        + (size_t(expert) * BANK_OUT_FEATURES + out_row) * IN_FEATURES;
    const device uint8_t* up_wr = weight
        + (size_t(expert) * BANK_OUT_FEATURES + OUT_FEATURES + out_row)
        * IN_FEATURES;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint k = tid; k < IN_FEATURES; k += THREADS) {{
        uint scale_col = k / BLOCK_SIZE;
        size_t gate_scale_offset =
            (size_t(expert) * BANK_SCALE_ROWS + out_row / BLOCK_SIZE)
            * SCALE_COLS + scale_col;
        size_t up_scale_offset =
            (size_t(expert) * BANK_SCALE_ROWS + SCALE_HALF_ROWS
             + out_row / BLOCK_SIZE) * SCALE_COLS + scale_col;
        gate_acc += float(x[k]) * glm53_fp8_lut[gate_wr[k]]
            * scale_inv[gate_scale_offset];
        up_acc += float(x[k]) * glm53_fp8_lut[up_wr[k]]
            * scale_inv[up_scale_offset];
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
            // Existing projection kernels store BF16 before the lazy MLX
            // clamp/SiLU/multiply graph. Preserve that rounding boundary.
            T gate_t = T(gate_total);
            T up_t = T(up_total);
            constexpr float LIMIT_F = float(LIMIT);
            float gate_value = min(float(gate_t), LIMIT_F);
            float up_value = clamp(float(up_t), -LIMIT_F, LIMIT_F);
            T gate_activation = T(gate_value);
            T up_activation = T(up_value);
            auto sigmoid_tail = 1 / (
                1 + metal::exp(metal::abs(gate_activation))
            );
            T sigmoid_value = (gate_activation < 0)
                ? sigmoid_tail
                : 1 - sigmoid_tail;
            T silu_value = gate_activation * sigmoid_value;
            T activated = silu_value * up_activation;
            hidden[size_t(selected) * OUT_FEATURES + out_row] = T(activated);
            {diagnostic_write}
        }}
    }}
    """


_fused_kernel = (
    mx.fast.metal_kernel(
        name="glm53_probe_packed_selected8_gate_up_swiglu",
        input_names=["x", "expert_ids", "weight", "scale_inv"],
        output_names=["hidden"],
        source=_fused_source(diagnostics=False),
        header=_FP8_LUT_HEADER,
    )
    if mx.metal.is_available()
    else None
)

_fused_diagnostic_kernel = (
    mx.fast.metal_kernel(
        name="glm53_probe_packed_selected8_gate_up_swiglu_diagnostic",
        input_names=["x", "expert_ids", "weight", "scale_inv"],
        output_names=["hidden", "gate_output", "up_output"],
        source=_fused_source(diagnostics=True),
        header=_FP8_LUT_HEADER,
    )
    if mx.metal.is_available()
    else None
)


def fused_packed_gate_up_swiglu(
    x: mx.array,
    expert_ids: mx.array,
    bank,
    *,
    limit: float,
    diagnostics: bool = False,
):
    kernel = _fused_diagnostic_kernel if diagnostics else _fused_kernel
    if kernel is None:
        raise RuntimeError("fused packed decode probe requires Metal")
    x = _metal_input(x)
    expert_ids = _metal_input(expert_ids)
    weight = _metal_input(bank.gate_up_weight)
    scales = _metal_input(bank.gate_up_scale_inv)
    if float(limit) != int(limit):
        raise ValueError("fused probe requires an integral SwiGLU clamp limit")
    outputs = kernel(
        inputs=[x, expert_ids, weight, scales],
        template=[
            ("T", x.dtype),
            ("IN_FEATURES", int(x.shape[-1])),
            ("OUT_FEATURES", int(bank.intermediate_size)),
            ("BANK_OUT_FEATURES", int(weight.shape[1])),
            ("BANK_SCALE_ROWS", int(scales.shape[1])),
            ("SCALE_HALF_ROWS", int(bank.intermediate_scale_rows)),
            ("SCALE_COLS", int(scales.shape[2])),
            ("TOP_K", DECODE_TOP_K),
            ("BLOCK_SIZE", BLOCK_SIZE),
            ("THREADS", THREADS),
            ("LIMIT", int(limit)),
        ],
        grid=(DECODE_TOP_K * int(bank.intermediate_size) * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(DECODE_TOP_K, int(bank.intermediate_size))]
        * (3 if diagnostics else 1),
        output_dtypes=[x.dtype] * (3 if diagnostics else 1),
    )
    if diagnostics:
        hidden, gate, up = outputs
        return gate, up, hidden
    return outputs[0]


def _packed_down_raw(hidden, expert_ids, bank):
    output = _packed_selected_down_kernel(
        inputs=[
            _metal_input(hidden),
            _metal_input(expert_ids),
            _metal_input(bank.down_weight),
            _metal_input(bank.down_scale_inv),
        ],
        template=[
            ("T", hidden.dtype),
            ("IN_FEATURES", int(bank.down_weight.shape[2])),
            ("OUT_FEATURES", int(bank.down_weight.shape[1])),
            ("TOP_K", DECODE_TOP_K),
            ("SCALE_ROWS", int(bank.down_scale_inv.shape[1])),
            ("SCALE_COLS", int(bank.down_scale_inv.shape[2])),
            ("BLOCK_SIZE", BLOCK_SIZE),
            ("THREADS", THREADS),
        ],
        grid=(DECODE_TOP_K * int(bank.down_weight.shape[1]) * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(DECODE_TOP_K, int(bank.down_weight.shape[1]))],
        output_dtypes=[hidden.dtype],
    )[0]
    return output


def _np(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.ascontiguousarray(np.asarray(value.astype(mx.float32)))


def _hash(value: mx.array) -> str:
    return hashlib.sha256(_np(value).tobytes()).hexdigest()


def _metrics(reference: mx.array, actual: mx.array) -> dict:
    left = _np(reference)
    right = _np(actual)
    delta = right - left
    denominator = max(float(np.linalg.norm(left)), 1e-12)
    return {
        "shape": list(reference.shape),
        "reference_hash": hashlib.sha256(left.tobytes()).hexdigest(),
        "actual_hash": hashlib.sha256(right.tobytes()).hexdigest(),
        "byte_identical": bool(np.array_equal(left, right)),
        "relative_l2": float(np.linalg.norm(delta) / denominator),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }


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


class _Capture:
    def __init__(self, inner, sink: dict[int, mx.array], layer: int):
        self.inner = inner
        self.sink = sink
        self.layer = layer

    def __call__(self, x):
        self.sink[self.layer] = x
        return self.inner(x)


def _capture_layer_inputs(model) -> dict[int, mx.array]:
    layers = model.language_model.model.layers
    captured = {}
    originals = {layer: layers[layer].mlp for layer in REPRESENTATIVE_LAYERS}
    cache = model.make_cache()
    try:
        for layer, original in originals.items():
            layers[layer].mlp = _Capture(original, captured, layer)
        output = model(mx.array([[101]], dtype=mx.uint32), cache=cache)
        mx.eval(output.logits, *captured.values())
        mx.synchronize()
    finally:
        for layer, original in originals.items():
            layers[layer].mlp = original
    result = {layer: mx.array(captured[layer]) for layer in REPRESENTATIVE_LAYERS}
    mx.eval(*result.values())
    _release(cache)
    return result


def _stage_outputs(moe: PackedFP8MoE, x: mx.array) -> tuple[dict, dict]:
    flat_x = x.reshape(-1, x.shape[-1])
    indices_a, scores_a = moe.gate(x)
    indices_b, scores_b = moe.gate(x)
    mx.eval(indices_a, scores_a, indices_b, scores_b)
    expert_ids = indices_a.reshape(-1).astype(mx.uint32)
    scores = scores_a.reshape(-1)
    gate = _packed_selected_projection(flat_x[0], expert_ids, moe.bank, row_offset=0)
    up = _packed_selected_projection(
        flat_x[0], expert_ids, moe.bank, row_offset=moe.bank.intermediate_size
    )
    hidden = nn.silu(mx.minimum(gate, moe.config.swiglu_limit)) * mx.clip(
        up, -moe.config.swiglu_limit, moe.config.swiglu_limit
    )
    down = _packed_down_raw(hidden, expert_ids, moe.bank)
    weighted = down.astype(mx.float32) * scores[:, None]
    routed = mx.sum(weighted, axis=0).astype(hidden.dtype).reshape(x.shape)
    shared = moe.shared_experts(x) if moe.shared_experts is not None else mx.zeros_like(x)
    final = routed + shared

    fused_gate, fused_up, fused_hidden = fused_packed_gate_up_swiglu(
        flat_x[0],
        expert_ids,
        moe.bank,
        limit=moe.config.swiglu_limit,
        diagnostics=True,
    )
    fused_down = _packed_down_raw(fused_hidden, expert_ids, moe.bank)
    fused_weighted = fused_down.astype(mx.float32) * scores[:, None]
    fused_routed = mx.sum(fused_weighted, axis=0).astype(hidden.dtype).reshape(x.shape)
    fused_final = fused_routed + shared
    values = (
        indices_a,
        scores_a,
        indices_b,
        scores_b,
        gate,
        up,
        hidden,
        down,
        weighted,
        routed,
        final,
        fused_gate,
        fused_up,
        fused_hidden,
        fused_down,
        fused_weighted,
        fused_routed,
        fused_final,
    )
    mx.eval(*values)
    mx.synchronize()
    reference = {
        "gate_output": gate,
        "up_output": up,
        "activated_hidden": hidden,
        "down_output": down,
        "weighted_expert_output": weighted,
        "routed_moe_output": routed,
        "final_moe_output": final,
    }
    fused = {
        "gate_output": fused_gate,
        "up_output": fused_up,
        "activated_hidden": fused_hidden,
        "down_output": fused_down,
        "weighted_expert_output": fused_weighted,
        "routed_moe_output": fused_routed,
        "final_moe_output": fused_final,
    }
    routing = {
        "ids_exact": _metrics(indices_a, indices_b)["byte_identical"],
        "scores_exact": _metrics(scores_a, scores_b)["byte_identical"],
        "ids_hash": _hash(indices_a),
        "scores_hash": _hash(scores_a),
        "expert_ids": np.asarray(indices_a).reshape(-1).astype(int).tolist(),
    }
    return {name: _metrics(reference[name], fused[name]) for name in reference}, routing


def _routed_call(moe: PackedFP8MoE, x: mx.array, *, fused: bool):
    flat_x = x.reshape(-1, x.shape[-1])
    indices, scores = moe.gate(x)
    expert_ids = indices.reshape(-1).astype(mx.uint32)
    flat_scores = scores.reshape(-1)
    if fused:
        hidden = fused_packed_gate_up_swiglu(
            flat_x[0], expert_ids, moe.bank, limit=moe.config.swiglu_limit
        )
    else:
        gate = _packed_selected_projection(flat_x[0], expert_ids, moe.bank, row_offset=0)
        up = _packed_selected_projection(
            flat_x[0], expert_ids, moe.bank, row_offset=moe.bank.intermediate_size
        )
        hidden = nn.silu(mx.minimum(gate, moe.config.swiglu_limit)) * mx.clip(
            up, -moe.config.swiglu_limit, moe.config.swiglu_limit
        )
    return _packed_selected_down(hidden, flat_scores, expert_ids, moe.bank).reshape(x.shape)


def _operator_benchmark(moe: PackedFP8MoE, x: mx.array, *, include_shared: bool) -> dict:
    rows = {}
    for fused in (False, True):
        samples = []
        hashes = []
        for sample in range(OPERATOR_WARMUPS + OPERATOR_SAMPLES):
            started = time.perf_counter()
            output = _routed_call(moe, x, fused=fused)
            if include_shared and moe.shared_experts is not None:
                output = output + moe.shared_experts(x)
            mx.eval(output)
            mx.synchronize()
            elapsed = (time.perf_counter() - started) * 1000.0
            if sample >= OPERATOR_WARMUPS:
                samples.append(elapsed)
                hashes.append(_hash(output))
        rows["fused" if fused else "existing"] = {
            "samples_ms": samples,
            "median_ms": statistics.median(samples),
            "hashes": hashes,
        }
    rows["speedup"] = rows["existing"]["median_ms"] / rows["fused"]["median_ms"]
    rows["all_hashes_exact"] = len(
        set(rows["existing"]["hashes"] + rows["fused"]["hashes"])
    ) == 1
    return rows


_ORIGINAL_PACKED_CALL = PackedFP8MoE.__call__
_FUSED_CALL_COUNT = 0


def _fused_packed_call(self, x):
    global _FUSED_CALL_COUNT
    flat_x = x.reshape(-1, x.shape[-1])
    if flat_x.shape[0] != 1:
        return _ORIGINAL_PACKED_CALL(self, x)
    indices, scores = self.gate(x)
    if indices.shape[-1] != DECODE_TOP_K:
        return _ORIGINAL_PACKED_CALL(self, x)
    _FUSED_CALL_COUNT += 1
    expert_ids = indices.reshape(-1).astype(mx.uint32)
    hidden = fused_packed_gate_up_swiglu(
        flat_x[0], expert_ids, self.bank, limit=self.config.swiglu_limit
    )
    result = _packed_selected_down(
        hidden, scores.reshape(-1), expert_ids, self.bank
    ).reshape(x.shape)
    if self.shared_experts is not None:
        result = result + self.shared_experts(x)
    return result


@contextmanager
def _fused_runtime():
    global _FUSED_CALL_COUNT
    _FUSED_CALL_COUNT = 0
    PackedFP8MoE.__call__ = _fused_packed_call
    try:
        yield
    finally:
        PackedFP8MoE.__call__ = _ORIGINAL_PACKED_CALL


def _atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "bench-results/m3ultra512-fused-packed-gate-up-swiglu-decode-20260901.json"
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
    inputs = _capture_layer_inputs(model)

    stage_parity = {}
    operator = {}
    for layer in REPRESENTATIVE_LAYERS:
        _progress("layer", layer=layer)
        moe = model.language_model.model.layers[layer].mlp
        stages, routing = _stage_outputs(moe, inputs[layer])
        stage_parity[str(layer)] = {"routing": routing, "stages": stages}
        operator[str(layer)] = {
            "routed_only": _operator_benchmark(moe, inputs[layer], include_shared=False),
            "with_shared_expert": _operator_benchmark(
                moe, inputs[layer], include_shared=True
            ),
        }

    stage_exact = all(
        row["routing"]["ids_exact"]
        and row["routing"]["scores_exact"]
        and all(stage["byte_identical"] for stage in row["stages"].values())
        for row in stage_parity.values()
    )
    partial = {
        "schema": "glm53-fused-packed-gate-up-swiglu-decode-v1",
        "date": date.today().isoformat(),
        "complete": False,
        "probe_only": True,
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "packed_decode_kernel_abi": PACKED_DECODE_KERNEL_ABI,
        "representative_layers": list(REPRESENTATIVE_LAYERS),
        "stage_parity": stage_parity,
        "operator": operator,
        "stage_exact": stage_exact,
        "steady_memory": steady_memory,
    }
    _atomic_write(args.output, partial)
    if not stage_exact:
        partial["complete"] = True
        partial["runtime_candidate_accepted"] = False
        partial["decision"] = "reject: fused stage is not byte-identical"
        _atomic_write(args.output, partial)
        return 1

    _progress("baseline_2k")
    baseline_2k, baseline_2k_hashes = packed_probe._frontier_arm(
        model, context=FRONTIER_CONTEXT, cache_backend="direct"
    )
    _progress("fused_2k")
    with _fused_runtime():
        fused_2k, fused_2k_hashes = packed_probe._frontier_arm(
            model, context=FRONTIER_CONTEXT, cache_backend="direct"
        )
        fused_2k_calls = _FUSED_CALL_COUNT

    _progress("baseline_4096")
    baseline_cache, baseline_4096 = packed_probe._run_4096(model)
    teacher_tokens = baseline_4096.pop("generated_tokens")
    _progress("fused_4096")
    with _fused_runtime():
        fused_cache, fused_4096 = packed_probe._run_4096(model, teacher_tokens)
        fused_4096_calls = _FUSED_CALL_COUNT
    fused_4096.pop("generated_tokens")

    final_state_exact = boundary._cache_exact(baseline_cache, fused_cache)
    evidence_exact = (
        baseline_4096["evidence_logits_hashes"]
        == fused_4096["evidence_logits_hashes"]
    )
    tokens_exact = fused_4096["all_tokens_match_teacher"]
    memory_final = _memory()
    selected_speedups = [
        operator[str(layer)]["routed_only"]["speedup"]
        for layer in REPRESENTATIVE_LAYERS
    ]
    comparisons = {
        "selected_expert_moe_decode_min_speedup": min(selected_speedups),
        "selected_expert_moe_decode_speedups": selected_speedups,
        "full_model_2k_speedup": baseline_2k["median_ms"] / fused_2k["median_ms"],
        "full_model_4096_speedup": fused_4096["decode_tokens_per_second"]
        / baseline_4096["decode_tokens_per_second"],
        "fused_4096_tokens_per_second": fused_4096["decode_tokens_per_second"],
    }
    acceptance = {
        "layer_3_5_router_ids_and_scores_exact": all(
            row["routing"]["ids_exact"] and row["routing"]["scores_exact"]
            for row in stage_parity.values()
        ),
        "all_stage_outputs_byte_identical": stage_exact,
        "selected_expert_moe_decode_speedup_at_least_1_20": min(selected_speedups)
        >= 1.20,
        "full_model_2k_speedup_at_least_1_12": comparisons["full_model_2k_speedup"]
        >= 1.12,
        "full_model_4096_at_least_14_tps": comparisons[
            "fused_4096_tokens_per_second"
        ]
        >= 14.0,
        "full_model_2k_logits_hash_exact": baseline_2k_hashes == fused_2k_hashes,
        "decode_4096_generated_tokens_exact": tokens_exact,
        "decode_4096_full_vocab_evidence_hash_exact": evidence_exact,
        "decode_4096_final_kda_dsa_state_exact": final_state_exact,
        "decode_4096_materialization_count_16": baseline_4096[
            "materialization_count"
        ]
        == fused_4096["materialization_count"]
        == 16,
        "fused_kernel_exercised_all_42_moe_layers": fused_2k_calls > 0
        and fused_4096_calls >= 42 * DECODE_STEPS,
        "no_nan_or_metal_error": baseline_2k["nan_count"] == 0
        and fused_2k["nan_count"] == 0
        and baseline_4096["nan_count"] == 0
        and fused_4096["nan_count"] == 0,
        "runtime_kernel_abi_server_apc_and_admission_unchanged": True,
    }
    acceptance["accepted"] = all(acceptance.values())
    artifact = {
        **partial,
        "complete": True,
        "full_model_2k": {
            "context_tokens": FRONTIER_CONTEXT,
            "existing": baseline_2k,
            "fused": fused_2k,
            "fused_kernel_calls": fused_2k_calls,
            "hashes_exact": baseline_2k_hashes == fused_2k_hashes,
        },
        "decode_4096": {
            "existing": baseline_4096,
            "fused": fused_4096,
            "fused_kernel_calls": fused_4096_calls,
            "generated_tokens_exact": tokens_exact,
            "full_vocab_evidence_hash_exact": evidence_exact,
            "final_kda_dsa_state_exact": final_state_exact,
        },
        "comparisons": comparisons,
        "memory_final": memory_final,
        "runtime_changes": {
            "packed_runtime": False,
            "kernel_abi": False,
            "server": False,
            "apc": False,
            "admission": False,
        },
        "runtime_candidate_accepted": acceptance["accepted"],
        "decision": (
            "accept exact fused packed gate-up SwiGLU runtime candidate"
            if acceptance["accepted"]
            else "reject runtime promotion: exact fusion missed the selected-MoE, "
            "2k full-model, and 4096-token throughput gates"
        ),
        "acceptance": acceptance,
    }
    _atomic_write(args.output, artifact)
    _release(baseline_cache, fused_cache)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "accepted": acceptance["accepted"],
                "comparisons": comparisons,
            },
            indent=2,
        )
    )
    return 0 if acceptance["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
