#!/usr/bin/env python3
"""Prove the final exactness gate for a compiled KDA Q-scale barrier."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import tempfile
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import localize_compiled_kda_numerical_barriers as metrics
import localize_compiled_kda_recurrent_readout_barrier as readout
import probe_compiled_packed_ffn_fp32_router as compiled_ffn_probe
import probe_exact_sigmoid_gate_metal_barrier as sigmoid_probe
import probe_functional_stateful_decode_executable as functional
import probe_long_context_first_decode_boundary as boundary
import probe_packed_decode_runtime as packed_probe
import probe_residual_packed_decode_moe_fusion as residual
from glm53_flash_mlx.abi import MLX_VLM_REVISION
from glm53_flash_mlx.loader import _make_config, load, warm_residency
from glm53_flash_mlx.manifest import EXPECTED_KDA, inspect_checkpoint


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-exact-compiled-kda-q-scale-barrier-20260902.json"
)
LOCAL_LAYERS = (0, 10, 22, 25, 42)
STEPS = 64
THREADS = 256
HEAD_DIM = 128
SCALE_BITS = 0x3DB504F3
SIGMOID_MODE = 7
HOST_BUILD_PREFERRED = 0.60
HOST_BUILD_FLOOR = 0.40
WORKING_PEAK_LIMIT = 64 * 2**20

STAGES = (
    "q_input_fp32",
    "q_l2normalized_fp32",
    "q_scaled_fp32",
    "normalized_q_bf16",
    "recurrent_output_bf16",
    "gated_norm_input_bf16",
    "final_projection_bf16",
)


_SCALE_F32_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= N) return;
    output[index] = input[index] * scale[0];
"""


_SCALE_BF16_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= N) return;
    float scaled = input[index] * scale[0];
    output[index] = OutT(scaled);
"""


_scale_f32_kernel = (
    mx.fast.metal_kernel(
        name="glm53_probe_exact_kda_q_scale_f32",
        input_names=["input", "scale"],
        output_names=["output"],
        source=_SCALE_F32_SOURCE,
    )
    if mx.metal.is_available()
    else None
)


_scale_bf16_kernel = (
    mx.fast.metal_kernel(
        name="glm53_probe_exact_kda_q_scale_bf16",
        input_names=["input", "scale"],
        output_names=["output"],
        source=_SCALE_BF16_SOURCE,
    )
    if mx.metal.is_available()
    else None
)


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


def _metal_input(value: mx.array) -> mx.array:
    return mx.contiguous(value, allow_col_major=False)


def exact_q_scale_f32(value: mx.array, scale: mx.array) -> mx.array:
    if _scale_f32_kernel is None:
        raise RuntimeError("exact Q-scale barrier requires Metal")
    if value.dtype != mx.float32 or scale.dtype != mx.float32:
        raise TypeError("exact Q-scale FP32 barrier requires FP32 inputs")
    if scale.shape != (1,):
        raise ValueError("exact Q-scale requires a one-element scale tensor")
    count = int(value.size)
    return _scale_f32_kernel(
        inputs=[_metal_input(value), _metal_input(scale)],
        template=[("N", count)],
        grid=((count + THREADS - 1) // THREADS * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[value.shape],
        output_dtypes=[mx.float32],
    )[0]


def exact_q_scale_bf16(value: mx.array, scale: mx.array) -> mx.array:
    if _scale_bf16_kernel is None:
        raise RuntimeError("exact Q-scale barrier requires Metal")
    if value.dtype != mx.float32 or scale.dtype != mx.float32:
        raise TypeError("exact Q-scale BF16 barrier requires FP32 inputs")
    if scale.shape != (1,):
        raise ValueError("exact Q-scale requires a one-element scale tensor")
    count = int(value.size)
    return _scale_bf16_kernel(
        inputs=[_metal_input(value), _metal_input(scale)],
        template=[("N", count), ("OutT", mx.bfloat16)],
        grid=((count + THREADS - 1) // THREADS * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[value.shape],
        output_dtypes=[mx.bfloat16],
    )[0]


def _scale_tensor() -> mx.array:
    bits = np.asarray([SCALE_BITS], dtype=np.uint32)
    return mx.array(bits.view(np.float32), dtype=mx.float32)


@contextmanager
def _exact_norm(attention):
    original = attention.o_norm
    attention.o_norm = sigmoid_probe.SigmoidBarrierNorm(
        original, mode=SIGMOID_MODE
    )
    try:
        yield
    finally:
        attention.o_norm = original


def _replace_scale(prefix: tuple, mode: str, scale: mx.array) -> tuple:
    q_l2_index = readout.PREFIX_STAGE_NAMES.index("q_l2normalized_fp32")
    q_scaled_index = readout.PREFIX_STAGE_NAMES.index("q_scaled_fp32")
    q_bf16_index = readout.PREFIX_STAGE_NAMES.index("normalized_q")
    q_l2 = prefix[q_l2_index]
    if mode == "B_compiled_constant":
        return prefix
    if mode == "C_runtime_scalar":
        q_scaled = q_l2 * scale[0]
        q_bf16 = q_scaled.astype(mx.bfloat16)
    elif mode == "D_metal_f32":
        q_scaled = exact_q_scale_f32(q_l2, scale)
        q_bf16 = q_scaled.astype(mx.bfloat16)
    elif mode == "E_metal_f32_bf16":
        q_scaled = exact_q_scale_f32(q_l2, scale)
        q_bf16 = exact_q_scale_bf16(q_l2, scale)
    else:
        raise ValueError(f"unsupported Q-scale arm: {mode}")
    values = list(prefix)
    values[q_scaled_index] = q_scaled
    values[q_bf16_index] = q_bf16
    return tuple(values)


def _full(attention, inputs, conv_state, recurrent_state, position, scale, mode):
    prefix = readout._prefix(attention, inputs, conv_state)
    prefix = _replace_scale(prefix, mode, scale)
    recurrent_output, next_state = readout._recurrence(prefix, recurrent_state)
    normalized, output = readout._tail(attention, recurrent_output, prefix[-1])
    return (
        *prefix,
        recurrent_output,
        next_state,
        normalized,
        output,
        position + mx.array(1, mx.int32),
    )


def _diagnostic_values(full: tuple) -> tuple:
    prefix = full[: len(readout.PREFIX_STAGE_NAMES)]
    recurrent, _, normalized, output = full[-5:-1]
    return (
        prefix[readout.PREFIX_STAGE_NAMES.index("q_input_fp32")],
        prefix[readout.PREFIX_STAGE_NAMES.index("q_l2normalized_fp32")],
        prefix[readout.PREFIX_STAGE_NAMES.index("q_scaled_fp32")],
        prefix[readout.PREFIX_STAGE_NAMES.index("normalized_q")],
        recurrent,
        normalized,
        output,
    )


def _compile(attention, mode: str, counter: dict[str, int]):
    def traced(inputs, conv, recurrent, position, scale):
        counter["calls"] += 1
        return _full(
            attention, inputs, conv, recurrent, position, scale, mode
        )

    return mx.compile(traced)


def _boundary_evidence(reference: tuple, actual: tuple) -> dict:
    scaled = metrics._metrics(reference[2], actual[2])
    rounded = metrics._metrics(reference[3], actual[3])
    evidence = {
        "scale_bits": f"0x{SCALE_BITS:08x}",
        "scaled_fp32": scaled,
        "scaled_bf16": rounded,
        "bf16_boundary": None,
    }
    first = rounded["first_difference"]
    if first is not None:
        index = first["flat_index"]
        ref_f32 = np.ascontiguousarray(
            np.asarray(reference[2], dtype=np.float32)
        ).reshape(-1)
        actual_f32 = np.ascontiguousarray(
            np.asarray(actual[2], dtype=np.float32)
        ).reshape(-1)
        ref_bf16 = np.ascontiguousarray(
            np.asarray(reference[3].view(mx.uint16), dtype=np.uint16)
        ).reshape(-1)
        actual_bf16 = np.ascontiguousarray(
            np.asarray(actual[3].view(mx.uint16), dtype=np.uint16)
        ).reshape(-1)
        reference_bits = int(ref_bf16[index])
        evidence["bf16_boundary"] = {
            "flat_index": index,
            "reference_fp32_bits": f"0x{ref_f32[index].view(np.uint32):08x}",
            "actual_fp32_bits": f"0x{actual_f32[index].view(np.uint32):08x}",
            "reference_bf16_bits": f"0x{reference_bits:04x}",
            "actual_bf16_bits": f"0x{int(actual_bf16[index]):04x}",
            "reference_bf16_predecessor_bits": f"0x{max(0, reference_bits - 1):04x}",
            "reference_bf16_successor_bits": f"0x{min(0xffff, reference_bits + 1):04x}",
        }
    return evidence


def _run_arm(attention, inputs, mode: str, *, compiled: bool) -> dict:
    scale = _scale_tensor()
    counter = {"calls": 0}
    callable_ = (
        _compile(attention, mode, counter)
        if compiled
        else lambda x, conv, recurrent, position, scalar: _full(
            attention, x, conv, recurrent, position, scalar, mode
        )
    )
    state = functional.initial_kda_state(attention)
    stages = []
    state_hashes = []
    build_ms = []
    step_ms = []
    mx.synchronize()
    active_before = int(mx.get_active_memory())
    mx.reset_peak_memory()
    for step in range(STEPS):
        started = time.perf_counter_ns()
        result = callable_(
            inputs, *state, mx.array(step, mx.int32), scale
        )
        built = time.perf_counter_ns()
        mx.eval(*result)
        mx.synchronize()
        finished = time.perf_counter_ns()
        stages.append(_diagnostic_values(result))
        prefix = result[: len(readout.PREFIX_STAGE_NAMES)]
        state = (prefix[7], result[-4])
        state_hashes.append(tuple(functional._hash(value) for value in state))
        build_ms.append((built - started) / 1e6)
        step_ms.append((finished - started) / 1e6)
    return {
        "stages": stages,
        "state_hashes": state_hashes,
        "state": state,
        "compile_trace_calls": counter["calls"],
        "host_build_median_ms": float(statistics.median(build_ms[1:])),
        "step_median_ms": float(statistics.median(step_ms[1:])),
        "working_peak_delta_bytes": max(
            0, int(mx.get_peak_memory()) - active_before
        ),
    }


class CompiledQScaleKDA(nn.Module):
    """Probe-only fixed-shape decode wrapper around one KDA attention layer."""

    def __init__(self, inner):
        super().__init__()
        inner.o_norm = sigmoid_probe.SigmoidBarrierNorm(
            inner.o_norm, mode=SIGMOID_MODE
        )
        self.inner = inner
        self.conv_kernel_size = inner.conv_kernel_size
        self.conv_dim = inner.conv_dim
        self.num_heads = inner.num_heads
        self.head_dim = inner.head_dim
        self.scale = _scale_tensor()
        self.compile_trace_calls = 0

        def decoded(inputs, conv_state, recurrent_state, scale):
            self.compile_trace_calls += 1
            prefix = readout._prefix(self.inner, inputs, conv_state)
            prefix = _replace_scale(prefix, "C_runtime_scalar", scale)
            recurrent, next_recurrent = readout._recurrence(
                prefix, recurrent_state
            )
            _, output = readout._tail(
                self.inner, recurrent, prefix[-1]
            )
            return output, prefix[7], next_recurrent

        self._compiled_decode = mx.compile(decoded)

    def __call__(self, inputs, mask=None, cache=None):
        batch, length, _ = inputs.shape
        if batch != 1 or length != 1 or mask is not None or cache is None:
            return self.inner(inputs, mask=mask, cache=cache)
        conv_state = cache[0]
        recurrent_state = cache[1]
        if conv_state is None or recurrent_state is None:
            initial = functional.initial_kda_state(
                self.inner, dtype=inputs.dtype
            )
            conv_state = initial[0] if conv_state is None else conv_state
            recurrent_state = (
                initial[1] if recurrent_state is None else recurrent_state
            )
        output, next_conv, next_recurrent = self._compiled_decode(
            inputs, conv_state, recurrent_state, self.scale
        )
        cache[0] = next_conv
        cache[1] = next_recurrent
        cache.advance(1)
        return output


def _screen_64(model) -> tuple[object, dict]:
    cache = boundary._synthetic_cache(model, 2049, "direct")
    latencies = []
    hashes = []
    nan_count = 0
    mx.synchronize()
    active_before = int(mx.get_active_memory())
    mx.reset_peak_memory()
    for step in range(66):
        token = mx.array([[3000 + step]], dtype=mx.uint32)
        started = time.perf_counter_ns()
        output = model(token, cache=cache)
        logits = output.logits[0, -1]
        nan = mx.sum(mx.isnan(logits))
        mx.eval(logits, nan)
        mx.synchronize()
        finished = time.perf_counter_ns()
        nan_count += int(nan.item())
        if step >= 2:
            latencies.append((finished - started) / 1e6)
            hashes.append(packed_probe._hash(logits))
    ordered = sorted(latencies)
    return cache, {
        "context_tokens": 2049,
        "warmups": 2,
        "steps": 64,
        "median_ms": float(statistics.median(latencies)),
        "p95_ms": float(ordered[int(0.95 * (len(ordered) - 1))]),
        "tokens_per_second": 1000.0 / statistics.median(latencies),
        "logits_hashes": hashes,
        "nan_count": nan_count,
        "active_memory_drift_bytes": int(mx.get_active_memory()) - active_before,
        "working_peak_delta_bytes": max(
            0, int(mx.get_peak_memory()) - active_before
        ),
    }


def _full_model_oracle(path: Path, report) -> dict:
    _progress("load_full_model")
    mx.set_wired_limit(int(440e9))
    mx.set_cache_limit(int(32e9))
    load_started = time.perf_counter()
    model, processor = load(path, experimental_packed_decode_moe=True)
    load_seconds = time.perf_counter() - load_started
    residency_started = time.perf_counter()
    warm_residency(model)
    residency_seconds = time.perf_counter() - residency_started

    _progress("official_oracle_baseline", tokens=128)
    baseline = sigmoid_probe._official_oracle(model, processor, report)
    wrappers = []
    for layer_id in EXPECTED_KDA:
        layer = model.language_model.model.layers[layer_id]
        wrapper = CompiledQScaleKDA(layer.self_attn)
        layer.self_attn = wrapper
        wrappers.append(wrapper)
    _progress("official_oracle_candidate", tokens=128)
    candidate = sigmoid_probe._official_oracle(model, processor, report)
    trace_counts = [wrapper.compile_trace_calls for wrapper in wrappers]
    _progress("full_token_screen", steps=64)
    with residual._runtime(residual.Arm("B1", True)):
        compiled_ffn_probe._configure_compile(model, True)
        for layer_id, wrapper in zip(EXPECTED_KDA, wrappers, strict=True):
            model.language_model.model.layers[layer_id].self_attn = wrapper.inner
        compile_warmup = compiled_ffn_probe._warm_compiled_ffn(model, True)
        baseline_cache, baseline_screen = _screen_64(model)
        for layer_id, wrapper in zip(EXPECTED_KDA, wrappers, strict=True):
            model.language_model.model.layers[layer_id].self_attn = wrapper
        candidate_cache, candidate_screen = _screen_64(model)
    screen_exact = (
        baseline_screen["logits_hashes"] == candidate_screen["logits_hashes"]
        and boundary._cache_exact(baseline_cache, candidate_cache)
    )
    return {
        "executed": True,
        "load_seconds": load_seconds,
        "warm_residency_seconds": residency_seconds,
        "moe_backend": getattr(model, "_glm53_moe_backend", "direct"),
        "installed_kda_layers": list(EXPECTED_KDA),
        "baseline": baseline,
        "candidate": candidate,
        "baseline_candidate_token_digest_exact": (
            baseline["generated_token_sha256"]
            == candidate["generated_token_sha256"]
        ),
        "all_compiled_layers_trace_once": all(count == 1 for count in trace_counts),
        "compile_trace_calls": trace_counts,
        "full_token_screen": {
            "backend": "residual-D + compiled sparse FFN + packed decode",
            "compile_warmup": compile_warmup,
            "baseline": baseline_screen,
            "candidate": candidate_screen,
            "all_logits_and_final_cache_exact": screen_exact,
            "candidate_speedup": (
                candidate_screen["tokens_per_second"]
                / baseline_screen["tokens_per_second"]
            ),
        },
        "memory": readout._memory(),
    }


def _comparison(reference: dict, candidate: dict) -> dict:
    divergent = []
    state_divergent = []
    first = None
    for step, (expected, actual) in enumerate(
        zip(reference["stages"], candidate["stages"], strict=True)
    ):
        rows = {
            name: metrics._metrics(left, right)
            for name, left, right in zip(STAGES, expected, actual, strict=True)
        }
        if not all(row["byte_identical"] for row in rows.values()):
            divergent.append(step)
            if first is None:
                first = {
                    "step_zero_based": step,
                    "first_stage": next(
                        name
                        for name, row in rows.items()
                        if not row["byte_identical"]
                    ),
                    "stages": rows,
                    "q_scale_boundary": _boundary_evidence(expected, actual),
                }
    state_divergent = [
        step
        for step, (expected, actual) in enumerate(
            zip(
                reference["state_hashes"],
                candidate["state_hashes"],
                strict=True,
            )
        )
        if expected != actual
    ]
    eager_build = reference["host_build_median_ms"]
    return {
        "all_64_steps_byte_identical": not divergent,
        "divergent_steps_zero_based": divergent,
        "state_divergent_steps_zero_based": state_divergent,
        "all_step_states_byte_identical": not state_divergent,
        "first_divergence": first,
        "conv_state_byte_identical": metrics._exact(
            reference["state"][0], candidate["state"][0]
        ),
        "recurrent_state_byte_identical": metrics._exact(
            reference["state"][1], candidate["state"][1]
        ),
        "compile_trace_calls": candidate["compile_trace_calls"],
        "host_build_median_ms": candidate["host_build_median_ms"],
        "host_build_reduction": 1.0
        - candidate["host_build_median_ms"] / eager_build,
        "step_median_ms": candidate["step_median_ms"],
        "step_speedup": reference["step_median_ms"]
        / candidate["step_median_ms"],
        "working_peak_delta_bytes": candidate["working_peak_delta_bytes"],
    }


def _layer(path: Path, config, layer_id: int, arms: tuple[str, ...]) -> dict:
    _progress("layer", layer=layer_id, arms=list(arms))
    attention = sigmoid_probe._load_kda_attention_layer(path, config, layer_id)
    inputs = functional._deterministic_input(attention.hidden_size)
    with _exact_norm(attention):
        reference = _run_arm(
            attention, inputs, "B_compiled_constant", compiled=False
        )
        candidates = {
            arm: _run_arm(attention, inputs, arm, compiled=True)
            for arm in arms
        }
    result = {
        "layer": layer_id,
        "reference": {
            "host_build_median_ms": reference["host_build_median_ms"],
            "step_median_ms": reference["step_median_ms"],
        },
        "arms": {
            arm: _comparison(reference, candidate)
            for arm, candidate in candidates.items()
        },
    }
    del attention, reference, candidates
    gc.collect()
    mx.clear_cache()
    return result


def _repeat_gate(path: Path, config, layer_id: int, winner: str) -> dict:
    attention = sigmoid_probe._load_kda_attention_layer(path, config, layer_id)
    inputs = functional._deterministic_input(attention.hidden_size)
    with _exact_norm(attention):
        first = _run_arm(attention, inputs, winner, compiled=True)
        second = _run_arm(attention, inputs, winner, compiled=True)
    exact = all(
        all(metrics._exact(left, right) for left, right in zip(a, b, strict=True))
        for a, b in zip(first["stages"], second["stages"], strict=True)
    ) and all(
        metrics._exact(left, right)
        for left, right in zip(first["state"], second["state"], strict=True)
    )
    del attention, first, second
    gc.collect()
    mx.clear_cache()
    return {"layer": layer_id, "winner": winner, "byte_identical": exact}


def _strided_gate(path: Path, config, winner: str) -> dict:
    attention = sigmoid_probe._load_kda_attention_layer(path, config, 0)
    contiguous = functional._deterministic_input(attention.hidden_size)
    strided = functional._deterministic_input(
        attention.hidden_size, strided=True
    )
    with _exact_norm(attention):
        left = _run_arm(attention, contiguous, winner, compiled=True)
        right = _run_arm(attention, strided, winner, compiled=True)
    exact = all(
        all(metrics._exact(a, b) for a, b in zip(x, y, strict=True))
        for x, y in zip(left["stages"], right["stages"], strict=True)
    ) and all(
        metrics._exact(a, b)
        for a, b in zip(left["state"], right["state"], strict=True)
    )
    del attention, left, right
    gc.collect()
    mx.clear_cache()
    return {"layer": 0, "byte_identical": exact}


def _winner(local: dict) -> str | None:
    for arm in (
        "C_runtime_scalar",
        "D_metal_f32",
        "E_metal_f32_bf16",
    ):
        if all(
            row["arms"][arm]["all_64_steps_byte_identical"]
            and row["arms"][arm]["all_step_states_byte_identical"]
            and row["arms"][arm]["conv_state_byte_identical"]
            and row["arms"][arm]["recurrent_state_byte_identical"]
            and row["arms"][arm]["compile_trace_calls"] == 1
            for row in local.values()
        ):
            return arm
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.time()
    report = inspect_checkpoint(args.model, require_server_ready=True)
    config = _make_config(json.loads((args.model / "config.json").read_text()))
    local_arms = (
        "B_compiled_constant",
        "C_runtime_scalar",
        "D_metal_f32",
        "E_metal_f32_bf16",
    )
    local = {
        str(layer): _layer(args.model, config, layer, local_arms)
        for layer in LOCAL_LAYERS
    }
    winner = _winner(local)
    all_layers = {}
    repeat = {"executed": False, "reason": "no exact local candidate"}
    strided = {"executed": False, "reason": "no exact local candidate"}
    if winner is not None:
        all_layers = {
            str(layer): (
                local[str(layer)]
                if layer in LOCAL_LAYERS
                else _layer(args.model, config, layer, (winner,))
            )
            for layer in EXPECTED_KDA
        }
        repeat = {
            "executed": True,
            "rows": [
                _repeat_gate(args.model, config, layer, winner)
                for layer in LOCAL_LAYERS
            ],
        }
        strided = {
            "executed": True,
            **_strided_gate(args.model, config, winner),
        }
    all_exact = bool(
        winner
        and len(all_layers) == len(EXPECTED_KDA)
        and all(
            row["arms"][winner]["all_64_steps_byte_identical"]
            and row["arms"][winner]["all_step_states_byte_identical"]
            and row["arms"][winner]["conv_state_byte_identical"]
            and row["arms"][winner]["recurrent_state_byte_identical"]
            and row["arms"][winner]["compile_trace_calls"] == 1
            for row in all_layers.values()
        )
    )
    third_blockers = []
    if winner is not None:
        third_blockers = [
            {
                "layer": int(layer),
                "first_divergence": row["arms"][winner]["first_divergence"],
            }
            for layer, row in all_layers.items()
            if not row["arms"][winner]["all_64_steps_byte_identical"]
            or not row["arms"][winner]["all_step_states_byte_identical"]
        ]
    reductions = (
        [row["arms"][winner]["host_build_reduction"] for row in all_layers.values()]
        if winner is not None
        else []
    )
    peak = (
        max(
            row["arms"][winner]["working_peak_delta_bytes"]
            for row in all_layers.values()
        )
        if winner is not None
        else None
    )
    repeat_exact = bool(
        repeat.get("executed")
        and all(row["byte_identical"] for row in repeat["rows"])
    )
    if all_exact and repeat_exact and strided.get("byte_identical", False):
        official_oracle = _full_model_oracle(args.model, report)
    else:
        official_oracle = {
            "executed": False,
            "reason": "all-34-layer final numerical gate did not pass",
        }
    official_exact = bool(
        official_oracle.get("executed", False)
        and official_oracle["baseline"]["first_16_match"]
        and official_oracle["baseline"]["full_128_match"]
        and official_oracle["candidate"]["first_16_match"]
        and official_oracle["candidate"]["full_128_match"]
        and official_oracle["baseline_candidate_token_digest_exact"]
        and official_oracle["all_compiled_layers_trace_once"]
    )
    full_token_screen = official_oracle.get("full_token_screen", {})
    screen_exact = bool(
        full_token_screen.get("all_logits_and_final_cache_exact", False)
    )
    screen_tps = full_token_screen.get("candidate", {}).get(
        "tokens_per_second"
    )
    screen_peak = full_token_screen.get("candidate", {}).get(
        "working_peak_delta_bytes"
    )
    artifact = {
        "schema": "glm53-exact-compiled-kda-q-scale-barrier-v1",
        "date": str(date.today()),
        "complete": True,
        "probe_only": True,
        "runtime_changes": False,
        "kernel_abi_changes": False,
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "steps": STEPS,
        "scale_contract": {
            "head_dim": HEAD_DIM,
            "fp32_bits_hex": f"0x{SCALE_BITS:08x}",
            "fp32_value": float(
                np.asarray([SCALE_BITS], dtype=np.uint32).view(np.float32)[0]
            ),
            "formula_regeneration_forbidden": True,
        },
        "arm_contract": {
            "A_eager": "eager normalization, eager scale, eager BF16 cast",
            "B_compiled_constant": "compiled existing Python constant expression",
            "C_runtime_scalar": "compiled explicit FP32 scalar input multiply",
            "D_metal_f32": "opaque Metal FP32 multiply, compiled BF16 cast",
            "E_metal_f32_bf16": "opaque Metal FP32 multiply and BF16 cast",
        },
        "local_layers": local,
        "selected_minimal_candidate": winner,
        "all_34_kda_layers": all_layers,
        "repeatability": repeat,
        "strided_contiguous": strided,
        "third_numerical_blockers": third_blockers,
        "official_oracle": official_oracle,
        "performance": {
            "host_build_reduction_min": min(reductions) if reductions else None,
            "host_build_reduction_median": (
                float(statistics.median(reductions)) if reductions else None
            ),
            "working_peak_delta_bytes_max": peak,
            "gpu_busy_regression": None,
            "additional_command_buffers_per_token": None,
            "full_token_64_step_tok_s": screen_tps,
            "full_token_screen": full_token_screen,
            "note": (
                "GPU busy and dynamic command-buffer deltas require the next "
                "bounded System Trace gate"
            ),
        },
        "acceptance": {
            "scale_bits_exact": SCALE_BITS == 0x3DB504F3,
            "local_candidate_exists": winner is not None,
            "all_34_layers_64_steps_byte_exact": all_exact,
            "repeatability_exact": repeat_exact,
            "strided_contiguous_exact": bool(strided.get("byte_identical", False)),
            "no_third_numerical_blocker": not third_blockers,
            "official_16_128_full_vocab_oracle_exact": official_exact,
            "full_token_64_step_logits_and_cache_exact": screen_exact,
            "full_token_64_step_at_least_14_7_tps": bool(
                screen_tps is not None and screen_tps >= 14.7
            ),
            "host_build_reduction_preferred_60pct": bool(
                reductions and min(reductions) >= HOST_BUILD_PREFERRED
            ),
            "host_build_reduction_hard_floor_40pct": bool(
                reductions and min(reductions) >= HOST_BUILD_FLOOR
            ),
            "working_peak_delta_at_most_64mib": bool(
                peak is not None and peak <= WORKING_PEAK_LIMIT
            ),
            "full_token_working_peak_delta_at_most_64mib": bool(
                screen_peak is not None and screen_peak <= WORKING_PEAK_LIMIT
            ),
            "new_q_scale_metal_primitive_required": winner
            in ("D_metal_f32", "E_metal_f32_bf16"),
            "q_scale_is_only_new_numerical_workaround": winner
            in ("C_runtime_scalar", "D_metal_f32", "E_metal_f32_bf16"),
            "runtime_unchanged": True,
        },
        "decision": (
            "proceed_to_bounded_system_trace"
            if (
                all_exact
                and repeat_exact
                and strided.get("byte_identical", False)
                and official_exact
                and screen_exact
                and screen_tps is not None
                and screen_tps >= 14.7
            )
            else "stop_mlx_compiled_kda_full_token_performance_gate"
            if official_exact and screen_tps is not None and screen_tps < 14.7
            else "stop_mlx_compiled_kda_official_oracle_mismatch"
            if all_exact and not official_exact
            else "stop_mlx_compiled_kda_after_third_numerical_blocker"
            if third_blockers
            else "stop_mlx_compiled_kda_q_scale_not_exact"
        ),
        "elapsed_seconds": time.time() - started,
        "memory": readout._memory(),
    }
    _atomic_write(args.output, artifact)
    _progress("complete", decision=artifact["decision"], winner=winner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
