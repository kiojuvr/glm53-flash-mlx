#!/usr/bin/env python3
"""Probe minimal and fused opaque Metal barriers for exact compiled KDA."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import tempfile
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np

import localize_compiled_kda_numerical_barriers as barriers
import probe_functional_stateful_decode_executable as functional
from glm53_flash_mlx.abi import KERNEL_ABI_VERSION, MLX_VLM_REVISION
from glm53_flash_mlx.loader import _make_config, load, warm_residency
from glm53_flash_mlx.manifest import EXPECTED_KDA, inspect_checkpoint
from glm53_flash_mlx.patch import apply_runtime_patch


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-exact-sigmoid-gate-metal-barrier-20260902.json"
)
STEPS = 64
OFFSET_FIXTURES = (0, 1, 255, 256, 2048)
BIT_CRITICAL_STEPS = (6, 31)
REPRESENTATIVE_KDA_LAYERS = (0, 20, 44)
HOST_BUILD_REDUCTION_GATE = 0.40
WORKING_PEAK_LIMIT = 64 * 2**20
THREADS = 256


_SIGMOID_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= N) return;
    float value = gate[index];
    float result;
    if (MODE == 0) {
        // MLX eager sigmoid ordering candidate.
        result = 1.0f / (1.0f + metal::exp(-value));
    } else if (MODE == 1) {
        // Numerically stable sign-branch candidate used by the exact SwiGLU
        // anchor.  It is measured, not assumed equivalent.
        float tail = 1.0f / (1.0f + metal::exp(metal::abs(value)));
        result = value < 0.0f ? tail : 1.0f - tail;
    } else if (MODE == 2) {
        if (value < 0.0f) {
            float numerator = metal::exp(value);
            result = numerator / (1.0f + numerator);
        } else {
            result = 1.0f / (1.0f + metal::exp(-value));
        }
    } else if (MODE == 3) {
        result = (metal::tanh(value * 0.5f) + 1.0f) * 0.5f;
    } else if (MODE == 4) {
        result = 0.5f * metal::tanh(0.5f * value) + 0.5f;
    } else if (MODE == 5) {
        float numerator = metal::exp(min(value, 0.0f));
        result = numerator / (1.0f + metal::exp(-metal::abs(value)));
    } else if (MODE == 6) {
        // Literal transcription of MLX v0.32.2 Sigmoid::operator().  Keep the
        // integer literals and auto temporary: bit parity, not algebraic
        // equivalence, is the contract of this probe.
        auto y = 1 / (1 + metal::exp(metal::abs(value)));
        result = (value < 0) ? y : 1 - y;
    } else {
        // The same MLX ordering with an explicit precise intrinsic so the
        // custom-JIT math mode cannot silently select the fast implementation.
        auto y = 1 / (1 + metal::precise::exp(metal::abs(value)));
        result = (value < 0) ? y : 1 - y;
    }
    output[index] = result;
"""


_FUSED_NORM_SOURCE = r"""
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_threadgroup;
    uint index = row * WIDTH + lane;
    float value = lane < WIDTH ? float(hidden[index]) : 0.0f;
    float square = value * value;
    float partial = simd_sum(square);
    constexpr uint NSIMD = THREADS_PER_GROUP / 32;
    threadgroup float sums[NSIMD];
    if (thread_index_in_simdgroup == 0) {
        sums[simdgroup_index_in_threadgroup] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_index_in_threadgroup == 0) {
        float subtotal = lane < NSIMD ? sums[lane] : 0.0f;
        subtotal = simd_sum(subtotal);
        if (lane == 0) sums[0] = subtotal;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane >= WIDTH) return;
    float inverse_rms = metal::precise::rsqrt(
        sums[0] / float(WIDTH) + epsilon[0]);
    float normalized = value * inverse_rms;
    float weighted = float(weight[lane]) * normalized;
    float gate_value = float(gate[index]);
    auto sigmoid_tail = 1 / (
        1 + metal::precise::exp(metal::abs(gate_value)));
    float sigmoid = gate_value < 0 ? sigmoid_tail : 1 - sigmoid_tail;
    output[index] = OutT(weighted * sigmoid);
"""


_sigmoid_kernel = (
    mx.fast.metal_kernel(
        name="glm53_probe_exact_sigmoid_gate",
        input_names=["gate"],
        output_names=["output"],
        source=_SIGMOID_SOURCE,
    )
    if mx.metal.is_available()
    else None
)


_fused_norm_kernel = (
    mx.fast.metal_kernel(
        name="glm53_probe_exact_gated_rmsnorm",
        input_names=["hidden", "gate", "weight", "epsilon"],
        output_names=["output"],
        source=_FUSED_NORM_SOURCE,
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


def exact_sigmoid_gate(gate_f32: mx.array, *, mode: int = 0) -> mx.array:
    if _sigmoid_kernel is None:
        raise RuntimeError("exact sigmoid barrier requires Metal")
    if gate_f32.dtype != mx.float32:
        raise TypeError("exact sigmoid barrier requires an FP32 input")
    if mode not in (0, 1, 2, 3, 4, 5, 6, 7):
        raise ValueError("unsupported sigmoid ordering mode")
    gate_f32 = _metal_input(gate_f32)
    count = int(gate_f32.size)
    return _sigmoid_kernel(
        inputs=[gate_f32],
        template=[("N", count), ("MODE", mode)],
        grid=((count + THREADS - 1) // THREADS * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[gate_f32.shape],
        output_dtypes=[mx.float32],
    )[0]


def validate_finite_gate(gate: mx.array) -> None:
    if gate.dtype not in (mx.bfloat16, mx.float32):
        raise TypeError("gate must be BF16 or FP32")
    if not bool(mx.all(mx.isfinite(gate)).item()):
        raise ValueError("NaN/Inf gate input is rejected before Metal dispatch")


def fused_gated_rmsnorm(hidden, gate, weight, eps: float):
    if _fused_norm_kernel is None:
        raise RuntimeError("fused gated RMSNorm barrier requires Metal")
    if hidden.shape != gate.shape:
        raise ValueError("hidden and gate shapes must match")
    width = int(hidden.shape[-1])
    if width != 128:
        raise ValueError("GLM-5.3 KDA gated RMSNorm width must be 128")
    if weight.shape != (width,):
        raise ValueError("gated RMSNorm weight shape mismatch")
    rows = int(hidden.size // width)
    return _fused_norm_kernel(
        inputs=[
            _metal_input(hidden),
            _metal_input(gate),
            _metal_input(weight),
            mx.array([eps], dtype=mx.float32),
        ],
        template=[
            ("InT", hidden.dtype),
            ("GateT", gate.dtype),
            ("WeightT", weight.dtype),
            ("OutT", hidden.dtype),
            ("WIDTH", width),
            ("THREADS_PER_GROUP", 128),
        ],
        grid=(rows * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[hidden.shape],
        output_dtypes=[hidden.dtype],
    )[0]


class SigmoidBarrierNorm:
    def __init__(self, inner, *, mode: int):
        self.weight = inner.weight
        self.eps = inner.eps
        self.mode = mode

    def __call__(self, hidden, gate):
        dtype = hidden.dtype
        value = hidden.astype(mx.float32)
        variance = (value * value).mean(-1, keepdims=True)
        value = value * mx.rsqrt(variance + self.eps)
        value = self.weight.astype(mx.float32) * value
        sigmoid = exact_sigmoid_gate(gate.astype(mx.float32), mode=self.mode)
        return (value * sigmoid).astype(dtype)


class FusedBarrierNorm:
    def __init__(self, inner):
        self.weight = inner.weight
        self.eps = inner.eps

    def __call__(self, hidden, gate):
        return fused_gated_rmsnorm(
            hidden, gate, self.weight, self.eps
        )


@contextmanager
def _norm(attention, replacement):
    original = attention.o_norm
    attention.o_norm = replacement
    try:
        yield
    finally:
        attention.o_norm = original


def _bit_fixture_values(actual_gate: mx.array) -> mx.array:
    actual = actual_gate.astype(mx.float32).reshape(-1)
    mx.eval(actual)
    index = min(74, int(actual.size) - 1)
    measured = actual[index : index + 1]
    measured_np = float(np.asarray(measured)[0])
    neighbors = np.array(
        [
            np.nextafter(np.float32(measured_np), np.float32(-np.inf)),
            np.float32(measured_np),
            np.nextafter(np.float32(measured_np), np.float32(np.inf)),
        ],
        dtype=np.float32,
    )
    finite = np.array(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            20.0,
            -20.0,
            np.finfo(np.float32).tiny,
            -np.finfo(np.float32).tiny,
            np.nextafter(np.float32(0.0), np.float32(1.0)),
            np.nextafter(np.float32(0.0), np.float32(-1.0)),
        ],
        dtype=np.float32,
    )
    return mx.array(np.concatenate([neighbors, finite]))


def _sigmoid_formula_gate(gate: mx.array) -> dict:
    actual_flat = gate.astype(mx.float32).reshape(-1)
    synthetic = _bit_fixture_values(gate)
    fixtures = {
        "actual_gate": actual_flat,
        "synthetic_boundaries": synthetic,
    }
    modes = {}
    formulas = (
        (0, "naive_exp_divide"),
        (1, "stable_tail_subtract"),
        (2, "stable_common_branch"),
        (3, "tanh_add_then_half"),
        (4, "tanh_half_then_add"),
        (5, "exp_min_over_abs_denominator"),
        (6, "mlx_v0322_literal"),
        (7, "mlx_v0322_precise_exp"),
    )
    for mode, name in formulas:
        rows = {}
        exact_all = True
        for fixture_name, values in fixtures.items():
            eager = mx.sigmoid(values)
            metal = exact_sigmoid_gate(values, mode=mode)
            mx.eval(eager, metal)
            metrics = barriers._metrics(eager, metal)
            rows[fixture_name] = metrics
            exact_all &= metrics["byte_identical"]
        modes[name] = {
            "mode": mode,
            "all_fixtures_byte_identical": exact_all,
            "fixtures": rows,
        }

    rejected = []
    for value in (
        mx.array([float("nan")], dtype=mx.float32),
        mx.array([float("inf")], dtype=mx.float32),
        mx.array([float("-inf")], dtype=mx.float32),
    ):
        try:
            validate_finite_gate(value)
        except ValueError:
            rejected.append(True)
        else:
            rejected.append(False)
    return {
        "modes": modes,
        "nan_inf_cases": 3,
        "nan_inf_rejected_before_dispatch": all(rejected),
        "selected_mode": next(
            (
                row["mode"]
                for row in modes.values()
                if row["all_fixtures_byte_identical"]
            ),
            None,
        ),
    }


def _run_path(
    attention,
    inputs,
    norm_replacement,
    *,
    compiled: bool,
    initial_state=None,
    steps: int = STEPS,
):
    counter = {"calls": 0}

    def decoded(inputs_, conv, recurrent, position):
        counter["calls"] += 1
        return functional.functional_kda_decode(
            attention, inputs_, conv, recurrent, position
        )

    with _norm(attention, norm_replacement):
        callable_ = mx.compile(decoded) if compiled else decoded
        state = (
            functional.initial_kda_state(attention)
            if initial_state is None
            else initial_state
        )
        outputs = []
        build_ms = []
        elapsed_ms = []
        peak_before = int(mx.get_peak_memory())
        for step in range(steps):
            started = time.perf_counter_ns()
            result = callable_(inputs, *state, mx.array(step, mx.int32))
            built = time.perf_counter_ns()
            mx.eval(*result)
            mx.synchronize()
            finished = time.perf_counter_ns()
            outputs.append(result[0])
            state = result[1:3]
            build_ms.append((built - started) / 1e6)
            elapsed_ms.append((finished - started) / 1e6)
        peak_after = int(mx.get_peak_memory())
    return {
        "outputs": outputs,
        "state": state,
        "compile_trace_calls": counter["calls"],
        "host_build_median_ms": float(statistics.median(build_ms[1:])),
        "step_median_ms": float(statistics.median(elapsed_ms[1:])),
        "working_peak_delta_bytes": max(0, peak_after - peak_before),
    }


def _load_kda_attention_layer(path: Path, config, layer_id: int):
    """Load one audited KDA layer without materializing the full checkpoint."""
    if layer_id not in EXPECTED_KDA:
        raise ValueError(f"layer {layer_id} is not an audited KDA layer")
    apply_runtime_patch()
    from mlx_vlm.models.glm5_next.language import Glm5NextLinearAttention

    attention = Glm5NextLinearAttention(config.text_config)
    prefix = f"model.language_model.layers.{layer_id}.self_attn."
    raw = functional._checkpoint_tensors(path, prefix)
    relative = {name[len(prefix) :]: value for name, value in raw.items()}
    qkv_conv = [relative.pop(f"{name}_conv1d.weight") for name in "qkv"]
    conv = mx.concatenate(qkv_conv, axis=0)
    if conv.ndim == 3 and conv.shape[-1] != 1:
        conv = conv.moveaxis(2, 1)
    weights = {
        "conv1d.weight": conv,
        "forget_gate.A_log": relative.pop("A_log").astype(mx.float32),
        "forget_gate.dt_bias": relative.pop("dt_bias").astype(mx.float32),
        "forget_gate.f_a_proj.weight": relative.pop("f_a_proj.weight"),
        "forget_gate.f_b_proj.weight": relative.pop("f_b_proj.weight"),
        **relative,
    }
    attention.load_weights(list(weights.items()), strict=True)
    attention.fuse_in = False
    attention.eval()
    return attention


def _nonzero_state(attention):
    conv, recurrent = functional.initial_kda_state(attention)
    conv_values = mx.arange(conv.size, dtype=mx.float32).reshape(conv.shape)
    recurrent_values = mx.arange(
        recurrent.size, dtype=mx.float32
    ).reshape(recurrent.shape)
    conv = (mx.sin(conv_values * mx.array(0.0009765625)) * 0.01).astype(
        conv.dtype
    )
    recurrent = mx.cos(
        recurrent_values * mx.array(0.0000152587890625)
    ) * mx.array(0.001, mx.float32)
    mx.eval(conv, recurrent)
    return conv, recurrent


def _snapshot_replay_gate(attention, inputs, replacement) -> dict:
    snapshot = _nonzero_state(attention)
    before = tuple(functional._hash(value) for value in snapshot)
    candidate = _run_path(
        attention,
        inputs,
        replacement,
        compiled=True,
        initial_state=snapshot,
        steps=16,
    )
    after_candidate = tuple(functional._hash(value) for value in snapshot)
    replay = _run_path(
        attention,
        inputs,
        attention.o_norm,
        compiled=False,
        initial_state=snapshot,
        steps=16,
    )
    return {
        "snapshot_unchanged": before == after_candidate,
        "all_replay_outputs_byte_identical": all(
            barriers._exact(left, right)
            for left, right in zip(
                replay["outputs"], candidate["outputs"], strict=True
            )
        ),
        "conv_state_byte_identical": barriers._exact(
            replay["state"][0], candidate["state"][0]
        ),
        "recurrent_state_byte_identical": barriers._exact(
            replay["state"][1], candidate["state"][1]
        ),
        "snapshot_hashes": list(before),
    }


def _layer_gate(
    path: Path,
    config,
    layer_id: int,
    *,
    candidate_name: str,
    mode: int,
) -> dict:
    _progress("kda_layer", layer=layer_id)
    attention = _load_kda_attention_layer(path, config, layer_id)
    inputs = functional._deterministic_input(attention.hidden_size)
    original = attention.o_norm
    if candidate_name == "B_sigmoid_only":
        replacement = SigmoidBarrierNorm(original, mode=mode)
    elif candidate_name == "C_fused_gated_rmsnorm":
        replacement = FusedBarrierNorm(original)
    else:
        raise ValueError(f"unsupported barrier candidate: {candidate_name}")
    reference = _run_path(attention, inputs, original, compiled=False)
    candidate = _run_path(attention, inputs, replacement, compiled=True)
    metrics = _path_metrics(reference, candidate)
    metrics["layer"] = layer_id
    metrics["candidate"] = candidate_name
    metrics["steps"] = STEPS
    metrics["host_build_reduction"] = 1.0 - (
        candidate["host_build_median_ms"] / reference["host_build_median_ms"]
    )
    if not metrics["all_outputs_byte_identical"]:
        metrics["first_divergence_stages"] = _first_divergence_stages(
            attention, inputs, replacement
        )
    if layer_id in REPRESENTATIVE_KDA_LAYERS:
        nonzero = _nonzero_state(attention)
        nonzero_reference = _run_path(
            attention,
            inputs,
            original,
            compiled=False,
            initial_state=nonzero,
        )
        nonzero_candidate = _run_path(
            attention,
            inputs,
            replacement,
            compiled=True,
            initial_state=nonzero,
        )
        metrics["nonzero_state"] = _path_metrics(
            nonzero_reference, nonzero_candidate
        )
        metrics["snapshot_restore_replay"] = _snapshot_replay_gate(
            attention, inputs, replacement
        )
    del attention, reference, candidate
    gc.collect()
    mx.clear_cache()
    return metrics


def _all_kda_gate(
    path: Path, config, *, candidate_name: str, mode: int
) -> dict:
    rows = {
        str(layer_id): _layer_gate(
            path,
            config,
            layer_id,
            candidate_name=candidate_name,
            mode=mode,
        )
        for layer_id in EXPECTED_KDA
    }
    exact = all(
        row["all_outputs_byte_identical"]
        and row["conv_state_byte_identical"]
        and row["recurrent_state_byte_identical"]
        and row["compile_trace_calls"] == 1
        for row in rows.values()
    )
    representative_exact = all(
        rows[str(layer_id)]["nonzero_state"]["all_outputs_byte_identical"]
        and rows[str(layer_id)]["nonzero_state"][
            "conv_state_byte_identical"
        ]
        and rows[str(layer_id)]["nonzero_state"][
            "recurrent_state_byte_identical"
        ]
        and all(
            value
            for key, value in rows[str(layer_id)][
                "snapshot_restore_replay"
            ].items()
            if key != "snapshot_hashes"
        )
        for layer_id in REPRESENTATIVE_KDA_LAYERS
    )
    reductions = [row["host_build_reduction"] for row in rows.values()]
    return {
        "executed": True,
        "candidate": candidate_name,
        "layers": list(EXPECTED_KDA),
        "representative_layers": list(REPRESENTATIVE_KDA_LAYERS),
        "all_34_layers_byte_identical": exact,
        "representative_nonzero_snapshot_replay_exact": representative_exact,
        "host_build_reduction_min": min(reductions),
        "host_build_reduction_median": float(statistics.median(reductions)),
        "rows": rows,
    }


def _official_oracle(model, processor, report) -> dict:
    import oracle_trace

    expected_16 = json.loads(
        (REPOSITORY / "oracles/glm53-official-greedy-16.json").read_text()
    )
    expected_128 = json.loads(
        (REPOSITORY / "oracles/glm53-official-greedy-128.json").read_text()
    )
    prompt = oracle_trace.DEFAULT_PROMPT
    formatted = processor.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = processor(formatted, return_tensors="np", add_special_tokens=True)
    prompt_ids = np.asarray(encoded["input_ids"], dtype=np.int32).reshape(1, -1)
    cache = model.make_cache()
    output = model(mx.array(prompt_ids), cache=cache)
    generated = []
    steps = []
    started = time.perf_counter()
    for step in range(128):
        logits = output.logits[0, -1].astype(mx.float32)
        mx.eval(logits)
        values = np.ascontiguousarray(np.asarray(logits), dtype=np.float32)
        top2 = np.argpartition(values, -2)[-2:]
        top2 = top2[np.argsort(values[top2])[::-1]]
        token = int(top2[0])
        generated.append(token)
        steps.append(
            {
                "step": step,
                "token": token,
                "logits_f32_sha256": hashlib.sha256(
                    values.tobytes()
                ).hexdigest(),
            }
        )
        if step + 1 < 128:
            output = model(mx.array([[token]], dtype=mx.uint32), cache=cache)

    def trace(count: int) -> dict:
        return {
            "schema": "glm53-greedy-oracle-v1",
            "official_hf_revision": report.official_revision,
            "checkpoint_fingerprint": report.fingerprint,
            "checkpoint_layout_digest": report.layout_digest,
            "kernel_abi": KERNEL_ABI_VERSION,
            "prompt": prompt,
            "formatted_prompt_sha256": hashlib.sha256(
                formatted.encode()
            ).hexdigest(),
            "prompt_token_count": int(prompt_ids.size),
            "prompt_token_ids_sha256": hashlib.sha256(
                prompt_ids.tobytes()
            ).hexdigest(),
            "generation_tokens": count,
            "generated_token_ids": generated[:count],
            "steps": steps[:count],
        }

    actual_16 = trace(16)
    actual_128 = trace(128)
    failures_16 = oracle_trace.compare_trace(actual_16, expected_16)
    failures_128 = oracle_trace.compare_trace(actual_128, expected_128)
    return {
        "executed": True,
        "prompt_tokens": int(prompt_ids.size),
        "generated_token_sha256": hashlib.sha256(
            np.asarray(generated, dtype=np.uint32).tobytes()
        ).hexdigest(),
        "first_16_match": not failures_16,
        "full_128_match": not failures_128,
        "all_full_vocab_logits_hashes_match": not failures_128,
        "failures_16": failures_16,
        "failures_128": failures_128,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _full_model_gate(
    path: Path, report, *, candidate_name: str, mode: int
) -> dict:
    _progress("load_full_model")
    mx.set_wired_limit(int(440e9))
    mx.set_cache_limit(int(32e9))
    started = time.perf_counter()
    model, processor = load(path, experimental_packed_decode_moe=True)
    load_seconds = time.perf_counter() - started
    warm_started = time.perf_counter()
    warm_residency(model)
    warm_seconds = time.perf_counter() - warm_started
    installed = []
    for layer_id in EXPECTED_KDA:
        attention = model.language_model.model.layers[layer_id].self_attn
        if candidate_name == "B_sigmoid_only":
            attention.o_norm = SigmoidBarrierNorm(attention.o_norm, mode=mode)
        elif candidate_name == "C_fused_gated_rmsnorm":
            attention.o_norm = FusedBarrierNorm(attention.o_norm)
        else:
            raise ValueError(f"unsupported barrier candidate: {candidate_name}")
        installed.append(layer_id)
    _progress("official_oracle", tokens=128)
    oracle = _official_oracle(model, processor, report)
    memory = functional._memory()
    return {
        "executed": True,
        "candidate": candidate_name,
        "moe_backend": getattr(model, "_glm53_moe_backend", "direct"),
        "installed_kda_layers": installed,
        "load_seconds": load_seconds,
        "warm_residency_seconds": warm_seconds,
        "official_oracle": oracle,
        "memory": memory,
    }


def _path_metrics(reference, candidate) -> dict:
    divergent = [
        step
        for step, (left, right) in enumerate(
            zip(reference["outputs"], candidate["outputs"], strict=True)
        )
        if not barriers._exact(left, right)
    ]
    return {
        "all_outputs_byte_identical": not divergent,
        "divergent_steps_zero_based": divergent,
        "conv_state_byte_identical": barriers._exact(
            reference["state"][0], candidate["state"][0]
        ),
        "recurrent_state_byte_identical": barriers._exact(
            reference["state"][1], candidate["state"][1]
        ),
        "critical_steps": {
            str(step): barriers._metrics(
                reference["outputs"][step], candidate["outputs"][step]
            )
            for step in BIT_CRITICAL_STEPS
        },
        "compile_trace_calls": candidate["compile_trace_calls"],
        "host_build_median_ms": candidate["host_build_median_ms"],
        "step_median_ms": candidate["step_median_ms"],
        "working_peak_delta_bytes": candidate["working_peak_delta_bytes"],
    }


def _first_divergence_stages(attention, inputs, replacement) -> dict:
    def diagnostic(inputs_, conv, recurrent, position):
        return functional.functional_kda_decode(
            attention,
            inputs_,
            conv,
            recurrent,
            position,
            diagnostics=True,
        )

    initial = functional.initial_kda_state(attention)
    with _norm(attention, replacement):
        compiled = mx.compile(diagnostic)
        traced = compiled(inputs, *initial, mx.array(0, mx.int32))
        mx.eval(*traced)

    eager_state = initial
    candidate_state = initial
    for step in range(STEPS):
        eager = diagnostic(inputs, *eager_state, mx.array(step, mx.int32))
        actual = compiled(
            inputs, *candidate_state, mx.array(step, mx.int32)
        )
        mx.eval(*eager, *actual)
        if not barriers._exact(eager[0], actual[0]):
            compiled_projection = mx.compile(
                lambda normalized: attention.o_proj(normalized)
            )
            projection_from_eager_norm = compiled_projection(eager[5])
            eager_projection_from_candidate_norm = attention.o_proj(actual[5])
            mx.eval(
                projection_from_eager_norm,
                eager_projection_from_candidate_norm,
            )
            return {
                "first_divergent_step_zero_based": step,
                "recurrent_output": barriers._metrics(eager[4], actual[4]),
                "gated_norm_output": barriers._metrics(eager[5], actual[5]),
                "final_projection_output": barriers._metrics(
                    eager[0], actual[0]
                ),
                "compiled_projection_from_eager_norm": barriers._metrics(
                    eager[0], projection_from_eager_norm
                ),
                "eager_projection_from_candidate_norm": barriers._metrics(
                    eager[0], eager_projection_from_candidate_norm
                ),
            }
        eager_state = eager[1:3]
        candidate_state = actual[1:3]
    return {"first_divergent_step_zero_based": None}


def _offset_gate(attention, inputs, replacement) -> dict:
    counter = {"calls": 0}

    def decoded(inputs_, conv, recurrent, position):
        counter["calls"] += 1
        return functional.functional_kda_decode(
            attention, inputs_, conv, recurrent, position
        )

    with _norm(attention, replacement):
        compiled = mx.compile(decoded)
        state = functional.initial_kda_state(attention)
        rows = []
        for offset in OFFSET_FIXTURES:
            result = compiled(inputs, *state, mx.array(offset, mx.int32))
            mx.eval(*result)
            rows.append(
                {
                    "offset": offset,
                    "position_after": int(result[3].item()),
                    "output_hash": functional._hash(result[0]),
                }
            )
    return {
        "compile_trace_calls": counter["calls"],
        "updates_exact": all(
            row["position_after"] == row["offset"] + 1 for row in rows
        ),
        "fixtures": rows,
    }


def _strided_gate(attention, replacement) -> bool:
    contiguous = functional._deterministic_input(attention.hidden_size)
    strided = functional._deterministic_input(
        attention.hidden_size, strided=True
    )
    state = functional.initial_kda_state(attention)
    with _norm(attention, replacement):
        decoded = mx.compile(
            lambda value, conv, recurrent, position: functional.functional_kda_decode(
                attention, value, conv, recurrent, position
            )
        )
        left = decoded(contiguous, *state, mx.array(0, mx.int32))
        right = decoded(strided, *state, mx.array(0, mx.int32))
        mx.eval(*left, *right)
    return all(
        barriers._exact(a, b) for a, b in zip(left[:3], right[:3], strict=True)
    )


def _invalid_state_gate(attention, inputs, replacement) -> dict:
    state = functional.initial_kda_state(attention)
    before = (functional._hash(state[0]), functional._hash(state[1]))
    rejected = 0
    candidates = (
        (state[0],),
        (state[0][:, :, :-1], state[1]),
        (state[0], state[1][:, :, :, :-1]),
    )
    with _norm(attention, replacement):
        for candidate in candidates:
            try:
                functional.validate_kda_state(attention, candidate)
            except (TypeError, ValueError):
                rejected += 1
    after = (functional._hash(state[0]), functional._hash(state[1]))
    return {
        "cases": len(candidates),
        "rejected_before_execution": rejected,
        "state_unchanged": before == after,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.time()
    report = inspect_checkpoint(args.model, require_server_ready=True)
    config = _make_config(json.loads((args.model / "config.json").read_text()))
    _progress("load_kda", layer=0)
    attention = functional._load_kda_attention(args.model, config)
    inputs = functional._deterministic_input(attention.hidden_size)
    gate = attention.g_b_proj(attention.g_a_proj(inputs)).reshape(
        1, 1, attention.num_heads, attention.head_dim
    )
    mx.eval(gate)
    _progress("sigmoid_formula")
    sigmoid = _sigmoid_formula_gate(gate)
    selected_mode = sigmoid["selected_mode"]

    original_norm = attention.o_norm
    _progress("arm_A")
    reference = _run_path(
        attention, inputs, original_norm, compiled=False
    )
    candidates = {}
    if selected_mode is not None:
        _progress("arm_B", mode=selected_mode)
        barrier_b = SigmoidBarrierNorm(original_norm, mode=selected_mode)
        candidate_b = _run_path(
            attention, inputs, barrier_b, compiled=True
        )
        candidates["B_sigmoid_only"] = {
            **_path_metrics(reference, candidate_b),
            "offset_gate": _offset_gate(attention, inputs, barrier_b),
            "strided_contiguous_byte_identical": _strided_gate(
                attention, barrier_b
            ),
            "invalid_state_gate": _invalid_state_gate(
                attention, inputs, barrier_b
            ),
        }
    else:
        candidates["B_sigmoid_only"] = {
            "executed": False,
            "reason": "no Metal sigmoid formula matched eager bits",
        }

    _progress("arm_C")
    barrier_c = FusedBarrierNorm(original_norm)
    candidate_c = _run_path(attention, inputs, barrier_c, compiled=True)
    candidates["C_fused_gated_rmsnorm"] = {
        **_path_metrics(reference, candidate_c),
        "offset_gate": _offset_gate(attention, inputs, barrier_c),
        "strided_contiguous_byte_identical": _strided_gate(attention, barrier_c),
        "invalid_state_gate": _invalid_state_gate(attention, inputs, barrier_c),
    }

    reference_build = reference["host_build_median_ms"]
    for row in candidates.values():
        if "host_build_median_ms" in row:
            row["host_build_reduction"] = 1.0 - (
                row["host_build_median_ms"] / reference_build
            )
            row["step_speedup_vs_eager"] = (
                reference["step_median_ms"] / row["step_median_ms"]
            )
    b = candidates["B_sigmoid_only"]
    c = candidates["C_fused_gated_rmsnorm"]
    b_exact = bool(b.get("all_outputs_byte_identical", False))
    c_exact = bool(c.get("all_outputs_byte_identical", False))
    layer0_retained = "B_sigmoid_only" if b_exact else (
        "C_fused_gated_rmsnorm" if c_exact else None
    )
    lower_tier_passed = layer0_retained is not None
    selected_formula = selected_mode if selected_mode is not None else -1
    all_kda_candidates = {}
    if b_exact:
        all_kda_candidates["B_sigmoid_only"] = _all_kda_gate(
            args.model,
            config,
            candidate_name="B_sigmoid_only",
            mode=selected_formula,
        )
    b_all = all_kda_candidates.get("B_sigmoid_only", {})
    b_all_exact = bool(
        b_all.get("all_34_layers_byte_identical", False)
        and b_all.get("representative_nonzero_snapshot_replay_exact", False)
    )
    if not b_all_exact and c_exact:
        all_kda_candidates["C_fused_gated_rmsnorm"] = _all_kda_gate(
            args.model,
            config,
            candidate_name="C_fused_gated_rmsnorm",
            mode=selected_formula,
        )
    c_all = all_kda_candidates.get("C_fused_gated_rmsnorm", {})
    c_all_exact = bool(
        c_all.get("all_34_layers_byte_identical", False)
        and c_all.get("representative_nonzero_snapshot_replay_exact", False)
    )
    retained = (
        "B_sigmoid_only"
        if b_all_exact
        else "C_fused_gated_rmsnorm" if c_all_exact else None
    )
    all_kda = all_kda_candidates.get(retained or layer0_retained or "", {
        "executed": False,
        "layers": list(EXPECTED_KDA),
        "reason": "neither candidate passed layer 0",
    })
    all_kda_exact = bool(
        all_kda.get("all_34_layers_byte_identical", False)
        and all_kda.get(
            "representative_nonzero_snapshot_replay_exact", False
        )
    )
    if all_kda_exact:
        full_model = _full_model_gate(
            args.model,
            report,
            candidate_name=retained,
            mode=selected_formula,
        )
    else:
        full_model = {
            "executed": False,
            "reason": "all-KDA exactness gate did not pass",
        }
    oracle_exact = bool(
        full_model.get("official_oracle", {}).get("first_16_match", False)
        and full_model.get("official_oracle", {}).get("full_128_match", False)
    )
    failure_localization = {}
    for candidate_name, candidate_gate in all_kda_candidates.items():
        failed = {
            layer_id: row
            for layer_id, row in candidate_gate["rows"].items()
            if not row["all_outputs_byte_identical"]
        }
        failure_localization[candidate_name] = {
            "failed_layers": [int(layer_id) for layer_id in failed],
            "failed_layer_count": len(failed),
            "final_conv_and_recurrent_state_exact": all(
                row["conv_state_byte_identical"]
                and row["recurrent_state_byte_identical"]
                for row in failed.values()
            ),
            "first_difference_precedes_gated_norm": all(
                not row["first_divergence_stages"]["recurrent_output"][
                    "byte_identical"
                ]
                for row in failed.values()
            ),
            "compiled_projection_from_eager_norm_exact": all(
                row["first_divergence_stages"][
                    "compiled_projection_from_eager_norm"
                ]["byte_identical"]
                for row in failed.values()
            ),
        }
    later = {
        "representative_layers": {
            "executed": bool(all_kda.get("executed", False)),
            "layers": list(REPRESENTATIVE_KDA_LAYERS),
            "all_exact": bool(
                all_kda.get(
                    "representative_nonzero_snapshot_replay_exact", False
                )
            ),
        },
        "all_kda_layers": all_kda,
        "full_model_oracles": full_model,
        "full_token_performance": {
            "executed": False,
            "reason": (
                "compiled KDA/full-token promotion requires an all-34-layer "
                "exact barrier; neither candidate passed"
            ),
            "correctness_claim": False,
            "performance_claim": False,
        },
    }
    artifact = {
        "schema": "glm53-exact-sigmoid-gate-metal-barrier-v1",
        "date": str(date.today()),
        "complete": True,
        "probe_only": True,
        "runtime_changes": False,
        "kernel_abi_changes": False,
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "sigmoid_formula": sigmoid,
        "reference": {
            "host_build_median_ms": reference["host_build_median_ms"],
            "step_median_ms": reference["step_median_ms"],
            "conv_state_hash": functional._hash(reference["state"][0]),
            "recurrent_state_hash": functional._hash(reference["state"][1]),
        },
        "candidates": candidates,
        "retained_candidate": retained,
        "layer0_retained_candidate": layer0_retained,
        "all_kda_candidates": all_kda_candidates,
        "failure_localization": failure_localization,
        "later_gates": later,
        "acceptance": {
            "actual_and_synthetic_sigmoid_bits_exact": selected_mode is not None,
            "B_layer0_64_step_exact": b_exact,
            "C_layer0_64_step_exact": c_exact,
            "layer0_candidate_exists": lower_tier_passed,
            "retained_candidate_exists": retained is not None,
            "globally_retained_candidate_exists": retained is not None,
            "all_34_kda_layers_exact": all_kda_exact,
            "representative_nonzero_snapshot_replay_exact": bool(
                all_kda.get(
                    "representative_nonzero_snapshot_replay_exact", False
                )
            ),
            "official_16_128_token_oracle_exact": oracle_exact,
            "official_oracle_skipped_after_all_kda_failure": bool(
                not all_kda_exact and not full_model["executed"]
            ),
            "retained_host_graph_build_reduction_at_least_40pct": bool(
                candidates.get(retained or "", {}).get(
                    "host_build_reduction", 0.0
                )
                >= HOST_BUILD_REDUCTION_GATE
            ),
            "retained_working_peak_at_most_64mib": bool(
                candidates.get(retained or "", {}).get(
                    "working_peak_delta_bytes", WORKING_PEAK_LIMIT + 1
                )
                <= WORKING_PEAK_LIMIT
            ),
            "runtime_unchanged": True,
        },
        "decision": (
            "retain_minimal_sigmoid_barrier_for_compiled_kda_integration"
            if lower_tier_passed and all_kda_exact and oracle_exact
            else "stop_exact_sigmoid_barrier_at_all_kda_recurrence_output"
        ),
        "elapsed_seconds": time.time() - started,
        "memory": functional._memory(),
    }
    _atomic_write(args.output, artifact)
    gc.collect()
    mx.clear_cache()
    _progress("complete", retained=retained)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
