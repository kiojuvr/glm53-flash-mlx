#!/usr/bin/env python3
"""Localize the all-layer compiled KDA recurrent readout exactness barrier."""

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
import mlx.nn as nn
import numpy as np

import localize_compiled_kda_numerical_barriers as metrics
import probe_exact_sigmoid_gate_metal_barrier as sigmoid_probe
import probe_functional_stateful_decode_executable as functional
from glm53_flash_mlx.abi import MLX_VLM_REVISION
from glm53_flash_mlx.loader import _make_config
from glm53_flash_mlx.manifest import inspect_checkpoint


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-compiled-kda-recurrent-readout-barrier-20260902.json"
)
FAILED_LAYERS = (10, 22, 25, 42)
CONTROL_LAYERS = (0, 20, 44)
LAYERS = CONTROL_LAYERS + FAILED_LAYERS
STEPS = 64
SIGMOID_MODE = 7
HOST_BUILD_REDUCTION_GATE = 0.40
OFFICIAL_HEAD_DIM = 128

PREFIX_STAGE_NAMES = (
    "q_projection",
    "k_projection",
    "v_projection",
    "forget_a_projection",
    "gate_a_projection",
    "beta_logits",
    "conv_output",
    "conv_state",
    "raw_q_bf16",
    "q_input_fp32",
    "q_square_fp32",
    "q_sum_fp32",
    "q_inverse_norm_fp32",
    "q_l2normalized_fp32",
    "q_scaled_fp32",
    "normalized_q",
    "normalized_k",
    "value",
    "forget_a",
    "decay_g",
    "beta",
    "output_gate",
)
FULL_STAGE_NAMES = PREFIX_STAGE_NAMES + (
    "recurrent_output_bf16",
    "updated_recurrent_state_fp32",
    "gated_norm_input_bf16",
    "final_projection_bf16",
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


def _hash(value: mx.array) -> str:
    return functional._hash(value)


def _exact(left: mx.array, right: mx.array) -> bool:
    return metrics._exact(left, right)


def _memory() -> dict[str, int]:
    mx.synchronize()
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }


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


def _prefix(attention, inputs, conv_state):
    """Functional KDA prefix through the recurrence kernel inputs."""
    from mlx_vlm.models.gated_delta import compute_g_safe
    from mlx_vlm.models.glm5_next.language import _l2norm

    q_out = attention.q_proj(inputs)
    k_out = attention.k_proj(inputs)
    v_out = attention.v_proj(inputs)
    fa_out = attention.forget_gate.f_a_proj(inputs)
    ga_out = attention.g_a_proj(inputs)
    beta_logits = attention.b_proj(inputs)
    mixed = mx.concatenate([q_out, k_out, v_out], axis=-1)
    conv_input = mx.concatenate([conv_state, mixed], axis=1)
    next_conv = mx.contiguous(
        conv_input[:, -(attention.conv_kernel_size - 1) :, :],
        allow_col_major=False,
    )
    conv_output = nn.silu(attention.conv1d(conv_input))
    raw_q, k, value = mx.split(
        conv_output,
        [attention.qkv_dim, 2 * attention.qkv_dim],
        axis=-1,
    )
    raw_q = raw_q.reshape(1, 1, attention.num_heads, attention.head_dim)
    k = k.reshape(1, 1, attention.num_heads, attention.head_dim)
    value = value.reshape(1, 1, attention.num_heads, attention.head_dim)
    forget_a = attention.forget_gate.f_b_proj(fa_out).reshape(
        1, 1, attention.num_heads, attention.head_dim
    )
    in_dtype = raw_q.dtype
    q_input = raw_q.astype(mx.float32)
    q_square = q_input * q_input
    q_sum = q_square.sum(axis=-1, keepdims=True)
    q_inverse_norm = mx.rsqrt(q_sum + 1e-6)
    q_l2normalized = q_input * q_inverse_norm
    q_scaled = q_l2normalized * (attention.head_dim**-0.5)
    q = q_scaled.astype(in_dtype)
    k = _l2norm(k.astype(mx.float32)).astype(in_dtype)
    forget = attention.forget_gate
    decay_g = compute_g_safe(
        forget.A_log.reshape(attention.num_heads, 1),
        forget_a,
        forget.dt_bias.reshape(attention.num_heads, attention.head_dim),
        forget.safe_gate_lower_bound,
    )
    beta = mx.sigmoid(beta_logits)
    output_gate = attention.g_b_proj(ga_out).reshape(
        1, 1, attention.num_heads, attention.head_dim
    )
    return (
        q_out,
        k_out,
        v_out,
        fa_out,
        ga_out,
        beta_logits,
        conv_output,
        next_conv,
        raw_q,
        q_input,
        q_square,
        q_sum,
        q_inverse_norm,
        q_l2normalized,
        q_scaled,
        q,
        k,
        value,
        forget_a,
        decay_g,
        beta,
        output_gate,
    )


def _recurrence(prefix, recurrent_state):
    from mlx_vlm.models.gated_delta import gated_delta_kernel

    q = prefix[PREFIX_STAGE_NAMES.index("normalized_q")]
    k = prefix[PREFIX_STAGE_NAMES.index("normalized_k")]
    value = prefix[PREFIX_STAGE_NAMES.index("value")]
    decay_g = prefix[PREFIX_STAGE_NAMES.index("decay_g")]
    beta = prefix[PREFIX_STAGE_NAMES.index("beta")]
    return gated_delta_kernel(q, k, value, decay_g, beta, recurrent_state)


def _tail(attention, recurrent_output, output_gate):
    normalized = attention.o_norm(recurrent_output, output_gate)
    normalized = normalized.reshape(1, 1, -1)
    return normalized, attention.o_proj(normalized)


def _full(attention, inputs, conv_state, recurrent_state):
    prefix = _prefix(attention, inputs, conv_state)
    recurrent_output, next_state = _recurrence(prefix, recurrent_state)
    normalized, output = _tail(attention, recurrent_output, prefix[-1])
    return (*prefix, recurrent_output, next_state, normalized, output)


def _compile(callable_, counter: dict[str, int], name: str):
    def traced(*args):
        counter[name] += 1
        return callable_(*args)

    return mx.compile(traced)


def _stage_metrics(reference, actual, names) -> dict:
    return {
        name: metrics._metrics(left, right)
        for name, left, right in zip(names, reference, actual, strict=True)
    }


def _first_difference(stage_rows: dict) -> str | None:
    return next(
        (
            name
            for name, row in stage_rows.items()
            if not row["byte_identical"]
        ),
        None,
    )


def _array_signature(value: mx.array) -> dict:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "strides_api_available": False,
        "layout_flags_api_available": False,
        "hash": _hash(value),
    }


def _capture_fp32_readout_inputs(prefix, state) -> dict:
    return {
        "normalized_q": _array_signature(
            prefix[PREFIX_STAGE_NAMES.index("normalized_q")]
        ),
        "updated_state": _array_signature(state),
    }


def _run_layer(model_path: Path, config, layer_id: int) -> dict:
    _progress("layer", layer=layer_id)
    attention = sigmoid_probe._load_kda_attention_layer(
        model_path, config, layer_id
    )
    inputs = functional._deterministic_input(attention.hidden_size)
    counters = {name: 0 for name in ("full", "prefix", "recurrence", "tail")}
    with _exact_norm(attention):
        compiled_full = _compile(
            lambda x, conv, state: _full(attention, x, conv, state),
            counters,
            "full",
        )
        compiled_prefix = _compile(
            lambda x, conv: _prefix(attention, x, conv),
            counters,
            "prefix",
        )
        compiled_recurrence = _compile(
            lambda *values: _recurrence(values[:-1], values[-1]),
            counters,
            "recurrence",
        )
        compiled_tail = _compile(
            lambda recurrent, gate: _tail(attention, recurrent, gate),
            counters,
            "tail",
        )

        states = {
            arm: functional.initial_kda_state(attention)
            for arm in ("A", "B", "C", "D", "E")
        }
        arm_exact = {arm: [] for arm in states}
        first_divergence = None
        eager_build = []
        compiled_build = []

        for step in range(STEPS):
            # A: eager prefix + eager recurrence + eager tail.
            started = time.perf_counter_ns()
            prefix_a = _prefix(attention, inputs, states["A"][0])
            recurrent_a, state_a = _recurrence(prefix_a, states["A"][1])
            norm_a, output_a = _tail(attention, recurrent_a, prefix_a[-1])
            eager_build.append((time.perf_counter_ns() - started) / 1e6)
            mx.eval(*prefix_a, recurrent_a, state_a, norm_a, output_a)

            # B: full outer compile with exact sigmoid barrier.
            started = time.perf_counter_ns()
            full_b = compiled_full(inputs, *states["B"])
            compiled_build.append((time.perf_counter_ns() - started) / 1e6)
            mx.eval(*full_b)
            prefix_b = full_b[: len(PREFIX_STAGE_NAMES)]
            recurrent_b, state_b, norm_b, output_b = full_b[-4:]

            # C: compiled prefix is explicitly materialized, recurrence/tail eager.
            prefix_c = compiled_prefix(inputs, states["C"][0])
            mx.eval(*prefix_c)
            recurrent_c, state_c = _recurrence(prefix_c, states["C"][1])
            mx.eval(recurrent_c, state_c)
            norm_c, output_c = _tail(attention, recurrent_c, prefix_c[-1])
            mx.eval(norm_c, output_c)

            # D: eager prefix/state anchor, compiled recurrent kernel wrapper.
            prefix_d = _prefix(attention, inputs, states["D"][0])
            mx.eval(*prefix_d)
            recurrent_d, state_d = compiled_recurrence(
                *prefix_d, states["D"][1]
            )
            mx.eval(recurrent_d, state_d)
            norm_d, output_d = _tail(attention, recurrent_d, prefix_d[-1])
            mx.eval(norm_d, output_d)

            # E: eager recurrence, materialized, then compiled exact tail.
            prefix_e = _prefix(attention, inputs, states["E"][0])
            recurrent_e, state_e = _recurrence(prefix_e, states["E"][1])
            mx.eval(*prefix_e, recurrent_e, state_e)
            norm_e, output_e = compiled_tail(recurrent_e, prefix_e[-1])
            mx.eval(norm_e, output_e)

            values = {
                "A": (prefix_a, recurrent_a, state_a, norm_a, output_a),
                "B": (prefix_b, recurrent_b, state_b, norm_b, output_b),
                "C": (prefix_c, recurrent_c, state_c, norm_c, output_c),
                "D": (prefix_d, recurrent_d, state_d, norm_d, output_d),
                "E": (prefix_e, recurrent_e, state_e, norm_e, output_e),
            }
            for arm, value in values.items():
                arm_exact[arm].append(
                    {
                        "prefix": all(
                            _exact(left, right)
                            for left, right in zip(
                                prefix_a, value[0], strict=True
                            )
                        ),
                        "recurrent_output": _exact(recurrent_a, value[1]),
                        "updated_state": _exact(state_a, value[2]),
                        "gated_norm": _exact(norm_a, value[3]),
                        "final_output": _exact(output_a, value[4]),
                    }
                )

            if first_divergence is None and not arm_exact["B"][-1][
                "final_output"
            ]:
                stages_b = _stage_metrics(
                    (*prefix_a, recurrent_a, state_a, norm_a, output_a),
                    (*prefix_b, recurrent_b, state_b, norm_b, output_b),
                    FULL_STAGE_NAMES,
                )
                stages_c = _stage_metrics(
                    (*prefix_a, recurrent_a, state_a, norm_a, output_a),
                    (*prefix_c, recurrent_c, state_c, norm_c, output_c),
                    FULL_STAGE_NAMES,
                )
                first_divergence = {
                    "step_zero_based": step,
                    "B_full_compiled": stages_b,
                    "C_materialized_prefix": stages_c,
                    "D_compiled_recurrence": _stage_metrics(
                        (recurrent_a, state_a, norm_a, output_a),
                        (recurrent_d, state_d, norm_d, output_d),
                        FULL_STAGE_NAMES[-4:],
                    ),
                    "E_compiled_tail": _stage_metrics(
                        (recurrent_a, state_a, norm_a, output_a),
                        (recurrent_e, state_e, norm_e, output_e),
                        FULL_STAGE_NAMES[-4:],
                    ),
                    "first_B_stage": _first_difference(stages_b),
                    "first_C_stage": _first_difference(stages_c),
                    "readout_inputs": _capture_fp32_readout_inputs(
                        prefix_a, state_a
                    ),
                    "state_before": _array_signature(states["A"][1]),
                    "state_after": _array_signature(state_a),
                }

            states = {
                "A": (prefix_a[7], state_a),
                "B": (prefix_b[7], state_b),
                "C": (prefix_c[7], state_c),
                "D": (prefix_d[7], state_d),
                "E": (prefix_e[7], state_e),
            }

    summaries = {}
    for arm, rows in arm_exact.items():
        summaries[arm] = {
            "steps": STEPS,
            "all_prefix_exact": all(row["prefix"] for row in rows),
            "all_recurrent_outputs_exact": all(
                row["recurrent_output"] for row in rows
            ),
            "all_updated_states_exact": all(
                row["updated_state"] for row in rows
            ),
            "all_gated_norm_exact": all(row["gated_norm"] for row in rows),
            "all_final_outputs_exact": all(row["final_output"] for row in rows),
            "first_recurrent_divergence": next(
                (
                    step
                    for step, row in enumerate(rows)
                    if not row["recurrent_output"]
                ),
                None,
            ),
            "first_state_divergence": next(
                (
                    step
                    for step, row in enumerate(rows)
                    if not row["updated_state"]
                ),
                None,
            ),
            "first_output_divergence": next(
                (
                    step
                    for step, row in enumerate(rows)
                    if not row["final_output"]
                ),
                None,
            ),
        }

    eager_median = float(statistics.median(eager_build[1:]))
    compiled_median = float(statistics.median(compiled_build[1:]))
    result = {
        "layer": layer_id,
        "classification": "failed" if layer_id in FAILED_LAYERS else "control",
        "arms": summaries,
        "first_divergence": first_divergence,
        "compile_trace_calls": counters,
        "eager_host_build_median_ms": eager_median,
        "compiled_host_build_median_ms": compiled_median,
        "host_build_reduction": 1.0 - compiled_median / eager_median,
        "final_state_hashes": {
            arm: [_hash(value) for value in state]
            for arm, state in states.items()
        },
    }
    del attention, states
    gc.collect()
    mx.clear_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.time()
    report = inspect_checkpoint(args.model, require_server_ready=True)
    config = _make_config(json.loads((args.model / "config.json").read_text()))
    rows = {
        str(layer_id): _run_layer(args.model, config, layer_id)
        for layer_id in LAYERS
    }
    failed = [rows[str(layer)] for layer in FAILED_LAYERS]
    controls = [rows[str(layer)] for layer in CONTROL_LAYERS]
    arm_c_exact = all(
        row["arms"]["C"]["all_final_outputs_exact"] for row in failed
    )
    arm_d_exact = all(
        row["arms"]["D"]["all_final_outputs_exact"] for row in failed
    )
    arm_e_exact = all(
        row["arms"]["E"]["all_final_outputs_exact"] for row in failed
    )
    compiled_prefix_exact = all(
        row["arms"]["C"]["all_prefix_exact"] for row in failed
    )
    recurrence_wrapper_exact = all(
        row["arms"]["D"]["all_recurrent_outputs_exact"]
        and row["arms"]["D"]["all_updated_states_exact"]
        for row in failed
    )
    pre_scale_stages = PREFIX_STAGE_NAMES[
        : PREFIX_STAGE_NAMES.index("q_scaled_fp32")
    ]
    q_scale_is_first_difference = all(
        row["first_divergence"]["first_B_stage"] == "q_scaled_fp32"
        and all(
            row["first_divergence"]["B_full_compiled"][stage][
                "byte_identical"
            ]
            for stage in pre_scale_stages
        )
        for row in failed
    )
    q_scale_changes_bf16_boundary = all(
        not row["first_divergence"]["B_full_compiled"]["q_scaled_fp32"][
            "byte_identical"
        ]
        and not row["first_divergence"]["B_full_compiled"]["normalized_q"][
            "byte_identical"
        ]
        for row in failed
    )
    final_state_exact_all_arms = all(
        len(
            {
                tuple(hashes)
                for hashes in row["final_state_hashes"].values()
            }
        )
        == 1
        for row in rows.values()
    )
    q_scale = np.float32(OFFICIAL_HEAD_DIM**-0.5)
    artifact = {
        "schema": "glm53-compiled-kda-recurrent-readout-barrier-v1",
        "date": str(date.today()),
        "complete": True,
        "probe_only": True,
        "runtime_changes": False,
        "kernel_abi_changes": False,
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "layers": list(LAYERS),
        "failed_layers_from_sigmoid_gate": list(FAILED_LAYERS),
        "control_layers": list(CONTROL_LAYERS),
        "steps": STEPS,
        "q_scale_contract": {
            "head_dim": OFFICIAL_HEAD_DIM,
            "python_value": OFFICIAL_HEAD_DIM**-0.5,
            "fp32_value": float(q_scale),
            "fp32_bits_hex": f"0x{q_scale.view(np.uint32).item():08x}",
            "eager_order": "q_l2normalized_fp32 * fp32(head_dim ** -0.5)",
        },
        "arm_contract": {
            "A": "eager prefix + eager recurrence + eager exact-sigmoid tail",
            "B": "compiled full KDA + exact sigmoid barrier",
            "C": "compiled prefix materialized + eager recurrence/tail",
            "D": "eager prefix/state anchor + compiled recurrence wrapper + eager tail",
            "E": "eager recurrence materialized + compiled exact-sigmoid tail",
        },
        "rows": rows,
        "acceptance": {
            "control_layers_all_arms_exact": all(
                all(
                    arm["all_final_outputs_exact"]
                    and arm["all_updated_states_exact"]
                    for arm in row["arms"].values()
                )
                for row in controls
            ),
            "failed_layers_full_compile_reproduces_divergence": all(
                not row["arms"]["B"]["all_final_outputs_exact"]
                for row in failed
            ),
            "failed_layers_updated_state_exact_every_step": all(
                row["arms"]["B"]["all_updated_states_exact"]
                for row in failed
            ),
            "compiled_prefix_materialization_restores_exactness": arm_c_exact,
            "compiled_prefix_tensors_are_exact": compiled_prefix_exact,
            "compiled_recurrence_with_eager_inputs_exact": recurrence_wrapper_exact,
            "compiled_tail_with_exact_sigmoid_exact": arm_e_exact,
            "q_scale_is_first_difference_in_all_failed_layers": (
                q_scale_is_first_difference
            ),
            "q_scale_difference_crosses_bf16_boundary": (
                q_scale_changes_bf16_boundary
            ),
            "final_state_exact_across_all_arms": final_state_exact_all_arms,
            "all_compiled_callables_trace_once": all(
                all(count == 1 for count in row["compile_trace_calls"].values())
                for row in rows.values()
            ),
            "host_build_reduction_at_least_40pct": min(
                row["host_build_reduction"] for row in rows.values()
            )
            >= HOST_BUILD_REDUCTION_GATE,
            "runtime_unchanged": True,
        },
        "decision": (
            "retain_materialization_identity_barrier"
            if arm_c_exact and compiled_prefix_exact
            else "probe_exact_q_scale_metal_barrier"
            if (
                not compiled_prefix_exact
                and q_scale_is_first_difference
                and recurrence_wrapper_exact
                and arm_e_exact
            )
            else "probe_exact_recurrent_readout_primitive"
            if not arm_d_exact and arm_e_exact
            else "stop_mlx_compiled_kda_barrier_stacking"
        ),
        "full_model_oracle": {
            "executed": False,
            "reason": "localization commit only; requires a 34-layer exact candidate",
        },
        "elapsed_seconds": time.time() - started,
        "memory": _memory(),
    }
    _atomic_write(args.output, artifact)
    _progress("complete", decision=artifact["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
