#!/usr/bin/env python3
"""Localize exactness barriers in compiled GLM-5.3 KDA post-processing."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np

import probe_functional_stateful_decode_executable as functional
from glm53_flash_mlx.abi import MLX_VLM_REVISION
from glm53_flash_mlx.loader import _make_config
from glm53_flash_mlx.manifest import inspect_checkpoint


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-compiled-kda-numerical-barriers-20260902.json"
)
STEPS = 64
BIT_PATTERN_STEPS = (0, 1, 6, 7, 8, 63)
ARM_ABI = {
    "A": {"gated_rmsnorm": "eager", "final_projection": "eager"},
    "B": {"gated_rmsnorm": "compiled", "final_projection": "compiled"},
    "C": {"gated_rmsnorm": "eager-materialized", "final_projection": "compiled"},
    "D": {"gated_rmsnorm": "compiled-materialized", "final_projection": "eager"},
    "E": {"gated_rmsnorm": "eager-materialized", "final_projection": "eager"},
}
NORM_STAGES = (
    "input_f32",
    "square",
    "mean_square",
    "inverse_rms",
    "normalized",
    "weighted",
    "gate_f32",
    "sigmoid_gate",
    "gated_f32",
    "dtype_rounding",
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


def _np(value: mx.array, *, canonical_f32: bool = False) -> np.ndarray:
    if canonical_f32 and value.dtype == mx.bfloat16:
        value = value.astype(mx.float32)
    mx.eval(value)
    return np.ascontiguousarray(np.asarray(value))


def _hash(value: mx.array) -> str:
    array = _np(value, canonical_f32=True)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _exact(left: mx.array, right: mx.array) -> bool:
    mx.eval(left, right)
    return bool(mx.array_equal(left, right).item())


def _ordered_bits(bits: np.ndarray) -> np.ndarray:
    width = bits.dtype.itemsize * 8
    sign = np.array(1 << (width - 1), dtype=bits.dtype)
    mask = np.array((1 << width) - 1, dtype=bits.dtype)
    return np.where((bits & sign) != 0, (~bits) & mask, bits | sign).astype(
        np.uint64
    )


def _bits(value: mx.array) -> tuple[np.ndarray, int]:
    if value.dtype == mx.bfloat16:
        array = _np(value.view(mx.uint16))
        return array.astype(np.uint16, copy=False), 16
    if value.dtype == mx.float32:
        array = _np(value).view(np.uint32)
        return array, 32
    raise TypeError(f"ULP evidence only supports BF16/FP32, got {value.dtype}")


def _metrics(reference: mx.array, actual: mx.array) -> dict:
    reference_f = _np(reference.astype(mx.float32))
    actual_f = _np(actual.astype(mx.float32))
    delta = actual_f - reference_f
    different = np.flatnonzero(delta.reshape(-1) != 0)
    row = {
        "byte_identical": _exact(reference, actual),
        "dtype": str(reference.dtype),
        "shape": list(reference.shape),
        "different_elements": int(different.size),
        "elements": int(delta.size),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "relative_l2": float(
            np.linalg.norm(delta)
            / max(float(np.linalg.norm(reference_f)), 1e-30)
        ),
        "reference_hash": hashlib.sha256(reference_f.tobytes()).hexdigest(),
        "actual_hash": hashlib.sha256(actual_f.tobytes()).hexdigest(),
        "first_difference": None,
    }
    if different.size:
        index = int(different[0])
        reference_bits, width = _bits(reference)
        actual_bits, _ = _bits(actual)
        reference_flat = reference_bits.reshape(-1)
        actual_flat = actual_bits.reshape(-1)
        ordered_reference = _ordered_bits(reference_flat)
        ordered_actual = _ordered_bits(actual_flat)
        ulp = abs(
            int(ordered_actual[index]) - int(ordered_reference[index])
        )
        row["first_difference"] = {
            "flat_index": index,
            "reference_value_f32": float(reference_f.reshape(-1)[index]),
            "actual_value_f32": float(actual_f.reshape(-1)[index]),
            "reference_bits": f"0x{int(reference_flat[index]):0{width // 4}x}",
            "actual_bits": f"0x{int(actual_flat[index]):0{width // 4}x}",
            "ulp_distance": ulp,
        }
    return row


def gated_rmsnorm_stages(attention, hidden: mx.array, gate: mx.array):
    norm = attention.o_norm
    dtype = hidden.dtype
    input_f32 = hidden.astype(mx.float32)
    square = input_f32 * input_f32
    mean_square = square.mean(-1, keepdims=True)
    inverse_rms = mx.rsqrt(mean_square + norm.eps)
    normalized = input_f32 * inverse_rms
    weighted = norm.weight.astype(mx.float32) * normalized
    gate_f32 = gate.astype(mx.float32)
    sigmoid_gate = mx.sigmoid(gate_f32)
    gated_f32 = weighted * sigmoid_gate
    dtype_rounding = gated_f32.astype(dtype)
    return (
        input_f32,
        square,
        mean_square,
        inverse_rms,
        normalized,
        weighted,
        gate_f32,
        sigmoid_gate,
        gated_f32,
        dtype_rounding,
    )


def _compiled(callable_, counter: dict[str, int], name: str):
    def traced(*args):
        counter[name] += 1
        return callable_(*args)

    return mx.compile(traced)


def _capture_recurrence(attention, inputs):
    state = functional.initial_kda_state(attention)
    gate_input = attention.g_a_proj(inputs)
    gate = attention.g_b_proj(gate_input).reshape(
        1, 1, attention.num_heads, attention.head_dim
    )
    recurrence = []
    state_hashes = []
    for step in range(STEPS):
        result = functional.functional_kda_decode(
            attention,
            inputs,
            *state,
            mx.array(step, mx.int32),
            diagnostics=True,
        )
        mx.eval(*result, gate)
        recurrence.append(result[4])
        state = result[1:3]
        state_hashes.append((_hash(state[0]), _hash(state[1])))
    return recurrence, gate, state, state_hashes


def _arm_outputs(attention, recurrence, gate):
    counters = {name: 0 for name in ("norm", "projection", "composite", "stages")}
    compiled_norm = _compiled(
        lambda hidden, gate_: attention.o_norm(hidden, gate_).reshape(1, 1, -1),
        counters,
        "norm",
    )
    compiled_projection = _compiled(
        lambda normalized: attention.o_proj(normalized),
        counters,
        "projection",
    )

    def composite(hidden, gate_):
        normalized = attention.o_norm(hidden, gate_).reshape(1, 1, -1)
        return normalized, attention.o_proj(normalized)

    compiled_composite = _compiled(composite, counters, "composite")
    compiled_stages = _compiled(
        lambda hidden, gate_: gated_rmsnorm_stages(attention, hidden, gate_),
        counters,
        "stages",
    )

    rows = {name: [] for name in ARM_ABI}
    fixture_evidence = {}
    stage_first_divergence = None
    for step, recurrent_output in enumerate(recurrence):
        eager_stages = gated_rmsnorm_stages(attention, recurrent_output, gate)
        mx.eval(*eager_stages)
        eager_norm = eager_stages[-1].reshape(1, 1, -1)
        eager_projection = attention.o_proj(eager_norm)
        mx.eval(eager_projection)

        compiled_norm_value, compiled_projection_value = compiled_composite(
            recurrent_output, gate
        )
        mx.eval(compiled_norm_value, compiled_projection_value)

        # C: materialize the eager numerical barrier before compiled projection.
        anchored_eager_norm = mx.array(eager_norm)
        mx.eval(anchored_eager_norm)
        c_output = compiled_projection(anchored_eager_norm)
        mx.eval(c_output)

        # D: materialize compiled norm and return projection to eager execution.
        anchored_compiled_norm = compiled_norm(recurrent_output, gate)
        mx.eval(anchored_compiled_norm)
        d_output = attention.o_proj(anchored_compiled_norm)
        mx.eval(d_output)

        # E is a separately evaluated eager/eager control.
        e_norm = attention.o_norm(recurrent_output, gate).reshape(1, 1, -1)
        mx.eval(e_norm)
        e_output = attention.o_proj(e_norm)
        mx.eval(e_output)

        values = {
            "A": (eager_norm, eager_projection),
            "B": (compiled_norm_value, compiled_projection_value),
            "C": (anchored_eager_norm, c_output),
            "D": (anchored_compiled_norm, d_output),
            "E": (e_norm, e_output),
        }
        for arm, (norm_value, output) in values.items():
            rows[arm].append(
                {
                    "norm_exact": _exact(eager_norm, norm_value),
                    "output_exact": _exact(eager_projection, output),
                    "norm_hash": _hash(norm_value),
                    "output_hash": _hash(output),
                }
            )

        compiled_stage_values = compiled_stages(recurrent_output, gate)
        mx.eval(*compiled_stage_values)
        stage_metrics = {
            name: _metrics(reference, actual)
            for name, reference, actual in zip(
                NORM_STAGES,
                eager_stages,
                compiled_stage_values,
                strict=True,
            )
        }
        first = next(
            (
                name
                for name in NORM_STAGES
                if not stage_metrics[name]["byte_identical"]
            ),
            None,
        )
        if stage_first_divergence is None and first is not None:
            stage_first_divergence = {"step": step, "stage": first}

        if step in BIT_PATTERN_STEPS:
            fixture_evidence[str(step)] = {
                "recurrent_output_hash": _hash(recurrent_output),
                "gate_hash": _hash(gate),
                "arms": {
                    arm: {
                        "norm": _metrics(eager_norm, value[0]),
                        "final_projection": _metrics(
                            eager_projection, value[1]
                        ),
                    }
                    for arm, value in values.items()
                },
                "gated_rmsnorm_stages": stage_metrics,
                "first_divergent_norm_stage": first,
            }

    summaries = {}
    for arm, arm_rows in rows.items():
        divergent_norm_steps = [
            index for index, row in enumerate(arm_rows) if not row["norm_exact"]
        ]
        divergent_final_steps = [
            index for index, row in enumerate(arm_rows) if not row["output_exact"]
        ]
        first_norm = next(
            (index for index, row in enumerate(arm_rows) if not row["norm_exact"]),
            None,
        )
        first_output = next(
            (
                index
                for index, row in enumerate(arm_rows)
                if not row["output_exact"]
            ),
            None,
        )
        summaries[arm] = {
            **ARM_ABI[arm],
            "steps": STEPS,
            "all_norm_byte_identical": first_norm is None,
            "all_final_projection_byte_identical": first_output is None,
            "first_norm_divergence_step_zero_based": first_norm,
            "first_final_divergence_step_zero_based": first_output,
            "exact_norm_steps": sum(row["norm_exact"] for row in arm_rows),
            "exact_final_steps": sum(row["output_exact"] for row in arm_rows),
            "divergent_norm_steps_zero_based": divergent_norm_steps,
            "divergent_final_steps_zero_based": divergent_final_steps,
        }
    return {
        "arms": summaries,
        "compile_trace_calls": counters,
        "bit_pattern_fixtures": fixture_evidence,
        "first_compiled_norm_stage_divergence": stage_first_divergence,
    }


def _full_compiled_confirmation(attention, inputs):
    counter = {"calls": 0}

    def decoded(inputs_, conv, recurrent, position):
        counter["calls"] += 1
        return functional.functional_kda_decode(
            attention, inputs_, conv, recurrent, position
        )

    compiled = mx.compile(decoded)
    eager_state = functional.initial_kda_state(attention)
    compiled_state = functional.initial_kda_state(attention)
    first_output = None
    first_conv = None
    first_recurrent = None
    for step in range(STEPS):
        position = mx.array(step, mx.int32)
        eager = functional.functional_kda_decode(
            attention, inputs, *eager_state, position
        )
        actual = compiled(inputs, *compiled_state, position)
        mx.eval(*eager, *actual)
        if first_output is None and not _exact(eager[0], actual[0]):
            first_output = step
        if first_conv is None and not _exact(eager[1], actual[1]):
            first_conv = step
        if first_recurrent is None and not _exact(eager[2], actual[2]):
            first_recurrent = step
        eager_state = eager[1:3]
        compiled_state = actual[1:3]
    return {
        "compile_trace_calls": counter["calls"],
        "first_output_divergence_step_zero_based": first_output,
        "first_conv_divergence_step_zero_based": first_conv,
        "first_recurrent_divergence_step_zero_based": first_recurrent,
        "final_conv_state_byte_identical": _exact(
            eager_state[0], compiled_state[0]
        ),
        "final_recurrent_state_byte_identical": _exact(
            eager_state[1], compiled_state[1]
        ),
    }


def _decision(result: dict) -> dict:
    arms = result["arms"]
    projection_independent = not arms["C"][
        "all_final_projection_byte_identical"
    ]
    norm_is_only_blocker = bool(
        arms["C"]["all_final_projection_byte_identical"]
        and not arms["D"]["all_final_projection_byte_identical"]
        and arms["E"]["all_final_projection_byte_identical"]
    )
    blocker_count = 1 if norm_is_only_blocker else (2 if projection_independent else 3)
    return {
        "gated_rmsnorm_is_only_observed_blocker": norm_is_only_blocker,
        "final_projection_is_independent_blocker": projection_independent,
        "unobserved_third_blocker": not arms["E"][
            "all_final_projection_byte_identical"
        ],
        "minimum_observed_blocker_count": blocker_count,
        "first_numerical_barrier": result[
            "first_compiled_norm_stage_divergence"
        ],
        "next_step": (
            "implement_probe_exact_gated_rmsnorm_primitive_with_eager_sigmoid_order"
            if norm_is_only_blocker
            else "stop_numerical_barrier_path"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.time()
    report = inspect_checkpoint(args.model, require_server_ready=True)
    raw_config = json.loads((args.model / "config.json").read_text())
    config = _make_config(raw_config)
    _progress("load_kda", layer=functional.KDA_LAYER)
    attention = functional._load_kda_attention(args.model, config)
    inputs = functional._deterministic_input(attention.hidden_size)
    _progress("capture_recurrence", steps=STEPS)
    recurrence, gate, final_state, state_hashes = _capture_recurrence(
        attention, inputs
    )
    _progress("anchor_arms", arms=list(ARM_ABI))
    localization = _arm_outputs(attention, recurrence, gate)
    _progress("full_compiled_confirmation")
    full = _full_compiled_confirmation(attention, inputs)
    decision = _decision(localization)
    acceptance = {
        "A_control_exact": localization["arms"]["A"][
            "all_final_projection_byte_identical"
        ],
        "B_reproduces_known_divergence": not localization["arms"]["B"][
            "all_final_projection_byte_identical"
        ],
        "C_eager_norm_anchor_exact": localization["arms"]["C"][
            "all_final_projection_byte_identical"
        ],
        "D_compiled_norm_remains_divergent": not localization["arms"]["D"][
            "all_final_projection_byte_identical"
        ],
        "E_control_exact": localization["arms"]["E"][
            "all_final_projection_byte_identical"
        ],
        "all_compile_signatures_single_trace": all(
            value == 1 for value in localization["compile_trace_calls"].values()
        ) and full["compile_trace_calls"] == 1,
        "full_compiled_state_exact": bool(
            full["final_conv_state_byte_identical"]
            and full["final_recurrent_state_byte_identical"]
        ),
        "runtime_unchanged": True,
    }
    artifact = {
        "schema": "glm53-compiled-kda-numerical-barriers-v1",
        "date": str(date.today()),
        "complete": True,
        "probe_only": True,
        "runtime_changes": False,
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "layer": functional.KDA_LAYER,
        "steps": STEPS,
        "bit_pattern_steps_zero_based": list(BIT_PATTERN_STEPS),
        "arm_abi": ARM_ABI,
        "localization": localization,
        "full_compiled_confirmation": full,
        "final_eager_state": {
            "conv_hash": _hash(final_state[0]),
            "recurrent_hash": _hash(final_state[1]),
            "per_step_hash_count": len(state_hashes),
        },
        "decision": decision,
        "acceptance": acceptance,
        "elapsed_seconds": time.time() - started,
        "memory": functional._memory(),
    }
    _atomic_write(args.output, artifact)
    _progress("complete", next_step=decision["next_step"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
