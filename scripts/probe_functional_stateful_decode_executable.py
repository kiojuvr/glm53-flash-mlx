#!/usr/bin/env python3
"""Probe fixed-signature functional stateful decode compilation, tier by tier.

The probe deliberately loads only the checkpoint tensors needed by the active
tier.  A failed tier records negative evidence and prevents every more costly
tier from loading or compiling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np

from glm53_flash_mlx.abi import MLX_VLM_REVISION
from glm53_flash_mlx.loader import _make_config
from glm53_flash_mlx.manifest import inspect_checkpoint
from glm53_flash_mlx.patch import apply_runtime_patch


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-functional-stateful-decode-executable-20260902.json"
)
KDA_LAYER = 0
KDA_STEPS = 64
KDA_OFFSET_FIXTURES = (0, 1, 255, 256, 2048)
KDA_HOST_BUILD_REDUCTION_GATE = 0.50
FULL_COMPILE_BUDGET_SECONDS = 15 * 60


@dataclass(frozen=True)
class KDAStateSchema:
    conv_shape: tuple[int, ...]
    conv_dtype: str
    recurrent_shape: tuple[int, ...]
    recurrent_dtype: str


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


def _array(value: mx.array) -> np.ndarray:
    mx.eval(value)
    # NumPy has no portable PEP-3118 view for MLX bfloat16.  Hash a canonical
    # FP32 representation; byte identity itself is checked on-device below.
    if value.dtype == mx.bfloat16:
        value = value.astype(mx.float32)
        mx.eval(value)
    return np.ascontiguousarray(np.asarray(value))


def _hash(value: mx.array) -> str:
    return hashlib.sha256(_array(value).tobytes()).hexdigest()


def _exact(left: mx.array, right: mx.array) -> bool:
    mx.eval(left, right)
    return bool(mx.array_equal(left, right).item())


def _metrics(left: mx.array, right: mx.array) -> dict:
    left_f = _array(left.astype(mx.float32))
    right_f = _array(right.astype(mx.float32))
    delta = right_f - left_f
    denominator = max(float(np.linalg.norm(left_f)), 1e-12)
    return {
        "byte_identical": _exact(left, right),
        "relative_l2": float(np.linalg.norm(delta) / denominator),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "different_elements": int(np.count_nonzero(delta)),
        "elements": int(delta.size),
        "reference_hash": hashlib.sha256(left_f.tobytes()).hexdigest(),
        "compiled_hash": hashlib.sha256(right_f.tobytes()).hexdigest(),
    }


def _memory() -> dict[str, int]:
    mx.synchronize()
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }


def _checkpoint_tensors(path: Path, prefix: str) -> dict[str, mx.array]:
    """Load only shards containing tensors below one checkpoint prefix."""
    index = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    names = {name: shard for name, shard in index.items() if name.startswith(prefix)}
    if not names:
        raise ValueError(f"checkpoint prefix has no tensors: {prefix}")
    tensors: dict[str, mx.array] = {}
    for shard in sorted(set(names.values())):
        loaded = mx.load(str(path / shard))
        for name in names:
            if names[name] == shard:
                tensors[name] = loaded[name]
    if set(tensors) != set(names):
        missing = sorted(set(names) - set(tensors))
        raise ValueError(f"selective checkpoint load missed tensors: {missing}")
    return tensors


def _load_kda_attention(path: Path, config):
    """Construct the official layer-0 KDA attention without loading the MoE."""
    apply_runtime_patch()
    from mlx_vlm.models.glm5_next.language import Glm5NextLinearAttention

    attention = Glm5NextLinearAttention(config.text_config)
    prefix = f"model.language_model.layers.{KDA_LAYER}.self_attn."
    raw = _checkpoint_tensors(path, prefix)
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


def initial_kda_state(attention, *, dtype=mx.bfloat16):
    conv = mx.zeros(
        (1, attention.conv_kernel_size - 1, attention.conv_dim), dtype=dtype
    )
    recurrent = mx.zeros(
        (
            1,
            attention.num_heads,
            attention.head_dim,
            attention.head_dim,
        ),
        dtype=mx.float32,
    )
    return conv, recurrent


def kda_state_schema(state) -> KDAStateSchema:
    if not isinstance(state, (tuple, list)) or len(state) != 2:
        raise ValueError("KDA functional state must contain exactly conv and recurrent")
    conv, recurrent = state
    if not isinstance(conv, mx.array) or not isinstance(recurrent, mx.array):
        raise TypeError("KDA functional state leaves must be MLX arrays")
    if conv.ndim != 3 or recurrent.ndim != 4:
        raise ValueError("KDA state rank mismatch")
    if conv.shape[0] != 1 or recurrent.shape[0] != 1:
        raise ValueError("KDA functional executable supports batch size 1 only")
    return KDAStateSchema(
        tuple(map(int, conv.shape)),
        str(conv.dtype),
        tuple(map(int, recurrent.shape)),
        str(recurrent.dtype),
    )


def validate_kda_state(attention, state) -> KDAStateSchema:
    schema = kda_state_schema(state)
    expected = initial_kda_state(attention, dtype=state[0].dtype)
    expected_schema = kda_state_schema(expected)
    if schema != expected_schema:
        raise ValueError(
            f"KDA state schema mismatch: expected {expected_schema}, got {schema}"
        )
    return schema


def functional_kda_decode(
    attention,
    inputs,
    conv_state,
    recurrent_state,
    position,
    *,
    diagnostics: bool = False,
):
    """The pinned GLM KDA decode tree with all mutable state made explicit."""
    from mlx_vlm.models.gated_delta import gated_delta_update
    from mlx_vlm.models.glm5_next.language import _l2norm
    import mlx.nn as nn

    batch, length, _ = inputs.shape
    if batch != 1 or length != 1:
        raise ValueError("functional KDA decode requires fixed [1, 1, hidden] input")

    q_out = attention.q_proj(inputs)
    k_out = attention.k_proj(inputs)
    v_out = attention.v_proj(inputs)
    fa_out = attention.forget_gate.f_a_proj(inputs)
    ga_out = attention.g_a_proj(inputs)
    beta_out = attention.b_proj(inputs)
    mixed = mx.concatenate([q_out, k_out, v_out], axis=-1)
    conv_input = mx.concatenate([conv_state, mixed], axis=1)
    next_conv = mx.contiguous(
        conv_input[:, -(attention.conv_kernel_size - 1) :, :],
        allow_col_major=False,
    )
    conv_out = nn.silu(attention.conv1d(conv_input))
    q, k, value = mx.split(
        conv_out,
        [attention.qkv_dim, 2 * attention.qkv_dim],
        axis=-1,
    )
    q = q.reshape(1, 1, attention.num_heads, attention.head_dim)
    k = k.reshape(1, 1, attention.num_heads, attention.head_dim)
    value = value.reshape(1, 1, attention.num_heads, attention.head_dim)
    forget = attention.forget_gate
    a = forget.f_b_proj(fa_out).reshape(
        1, 1, attention.num_heads, attention.head_dim
    )
    in_dtype = q.dtype
    q = (_l2norm(q.astype(mx.float32)) * (attention.head_dim**-0.5)).astype(
        in_dtype
    )
    k = _l2norm(k.astype(mx.float32)).astype(in_dtype)
    recurrent_output, next_recurrent = gated_delta_update(
        q,
        k,
        value,
        a,
        beta_out,
        forget.A_log.reshape(attention.num_heads, 1),
        forget.dt_bias.reshape(attention.num_heads, attention.head_dim),
        state=recurrent_state,
        lower_bound=forget.safe_gate_lower_bound,
    )
    gate = attention.g_b_proj(ga_out).reshape(
        1, 1, attention.num_heads, attention.head_dim
    )
    gated_output = attention.o_norm(recurrent_output, gate).reshape(1, 1, -1)
    output = attention.o_proj(gated_output)
    result = output, next_conv, next_recurrent, position + mx.array(1, mx.int32)
    if diagnostics:
        return (*result, recurrent_output, gated_output)
    return result


def _deterministic_input(hidden_size: int, *, strided: bool = False) -> mx.array:
    wide = mx.arange(hidden_size * 2, dtype=mx.float32).reshape(1, 1, -1)
    value = mx.sin(wide * mx.array(0.0009765625, mx.float32))
    selected = value[..., 1::2].astype(mx.bfloat16)
    return selected if strided else mx.contiguous(selected, allow_col_major=False)


def _eager_kda_step(attention, inputs, state):
    from mlx_vlm.models.cache import ArraysCache

    cache = ArraysCache(size=2)
    cache.state = [state[0], state[1]]
    output = attention(inputs, cache=cache)
    return output, cache[0], cache[1]


def _tier0(attention) -> dict:
    state = initial_kda_state(attention)
    schema = validate_kda_state(attention, state)
    before = (_hash(state[0]), _hash(state[1]))
    rejected = []
    invalid = (
        (state[0],),
        (state[0][:, :, :-1], state[1]),
        (state[0], state[1][:, :, :, :-1]),
    )
    for candidate in invalid:
        try:
            validate_kda_state(attention, candidate)
        except (TypeError, ValueError) as error:
            rejected.append(type(error).__name__)
        else:
            raise AssertionError("invalid KDA state was accepted")
    after = (_hash(state[0]), _hash(state[1]))

    # Prefill -> decode transition using actual recurrence and cache ownership.
    prefill = mx.broadcast_to(
        _deterministic_input(attention.hidden_size),
        (1, attention.conv_kernel_size, attention.hidden_size),
    )
    prefill_output, conv, recurrent = _eager_kda_step(attention, prefill, state)
    mx.eval(prefill_output, conv, recurrent)
    decode_input = _deterministic_input(attention.hidden_size)
    eager = _eager_kda_step(attention, decode_input, (conv, recurrent))
    functional = functional_kda_decode(
        attention,
        decode_input,
        conv,
        recurrent,
        mx.array(attention.conv_kernel_size, mx.int32),
    )
    mx.eval(*eager, *functional)
    transition_exact = all(
        _exact(left, right)
        for left, right in zip(eager, functional[:3], strict=True)
    )
    return {
        "passed": bool(
            len(rejected) == len(invalid)
            and before == after
            and transition_exact
        ),
        "state_schema": {
            "conv_shape": list(schema.conv_shape),
            "conv_dtype": schema.conv_dtype,
            "recurrent_shape": list(schema.recurrent_shape),
            "recurrent_dtype": schema.recurrent_dtype,
            "leaf_count": 2,
        },
        "invalid_state_cases": len(invalid),
        "invalid_state_cases_rejected": len(rejected),
        "rejected_state_unchanged": before == after,
        "prefill_tokens": attention.conv_kernel_size,
        "prefill_to_decode_output_exact": _exact(eager[0], functional[0]),
        "prefill_to_decode_conv_exact": _exact(eager[1], functional[1]),
        "prefill_to_decode_recurrent_exact": _exact(eager[2], functional[2]),
        "transition_exact": transition_exact,
    }


def _run_steps(callable_, inputs, initial, *, steps: int):
    state = initial
    build_ms = []
    outputs = []
    for step in range(steps):
        started = time.perf_counter_ns()
        result = callable_(inputs, *state, mx.array(step, mx.int32))
        build_ms.append((time.perf_counter_ns() - started) / 1e6)
        mx.eval(*result)
        outputs.append(result[0])
        state = result[1:3]
    mx.synchronize()
    return outputs, state, build_ms


def _paired_kda_steps(
    attention, compiled, inputs, initial, *, steps: int, diagnostics: bool = False
):
    eager_state = initial
    compiled_state = initial
    first_output_divergence = None
    first_conv_divergence = None
    first_recurrent_divergence = None
    first_output_metrics = None
    first_stage_metrics = None
    for step in range(steps):
        position = mx.array(step, mx.int32)
        eager = functional_kda_decode(
            attention,
            inputs,
            *eager_state,
            position,
            diagnostics=diagnostics,
        )
        actual = compiled(inputs, *compiled_state, position)
        mx.eval(*eager, *actual)
        if first_output_divergence is None and not _exact(eager[0], actual[0]):
            first_output_divergence = step + 1
            first_output_metrics = _metrics(eager[0], actual[0])
            if diagnostics:
                first_stage_metrics = {
                    "recurrent_output": _metrics(eager[4], actual[4]),
                    "gated_norm_output": _metrics(eager[5], actual[5]),
                    "final_projection_output": _metrics(eager[0], actual[0]),
                }
        if first_conv_divergence is None and not _exact(eager[1], actual[1]):
            first_conv_divergence = step + 1
        if first_recurrent_divergence is None and not _exact(eager[2], actual[2]):
            first_recurrent_divergence = step + 1
        eager_state = eager[1:3]
        compiled_state = actual[1:3]
    return {
        "all_outputs_exact": first_output_divergence is None,
        "all_conv_states_exact": first_conv_divergence is None,
        "all_recurrent_states_exact": first_recurrent_divergence is None,
        "first_output_divergence": first_output_divergence,
        "first_conv_divergence": first_conv_divergence,
        "first_recurrent_divergence": first_recurrent_divergence,
        "first_output_metrics": first_output_metrics,
        "first_stage_metrics": first_stage_metrics,
        "eager_state": eager_state,
        "compiled_state": compiled_state,
    }


def _tier1(attention) -> dict:
    contiguous = _deterministic_input(attention.hidden_size)
    strided = _deterministic_input(attention.hidden_size, strided=True)
    initial = initial_kda_state(attention)
    validate_kda_state(attention, initial)

    trace_counter = {"calls": 0}

    def traced(inputs, conv, recurrent, position):
        trace_counter["calls"] += 1
        return functional_kda_decode(
            attention, inputs, conv, recurrent, position
        )

    compiled = mx.compile(traced)
    cold_started = time.perf_counter()
    cold = compiled(contiguous, *initial, mx.array(0, mx.int32))
    mx.eval(*cold)
    mx.synchronize()
    cold_seconds = time.perf_counter() - cold_started

    signatures = []
    for offset in KDA_OFFSET_FIXTURES:
        result = compiled(contiguous, *initial, mx.array(offset, mx.int32))
        mx.eval(*result)
        signatures.append(
            {
                "offset": offset,
                "position_after": int(result[3].item()),
                "output_hash": _hash(result[0]),
                "trace_calls_after": trace_counter["calls"],
            }
        )

    eager_outputs, eager_state, eager_build = _run_steps(
        lambda x, conv, recurrent, position: functional_kda_decode(
            attention, x, conv, recurrent, position
        ),
        contiguous,
        initial,
        steps=KDA_STEPS,
    )
    compiled_outputs, compiled_state, compiled_build = _run_steps(
        compiled, contiguous, initial, steps=KDA_STEPS
    )
    eager_repeat_outputs, eager_repeat_state, _ = _run_steps(
        lambda x, conv, recurrent, position: functional_kda_decode(
            attention, x, conv, recurrent, position
        ),
        contiguous,
        initial,
        steps=KDA_STEPS,
    )
    compiled_repeat_outputs, compiled_repeat_state, _ = _run_steps(
        compiled, contiguous, initial, steps=KDA_STEPS
    )
    eager_repeat_exact = all(
        _exact(left, right)
        for left, right in zip(
            eager_outputs, eager_repeat_outputs, strict=True
        )
    ) and all(
        _exact(left, right)
        for left, right in zip(eager_state, eager_repeat_state, strict=True)
    )
    compiled_repeat_exact = all(
        _exact(left, right)
        for left, right in zip(
            compiled_outputs, compiled_repeat_outputs, strict=True
        )
    ) and all(
        _exact(left, right)
        for left, right in zip(
            compiled_state, compiled_repeat_state, strict=True
        )
    )
    paired = _paired_kda_steps(
        attention, compiled, contiguous, initial, steps=KDA_STEPS
    )
    output_exact = paired["all_outputs_exact"]
    state_exact = bool(
        paired["all_conv_states_exact"]
        and paired["all_recurrent_states_exact"]
    )
    eager_median = float(statistics.median(eager_build))
    compiled_median = float(statistics.median(compiled_build))
    reduction = 1.0 - compiled_median / eager_median

    contiguous_result = compiled(contiguous, *initial, mx.array(0, mx.int32))
    strided_result = compiled(strided, *initial, mx.array(0, mx.int32))
    mx.eval(*contiguous_result, *strided_result)
    strided_exact = all(
        _exact(left, right)
        for left, right in zip(
            contiguous_result[:3], strided_result[:3], strict=True
        )
    )
    fixed_signature = trace_counter["calls"] == 1
    offset_exact = all(
        row["position_after"] == row["offset"] + 1 for row in signatures
    )
    diagnostic_eager = functional_kda_decode(
        attention,
        contiguous,
        *initial,
        mx.array(0, mx.int32),
        diagnostics=True,
    )

    def diagnostic_tree(inputs, conv, recurrent, position):
        return functional_kda_decode(
            attention,
            inputs,
            conv,
            recurrent,
            position,
            diagnostics=True,
        )

    diagnostic_compiled = mx.compile(diagnostic_tree)(
        contiguous, *initial, mx.array(0, mx.int32)
    )
    mx.eval(*diagnostic_eager, *diagnostic_compiled)
    stage_metrics = {
        "recurrent_output": _metrics(
            diagnostic_eager[4], diagnostic_compiled[4]
        ),
        "gated_norm_output": _metrics(
            diagnostic_eager[5], diagnostic_compiled[5]
        ),
        "final_projection_output": _metrics(
            diagnostic_eager[0], diagnostic_compiled[0]
        ),
        "next_conv_state": _metrics(
            diagnostic_eager[1], diagnostic_compiled[1]
        ),
        "next_recurrent_state": _metrics(
            diagnostic_eager[2], diagnostic_compiled[2]
        ),
    }
    diagnostic_callable = mx.compile(diagnostic_tree)
    anchored_paired = _paired_kda_steps(
        attention,
        diagnostic_callable,
        contiguous,
        initial,
        steps=KDA_STEPS,
        diagnostics=True,
    )
    first_divergent_step = paired["first_output_divergence"]
    passed = bool(
        output_exact
        and state_exact
        and strided_exact
        and fixed_signature
        and offset_exact
        and reduction >= KDA_HOST_BUILD_REDUCTION_GATE
    )
    return {
        "passed": passed,
        "layer": KDA_LAYER,
        "steps": KDA_STEPS,
        "process_first_compile_seconds": cold_seconds,
        "compiler_persistent_cache_cleared": False,
        "compile_trace_calls": trace_counter["calls"],
        "fixed_signature": fixed_signature,
        "offset_fixtures": signatures,
        "offset_updates_exact": offset_exact,
        "all_step_outputs_byte_identical": output_exact,
        "eager_repeated_execution_byte_identical": eager_repeat_exact,
        "compiled_repeated_execution_byte_identical": compiled_repeat_exact,
        "final_conv_state_byte_identical": _exact(
            eager_state[0], compiled_state[0]
        ),
        "final_recurrent_state_byte_identical": _exact(
            eager_state[1], compiled_state[1]
        ),
        "final_state_byte_identical": state_exact,
        "strided_contiguous_byte_identical": strided_exact,
        "first_divergent_output_step": first_divergent_step,
        "first_divergent_conv_state_step": paired["first_conv_divergence"],
        "first_divergent_recurrent_state_step": paired[
            "first_recurrent_divergence"
        ],
        "first_divergent_output_metrics": paired["first_output_metrics"],
        "first_step_stage_metrics": stage_metrics,
        "explicit_boundary_anchor": {
            "returned_auxiliary_tensors": [
                "recurrent_output",
                "gated_norm_output",
            ],
            "all_step_outputs_byte_identical": anchored_paired[
                "all_outputs_exact"
            ],
            "all_conv_states_byte_identical": anchored_paired[
                "all_conv_states_exact"
            ],
            "all_recurrent_states_byte_identical": anchored_paired[
                "all_recurrent_states_exact"
            ],
            "first_output_divergence": anchored_paired[
                "first_output_divergence"
            ],
            "first_conv_divergence": anchored_paired[
                "first_conv_divergence"
            ],
            "first_recurrent_divergence": anchored_paired[
                "first_recurrent_divergence"
            ],
            "first_divergent_stage_metrics": anchored_paired[
                "first_stage_metrics"
            ],
        },
        "state_leaf_count": 2,
        "eager_host_build_median_ms": eager_median,
        "compiled_host_build_median_ms": compiled_median,
        "host_build_reduction": reduction,
        "host_build_reduction_gate": KDA_HOST_BUILD_REDUCTION_GATE,
        "output_hash": _hash(compiled_outputs[-1]),
        "output_dtype": str(compiled_outputs[-1].dtype),
        "conv_state_hash": _hash(compiled_state[0]),
        "recurrent_state_hash": _hash(compiled_state[1]),
    }


def _skipped(reason: str) -> dict:
    return {
        "executed": False,
        "passed": False,
        "reason": reason,
        "correctness_claim": False,
        "performance_claim": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tier", type=int, choices=range(0, 5), default=4)
    parser.add_argument(
        "--cold-compile-budget-seconds",
        type=float,
        default=FULL_COMPILE_BUDGET_SECONDS,
    )
    args = parser.parse_args()
    started = time.time()
    checkpoint = inspect_checkpoint(args.model, require_server_ready=True)
    artifact = {
        "schema": "glm53-functional-stateful-decode-executable-v1",
        "date": str(date.today()),
        "complete": False,
        "probe_only": True,
        "runtime_changes": False,
        "checkpoint_revision": checkpoint.official_revision,
        "checkpoint_fingerprint": checkpoint.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "max_tier_requested": args.max_tier,
        "cold_compile_budget_seconds": args.cold_compile_budget_seconds,
        "tiers": {},
        "memory_before": _memory(),
    }
    _atomic_write(args.output, artifact)
    try:
        raw_config = json.loads((args.model / "config.json").read_text())
        config = _make_config(raw_config)
        _progress("load_kda", layer=KDA_LAYER)
        attention = _load_kda_attention(args.model, config)

        _progress("tier0")
        tier0 = _tier0(attention)
        artifact["tiers"]["tier0_state_schema"] = tier0
        _atomic_write(args.output, artifact)
        if not tier0["passed"] or args.max_tier == 0:
            reason = "Tier 0 failed" if not tier0["passed"] else "max tier is 0"
            artifact["tiers"]["tier1_kda"] = _skipped(reason)
        else:
            _progress("tier1", layer=KDA_LAYER)
            tier1 = _tier1(attention)
            tier1["executed"] = True
            artifact["tiers"]["tier1_kda"] = tier1
            _atomic_write(args.output, artifact)

        tier1 = artifact["tiers"]["tier1_kda"]
        if args.max_tier < 2:
            reason = "max tier is below 2"
        elif not tier1.get("passed", False):
            reason = "Tier 1 KDA gate failed; DSA was not loaded"
        else:
            reason = "Tier 2 implementation is not present"
        artifact["tiers"]["tier2_dsa"] = _skipped(reason)
        artifact["tiers"]["tier3_complete_layer"] = _skipped(
            "Tier 2 did not pass"
        )
        artifact["tiers"]["tier4_full_executable"] = _skipped(
            "Tier 1-3 did not all pass"
        )
        artifact["decision"] = (
            "continue_to_tier2"
            if tier1.get("passed", False) and args.max_tier >= 2
            else "reject_functional_stateful_decode_at_kda_exactness"
        )
        artifact["failure_condition"] = (
            None
            if tier1.get("passed", False)
            else "mx.compile changes the eager gated-RMSNorm numerical boundary"
        )
        artifact["acceptance"] = {
            "tier0_state_invariants": bool(tier0["passed"]),
            "tier1_single_compile_signature": bool(
                tier1.get("fixed_signature", False)
            ),
            "tier1_runtime_tensor_offsets": bool(
                tier1.get("offset_updates_exact", False)
            ),
            "tier1_state_byte_identical": bool(
                tier1.get("final_state_byte_identical", False)
            ),
            "tier1_output_byte_identical": bool(
                tier1.get("all_step_outputs_byte_identical", False)
            ),
            "tier1_host_build_reduction_gate": bool(
                tier1.get("host_build_reduction", 0.0)
                >= KDA_HOST_BUILD_REDUCTION_GATE
            ),
            "tier2_not_loaded_after_tier1_failure": bool(
                not artifact["tiers"]["tier2_dsa"]["executed"]
            ),
            "runtime_unchanged": True,
        }
        artifact["evidence_complete"] = bool(
            tier0["passed"] and tier1.get("executed", False)
        )
        artifact["complete"] = True
        artifact["elapsed_seconds"] = time.time() - started
        artifact["memory_after"] = _memory()
        _atomic_write(args.output, artifact)
        _progress("complete", decision=artifact["decision"])
        return 0
    except Exception as error:
        artifact["complete"] = False
        artifact["error"] = f"{type(error).__name__}: {error}"
        artifact["traceback"] = traceback.format_exc()
        artifact["elapsed_seconds"] = time.time() - started
        _atomic_write(args.output, artifact)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
