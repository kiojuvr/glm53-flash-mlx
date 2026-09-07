#!/usr/bin/env python3
"""Localize the native routed-hidden mismatch inside projection/SwiGLU."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import sys
import tempfile
import traceback
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np

from glm53_flash_mlx.abi import MLX_VLM_REVISION
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
NATIVE_PACKAGE = ROOT / "native_execution"
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-routed-hidden-numerical-localization-20260907.json"
)
SOURCE_ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-shape-specialized-native-moe-repair-20260907.json"
)
TARGET_STEP = 29
TARGET_LAYER = 41


_ACTIVATION_DIAGNOSTIC_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= ELEMENTS) return;
    T gate_activation = gate[index];
    T up_activation = up[index];
    auto sigmoid_tail = 1 / (
        1 + metal::exp(metal::abs(gate_activation))
    );
    T sigmoid_value = (gate_activation < 0)
        ? sigmoid_tail
        : 1 - sigmoid_tail;
    T silu_value = gate_activation * sigmoid_value;
    T activated = silu_value * up_activation;
    sigmoid_output[index] = sigmoid_value;
    silu_output[index] = silu_value;
    hidden[index] = activated;
"""


_activation_diagnostic_kernel = (
    mx.fast.metal_kernel(
        name="glm53_probe_routed_activation_diagnostic",
        input_names=["gate", "up"],
        output_names=["sigmoid_output", "silu_output", "hidden"],
        source=_ACTIVATION_DIAGNOSTIC_SOURCE,
    )
    if mx.metal.is_available()
    else None
)


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


def _atomic_write(path: Path, value: dict) -> None:
    def json_default(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    # Serialize before creating the temporary file.  A diagnostic conversion
    # bug must not leave a truncated artifact that looks recoverable.
    payload = json.dumps(value, indent=2, sort_keys=True, default=json_default)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _load_helpers():
    for path in (str(SCRIPTS), str(NATIVE_PACKAGE)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from glm53_native_execution import NativePackedMoERoutedDiagnostic
    import localize_native_packed_moe_divergence as localizer

    plan_type, oracle_trace, d99f, native_probe, residual = (
        localizer._load_helpers()
    )
    return (
        NativePackedMoERoutedDiagnostic,
        localizer,
        plan_type,
        oracle_trace,
        d99f,
        native_probe,
        residual,
    )


def _metrics(reference: mx.array, actual: mx.array) -> dict:
    mx.eval(reference, actual)
    reference_f32 = np.ascontiguousarray(
        np.asarray(reference.astype(mx.float32)), dtype=np.float32
    )
    actual_f32 = np.ascontiguousarray(
        np.asarray(actual.astype(mx.float32)), dtype=np.float32
    )
    reference_bits = np.ascontiguousarray(
        np.asarray(reference.view(mx.uint16)), dtype=np.uint16
    ).reshape(-1)
    actual_bits = np.ascontiguousarray(
        np.asarray(actual.view(mx.uint16)), dtype=np.uint16
    ).reshape(-1)
    different = np.flatnonzero(reference_bits != actual_bits)
    result = {
        "byte_identical": not bool(different.size),
        "shape": list(reference.shape),
        "different_elements": int(different.size),
        "reference_hash": hashlib.sha256(reference_bits.tobytes()).hexdigest(),
        "actual_hash": hashlib.sha256(actual_bits.tobytes()).hexdigest(),
        "max_abs": float(np.max(np.abs(actual_f32 - reference_f32))),
        "first_difference": None,
    }
    if different.size:
        flat_index = int(different[0])
        coordinate = [
            int(value)
            for value in np.unravel_index(flat_index, reference.shape)
        ]
        result["first_difference"] = {
            "flat_index": flat_index,
            "coordinate": coordinate,
            "reference_value_f32": float(reference_f32.reshape(-1)[flat_index]),
            "actual_value_f32": float(actual_f32.reshape(-1)[flat_index]),
            "reference_bf16_bits": f"0x{int(reference_bits[flat_index]):04x}",
            "actual_bf16_bits": f"0x{int(actual_bits[flat_index]):04x}",
        }
    return result


def _activation_diagnostics(gate: mx.array, up: mx.array):
    if _activation_diagnostic_kernel is None:
        raise RuntimeError("routed activation diagnostic requires Metal")
    elements = int(gate.size)
    return _activation_diagnostic_kernel(
        inputs=[gate, up],
        template=[("T", gate.dtype), ("ELEMENTS", elements)],
        grid=(elements, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[tuple(gate.shape)] * 3,
        output_dtypes=[gate.dtype] * 3,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    args = parser.parse_args()
    artifact = {
        "schema": "glm53-native-routed-hidden-numerical-localization-v1",
        "date": date.today().isoformat(),
        "complete": False,
        "accepted": False,
        "diagnostic_only": True,
        "source_artifact": str(SOURCE_ARTIFACT.relative_to(ROOT)),
        "target": {"step": TARGET_STEP, "layer": TARGET_LAYER},
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "runtime_changes": {
            "runtime": False,
            "server": False,
            "apc": False,
            "cache_abi": False,
            "production_kernel_abi": False,
        },
    }
    try:
        source = json.loads(SOURCE_ARTIFACT.read_text())
        divergence = source["owned_activation_replay"]["first_divergence"]
        if not source["complete"] or source["accepted"]:
            raise RuntimeError("expected a completed rejected repair source")
        if divergence["step"] != TARGET_STEP or divergence["layer"] != TARGET_LAYER:
            raise RuntimeError("repair source does not reproduce the target boundary")
        (
            diagnostic_type,
            localizer,
            plan_type,
            oracle_trace,
            d99f,
            native_probe,
            residual,
        ) = _load_helpers()
        report = inspect_checkpoint(args.model, require_server_ready=True)
        artifact["checkpoint_fingerprint"] = report.fingerprint
        artifact["official_hf_revision"] = report.official_revision
        mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
        mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
        _progress("load_model")
        model, processor = load(
            args.model,
            experimental_packed_decode_moe=True,
            experimental_compact_nope_dsa_cache=True,
        )
        warm_residency(model)
        _progress("capture_target_activation", step=TARGET_STEP, layer=TARGET_LAYER)
        capture, _, _ = localizer._capture_trajectory(
            model, processor, oracle_trace, residual, TARGET_STEP
        )
        x = capture.inputs[TARGET_LAYER][TARGET_STEP - 1]
        expected_moe_output = capture.outputs[TARGET_LAYER][TARGET_STEP - 1]
        moe = model.language_model.model.layers[TARGET_LAYER].mlp
        registry = native_probe._Registry(plan_type)
        candidate_moe_output = native_probe._native_call(registry, moe, x)
        localizer._eval(candidate_moe_output)
        plan = registry.get(moe)
        plan_hidden = mx.array(plan.debug_routed_hidden)

        flat = mx.contiguous(x.reshape(-1, x.shape[-1]), allow_col_major=False)[0]
        indices, scores = moe.gate(x)
        expert_ids = mx.contiguous(
            indices.reshape(-1).astype(mx.uint32), allow_col_major=False
        )
        dependencies = (
            flat,
            expert_ids,
            moe.bank.gate_up_weight,
            moe.bank.gate_up_scale_inv,
        )
        mx.async_eval(*dependencies)
        diagnostic = diagnostic_type(int(moe.bank.expert_count))
        diagnostic_hidden = diagnostic.execute(*dependencies)
        reference_gate, reference_up, reference_hidden = (
            d99f.fused_packed_gate_up_swiglu(
                flat,
                expert_ids,
                moe.bank,
                limit=moe.config.swiglu_limit,
                diagnostics=True,
            )
        )
        reference_sigmoid, reference_silu, activation_hidden = (
            _activation_diagnostics(reference_gate, reference_up)
        )
        localizer._eval(
            (
                expected_moe_output,
                candidate_moe_output,
                plan_hidden,
                diagnostic_hidden,
                reference_gate,
                reference_up,
                reference_hidden,
                reference_sigmoid,
                reference_silu,
                activation_hidden,
                diagnostic.gate,
                diagnostic.up,
                diagnostic.sigmoid,
                diagnostic.silu,
                scores,
            )
        )
        stages = {
            "gate_projection_bf16": _metrics(reference_gate, diagnostic.gate),
            "up_projection_bf16": _metrics(reference_up, diagnostic.up),
            "sigmoid_bf16": _metrics(reference_sigmoid, diagnostic.sigmoid),
            "silu_bf16": _metrics(reference_silu, diagnostic.silu),
            "activated_hidden_bf16": _metrics(
                reference_hidden, diagnostic.hidden
            ),
        }
        first_stage = next(
            (name for name, row in stages.items() if not row["byte_identical"]),
            None,
        )
        artifact["evidence"] = {
            "router_indices_hash": localizer._hash(indices),
            "router_scores_hash": localizer._hash(scores),
            "source_moe_output_mismatch_reproduced": not bool(
                mx.array_equal(expected_moe_output, candidate_moe_output).item()
            ),
            "diagnostic_hidden_matches_plan_hidden": bool(
                mx.array_equal(diagnostic_hidden, plan_hidden).item()
            ),
            "jit_activation_diagnostic_matches_exact_hidden": bool(
                mx.array_equal(activation_hidden, reference_hidden).item()
            ),
            "first_differing_stage": first_stage,
            "stages": stages,
            "diagnostic_execution_count": int(diagnostic.execution_count),
            "diagnostic_scratch_bytes": int(diagnostic.scratch_bytes),
            "diagnostic_buffer_identities": list(diagnostic.buffer_identities),
        }
        valid = (
            artifact["evidence"]["source_moe_output_mismatch_reproduced"]
            and artifact["evidence"]["diagnostic_hidden_matches_plan_hidden"]
            and artifact["evidence"][
                "jit_activation_diagnostic_matches_exact_hidden"
            ]
            and first_stage is not None
        )
        if first_stage in ("gate_projection_bf16", "up_projection_bf16"):
            decision = "repair_native_routed_projection_reduction"
        elif first_stage == "sigmoid_bf16":
            decision = "repair_native_routed_sigmoid_rounding"
        elif first_stage in ("silu_bf16", "activated_hidden_bf16"):
            decision = "repair_native_routed_activation_rounding"
        else:
            decision = "stop_unreliable_routed_hidden_diagnostic"
        artifact["process_peak_memory_bytes"] = int(mx.get_peak_memory())
        artifact["complete"] = True
        artifact["accepted"] = valid
        artifact["decision"] = decision if valid else (
            "stop_unreliable_routed_hidden_diagnostic"
        )
        del capture, model
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
    except Exception as error:
        artifact.update(
            complete=True,
            accepted=False,
            decision="native_routed_hidden_localization_failed",
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
    _atomic_write(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": artifact["complete"],
                "accepted": artifact["accepted"],
                "decision": artifact["decision"],
                "first_differing_stage": artifact.get("evidence", {}).get(
                    "first_differing_stage"
                ),
            },
            indent=2,
        )
    )
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
