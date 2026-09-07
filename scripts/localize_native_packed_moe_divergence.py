#!/usr/bin/env python3
"""Localize the official step-68 native packed-MoE divergence.

Capture the exact-composition L=1 input/output of every sparse layer for the
official greedy trajectory, then replay those owned activations through the
same 42 persistent native plans in temporal order.  On the first mismatch,
compare router, routed hidden/down/reduction, shared hidden/down, and final add
without changing production runtime or the rejected plan.
"""

from __future__ import annotations

import argparse
import contextlib
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
from glm53_flash_mlx.fp8 import block_fp8_linear
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint
from glm53_flash_mlx.packed import PackedFP8MoE


ROOT = Path(__file__).resolve().parents[1]
NATIVE_PACKAGE = ROOT / "native_execution"
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-packed-moe-divergence-localization-20260907.json"
)
SOURCE_ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-packed-moe-execution-plan-20260907.json"
)
DEFAULT_TARGET_STEP = 68


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


def _atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _load_helpers():
    for path in (str(ROOT / "scripts"), str(NATIVE_PACKAGE)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from glm53_native_execution import NativePackedMoEDecodePlan
    import oracle_trace
    import probe_fused_packed_gate_up_swiglu_decode as d99f
    import probe_native_packed_moe_execution_plan as native_probe
    import probe_residual_packed_decode_moe_fusion as residual

    return NativePackedMoEDecodePlan, oracle_trace, d99f, native_probe, residual


def _arrays(value):
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _arrays(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _arrays(child)


def _eval(value) -> None:
    arrays = list(_arrays(value))
    if arrays:
        mx.eval(*arrays)
    mx.synchronize()


def _hash(value: mx.array) -> str:
    host = np.ascontiguousarray(np.asarray(value.astype(mx.float32)))
    return hashlib.sha256(host.tobytes()).hexdigest()


class _ReferenceCapture:
    def __init__(self, model, residual):
        self.residual = residual
        self.layers = {
            id(layer.mlp): layer_id
            for layer_id, layer in enumerate(model.language_model.model.layers)
            if isinstance(layer.mlp, PackedFP8MoE)
        }
        self.step = None
        self.inputs = {layer_id: [] for layer_id in self.layers.values()}
        self.outputs = {layer_id: [] for layer_id in self.layers.values()}
        self.pending = []

    def call(self, moe, x):
        flat = x.reshape(-1, x.shape[-1])
        if int(flat.shape[0]) != 1:
            return self.residual._ORIGINAL_PACKED_CALL(moe, x)
        result = self.residual._moe_variant(moe, x, "B1", True)
        if self.step is not None:
            layer_id = self.layers[id(moe)]
            captured_input = mx.array(x)
            captured_output = mx.array(result)
            self.inputs[layer_id].append(captured_input)
            self.outputs[layer_id].append(captured_output)
            self.pending.extend((captured_input, captured_output))
        return result


@contextlib.contextmanager
def _capture_runtime(capture):
    previous = PackedFP8MoE.__call__

    def wrapped(moe, x):
        return capture.call(moe, x)

    PackedFP8MoE.__call__ = wrapped
    try:
        yield
    finally:
        PackedFP8MoE.__call__ = previous


def _capture_trajectory(model, processor, oracle_trace, residual, target_step):
    formatted = processor.apply_chat_template(
        [{"role": "user", "content": oracle_trace.DEFAULT_PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = processor(formatted, return_tensors="np", add_special_tokens=True)
    prompt_ids = np.asarray(encoded["input_ids"], dtype=np.int32).reshape(1, -1)
    capture = _ReferenceCapture(model, residual)
    cache = model.make_cache()
    generated = []
    logits_hashes = []
    with _capture_runtime(capture):
        output = model(mx.array(prompt_ids), cache=cache)
        for output_step in range(target_step + 1):
            _eval((output.logits, capture.pending))
            capture.pending.clear()
            logits = output.logits[0, -1].astype(mx.float32)
            _eval(logits)
            logits_hashes.append(_hash(logits))
            token = int(mx.argmax(logits).item())
            generated.append(token)
            if output_step < target_step:
                capture.step = output_step + 1
                output = model(mx.array([[token]], dtype=mx.uint32), cache=cache)
                capture.step = None
    for layer_id in capture.inputs:
        if len(capture.inputs[layer_id]) != target_step:
            raise RuntimeError(
                f"layer {layer_id} capture count {len(capture.inputs[layer_id])} "
                f"!= {target_step}"
            )
    cache.clear()
    del output, cache
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    return capture, generated, logits_hashes


def _diagnose_stages(moe, x, plan, d99f, residual):
    flat = mx.contiguous(x.reshape(-1, x.shape[-1]), allow_col_major=False)[0]
    indices, scores = moe.gate(x)
    expert_ids = mx.contiguous(
        indices.reshape(-1).astype(mx.uint32), allow_col_major=False
    )
    flat_scores = mx.contiguous(scores.reshape(-1), allow_col_major=False)
    shared = moe.shared_experts
    routed_hidden = d99f.fused_packed_gate_up_swiglu(
        flat, expert_ids, moe.bank, limit=moe.config.swiglu_limit
    )
    routed_down = d99f._packed_down_raw(routed_hidden, expert_ids, moe.bank)
    routed_output = residual.aggregate_b1(routed_down, flat_scores)
    shared_hidden = residual.fused_shared_gate_up_swiglu(
        flat, shared, limit=moe.config.swiglu_limit
    )
    shared_down = block_fp8_linear(
        shared_hidden,
        shared.down_proj.weight,
        shared.down_proj.weight_scale_inv,
    )
    final = (routed_output + shared_down).reshape(x.shape)
    dependencies = (
        flat,
        expert_ids,
        flat_scores,
        moe.bank.gate_up_weight,
        moe.bank.gate_up_scale_inv,
        moe.bank.down_weight,
        moe.bank.down_scale_inv,
        shared.gate_proj.weight,
        shared.gate_proj.weight_scale_inv,
        shared.up_proj.weight,
        shared.up_proj.weight_scale_inv,
        shared.down_proj.weight,
        shared.down_proj.weight_scale_inv,
    )
    mx.async_eval(*dependencies)
    candidate = plan.execute(*dependencies)
    stages = {
        "routed_hidden": (routed_hidden, plan.debug_routed_hidden),
        "routed_down": (routed_down, plan.debug_routed_down),
        "routed_output": (routed_output, plan.debug_routed_output),
        "shared_hidden": (shared_hidden, plan.debug_shared_hidden),
        "shared_down": (shared_down, plan.debug_shared_down),
        "final_output": (final, candidate),
    }
    _eval(tuple(value for pair in stages.values() for value in pair))
    metrics = {
        name: d99f._metrics(reference, actual)
        for name, (reference, actual) in stages.items()
    }
    first = next(
        (name for name, row in metrics.items() if not row["byte_identical"]),
        None,
    )
    return {
        "router_indices_hash": _hash(indices),
        "router_scores_hash": _hash(scores),
        "first_differing_stage": first,
        "stages": metrics,
    }


def _replay(model, capture, native_probe, plan_type, d99f, residual, target_step):
    registry = native_probe._Registry(plan_type)
    layers = model.language_model.model.layers
    sparse_layers = sorted(capture.inputs)
    exact_comparisons = 0
    for step in range(1, target_step + 1):
        candidates = {
            layer_id: native_probe._native_call(
                registry, layers[layer_id].mlp, capture.inputs[layer_id][step - 1]
            )
            for layer_id in sparse_layers
        }
        _eval(list(candidates.values()))
        for layer_id in sparse_layers:
            reference = capture.outputs[layer_id][step - 1]
            candidate = candidates[layer_id]
            if not bool(mx.array_equal(reference, candidate).item()):
                divergence = {
                    "step": step,
                    "layer": layer_id,
                    "reference_output_hash": _hash(reference),
                    "candidate_output_hash": _hash(candidate),
                    "stage_localization": _diagnose_stages(
                        layers[layer_id].mlp,
                        capture.inputs[layer_id][step - 1],
                        registry.get(layers[layer_id].mlp),
                        d99f,
                        residual,
                    ),
                }
                return divergence, exact_comparisons, registry.evidence()
            exact_comparisons += 1
        if step % 8 == 0 or step == target_step:
            _progress("native_replay", step=step, target=target_step)
    return None, exact_comparisons, registry.evidence()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-step", type=int, default=DEFAULT_TARGET_STEP)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    args = parser.parse_args()
    artifact = {
        "schema": "glm53-native-packed-moe-divergence-localization-v1",
        "date": date.today().isoformat(),
        "complete": False,
        "accepted": False,
        "diagnostic_only": True,
        "target_step": args.target_step,
        "source_artifact": str(SOURCE_ARTIFACT.relative_to(ROOT)),
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "runtime_changes": {
            "runtime": False,
            "server": False,
            "apc": False,
            "cache_abi": False,
            "kernel_abi": False,
        },
    }
    try:
        source = json.loads(SOURCE_ARTIFACT.read_text())
        expected_failure = f"step {args.target_step} logits hash mismatch"
        if expected_failure not in source["official_oracle"][
            "B_native_packed_moe_plan"
        ]["failures_128"]:
            raise RuntimeError("target step is not a recorded oracle divergence")
        plan_type, oracle_trace, d99f, native_probe, residual = _load_helpers()
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
        _progress("capture_exact_composition", target_step=args.target_step)
        capture, generated, logits_hashes = _capture_trajectory(
            model, processor, oracle_trace, residual, args.target_step
        )
        artifact["capture"] = {
            "sparse_layers": sorted(capture.inputs),
            "steps_per_layer": args.target_step,
            "owned_activation_count": sum(len(row) for row in capture.inputs.values()),
            "owned_output_count": sum(len(row) for row in capture.outputs.values()),
            "generated_tokens_sha256": hashlib.sha256(
                np.asarray(generated, dtype=np.int32).tobytes()
            ).hexdigest(),
            "target_logits_hash": logits_hashes[args.target_step],
        }
        _progress("replay_native_plans", target_step=args.target_step)
        divergence, exact_comparisons, evidence = _replay(
            model,
            capture,
            native_probe,
            plan_type,
            d99f,
            residual,
            args.target_step,
        )
        artifact["replay"] = {
            "first_divergence": divergence,
            "exact_layer_step_comparisons_before_divergence": exact_comparisons,
            "native_plan_evidence": evidence,
        }
        artifact["process_peak_memory_bytes"] = int(mx.get_peak_memory())
        artifact["complete"] = True
        artifact["accepted"] = divergence is not None
        artifact["decision"] = (
            "repair_localized_native_moe_stage"
            if divergence is not None
            else "investigate_full_graph_or_fixed_arena_lifetime"
        )
    except Exception as error:
        artifact.update(
            complete=True,
            accepted=False,
            decision="native_packed_moe_localization_failed",
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
    _atomic_write(args.output, artifact)
    first_divergence = artifact.get("replay", {}).get("first_divergence")
    summary = (
        None
        if first_divergence is None
        else {
            "step": first_divergence["step"],
            "layer": first_divergence["layer"],
            "stage": first_divergence["stage_localization"][
                "first_differing_stage"
            ],
        }
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": artifact["complete"],
                "accepted": artifact["accepted"],
                "decision": artifact["decision"],
                "first_divergence": summary,
            },
            indent=2,
        )
    )
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
