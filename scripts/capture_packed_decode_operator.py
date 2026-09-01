#!/usr/bin/env python3
"""Capture bounded replayable traces for one packed MoE operator stage."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from capture_budget import CaptureBudget, atomic_write, supervise_capture
from capture_steady_packed_decode_critical_path import _trace_identity


LAYERS = (3, 24, 44)
STAGES = ("router", "routed", "shared", "ffn-add", "full-ffn")
WARMUPS = 2
REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-packed-decode-operator-microcapture-20260901.json"
)


def _validate_trace(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.suffix != ".gputrace":
        raise ValueError("--trace must end in .gputrace")
    if path.is_relative_to(REPOSITORY):
        raise ValueError("operator .gputrace must live outside the repository")
    if path.exists():
        raise FileExistsError(f"trace path already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _child(args) -> int:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    import probe_fused_packed_gate_up_swiglu_decode as d99f
    import probe_packed_decode_runtime as packed_probe
    import probe_residual_packed_decode_moe_fusion as residual

    from glm53_flash_mlx.fp8 import DirectFP8MoE
    from glm53_flash_mlx.loader import load_model
    from glm53_flash_mlx.manifest import inspect_checkpoint
    from glm53_flash_mlx.packed import PackedFP8ExpertBank, PackedFP8MoE

    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        raise RuntimeError("set MTL_CAPTURE_ENABLED=1 before importing MLX")
    report = inspect_checkpoint(args.model, require_server_ready=True)
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    mx.reset_peak_memory()
    model, _ = load_model(args.model, strict=True)
    direct = model.language_model.model.layers[args.layer].mlp
    if not isinstance(direct, DirectFP8MoE):
        raise RuntimeError(f"layer {args.layer} is not a sparse DirectFP8MoE")
    bank = PackedFP8ExpertBank.pack(direct.experts)
    packed = PackedFP8MoE(bank, direct.config, direct.gate, direct.shared_experts)
    values = [value for _, value in tree_flatten(packed.parameters())]
    mx.eval(*values)
    mx.synchronize()
    del values, direct, model
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    steady_active = int(mx.get_active_memory())
    steady_peak = int(mx.get_peak_memory())

    x = (
        mx.sin(mx.arange(packed.config.hidden_size, dtype=mx.float32) * 0.00390625)
        * 0.5
    ).astype(mx.bfloat16).reshape(1, 1, -1)
    flat = x.reshape(-1, x.shape[-1])
    indices, scores = packed.gate(x)
    expert_ids = indices.reshape(-1).astype(mx.uint32)
    flat_scores = scores.reshape(-1)
    mx.eval(indices, scores, expert_ids, flat_scores)

    def routed_value():
        hidden = d99f.fused_packed_gate_up_swiglu(
            flat[0], expert_ids, packed.bank, limit=packed.config.swiglu_limit
        )
        raw_down = d99f._packed_down_raw(hidden, expert_ids, packed.bank)
        return residual.aggregate_b1(raw_down, flat_scores)

    def shared_value():
        return residual._fused_shared(
            packed.shared_experts,
            flat[0],
            packed.config.swiglu_limit,
        )

    routed_anchor = routed_value()
    shared_anchor = shared_value()
    mx.eval(routed_anchor, shared_anchor)
    mx.synchronize()

    def operation():
        if args.stage == "router":
            return packed.gate(x)
        if args.stage == "routed":
            return routed_value()
        if args.stage == "shared":
            return shared_value()
        if args.stage == "ffn-add":
            return routed_anchor.reshape(x.shape) + shared_anchor.reshape(x.shape)
        if args.stage == "full-ffn":
            return residual._moe_variant(packed, x, "B1", True)
        raise AssertionError(args.stage)

    for _ in range(WARMUPS):
        output = operation()
        mx.eval(output)
        mx.synchronize()
    trace = _validate_trace(args.trace)
    started = time.perf_counter()
    mx.metal.start_capture(str(trace))
    try:
        output = operation()
        mx.eval(output)
        mx.synchronize()
    finally:
        mx.metal.stop_capture()
    captured_wall = time.perf_counter() - started
    leaves = list(output) if isinstance(output, tuple) else [output]
    hashes = [packed_probe._hash(value) for value in leaves]
    identity = _trace_identity(trace)
    identity.update(path=str(trace), stored_in_repository=False)
    row = {
        "schema": "glm53-packed-decode-operator-microcapture-case-v1",
        "complete": True,
        "layer": args.layer,
        "stage": args.stage,
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "activation": {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "source": "deterministic-checkpoint-native-layout",
        },
        "router_indices_hash": packed_probe._hash(indices),
        "router_scores_hash": packed_probe._hash(scores),
        "output_hashes": hashes,
        "trace": identity,
        "capture_process_wall_seconds": captured_wall,
        "capture_process_wall_is_performance_metric": False,
        "resident_payload_scope": "one packed MoE layer only",
        "full_model_payload_resident": False,
        "steady_active_bytes": steady_active,
        "peak_bytes": steady_peak,
        "bank_bytes": bank.nbytes,
        "canonical_storage": {
            "weight": str(bank.gate_up_weight.dtype),
            "scale": str(bank.gate_up_scale_inv.dtype),
        },
    }
    artifact = {
        "schema": "glm53-packed-decode-operator-microcapture-v1",
        "probe_only": True,
        "cases": {},
        "coverage_complete": False,
        "runtime_changes": False,
    }
    if args.output.exists():
        artifact = json.loads(args.output.read_text())
    artifact["cases"][f"layer-{args.layer}:{args.stage}"] = row
    expected = {f"layer-{layer}:{stage}" for layer in LAYERS for stage in STAGES}
    artifact["coverage_complete"] = set(artifact["cases"]) == expected
    atomic_write(args.output, artifact)
    print(json.dumps(row, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, choices=LAYERS, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--negative-output", type=Path)
    parser.add_argument("--wired-limit-gb", type=float, default=64.0)
    parser.add_argument("--cache-limit-gb", type=float, default=16.0)
    parser.add_argument("--capture-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.capture_child:
        return _child(args)
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        raise RuntimeError("set MTL_CAPTURE_ENABLED=1 before starting capture")
    trace = _validate_trace(args.trace)
    negative_output = args.negative_output or args.output.with_name(
        f"{args.output.stem}-negative-layer-{args.layer}-{args.stage}.json"
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(args.model),
        "--layer",
        str(args.layer),
        "--stage",
        args.stage,
        "--trace",
        str(trace),
        "--output",
        str(args.output),
        "--wired-limit-gb",
        str(args.wired_limit_gb),
        "--cache-limit-gb",
        str(args.cache_limit_gb),
        "--capture-child",
    ]
    return supervise_capture(
        command,
        trace_path=trace,
        evidence_path=negative_output,
        budget=CaptureBudget(),
        metadata={
            "capture_kind": "single-layer-packed-moe-operator",
            "layer": args.layer,
            "stage": args.stage,
            "trace_path": str(trace),
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
