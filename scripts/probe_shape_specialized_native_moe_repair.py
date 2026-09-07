#!/usr/bin/env python3
"""Probe a shape-specialized repair for the native routed-MoE divergence.

The exact JIT Metal oracle compiles GLM-5.3's routed gate/up geometry as
constants.  The rejected AOT plan accepted those dimensions as runtime values
and first differed at decode step 29, layer 41.  Replay the same 68-step owned
activation corpus through the shape-specialized AOT kernel, then run the
official 16/128-token oracle before allowing any performance requalification.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import sys
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
    / "m3ultra512-shape-specialized-native-moe-repair-20260907.json"
)
LOCALIZATION_ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-packed-moe-divergence-localization-20260907.json"
)
TARGET_STEP = 68
EXPECTED_FIRST_DIVERGENCE = {
    "step": 29,
    "layer": 41,
    "stage": "routed_hidden",
}


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


def _load_helpers():
    for path in (str(SCRIPTS), str(NATIVE_PACKAGE)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import localize_native_packed_moe_divergence as localizer
    import probe_exact_sigmoid_gate_metal_barrier as oracle_probe

    plan_type, oracle_trace, d99f, native_probe, residual = (
        localizer._load_helpers()
    )
    return (
        localizer,
        oracle_probe,
        plan_type,
        oracle_trace,
        d99f,
        native_probe,
        residual,
    )


def _validate_source() -> dict:
    source = json.loads(LOCALIZATION_ARTIFACT.read_text())
    first = source["replay"]["first_divergence"]
    observed = {
        "step": first["step"],
        "layer": first["layer"],
        "stage": first["stage_localization"]["first_differing_stage"],
    }
    if not source["complete"] or not source["accepted"]:
        raise RuntimeError("native MoE localization source is not accepted")
    if observed != EXPECTED_FIRST_DIVERGENCE:
        raise RuntimeError(f"unexpected localization source: {observed}")
    return source


def _specialization_evidence(rows: list[dict]) -> dict:
    return {
        "plan_count": len(rows),
        "all_shape_specialized": bool(rows)
        and all(row["uses_shape_specialized_routed_gate_up"] for row in rows),
        "all_fixed_arena_invariants": bool(rows)
        and all(
            row["buffer_identities_stable"]
            and row["dynamic_allocation_count"] == 0
            and row["graph_node_count"] == 0
            and row["shape_discovery_count"] == 0
            and row["host_synchronization_count"] == 0
            and row["returned_intermediate_tensor_bytes"] == 0
            for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    args = parser.parse_args()
    artifact = {
        "schema": "glm53-shape-specialized-native-moe-repair-v1",
        "date": date.today().isoformat(),
        "complete": False,
        "accepted": False,
        "probe_only": True,
        "localization_source": str(LOCALIZATION_ARTIFACT.relative_to(ROOT)),
        "expected_first_divergence": EXPECTED_FIRST_DIVERGENCE,
        "repair": {
            "boundary": "routed gate+up projection and SwiGLU",
            "strategy": "compile-time GLM-5.3 execution geometry",
            "hidden_size": 4096,
            "intermediate_size": 2048,
            "intermediate_scale_rows": 16,
            "hidden_scale_rows": 32,
            "swiglu_limit": 10,
        },
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
        source = _validate_source()
        (
            localizer,
            oracle_probe,
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

        _progress("capture_exact_composition", target_step=TARGET_STEP)
        capture, generated, logits_hashes = localizer._capture_trajectory(
            model, processor, oracle_trace, residual, TARGET_STEP
        )
        generated_hash = hashlib.sha256(
            np.asarray(generated, dtype=np.int32).tobytes()
        ).hexdigest()
        artifact["capture_reproduces_localization_source"] = {
            "generated_tokens_sha256": generated_hash,
            "generated_tokens_exact": generated_hash
            == source["capture"]["generated_tokens_sha256"],
            "target_logits_hash": logits_hashes[TARGET_STEP],
            "target_logits_exact": logits_hashes[TARGET_STEP]
            == source["capture"]["target_logits_hash"],
        }

        _progress("replay_shape_specialized_native_plans", target_step=TARGET_STEP)
        divergence, comparisons, evidence = localizer._replay(
            model,
            capture,
            native_probe,
            plan_type,
            d99f,
            residual,
            TARGET_STEP,
        )
        artifact["owned_activation_replay"] = {
            "first_divergence": divergence,
            "exact_layer_step_comparisons": comparisons,
            "expected_comparisons": 42 * TARGET_STEP,
            "specialization": _specialization_evidence(evidence),
        }
        del capture
        gc.collect()
        mx.clear_cache()
        mx.synchronize()

        if divergence is None:
            reference_arm = residual.Arm("B1", True)
            _progress("official_oracle", arm="exact-composition", tokens=128)
            with residual._runtime(reference_arm):
                reference_oracle = oracle_probe._official_oracle(
                    model, processor, report
                )
            _progress("official_oracle", arm="shape-specialized-native", tokens=128)
            oracle_registry = native_probe._Registry(plan_type)
            with native_probe._native_moe_arm(oracle_registry):
                candidate_oracle = oracle_probe._official_oracle(
                    model, processor, report
                )
            oracle_evidence = oracle_registry.evidence()
            artifact["official_oracle"] = {
                "A_exact_composition": reference_oracle,
                "B_shape_specialized_native": candidate_oracle,
                "specialization": _specialization_evidence(oracle_evidence),
            }

        replay = artifact["owned_activation_replay"]
        reproduction = artifact["capture_reproduces_localization_source"]
        official = artifact.get("official_oracle", {})
        artifact["acceptance"] = {
            "localization_source_reproduced": (
                reproduction["generated_tokens_exact"]
                and reproduction["target_logits_exact"]
            ),
            "all_2856_owned_layer_steps_byte_exact": (
                replay["first_divergence"] is None
                and replay["exact_layer_step_comparisons"]
                == replay["expected_comparisons"]
            ),
            "all_42_plans_use_shape_specialized_kernel": replay[
                "specialization"
            ]["plan_count"]
            == 42
            and replay["specialization"]["all_shape_specialized"],
            "fixed_arena_invariants_preserved": replay["specialization"][
                "all_fixed_arena_invariants"
            ],
            "official_16_128_oracle_exact_both_arms": bool(official)
            and official["A_exact_composition"]["all_full_vocab_logits_hashes_match"]
            and official["B_shape_specialized_native"][
                "all_full_vocab_logits_hashes_match"
            ],
            "official_native_plans_all_shape_specialized": bool(official)
            and official["specialization"]["plan_count"] == 42
            and official["specialization"]["all_shape_specialized"],
        }
        artifact["process_peak_memory_bytes"] = int(mx.get_peak_memory())
        artifact["complete"] = True
        artifact["accepted"] = all(artifact["acceptance"].values())
        artifact["decision"] = (
            "advance_shape_specialized_native_moe_to_performance_requalification"
            if artifact["accepted"]
            else "stop_or_relocalize_native_routed_hidden_repair"
        )
    except Exception as error:
        artifact.update(
            complete=True,
            accepted=False,
            decision="shape_specialized_native_moe_repair_probe_failed",
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
    localizer = sys.modules.get("localize_native_packed_moe_divergence")
    if localizer is None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.output)
    else:
        localizer._atomic_write(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": artifact["complete"],
                "accepted": artifact["accepted"],
                "decision": artifact["decision"],
                "failed_gates": [
                    name
                    for name, passed in artifact.get("acceptance", {}).items()
                    if not passed
                ],
            },
            indent=2,
        )
    )
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
