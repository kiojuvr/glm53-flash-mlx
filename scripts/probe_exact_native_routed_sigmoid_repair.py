#!/usr/bin/env python3
"""Probe the exact fast-BF16 repair for native routed-MoE sigmoid."""

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
    / "m3ultra512-exact-native-routed-fast-sigmoid-repair-20260907.json"
)
LOCALIZATION_ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-routed-hidden-numerical-localization-20260907.json"
)
SIGMOID_FORMULA_ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-routed-sigmoid-formula-sweep-20260907.json"
)
TARGET_STEP = 68


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


def _atomic_write(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True)
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


def _validate_sources() -> dict:
    localization = json.loads(LOCALIZATION_ARTIFACT.read_text())
    formula = json.loads(SIGMOID_FORMULA_ARTIFACT.read_text())
    evidence = localization["evidence"]
    if not localization["complete"] or not localization["accepted"]:
        raise RuntimeError("routed-hidden localization source is not accepted")
    if evidence["first_differing_stage"] != "sigmoid_bf16":
        raise RuntimeError("routed-hidden localization is not a sigmoid failure")
    if not evidence["stages"]["gate_projection_bf16"]["byte_identical"]:
        raise RuntimeError("gate projection must be exact before sigmoid repair")
    if not evidence["stages"]["up_projection_bf16"]["byte_identical"]:
        raise RuntimeError("up projection must be exact before sigmoid repair")
    if not formula["complete"] or not formula["accepted"]:
        raise RuntimeError("native sigmoid formula sweep is not accepted")
    if formula["selected_formula"] != "fast_bf16":
        raise RuntimeError("fast BF16 sigmoid is not the unique recorded formula")
    if formula["exact_formulas"] != ["fast_bf16"]:
        raise RuntimeError("native sigmoid formula sweep is not unique")
    return {
        "localization": localization,
        "formula": formula,
    }


def _plan_evidence(rows: list[dict]) -> dict:
    return {
        "plan_count": len(rows),
        "all_use_shape_specialized_kernel": bool(rows)
        and all(row["uses_shape_specialized_routed_gate_up"] for row in rows),
        "all_use_fast_bf16_routed_sigmoid": bool(rows)
        and all(row["uses_fast_bf16_routed_sigmoid"] for row in rows),
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
        "schema": "glm53-exact-native-routed-fast-sigmoid-repair-v2",
        "date": date.today().isoformat(),
        "complete": False,
        "accepted": False,
        "probe_only": True,
        "target_step": TARGET_STEP,
        "source_artifacts": {
            "numerical_localization": str(LOCALIZATION_ARTIFACT.relative_to(ROOT)),
            "native_sigmoid_formula_sweep": str(
                SIGMOID_FORMULA_ARTIFACT.relative_to(ROOT)
            ),
        },
        "repair": {
            "stage": "routed sigmoid BF16",
            "formula": "fast_bf16",
            "change": "metal::exp to metal::fast::exp",
            "generic_kernel_changed": False,
            "shared_expert_kernel_changed": False,
            "projection_reduction_changed": False,
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
        sources = _validate_sources()
        artifact["source_evidence"] = {
            "first_differing_stage": sources["localization"]["evidence"][
                "first_differing_stage"
            ],
            "different_sigmoid_elements": sources["localization"]["evidence"][
                "stages"
            ]["sigmoid_bf16"]["different_elements"],
            "first_difference": sources["localization"]["evidence"]["stages"][
                "sigmoid_bf16"
            ]["first_difference"],
            "selected_formula": sources["formula"]["selected_formula"],
            "formula_domain_elements": sources["formula"]["domain"]["elements"],
        }
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
        artifact["capture"] = {
            "generated_tokens_sha256": hashlib.sha256(
                np.asarray(generated, dtype=np.int32).tobytes()
            ).hexdigest(),
            "target_logits_hash": logits_hashes[TARGET_STEP],
            "owned_activation_count": sum(
                len(values) for values in capture.inputs.values()
            ),
            "owned_output_count": sum(
                len(values) for values in capture.outputs.values()
            ),
        }
        _progress("replay_fast_bf16_sigmoid_native_plans", target_step=TARGET_STEP)
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
            "plan_evidence": _plan_evidence(evidence),
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
            _progress("official_oracle", arm="fast-bf16-sigmoid-native", tokens=128)
            oracle_registry = native_probe._Registry(plan_type)
            with native_probe._native_moe_arm(oracle_registry):
                candidate_oracle = oracle_probe._official_oracle(
                    model, processor, report
                )
            artifact["official_oracle"] = {
                "A_exact_composition": reference_oracle,
                "B_fast_bf16_sigmoid_native": candidate_oracle,
                "plan_evidence": _plan_evidence(oracle_registry.evidence()),
            }

        replay = artifact["owned_activation_replay"]
        plans = replay["plan_evidence"]
        official = artifact.get("official_oracle", {})
        artifact["acceptance"] = {
            "all_2856_owned_layer_steps_byte_exact": (
                replay["first_divergence"] is None
                and replay["exact_layer_step_comparisons"]
                == replay["expected_comparisons"]
            ),
            "all_42_plans_use_fast_bf16_routed_sigmoid": plans["plan_count"] == 42
            and plans["all_use_shape_specialized_kernel"]
            and plans["all_use_fast_bf16_routed_sigmoid"],
            "fixed_arena_invariants_preserved": plans[
                "all_fixed_arena_invariants"
            ],
            "official_16_128_oracle_exact_both_arms": bool(official)
            and official["A_exact_composition"]["all_full_vocab_logits_hashes_match"]
            and official["B_fast_bf16_sigmoid_native"][
                "all_full_vocab_logits_hashes_match"
            ],
            "official_native_plans_all_fast_bf16": bool(official)
            and official["plan_evidence"]["plan_count"] == 42
            and official["plan_evidence"]["all_use_fast_bf16_routed_sigmoid"],
        }
        artifact["process_peak_memory_bytes"] = int(mx.get_peak_memory())
        artifact["complete"] = True
        artifact["accepted"] = all(artifact["acceptance"].values())
        artifact["decision"] = (
            "advance_exact_native_moe_to_performance_requalification"
            if artifact["accepted"]
            else "stop_or_relocalize_fast_bf16_native_sigmoid_repair"
        )
    except Exception as error:
        artifact.update(
            complete=True,
            accepted=False,
            decision="exact_native_routed_sigmoid_repair_probe_failed",
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
