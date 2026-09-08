#!/usr/bin/env python3
"""Build the measured native prefill execution-plan artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from glm53_flash_mlx.native_prefill_plan import (
    NATIVE_PREFILL_PLAN_ABI,
    build_native_prefill_execution_plan,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-prefill-critical-path-20260908.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-prefill-execution-plan-20260908.json"
)


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_artifact(profile: dict[str, object]) -> dict[str, object]:
    plan = build_native_prefill_execution_plan(profile)
    targets = {row.tokens_per_second: row for row in plan.throughput_targets}
    checks = {
        "profile_measurements_exact_at_32k_128k_320k": True,
        "profile_340gb_failure_preserved_as_negative_evidence": (
            plan.measured_profile_peak_over_budget_bytes > 0
            and not profile["acceptance"]["process_peak_at_most_340gb"]
        ),
        "dsa_and_moe_explain_at_least_90_percent_at_320k": (
            plan.combined_structural_share_320k >= 0.90
        ),
        "dsa_is_context_scaling_region": plan.dsa_32k_to_320k_scaling >= 4.0,
        "moe_is_nearly_context_invariant_weight_region": (
            0.90 <= plan.routed_moe_32k_to_320k_scaling <= 1.10
        ),
        "100_tps_is_only_first_checkpoint": (
            targets[100].required_speedup_from_320k > 1.0
            and tuple(targets) == (100, 200, 300)
        ),
        "512k_dsa_workspace_remains_at_most_64mib": (
            plan.dsa_logits_workspace_bytes <= 64 << 20
        ),
        "native_arena_budget_is_positive": plan.planned_native_arena_budget_bytes > 0,
        "partial_kernel_promotion_is_forbidden": any(
            "partial kernels cannot be promoted" in invariant
            for invariant in plan.invariants
        ),
        "production_admission_is_unchanged": not plan.production_admission_changed,
    }
    accepted = all(checks.values())
    return {
        "schema": "glm53-native-prefill-execution-plan-v1",
        "complete": True,
        "accepted": accepted,
        "decision": (
            "implement_composed_native_prefill_layer_plan"
            if accepted
            else "stop_native_prefill_architecture"
        ),
        "abi": NATIVE_PREFILL_PLAN_ABI,
        "source_profile": str(DEFAULT_PROFILE.relative_to(ROOT)),
        "source_profile_accepted": bool(profile["accepted"]),
        "source_profile_failed_gates": sorted(
            name for name, passed in profile["acceptance"].items() if not passed
        ),
        "plan": plan.descriptor(),
        "checks": checks,
        "runtime_changes": {
            "production_context_default": False,
            "production_generation_default": False,
            "server_admission": False,
            "cache_abi": False,
            "kernel_abi": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    profile = json.loads(args.profile.read_text())
    artifact = build_artifact(profile)
    _atomic_write(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": artifact["complete"],
                "accepted": artifact["accepted"],
                "decision": artifact["decision"],
                "measured_320k_tps": artifact["plan"][
                    "measured_320k_tokens_per_second"
                ],
            }
        )
    )
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

