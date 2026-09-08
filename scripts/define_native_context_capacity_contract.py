#!/usr/bin/env python3
"""Record the deterministic 512K native context design contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from glm53_flash_mlx.context_capacity import (
    DESIGN_CODING_AGENT_PROMPT_TOKENS,
    DESIGN_MAX_GENERATION_TOKENS,
    DESIGN_TOTAL_CONTEXT_TOKENS,
    NATIVE_CONTEXT_CAPACITY_CONTRACT,
    native_prefill_profile_geometries,
    plan_native_context_capacity,
)
from glm53_flash_mlx.server import (
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_GENERATION_TOKENS,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-context-capacity-contract-20260908.json"
)


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_artifact() -> dict[str, object]:
    plan = plan_native_context_capacity()
    prefill = native_prefill_profile_geometries()
    checks = {
        "total_context_is_512k": plan.total_context_tokens == 512 << 10,
        "coding_agent_prompt_is_320k": (
            plan.coding_agent_prompt_tokens == 320 << 10
        ),
        "max_generation_is_128k": plan.max_generation_tokens == 128 << 10,
        "target_has_64k_headroom": plan.target_headroom_tokens == 64 << 10,
        "pool_count_is_tail_exact_kpool4": plan.logical_pool_rows == 131_072,
        "physical_cache_is_256_token_aligned": (
            plan.physical_capacity_tokens == plan.total_context_tokens
            and plan.physical_pool_rows % 64 == 0
        ),
        "dsa_logits_workspace_is_at_most_64mib": (
            plan.fp32_logits_workspace_bytes <= plan.max_workspace_bytes
            == 64 << 20
        ),
        "512k_q256_uses_128_rows_two_blocks": (
            plan.prefill_query_block_rows == 128
            and plan.prefill_query_block_count == 2
        ),
        "selected_output_width_remains_2051": (
            plan.selected_token_width == 2_051
        ),
        "128k_generation_materializes_512_times": (
            plan.generation_materialization_count == 512
        ),
        "current_native_decode_geometry_is_explicitly_unqualified": (
            not plan.native_decode_geometry_qualified
            and plan.native_pool_row_deficit == 65_472
        ),
        "prefill_profiles_are_32k_128k_320k": (
            tuple(row["context_tokens"] for row in prefill)
            == (32 << 10, 128 << 10, 320 << 10)
        ),
        "production_defaults_are_not_promoted_by_contract": (
            DEFAULT_MAX_CONTEXT_TOKENS == 36_864
            and DEFAULT_MAX_GENERATION_TOKENS == 4_096
            and plan.production_default_changed is False
        ),
    }
    accepted = all(checks.values())
    return {
        "schema": "glm53-native-context-capacity-contract-v1",
        "date": "2026-09-08",
        "complete": True,
        "accepted": accepted,
        "decision": (
            "advance_to_native_prefill_profiling_and_expand_decode_geometry"
            if accepted
            else "stop_native_512k_capacity_design"
        ),
        "contract": NATIVE_CONTEXT_CAPACITY_CONTRACT,
        "targets": {
            "total_context_tokens": DESIGN_TOTAL_CONTEXT_TOKENS,
            "coding_agent_prompt_tokens": DESIGN_CODING_AGENT_PROMPT_TOKENS,
            "max_generation_tokens": DESIGN_MAX_GENERATION_TOKENS,
        },
        "plan": plan.descriptor(),
        "native_prefill_profile_geometries": list(prefill),
        "checks": checks,
        "runtime_changes": {
            "production_context_default": False,
            "production_generation_default": False,
            "server_admission": False,
            "cache_abi": False,
            "native_decode_qualified_range": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    artifact = build_artifact()
    _atomic_write(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": artifact["complete"],
                "accepted": artifact["accepted"],
                "decision": artifact["decision"],
            }
        )
    )
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
