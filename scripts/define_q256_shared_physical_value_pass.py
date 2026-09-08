#!/usr/bin/env python3
"""Define the shared physical-BK16 Q256 value-pass contract."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from glm53_flash_mlx.native_prefill_plan import (
    build_native_q256_shared_value_pass_plan,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_Q4 = ROOT / "bench-results" / (
    "m3ultra512-q4-union-tiled-full-attention-20260909.json"
)
DEFAULT_Q256 = ROOT / "bench-results" / (
    "m3ultra512-q256-union-tiled-full-attention-20260909.json"
)
DEFAULT_QK320 = ROOT / "bench-results" / (
    "m3ultra512-q256-projected-qk-320k-tiles-20260909.json"
)
DEFAULT_PREFILL = ROOT / "bench-results" / (
    "m3ultra512-native-prefill-execution-plan-20260908.json"
)
DEFAULT_OUTPUT = ROOT / "bench-results" / (
    "m3ultra512-q256-shared-physical-value-pass-plan-20260909.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q4", type=Path, default=DEFAULT_Q4)
    parser.add_argument("--q256", type=Path, default=DEFAULT_Q256)
    parser.add_argument("--qk320", type=Path, default=DEFAULT_QK320)
    parser.add_argument("--prefill", type=Path, default=DEFAULT_PREFILL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    plan = build_native_q256_shared_value_pass_plan(
        _load(args.q4), _load(args.q256), _load(args.qk320), _load(args.prefill)
    )
    checks = {
        "q256_is_four_bm64_query_blocks": (
            plan.query_block_rows == 64 and plan.query_blocks == 4
        ),
        "512k_is_eight_65k_physical_tiles": (
            plan.physical_value_tile_rows == 65_536
            and plan.maximum_physical_tiles_512k == 8
        ),
        "eliminates_at_least_8gib_query_local_selected_v": (
            plan.rejected_query_local_selected_value_bytes >= 8 << 30
        ),
        "shared_value_tile_is_exactly_1gib": (
            plan.projected_value_tile_bytes == 1 << 30
        ),
        "fp32_accumulator_is_exactly_8mib": (
            plan.fp32_attention_accumulator_bytes == 8 << 20
        ),
        "phase_aliased_arena_fits_native_budget": (
            plan.maximum_phase_arena_bytes <= plan.available_native_arena_bytes
        ),
        "q256_rejection_is_preserved_as_design_evidence": (
            plan.measured_q256_rejected_ms > plan.measured_q256_direct_ms
        ),
        "production_admission_unchanged": not plan.production_admission_changed,
    }
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-q256-shared-physical-value-pass-plan-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "implement_exact_bm64_shared_physical_value_tile"
            if accepted
            else "stop_q256_shared_physical_value_architecture"
        ),
        "plan": plan.descriptor(),
        "checks": checks,
        "runtime_changes": False,
    }
    _atomic_write(args.output, artifact)
    print(json.dumps({
        "output": str(args.output),
        "complete": True,
        "accepted": accepted,
        "decision": artifact["decision"],
        "maximum_phase_arena_bytes": plan.maximum_phase_arena_bytes,
    }))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
