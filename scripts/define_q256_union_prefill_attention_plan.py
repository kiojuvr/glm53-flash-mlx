#!/usr/bin/env python3
"""Build the bounded Q256 union prefill-attention execution contract."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from glm53_flash_mlx.native_prefill_plan import (
    build_native_q256_union_attention_plan,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REUSE = ROOT / "bench-results" / (
    "m3ultra512-selected-kv-projection-reuse-frontier-20260908.json"
)
DEFAULT_UNION = ROOT / "bench-results" / (
    "m3ultra512-device-resident-q256-selected-union-20260908.json"
)
DEFAULT_INDIRECT = ROOT / "bench-results" / (
    "m3ultra512-indirect-q256-selected-latent-20260908.json"
)
DEFAULT_PREFILL = ROOT / "bench-results" / (
    "m3ultra512-native-prefill-execution-plan-20260908.json"
)
DEFAULT_OUTPUT = ROOT / "bench-results" / (
    "m3ultra512-q256-union-prefill-attention-plan-20260908.json"
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
    parser.add_argument("--reuse", type=Path, default=DEFAULT_REUSE)
    parser.add_argument("--union", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--indirect", type=Path, default=DEFAULT_INDIRECT)
    parser.add_argument("--prefill", type=Path, default=DEFAULT_PREFILL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    plan = build_native_q256_union_attention_plan(
        _load(args.reuse), _load(args.union), _load(args.indirect),
        _load(args.prefill),
    )
    checks = {
        "projection_tile_is_65536_rows": plan.projection_tile_rows == 65_536,
        "512k_requires_at_most_8_union_tiles": (
            plan.maximum_union_tiles_512k <= 8
        ),
        "phase_aliased_arena_fits_native_budget": (
            plan.maximum_phase_arena_bytes <= plan.available_native_arena_bytes
        ),
        "forbids_at_least_40gib_full_projected_kv": (
            plan.full_512k_projected_key_value_bytes_forbidden >= 40 << 30
        ),
        "forbids_per_query_selected_kv_over_native_budget": (
            plan.per_query_selected_key_value_bytes_forbidden
            > plan.available_native_arena_bytes
        ),
        "measured_q256_union_is_320k_minimum": (
            plan.measured_320k_union_vs_full_history < 0.40
        ),
        "device_union_and_indirect_gather_under_5ms_each": (
            plan.measured_320k_union_build_ms <= 5.0
            and plan.measured_320k_indirect_gather_ms <= 5.0
        ),
        "exact_two_pass_boundaries_are_explicit": (
            "key_tiles_project_once_and_fuse_into_query_edge_qk"
            in plan.execution_phases
            and "direct_order_virtual_bk16_av" in plan.execution_phases
        ),
        "production_admission_unchanged": not plan.production_admission_changed,
    }
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-q256-union-prefill-attention-plan-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "implement_exact_fused_projected_qk_union_tile"
            if accepted
            else "stop_q256_union_attention_architecture"
        ),
        "plan": plan.descriptor(),
        "checks": checks,
        "source_evidence": {
            "projection_reuse": str(args.reuse),
            "device_union": str(args.union),
            "indirect_gather": str(args.indirect),
            "native_prefill": str(args.prefill),
        },
        "runtime_changes": False,
    }
    _atomic_write(args.output, artifact)
    print(json.dumps({
        "output": str(args.output),
        "complete": artifact["complete"],
        "accepted": artifact["accepted"],
        "decision": artifact["decision"],
        "maximum_phase_arena_bytes": plan.maximum_phase_arena_bytes,
    }))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
