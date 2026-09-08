#!/usr/bin/env python3
"""Build the next native prefill DSA plan from measured exactness evidence."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from glm53_flash_mlx.native_prefill_plan import (
    build_native_sparse_prefill_attention_plan,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REORDERING = (
    ROOT
    / "bench-results"
    / "m3ultra512-sparse-prefill-reordering-equivalence-20260908.json"
)
DEFAULT_MICROTILE = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-dsa-prefill-streaming-microtile-20260908.json"
)
DEFAULT_LOCALIZATION = (
    ROOT
    / "bench-results"
    / "m3ultra512-sparse-prefill-attention-reduction-localization-20260908.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-exact-sparse-prefill-attention-plan-20260908.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reordering", type=Path, default=DEFAULT_REORDERING)
    parser.add_argument("--microtile", type=Path, default=DEFAULT_MICROTILE)
    parser.add_argument("--localization", type=Path, default=DEFAULT_LOCALIZATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    plan = build_native_sparse_prefill_attention_plan(
        _load(args.reordering), _load(args.microtile), _load(args.localization)
    )
    checks = {
        "selected_projection_reordering_is_byte_exact": (
            plan.selected_projection_reordering_exact
        ),
        "ordinary_compact_sdpa_is_forbidden": (
            not plan.ordinary_compact_sdpa_allowed
        ),
        "compact_qk_and_softmax_are_exact": (
            plan.compact_qk_exact and plan.compact_precise_softmax_exact
        ),
        "compact_av_is_forbidden": not plan.compact_av_allowed,
        "virtual_full_kv_av_reduction_is_required": (
            plan.requires_virtual_full_kv_av_reduction_topology
        ),
        "total_native_scratch_at_most_256mib": (
            plan.total_scratch_bytes <= plan.max_scratch_bytes
        ),
        "avoids_at_least_40gib_full_projected_kv": (
            plan.full_projected_key_value_bytes_avoided >= 40 << 30
        ),
        "avoids_128mib_q256_full_sparse_mask": (
            plan.full_sparse_mask_bytes_avoided == 128 << 20
        ),
        "production_admission_unchanged": (
            not plan.production_admission_changed
        ),
    }
    artifact = {
        "schema": "glm53-exact-sparse-prefill-attention-plan-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": all(checks.values()),
        "decision": "implement_exact_virtual_mask_sparse_attention_primitive",
        "plan": plan.descriptor(),
        "checks": checks,
        "source_evidence": {
            "reordering": str(args.reordering),
            "microtile": str(args.microtile),
            "localization": str(args.localization),
        },
    }
    _atomic_write(args.output, artifact)
    print(json.dumps({
        "output": str(args.output), "complete": artifact["complete"],
        "accepted": artifact["accepted"], "decision": artifact["decision"]
    }))
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
