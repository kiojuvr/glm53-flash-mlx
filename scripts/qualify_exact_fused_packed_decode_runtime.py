#!/usr/bin/env python3
"""Qualify the exact fused packed-decode topology as the real opt-in runtime.

The underlying qualification deliberately reuses the original packed runtime
suite: Direct remains the independent correctness oracle, prefill must retain
Direct semantics, decode runs for 4,096 steps, the 256K synthetic frontier and
RAM APC are exercised, and a fresh server process must become healthy.  This
wrapper adds the v2 topology provenance and the 15 tok/s promotion gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-exact-fused-packed-decode-runtime-20260907.json"
)
RESIDUAL_SOURCE = (
    ROOT
    / "bench-results"
    / "m3ultra512-residual-packed-decode-moe-fusion-20260901.json"
)
COMPOSITION_SOURCE = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-fused-decode-composition-20260907.json"
)
TARGET_2K_TPS = 15.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _validate_sources() -> dict:
    residual = json.loads(RESIDUAL_SOURCE.read_text())
    composition = json.loads(COMPOSITION_SOURCE.read_text())
    if residual.get("selected_aggregation") != "B1":
        raise RuntimeError("qualified residual aggregation is not B1")
    if not residual.get("aggregation_exact", {}).get("B1"):
        raise RuntimeError("B1 weighted reduction source is not exact")
    if not residual.get("shared_exact"):
        raise RuntimeError("shared gate/up/SiLU source is not exact")
    if not all(residual.get("correctness", {}).values()):
        raise RuntimeError("residual fused source correctness is incomplete")
    if not composition.get("accepted") or not all(
        composition.get("acceptance", {}).values()
    ):
        raise RuntimeError("fused composition source was not accepted")
    return {
        "residual": {
            "path": str(RESIDUAL_SOURCE.relative_to(ROOT)),
            "sha256": _sha256(RESIDUAL_SOURCE),
            "selected_aggregation": "B1",
            "shared_fused": True,
        },
        "composition": {
            "path": str(COMPOSITION_SOURCE.relative_to(ROOT)),
            "sha256": _sha256(COMPOSITION_SOURCE),
            "decision": composition["decision"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    parser.add_argument("--skip-server-smoke", action="store_true")
    args = parser.parse_args()

    sources = _validate_sources()
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import probe_packed_decode_runtime as qualification

    delegated = [
        str(Path(qualification.__file__)),
        str(args.model),
        "--output",
        str(args.output),
        "--wired-limit-gb",
        str(args.wired_limit_gb),
        "--cache-limit-gb",
        str(args.cache_limit_gb),
    ]
    if args.skip_server_smoke:
        delegated.append("--skip-server-smoke")
    previous = sys.argv
    try:
        sys.argv = delegated
        qualification_status = qualification.main()
    finally:
        sys.argv = previous

    artifact = json.loads(args.output.read_text())
    from glm53_flash_mlx.abi import PACKED_DECODE_KERNEL_ABI

    short_tps = artifact.get("packed_decode", {}).get("frontier", {}).get(
        "direct:2049", {}
    ).get("tokens_per_second", 0.0)
    promotion = {
        "source_artifacts": sources,
        "packed_decode_kernel_abi": PACKED_DECODE_KERNEL_ABI,
        "runtime_topology": [
            "routed exact gate+up+SwiGLU",
            "routed existing BF16 down projection",
            "routed exact FP32 weighted top-8 reduction",
            "shared exact gate+up+SwiGLU",
            "shared existing BF16 down projection",
        ],
        "prefill_semantics": "Direct packed-bank semantics",
        "grouped_kernel_calls": 0,
    }
    promotion_acceptance = {
        "delegated_runtime_qualification_accepted": qualification_status == 0
        and artifact.get("acceptance", {}).get("accepted") is True,
        "2k_decode_at_least_15_tps": short_tps >= TARGET_2K_TPS,
        "v2_exact_fused_kernel_abi": PACKED_DECODE_KERNEL_ABI.startswith(
            "glm53-packed-selected8-fp8-v2-"
        ),
        "source_fused_composition_prequalified": True,
    }
    artifact["base_schema"] = artifact["schema"]
    artifact["schema"] = "glm53-exact-fused-packed-decode-runtime-v2"
    artifact["promotion"] = promotion
    artifact["promotion_acceptance"] = promotion_acceptance
    artifact["accepted"] = all(promotion_acceptance.values())
    artifact["decision"] = (
        "keep_exact_fused_packed_decode_runtime"
        if artifact["accepted"]
        else "stop_or_requalify_exact_fused_packed_decode_runtime"
    )
    artifact["runtime_changes"] = {
        "packed_decode_execution": True,
        "packed_decode_kernel_abi": True,
        "default_backend": False,
        "prompt_admission": False,
        "cache_abi": False,
        "grouped_backend": False,
    }
    _atomic_write(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": artifact.get("complete"),
                "accepted": artifact["accepted"],
                "decision": artifact["decision"],
                "2k_tokens_per_second": short_tps,
                "failed_gates": [
                    name
                    for name, passed in promotion_acceptance.items()
                    if not passed
                ],
            },
            indent=2,
        )
    )
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
