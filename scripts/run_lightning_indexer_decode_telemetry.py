#!/usr/bin/env python3
"""Run and merge all resumable Lightning Indexer bounded telemetry contexts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
CONTEXTS = (2_048, 32_768, 131_072, 262_144)
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_PROFILE = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-lightning-indexer-decode-critical-path-20260905.json"
)


def _complete(path: Path) -> bool:
    return path.exists() and bool(json.loads(path.read_text()).get("complete"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--profile-output", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    capture = REPOSITORY / "scripts" / "capture_lightning_indexer_decode_telemetry.py"
    profile = REPOSITORY / "scripts" / "profile_lightning_indexer_decode_critical_path.py"
    for context in CONTEXTS:
        telemetry = (
            REPOSITORY
            / "bench-results"
            / f"m3ultra512-lightning-indexer-telemetry-{context}-20260905.json"
        )
        if not _complete(telemetry):
            subprocess.run(
                [
                    sys.executable,
                    str(capture),
                    str(args.model),
                    "--context",
                    str(context),
                    "--trace-dir",
                    str(args.trace_root / str(context)),
                    "--output",
                    str(telemetry),
                    "--repetitions",
                    str(args.repetitions),
                ],
                check=True,
            )
        subprocess.run(
            [
                sys.executable,
                str(profile),
                str(args.model),
                "--phase",
                "merge-telemetry",
                "--output",
                str(args.profile_output),
                "--telemetry",
                str(telemetry),
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            str(profile),
            str(args.model),
            "--phase",
            "finalize",
            "--output",
            str(args.profile_output),
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
