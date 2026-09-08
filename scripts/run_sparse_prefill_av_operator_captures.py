#!/usr/bin/env python3
"""Capture Direct and compact sparse-prefill AV operators separately."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from capture_sparse_prefill_av_operator import ARMS, DEFAULT_OUTPUT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    existing = {}
    if args.output.exists():
        existing = json.loads(args.output.read_text()).get("cases", {})
    env = dict(os.environ, MTL_CAPTURE_ENABLED="1")
    for arm in ARMS:
        if arm in existing:
            print(json.dumps({"arm": arm, "status": "already-complete"}), flush=True)
            continue
        trace = args.trace_dir / f"sparse-prefill-av-{arm}.gputrace"
        command = [
            sys.executable,
            str(Path(__file__).with_name("capture_sparse_prefill_av_operator.py")),
            "--arm",
            arm,
            "--trace",
            str(trace),
            "--output",
            str(args.output),
        ]
        print(json.dumps({"arm": arm, "status": "starting"}), flush=True)
        completed = subprocess.run(command, env=env, check=False)
        if completed.returncode:
            return completed.returncode
    artifact = json.loads(args.output.read_text())
    if artifact["complete"]:
        analyzed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name(
                    "analyze_sparse_prefill_av_operator_captures.py"
                )),
                "--artifact",
                str(args.output),
            ],
            check=False,
        )
        if analyzed.returncode:
            return analyzed.returncode
        artifact = json.loads(args.output.read_text())
    print(json.dumps({
        "output": str(args.output),
        "complete": artifact["complete"],
        "accepted": artifact["accepted"],
        "decision": artifact["decision"],
    }))
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
