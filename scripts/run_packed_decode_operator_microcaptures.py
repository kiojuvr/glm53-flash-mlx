#!/usr/bin/env python3
"""Run the bounded layer/stage packed-MoE microcapture matrix sequentially."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from capture_packed_decode_operator import DEFAULT_OUTPUT, LAYERS, STAGES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MTL_CAPTURE_ENABLED="1")
    existing = {}
    if args.output.exists():
        existing = json.loads(args.output.read_text()).get("cases", {})
    for layer in LAYERS:
        for stage in STAGES:
            key = f"layer-{layer}:{stage}"
            if key in existing:
                print(json.dumps({"case": key, "status": "already-complete"}), flush=True)
                continue
            trace = args.trace_dir / f"layer-{layer}-{stage}.gputrace"
            command = [
                sys.executable,
                str(Path(__file__).with_name("capture_packed_decode_operator.py")),
                str(args.model),
                "--layer",
                str(layer),
                "--stage",
                stage,
                "--trace",
                str(trace),
                "--output",
                str(args.output),
            ]
            print(json.dumps({"case": key, "status": "starting"}), flush=True)
            completed = subprocess.run(command, env=env, check=False)
            if completed.returncode != 0:
                return completed.returncode
    return subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("summarize_packed_decode_operator_traces.py")),
            "--artifact",
            str(args.output),
        ],
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
