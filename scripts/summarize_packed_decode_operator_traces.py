#!/usr/bin/env python3
"""Extract bounded custom-kernel label evidence from operator gputrace bundles."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from capture_budget import atomic_write
from capture_packed_decode_operator import DEFAULT_OUTPUT, LAYERS, STAGES


LABEL = re.compile(rb"custom_kernel_[A-Za-z0-9_]+")
MAX_SCANNED_FILE_BYTES = 100 << 20


def _labels(trace: Path) -> tuple[list[str], int, int]:
    labels = set()
    scanned_bytes = 0
    scanned_files = 0
    for path in trace.rglob("*"):
        if not path.is_file() or path.stat().st_size > MAX_SCANNED_FILE_BYTES:
            continue
        data = path.read_bytes()
        scanned_bytes += len(data)
        scanned_files += 1
        labels.update(value.decode("ascii") for value in LABEL.findall(data))
    return sorted(labels), scanned_files, scanned_bytes


def summarize(path: Path) -> dict:
    artifact = json.loads(path.read_text())
    for row in artifact["cases"].values():
        labels, files, size = _labels(Path(row["trace"]["path"]))
        row["trace_static_analysis"] = {
            "custom_kernel_labels": labels,
            "unique_custom_kernel_label_count": len(labels),
            "scanned_files": files,
            "scanned_bytes": size,
            "dispatch_count_requires_xcode_event_view": True,
        }
    layer_analysis = {}
    for layer in LAYERS:
        rows = {
            stage: artifact["cases"][f"layer-{layer}:{stage}"]
            ["trace_static_analysis"]["custom_kernel_labels"]
            for stage in STAGES
        }
        full = set(rows["full-ffn"])
        layer_analysis[str(layer)] = {
            "static_custom_kernel_inventory_counts": {
                stage: len(labels) for stage, labels in rows.items()
            },
            "static_inventory_identical_across_stages": len(
                {tuple(labels) for labels in rows.values()}
            )
            == 1,
            "full_output_equals_ffn_add": artifact["cases"][
                f"layer-{layer}:full-ffn"
            ]["output_hashes"]
            == artifact["cases"][f"layer-{layer}:ffn-add"]["output_hashes"],
            "standalone_custom_clamp_or_swiglu_kernel": any(
                ("clamp" in label or "swiglu" in label)
                and "gate_up_swiglu" not in label
                for label in full
            ),
        }
    artifact["static_trace_analysis"] = {
        "layers": layer_analysis,
        "all_static_inventories_identical_across_stages": all(
            row["static_inventory_identical_across_stages"]
            for row in layer_analysis.values()
        ),
        "all_full_outputs_equal_ffn_add": all(
            row["full_output_equals_ffn_add"] for row in layer_analysis.values()
        ),
        "standalone_custom_clamp_or_swiglu_in_static_glm53_inventory": any(
            row["standalone_custom_clamp_or_swiglu_kernel"]
            for row in layer_analysis.values()
        ),
        "labels_prove_dispatch": False,
        "scope": (
            "static resident metallib inventory only; stage dispatch counts, "
            "intermediate allocations, and gaps require Xcode event view"
        ),
    }
    atomic_write(path, artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = summarize(args.artifact)
    print(json.dumps(artifact["static_trace_analysis"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
