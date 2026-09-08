#!/usr/bin/env python3
"""Attach captured AV pipeline-resource evidence to the repository artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from capture_budget import atomic_write
from capture_sparse_prefill_av_operator import (
    ARMS,
    DEFAULT_OUTPUT,
    _pipeline_labels,
)


EXPECTED_PIPELINE = (
    "steel_gemm_fused_nn_bfloat16_bfloat16_"
    "bm64_bn64_bk16_wm1_wn2_has_batch_n_use_out_source_n_do_axpby_n_"
    "align_M_n_align_N_t_align_K_t"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text())
    if set(artifact.get("cases", {})) != set(ARMS):
        raise RuntimeError("both Direct and compact captures are required")
    topologies = {}
    for arm in ARMS:
        trace = Path(artifact["cases"][arm]["trace"]["path"])
        labels = _pipeline_labels(trace)
        artifact["cases"][arm]["captured_pipeline_resource_labels"] = labels
        topologies[arm] = labels
    same = topologies["direct"] == topologies["compact"]
    expected = topologies["direct"] == [EXPECTED_PIPELINE]
    checks = {
        "both_capture_bundles_available": all(
            Path(artifact["cases"][arm]["trace"]["path"]).exists()
            for arm in ARMS
        ),
        "single_av_gemm_pipeline_resource_per_arm": all(
            len(topologies[arm]) == 1 for arm in ARMS
        ),
        "direct_and_compact_select_same_pipeline": same,
        "captured_pipeline_is_bf16_bm64_bn64_bk16": expected,
        "only_reduction_k_geometry_differs": (
            artifact["cases"]["direct"]["reduction_k"] == 32_768
            and artifact["cases"]["compact"]["reduction_k"] == 2_051
        ),
    }
    artifact["dynamic_topology"] = {
        "checks": checks,
        "direct_pipeline_resources": topologies["direct"],
        "compact_pipeline_resources": topologies["compact"],
        "shared_bk": 16,
        "cause": "same Steel pipeline, different physical-K lane grouping",
        "required_native_topology": (
            "traverse selected values in original physical BK16 groups; "
            "preserve zero lanes and skip only wholly empty groups"
        ),
        "dispatch_association_requires_xcode_view": True,
    }
    artifact["accepted"] = artifact["accepted"] and all(checks.values())
    artifact["decision"] = (
        "implement_virtual_physical_bk16_av_reduction"
        if artifact["accepted"]
        else "inspect_capture_in_xcode_before_av_implementation"
    )
    atomic_write(args.artifact, artifact)
    print(json.dumps({
        "output": str(args.artifact),
        "complete": artifact["complete"],
        "accepted": artifact["accepted"],
        "decision": artifact["decision"],
        "pipelines": topologies,
    }))
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
