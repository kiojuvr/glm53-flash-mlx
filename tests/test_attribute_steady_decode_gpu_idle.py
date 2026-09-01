from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts" / "attribute_steady_decode_gpu_idle.py"
ARTIFACT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-steady-decode-dynamic-idle-attribution-20260901.json"
)


def _module():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("dynamic_idle_attribution_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gpu_gap_splits_application_and_driver_time():
    module = _module()
    gap = module.GpuGap(
        previous_end_ns=100,
        next_commit_ns=160,
        next_start_ns=220,
        previous_frame=10,
        next_frame=12,
        previous_stage="previous",
        next_stage="next",
        commit_source="application_submission",
    )
    assert gap.total_ns == 120
    assert gap.application_starvation_ns == 60
    assert gap.driver_or_dependency_ns == 60


def test_overlapping_gpu_events_are_merged_before_gap_reconstruction():
    module = _module()
    events = [
        module.GpuEvent(0, 10, 0, 1, 1, "one", "application_submission"),
        module.GpuEvent(5, 15, 0, 2, 2, "two", "application_submission"),
        module.GpuEvent(25, 30, 20, 3, 3, "three", "application_submission"),
    ]
    gaps, summary = module.reconstruct_gaps(events)
    assert len(gaps) == 1
    assert gaps[0].total_ns == 10
    assert gaps[0].application_starvation_ns == 5
    assert gaps[0].driver_or_dependency_ns == 5
    assert summary == {
        "gpu_busy_ns": 20,
        "gpu_idle_ns": 10,
        "gpu_span_ns": 30,
        "merged_interval_count": 2,
        "reconstructed_gap_ns": 10,
    }


def test_only_dynamic_periodic_starvation_is_a_token_boundary():
    module = _module()
    token = module.GpuGap(
        0,
        600_000,
        1_000_000,
        370,
        372,
        "",
        "",
        "application_submission",
    )
    within = module.GpuGap(
        0,
        0,
        300_000,
        866,
        870,
        "",
        "",
        "application_submission",
    )
    boundary, classified = module._classify_gap(token)
    assert boundary == "readback_to_next_token"
    assert classified.previous_stage == "argmax/eval/readback"
    assert module._classify_gap(within)[0] == "within_token_unclassified"


def test_committed_attribution_reconstructs_all_idle_and_localizes_delta():
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"]
    assert artifact["stage_assignment_contract"]["static_kernel_labels_used"] is False
    assert artifact["acceptance"]["compiled_idle_delta_attributed_ge_80pct"]
    assert artifact["compiled_vs_eager"]["readback_boundary_attribution_fraction"] > 1.0
    assert artifact["compiled_vs_eager"]["readback_boundary_reduction_ms_per_token"] > 1.3
    expected_trace_hashes = {
        "A": "28561e60d39c6f0a8d5c9a6368c79b3bfcbace86a972c70789c87fd1afee7526",
        "B": "31e91f0367289e00b45738fb5a1f0effa6ebb85496b15b7140f44835666a1810",
    }
    for arm, expected_hash in expected_trace_hashes.items():
        row = artifact["arms"][arm]
        assert row["trace"]["sha256"] == expected_hash
        assert row["idle_reconstruction_relative_error"] == 0.0
        assert row["within_token_unclassified_idle_fraction"] <= 0.10
        assert row["dynamic_boundaries"]["readback_to_next_token"]["count"] == 15
