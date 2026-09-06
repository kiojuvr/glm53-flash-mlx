from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "scripts" / "profile_lightning_indexer_decode_critical_path.py"
TELEMETRY = ROOT / "scripts" / "capture_lightning_indexer_decode_telemetry.py"
RUNNER = ROOT / "scripts" / "run_lightning_indexer_decode_telemetry.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-lightning-indexer-decode-critical-path-20260905.json"
)


def test_profile_is_profiling_only_and_covers_requested_frontier():
    source = PROFILE.read_text()
    ast.parse(source)
    assert "CONTEXTS = (2_048, 32_768, 131_072, 262_144)" in source
    for phase in (
        "index_query_projection",
        "indexpool_update",
        "pooled_key_score",
        "topk_selection",
        "pool_token_expansion",
        "sanitize_gather_preparation",
        "sparse_attention",
    ):
        assert f'"{phase}"' in source
    assert '"profiling_only": True' in source
    assert '"phase_sum_is_token_wall": False' in source
    assert '"kernel": False' in source
    assert '"server": False' in source
    assert '"admission": False' in source


def test_counterfactual_retains_authoritative_update_and_requires_exactness():
    source = PROFILE.read_text()
    assert "CompactIndexPoolCache._decode_selection = replay" in source
    assert "authoritative pool update retained" in source
    assert "capture_replay_selected_indices_byte_exact" in source
    assert "capture_replay_logits_byte_exact" in source
    assert "capture_replay_state_byte_exact" in source
    assert "selection_free_headroom_ms" in source
    assert "stop_lightning_indexer_optimization" in source


def test_bounded_telemetry_uses_dynamic_per_stage_traces():
    source = TELEMETRY.read_text()
    ast.parse(source)
    assert '"Metal System Trace"' in source
    assert "TRACE_STAGES = profiler.PHASES + (\"full_model_decode\",)" in source
    assert '"dynamic_pid_attribution": True' in source
    assert '"static_kernel_labels_used": False' in source
    assert '"full_model_resources_embedded": False' in source
    assert "MAX_TRACE_BYTES = 2 << 30" in source
    assert "Metal System Trace directory must be outside the repository" in source
    assert "_export_telemetry(trace, pid)" in source
    runner = RUNNER.read_text()
    ast.parse(runner)
    assert "CONTEXTS = (2_048, 32_768, 131_072, 262_144)" in runner
    assert '"merge-telemetry"' in runner
    assert '"finalize"' in runner


def test_profile_artifact_is_complete_when_present():
    if not ARTIFACT.exists():
        pytest.skip("M3 Ultra Lightning Indexer profile has not been generated yet")
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "glm53-lightning-indexer-decode-critical-path-v1"
    if "steady_trajectory_profile_and_replay_exact" not in artifact["acceptance"]:
        pytest.skip("bounded telemetry is complete; steady full-model retime is pending")
    if not artifact["complete"]:
        assert artifact["model_phase_complete"] is True
        assert artifact["decision"] == "await_bounded_system_trace"
        assert artifact["acceptance"][
            "bounded_system_trace_gpu_cb_submission_metrics_complete"
        ] is False
        pytest.skip("model phase is complete; bounded telemetry is still pending")
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert all(artifact["acceptance"].values())
    assert set(map(int, artifact["contexts"])) == {2048, 32768, 131072, 262144}
    assert artifact["decision"] == "single_dominant_candidate_selected"
    assert artifact["optimization_decision"]["selected_single_next_candidate"]
    for row in artifact["contexts"].values():
        assert row["full_model"]["evidence"][
            "profile_off_capture_logits_byte_exact"
        ]
        assert row["full_model"]["evidence"][
            "capture_replay_logits_byte_exact"
        ]
        assert row["full_model"]["bounded_system_trace"][
            "command_buffers_per_token"
        ] > 0
