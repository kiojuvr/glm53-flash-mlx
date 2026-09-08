import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "scripts" / "capture_sparse_prefill_av_operator.py"
ANALYZER_PATH = (
    ROOT / "scripts" / "analyze_sparse_prefill_av_operator_captures.py"
)
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-sparse-prefill-av-operator-capture-20260908.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("capture_sparse_av", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_capture_is_model_free_bounded_and_keeps_arms_separate():
    source = PROBE_PATH.read_text()
    assert 'ARMS = ("direct", "compact")' in source
    assert 'max_trace_bytes=4 << 30' in source
    assert 'max_elapsed_s=300.0' in source
    assert '"full_model_payload_resident": False' in source
    assert '"model-free 32K AV operands only"' in source


def test_capture_rejects_repository_and_existing_trace_paths(tmp_path):
    probe = _module()
    with pytest.raises(ValueError):
        probe._validate_trace(ROOT / "bad.gputrace")
    trace = tmp_path / "one.gputrace"
    trace.write_bytes(b"exists")
    with pytest.raises(FileExistsError):
        probe._validate_trace(trace)


def test_capture_records_dynamic_pipeline_labels_as_best_effort_only():
    source = PROBE_PATH.read_text()
    assert '"pipeline_labels_best_effort"' in source
    assert '"pipeline_labels_authoritative": False' in source
    assert '"capture_dynamic_av_gemm_geometry"' in source


def test_analyzer_requires_shared_bk16_pipeline_and_preserves_xcode_caveat():
    source = ANALYZER_PATH.read_text()
    assert "bm64_bn64_bk16_wm1_wn2" in source
    assert '"direct_and_compact_select_same_pipeline"' in source
    assert '"only_reduction_k_geometry_differs"' in source
    assert '"dispatch_association_requires_xcode_view": True' in source
    assert '"implement_virtual_physical_bk16_av_reduction"' in source


def test_capture_artifact_is_complete_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert set(artifact["cases"]) == {"direct", "compact"}
    for row in artifact["cases"].values():
        assert row["full_model_payload_resident"] is False
        assert row["trace"]["stored_in_repository"] is False
        assert row["expected_exactness_observed"] is True
    if "dynamic_topology" in artifact:
        assert all(artifact["dynamic_topology"]["checks"].values())
        assert artifact["dynamic_topology"]["shared_bk"] == 16
