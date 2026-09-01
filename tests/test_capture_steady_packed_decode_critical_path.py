import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "capture_steady_packed_decode_critical_path.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-steady-packed-decode-critical-path-20260901.json"
)


def _load_probe():
    spec = importlib.util.spec_from_file_location("steady_capture_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_geometry_and_two_arm_contract_are_explicit():
    probe = _load_probe()
    assert probe.ARMS == {
        "A": {"compile_ffn": False, "router_weight_dtype": "bfloat16"},
        "B": {"compile_ffn": True, "router_weight_dtype": "bfloat16"},
    }
    assert probe.INITIAL_CONTEXT_TOKENS == 2049
    assert probe.WARMUP_TOKENS == 2
    assert probe.CAPTURE_TOKENS == 8
    assert probe.CACHE_BACKEND == "direct"


def test_trace_paths_are_fail_closed_and_outside_repository(tmp_path):
    probe = _load_probe()
    trace = tmp_path / "capture.gputrace"
    assert probe._validate_trace_path(trace) == trace.resolve()
    trace.mkdir()
    with pytest.raises(FileExistsError):
        probe._validate_trace_path(trace)
    with pytest.raises(ValueError):
        probe._validate_trace_path(tmp_path / "capture.bin")
    with pytest.raises(ValueError):
        probe._validate_trace_path(ROOT / "capture.gputrace")


def test_bundle_identity_covers_relative_names_and_contents(tmp_path):
    probe = _load_probe()
    trace = tmp_path / "trace.gputrace"
    (trace / "nested").mkdir(parents=True)
    (trace / "capture").write_bytes(b"capture")
    (trace / "nested" / "metadata").write_bytes(b"metadata")
    first = probe._trace_identity(trace)
    second = probe._trace_identity(trace)
    assert first == second
    assert first["kind"] == "bundle-directory"
    assert first["file_count"] == 2
    assert first["bytes"] == len(b"capturemetadata")
    (trace / "nested" / "metadata").write_bytes(b"changed")
    assert probe._trace_identity(trace)["sha256"] != first["sha256"]


def test_capture_interval_excludes_evidence_and_cache_maintenance():
    source = SCRIPT.read_text()
    capture = source.index("mx.metal.start_capture")
    stop = source.index("mx.metal.stop_capture", capture)
    interval = source[capture:stop]
    assert "mx.argmax" in interval
    assert ".item()" in interval
    assert "mx.synchronize()" in interval
    assert "mx.clear_cache" not in interval
    assert "_hash(" not in interval
    assert "_memory(" not in interval
    assert "print(" not in interval
    assert 'residual.Arm("B1", True)' in source
    assert "experimental_packed_decode_moe=True" in source
    assert "capture-process wall latency includes GPUTools resource download" in source
    assert 'artifact.pop("compiled_speedup", None)' in source
    assert '"xcode_gpu_event_measurement_required": True' in source
    assert "capture_tokens_per_second" not in source


def test_gputrace_is_ignored():
    assert "*.gputrace" in (ROOT / ".gitignore").read_text().splitlines()


def test_artifact_is_complete_when_present():
    if not ARTIFACT.exists():
        pytest.skip("M3 Ultra steady packed decode capture has not been generated")
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "glm53-steady-packed-decode-critical-path-v1"
    assert artifact["complete"] is True
    assert artifact["probe_only"] is True
    assert set(artifact["arms"]) == {"A", "B"}
    assert artifact["capture_correctness_passed"] is True
    assert all(artifact["capture_correctness"].values())
    assert all(not row["trace"]["stored_in_repository"] for row in artifact["arms"].values())
    assert artifact["runtime_changes"] == {
        "admission": False,
        "apc": False,
        "cache_abi": False,
        "kernel_abi": False,
        "packed_runtime": False,
        "server": False,
    }
