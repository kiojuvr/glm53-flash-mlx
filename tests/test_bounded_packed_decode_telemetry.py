import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
NEGATIVE = (
    ROOT
    / "bench-results"
    / "m3ultra512-full-model-gputrace-negative-evidence-20260901.json"
)


def _load(name: str):
    path = SCRIPTS / f"{name}.py"
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_full_model_capture_budget_contract_and_negative_evidence():
    budget = _load("capture_budget")
    value = budget.CaptureBudget()
    assert value.max_elapsed_s == 900.0
    assert value.max_trace_bytes == 32 << 30
    assert value.min_free_bytes == 64 << 30
    artifact = json.loads(NEGATIVE.read_text())
    assert artifact["capture_complete"] is False
    assert artifact["correctness_claim"] is False
    assert artifact["observed_trace_bytes"] == 130 << 30
    assert artifact["partial_trace_deleted"] is True


def test_budget_supervisor_deletes_partial_trace(tmp_path):
    budget = _load("capture_budget")
    trace = tmp_path / "partial.gputrace"
    evidence = tmp_path / "negative.json"
    code = (
        "from pathlib import Path; import time; "
        f"p=Path({str(trace)!r}); p.mkdir(); "
        "(p/'buffer').write_bytes(b'x'*(2<<20)); time.sleep(30)"
    )
    result = budget.supervise_capture(
        [sys.executable, "-c", code],
        trace_path=trace,
        evidence_path=evidence,
        budget=budget.CaptureBudget(
            max_elapsed_s=10.0,
            max_trace_bytes=1 << 20,
            min_free_bytes=0,
        ),
        metadata={"fixture": True},
        poll_s=0.01,
    )
    assert result == 2
    assert not trace.exists()
    value = json.loads(evidence.read_text())
    assert value["status"] == "aborted_resource_budget"
    assert value["violation"] == "max_trace_bytes"
    assert value["partial_trace_deleted"] is True


def test_non_replayable_telemetry_geometry_and_parser(tmp_path):
    telemetry = _load("characterize_packed_decode_bounded_telemetry")
    assert telemetry.ARMS == {"A": False, "B": True}
    assert telemetry.INITIAL_CONTEXT == 2049
    assert telemetry.WARMUPS == 2
    assert telemetry.TELEMETRY_TOKENS == 272
    assert telemetry.TRACED_TOKENS == 16
    assert telemetry.MATERIALIZATION_STEP == 256
    assert telemetry.TRACE_TIME_LIMIT_SECONDS == 8
    assert telemetry._round_up(2049 + 2 + 272 + 1) == 2560
    xml = tmp_path / "gpu.xml"
    xml.write_text(
        """<?xml version='1.0'?>
<trace-query-result><node><schema name='metal-gpu-intervals'>
<col><mnemonic>start</mnemonic></col><col><mnemonic>duration</mnemonic></col>
<col><mnemonic>process</mnemonic></col></schema>
<row><start-time id='1'>0</start-time><duration id='2'>10</duration>
<process id='3' fmt='python (42)'>python</process></row>
<row><start-time id='4'>15</start-time><duration ref='2'/><process ref='3'/></row>
<row><start-time id='5'>30</start-time><duration ref='2'/>
<process id='6' fmt='other (7)'>other</process></row>
</node></trace-query-result>"""
    )
    rows = telemetry._parse_export(xml, 42)
    assert len(rows) == 2
    stats = telemetry._merged_interval_stats(rows)
    assert stats["gpu_interval_count"] == 2
    assert stats["gpu_busy_ms"] == 20 / 1e6
    assert stats["gpu_idle_gap_ms"] == 5 / 1e6


def test_operator_microcapture_is_layer_scoped_and_bounded():
    operator = _load("capture_packed_decode_operator")
    assert operator.LAYERS == (3, 24, 44)
    assert operator.STAGES == (
        "router",
        "routed",
        "shared",
        "ffn-add",
        "full-ffn",
    )
    source = (SCRIPTS / "capture_packed_decode_operator.py").read_text()
    assert "load_model(args.model, strict=True)" in source
    assert "warm_residency" not in source
    assert '"full_model_payload_resident": False' in source
    assert "CaptureBudget()" in source
    runner = _load("run_packed_decode_operator_microcaptures")
    assert runner.LAYERS == operator.LAYERS
    assert runner.STAGES == operator.STAGES


def test_full_model_capture_uses_external_budget_supervisor():
    source = (
        SCRIPTS / "capture_steady_packed_decode_critical_path.py"
    ).read_text()
    assert "CaptureBudget(" in source
    assert "supervise_capture(" in source
    assert '"--capture-child"' in source
    assert '"capture_kind": "full-model-replayable-gputrace"' in source


def test_operator_static_trace_label_summary(tmp_path):
    summary = _load("summarize_packed_decode_operator_traces")
    trace = tmp_path / "one.gputrace"
    trace.mkdir()
    (trace / "capture").write_bytes(
        b"custom_kernel_glm53_one\0custom_kernel_glm53_two\0"
        b"custom_kernel_glm53_one"
    )
    labels, files, size = summary._labels(trace)
    assert labels == ["custom_kernel_glm53_one", "custom_kernel_glm53_two"]
    assert files == 1
    assert size > 0


def test_bounded_telemetry_artifact_is_complete():
    path = (
        ROOT
        / "bench-results"
        / "m3ultra512-packed-decode-bounded-telemetry-20260901.json"
    )
    artifact = json.loads(path.read_text())
    assert artifact["complete"] is True
    assert all(artifact["correctness"].values())
    assert artifact["compiled_vs_eager"]["gpu_interval_delta"] == 0
    assert artifact["compiled_vs_eager"]["gpu_idle_reduction_ms_per_token"] > 1.0
    assert all(row["trace"]["bytes"] < 4 << 30 for row in artifact["arms"].values())


def test_operator_microcapture_artifact_has_full_coverage():
    path = (
        ROOT
        / "bench-results"
        / "m3ultra512-packed-decode-operator-microcapture-20260901.json"
    )
    artifact = json.loads(path.read_text())
    assert artifact["coverage_complete"] is True
    assert len(artifact["cases"]) == 15
    assert artifact["static_trace_analysis"]["labels_prove_dispatch"] is False
    assert artifact["static_trace_analysis"]["all_full_outputs_equal_ffn_add"] is True
    assert all(not row["full_model_payload_resident"] for row in artifact["cases"].values())
    assert all(row["trace"]["bytes"] < 32 << 30 for row in artifact["cases"].values())
