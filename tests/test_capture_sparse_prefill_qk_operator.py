import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "capture_sparse_prefill_qk_operator.py"


def test_capture_script_is_model_free_and_bounded():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert "load_model" not in source
    assert 'CONTEXT = 2_048' in source
    assert 'max_trace_bytes=2 << 30' in source
    assert 'full_model_payload_resident": False' in source
    assert "_validate_trace(args.trace)" in source


def test_capture_records_dynamic_pipeline_and_exact_output():
    source = SCRIPT.read_text()
    assert "captured_pipeline_resource_labels" in source
    assert "gemv_bfloat16_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0" in source
    assert "output_byte_exact" in source
    assert "model-free compact QK operands only" in source
    assert "instantiate_captured_compact_qk_topology" in source


def test_capture_artifact_records_the_expected_qk_gemv_when_present():
    artifact = (
        ROOT
        / "bench-results"
        / "m3ultra512-sparse-prefill-qk-operator-capture-20260908.json"
    )
    if not artifact.exists():
        return
    import json

    value = json.loads(artifact.read_text())
    assert value["complete"] is True
    assert value["accepted"] is True
    assert (
        "gemv_bfloat16_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0"
        in value["case"]["captured_pipeline_resource_labels"]
    )
