import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "native_execution" / "native_prefill_45_layer_submission_plan.h"
SOURCE = ROOT / "native_execution" / "native_prefill_45_layer_submission_plan.cpp"
BINDINGS = ROOT / "native_execution" / "bindings.cpp"
PACKAGE = ROOT / "native_execution" / "glm53_native_execution" / "__init__.py"
SCRIPT = ROOT / "scripts" / "probe_native_prefill_45_layer_submission_topology.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-native-prefill-45-layer-submission-topology-20260909.json"
)


def test_native_model_plan_has_one_call_and_exact_layer_map():
    header = HEADER.read_text()
    source = SOURCE.read_text()
    assert "kLayerCount = 45" in header
    assert "native_calls_per_all_execute() const { return 1; }" in header
    assert "native_calls_per_layerwise_execute() const { return kLayerCount; }" in header
    assert 'layer % 4 == 3 ? "dsa" : "kda"' in source
    assert 'layer < 3 ? "dense" : "moe"' in source


def test_execute_all_encodes_all_layers_without_graph_or_sync():
    source = SOURCE.read_text()
    execute = source[source.index("NativePrefill45LayerSubmissionPlan::execute_all(") :]
    assert "for (int layer = 0; layer < kLayerCount; ++layer)" in execute
    assert "mx::eval" not in execute
    assert "synchronize" not in execute
    assert "allocator::malloc" not in execute


def test_binding_and_probe_keep_arithmetic_coverage_fail_closed():
    assert "NativePrefill45LayerSubmissionPlan" in BINDINGS.read_text()
    assert "NativePrefill45LayerSubmissionPlan" in PACKAGE.read_text()
    ast.parse(SCRIPT.read_text())
    text = SCRIPT.read_text()
    assert "python_native_calls_reduce_from_45_to_1" in text
    assert '"pass_through_only": True' in text
    assert "replace_copy_slots_with_exact_kda_dsa_dense_moe_layer_encoders" in text
    if ARTIFACT.exists():
        artifact = json.loads(ARTIFACT.read_text())
        assert artifact["complete"] is True
        assert artifact["arithmetic_coverage"]["pass_through_only"] is True
        if artifact["accepted"]:
            assert all(artifact["checks"].values())
        else:
            failed = [
                name for name, passed in artifact["checks"].items() if not passed
            ]
            assert failed == ["host_submission_speedup_at_least_1_20x"]
