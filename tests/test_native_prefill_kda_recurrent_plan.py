import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "native_execution" / "native_prefill_kda_recurrent_plan.h"
SOURCE = ROOT / "native_execution" / "native_prefill_kda_recurrent_plan.cpp"
METAL = ROOT / "native_execution" / "native_indexer_plan.metal"
SCRIPT = ROOT / "scripts" / "probe_native_prefill_kda_recurrent_plan.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-native-prefill-kda-recurrent-plan-20260909.json"
)


def test_native_kda_plan_has_fixed_r4_q256_geometry_and_owned_state():
    header = HEADER.read_text()
    source = SOURCE.read_text()
    metal = METAL.read_text()
    assert "kQueryRows = 256" in header
    assert "kHeads = 64" in header
    assert "kKeyDim = 128" in header
    assert "kValueDim = 128" in header
    assert "kRowBlock = 4" in header
    assert "output_" in header and "next_state_" in header
    assert "glm53_native_prefill_kda_recurrent_r4_bfloat16" in source
    assert "local_state[ROW_BLOCK][N_PER_THREAD]" in metal


def test_native_kda_execute_has_no_graph_allocation_or_sync():
    source = SOURCE.read_text()
    execute = source[source.index("NativePrefillKDARecurrentPlan::execute(") :]
    assert "allocator::malloc" not in execute
    assert "mx::eval" not in execute
    assert "synchronize" not in execute
    assert "dispatch_threadgroups" in execute


def test_probe_preserves_exactness_and_does_not_overclaim_layer_coverage():
    ast.parse(SCRIPT.read_text())
    text = SCRIPT.read_text()
    assert "zero_nonzero_and_mask_fixtures_byte_exact" in text
    assert "operator_speedup_at_least_1_15x" in text
    assert '"qkv_projection": False' in text
    assert '"exact_recurrence": True' in text
    if ARTIFACT.exists():
        artifact = json.loads(ARTIFACT.read_text())
        assert artifact["complete"] is True
        assert artifact["accepted"] is True
        assert all(artifact["checks"].values())
