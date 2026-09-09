import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "native_execution" / "native_q256_dsa_prefill_plan.h"
SOURCE = ROOT / "native_execution" / "native_q256_dsa_prefill_plan.cpp"
BINDINGS = ROOT / "native_execution" / "bindings.cpp"
SCRIPT = ROOT / "scripts" / "probe_actual_q256_native_dsa_prefill_island.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-actual-q256-native-dsa-prefill-island-20260909.json"
)


def test_q256_dsa_plan_carries_actual_value_dim_through_one_native_scope():
    header = HEADER.read_text()
    source = SOURCE.read_text()
    bindings = BINDINGS.read_text()
    assert "int value_dim = 128" in header
    assert "value_plan_.value_dim()" in header
    assert "value_plan_(physical_k, tile_rows, 256, value_dim)" in source
    assert '"value_dim"_a = 128' in bindings


def test_actual_q256_dsa_probe_closes_body_before_45_layer_plan():
    ast.parse(SCRIPT.read_text())
    text = SCRIPT.read_text()
    assert "actual_qk_precise_softmax_is_byte_exact" in text
    assert "actual_v256_attention_output_is_byte_exact" in text
    assert "close_dsa_body_and_advance_45_layer_native_prefill_plan" in text
    if ARTIFACT.exists():
        artifact = json.loads(ARTIFACT.read_text())
        assert artifact["complete"] is True
        assert artifact["accepted"] is True
        assert all(artifact["checks"].values())
