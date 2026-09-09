import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "native_execution" / "native_shared_physical_value_tile_plan.h"
SOURCE = ROOT / "native_execution" / "native_shared_physical_value_tile_plan.cpp"
METAL = ROOT / "native_execution" / "native_indexer_plan.metal"
BINDINGS = ROOT / "native_execution" / "bindings.cpp"
SCRIPT = ROOT / "scripts" / "probe_actual_q256_shared_physical_value_pass.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-actual-q256-shared-physical-value-pass-20260909.json"
)


def test_shared_value_plan_exposes_actual_v256_as_explicit_abi():
    header = HEADER.read_text()
    source = SOURCE.read_text()
    metal = METAL.read_text()
    bindings = BINDINGS.read_text()
    assert "int value_dim = 128" in header
    assert "value_dim_" in header
    assert "value_dim != 128 && value_dim != 256" in source
    assert "encoder.set_bytes(value_dim_, 9)" in source
    assert "constant const int& value_dim [[buffer(9)]]" in metal
    assert '"value_dim"_a = 128' in bindings


def test_actual_v256_probe_is_exact_and_is_a_45_layer_coverage_gate():
    ast.parse(SCRIPT.read_text())
    text = SCRIPT.read_text()
    assert "actual_v256_matches_two_exact_v128_halves_byte_exact" in text
    assert "one_shared_probability_scatter_for_all_four_bn64_value_blocks" in text
    assert "advance_actual_dsa_attention_into_45_layer_native_prefill_plan" in text
    if ARTIFACT.exists():
        artifact = json.loads(ARTIFACT.read_text())
        assert artifact["complete"] is True
        assert artifact["accepted"] is True
        assert all(artifact["checks"].values())
