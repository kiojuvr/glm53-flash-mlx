import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native_execution" / "native_shared_physical_value_tile_plan.cpp"
HEADER = ROOT / "native_execution" / "native_shared_physical_value_tile_plan.h"
METAL = ROOT / "native_execution" / "native_indexer_plan.metal"
SCRIPT = ROOT / "scripts" / "probe_exact_bm64_shared_physical_value_tile.py"
MULTITILE_SCRIPT = ROOT / "scripts" / (
    "probe_exact_bm64_shared_physical_value_multitile.py"
)
Q256_SCRIPT = ROOT / "scripts" / "probe_exact_q256_shared_physical_value_pass.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-exact-bm64-shared-physical-value-tile-20260909.json"
)
MULTITILE_ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-exact-bm64-shared-physical-value-multitile-20260909.json"
)
Q256_ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-exact-q256-shared-physical-value-pass-20260909.json"
)


def test_native_plan_has_direct_bm64_and_shared_physical_contract():
    source = SOURCE.read_text()
    header = HEADER.read_text()
    metal = METAL.read_text()
    assert "glm53_native_bm64_physical_value_tile_continue_bfloat16" in source
    assert "kBM = 64" in metal
    assert "kWM = 2" in metal
    assert "kWN = 2" in metal
    assert "glm53_native_scatter_bm64_probabilities_to_physical_bfloat16" in metal
    assert "materialized_query_local_selected_value_bytes() const { return 0; }" in header
    assert "{kHeads, kQueryBlockRows, tile_rows_}" in source
    assert "{kHeads, tile_rows_, kValueDim}" in source


def test_probe_contract_and_artifact():
    ast.parse(SCRIPT.read_text())
    text = SCRIPT.read_text()
    assert "physical_probability_scatter" in text
    assert "shared_value_projection" in text
    assert "bm64_physical_attention_output" in text
    assert "query_local_selected_reduction_is_not_the_q256_oracle" in text
    assert "advance_shared_physical_value_pass_to_multitile_accumulator" in text
    if ARTIFACT.exists():
        artifact = json.loads(ARTIFACT.read_text())
        assert artifact["complete"] is True
        assert artifact["accepted"] is True
        assert all(artifact["checks"].values())


def test_multitile_probe_preserves_raw_fp32_mma_fragments():
    ast.parse(MULTITILE_SCRIPT.read_text())
    text = MULTITILE_SCRIPT.read_text()
    metal = METAL.read_text()
    assert "fp32_mma_fragment_store_reload_byte_exact" in text
    assert "two_tile_output_matches_single_continuous_bm64" in text
    assert "mma_op.Ctile.template load<float" in metal
    assert "mma_op.Ctile.template store<float" in metal
    if MULTITILE_ARTIFACT.exists():
        artifact = json.loads(MULTITILE_ARTIFACT.read_text())
        assert artifact["accepted"] is True
        assert all(artifact["checks"].values())


def test_q256_probe_reuses_each_value_tile_across_four_bm64_blocks():
    ast.parse(Q256_SCRIPT.read_text())
    text = Q256_SCRIPT.read_text()
    source = SOURCE.read_text()
    assert "q256_four_bm64_blocks_byte_exact" in text
    assert "one_value_projection_shared_across_four_query_blocks" in text
    assert "advance_exact_q256_shared_value_pass_to_320k_composed_prefill" in text
    assert "for (int query_offset = 0; query_offset < query_rows_;" in source
    if Q256_ARTIFACT.exists():
        artifact = json.loads(Q256_ARTIFACT.read_text())
        assert artifact["accepted"] is True
        assert all(artifact["checks"].values())
