from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from glm53_flash_mlx.native_prefill_plan import (
    NativePrefillPlanError,
    plan_native_dsa_prefill_streaming_geometry,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_native_dsa_prefill_streaming_microtile.py"
NATIVE = ROOT / "native_execution"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-dsa-prefill-streaming-microtile-20260908.json"
)


def test_q4_makes_m128_and_bounds_512k_exact_score_scratch():
    plan = plan_native_dsa_prefill_streaming_geometry()
    assert plan.query_rows_per_microtile == 4
    assert plan.indexer_heads == 32
    assert plan.matrix_rows == 128
    assert plan.steel_matrix_rows_aligned is True
    assert plan.microtiles_per_256_row_chunk == 64
    assert plan.bf16_head_score_scratch_bytes == 32 << 20
    assert plan.bf16_index_score_scratch_bytes == 1 << 20
    assert plan.total_score_selection_scratch_bytes < 34 << 20
    assert plan.full_q256_head_score_bytes_avoided == 2 << 30


@pytest.mark.parametrize("rows", [511, 513, 65_601, 131_136, True])
def test_streaming_geometry_rejects_incompatible_pool_capacity(rows):
    with pytest.raises(NativePrefillPlanError):
        plan_native_dsa_prefill_streaming_geometry(physical_pool_rows=rows)


def test_native_score_plan_explicitly_admits_512k_pool_geometry():
    cpp = (NATIVE / "native_dsa_score_plan.cpp").read_text()
    assert "physical_pool_rows > 131072" in cpp
    assert "64-aligned in [512, 131072]" in cpp


def test_probe_preserves_bf16_head_score_boundary_and_forbids_promotion():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert "CONTEXTS = PROFILE_CONTEXTS + (512 << 10,)" in source
    assert '"bf16_head_scores"' in source
    assert '"bf16_index_scores"' in source
    assert '"partial_score_selection": False' in source
    assert '"requires_composed_gather_attention": True' in source
    assert '"timing_is_production_prefill_claim": False' in source


def test_artifact_when_present_is_exact_and_bounded():
    if not ARTIFACT.exists():
        pytest.skip("native DSA streaming microtile probe is pending")
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert all(artifact["checks"].values())
    assert artifact["promotion"]["runtime"] is False
    assert artifact["promotion"]["partial_score_selection"] is False
