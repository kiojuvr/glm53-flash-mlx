import ast
import json
from pathlib import Path

import pytest

from glm53_flash_mlx.native_prefill_plan import (
    NativePrefillPlanError,
    build_native_q256_shared_value_pass_plan,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "define_q256_shared_physical_value_pass.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-q256-shared-physical-value-pass-plan-20260909.json"
)


def _load(name: str) -> dict[str, object]:
    return json.loads((ROOT / "bench-results" / name).read_text())


def _sources():
    return (
        _load("m3ultra512-q4-union-tiled-full-attention-20260909.json"),
        _load("m3ultra512-q256-union-tiled-full-attention-20260909.json"),
        _load("m3ultra512-q256-projected-qk-320k-tiles-20260909.json"),
        _load("m3ultra512-native-prefill-execution-plan-20260908.json"),
    )


def test_plan_replaces_query_local_v_with_shared_bm64_physical_tiles():
    plan = build_native_q256_shared_value_pass_plan(*_sources())
    assert plan.query_block_rows == 64
    assert plan.query_blocks == 4
    assert plan.physical_value_tile_rows == 65_536
    assert plan.maximum_physical_tiles_512k == 8
    assert plan.projected_value_tile_bytes == 1 << 30
    assert plan.fp32_attention_accumulator_bytes == 8 << 20
    assert plan.rejected_query_local_selected_value_bytes >= 8 << 30
    assert plan.maximum_phase_arena_bytes <= plan.available_native_arena_bytes
    assert "query-local selected V materialization is forbidden" in plan.invariants


def test_plan_rejects_erasing_the_measured_q256_failure():
    sources = list(_sources())
    sources[1] = dict(sources[1], accepted=True)
    with pytest.raises(NativePrefillPlanError):
        build_native_q256_shared_value_pass_plan(*sources)


def test_definition_script_and_artifact_contract():
    ast.parse(SCRIPT.read_text())
    assert "implement_exact_bm64_shared_physical_value_tile" in SCRIPT.read_text()
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert all(artifact["checks"].values())
