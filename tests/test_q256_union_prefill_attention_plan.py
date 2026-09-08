import ast
import json
from pathlib import Path

import pytest

from glm53_flash_mlx.native_prefill_plan import (
    NativePrefillPlanError,
    build_native_q256_union_attention_plan,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "define_q256_union_prefill_attention_plan.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-q256-union-prefill-attention-plan-20260908.json"
)


def _load(name: str):
    return json.loads((ROOT / "bench-results" / name).read_text())


def _sources():
    return (
        _load("m3ultra512-selected-kv-projection-reuse-frontier-20260908.json"),
        _load("m3ultra512-device-resident-q256-selected-union-20260908.json"),
        _load("m3ultra512-indirect-q256-selected-latent-20260908.json"),
        _load("m3ultra512-native-prefill-execution-plan-20260908.json"),
    )


def test_plan_fits_two_pass_alias_arena_and_forbids_large_materializations():
    plan = build_native_q256_union_attention_plan(*_sources())
    assert plan.projection_tile_rows == 65_536
    assert plan.maximum_union_tiles_512k == 8
    assert plan.maximum_phase_arena_bytes <= plan.available_native_arena_bytes
    assert plan.full_512k_projected_key_value_bytes_forbidden >= 40 << 30
    assert (
        plan.per_query_selected_key_value_bytes_forbidden
        > plan.available_native_arena_bytes
    )
    assert "key_tiles_project_once_and_fuse_into_query_edge_qk" in (
        plan.execution_phases
    )
    assert "direct_order_virtual_bk16_av" in plan.execution_phases


def test_plan_rejects_unaccepted_source_evidence():
    sources = list(_sources())
    sources[2] = {**sources[2], "accepted": False}
    with pytest.raises(NativePrefillPlanError):
        build_native_q256_union_attention_plan(*sources)


def test_script_and_accepted_artifact_contract():
    ast.parse(SCRIPT.read_text())
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert all(artifact["checks"].values())
    assert artifact["decision"] == "implement_exact_fused_projected_qk_union_tile"
