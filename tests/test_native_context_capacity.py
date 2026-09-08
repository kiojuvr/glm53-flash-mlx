from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from glm53_flash_mlx.cache_geometry import plan_nope_cache_capacity
from glm53_flash_mlx.context_capacity import (
    DESIGN_CODING_AGENT_PROMPT_TOKENS,
    DESIGN_MAX_GENERATION_TOKENS,
    DESIGN_PREFILL_PROFILE_CONTEXTS,
    DESIGN_TOTAL_CONTEXT_TOKENS,
    MODEL_NATIVE_CONTEXT_TOKENS,
    NativeContextCapacityError,
    native_prefill_profile_geometries,
    plan_native_context_capacity,
)
from glm53_flash_mlx.server import (
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_GENERATION_TOKENS,
    validate_admission,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "define_native_context_capacity_contract.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-context-capacity-contract-20260908.json"
)


def _script_module():
    spec = importlib.util.spec_from_file_location("native_capacity_contract", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_512k_contract_reserves_320k_prompt_128k_generation_and_64k_headroom():
    plan = plan_native_context_capacity()
    assert DESIGN_TOTAL_CONTEXT_TOKENS == 524_288
    assert DESIGN_CODING_AGENT_PROMPT_TOKENS == 327_680
    assert DESIGN_MAX_GENERATION_TOKENS == 131_072
    assert plan.target_headroom_tokens == 65_536
    assert plan.max_prompt_at_max_generation == 393_216
    validate_admission(
        DESIGN_CODING_AGENT_PROMPT_TOKENS,
        DESIGN_MAX_GENERATION_TOKENS,
        max_generation_tokens=DESIGN_MAX_GENERATION_TOKENS,
        max_context_tokens=DESIGN_TOTAL_CONTEXT_TOKENS,
    )


def test_contract_does_not_silently_promote_current_production_defaults():
    plan = plan_native_context_capacity()
    assert DEFAULT_MAX_CONTEXT_TOKENS == 36_864
    assert DEFAULT_MAX_GENERATION_TOKENS == 4_096
    assert plan.production_default_changed is False


@pytest.mark.parametrize(
    ("logical", "physical", "logical_rows", "physical_rows"),
    [
        (524_287, 524_288, 131_072, 131_072),
        (524_288, 524_288, 131_072, 131_072),
        (524_289, 524_544, 131_073, 131_136),
    ],
)
def test_512k_cache_capacity_minus_exact_plus_boundaries(
    logical, physical, logical_rows, physical_rows
):
    plan = plan_nope_cache_capacity(logical)
    assert plan.physical_capacity_tokens == physical
    assert plan.logical_pool_rows == logical_rows
    assert plan.physical_pool_rows == physical_rows


def test_512k_dsa_workspace_stays_bounded_by_reducing_query_rows():
    plan = plan_native_context_capacity()
    assert plan.logical_pool_rows == 131_072
    assert plan.prefill_query_rows == 256
    assert plan.prefill_query_block_rows == 128
    assert plan.prefill_query_block_count == 2
    assert plan.fp32_logits_workspace_bytes == 64 << 20
    assert plan.selected_token_width == 2_051


def test_prefill_profile_geometries_cover_32k_128k_and_320k():
    rows = native_prefill_profile_geometries()
    assert DESIGN_PREFILL_PROFILE_CONTEXTS == (32_768, 131_072, 327_680)
    assert tuple(row["context_tokens"] for row in rows) == (
        32_768,
        131_072,
        327_680,
    )
    assert rows[0] == {
        "context_tokens": 32_768,
        "pool_count": 8_192,
        "query_block_rows": 256,
        "query_block_count": 1,
        "fp32_logits_workspace_bytes": 8 << 20,
    }
    assert rows[1]["fp32_logits_workspace_bytes"] == 32 << 20
    assert rows[2]["query_block_rows"] == 204
    assert rows[2]["query_block_count"] == 2
    assert rows[2]["fp32_logits_workspace_bytes"] <= 64 << 20


def test_128k_generation_has_exactly_512_materialization_boundaries():
    plan = plan_native_context_capacity()
    assert plan.materialization_interval_tokens == 256
    assert plan.generation_materialization_count == 512


def test_persistent_dsa_capacity_is_explicitly_accounted():
    plan = plan_native_context_capacity()
    assert plan.persistent_nope_latent_bytes_all_dsa_layers == 5_905_580_032
    assert plan.persistent_indexpool_bytes_all_dsa_layers == 416_677_888


def test_current_native_decode_range_is_not_misrepresented_as_512k_ready():
    plan = plan_native_context_capacity()
    assert plan.currently_qualified_native_pool_rows == 65_600
    assert plan.physical_pool_rows == 131_072
    assert plan.native_pool_row_deficit == 65_472
    assert plan.native_decode_geometry_qualified is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total_context_tokens": 0},
        {"coding_agent_prompt_tokens": True},
        {"max_generation_tokens": 0},
        {"prefill_query_rows": 0},
        {"max_workspace_bytes": 0},
        {"total_context_tokens": MODEL_NATIVE_CONTEXT_TOKENS + 1},
        {
            "total_context_tokens": 100,
            "coding_agent_prompt_tokens": 60,
            "max_generation_tokens": 41,
        },
    ],
)
def test_invalid_capacity_contract_fails_closed(kwargs):
    with pytest.raises(NativeContextCapacityError):
        plan_native_context_capacity(**kwargs)


def test_contract_artifact_is_deterministic_and_authoritative():
    expected = _script_module().build_artifact()
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact == expected
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert all(artifact["checks"].values())
    assert not any(artifact["runtime_changes"].values())
    assert artifact["decision"] == (
        "advance_to_native_prefill_profiling_and_expand_decode_geometry"
    )
