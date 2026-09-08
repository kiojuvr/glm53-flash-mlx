import json
from pathlib import Path

import pytest

from glm53_flash_mlx.native_prefill_plan import (
    NativePrefillPlanError,
    build_native_sparse_prefill_attention_plan,
)


ROOT = Path(__file__).resolve().parents[1]
REORDERING = (
    ROOT
    / "bench-results"
    / "m3ultra512-sparse-prefill-reordering-equivalence-20260908.json"
)
MICROTILE = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-dsa-prefill-streaming-microtile-20260908.json"
)
LOCALIZATION = (
    ROOT
    / "bench-results"
    / "m3ultra512-sparse-prefill-attention-reduction-localization-20260908.json"
)
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-exact-sparse-prefill-attention-plan-20260908.json"
)


def _evidence():
    return (
        json.loads(REORDERING.read_text()),
        json.loads(MICROTILE.read_text()),
        json.loads(LOCALIZATION.read_text()),
    )


def test_plan_uses_q4_selection_q1_attention_and_bounded_scratch():
    plan = build_native_sparse_prefill_attention_plan(*_evidence())
    assert plan.logical_context_tokens == 512 << 10
    assert plan.score_microtile_rows == 4
    assert plan.attention_microtile_rows == 1
    assert plan.selected_width == 2051
    assert plan.total_scratch_bytes < 256 << 20
    assert plan.full_projected_key_value_bytes_avoided == 40 << 30
    assert plan.full_sparse_mask_bytes_avoided == 128 << 20


def test_plan_keeps_measured_compact_attention_barrier_explicit():
    plan = build_native_sparse_prefill_attention_plan(*_evidence())
    assert plan.selected_projection_reordering_exact is True
    assert plan.compact_qk_exact is True
    assert plan.compact_precise_softmax_exact is True
    assert plan.ordinary_compact_sdpa_allowed is False
    assert plan.compact_av_allowed is False
    assert plan.requires_virtual_full_kv_av_reduction_topology is True
    assert any("split-K topology" in item for item in plan.invariants)


def test_plan_rejects_evidence_that_hides_measured_barrier():
    reordering, microtile, localization = _evidence()
    reordering["checks"]["sorted_compact_attention_matches_dense_sparse_mask"] = True
    with pytest.raises(NativePrefillPlanError):
        build_native_sparse_prefill_attention_plan(
            reordering, microtile, localization
        )


def test_plan_artifact_is_complete_and_accepted_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert all(artifact["checks"].values())
