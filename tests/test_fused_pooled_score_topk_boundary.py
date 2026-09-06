import ast
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_fused_pooled_score_topk_boundary.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-fused-pooled-score-topk-boundary-20260906.json"
)


def test_probe_has_three_isolated_arms_and_two_context_screen():
    source = PROBE.read_text()
    ast.parse(source)
    assert "CONTEXTS = (2_048, 262_144)" in source
    assert '"A_eager_score_mlx_topk"' in source
    assert '"B_compiled_score_mlx_topk"' in source
    assert '"C_compiled_score_exact_topk"' in source
    assert "partial_topk.exact_partial_topk(scored)[1]" in source
    assert '"score_expression_arithmetic_changed": False' in source
    assert '"score_tensor_external_from_C": False' in source


def test_score_expression_matches_production_order_and_padding_is_explicit():
    source = PROBE.read_text()
    ordered = (
        "scores = query @ pool_keys[:, None].swapaxes(-1, -2)",
        "scores = mx.maximum(scores * softmax_scale, 0.0)",
        "index_scores = mx.sum(weights[..., None] * scores, axis=2)",
        "return mx.where(valid_candidates, index_scores, -1e30)",
    )
    positions = [source.index(fragment) for fragment in ordered]
    assert positions == sorted(positions)
    assert '"physical_pool_shape_fixed_across_decode": True' in source
    assert '"padding_is_negative_sentinel": True' in source
    assert "B_padding_all_sentinel" in source


def test_exactness_covers_score_selection_expansion_attention_and_state():
    source = PROBE.read_text()
    for field in (
        "B_score_bits_exact",
        "B_indices_exact",
        "C_indices_exact",
        "B_expanded_exact",
        "C_expanded_exact",
        "B_gather_exact",
        "C_gather_exact",
        "B_attention_exact",
        "C_attention_exact",
        "B_state_exact",
        "C_state_exact",
        "B_all_logits_exact",
        "C_all_logits_exact",
        "B_post_state_exact",
        "C_post_state_exact",
    ):
        assert f'"{field}"' in source


def test_wall_measurement_is_interleaved_and_fixed_gates_are_present():
    source = PROBE.read_text()
    assert 'result["interleaved_order"] = True' in source
    assert '"interleaved_arm_order": True' in source
    assert "MIN_FULL_MODEL_SAVING_MS = 0.75" in source
    assert "MAX_2K_REGRESSION_FRACTION = 0.01" in source
    assert "MAX_WORKING_PEAK_DELTA = 32 << 20" in source
    assert '"command_buffer_count_inferred": False' in source
    assert '"bounded_system_trace": "required only if C passes wall screen"' in source
    assert '"bounded_system_trace_executed": False' in source
    assert '"working_peak_measurement_scope"' in source


def test_probe_does_not_change_production_runtime_or_abis():
    source = PROBE.read_text()
    assert '"probe_only": True' in source
    for component in ("runtime", "server", "apc", "cache_abi", "kernel_abi"):
        assert f'"{component}": False' in source


def test_artifact_records_a_consistent_outcome_when_present():
    if not ARTIFACT.exists():
        pytest.skip("M3 Ultra fused score/top-k artifact has not been generated")
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "glm53-fused-pooled-score-topk-boundary-v1"
    assert artifact["complete"] is True
    if artifact["accepted"]:
        assert all(artifact["acceptance"].values())
        assert artifact["decision"] == "await_bounded_system_trace"
    else:
        assert artifact["decision"] in {
            "reject_score_envelope_exactness",
            "reject_fused_score_topk_fixed_gate_not_met",
            "reject_fused_score_topk_boundary_tax",
        }
        assert not all(artifact["acceptance"].values())
        if artifact["decision"] == "reject_fused_score_topk_boundary_tax":
            derived = artifact["derived_interpretation"]
            assert derived["bounded_system_trace_executed"] is False
            assert (
                derived["arms"]["C_compiled_score_exact_topk"]
                ["full_model_saving_ms"]
                < 0
            )
