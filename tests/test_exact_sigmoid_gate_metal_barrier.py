import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_exact_sigmoid_gate_metal_barrier.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-exact-sigmoid-gate-metal-barrier-20260902.json"
)


def _source() -> str:
    return SCRIPT.read_text()


def test_minimal_and_fused_barriers_are_separate_probe_arms():
    source = _source()
    assert "class SigmoidBarrierNorm" in source
    assert "class FusedBarrierNorm" in source
    assert '"B_sigmoid_only"' in source
    assert '"C_fused_gated_rmsnorm"' in source
    assert '"runtime_changes": False' in source
    assert '"kernel_abi_changes": False' in source


def test_metal_sigmoid_avoids_fast_namespace_and_records_ordering():
    source = _source()
    assert "metal::exp" in source
    assert "fast::exp" not in source
    assert "naive_exp_divide" in source
    assert "stable_tail_subtract" in source
    assert "stable_common_branch" in source
    assert "tanh_add_then_half" in source
    assert "tanh_half_then_add" in source
    assert "exp_min_over_abs_denominator" in source
    assert "mlx_v0322_literal" in source
    assert "mlx_v0322_precise_exp" in source
    assert "auto y = 1 / (1 + metal::exp(metal::abs(value)))" in source
    assert "metal::precise::exp" in source
    assert "metal::precise::rsqrt" in source
    assert "1.0f / (1.0f + metal::exp(-value))" in source


def test_boundary_and_state_gates_are_explicit():
    source = _source()
    assert "BIT_CRITICAL_STEPS = (6, 31)" in source
    assert "OFFSET_FIXTURES = (0, 1, 255, 256, 2048)" in source
    assert "NaN/Inf gate input is rejected before Metal dispatch" in source
    assert "strided_contiguous_byte_identical" in source
    assert "rejected_before_execution" in source
    assert "HOST_BUILD_REDUCTION_GATE = 0.40" in source
    assert "WORKING_PEAK_LIMIT = 64 * 2**20" in source


def test_committed_artifact_has_decisive_layer0_and_all_kda_gates():
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["runtime_changes"] is False
    assert artifact["kernel_abi_changes"] is False
    assert artifact["sigmoid_formula"]["selected_mode"] == 7
    assert artifact["layer0_retained_candidate"] == "B_sigmoid_only"
    assert artifact["retained_candidate"] is None
    for candidate_name in ("B_sigmoid_only", "C_fused_gated_rmsnorm"):
        candidate = artifact["candidates"][candidate_name]
        assert candidate["all_outputs_byte_identical"] is True
        assert candidate["conv_state_byte_identical"] is True
        assert candidate["recurrent_state_byte_identical"] is True
        assert candidate["critical_steps"]["6"]["byte_identical"] is True
        assert candidate["critical_steps"]["31"]["byte_identical"] is True
        assert candidate["compile_trace_calls"] == 1
        all_kda = artifact["all_kda_candidates"][candidate_name]
        assert all_kda["all_34_layers_byte_identical"] is False
        failure = artifact["failure_localization"][candidate_name]
        assert failure["failed_layers"] == [10, 22, 25, 42]
        assert failure["final_conv_and_recurrent_state_exact"] is True
        assert failure["first_difference_precedes_gated_norm"] is True
        assert failure["compiled_projection_from_eager_norm_exact"] is True
    assert artifact["acceptance"]["all_34_kda_layers_exact"] is False
    assert artifact["acceptance"][
        "representative_nonzero_snapshot_replay_exact"
    ] is True
    assert artifact["acceptance"][
        "official_oracle_skipped_after_all_kda_failure"
    ] is True
    assert artifact["decision"] == (
        "stop_exact_sigmoid_barrier_at_all_kda_recurrence_output"
    )
