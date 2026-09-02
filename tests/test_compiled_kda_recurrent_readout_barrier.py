import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/localize_compiled_kda_recurrent_readout_barrier.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-compiled-kda-recurrent-readout-barrier-20260902.json"
)


def _source() -> str:
    return SCRIPT.read_text()


def test_probe_has_five_causal_arms_and_fixed_boundaries():
    source = _source()
    for arm in ('"A"', '"B"', '"C"', '"D"', '"E"'):
        assert arm in source
    for stage in (
        "conv_output",
        "conv_state",
        "raw_q_bf16",
        "q_square_fp32",
        "q_sum_fp32",
        "q_inverse_norm_fp32",
        "q_scaled_fp32",
        "decay_g",
        "beta",
        "recurrent_output_bf16",
        "updated_recurrent_state_fp32",
        "gated_norm_input_bf16",
    ):
        assert stage in source
    assert "FAILED_LAYERS = (10, 22, 25, 42)" in source
    assert "CONTROL_LAYERS = (0, 20, 44)" in source


def test_probe_retains_exact_sigmoid_and_does_not_change_runtime_abi():
    source = _source()
    assert "SigmoidBarrierNorm" in source
    assert "SIGMOID_MODE = 7" in source
    assert '"runtime_changes": False' in source
    assert '"kernel_abi_changes": False' in source
    assert '"executed": False' in source


def test_artifact_is_decisive_and_preserves_state_exactness():
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["probe_only"] is True
    assert artifact["runtime_changes"] is False
    assert artifact["kernel_abi_changes"] is False
    acceptance = artifact["acceptance"]
    assert acceptance["control_layers_all_arms_exact"] is True
    assert acceptance["failed_layers_full_compile_reproduces_divergence"] is True
    assert acceptance["failed_layers_updated_state_exact_every_step"] is True
    assert acceptance["compiled_prefix_materialization_restores_exactness"] is False
    assert acceptance["compiled_recurrence_with_eager_inputs_exact"] is True
    assert acceptance["compiled_tail_with_exact_sigmoid_exact"] is True
    assert acceptance["q_scale_is_first_difference_in_all_failed_layers"] is True
    assert acceptance["q_scale_difference_crosses_bf16_boundary"] is True
    assert acceptance["final_state_exact_across_all_arms"] is True
    assert acceptance["all_compiled_callables_trace_once"] is True
    assert artifact["decision"] == "probe_exact_q_scale_metal_barrier"
    assert artifact["q_scale_contract"]["head_dim"] == 128
    assert artifact["q_scale_contract"]["fp32_bits_hex"] == "0x3db504f3"
    for layer in (10, 22, 25, 42):
        first = artifact["rows"][str(layer)]["first_divergence"]
        assert first["first_B_stage"] == "q_scaled_fp32"
        assert first["B_full_compiled"]["q_l2normalized_fp32"][
            "byte_identical"
        ] is True
        assert first["B_full_compiled"]["q_scaled_fp32"][
            "byte_identical"
        ] is False
        assert first["B_full_compiled"]["normalized_q"][
            "byte_identical"
        ] is False
    assert artifact["full_model_oracle"]["executed"] is False
