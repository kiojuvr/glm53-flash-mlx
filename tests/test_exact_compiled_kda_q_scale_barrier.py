import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/probe_exact_compiled_kda_q_scale_barrier.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-exact-compiled-kda-q-scale-barrier-20260902.json"
)


def _source() -> str:
    return SCRIPT.read_text()


def test_probe_has_ordered_q_scale_arms_and_fixed_bit_contract():
    source = _source()
    for arm in (
        "B_compiled_constant",
        "C_runtime_scalar",
        "D_metal_f32",
        "E_metal_f32_bf16",
    ):
        assert arm in source
    assert "SCALE_BITS = 0x3DB504F3" in source
    assert "exact_q_scale_f32" in source
    assert "exact_q_scale_bf16" in source
    assert "C_runtime_scalar" in source.split("def _winner", 1)[1]


def test_probe_declares_final_numerical_and_performance_stop_gates():
    source = _source()
    assert '"runtime_changes": False' in source
    assert '"kernel_abi_changes": False' in source
    assert '"no_third_numerical_blocker"' in source
    assert 'screen_tps >= 14.7' in source
    assert '"stop_mlx_compiled_kda_full_token_performance_gate"' in source
    assert '"stop_mlx_compiled_kda_after_third_numerical_blocker"' in source


def test_artifact_proves_exact_runtime_scalar_but_stops_on_speed():
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["probe_only"] is True
    assert artifact["runtime_changes"] is False
    assert artifact["kernel_abi_changes"] is False
    assert artifact["scale_contract"]["fp32_bits_hex"] == "0x3db504f3"
    assert artifact["selected_minimal_candidate"] == "C_runtime_scalar"
    acceptance = artifact["acceptance"]
    assert acceptance["all_34_layers_64_steps_byte_exact"] is True
    assert acceptance["repeatability_exact"] is True
    assert acceptance["strided_contiguous_exact"] is True
    assert acceptance["no_third_numerical_blocker"] is True
    assert acceptance["official_16_128_full_vocab_oracle_exact"] is True
    assert acceptance["full_token_64_step_logits_and_cache_exact"] is True
    assert acceptance["host_build_reduction_hard_floor_40pct"] is True
    assert acceptance["working_peak_delta_at_most_64mib"] is True
    assert acceptance["full_token_working_peak_delta_at_most_64mib"] is True
    assert acceptance["new_q_scale_metal_primitive_required"] is False
    assert acceptance["full_token_64_step_at_least_14_7_tps"] is False
    assert artifact["performance"]["full_token_64_step_tok_s"] < 14.7
    assert artifact["decision"] == (
        "stop_mlx_compiled_kda_full_token_performance_gate"
    )


def test_all_kda_layers_have_exact_q_scale_output_and_state():
    artifact = json.loads(ARTIFACT.read_text())
    assert len(artifact["all_34_kda_layers"]) == 34
    for row in artifact["all_34_kda_layers"].values():
        candidate = row["arms"]["C_runtime_scalar"]
        assert candidate["all_64_steps_byte_identical"] is True
        assert candidate["all_step_states_byte_identical"] is True
        assert candidate["conv_state_byte_identical"] is True
        assert candidate["recurrent_state_byte_identical"] is True
        assert candidate["compile_trace_calls"] == 1


def test_official_oracles_and_full_token_screen_are_exact():
    artifact = json.loads(ARTIFACT.read_text())
    oracle = artifact["official_oracle"]
    assert oracle["executed"] is True
    assert oracle["baseline"]["first_16_match"] is True
    assert oracle["baseline"]["full_128_match"] is True
    assert oracle["candidate"]["first_16_match"] is True
    assert oracle["candidate"]["full_128_match"] is True
    assert oracle["baseline_candidate_token_digest_exact"] is True
    assert oracle["all_compiled_layers_trace_once"] is True
    screen = oracle["full_token_screen"]
    assert screen["all_logits_and_final_cache_exact"] is True
    assert screen["candidate"]["tokens_per_second"] < 14.7
    assert screen["candidate"]["nan_count"] == 0
