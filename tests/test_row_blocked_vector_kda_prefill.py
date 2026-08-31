import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "probe_row_blocked_vector_kda_prefill.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-row-blocked-vector-kda-prefill-20260831.json"
)


def test_probe_scope_and_gates_are_explicit():
    source = SCRIPT.read_text()
    assert "ROW_BLOCKS = (1, 2, 4, 8)" in source
    assert "TOKENS = (128, 256, 512, 2048, 4096, 8192, 16384)" in source
    assert "KDA_LAYERS = tuple(" in source
    assert "REPRESENTATIVE_KDA_LAYERS = (0, 20, 44)" in source
    assert "all_operator_outputs_and_final_states_byte_identical" in source
    assert "whole_model_prefill_2k_and_4k_speedup_at_least_1_02" in source
    assert "decode_regression_at_most_1_percent_and_exact" in source
    assert '"probe_only": True' in source
    assert '"kernel_abi": False' in source
    assert '"server": False' in source
    assert '"admission": False' in source


def test_true_nope_fixture_is_structural_and_observed():
    source = SCRIPT.read_text()
    assert "attention.use_nope" in source
    assert "attention.qk_rope_head_dim == 0" in source
    assert 'not hasattr(attention, "rotary_emb")' in source
    assert "nn.RoPE.__call__" in source
    assert "mx.fast.rope" in source
    assert "selected_index_hashes_unchanged" in source
    assert "indexpool_token_position_processing_preserved" in source


def test_row_blocked_kernel_matches_current_on_metal():
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("MLX/Metal is unavailable")
    if not mx.metal.is_available():
        pytest.skip("MLX/Metal is unavailable")
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("row_blocked_kda_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    q, k, v, g, beta, state = module._operator_inputs(9, heads=2, strided=True)
    expected_y, expected_state = module._reference(q, k, v, g, beta, state)
    for block in module.ROW_BLOCKS:
        actual_y, actual_state = module.row_blocked_vector_kda(
            q, k, v, g, beta, state, row_block=block
        )
        mx.eval(expected_y, expected_state, actual_y, actual_state)
        assert mx.array_equal(expected_y, actual_y).item()
        assert mx.array_equal(expected_state, actual_state).item()


def test_artifact_is_complete_when_present():
    if not ARTIFACT.exists():
        pytest.skip("M3 Ultra artifact has not been generated yet")
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "glm53-row-blocked-vector-kda-prefill-v1"
    assert artifact["probe_only"] is True
    assert artifact["complete"] is True
    assert artifact["winner"] in (1, 2, 4, 8)
    assert artifact["acceptance"]["accepted"] == all(
        value
        for key, value in artifact["acceptance"].items()
        if key != "accepted"
    )
    assert artifact["acceptance"][
        "all_operator_outputs_and_final_states_byte_identical"
    ]
    assert artifact["acceptance"]["early_middle_late_and_all_34_kda_layers_exact"]
    assert artifact["acceptance"]["official_16_128_token_oracle_exact"]
    assert artifact["runtime_changes"] == {
        "kernel_abi": False,
        "server": False,
        "admission": False,
        "default_backend": False,
    }
