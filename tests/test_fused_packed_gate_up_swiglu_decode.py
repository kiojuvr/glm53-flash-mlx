import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "probe_fused_packed_gate_up_swiglu_decode.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-fused-packed-gate-up-swiglu-decode-20260901.json"
)


def test_probe_scope_and_release_gates_are_explicit():
    source = SCRIPT.read_text()
    assert "REPRESENTATIVE_LAYERS = (3, 5)" in source
    assert "DECODE_STEPS = 4096" in source
    assert "gate_output" in source
    assert "up_output" in source
    assert "activated_hidden" in source
    assert "weighted_expert_output" in source
    assert "selected_expert_moe_decode_speedup_at_least_1_20" in source
    assert "full_model_2k_speedup_at_least_1_12" in source
    assert "full_model_4096_at_least_14_tps" in source
    assert "decode_4096_final_kda_dsa_state_exact" in source
    assert '"packed_runtime": False' in source
    assert '"kernel_abi": False' in source


def test_fused_kernel_preserves_projection_rounding_boundary():
    source = SCRIPT.read_text()
    assert "T gate_t = T(gate_total);" in source
    assert "T up_t = T(up_total);" in source
    assert "metal::exp(metal::abs(gate_activation))" in source
    assert "(gate_activation < 0)" in source
    assert "T silu_value = gate_activation * sigmoid_value;" in source
    assert "T activated = silu_value * up_activation;" in source


def _load_probe():
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("MLX/Metal is unavailable")
    if not mx.metal.is_available():
        pytest.skip("MLX/Metal is unavailable")
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("fused_packed_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, mx


def test_fused_diagnostic_matches_existing_projection_on_metal():
    probe, mx = _load_probe()
    from glm53_flash_mlx.packed import PackedFP8ExpertBank

    experts = 8
    hidden = 128
    intermediate = 128
    weights = (
        mx.arange(experts * 2 * intermediate * hidden, dtype=mx.uint32) % 256
    ).astype(mx.uint8).reshape(experts, 2 * intermediate, hidden)
    scales = mx.full((experts, 2, 1), 0.015625, dtype=mx.float32)
    down = mx.zeros((experts, hidden, intermediate), dtype=mx.uint8)
    down_scales = mx.ones((experts, 1, 1), dtype=mx.float32)
    bank = PackedFP8ExpertBank(
        weights,
        scales,
        down,
        down_scales,
        intermediate_size=intermediate,
    )
    x = (mx.sin(mx.arange(hidden, dtype=mx.float32) * 0.03125) * 0.25).astype(
        mx.bfloat16
    )
    expert_ids = mx.arange(experts, dtype=mx.uint32)
    gate = probe._packed_selected_projection(x, expert_ids, bank, row_offset=0)
    up = probe._packed_selected_projection(
        x, expert_ids, bank, row_offset=intermediate
    )
    expected_hidden = probe.nn.silu(mx.minimum(gate, 20.0)) * mx.clip(
        up, -20.0, 20.0
    )
    actual_gate, actual_up, actual_hidden = probe.fused_packed_gate_up_swiglu(
        x, expert_ids, bank, limit=20.0, diagnostics=True
    )
    mx.eval(gate, up, expected_hidden, actual_gate, actual_up, actual_hidden)
    assert mx.array_equal(gate, actual_gate).item()
    assert mx.array_equal(up, actual_up).item()
    assert mx.array_equal(expected_hidden, actual_hidden).item()


def test_artifact_is_complete_when_present():
    if not ARTIFACT.exists():
        pytest.skip("M3 Ultra fused packed artifact has not been generated yet")
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "glm53-fused-packed-gate-up-swiglu-decode-v1"
    assert artifact["complete"] is True
    assert artifact["probe_only"] is True
    if "acceptance" not in artifact:
        assert artifact["runtime_candidate_accepted"] is False
        assert artifact["stage_exact"] is False
    else:
        assert artifact["acceptance"]["accepted"] == all(
            value
            for key, value in artifact["acceptance"].items()
            if key != "accepted"
        )
        assert artifact["runtime_changes"] == {
            "admission": False,
            "apc": False,
            "kernel_abi": False,
            "packed_runtime": False,
            "server": False,
        }
