import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "probe_residual_packed_decode_moe_fusion.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-residual-packed-decode-moe-fusion-20260901.json"
)


def test_probe_scope_and_four_arms_are_explicit():
    source = SCRIPT.read_text()
    assert '"A": Arm("existing", False)' in source
    assert '"B": Arm(aggregation, False)' in source
    assert '"C": Arm("existing", True)' in source
    assert '"D": Arm(aggregation, True)' in source
    assert 'for variant in ("existing", "B1", "B2")' in source
    assert "D_at_least_15_tps" in source
    assert "D_median_at_most_66_67_ms" in source
    assert '"packed_runtime": False' in source
    assert '"kernel_abi": False' in source


def test_exact_mlx_sigmoid_tree_and_bf16_boundaries_are_preserved():
    source = SCRIPT.read_text()
    assert "metal::exp(metal::abs(gate_activation))" in source
    assert "T silu_value = gate_activation * sigmoid_value;" in source
    assert "T down_value = T(reduced);" in source
    assert "raw_down.astype(mx.float32) * flat_scores[:, None]" in source
    assert "mx.sum(weighted, axis=0)" in source


def _load_probe():
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("MLX/Metal is unavailable")
    if not mx.metal.is_available():
        pytest.skip("MLX/Metal is unavailable")
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("residual_moe_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, mx


def test_custom_down_aggregation_is_exact_on_metal():
    probe, mx = _load_probe()
    from glm53_flash_mlx.packed import PackedFP8ExpertBank
    from glm53_flash_mlx.ownership import TensorLayout, owned_tensor

    experts = 8
    hidden_size = 128
    intermediate = 128
    gate_up = (
        mx.arange(experts * 2 * intermediate * hidden_size, dtype=mx.uint32) % 256
    ).astype(mx.uint8).reshape(experts, 2 * intermediate, hidden_size)
    gate_up_scale = mx.full((experts, 2, 1), 0.015625, dtype=mx.float32)
    down = (
        mx.arange(experts * hidden_size * intermediate, dtype=mx.uint32) * 13 % 256
    ).astype(mx.uint8).reshape(experts, hidden_size, intermediate)
    down_scale = mx.full((experts, 1, 1), 0.0078125, dtype=mx.float32)
    bank = PackedFP8ExpertBank(
        owned_tensor(gate_up, layout=TensorLayout.ROW_MAJOR_CONTIGUOUS),
        owned_tensor(gate_up_scale, layout=TensorLayout.ROW_MAJOR_CONTIGUOUS),
        owned_tensor(down, layout=TensorLayout.ROW_MAJOR_CONTIGUOUS),
        owned_tensor(down_scale, layout=TensorLayout.ROW_MAJOR_CONTIGUOUS),
        intermediate_size=intermediate,
    )
    activated = (
        mx.sin(mx.arange(experts * intermediate, dtype=mx.float32) * 0.015625)
        * 0.5
    ).astype(mx.bfloat16).reshape(experts, intermediate)
    expert_ids = mx.arange(experts, dtype=mx.uint32)
    scores = mx.array(
        [0.28125, 0.203125, 0.15625, 0.125, 0.09375, 0.0625, 0.046875, 0.03125],
        dtype=mx.float32,
    )
    raw = probe.d99f._packed_down_raw(activated, expert_ids, bank)
    expected_weighted = raw.astype(mx.float32) * scores[:, None]
    expected_reduced = mx.sum(expected_weighted, axis=0)
    expected = expected_reduced.astype(mx.bfloat16)
    b1, b1_weighted, b1_reduced = probe.aggregate_b1(
        raw, scores, diagnostics=True
    )
    b2_weighted = probe.weighted_down_b2(activated, scores, expert_ids, bank)
    b2, b2_reduced = probe.reduce_b2(
        b2_weighted, output_dtype=mx.bfloat16, diagnostics=True
    )
    mx.eval(
        expected_weighted,
        expected_reduced,
        expected,
        b1,
        b1_weighted,
        b1_reduced,
        b2,
        b2_weighted,
        b2_reduced,
    )
    assert mx.array_equal(expected_weighted, b1_weighted).item()
    assert mx.array_equal(expected_reduced, b1_reduced).item()
    assert mx.array_equal(expected, b1).item()
    assert mx.array_equal(expected_weighted, b2_weighted).item()
    assert mx.array_equal(expected_reduced, b2_reduced).item()
    assert mx.array_equal(expected, b2).item()


def test_shared_gate_up_swiglu_is_exact_on_metal():
    probe, mx = _load_probe()
    from glm53_flash_mlx.fp8 import BlockFP8Linear

    hidden_size = 128
    intermediate = 128
    shared = SimpleNamespace(
        gate_proj=BlockFP8Linear(hidden_size, intermediate),
        up_proj=BlockFP8Linear(hidden_size, intermediate),
        down_proj=BlockFP8Linear(intermediate, hidden_size),
    )
    shared.gate_proj.weight = (
        mx.arange(intermediate * hidden_size, dtype=mx.uint32) % 256
    ).astype(mx.uint8).reshape(intermediate, hidden_size)
    shared.up_proj.weight = (
        (mx.arange(intermediate * hidden_size, dtype=mx.uint32) * 7 + 3) % 256
    ).astype(mx.uint8).reshape(intermediate, hidden_size)
    shared.down_proj.weight = mx.zeros(
        (hidden_size, intermediate), dtype=mx.uint8
    )
    shared.gate_proj.weight_scale_inv = mx.full(
        (1, 1), 0.015625, dtype=mx.float32
    )
    shared.up_proj.weight_scale_inv = mx.full(
        (1, 1), 0.0078125, dtype=mx.float32
    )
    x = (mx.cos(mx.arange(hidden_size, dtype=mx.float32) * 0.03125) * 0.25).astype(
        mx.bfloat16
    )
    gate, up, activated, _ = probe._existing_shared(shared, x, 10.0)
    actual_gate, actual_up, actual_activated = probe.fused_shared_gate_up_swiglu(
        x, shared, limit=10.0, diagnostics=True
    )
    mx.eval(gate, up, activated, actual_gate, actual_up, actual_activated)
    assert mx.array_equal(gate, actual_gate).item()
    assert mx.array_equal(up, actual_up).item()
    assert mx.array_equal(activated, actual_activated).item()


def test_artifact_is_complete_when_present():
    if not ARTIFACT.exists():
        pytest.skip("M3 Ultra residual MoE fusion artifact has not been generated")
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "glm53-residual-packed-decode-moe-fusion-v1"
    assert artifact["complete"] is True
    assert artifact["probe_only"] is True
    assert artifact["runtime_candidate_accepted"] == (
        all(artifact["correctness"].values())
        and all(artifact["performance"].values())
    )
    assert artifact["runtime_changes"] == {
        "admission": False,
        "apc": False,
        "kernel_abi": False,
        "packed_runtime": False,
        "server": False,
    }
