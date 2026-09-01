import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "probe_compiled_packed_ffn_fp32_router.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-compiled-packed-ffn-fp32-router-20260901.json"
)


def test_four_arms_and_process_isolation_are_explicit():
    source = SCRIPT.read_text()
    assert '"A": {"compile_ffn": False, "resident_fp32_router": False}' in source
    assert '"B": {"compile_ffn": True, "resident_fp32_router": False}' in source
    assert '"C": {"compile_ffn": False, "resident_fp32_router": True}' in source
    assert '"D": {"compile_ffn": True, "resident_fp32_router": True}' in source
    assert "subprocess.run(" in source
    assert '"four_distinct_processes"' in source
    assert 'residual.Arm("B1", True)' in source


def test_compile_and_router_acceptance_contract_is_explicit():
    source = SCRIPT.read_text()
    assert "layer._ffn_c = None" in source
    assert "mx.compile(lambda value" in source
    assert "gate.weight.astype(mx.float32)" in source
    assert "router_raw_logits_indices_scores_exact" in source
    assert "layer_3_5_compiled_moe_and_hc_output_exact" in source
    assert "late_early_retention_at_least_0_95" in source
    assert "peak_at_most_340_gb" in source
    assert '"packed_runtime": False' in source
    assert '"kernel_abi": False' in source


def _load_probe():
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("MLX/Metal is unavailable")
    if not mx.metal.is_available():
        pytest.skip("MLX/Metal is unavailable")
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("compiled_ffn_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, mx


def test_resident_fp32_router_matmul_is_exact_on_metal():
    _, mx = _load_probe()
    x = (
        mx.sin(mx.arange(4 * 128, dtype=mx.float32) * 0.03125) * 0.5
    ).astype(mx.bfloat16).reshape(4, 128)
    weight = (
        mx.cos(mx.arange(16 * 128, dtype=mx.float32) * 0.015625) * 0.25
    ).astype(mx.bfloat16).reshape(16, 128)
    existing = x.astype(mx.float32) @ weight.astype(mx.float32).T
    resident = mx.contiguous(weight.astype(mx.float32))
    mx.eval(resident)
    actual = x.astype(mx.float32) @ resident.T
    mx.eval(existing, actual)
    assert mx.array_equal(existing, actual).item()


def test_percentile_uses_nearest_rank():
    probe, _ = _load_probe()
    assert probe._percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
    assert probe._percentile([], 0.95) == 0.0


def test_artifact_is_complete_when_present():
    if not ARTIFACT.exists():
        pytest.skip("M3 Ultra compiled FFN/router artifact has not been generated")
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "glm53-compiled-packed-ffn-fp32-router-v1"
    assert artifact["complete"] is True
    assert artifact["probe_only"] is True
    assert artifact["process_isolation"] is True
    assert set(artifact["screens"]) == {"A", "B", "C", "D"}
    assert artifact["runtime_candidate_accepted"] == bool(artifact["accepted_arms"])
    assert artifact["runtime_changes"] == {
        "admission": False,
        "apc": False,
        "kernel_abi": False,
        "packed_runtime": False,
        "server": False,
    }
