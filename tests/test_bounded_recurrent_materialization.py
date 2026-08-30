import importlib.util
from pathlib import Path

import pytest

try:
    import mlx.core  # noqa: F401
except ImportError:
    pytest.skip("MLX/Metal is unavailable", allow_module_level=True)


def _load_probe():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "probe_bounded_recurrent_materialization.py"
    )
    spec = importlib.util.spec_from_file_location("bounded_materialization_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evidence_steps_cover_production_boundaries():
    probe = _load_probe()
    steps = probe.evidence_steps(4096)
    for value in (255, 256, 257, 511, 512, 4095, 4096):
        assert value in steps


def test_probe_pins_direct_4096_gate_and_policy_identity():
    probe = _load_probe()
    source = Path(probe.__file__).read_text()
    assert probe.DEFAULT_STEPS == 4096
    assert probe.MATERIALIZATION_INTERVAL_TOKENS == 256
    assert probe.MATERIALIZATION_POLICY == "nested-cache-eval-clear-v1"
    assert '"cache_backend": "direct"' in source
    assert "experimental_compact_nope_dsa_cache" not in source
    assert "metal_buffer_count" not in source


def test_finalize_requires_exact_production_boundaries_and_hashes():
    probe = _load_probe()
    hashes = {str(step): f"h-{step}" for step in probe.evidence_steps(4096)}
    tokens = [7] * 4096
    reference = {
        "complete": True,
        "steps": 4096,
        "token_sha256": "same",
        "logits_hashes": hashes,
        "materialization_count": 81,
        "materialization_steps": list(range(50, 4097, 50)),
        "nan_count": 0,
        "metal_error": None,
        "end_to_end_tokens_per_second": 10.0,
        "telemetry": {"completed_materializations": 81},
        "token_trace": tokens,
    }
    production = {
        **reference,
        "materialization_count": 16,
        "materialization_steps": list(range(256, 4097, 256)),
        "end_to_end_tokens_per_second": 10.1,
        "telemetry": {
            "completed_materializations": 16,
            "last_materialization_step": 4096,
        },
    }
    artifact = {
        "steps": 4096,
        "arms": {"50": reference, "256": production},
        "compact_100k_policy_evidence": {
            "complete": True,
            "accepted": True,
            "interval_tokens": 256,
        },
    }
    probe._finalize(artifact)
    assert artifact["acceptance"]["accepted"] is True
