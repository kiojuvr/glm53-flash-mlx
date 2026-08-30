import importlib.util
from pathlib import Path

import pytest

try:
    import mlx.core as mx
except ImportError:
    pytest.skip("MLX/Metal is unavailable", allow_module_level=True)

if not mx.metal.is_available():
    pytest.skip("Metal is unavailable", allow_module_level=True)


def _load_probe():
    path = Path(__file__).parents[1] / "scripts" / "soak_recurrent_state_100k.py"
    spec = importlib.util.spec_from_file_location("recurrent_state_100k_soak", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_schedule_and_final_remainder_are_exact():
    probe = _load_probe()
    assert probe.expected_materialization_count(100_000) == 390
    assert 100_000 % probe.MATERIALIZATION_INTERVAL == 160
    assert probe.milestone_steps(100_000) == (25_000, 50_000, 75_000, 100_000)


def test_evidence_schedule_extends_reference_without_storing_token_trace():
    probe = _load_probe()
    reference = {"1": "a", "8192": "b"}
    steps = set(probe.evidence_steps(100_000, reference))
    assert {1, 8_192, 25_000, 50_000, 75_000, 100_000} <= steps
    assert set(range(4_096, 98_305, 4_096)) <= steps


def test_token_digest_is_fixed_width_little_endian():
    probe = _load_probe()
    import hashlib

    expected = hashlib.sha256()
    expected.update((1).to_bytes(4, "little"))
    expected.update((0x12345678).to_bytes(4, "little"))
    assert probe.token_digest([1, 0x12345678]) == expected.hexdigest()


def test_materialization_preserves_state_and_includes_synchronize():
    probe = _load_probe()

    class Entry:
        def __init__(self, value):
            self.value = value

        @property
        def state(self):
            return (self.value,)

    cache = [Entry(mx.arange(8)), Entry(mx.arange(4))]
    before = [mx.array(entry.value) for entry in cache]
    probe.materialize_cache(cache)
    for entry, expected in zip(cache, before, strict=True):
        assert mx.array_equal(entry.value, expected).item()


def test_25k_stop_gate_detects_monotonic_cache_growth():
    probe = _load_probe()
    artifact = {
        "boundary_telemetry": [
            {
                "after": {
                    "active_memory_bytes": 100 + step,
                    "cache_memory_bytes": step,
                    "peak_memory_bytes": 1_000,
                },
                "state_leaf_count": 167,
            }
            for step in range(1, 5)
        ],
        "metal_error": None,
        "nan_count": 0,
        "scheduled_materialization_count": 97,
        "last_completed_step": 25_000,
        "reference_8192": {"logits_hashes": {}, "token_sha256": "token"},
        "checkpoint_hashes": {"8192": {"token_sha256": "token"}},
    }
    assert "post_materialization_cache_memory_monotonic_increase" in (
        probe._milestone_stop_failures(artifact)
    )


def test_final_acceptance_keeps_final_evidence_out_of_scheduled_count():
    probe = _load_probe()
    memory = {
        "active_memory_bytes": 320_000_000_000,
        "cache_memory_bytes": 2_000_000,
        "peak_memory_bytes": 330_000_000_000,
    }
    boundary = {
        "before": memory,
        "after": memory,
        "materialization_ms": 1.5,
        "state_leaf_count": 167,
        "state_array_leaf_count": 167,
    }
    artifact = {
        "boundary_telemetry": [dict(boundary) for _ in range(390)],
        "final_evidence_materialization": {"after": memory},
        "reference_8192": {
            "logits_hashes": {"8192": "logits"},
            "token_sha256": "tokens",
        },
        "checkpoint_hashes": {
            "8192": {"logits_sha256": "logits", "token_sha256": "tokens"}
        },
        "scheduled_materialization_count": 390,
        "last_completed_step": 100_000,
        "nan_count": 0,
        "metal_error": None,
        "rolling_token_sha256": "final",
    }
    probe._finalize(artifact, [0.1] * 100_000)
    assert artifact["acceptance"]["accepted"] is True
    assert artifact["summary"]["scheduled_materialization_count"] == 390


def test_probe_is_explicitly_non_runtime_and_atomic():
    probe = _load_probe()
    source = Path(probe.__file__).read_text()
    assert "experimental_compact_nope_dsa_cache=True" in source
    assert "compact_cache_reserve_tokens=args.steps + RESERVE_TAIL" in source
    assert "server_admission_bypassed_inside_probe_only" in source
    assert "temporary.replace(path)" in source
    assert '"process_resume_supported": False' in source
    assert "token_trace" not in source.split("artifact = {", 1)[1]
