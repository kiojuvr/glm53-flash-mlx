from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import mlx.core  # noqa: F401
except ImportError:
    pytest.skip("MLX/Metal is unavailable", allow_module_level=True)


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
ARTIFACT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-stateless-decode-compilation-envelopes-20260901.json"
)


def _load(name: str, path: Path):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_envelope_arm_contract_keeps_stateful_values_outside_compile():
    module = _load(
        "stateless_decode_envelopes_test",
        SCRIPTS / "stateless_decode_envelopes.py",
    )
    assert module.ARMS == {
        "A": "compiled-ffn-baseline",
        "B": "post-attention-through-ffn",
        "C": "ffn-through-next-attention-handoff",
        "D": "post-attention-through-next-attention-handoff",
    }
    for function in (
        module._finish_attention_and_ffn,
        module._finish_ffn_and_prepare_next,
        module._finish_attention_ffn_and_prepare_next,
    ):
        parameters = inspect.signature(function).parameters
        assert "cache" not in parameters
        assert "offset" not in parameters
        assert "materialization" not in parameters
    source = inspect.getsource(module.StatelessDecodeEnvelopeRunner.__init__)
    assert "self_attn" not in source
    assert "cache" not in source


def test_every_arm_builds_exactly_42_sparse_compile_callables():
    module = _load(
        "stateless_decode_envelopes_runner_test",
        SCRIPTS / "stateless_decode_envelopes.py",
    )

    class Layer:
        compile_ffn = False
        _ffn_c = None

    language = SimpleNamespace(
        model=SimpleNamespace(layers=[Layer() for _ in range(45)])
    )
    for arm in module.ARMS:
        runner = module.StatelessDecodeEnvelopeRunner(language, arm)
        assert runner.policy.compiled_sparse_layers == 42
        assert runner.policy.compiled_callable_count == 42
        assert not runner.policy.stateful_attention_inside_compiled_callable
        assert not runner.policy.mutable_cache_inside_compiled_callable


def _row(module, *, idle: float, busy: float, submissions: int):
    child = {
        "generated_token_sha256": "tokens",
        "evidence_full_vocab_hashes": {"1": "logits"},
        "post_cache_state_hash": "cache",
        "direct_compact_parity": {"full_vocab_exact": True},
        "physical_capacity_unchanged": True,
        "cache_leaf_count_constant": True,
        "idle_without_forward_state_unchanged": True,
        "nan_count": 0,
        "metal_error": None,
        "materialization_count": 1,
        "active_memory_drift_bytes": 0,
        "latency_p50_ms": 66.0,
        "decode_tokens_per_second": 1000.0 / 66.0,
        "compile_warmup": {
            "active_delta_bytes": 1 << 20,
            "active_before_bytes": 100 << 20,
            "peak_bytes": 101 << 20,
            "working_peak_bytes": 1 << 20,
        },
    }
    return {
        "child": child,
        "telemetry": {
            "gpu_busy_ms": busy * module.TRACED_TOKENS,
            "gpu_idle_gap_ms": idle * module.TRACED_TOKENS,
            "command_buffer_submission_rows": submissions
            * module.TRACED_TOKENS,
        },
    }


def test_sweep_merge_separates_correctness_from_screening(tmp_path):
    module = _load(
        "stateless_decode_sweep_test",
        SCRIPTS / "sweep_stateless_decode_compilation_envelopes.py",
    )
    output = tmp_path / "artifact.json"
    module._merge(output, "A", _row(module, idle=6.0, busy=63.0, submissions=10))
    module._merge(output, "B", _row(module, idle=5.0, busy=63.1, submissions=7))
    module._merge(output, "C", _row(module, idle=5.8, busy=63.0, submissions=9))
    artifact = module._merge(
        output, "D", _row(module, idle=4.9, busy=63.1, submissions=7)
    )
    assert artifact["complete"]
    assert artifact["correctness_complete"]
    assert artifact["screening_passed_arms"] == ["B", "D"]
    assert artifact["production_candidate_arms"] == ["B", "C", "D"]


def test_committed_m3ultra_sweep_is_exact_and_rejects_all_candidates():
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"]
    assert artifact["correctness_complete"]
    assert artifact["screening_passed_arms"] == []
    assert artifact["production_candidate_arms"] == []
    assert set(artifact["arms"]) == {"A", "B", "C", "D"}
    for arm in artifact["arms"].values():
        child = arm["child"]
        assert child["nan_count"] == 0
        assert child["metal_error"] is None
        assert child["materialization_count"] == 1
        assert child["cache_leaf_count_constant"]
        assert child["physical_capacity_unchanged"]
        assert child["direct_compact_parity"]["full_vocab_exact"]
        assert not child["capture_attached_first_token_is_steady_evidence"]
    for name in ("B", "C", "D"):
        assert not artifact["comparisons_to_A"][name]["short_screen_passed"]
