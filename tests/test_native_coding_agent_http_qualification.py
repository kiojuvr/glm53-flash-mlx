from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qualify_native_coding_agent_http.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-coding-agent-http-20260908.json"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "native_coding_agent_http_qualification", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _request_row(*, cached: int, digest: str, native_delta: int = 0):
    return {
        "usage": {
            "prompt_tokens": 32_768 if cached == 0 else 32_834,
            "completion_tokens": 256,
            "total_tokens": 33_024 if cached == 0 else 33_090,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
        "cached_tokens": cached,
        "choice_sha256": digest,
        "metrics": {"decode_tok_s": 15.1, "peak_memory_gb": 337.0},
        "native_execution_count_delta": native_delta,
    }


def _phase_row(*, native: bool):
    per_request = 255 * 11 if native else 0
    return {
        "fixture": {"prompt": "same"},
        "requests": {
            "cold_base": _request_row(
                cached=0, digest="cold", native_delta=per_request
            ),
            "tool_suffix": _request_row(
                cached=30_720, digest="suffix", native_delta=per_request
            ),
            "tool_suffix_warm": _request_row(
                cached=32_768, digest="suffix", native_delta=per_request
            ),
        },
        "cache_stats": {"exact_hits": 2, "rejects": 0, "evictions": 0},
    }


def test_script_is_split_across_separate_baseline_and_native_servers():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert 'choices=("baseline", "native")' in source
    assert "server_processes_are_separate" in source
    assert "native phase requires an accepted baseline phase first" in source
    assert 'artifact.setdefault("phases", {}).pop("native", None)' in source
    assert "temporary.replace(path)" in source
    assert "/v1/cache/reset" in source
    assert "/v1/cache/stats" in source
    assert "/v1/metrics" in source


def test_workload_is_real_32k_coding_agent_multiturn_and_bounded():
    module = _module()
    assert module.CONTEXT_TOKENS == 32_768
    assert module.DECODE_TOKENS == 256
    assert module.REQUEST_NAMES == (
        "cold_base",
        "tool_suffix",
        "tool_suffix_warm",
    )
    source = SCRIPT.read_text()
    assert "_repository_corpus(ROOT)" in source
    assert "build_exact_http_messages" in source
    assert "_extended_messages(base_messages)" in source
    assert "has_tool_call_and_result_suffix" in source


def test_openai_tool_arguments_are_normalized_only_for_template_rendering():
    module = _module()
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"README.md"}',
                    },
                }
            ],
        }
    ]
    normalized = module._messages_as_server_template_input(messages)
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == (
        '{"path":"README.md"}'
    )
    assert normalized[0]["tool_calls"][0]["function"]["arguments"] == {
        "path": "README.md"
    }


def test_phase_gates_prefix_reuse_native_execution_and_peak():
    module = _module()
    baseline = _phase_row(native=False)
    native = _phase_row(native=True)
    assert all(module._phase_local_checks("baseline", baseline).values())
    assert all(module._phase_local_checks("native", native).values())
    assert native["resource"]["observed_native_execution_count"] == 8_415
    assert native["resource"]["expected_native_execution_count"] == 8_415


def test_cross_backend_gate_requires_exact_outputs_usage_and_15_tps():
    module = _module()
    baseline = _phase_row(native=False)
    native = _phase_row(native=True)
    module._phase_local_checks("baseline", baseline)
    module._phase_local_checks("native", native)
    artifact = {
        "phases": {"baseline": baseline, "native": native},
        "source_native_runtime": {
            "decision": "keep_native_indexpool_runtime_and_pass_15_tps"
        },
    }
    assert all(module._cross_arm_checks(artifact).values())
    native["requests"]["tool_suffix_warm"]["metrics"]["decode_tok_s"] = 14.99
    assert not module._cross_arm_checks(artifact)[
        "native_warm_tool_suffix_decode_at_least_15_tps"
    ]


def test_qualification_artifact_when_present_is_authoritative():
    if not ARTIFACT.exists():
        pytest.skip("user-launched baseline/native HTTP qualification is pending")
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "glm53-native-coding-agent-http-v1"
    if not artifact["complete"]:
        pytest.skip("artifact is an atomic progress record")
    assert artifact["accepted"] is True
    assert artifact["decision"] == "qualify_native_backend_for_real_coding_agent_http"
    assert all(artifact["cross_arm_checks"].values())
    assert set(artifact["phases"]) == {"baseline", "native"}
    for phase in artifact["phases"].values():
        assert phase["accepted"] is True
        assert all(phase["checks"].values())
