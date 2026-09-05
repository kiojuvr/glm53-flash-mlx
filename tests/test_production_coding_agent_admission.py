from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualify_production_coding_agent_admission.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-production-coding-agent-admission-20260905.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("production_admission_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _CharacterTokenizer:
    def encode(self, value, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in value]

    def apply_chat_template(
        self, messages, *, tools, add_generation_prompt, tokenize
    ):
        assert tools
        assert add_generation_prompt is True
        rendered = "<tools>" + json.dumps(tools, sort_keys=True) + "</tools>"
        for message in messages:
            rendered += f"<{message['role']}>" + str(message.get("content", ""))
        rendered += "<assistant><think>"
        return self.encode(rendered) if tokenize else rendered


def test_probe_has_separate_residency_and_http_phases():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert 'choices=("model", "http")' in source
    assert "MODEL_PREFIX_TOKENS = 32768" in source
    assert "MODEL_CONTINUATION_TOKENS = 4096" in source
    assert "HTTP_CONTEXTS = (8192, 16384, 32768)" in source
    assert "HTTP_DECODE_TOKENS = 256" in source
    assert "HTTP_PREFILL_ALIGNMENT_TOKENS = 2048" in source
    assert "DEFAULT_FIRST_TOKEN_TIMEOUT_SECONDS" in source
    assert '"--http-contexts"' in source
    assert "EXACT_APC_PREFIX_GUARD_TOKENS" in source
    assert "EXACT_APC_STORE_POLICY" in source
    assert "temporary.replace(path)" in source
    assert "/v1/cache/reset" in source
    assert "/v1/metrics" in source
    assert "QualificationPreconditionError" in source
    assert "requires an accepted model phase first" in source


def test_exact_http_fixture_hits_requested_lengths():
    module = _module()
    tokenizer = _CharacterTokenizer()
    corpus = "repository evidence line\n" * 4096
    for target in (8192, 16384, 32768, 32769):
        messages, count = module.build_exact_http_messages(
            tokenizer, corpus, target
        )
        assert count == target
        assert module.prior.REPOSITORY_MARKER not in messages[-1]["content"]


def test_artifact_when_present_proves_model_http_and_boundary():
    if not ARTIFACT.exists():
        pytest.skip("user-launched production admission qualification is pending")
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "glm53-production-coding-agent-admission-v1"
    if not artifact["complete"]:
        pytest.skip("artifact is an atomic progress record")
    assert artifact["accepted"] is True
    assert all(artifact["acceptance"].values())
    model = artifact["model_32k_4096"]
    assert model["accepted"] is True
    assert model["baseline"]["steps"] == 4096
    assert model["replay"]["steps"] == 4096
    assert model["baseline"]["materializations"] == 16
    assert model["replay"]["materializations"] == 16
    assert all(model["checks"].values())

    http = artifact["http"]
    assert http["accepted"] is True
    assert set(http["cases"]) == {"8192", "16384", "32768"}
    assert all(row["accepted"] for row in http["cases"].values())
    assert all(all(row["checks"].values()) for row in http["cases"].values())
    assert all(http["boundary"]["checks"].values())
    assert http["boundary"]["status"] == 400


def test_total_context_boundary_is_authoritative():
    module = _module()
    module.validate_admission(
        32768,
        4096,
        max_generation_tokens=4096,
        max_context_tokens=36864,
    )
    with pytest.raises(ValueError, match="total context"):
        module.validate_admission(
            32769,
            4096,
            max_generation_tokens=4096,
            max_context_tokens=36864,
        )


@pytest.mark.parametrize(
    ("prompt_tokens", "expected_prefix"),
    [
        (8192, 6144),
        (8258, 8192),
        (16384, 14336),
        (32768, 30720),
    ],
)
def test_http_exact_apc_prefix_matches_absolute_prefill_geometry(
    prompt_tokens, expected_prefix
):
    module = _module()
    assert module._expected_exact_apc_prefix(prompt_tokens) == expected_prefix


def test_choice_signature_ignores_transport_tool_call_ids():
    module = _module()
    left = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "reasoning_content": "inspect",
                    "tool_calls": [
                        {
                            "id": "call-random-a",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                "logprobs": {"content": []},
            }
        ],
        "usage": {"completion_tokens": 12},
    }
    right = json.loads(json.dumps(left))
    right["choices"][0]["message"]["tool_calls"][0]["id"] = "call-random-b"
    assert module._choice_signature(left) == module._choice_signature(right)
