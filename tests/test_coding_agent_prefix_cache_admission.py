from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualify_coding_agent_prefix_cache_admission.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-coding-agent-prefix-cache-admission-20260905.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("coding_agent_admission_probe", SCRIPT)
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


def test_script_is_probe_only_atomic_and_uses_exact_hybrid_apc():
    source = SCRIPT.read_text()
    ast.parse(source)
    for required in (
        "APCManager",
        "model_apc_mode",
        "store_exact_cache",
        "lookup_exact_cache",
        "semantic_cache_digest",
        "semantic_component_digests",
        "semantic_cache_storage_alias_count",
        "_qualify_official_oracle",
        "uncached_suffix_reference",
        "warm_suffix",
        "temporary.replace(path)",
    ):
        assert required in source
    assert '"moe_backend": "direct"' in source
    assert '"cache_backend": "direct"' in source
    assert '"production_admission_changed": False' in source
    assert "experimental_compact_nope_dsa_cache" not in source
    assert "experimental_packed_grouped_moe" not in source


def test_controlled_fixture_is_exact_length_and_has_agent_semantics():
    module = _module()
    tokenizer = _CharacterTokenizer()
    corpus = "repository file\n" * 4096
    for context in module.CONTEXTS:
        fixture = module.build_coding_agent_fixture(tokenizer, context, corpus)
        assert len(fixture["prefix_token_ids"]) == context
        assert fixture["repository_payload_tokens"] > 0
        assert fixture["has_system_prompt"] is True
        assert fixture["has_tool_definitions"] is True
        assert fixture["has_repository_context"] is True
        assert fixture["has_conversation_history"] is True
        assert fixture["has_assistant_generation_boundary"] is True
        assert fixture["has_tool_call_and_result_suffix"] is True
        assert fixture["suffix_after_predicted_token_ids"]

    messages, tokens = module.build_server_smoke_messages(
        tokenizer, corpus, target_tokens=4096
    )
    assert 4096 - 64 <= tokens <= 4096
    assert module.REPOSITORY_MARKER not in messages[-1]["content"]


def test_context_parser_requires_unique_increasing_positive_values():
    module = _module()
    assert module._contexts("4096,8192,16384,32768") == module.CONTEXTS
    for value in ("", "0", "8192,4096", "4096,4096", "4096,-1"):
        with pytest.raises(Exception):
            module._contexts(value)


def test_artifact_when_present_proves_every_context_and_exact_state():
    if not ARTIFACT.exists():
        pytest.skip("long user-launched qualification has not produced its artifact")
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "glm53-coding-agent-prefix-cache-admission-v1"
    assert artifact["contexts"] == [4096, 8192, 16384, 32768]
    assert artifact["execution"] == {
        "moe_backend": "direct",
        "cache_backend": "direct",
        "apc_mode": "exact-hybrid-ram",
        "server_admission_bypassed_by_probe_only": True,
        "production_admission_changed": False,
        "runtime_backend_changed": False,
        "disk_apc_used": False,
    }
    if not artifact["complete"]:
        pytest.skip(
            "qualification artifact is an atomic progress record, not final evidence"
        )
    assert artifact["accepted"] is True
    assert artifact["last_completed_phase"] == "complete"
    assert artifact["server_smoke"]["accepted"] is True
    assert artifact["server_smoke"]["exact_prefix_hit_observed"] is True
    assert artifact["official_oracles"]["accepted"] is True
    assert all(
        row["generated_token_ids_exact"]
        and row["full_vocab_logits_hashes_exact"]
        for row in artifact["official_oracles"]["cases"].values()
    )
    assert set(artifact["cases"]) == {"4096", "8192", "16384", "32768"}
    for context, case in artifact["cases"].items():
        assert case["context_tokens"] == int(context)
        assert case["accepted"] is True
        assert all(case["acceptance"].values())
        assert case["apc"]["mode"] == "exact"
        assert case["apc"]["live_to_stored_alias_count"] == 0
        assert case["apc"]["stored_digest_immutable"] is True
        assert case["apc"]["stored_resident_bounded"] is True
        assert len(case["warm_turns"]) == 3
        for row in case["warm_turns"]:
            assert row["prefix_hit_tokens"] == int(context)
            assert row["restored_prefix_state_exact"] is True
            assert row["final_logits_exact"] is True
            assert row["final_state_exact"] is True
            assert row["component_state_exact"] is True
            assert row["stored_to_restored_alias_count"] == 0
            assert row["accounting"]["anonymous_bytes"] == 0


def test_script_separates_optional_http_smoke_from_model_residency():
    source = SCRIPT.read_text()
    assert 'choices=("model", "server-smoke")' in source
    assert "/v1/chat/completions" in source
    assert 'f"{args.server_url}/health"' in source
    assert "Run separately against an explicitly long-admission server" in source
