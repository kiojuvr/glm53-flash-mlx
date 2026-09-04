from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualify_latest_official_chat_template_revision.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-latest-official-chat-template-revision-20260905.json"
)
CHECKPOINT_DIGEST = (
    "fa0e072f4e9bcde1c8b0ce3e0a35387ecda1d7a6fc23d7a1e6cbc9063043e708"
)
TOKENIZER_DIGEST = (
    "a6c68bb04007faf831fe85cf8d62da7389c5590df7c519365800950a163787bb"
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def test_qualification_script_is_read_only_and_covers_semantic_fixtures():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert "ThreadPoolExecutor(max_workers=2)" in source
    assert "AutoTokenizer.from_pretrained" in source
    assert "load_tokenizer(root)._tokenizer" in source
    assert "server_not_started" in source
    for fixture in (
        "accepted_oracle_prompt",
        "system_user",
        "multi_turn",
        "assistant",
        "tool_declaration",
        "tool_call",
        "tool_result",
        "reasoning_effort",
        "clear_thinking",
        "none_content",
    ):
        assert f'"{fixture}"' in source


def test_artifact_proves_template_only_revision_and_all_gates_pass():
    artifact = _artifact()
    assert artifact["schema"] == "glm53-latest-official-chat-template-revision-v1"
    assert artifact["complete"] is True
    assert artifact["last_completed_phase"] == "complete"
    assert len(artifact["acceptance"]) == 20
    assert all(artifact["acceptance"].values())
    assert artifact["classification"] == {
        "kind": "runtime-metadata-template-only",
        "weights_changed": False,
        "tokenizer_changed": False,
        "processor_changed": False,
        "chat_template_changed": True,
        "ancillary_repository_files_changed": ["README.md"],
        "requires_fp8_kernel_requalification": False,
        "requires_template_semantics_qualification": True,
    }


def test_all_weight_payloads_and_non_template_metadata_are_byte_identical():
    artifact = _artifact()
    baseline = artifact["weights"]["baseline"]
    candidate = artifact["weights"]["candidate"]
    assert baseline == candidate
    assert baseline["shard_count"] == 62
    assert baseline["file_count"] == 63
    assert baseline["total_bytes"] == 328_345_862_285
    assert baseline["checkpoint_digest"] == CHECKPOINT_DIGEST

    metadata = artifact["metadata"]
    changed = {
        name
        for name in metadata["baseline"]
        if metadata["baseline"][name]["sha256"]
        != metadata["candidate"][name]["sha256"]
    }
    assert changed == {"chat_template.jinja", "README.md"}
    assert artifact["metadata_unified_diffs"]["README.md"]
    assert artifact["metadata_unified_diffs"]["config.json"] == ""
    assert artifact["metadata_unified_diffs"]["tokenizer_config.json"] == ""
    assert artifact["metadata_unified_diffs"]["generation_config.json"] == ""


def test_checkpoint_tokenizer_and_template_identity_are_independent():
    artifact = _artifact()
    identities = artifact["revision_identity"]
    baseline = identities["baseline"]
    candidate = identities["candidate"]
    assert baseline["checkpoint_digest"] == candidate["checkpoint_digest"] == (
        CHECKPOINT_DIGEST
    )
    assert baseline["tokenizer_digest"] == candidate["tokenizer_digest"] == (
        TOKENIZER_DIGEST
    )
    assert baseline["chat_template_digest"] != candidate["chat_template_digest"]
    assert baseline["chat_template_revision"] != candidate["chat_template_revision"]
    assert baseline["namespace_sha256"] != candidate["namespace_sha256"]
    assert artifact["provenance"]["runtime_source_release"] == (
        "a9d85675e3ebd559868e37a297a3b3cc7c74833e"
    )
    for name in ("baseline", "candidate"):
        report = artifact["checkpoint_reports"][name]
        assert report["server_ready"] is True
        assert report["tokenizer_digest"] == TOKENIZER_DIGEST
        assert report["chat_template_digest"] == identities[name][
            "chat_template_digest"
        ]


def test_runtime_render_matches_official_and_records_generation_boundaries():
    cases = _artifact()["template_cases"]
    assert all(
        row["baseline_runtime_exact"] and row["candidate_runtime_exact"]
        for row in cases.values()
    )
    assert set(cases) == {
        "accepted_oracle_prompt",
        "system_user",
        "multi_turn",
        "assistant",
        "tool_declaration",
        "tool_call",
        "tool_result",
        "reasoning_effort",
        "clear_thinking",
        "none_content",
    }
    assert all(
        row["candidate"]["boundaries"]["markers"]["<|assistant|>"][
            "token_offsets"
        ]
        for row in cases.values()
    )
    assert cases["accepted_oracle_prompt"]["token_ids_changed"] is False
    assert cases["accepted_oracle_prompt"]["rendered_text_changed"] is False


def test_candidate_fixes_none_content_and_preserves_declared_tool_order():
    cases = _artifact()["template_cases"]
    assert "None" in cases["none_content"]["baseline"]["rendered_text"]
    assert "None" not in cases["none_content"]["candidate"]["rendered_text"]
    assert "None" not in cases["tool_call"]["candidate"]["rendered_text"]
    rendered = cases["tool_result"]["candidate"]["rendered_text"]
    assert rendered.index("sunny") < rendered.index("12:00")
    assert cases["reasoning_effort"]["rendered_text_changed"] is False
    assert cases["clear_thinking"]["rendered_text_changed"] is False
