import hashlib
import json
import struct
from pathlib import Path

import pytest

from glm53_flash_mlx import manifest
from glm53_flash_mlx.manifest import (
    APPROVED_CHAT_TEMPLATE_REVISIONS,
    EXPECTED_CHECKPOINT_DIGEST,
    EXPECTED_TOKENIZER_DIGEST,
    ManifestError,
    attest_checkpoint,
    checkpoint_content_digest,
    inspect_checkpoint,
    revision_identity_from_digests,
)


def _write_safetensors(path: Path, tensor_names: list[str]):
    header = {}
    offset = 0
    for name in tensor_names:
        header[name] = {"dtype": "BF16", "shape": [1], "data_offsets": [offset, offset + 2]}
        offset += 2
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)


def make_checkpoint(tmp_path: Path, *, quantized=True):
    layer_types = [
        "deepseek_sparse_attention" if i % 4 == 3 else "linear_attention"
        for i in range(45)
    ]
    config = {
        "model_type": "glm5_next",
        "text_config": {
            "num_hidden_layers": 45,
            "layer_types": layer_types,
            "n_routed_experts": 288,
            "num_experts_per_tok": 8,
            "hidden_size": 4096,
            "max_position_embeddings": 1048576,
            "qk_rope_head_dim": 0,
            "mla_use_nope": True,
            "kv_lora_rank": 512,
            "index_topk": 2048,
            "index_kpool": 4,
        },
        "vision_config": {},
    }
    if quantized:
        config["quantization"] = {"group_size": 64, "bits": 6}
    else:
        config["quantization_config"] = {"quant_method": "fp8"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    weight_map = {
        f"language_model.model.layers.{i}.input_layernorm.weight": "one.safetensors"
        for i in range(45)
    }
    weight_map["language_model.model.embed_tokens.weight"] = "one.safetensors"
    _write_safetensors(tmp_path / "one.safetensors", list(weight_map))
    index = {
        "metadata": {"total_size": len(weight_map) * 2},
        "weight_map": weight_map,
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))


def test_incomplete_checkpoint_is_not_server_ready(tmp_path):
    make_checkpoint(tmp_path, quantized=False)
    report = inspect_checkpoint(tmp_path)
    assert not report.server_ready
    assert report.source_format == "hf-fp8"
    assert report.kda_layers == tuple(i for i in range(45) if i % 4 != 3)
    assert report.dsa_layers == tuple(range(3, 45, 4))
    assert any("shards=1" in failure for failure in report.audit_failures)
    with pytest.raises(ManifestError, match="strict official-FP8 audit"):
        inspect_checkpoint(tmp_path, require_server_ready=True)


def test_converted_checkpoint_is_rejected(tmp_path):
    make_checkpoint(tmp_path, quantized=True)
    assert inspect_checkpoint(tmp_path).source_format == "mlx-affine"
    with pytest.raises(ManifestError, match="strict official-FP8 audit"):
        inspect_checkpoint(tmp_path, require_server_ready=True)


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    [
        ("qk_rope_head_dim", 64, "qk_rope_head_dim != 0"),
        ("mla_use_nope", False, "mla_use_nope is not true"),
        ("kv_lora_rank", 256, "kv_lora_rank != 512"),
    ],
)
def test_nope_dsa_schema_is_explicitly_audited(tmp_path, field, value, failure):
    make_checkpoint(tmp_path, quantized=False)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["text_config"][field] = value
    config_path.write_text(json.dumps(config))

    report = inspect_checkpoint(tmp_path)

    assert failure in report.audit_failures
    assert report.attention_cache_abi.startswith("glm53-nope-dsa-v1")


def test_missing_shard_is_rejected(tmp_path):
    make_checkpoint(tmp_path)
    (tmp_path / "one.safetensors").unlink()
    with pytest.raises(ManifestError, match="missing 1 shard"):
        inspect_checkpoint(tmp_path)


def test_disk_cache_content_digest_changes_when_payload_changes(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text('{"version":"1"}')
    (tmp_path / "one.safetensors").write_bytes(b"same-size-payload-a")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "one.safetensors"}})
    )
    monkeypatch.setattr(
        manifest,
        "inspect_checkpoint",
        lambda *_args, **_kwargs: type("Report", (), {"layout_digest": "layout"})(),
    )
    before = checkpoint_content_digest(tmp_path, chunk_mb=1)
    (tmp_path / "one.safetensors").write_bytes(b"same-size-payload-b")
    after = checkpoint_content_digest(tmp_path, chunk_mb=1)
    assert before != after


def test_attest_checkpoint_returns_split_revision_namespace(monkeypatch):
    identity = revision_identity_from_digests(
        checkpoint_digest=EXPECTED_CHECKPOINT_DIGEST,
        tokenizer_digest=EXPECTED_TOKENIZER_DIGEST,
        chat_template_digest=next(iter(APPROVED_CHAT_TEMPLATE_REVISIONS)),
    )
    monkeypatch.setattr(manifest, "attest_revision_identity", lambda *_a, **_k: identity)
    assert attest_checkpoint("/unused") == identity.namespace_sha256


def test_config_and_chat_template_are_authenticated(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    template = tmp_path / "chat_template.jinja"
    config.write_text('{"text_config":{"swiglu_limit":10.0}}')
    template.write_text("official template")
    monkeypatch.setattr(
        manifest,
        "EXPECTED_METADATA_SHA256",
        {
            "config.json": hashlib.sha256(config.read_bytes()).hexdigest(),
            "chat_template.jinja": hashlib.sha256(template.read_bytes()).hexdigest(),
        },
    )
    assert manifest._audit_official_metadata(tmp_path) == []

    config.write_text('{"text_config":{"swiglu_limit":9.0}}')
    assert manifest._audit_official_metadata(tmp_path) == [
        "official metadata digest mismatch: config.json"
    ]
    config.write_text('{"text_config":{"swiglu_limit":10.0}}')
    template.write_text("modified template")
    assert manifest._audit_official_metadata(tmp_path) == [
        "official metadata digest mismatch: chat_template.jinja"
    ]


def test_chat_template_metadata_accepts_only_explicit_approved_revisions(
    tmp_path, monkeypatch
):
    template = tmp_path / "chat_template.jinja"
    template.write_text("candidate")
    candidate = hashlib.sha256(template.read_bytes()).hexdigest()
    monkeypatch.setattr(
        manifest,
        "EXPECTED_METADATA_SHA256",
        {"chat_template.jinja": ("baseline-digest", candidate)},
    )
    assert manifest._audit_official_metadata(tmp_path) == []

    template.write_text("unapproved")
    assert manifest._audit_official_metadata(tmp_path) == [
        "official metadata digest mismatch: chat_template.jinja"
    ]


def test_revision_identity_separates_checkpoint_tokenizer_and_template():
    baseline_template, candidate_template = APPROVED_CHAT_TEMPLATE_REVISIONS
    baseline = revision_identity_from_digests(
        checkpoint_digest=EXPECTED_CHECKPOINT_DIGEST,
        tokenizer_digest=EXPECTED_TOKENIZER_DIGEST,
        chat_template_digest=baseline_template,
    )
    candidate = revision_identity_from_digests(
        checkpoint_digest=EXPECTED_CHECKPOINT_DIGEST,
        tokenizer_digest=EXPECTED_TOKENIZER_DIGEST,
        chat_template_digest=candidate_template,
    )

    assert baseline.checkpoint_digest == candidate.checkpoint_digest
    assert baseline.tokenizer_digest == candidate.tokenizer_digest
    assert baseline.chat_template_revision != candidate.chat_template_revision
    assert baseline.chat_template_digest != candidate.chat_template_digest
    assert baseline.namespace_sha256 != candidate.namespace_sha256


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("checkpoint_digest", "checkpoint weight digest mismatch"),
        ("tokenizer_digest", "tokenizer digest mismatch"),
        ("chat_template_digest", "unapproved official chat template digest"),
    ],
)
def test_revision_identity_rejects_each_unapproved_component(field, message):
    values = {
        "checkpoint_digest": EXPECTED_CHECKPOINT_DIGEST,
        "tokenizer_digest": EXPECTED_TOKENIZER_DIGEST,
        "chat_template_digest": next(iter(APPROVED_CHAT_TEMPLATE_REVISIONS)),
    }
    values[field] = "0" * 64
    with pytest.raises(ManifestError, match=message):
        revision_identity_from_digests(**values)


def test_fp8_scale_shape_and_dtype_are_audited():
    tensors = {
        "layer.weight": manifest.TensorHeader(
            "one.safetensors", "F8_E4M3", (129, 257), 0, 129 * 257
        ),
        "layer.weight_scale_inv": manifest.TensorHeader(
            "one.safetensors", "F16", (2, 2), 129 * 257, 129 * 257 + 8
        ),
    }
    failures = manifest._validate_fp8_pairs(tensors)
    assert failures == [
        "invalid FP8 scale layer.weight_scale_inv: F16(2, 2), expected F32(2, 3)"
    ]
