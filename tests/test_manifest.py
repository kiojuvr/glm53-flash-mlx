import hashlib
import json
import struct
from pathlib import Path

import pytest

from glm53_flash_mlx import manifest
from glm53_flash_mlx.manifest import (
    ManifestError,
    attest_checkpoint,
    checkpoint_content_digest,
    inspect_checkpoint,
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
    monkeypatch.setattr(manifest, "EXPECTED_CONTENT_DIGEST", before)
    assert attest_checkpoint(tmp_path, chunk_mb=1) == before
    (tmp_path / "one.safetensors").write_bytes(b"same-size-payload-b")
    after = checkpoint_content_digest(tmp_path, chunk_mb=1)
    assert before != after
    with pytest.raises(ManifestError, match="content digest mismatch"):
        attest_checkpoint(tmp_path, chunk_mb=1)


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
