import json
from pathlib import Path

import pytest

from glm53_flash_mlx.manifest import ManifestError, inspect_checkpoint


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
    (tmp_path / "one.safetensors").write_bytes(b"fixture")
    weight_map = {
        f"language_model.model.layers.{i}.input_layernorm.weight": "one.safetensors"
        for i in range(45)
    }
    weight_map["language_model.model.embed_tokens.weight"] = "one.safetensors"
    index = {
        "metadata": {"total_size": 7},
        "weight_map": weight_map,
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))


def test_inspect_official_fp8_checkpoint(tmp_path):
    make_checkpoint(tmp_path, quantized=False)
    report = inspect_checkpoint(tmp_path, require_server_ready=True)
    assert report.server_ready
    assert report.source_format == "hf-fp8"
    assert report.kda_layers == tuple(i for i in range(45) if i % 4 != 3)
    assert report.dsa_layers == tuple(range(3, 45, 4))


def test_converted_checkpoint_is_rejected(tmp_path):
    make_checkpoint(tmp_path, quantized=True)
    assert inspect_checkpoint(tmp_path).source_format == "mlx-affine"
    with pytest.raises(ManifestError, match="requires the official HF FP8"):
        inspect_checkpoint(tmp_path, require_server_ready=True)


def test_missing_shard_is_rejected(tmp_path):
    make_checkpoint(tmp_path)
    (tmp_path / "one.safetensors").unlink()
    with pytest.raises(ManifestError, match="missing 1 shard"):
        inspect_checkpoint(tmp_path)
