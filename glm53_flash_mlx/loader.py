"""Zero-copy loader for the official GLM-5.3 block-FP8 checkpoint."""

from __future__ import annotations

import gc
import glob
import json
import logging
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .fp8 import BlockFP8Linear, DirectFP8MoE
from .grouped_fp8 import SortedGroupedFP8MoE
from .manifest import inspect_checkpoint
from .packed import PackedFP8ExpertBank
from .patch import apply_runtime_patch


def _make_config(raw: dict):
    from mlx_vlm.models.glm5_next import ModelConfig, TextConfig, VisionConfig

    config = ModelConfig.from_dict(raw)
    config.text_config = TextConfig.from_dict(raw["text_config"])
    config.vision_config = VisionConfig.from_dict(raw["vision_config"])
    return config


def _load_raw(path: Path) -> dict:
    weights = {}
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())["weight_map"]
        files = [path / name for name in sorted(set(index.values()))]
    else:
        files = [Path(name) for name in sorted(glob.glob(str(path / "*.safetensors")))]
    for file in files:
        weights.update(mx.load(str(file)))
    return weights


def _remap_language(weights: dict, config) -> dict:
    """Apply GLM naming/layout transforms without dequantizing FP8 tensors."""
    remapped = {}
    conv_parts: dict[str, dict[str, mx.array]] = {}
    fg_parts = ("A_log", "dt_bias", "f_a_proj.weight", "f_b_proj.weight")

    for key, value in weights.items():
        if key.startswith("model.visual.") or key.startswith("visual."):
            continue
        if key.startswith("model.language_model."):
            key = "language_model.model." + key[len("model.language_model.") :]
        elif key.startswith("lm_head."):
            key = "language_model." + key
        else:
            continue

        layer_match = re.search(r"\.layers\.(\d+)\.", key)
        if layer_match and int(layer_match.group(1)) >= config.text_config.num_hidden_layers:
            continue
        if ".mtp." in key:
            continue
        key = key.replace(".hc_attn_", ".attn_hc.").replace(
            ".hc_ffn_", ".ffn_hc."
        )

        fused = False
        for part in ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"):
            suffix = ".self_attn." + part
            if key.endswith(suffix):
                prefix = key[: -len(part)]
                conv_parts.setdefault(prefix, {})[part[0]] = value
                fused = True
                break
        if fused:
            continue
        for part in fg_parts:
            suffix = ".self_attn." + part
            if key.endswith(suffix):
                key = key[: -len(part)] + "forget_gate." + part
                break
        remapped[key] = value

    for prefix, parts in conv_parts.items():
        if set(parts) != {"q", "k", "v"}:
            raise ValueError(f"incomplete q/k/v conv bundle at {prefix}")
        value = mx.concatenate([parts["q"], parts["k"], parts["v"]], axis=0)
        if value.ndim == 3 and value.shape[-1] != 1:
            value = value.moveaxis(2, 1)
        remapped[prefix + "conv1d.weight"] = value

    # Absorbed NoPE MLA uses two views derived from the BF16 kv_b projection.
    tc = config.text_config
    for layer in range(tc.num_hidden_layers):
        prefix = f"language_model.model.layers.{layer}.self_attn"
        key = prefix + ".kv_b_proj.weight"
        if key not in remapped:
            continue
        value = remapped.pop(key)
        if key + "_scale_inv" in remapped:
            raise ValueError("FP8 kv_b_proj requires a dedicated absorbed FP8 kernel")
        head_width = tc.qk_nope_head_dim + tc.v_head_dim
        value = value.reshape(tc.num_attention_heads, head_width, -1)
        remapped[prefix + ".embed_q.weight"] = mx.contiguous(
            value[:, : tc.qk_nope_head_dim, :].swapaxes(-1, -2)
        )
        remapped[prefix + ".unembed_out.weight"] = mx.contiguous(
            value[:, tc.qk_nope_head_dim :, :]
        )
    return remapped


def _install_direct_modules(model, weights, config) -> None:
    from mlx_vlm.models.glm5_next import language as glm

    # Preserve the gate and shared expert modules/names, but keep all 288
    # routed experts separate instead of stacking them into another 283 GiB.
    for layer in model.language_model.model.layers:
        if not hasattr(layer.mlp, "gate"):
            continue
        old = layer.mlp
        layer.mlp = DirectFP8MoE(config.text_config, old.gate, old.shared_experts)
        layer.compile_ffn = False
    for layer in model.language_model.model.layers:
        if getattr(layer, "is_linear", False):
            layer.self_attn.fuse_in = False

    replacements = []
    for path, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if not isinstance(module, nn.Linear):
            continue
        if f"{path}.weight_scale_inv" in weights:
            replacements.append((path, BlockFP8Linear.from_linear(module)))
    if replacements:
        model.update_modules(tree_unflatten(replacements))


def install_packed_grouped_moe(model) -> dict:
    """Replace routed MoE layers one at a time without a model-sized duplicate."""
    logger = logging.getLogger(__name__)
    converted = []
    layers = model.language_model.model.layers
    for layer_id, layer in enumerate(layers):
        direct = layer.mlp
        if not isinstance(direct, DirectFP8MoE) or isinstance(
            direct, SortedGroupedFP8MoE
        ):
            continue
        active_before = mx.get_active_memory()
        bank = PackedFP8ExpertBank.pack(direct.experts)
        bank_tensors = [value for _, value in tree_flatten(bank.parameters())]
        mx.eval(*bank_tensors)
        mx.synchronize()
        active_materialized = mx.get_active_memory()
        layer.mlp = SortedGroupedFP8MoE(
            bank,
            direct.config,
            direct.gate,
            direct.shared_experts,
        )
        layer.compile_ffn = False
        old_expert_count = len(direct.experts)
        del bank_tensors, direct
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        active_after_clear = mx.get_active_memory()
        converted.append(
            {
                "layer": layer_id,
                "old_expert_count": old_expert_count,
                "bank_bytes": bank.nbytes,
                "active_before_bytes": active_before,
                "active_materialized_bytes": active_materialized,
                "active_after_clear_bytes": active_after_clear,
                "peak_bytes": mx.get_peak_memory(),
                "old_expert_modules_detached": not hasattr(layer.mlp, "experts"),
            }
        )
        logger.info(
            "packed grouped MoE layer=%d active=%.3f GB peak=%.3f GB",
            layer_id,
            active_after_clear / 1e9,
            mx.get_peak_memory() / 1e9,
        )

    remaining_direct = [
        layer_id
        for layer_id, layer in enumerate(layers)
        if isinstance(layer.mlp, DirectFP8MoE)
        and not isinstance(layer.mlp, SortedGroupedFP8MoE)
    ]
    report = {
        "converted_layers": [row["layer"] for row in converted],
        "converted_count": len(converted),
        "remaining_direct_layers": remaining_direct,
        "all_old_expert_modules_detached": all(
            row["old_expert_modules_detached"] for row in converted
        ),
        "layers": converted,
    }
    model._glm53_moe_backend = "packed-grouped"
    model._glm53_packed_grouped_report = report
    return report


def load_model(
    path: str | Path,
    *,
    strict: bool = True,
    experimental_packed_grouped_moe: bool = False,
):
    path = Path(path).expanduser().resolve()
    report = inspect_checkpoint(path, require_server_ready=True)
    if report.source_format != "hf-fp8":
        raise ValueError(f"direct runtime requires the official HF FP8 checkpoint, got {report.source_format}")
    apply_runtime_patch()
    from mlx_vlm.models.glm5_next import Model

    raw_config = json.loads((path / "config.json").read_text())
    config = _make_config(raw_config)
    model = Model(config)
    weights = _remap_language(_load_raw(path), config)
    _install_direct_modules(model, weights, config)
    model.vision_model = None
    weight_items = list(weights.items())
    model.load_weights(weight_items, strict=strict)
    weight_items.clear()
    weights.clear()
    del weight_items, weights
    gc.collect()
    if experimental_packed_grouped_moe:
        install_packed_grouped_moe(model)
    else:
        model._glm53_moe_backend = "direct"
    model.model_path = path
    model.eval()
    return model, raw_config


def load(path: str | Path, adapter_path=None, **kwargs):
    if adapter_path is not None:
        raise ValueError("LoRA adapters are not supported by the direct FP8 runtime")
    # mlx-vlm's server forwards this generic loader option.  This runtime uses
    # its pinned local implementation and never executes checkpoint code.
    kwargs.pop("trust_remote_code", None)
    model, config = load_model(
        path,
        strict=kwargs.pop("strict", True),
        experimental_packed_grouped_moe=kwargs.pop(
            "experimental_packed_grouped_moe", False
        ),
    )
    if kwargs:
        raise TypeError(f"unsupported loader options: {sorted(kwargs)}")
    # AutoProcessor also instantiates GLM's video processor and therefore
    # pulls PyTorch/torchvision into a text-only runtime.  The server only
    # needs the tokenizer, chat template and streaming detokenizer.
    from mlx_vlm.tokenizer_utils import load_tokenizer
    from mlx_vlm.utils import StoppingCriteria

    wrapped_tokenizer = load_tokenizer(Path(path))
    processor = wrapped_tokenizer._tokenizer
    processor.detokenizer = wrapped_tokenizer.detokenizer
    eos = config["text_config"].get("eos_token_id", processor.eos_token_id)
    processor.additional_eos_token_ids = []
    processor.stopping_criteria = StoppingCriteria(eos, processor)
    return model, processor


def warm_residency(model, *, batch_tensors: int = 128) -> int:
    """Materialize every canonical text tensor in unified memory.

    ``mx.load`` is lazy.  Without this pass, new router choices page in expert
    banks during interactive decode.  Evaluation preserves uint8 FP8 storage;
    it does not construct dense BF16 copies.
    """
    tensors = [value for _, value in tree_flatten(model.parameters())]
    total = sum(value.nbytes for value in tensors)
    logger = logging.getLogger(__name__)
    done = 0
    for start in range(0, len(tensors), batch_tensors):
        batch = tensors[start : start + batch_tensors]
        mx.eval(*batch)
        done += sum(value.nbytes for value in batch)
        batch_index = start // batch_tensors
        if batch_index % 16 == 15 or start + len(batch) == len(tensors):
            logger.info("FP8 residency %.1f/%.1f GiB", done / 2**30, total / 2**30)
    mx.synchronize()
    return total


def prefault_checkpoint(path: str | Path, *, chunk_mb: int = 64) -> int:
    """Sequentially fault canonical shards into the unified-memory page cache."""
    root = Path(path)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    total = 0
    chunk = chunk_mb * 1024 * 1024
    for name in sorted(set(index.values())):
        with open(root / name, "rb", buffering=0) as handle:
            while data := handle.read(chunk):
                total += len(data)
    return total
