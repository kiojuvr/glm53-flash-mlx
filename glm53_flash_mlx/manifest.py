"""Read-only checkpoint validation; does not materialize tensor payloads."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXPECTED_KDA = tuple(i for i in range(45) if i % 4 != 3)
EXPECTED_DSA = tuple(range(3, 45, 4))


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class CheckpointReport:
    path: str
    fingerprint: str
    source_format: str
    shard_count: int
    tensor_count: int
    declared_bytes: int
    layers: int
    kda_layers: tuple[int, ...]
    dsa_layers: tuple[int, ...]
    experts: int
    experts_per_token: int
    context_length: int
    quantization: dict[str, Any] | None
    server_ready: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_digest(*objects: Any) -> str:
    h = hashlib.sha256()
    for obj in objects:
        h.update(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())
        h.update(b"\0")
    return h.hexdigest()


def inspect_checkpoint(path: str | Path, *, require_server_ready: bool = False) -> CheckpointReport:
    root = Path(path).expanduser().resolve()
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    if not config_path.is_file():
        raise ManifestError(f"config.json not found: {config_path}")
    if not index_path.is_file():
        raise ManifestError(f"model.safetensors.index.json not found: {index_path}")

    config = json.loads(config_path.read_text())
    index = json.loads(index_path.read_text())
    text = config.get("text_config") or {}
    if config.get("model_type") != "glm5_next":
        raise ManifestError(f"expected model_type=glm5_next, got {config.get('model_type')!r}")

    layers = int(text.get("num_hidden_layers", -1))
    layer_types = text.get("layer_types") or []
    kda = tuple(i for i, kind in enumerate(layer_types) if kind == "linear_attention")
    dsa = tuple(i for i, kind in enumerate(layer_types) if kind == "deepseek_sparse_attention")
    failures = []
    if layers != 45 or len(layer_types) != 45:
        failures.append(f"layers={layers}/{len(layer_types)} (expected 45)")
    if kda != EXPECTED_KDA:
        failures.append("KDA layer pattern mismatch")
    if dsa != EXPECTED_DSA:
        failures.append("DSA layer pattern mismatch")
    if int(text.get("n_routed_experts", -1)) != 288:
        failures.append("n_routed_experts != 288")
    if int(text.get("num_experts_per_tok", -1)) != 8:
        failures.append("num_experts_per_tok != 8")
    if int(text.get("hidden_size", -1)) != 4096:
        failures.append("hidden_size != 4096")
    if failures:
        raise ManifestError("incompatible GLM-5.3 checkpoint: " + "; ".join(failures))

    weight_map = index.get("weight_map") or {}
    shards = sorted(set(weight_map.values()))
    missing = [name for name in shards if not (root / name).is_file()]
    if missing:
        raise ManifestError(f"missing {len(missing)} shard(s), first: {missing[0]}")

    mlx_quant = config.get("quantization")
    hf_quant = config.get("quantization_config") or {}
    raw_fp8 = hf_quant.get("quant_method") == "fp8"
    source_format = "mlx-affine" if mlx_quant else ("hf-fp8" if raw_fp8 else "dense")
    quant = mlx_quant or hf_quant or None
    covered_layers = set()
    for key in weight_map:
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", key)
        if match:
            covered_layers.add(int(match.group(1)))
    target_complete = set(range(45)).issubset(covered_layers)
    runtime_meta = config.get("glm53_runtime") or {}
    server_ready = (
        raw_fp8
        and target_complete
        and runtime_meta.get("complete_model", True) is not False
    )
    if require_server_ready and not server_ready:
        if not raw_fp8:
            raise ManifestError(
                f"direct runtime requires the official HF FP8 checkpoint, got {source_format}"
            )
        missing_layers = sorted(set(range(45)) - covered_layers)
        raise ManifestError(
            "checkpoint is not a complete official GLM-5.3 FP8 target; missing layer(s): "
            + ", ".join(map(str, missing_layers[:12]))
        )

    declared_bytes = int((index.get("metadata") or {}).get("total_size", 0))
    digest_config = {k: config.get(k) for k in (
        "model_type", "text_config", "vision_config", "quantization", "quantization_config"
    )}
    fingerprint = _canonical_digest(digest_config, weight_map)
    return CheckpointReport(
        path=str(root), fingerprint=fingerprint, source_format=source_format,
        shard_count=len(shards), tensor_count=len(weight_map),
        declared_bytes=declared_bytes, layers=layers, kda_layers=kda, dsa_layers=dsa,
        experts=int(text["n_routed_experts"]),
        experts_per_token=int(text["num_experts_per_tok"]),
        context_length=int(text["max_position_embeddings"]), quantization=quant,
        server_ready=server_ready,
    )
