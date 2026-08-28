"""Strict, read-only audit of the official GLM-5.3-Flash checkpoint."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

EXPECTED_KDA = tuple(i for i in range(45) if i % 4 != 3)
EXPECTED_DSA = tuple(range(3, 45, 4))
EXPECTED_SHARDS = 62
EXPECTED_TENSORS = 76_108
EXPECTED_DECLARED_BYTES = 328_326_771_576
# Canonical tensor metadata digest of the public checkpoint. Filled after the
# audit algorithm itself is validated against that checkpoint.
EXPECTED_LAYOUT_DIGEST = "c21cc9b55c8b977434e5932682313a3b84ac87e31c61a12e2768dd57e2954a72"
FP8_FORMAT = "e4m3"
FP8_BLOCK = (128, 128)

_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "U16": 2, "I16": 2, "F16": 2,
    "BF16": 2, "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8,
    "F64": 8,
}


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class TensorHeader:
    shard: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class CheckpointReport:
    path: str
    fingerprint: str
    layout_digest: str
    source_format: str
    shard_count: int
    tensor_count: int
    declared_bytes: int
    fp8_tensor_count: int
    layers: int
    kda_layers: tuple[int, ...]
    dsa_layers: tuple[int, ...]
    experts: int
    experts_per_token: int
    context_length: int
    quantization: dict[str, Any] | None
    server_ready: bool
    audit_failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_digest(*objects: Any) -> str:
    h = hashlib.sha256()
    for obj in objects:
        h.update(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())
        h.update(b"\0")
    return h.hexdigest()


def _read_exact(handle: BinaryIO, size: int, label: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ManifestError(f"truncated {label}: expected {size} bytes, got {len(data)}")
    return data


def _read_shard_header(path: Path) -> tuple[dict[str, Any], int, int]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", _read_exact(handle, 8, str(path)))[0]
        if header_len <= 0 or header_len > file_size - 8:
            raise ManifestError(f"invalid safetensors header length in {path.name}: {header_len}")
        raw = _read_exact(handle, header_len, f"{path.name} header")
    try:
        header = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"invalid safetensors JSON header in {path.name}: {exc}") from exc
    if not isinstance(header, dict):
        raise ManifestError(f"safetensors header is not an object: {path.name}")
    return header, header_len, file_size


def _audit_headers(
    root: Path, shards: list[str], weight_map: dict[str, str]
) -> tuple[dict[str, TensorHeader], str, list[str]]:
    tensors: dict[str, TensorHeader] = {}
    failures: list[str] = []
    layout = hashlib.sha256()

    for shard in shards:
        header, header_len, file_size = _read_shard_header(root / shard)
        entries = {key: value for key, value in header.items() if key != "__metadata__"}
        shard_end = 0
        ranges: list[tuple[int, int, str]] = []
        layout.update(shard.encode())
        layout.update(struct.pack("<Q", file_size))
        for key in sorted(entries):
            meta = entries[key]
            try:
                dtype = str(meta["dtype"])
                shape = tuple(int(dim) for dim in meta["shape"])
                start, end = (int(value) for value in meta["data_offsets"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ManifestError(f"malformed tensor header {key!r} in {shard}") from exc
            if dtype not in _DTYPE_BYTES:
                failures.append(f"unsupported dtype {dtype} for {key}")
                continue
            expected_bytes = math.prod(shape) * _DTYPE_BYTES[dtype]
            if start < 0 or end < start or end - start != expected_bytes:
                failures.append(f"invalid data range for {key}")
            if key in tensors:
                failures.append(f"duplicate tensor header {key}")
            tensors[key] = TensorHeader(shard, dtype, shape, start, end)
            ranges.append((start, end, key))
            shard_end = max(shard_end, end)
            layout.update(
                json.dumps([key, dtype, shape, start, end], separators=(",", ":")).encode()
            )
            layout.update(b"\0")
        ranges.sort()
        for previous, current in zip(ranges, ranges[1:]):
            if previous[1] > current[0]:
                failures.append(f"overlapping tensors {previous[2]} and {current[2]}")
        if 8 + header_len + shard_end != file_size:
            failures.append(f"shard size/header mismatch: {shard}")

    mapped = set(weight_map)
    present = set(tensors)
    if mapped != present:
        missing = sorted(mapped - present)
        extra = sorted(present - mapped)
        if missing:
            failures.append(f"{len(missing)} indexed tensor(s) absent from headers: {missing[0]}")
        if extra:
            failures.append(f"{len(extra)} unindexed tensor(s) in headers: {extra[0]}")
    for key, shard in weight_map.items():
        tensor = tensors.get(key)
        if tensor is not None and tensor.shard != shard:
            failures.append(f"weight_map shard mismatch for {key}")
    return tensors, layout.hexdigest(), failures


def _validate_fp8_pairs(tensors: dict[str, TensorHeader]) -> list[str]:
    failures: list[str] = []
    for key, tensor in tensors.items():
        if tensor.dtype != "F8_E4M3":
            continue
        if not key.endswith(".weight") or len(tensor.shape) != 2:
            failures.append(f"unexpected U8 tensor without 2-D weight semantics: {key}")
            continue
        scale_key = key + "_scale_inv"
        scale = tensors.get(scale_key)
        if scale is None:
            failures.append(f"missing FP8 scale tensor: {scale_key}")
            continue
        expected_shape = (
            math.ceil(tensor.shape[0] / FP8_BLOCK[0]),
            math.ceil(tensor.shape[1] / FP8_BLOCK[1]),
        )
        if scale.dtype != "F32" or scale.shape != expected_shape:
            failures.append(
                f"invalid FP8 scale {scale_key}: {scale.dtype}{scale.shape}, "
                f"expected F32{expected_shape}"
            )
    return failures


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
    failures: list[str] = []
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

    weight_map = index.get("weight_map") or {}
    shards = sorted(set(weight_map.values()))
    missing_shards = [name for name in shards if not (root / name).is_file()]
    if missing_shards:
        raise ManifestError(f"missing {len(missing_shards)} shard(s), first: {missing_shards[0]}")
    tensors, layout_digest, header_failures = _audit_headers(root, shards, weight_map)
    failures.extend(header_failures)
    failures.extend(_validate_fp8_pairs(tensors))

    mlx_quant = config.get("quantization")
    hf_quant = config.get("quantization_config") or {}
    raw_fp8 = hf_quant.get("quant_method") == "fp8"
    if hf_quant.get("fmt") != FP8_FORMAT:
        failures.append(f"quantization fmt != {FP8_FORMAT}")
    if tuple(hf_quant.get("weight_block_size") or ()) != FP8_BLOCK:
        failures.append(f"weight_block_size != {list(FP8_BLOCK)}")
    source_format = "mlx-affine" if mlx_quant else ("hf-fp8" if raw_fp8 else "dense")
    quant = mlx_quant or hf_quant or None
    declared_bytes = int((index.get("metadata") or {}).get("total_size", 0))
    tensor_bytes = sum(tensor.nbytes for tensor in tensors.values())
    if declared_bytes != tensor_bytes:
        failures.append(f"declared bytes {declared_bytes} != header bytes {tensor_bytes}")
    if len(shards) != EXPECTED_SHARDS:
        failures.append(f"shards={len(shards)} (expected {EXPECTED_SHARDS})")
    if len(tensors) != EXPECTED_TENSORS:
        failures.append(f"tensors={len(tensors)} (expected {EXPECTED_TENSORS})")
    if declared_bytes != EXPECTED_DECLARED_BYTES:
        failures.append(f"declared bytes={declared_bytes} (expected {EXPECTED_DECLARED_BYTES})")
    if layout_digest != EXPECTED_LAYOUT_DIGEST:
        failures.append("official checkpoint layout digest mismatch")

    digest_config = {
        key: config.get(key)
        for key in ("model_type", "text_config", "vision_config", "quantization", "quantization_config")
    }
    fingerprint = _canonical_digest(digest_config, layout_digest)
    server_ready = raw_fp8 and not failures
    report = CheckpointReport(
        path=str(root), fingerprint=fingerprint, layout_digest=layout_digest,
        source_format=source_format, shard_count=len(shards), tensor_count=len(tensors),
        declared_bytes=declared_bytes,
        fp8_tensor_count=sum(tensor.dtype == "F8_E4M3" for tensor in tensors.values()),
        layers=layers, kda_layers=kda, dsa_layers=dsa,
        experts=int(text.get("n_routed_experts", -1)),
        experts_per_token=int(text.get("num_experts_per_tok", -1)),
        context_length=int(text.get("max_position_embeddings", -1)), quantization=quant,
        server_ready=server_ready, audit_failures=tuple(failures),
    )
    if require_server_ready and not server_ready:
        detail = "; ".join(failures[:8]) or f"source format is {source_format}"
        raise ManifestError(f"checkpoint failed strict official-FP8 audit: {detail}")
    return report


def checkpoint_content_digest(path: str | Path, *, chunk_mb: int = 16) -> str:
    """Hash config/tokenizer/index and every shard byte for disk-cache identity."""
    root = Path(path).expanduser().resolve()
    report = inspect_checkpoint(root, require_server_ready=True)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    metadata_files = (
        "config.json", "generation_config.json", "model.safetensors.index.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "chat_template.jinja",
    )
    digest = hashlib.sha256()
    digest.update(b"glm53-checkpoint-content-v1\0")
    digest.update(report.layout_digest.encode())
    for name in metadata_files + tuple(shards):
        file = root / name
        if not file.is_file():
            continue
        digest.update(name.encode())
        digest.update(struct.pack("<Q", file.stat().st_size))
        with file.open("rb", buffering=0) as handle:
            while chunk := handle.read(chunk_mb * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()
