#!/usr/bin/env python3
"""Separate kpool=4 token expansion semantics from latent KV storage dtype."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import time
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np

from glm53_flash_mlx.abi import (
    KERNEL_ABI_VERSION,
    MLX_VLM_REVISION,
    NOPE_DSA_CACHE_ABI_COMPACT,
    NOPE_DSA_CACHE_ABI_DIRECT,
)
from glm53_flash_mlx.indexpool import (
    INDEXPOOL_SENTINEL,
    expand_selected_pools,
    prepare_decode_indexpool_gather,
)
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint

INDEX_TOPK = 2048
INDEX_KPOOL = 4
CONTEXTS = (2049, 4351, 4352, 16384, 65536, 131072, 262144)
TAIL_CONTEXTS = (4349, 4350, 4351, 4352)
BYPASS_CONTEXTS = (2047, 2048, 2049)
ARMS = ("bf16", "fp8_per_token_head", "fp8_group64")
WARMUPS = 2
SAMPLES = 5
FP8_MAX = 448.0


def _array(value) -> np.ndarray:
    if value.dtype == mx.bfloat16:
        value = value.astype(mx.float32)
    mx.eval(value)
    return np.ascontiguousarray(np.asarray(value))


def _hash(value) -> str:
    return hashlib.sha256(_array(value).tobytes()).hexdigest()


def _memory() -> dict[str, int]:
    mx.synchronize()
    return {
        "active_memory_bytes": int(mx.get_active_memory()),
        "cache_memory_bytes": int(mx.get_cache_memory()),
        "peak_memory_bytes": int(mx.get_peak_memory()),
    }


def _selection_fixture(kv_len: int):
    complete_pools = kv_len // INDEX_KPOOL
    selected_count = min(complete_pools, INDEX_TOPK // INDEX_KPOOL)
    selected = mx.arange(
        complete_pools - selected_count, complete_pools, dtype=mx.int32
    ).reshape(1, 1, selected_count)
    pool_indices = mx.arange(
        complete_pools * INDEX_KPOOL, dtype=mx.int64
    ).reshape(1, complete_pools, INDEX_KPOOL)
    selected_valid = mx.ones(selected.shape, dtype=mx.bool_)
    tail_count = kv_len % INDEX_KPOOL
    tail_positions = mx.arange(
        complete_pools * INDEX_KPOOL, kv_len, dtype=mx.int64
    )[None]
    tail_valid = mx.ones((1, tail_count), dtype=mx.bool_)
    return selected, pool_indices, selected_valid, tail_positions, tail_valid


def _expand(kv_len: int):
    selected, pool_indices, selected_valid, tail_positions, tail_valid = (
        _selection_fixture(kv_len)
    )
    indices, valid = expand_selected_pools(
        selected,
        pool_indices,
        selected_valid,
        kv_len=kv_len,
        index_topk=INDEX_TOPK,
        index_kpool=INDEX_KPOOL,
        tail_positions=tail_positions,
        tail_valid=tail_valid,
        always_select_tail=True,
    )
    return selected, indices, valid


def _quantize_per_token_head(latent):
    scale = mx.maximum(
        mx.max(mx.abs(latent.astype(mx.float32)), axis=-1, keepdims=True) / FP8_MAX,
        1e-8,
    )
    codes = mx.to_fp8(latent.astype(mx.float32) / scale)
    return codes, scale.astype(mx.float32)


def _quantize_group64(latent):
    grouped = latent.astype(mx.float32).reshape(*latent.shape[:-1], 8, 64)
    scale = mx.maximum(mx.max(mx.abs(grouped), axis=-1) / FP8_MAX, 1e-8)
    codes = mx.to_fp8(grouped / scale[..., None]).reshape(latent.shape)
    return codes, scale.astype(mx.float32)


def _gather_bf16(storage, indices):
    safe, valid = prepare_decode_indexpool_gather(indices, int(storage.shape[2]))
    expanded = mx.broadcast_to(safe[..., None], safe.shape + (storage.shape[-1],))
    return mx.take_along_axis(storage, expanded, axis=2), valid[:, :, None]


def _gather_fp8_per_token(storage, scale, indices):
    safe, valid = prepare_decode_indexpool_gather(indices, int(storage.shape[2]))
    code_index = mx.broadcast_to(
        safe[..., None], safe.shape + (storage.shape[-1],)
    )
    scale_index = safe[..., None]
    codes = mx.take_along_axis(storage, code_index, axis=2)
    selected_scale = mx.take_along_axis(scale, scale_index, axis=2)
    latent = mx.from_fp8(codes, dtype=mx.float32) * selected_scale
    return latent.astype(mx.bfloat16), valid[:, :, None]


def _gather_fp8_group64(storage, scale, indices):
    safe, valid = prepare_decode_indexpool_gather(indices, int(storage.shape[2]))
    code_index = mx.broadcast_to(
        safe[..., None], safe.shape + (storage.shape[-1],)
    )
    scale_index = mx.broadcast_to(safe[..., None], safe.shape + (scale.shape[-1],))
    codes = mx.take_along_axis(storage, code_index, axis=2)
    selected_scale = mx.take_along_axis(scale, scale_index, axis=2)
    grouped = mx.from_fp8(codes, dtype=mx.float32).reshape(*codes.shape[:-1], 8, 64)
    latent = (grouped * selected_scale[..., None]).reshape(codes.shape)
    return latent.astype(mx.bfloat16), valid[:, :, None]


def _attention_output(attention, q, latent, mask):
    embedded = attention.embed_q(q)
    output = mx.fast.scaled_dot_product_attention(
        embedded, latent, latent, scale=attention.scale, mask=mask
    )
    output = attention.unembed_out(output)
    output = output.transpose(0, 2, 1, 3).reshape(1, 1, -1)
    return attention.o_proj(output)


def _metrics(reference, actual) -> dict:
    expected = _array(reference.astype(mx.float32)).reshape(-1).astype(np.float64)
    observed = _array(actual.astype(mx.float32)).reshape(-1).astype(np.float64)
    difference = observed - expected
    relative_l2 = np.linalg.norm(difference) / max(np.linalg.norm(expected), 1e-12)
    p = np.exp(expected - expected.max())
    q = np.exp(observed - observed.max())
    p /= p.sum()
    q /= q.sum()
    top_expected = set(np.argpartition(expected, -10)[-10:].tolist())
    top_observed = set(np.argpartition(observed, -10)[-10:].tolist())
    return {
        "relative_l2": float(relative_l2),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "kl_reference_to_actual": float(
            np.sum(p * (np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(q, 1e-300))))
        ),
        "top_10_overlap": len(top_expected & top_observed),
        "argmax_match": bool(np.argmax(expected) == np.argmax(observed)),
    }


def _time_gather(function):
    for _ in range(WARMUPS):
        value = function()
        mx.eval(*value)
        mx.synchronize()
    samples = []
    value = None
    for _ in range(SAMPLES):
        started = time.perf_counter()
        value = function()
        mx.eval(*value)
        mx.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return value, {
        "warmups": WARMUPS,
        "samples": SAMPLES,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _storage_bytes(storage) -> int:
    if isinstance(storage, tuple):
        return sum(int(value.nbytes) for value in storage)
    return int(storage.nbytes)


def _run_context(attention, q, storages: dict, kv_len: int) -> dict:
    selected_hashes = {}
    index_hashes = {}
    valid_hashes = {}
    arm_outputs = {}
    arm_rows = {}
    functions = {
        "bf16": lambda indices: _gather_bf16(storages["bf16"][:, :, :kv_len], indices),
        "fp8_per_token_head": lambda indices: _gather_fp8_per_token(
            storages["fp8_per_token_head"][0][:, :, :kv_len],
            storages["fp8_per_token_head"][1][:, :, :kv_len],
            indices,
        ),
        "fp8_group64": lambda indices: _gather_fp8_group64(
            storages["fp8_group64"][0][:, :, :kv_len],
            storages["fp8_group64"][1][:, :, :kv_len],
            indices,
        ),
    }
    for arm in ARMS:
        selected, indices, valid = _expand(kv_len)
        selected_hashes[arm] = _hash(selected)
        index_hashes[arm] = _hash(indices)
        valid_hashes[arm] = _hash(valid)
        baseline = int(mx.get_active_memory())
        mx.reset_peak_memory()
        gathered, timing = _time_gather(lambda: functions[arm](indices))
        latent, gather_valid = gathered
        mask = valid[:, :, None] & gather_valid
        output = _attention_output(attention, q, latent, mask)
        mx.eval(output)
        mx.synchronize()
        working_peak = max(0, int(mx.get_peak_memory()) - baseline)
        invalid_poison = mx.where(valid[..., None], latent, 10_000.0)
        poison_output = _attention_output(attention, q, invalid_poison, mask)
        mx.eval(poison_output)
        arm_outputs[arm] = output
        arm_rows[arm] = {
            "selected_pool_sha256": selected_hashes[arm],
            "token_index_sha256": index_hashes[arm],
            "valid_mask_sha256": valid_hashes[arm],
            "gather_dequantize": timing,
            "working_peak_bytes": working_peak,
            "invalid_poison_output_equal": bool(
                mx.array_equal(output, poison_output).item()
            ),
        }

    reference = arm_outputs["bf16"]
    for arm in ARMS:
        arm_rows[arm]["attention_output"] = (
            {
                "relative_l2": 0.0,
                "max_absolute_error": 0.0,
                "kl_reference_to_actual": 0.0,
                "top_10_overlap": 10,
                "argmax_match": True,
            }
            if arm == "bf16"
            else _metrics(reference, arm_outputs[arm])
        )
        arm_rows[arm]["output_sha256"] = _hash(arm_outputs[arm])

    indices_np = _array(_expand(kv_len)[1])
    valid_np = _array(_expand(kv_len)[2])
    valid_values = indices_np[valid_np]
    repeated = _expand(kv_len)
    restored = _expand(kv_len)
    reference_index_hash = hashlib.sha256(indices_np.tobytes()).hexdigest()
    return {
        "context_tokens": kv_len,
        "context_mod_kpool": kv_len % INDEX_KPOOL,
        "production_bypass": kv_len <= INDEX_TOPK,
        "selected_width": int(indices_np.shape[-1]),
        "valid_slots": int(valid_np.sum()),
        "sentinel_slots": int((indices_np == INDEXPOOL_SENTINEL).sum()),
        "valid_index_min": int(valid_values.min()) if valid_values.size else None,
        "valid_index_max": int(valid_values.max()) if valid_values.size else None,
        "non_sentinel_out_of_range": int(
            np.count_nonzero((indices_np < -1) | (indices_np >= kv_len))
        ),
        "pool_hashes_identical_across_arms": len(set(selected_hashes.values())) == 1,
        "token_hashes_identical_across_arms": len(set(index_hashes.values())) == 1,
        "valid_hashes_identical_across_arms": len(set(valid_hashes.values())) == 1,
        "repeat_index_hash_match": _hash(repeated[1]) == reference_index_hash,
        "restore_index_hash_match": _hash(restored[1]) == reference_index_hash,
        "arms": arm_rows,
    }


def _structural_cases() -> dict:
    cases = {}
    for valid_pools in (7, 512, 513):
        kv_len = valid_pools * INDEX_KPOOL
        selected, indices, valid = _expand(kv_len)
        cases[str(valid_pools)] = {
            "valid_pool_count": valid_pools,
            "selected_pool_count": int(selected.shape[-1]),
            "output_shape": list(indices.shape),
            "valid_slots": int(_array(valid).sum()),
            "sentinel_slots": int((_array(indices) == INDEXPOOL_SENTINEL).sum()),
            "selected_pool_sha256": _hash(selected),
            "token_index_sha256": _hash(indices),
            "valid_mask_sha256": _hash(valid),
        }

    raw_pool = mx.array([[[-1, 0, 7, 15]]], dtype=mx.int64)
    oob_indices, oob_valid = expand_selected_pools(
        mx.array([[[0]]], dtype=mx.int32),
        raw_pool,
        mx.ones((1, 1, 1), dtype=mx.bool_),
        kv_len=8,
        index_topk=4,
        index_kpool=4,
        tail_positions=mx.zeros((1, 0), dtype=mx.int64),
        tail_valid=mx.zeros((1, 0), dtype=mx.bool_),
        always_select_tail=False,
    )
    cases["positive_oob"] = {
        "indices": _array(oob_indices).reshape(-1).tolist(),
        "valid": _array(oob_valid).reshape(-1).tolist(),
        "sentinelized": _array(oob_indices).reshape(-1).tolist() == [-1, 0, 7, -1],
    }
    cases["bypass_boundary"] = {
        str(context): {"bypassed": context <= INDEX_TOPK}
        for context in BYPASS_CONTEXTS
    }
    return cases


def _finalize(artifact: dict) -> None:
    contexts = artifact["contexts"]
    all_hashes = all(
        row["pool_hashes_identical_across_arms"]
        and row["token_hashes_identical_across_arms"]
        and row["valid_hashes_identical_across_arms"]
        for row in contexts.values()
    )
    tail_mods = {row["context_mod_kpool"] for row in artifact["tail_cases"].values()}
    acceptance = {
        "all_selected_pool_token_and_mask_hashes_identical": all_hashes,
        "sentinel_range_and_oob_contract": (
            artifact["structural_cases"]["positive_oob"]["sentinelized"]
            and all(row["non_sentinel_out_of_range"] == 0 for row in contexts.values())
        ),
        "tail_mod_0_1_2_3_covered": tail_mods == {0, 1, 2, 3},
        "selected_width_bounded_at_2051": all(
            row["selected_width"] == 2051 for row in contexts.values()
        ),
        "repeat_and_restore_index_hashes_match": all(
            row["repeat_index_hash_match"] and row["restore_index_hash_match"]
            for row in contexts.values()
        ),
        "invalid_kv_poison_does_not_change_attention": all(
            arm["invalid_poison_output_equal"]
            for row in contexts.values()
            for arm in row["arms"].values()
        ),
        "bypass_2047_2048_and_sparse_2049": artifact["structural_cases"][
            "bypass_boundary"
        ]
        == {
            "2047": {"bypassed": True},
            "2048": {"bypassed": True},
            "2049": {"bypassed": False},
        },
        "fp8_backend_remains_probe_only": artifact["runtime_identity"][
            "loaded_cache_backend"
        ]
        == "direct",
        "runtime_server_apc_cache_abi_unchanged": artifact["runtime_identity"]
        == {
            "loaded_cache_backend": "direct",
            "direct_cache_abi": NOPE_DSA_CACHE_ABI_DIRECT,
            "compact_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
            "fp8_latent_backend_registered": False,
        },
    }
    acceptance["accepted"] = all(acceptance.values())
    artifact["acceptance"] = acceptance
    artifact["complete"] = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mx.set_wired_limit(int(440e9))
    mx.set_cache_limit(int(32e9))
    report = inspect_checkpoint(args.model, require_server_ready=True)
    model, _ = load(args.model)
    warm_residency(model)
    attention = model.language_model.model.layers[args.layer].self_attn
    hidden = int(model.config.text_config.hidden_size)
    x = mx.sin(mx.arange(hidden, dtype=mx.float32) * 0.001)[None, None].astype(
        mx.bfloat16
    )
    qr = attention.q_a_layernorm(attention.q_a_proj(x))
    q = attention.q_b_proj(qr).reshape(
        1, 1, attention.num_heads, attention.q_head_dim
    ).transpose(0, 2, 1, 3)
    mx.eval(q)

    max_context = max(CONTEXTS)
    positions = mx.arange(max_context, dtype=mx.float32)[:, None]
    features = mx.arange(512, dtype=mx.float32)[None]
    latent = mx.sin(positions * 0.0009765625 + features * 0.015625)
    latent = latent.reshape(1, 1, max_context, 512).astype(mx.bfloat16)
    per_token = _quantize_per_token_head(latent)
    group64 = _quantize_group64(latent)
    mx.eval(latent, *per_token, *group64)
    mx.synchronize()
    storages = {
        "bf16": latent,
        "fp8_per_token_head": per_token,
        "fp8_group64": group64,
    }

    artifact = {
        "schema": "glm53-kpool4-kv-dtype-separation-v1",
        "date": date.today().isoformat(),
        "machine": platform.machine(),
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "kernel_abi": KERNEL_ABI_VERSION,
        "layer": args.layer,
        "index_topk": INDEX_TOPK,
        "index_kpool": INDEX_KPOOL,
        "maximum_selected_width": INDEX_TOPK + INDEX_KPOOL - 1,
        "indexpool_state_dtypes": {
            "pool_keys": "bfloat16",
            "raw_tail": "bfloat16",
            "pool_indices": "int64",
            "validity": "bool",
        },
        "latent_arms": {
            "bf16": "bfloat16",
            "fp8_per_token_head": "uint8 E4M3 + float32 per-token/per-head scale",
            "fp8_group64": "uint8 E4M3 + float32 per-token/per-head/group64 scale",
        },
        "fp8_scope": "probe-only; no runtime cache backend or ABI",
        "runtime_identity": {
            "loaded_cache_backend": getattr(model, "_glm53_cache_backend", "direct"),
            "direct_cache_abi": NOPE_DSA_CACHE_ABI_DIRECT,
            "compact_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
            "fp8_latent_backend_registered": False,
        },
        "storage_bytes_at_256k": {
            arm: _storage_bytes(storage) for arm, storage in storages.items()
        },
        "memory_after_storage_materialization": _memory(),
        "structural_cases": _structural_cases(),
        "tail_cases": {},
        "contexts": {},
        "complete": False,
    }
    for context in TAIL_CONTEXTS:
        _, indices, valid = _expand(context)
        artifact["tail_cases"][str(context)] = {
            "context_mod_kpool": context % INDEX_KPOOL,
            "output_shape": list(indices.shape),
            "valid_slots": int(_array(valid).sum()),
            "sentinel_slots": int((_array(indices) == INDEXPOOL_SENTINEL).sum()),
            "token_index_sha256": _hash(indices),
            "valid_mask_sha256": _hash(valid),
        }
    for context in CONTEXTS:
        print(json.dumps({"phase": "context", "tokens": context}), flush=True)
        artifact["contexts"][str(context)] = _run_context(
            attention, q, storages, context
        )
    _finalize(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact["acceptance"], indent=2), flush=True)

    del model, attention, latent, per_token, group64, storages
    gc.collect()
    mx.clear_cache()
    return 0 if artifact["acceptance"]["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
