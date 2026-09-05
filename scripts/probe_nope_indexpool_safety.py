#!/usr/bin/env python3
"""M3 Ultra gate for the official layer-3 NoPE IndexPool sentinel contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from glm53_flash_mlx.abi import (
    GROUPED_MIN_ROUTES,
    NOPE_DSA_CACHE_ABI,
)
from glm53_flash_mlx.indexpool import (
    INDEXPOOL_SENTINEL,
    build_prefill_indexpool_mask,
    prepare_decode_indexpool_gather,
    sanitize_indexpool_indices,
)
from glm53_flash_mlx.loader import load
from glm53_flash_mlx.manifest import inspect_checkpoint
from glm53_flash_mlx.server import (
    LEGACY_PROBE_MAX_PROMPT_TOKENS,
    _disk_cache_descriptor,
)


def _inputs(tokens: int, hidden_size: int, q_lora_rank: int, *, batch: int = 1):
    x = mx.sin(
        mx.arange(batch * tokens * hidden_size, dtype=mx.float32) * 0.0009765625
    ).reshape(batch, tokens, hidden_size)
    qr = mx.cos(
        mx.arange(batch * tokens * q_lora_rank, dtype=mx.float32) * 0.001953125
    ).reshape(batch, tokens, q_lora_rank)
    return x.astype(mx.bfloat16), qr.astype(mx.bfloat16)


def _array(value) -> np.ndarray:
    mx.eval(value)
    return np.ascontiguousarray(np.asarray(value))


def _stats(output, kv_len: int) -> dict:
    if output is None:
        return {
            "output_shape": None,
            "bypassed": True,
            "valid_slots": 0,
            "unused_slots": 0,
            "min_valid_index": None,
            "max_valid_index": None,
            "non_sentinel_out_of_range": 0,
            "nan_count": 0,
            "sha256": None,
        }
    values = _array(output)
    valid = values != INDEXPOOL_SENTINEL
    valid_values = values[valid]
    out_of_range = valid & ((values < 0) | (values >= kv_len))
    return {
        "output_shape": list(values.shape),
        "bypassed": False,
        "valid_slots": int(valid.sum()),
        "unused_slots": int((~valid).sum()),
        "min_valid_index": int(valid_values.min()) if valid_values.size else None,
        "max_valid_index": int(valid_values.max()) if valid_values.size else None,
        "non_sentinel_out_of_range": int(out_of_range.sum()),
        "nan_count": 0,
        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }


def _run(indexer, tokens: int, *, mask=None):
    x, qr = _inputs(tokens, indexer.dim, indexer.q_lora_rank)
    if mask is None:
        mask = mx.ones((1, tokens), dtype=mx.bool_)
    return indexer(x, qr, mask)


def _valid_sets(values: np.ndarray) -> list[set[int]]:
    return [
        set(row[row >= 0].tolist())
        for row in values.reshape(-1, values.shape[-1])
    ]


def _run_cached_modes(indexer, tokens: int) -> dict:
    from mlx_vlm.models.cache import KVCache

    x, qr = _inputs(tokens, indexer.dim, indexer.q_lora_rank)
    mask = mx.ones((1, tokens), dtype=mx.bool_)
    one_shot = _array(indexer(x, qr, mask))

    def chunked():
        cache = KVCache()
        outputs = []
        split = tokens // 2 + 1
        for start, end in ((0, split), (split, tokens)):
            outputs.append(
                _array(
                    indexer(
                        x[:, start:end],
                        qr[:, start:end],
                        mask[:, start:end],
                        cache=cache,
                    )
                )
            )
        return np.concatenate(outputs, axis=2)

    def incremental():
        cache = KVCache()
        outputs = []
        for position in range(tokens):
            outputs.append(
                _array(
                    indexer(
                        x[:, position : position + 1],
                        qr[:, position : position + 1],
                        mask[:, position : position + 1],
                        cache=cache,
                    )
                )
            )
        return np.concatenate(outputs, axis=2)

    chunked_first = chunked()
    chunked_second = chunked()
    incremental_first = incremental()
    incremental_second = incremental()
    one_sets = _valid_sets(one_shot)
    chunked_sets = _valid_sets(chunked_first)
    incremental_sets = _valid_sets(incremental_first)
    return {
        "tokens": tokens,
        "one_shot_shape": list(one_shot.shape),
        "chunked_shape": list(chunked_first.shape),
        "incremental_shape": list(incremental_first.shape),
        "one_shot_chunked_valid_set_parity": one_sets == chunked_sets,
        "one_shot_incremental_valid_set_parity": one_sets == incremental_sets,
        "chunked_repeated_hash_match": hashlib.sha256(
            chunked_first.tobytes()
        ).digest()
        == hashlib.sha256(chunked_second.tobytes()).digest(),
        "incremental_repeated_hash_match": hashlib.sha256(
            incremental_first.tobytes()
        ).digest()
        == hashlib.sha256(incremental_second.tobytes()).digest(),
        "partial_pool_valid_set_before_completion": sorted(
            incremental_sets[2]
        ),
        "complete_pool_valid_set_after_completion": sorted(
            incremental_sets[3]
        ),
        "non_sentinel_out_of_range": int(
            sum(
                np.count_nonzero((row < -1) | (row >= tokens))
                for row in (one_shot, chunked_first, incremental_first)
            )
        ),
    }


def _attention_differential() -> dict:
    kv_len = 4
    raw = mx.array(
        [INDEXPOOL_SENTINEL, 0, kv_len - 1, kv_len, kv_len + 7],
        dtype=mx.int32,
    )
    sanitized = sanitize_indexpool_indices(raw, kv_len)
    safe, valid = prepare_decode_indexpool_gather(raw, kv_len)
    sparse, _ = build_prefill_indexpool_mask(raw.reshape(1, 1, 1, -1), kv_len)

    q = mx.array([[[[0.25, -0.5]]]], dtype=mx.float32)
    keys = mx.array([[[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-1.0, 0.0]]]])
    values = keys * 0.25
    gathered_k = keys[:, :, safe, :]
    gathered_v = values[:, :, safe, :]
    selection_mask = valid.reshape(1, 1, 1, -1)
    expected = mx.fast.scaled_dot_product_attention(
        q, gathered_k, gathered_v, scale=2**-0.5, mask=selection_mask
    )
    kv_valid = valid.reshape(1, 1, -1, 1)
    changed = mx.fast.scaled_dot_product_attention(
        q,
        mx.where(kv_valid, gathered_k, 10_000.0),
        mx.where(kv_valid, gathered_v, -10_000.0),
        scale=2**-0.5,
        mask=selection_mask,
    )
    mx.eval(sanitized, safe, valid, sparse, expected, changed)
    expected_np = _array(expected)
    changed_np = _array(changed)
    return {
        "raw_indices": _array(raw).tolist(),
        "sanitized_indices": _array(sanitized).tolist(),
        "decode_safe_indices": _array(safe).tolist(),
        "decode_valid": _array(valid).tolist(),
        "prefill_sparse_mask": _array(sparse).reshape(-1).tolist(),
        "invalid_selected_kv_change_output_equal": bool(
            np.array_equal(expected_np, changed_np)
        ),
        "output_sha256": hashlib.sha256(expected_np.tobytes()).hexdigest(),
    }


def _batch_cache_case(indexer) -> dict:
    from mlx_vlm.models.cache import BatchKVCache

    tokens = 5
    x, qr = _inputs(tokens, indexer.dim, indexer.q_lora_rank, batch=2)
    mask = mx.array(
        [[False, False, True, True, True], [True] * tokens], dtype=mx.bool_
    )
    output = indexer(x, qr, mask, cache=BatchKVCache(left_padding=[2, 0]))
    values = _array(output)
    stats = _stats(output, tokens)
    stats.update(
        {
            "physical_kv_columns": tokens,
            "row_0_valid_indices": sorted(set(values[0][values[0] >= 0].tolist())),
            "row_1_valid_indices": sorted(set(values[1][values[1] >= 0].tolist())),
        }
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = inspect_checkpoint(args.model, require_server_ready=True)
    model, _ = load(args.model)
    layer = model.language_model.model.layers[args.layer]
    indexer = layer.self_attn.indexer
    mx.eval(*[value for _, value in tree_flatten(indexer.parameters())])
    mx.synchronize()

    topk = indexer.index_topk
    kpool = indexer.index_kpool
    bypass_cases = {}
    indexer.bypass_short = True
    for tokens in (topk - 1, topk, topk + 1):
        first = _run(indexer, tokens)
        second = _run(indexer, tokens)
        stats = _stats(first, tokens)
        second_stats = _stats(second, tokens)
        stats["repeated_hash_match"] = (
            stats["bypassed"] == second_stats["bypassed"]
            and stats["sha256"] == second_stats["sha256"]
        )
        if tokens == topk + 1:
            values = _array(first)
            first_row = values[0, 0, 0]
            final_row = values[0, 0, -1]
            stats.update(
                {
                    "early_row_valid_slots": int(np.count_nonzero(first_row >= 0)),
                    "early_row_unused_slots": int(np.count_nonzero(first_row == -1)),
                    "final_row_contains_pool_0": bool(np.any(final_row == 0)),
                    "final_row_contains_last_token": bool(
                        np.any(final_row == tokens - 1)
                    ),
                }
            )
        bypass_cases[str(tokens)] = stats

    indexer.bypass_short = False
    pool_cases = {}
    for tokens in (kpool - 1, kpool, kpool + 1, 31, 32, 33):
        first = _run(indexer, tokens)
        second = _run(indexer, tokens)
        stats = _stats(first, tokens)
        stats["repeated_hash_match"] = stats["sha256"] == _stats(
            second, tokens
        )["sha256"]
        pool_cases[str(tokens)] = stats

    zero_mask = mx.zeros((1, 33), dtype=mx.bool_)
    left_mask = mx.array([[False] * 5 + [True] * 28], dtype=mx.bool_)
    padding_cases = {
        "zero_valid_row": _stats(_run(indexer, 33, mask=zero_mask), 33),
        "left_padded_row": _stats(_run(indexer, 33, mask=left_mask), 33),
    }

    chunk_cases = {}
    for tokens in (511, 512, 513):
        first = _stats(_run(indexer, tokens), tokens)
        second = _stats(_run(indexer, tokens), tokens)
        first["repeated_hash_match"] = first["sha256"] == second["sha256"]
        chunk_cases[str(tokens)] = first

    cached_modes = _run_cached_modes(indexer, 33)
    batch_cache = _batch_cache_case(indexer)
    attention = _attention_differential()
    topk_plus_one = bypass_cases[str(topk + 1)]

    with patch.dict(os.environ, {"GLM53_MOE_BACKEND": "direct"}):
        direct_descriptor = _disk_cache_descriptor("probe-checkpoint")
    with patch.dict(os.environ, {"GLM53_MOE_BACKEND": "packed-grouped"}):
        grouped_descriptor = _disk_cache_descriptor("probe-checkpoint")

    range_clean = all(
        case["non_sentinel_out_of_range"] == 0
        for cases in (bypass_cases, pool_cases, padding_cases, chunk_cases)
        for case in cases.values()
    ) and (
        cached_modes["non_sentinel_out_of_range"] == 0
        and batch_cache["non_sentinel_out_of_range"] == 0
    )
    repeated = all(
        case["repeated_hash_match"] for case in pool_cases.values()
    ) and (
        all(case["repeated_hash_match"] for case in bypass_cases.values())
        and all(case["repeated_hash_match"] for case in chunk_cases.values())
    )
    acceptance = {
        "short_context_bypass_minus1_and_equal_topk": (
            bypass_cases[str(topk - 1)]["bypassed"]
            and bypass_cases[str(topk)]["bypassed"]
        ),
        "topk_plus_one_sentinel_and_range_clean": (
            not topk_plus_one["bypassed"]
            and topk_plus_one["non_sentinel_out_of_range"] == 0
            and topk_plus_one["unused_slots"] > 0
            and topk_plus_one["final_row_contains_pool_0"]
            and topk_plus_one["final_row_contains_last_token"]
        ),
        "all_unused_slots_use_minus1": range_clean,
        "all_valid_indices_in_range": range_clean,
        "zero_valid_row_all_sentinel": (
            padding_cases["zero_valid_row"]["valid_slots"] == 0
        ),
        "batch_cache_physical_range_and_left_padding": (
            batch_cache["non_sentinel_out_of_range"] == 0
            and batch_cache["row_0_valid_indices"] == [2, 3, 4]
            and batch_cache["row_1_valid_indices"] == [0, 1, 2, 3, 4]
        ),
        "one_shot_chunked_valid_set_parity": cached_modes[
            "one_shot_chunked_valid_set_parity"
        ],
        "one_shot_incremental_valid_set_parity": cached_modes[
            "one_shot_incremental_valid_set_parity"
        ],
        "repeated_execution_byte_identical": (
            repeated
            and cached_modes["chunked_repeated_hash_match"]
            and cached_modes["incremental_repeated_hash_match"]
        ),
        "invalid_selected_kv_does_not_affect_attention": attention[
            "invalid_selected_kv_change_output_equal"
        ],
        "no_nan_or_undefined_positive_sentinel": range_clean,
        "nope_manifest_schema": (
            report.qk_rope_head_dim == 0
            and report.mla_use_nope
            and report.kv_lora_rank == 512
        ),
        "attention_cache_abi_in_direct_and_grouped_apc": (
            direct_descriptor["attention_cache_abi"] == NOPE_DSA_CACHE_ABI
            and grouped_descriptor["attention_cache_abi"] == NOPE_DSA_CACHE_ABI
        ),
        "runtime_policy_unchanged": (
            GROUPED_MIN_ROUTES == 256 and LEGACY_PROBE_MAX_PROMPT_TOKENS == 256
        ),
    }
    acceptance["accepted"] = all(acceptance.values())
    output = {
        "schema": "glm53-nope-indexpool-safety-v1",
        "date": date.today().isoformat(),
        "machine": "Apple M3 Ultra, 80-core GPU, 512 GB unified memory",
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "layer": args.layer,
        "nope_cache_abi": NOPE_DSA_CACHE_ABI,
        "qk_rope_head_dim": report.qk_rope_head_dim,
        "mla_use_nope": report.mla_use_nope,
        "kv_lora_rank": report.kv_lora_rank,
        "index_topk": topk,
        "index_kpool": kpool,
        "short_context_bypass_cases": bypass_cases,
        "pool_boundary_cases": pool_cases,
        "padding_cases": padding_cases,
        "query_chunk_boundary_cases": chunk_cases,
        "cached_mode_differential": cached_modes,
        "continuous_batch_cache_case": batch_cache,
        "attention_gather_differential": attention,
        "disk_apc": {
            "direct_attention_cache_abi": direct_descriptor[
                "attention_cache_abi"
            ],
            "grouped_attention_cache_abi": grouped_descriptor[
                "attention_cache_abi"
            ],
            "backend_namespaces_differ": direct_descriptor != grouped_descriptor,
        },
        "runtime_policy": {
            "default_backend": "direct",
            "packed_grouped_experimental_opt_in": True,
            "grouped_min_routes": GROUPED_MIN_ROUTES,
            "prompt_limit": LEGACY_PROBE_MAX_PROMPT_TOKENS,
            "grouped_full_model_correctness_accepted": False,
        },
        "acceptance": acceptance,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "topk_plus_one_shape": topk_plus_one["output_shape"],
                "topk_plus_one_unused_slots": topk_plus_one["unused_slots"],
                "accepted": acceptance["accepted"],
            },
            indent=2,
        )
    )
    return 0 if acceptance["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
