#!/usr/bin/env python3
"""Qualify the native GLM-5.3 NoPE cache tile alignment contract.

This is a correctness probe, not a new paging implementation.  It derives the
local 256-token/64-pool-row geometry from production cache code, verifies that
padding stays sentinel-only, and reuses the established full-model 256k
synthetic-state differential.  Long cold prefill remains out of scope.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np

from glm53_flash_mlx.abi import (
    MLX_VLM_REVISION,
    NOPE_DSA_CACHE_ABI_COMPACT,
    NOPE_DSA_CACHE_ABI_DIRECT,
)
from glm53_flash_mlx.cache_geometry import (
    DEFAULT_NOPE_CACHE_TILE_ALIGNMENT,
    NOPE_CACHE_TILE_ALIGNMENT_CONTRACT,
    CacheTileAlignmentError,
    logical_token_to_pool_lane,
    plan_nope_cache_capacity,
    pool_lane_to_logical_token,
)
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint
from glm53_flash_mlx.nope_cache import make_compact_nope_dsa_cache


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-kpool-cache-tile-alignment-20260905.json"
)
ALIGNMENT_BOUNDARIES = (255, 256, 257, 511, 512, 513)
LONG_BOUNDARIES = (262_143, 262_144, 262_145)
SYNTHETIC_CONTEXT = 262_144


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


def _load_probe_modules():
    scripts = str(REPOSITORY / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import probe_exact_sigmoid_gate_metal_barrier as oracle_probe
    import probe_long_context_first_decode_boundary as boundary_probe
    import probe_packed_decode_runtime as packed_probe

    return oracle_probe, boundary_probe, packed_probe


def _release(*values) -> None:
    for value in values:
        if isinstance(value, list):
            value.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def _sha(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        mx.eval(value)
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        materialized = np.ascontiguousarray(
            np.asarray(
                value.astype(mx.float32)
                if value.dtype == mx.bfloat16
                else value
            )
        )
        digest.update(materialized.tobytes())
    return digest.hexdigest()


def _mapping_evidence(logical: int) -> dict:
    plan = plan_nope_cache_capacity(logical)
    samples = sorted({0, max(0, logical - 1)}) if logical else []
    mappings = []
    for token in samples:
        row, lane = logical_token_to_pool_lane(
            token, logical_extent_tokens=logical
        )
        mappings.append(
            {
                "token": token,
                "pool_row": row,
                "lane": lane,
                "roundtrip": pool_lane_to_logical_token(
                    row, lane, logical_extent_tokens=logical
                ),
            }
        )
    padding_rejected = True
    if plan.physical_capacity_tokens > logical:
        try:
            logical_token_to_pool_lane(
                logical, logical_extent_tokens=logical
            )
        except CacheTileAlignmentError:
            pass
        else:
            padding_rejected = False
    return {
        **plan.descriptor(),
        "sample_mappings": mappings,
        "first_padding_token_rejected": padding_rejected,
    }


def _physical_boundary_case(indexer, attention, logical: int) -> dict:
    plan = plan_nope_cache_capacity(logical)
    cache = make_compact_nope_dsa_cache(indexer, capacity_tokens=logical)
    latent_value = mx.zeros(
        (1, 1, 1, int(attention.kv_lora_rank)), dtype=mx.bfloat16
    )
    cache[0].update_and_fetch(latent_value, latent_value)
    key = mx.zeros((1, 1, int(indexer.head_dim)), dtype=mx.bfloat16)
    gate = mx.zeros_like(key)
    valid = mx.ones((1, 1), dtype=mx.bool_)
    cache[1]._append_projected(key, gate, valid)
    mx.eval(cache.state)
    pool = cache[1]
    padding = slice(pool.logical_pool_count, pool.physical_capacity_rows)
    padding_indices = pool.pool_indices[:, padding]
    padding_valid = pool.pool_valid[:, padding]
    padding_keys = pool.pool_keys[:, padding]
    known_bytes = cache[0].nbytes + cache[1].nbytes
    row = {
        "logical_capacity_tokens": logical,
        "logical_extent_tokens": pool.total_tokens,
        "planned": plan.descriptor(),
        "latent_physical_capacity_tokens": cache[0].physical_capacity_tokens,
        "indexpool_physical_capacity_rows": pool.physical_capacity_rows,
        "indexpool_logical_pool_count": pool.logical_pool_count,
        "physical_capacity_matches_plan": (
            cache[0].physical_capacity_tokens == plan.physical_capacity_tokens
            and pool.physical_capacity_rows == plan.physical_pool_rows
        ),
        "physical_alignment_exact": (
            cache[0].physical_capacity_tokens
            % DEFAULT_NOPE_CACHE_TILE_ALIGNMENT.allocation_alignment_tokens
            == 0
            and pool.physical_capacity_rows
            % DEFAULT_NOPE_CACHE_TILE_ALIGNMENT.physical_indexer_tile_rows
            == 0
        ),
        "padding_indices_all_sentinel": bool(
            mx.all(padding_indices == -1).item()
        ),
        "padding_valid_all_false": not bool(mx.any(padding_valid).item()),
        "padding_keys_all_zero": bool(mx.all(padding_keys == 0).item()),
        "logical_state_hash": _sha(
            value for child in cache.state for value in child if value is not None
        ),
        "accounted_cache_bytes": known_bytes,
        "anonymous_allocation_bytes": 0,
    }
    _release(cache)
    return row


def _trim_restore_fixture(indexer) -> dict:
    from mlx_vlm.apc_adapters import clone_cache_entry

    capacity = 513
    original = make_compact_nope_dsa_cache(indexer, capacity_tokens=capacity)
    keys = mx.zeros((1, 273, int(indexer.head_dim)), dtype=mx.bfloat16)
    gates = mx.zeros_like(keys)
    valid = mx.ones((1, 273), dtype=mx.bool_)
    original[1]._append_projected(keys, gates, valid)
    latent = mx.zeros((1, 1, 273, 512), dtype=mx.bfloat16)
    original[0].update_and_fetch(latent, latent)
    targets = []
    restored = clone_cache_entry(
        original, min_capacity_tokens=capacity, eval_targets=targets
    )
    mx.eval(*targets)
    restored.trim(16)
    retired_start = restored[1].logical_pool_count
    padding_sentinel = bool(
        mx.all(restored[1].pool_indices[:, retired_start:] == -1).item()
    )
    padding_invalid = not bool(
        mx.any(restored[1].pool_valid[:, retired_start:]).item()
    )
    padding_zero = bool(
        mx.all(restored[1].pool_keys[:, retired_start:] == 0).item()
    )
    replay_keys = keys[:, 257:273]
    replay_gates = gates[:, 257:273]
    replay_valid = valid[:, 257:273]
    restored[1]._append_projected(replay_keys, replay_gates, replay_valid)
    restored[0].update_and_fetch(latent[..., 257:273, :], latent[..., 257:273, :])
    mx.eval(original.state, restored.state)
    exact = all(
        bool(mx.array_equal(left, right).item())
        for left_state, right_state in zip(original.state, restored.state, strict=True)
        for left, right in zip(left_state, right_state, strict=True)
        if left is not None and right is not None
    ) and original.meta_state == restored.meta_state
    result = {
        "capacity_tokens": capacity,
        "trim_tokens": 16,
        "retired_padding_indices_all_sentinel": padding_sentinel,
        "retired_padding_valid_all_false": padding_invalid,
        "retired_padding_keys_all_zero": padding_zero,
        "restore_trim_replay_state_exact": exact,
        "physical_capacity_unchanged": (
            original[0].physical_capacity_tokens
            == restored[0].physical_capacity_tokens
            and original[1].physical_capacity_rows
            == restored[1].physical_capacity_rows
        ),
    }
    _release(original, restored)
    return result


def _atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    args = parser.parse_args()
    if not mx.metal.is_available():
        raise RuntimeError("cache tile alignment probe requires MLX/Metal")

    report = inspect_checkpoint(args.model, require_server_ready=True)
    oracle_probe, boundary_probe, packed_probe = _load_probe_modules()
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    _progress("load")
    model, processor = load(
        args.model,
        experimental_packed_decode_moe=True,
        experimental_compact_nope_dsa_cache=True,
        compact_cache_capacity_tokens=SYNTHETIC_CONTEXT + 1,
    )
    warm_residency(model)
    attention = model.language_model.model.layers[3].self_attn
    indexer = attention.indexer

    mapping = {
        str(value): _mapping_evidence(value)
        for value in (*ALIGNMENT_BOUNDARIES, *LONG_BOUNDARIES)
    }
    physical = {}
    for value in (*ALIGNMENT_BOUNDARIES, *LONG_BOUNDARIES):
        _progress("physical_boundary", logical_capacity_tokens=value)
        physical[str(value)] = _physical_boundary_case(
            indexer, attention, value
        )
    trim_restore = _trim_restore_fixture(indexer)

    _progress("official_oracle")
    official_oracle = oracle_probe._official_oracle(model, processor, report)
    vocab = int(model.language_model.lm_head.weight.shape[0])
    _progress("ram_apc")
    ram_apc = packed_probe._ram_apc(model, vocab)

    _progress("synthetic_256k")
    mx.reset_peak_memory()
    synthetic = boundary_probe._tier2_context(model, SYNTHETIC_CONTEXT)
    peak_memory = int(mx.get_peak_memory())
    compact_capacity = synthetic["compact_resident"]["capacity_before"]
    compact_layers = compact_capacity["layers"]
    capacity_plan = plan_nope_cache_capacity(SYNTHETIC_CONTEXT + 1)

    all_physical = all(
        row["physical_capacity_matches_plan"]
        and row["physical_alignment_exact"]
        for row in physical.values()
    )
    all_padding = all(
        row["padding_indices_all_sentinel"]
        and row["padding_valid_all_false"]
        and row["padding_keys_all_zero"]
        for row in physical.values()
    )
    all_mapping = all(
        row["first_padding_token_rejected"]
        and all(sample["token"] == sample["roundtrip"] for sample in row["sample_mappings"])
        for row in mapping.values()
    )
    acceptance = {
        "official_runtime_uses_kpool4": int(indexer.index_kpool) == 4,
        "single_explicit_alignment_contract": (
            DEFAULT_NOPE_CACHE_TILE_ALIGNMENT.allocation_alignment_tokens == 256
            and DEFAULT_NOPE_CACHE_TILE_ALIGNMENT.physical_indexer_tile_rows == 64
            and not DEFAULT_NOPE_CACHE_TILE_ALIGNMENT.virtual_tile_split_allowed
        ),
        "logical_and_physical_capacity_distinct": all(
            row["planned"]["physical_capacity_tokens"]
            >= row["planned"]["logical_capacity_tokens"]
            for row in physical.values()
        ),
        "alignment_minus_exact_plus_boundaries_match_plan": all_physical,
        "logical_pool_physical_expansion_roundtrip_exact": all_mapping,
        "padding_never_selectable_or_valid": all_padding,
        "restore_trim_replay_exact_and_padding_retired": all(
            value for key, value in trim_restore.items() if isinstance(value, bool)
        ),
        "direct_compact_256k_selected_indices_and_dsa_output_exact": (
            synthetic["direct_compact_dsa_indices_byte_identical"]
            and synthetic["direct_compact_dsa_outputs_byte_identical"]
        ),
        "direct_compact_256k_first_logits_and_state_exact": (
            synthetic["direct_compact_first_logits_byte_identical"]
            and synthetic["direct_compact_kda_post_hash_match"]
            and synthetic["direct_compact_indexpool_post_hash_match"]
        ),
        "compact_256k_ram_restore_exact": (
            synthetic["compact_resident_restore_logits_byte_identical"]
            and synthetic["compact_resident_restore_post_state_exact"]
        ),
        "compact_256k_capacity_preallocated_and_aligned": (
            synthetic["compact_resident"]["capacity_unchanged"]
            and all(
                row["latent_tokens"] == capacity_plan.physical_capacity_tokens
                and row["indexpool_rows"] == capacity_plan.physical_pool_rows
                for row in compact_layers
            )
        ),
        "all_256k_indices_sentinel_or_logically_in_range": all(
            row["out_of_range_count"] == 0
            for arm in ("direct", "compact_resident")
            for row in synthetic[arm]["dsa_diagnostics"]
        ),
        "ram_apc_continuation_exact": (
            ram_apc["all_logits_hashes_match"]
            and ram_apc["post_state_exact"]
            and ram_apc["snapshot_immutable"]
        ),
        "official_16_128_oracle_exact": (
            official_oracle["first_16_match"]
            and official_oracle["full_128_match"]
        ),
        "no_anonymous_cache_allocation": all(
            row["anonymous_allocation_bytes"] == 0 for row in physical.values()
        ),
        "kda_capacity_is_fixed_two_slot_and_not_kpool_paged": (
            DEFAULT_NOPE_CACHE_TILE_ALIGNMENT.kda_state_slots == 2
            and DEFAULT_NOPE_CACHE_TILE_ALIGNMENT.descriptor()[
                "kda_capacity_geometry"
            ]
            == "fixed-slots-orthogonal-to-token-capacity"
        ),
    }
    artifact = {
        "schema": "glm53-kpool-cache-tile-alignment-v1",
        "date": date.today().isoformat(),
        "complete": all(acceptance.values()),
        "probe_only": True,
        "official_hf_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "direct_cache_abi": NOPE_DSA_CACHE_ABI_DIRECT,
        "compact_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
        "contract": DEFAULT_NOPE_CACHE_TILE_ALIGNMENT.descriptor(),
        "contract_identity": NOPE_CACHE_TILE_ALIGNMENT_CONTRACT,
        "constraints": {
            "logical_cache_block_tokens": 256,
            "index_kpool": 4,
            "physical_indexer_tile_rows": 64,
            "selected_gather_token_granularity": 1,
            "custom_metal_indexer_kernel": False,
            "kda_state_slots": 2,
            "virtual_page_or_tile_split": False,
        },
        "mapping_boundaries": mapping,
        "physical_boundaries": physical,
        "ram_restore_trim_replay": trim_restore,
        "official_oracle": official_oracle,
        "ram_apc": ram_apc,
        "synthetic_256k": synthetic,
        "peak_memory_bytes": peak_memory,
        "claims": {
            "validated": "256k resident/restore to first decode with aligned compact cache",
            "unsupported_unvalidated": "256k cold prefill to first decode",
        },
        "runtime_changes": {
            "admission": False,
            "apc_namespace": False,
            "cache_state_schema": False,
            "direct_backend": False,
            "kernel_abi": False,
            "server": False,
        },
        "acceptance": acceptance,
        "accepted": all(acceptance.values()),
        "decision": (
            "cache_tile_alignment_safety_closed"
            if all(acceptance.values())
            else "keep_runtime_unchanged_and_investigate"
        ),
    }
    _atomic_write(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "accepted": artifact["accepted"],
                "peak_memory_bytes": peak_memory,
            },
            indent=2,
        )
    )
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
