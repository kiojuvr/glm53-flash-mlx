#!/usr/bin/env python3
"""Qualify pooled DSA Indexer sizing and exact query-row blocking."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np

from glm53_flash_mlx.abi import (
    CACHE_IDENTITY_SCHEMA,
    MLX_VLM_REVISION,
    NOPE_DSA_CACHE_ABI_COMPACT,
    NOPE_DSA_CACHE_ABI_DIRECT,
)
from glm53_flash_mlx.dsa_workspace import (
    DEFAULT_INDEX_KPOOL,
    DEFAULT_INDEX_TOPK,
    DEFAULT_MAX_WORKSPACE_BYTES,
    DSA_INDEXER_WORKSPACE_CONTRACT,
    account_dsa_indexer_memory,
    plan_dsa_indexer_workspace,
)
from glm53_flash_mlx.indexpool import INDEXPOOL_SENTINEL, expand_selected_pools
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-dsa-indexer-workspace-geometry-20260904.json"
)
QUERY_BOUNDARIES = (1, 63, 64, 65, 127, 128, 255, 256)
CONTEXT_BOUNDARIES = (
    1,
    2,
    3,
    4,
    5,
    2_047,
    2_048,
    2_049,
    4_095,
    4_096,
    4_097,
)
QUALIFICATION_CONTEXTS = (131_072, 262_144)


def _atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _array(value) -> np.ndarray:
    mx.eval(value)
    raw = value.view(mx.uint16) if value.dtype == mx.bfloat16 else value
    return np.ascontiguousarray(np.asarray(raw))


def _hash(value) -> str:
    return hashlib.sha256(_array(value).tobytes()).hexdigest()


def _deterministic_logits(start: int, end: int, pool_count: int) -> mx.array:
    """Produce unique FP32 pool scores using absolute query row identities."""
    queries = mx.arange(start, end, dtype=mx.int32)[:, None]
    pools = mx.arange(pool_count, dtype=mx.int32)[None, :]
    centers = (queries * 1543 + 17) % max(pool_count, 1)
    distance = mx.abs(pools - centers).astype(mx.float32)
    tie_break = (pools % 251).astype(mx.float32) * (2.0**-20)
    return (-distance + tie_break).astype(mx.float32)[None]


def _pool_state(context_tokens: int):
    geometry = plan_dsa_indexer_workspace(
        context_tokens=context_tokens,
        num_query_rows=1,
    )
    pool_count = geometry.pool_count
    positions = mx.arange(pool_count * DEFAULT_INDEX_KPOOL, dtype=mx.int64).reshape(
        1, pool_count, DEFAULT_INDEX_KPOOL
    )
    in_range = positions < context_tokens
    pool_indices = mx.where(in_range, positions, INDEXPOOL_SENTINEL)
    pool_valid = mx.all(in_range, axis=-1)
    active = context_tokens % DEFAULT_INDEX_KPOOL
    if active:
        tail_positions = mx.arange(
            context_tokens - active, context_tokens, dtype=mx.int64
        )[None]
        tail_valid = mx.ones((1, active), dtype=mx.bool_)
    else:
        tail_positions = mx.zeros((1, 0), dtype=mx.int64)
        tail_valid = mx.zeros((1, 0), dtype=mx.bool_)
    return geometry, pool_indices, pool_valid, tail_positions, tail_valid


def _select_block(logits, pool_valid, select_k: int) -> dict:
    valid = mx.broadcast_to(pool_valid[:, None], logits.shape)
    masked = mx.where(valid, logits, mx.array(-1e30, dtype=mx.float32))
    order = mx.argsort(-masked, axis=-1)[..., :select_k]
    selected_scores = mx.take_along_axis(masked, order, axis=-1)
    selected_valid = mx.take_along_axis(valid, order, axis=-1)
    return {
        "logits": logits,
        "masked_logits": masked,
        "selected_scores": selected_scores,
        "selected_pool_rows": order,
        "selected_valid": selected_valid,
    }


def _reference_selection(num_queries: int, pool_valid) -> dict:
    pool_count = int(pool_valid.shape[1])
    select_k = min(DEFAULT_INDEX_TOPK // DEFAULT_INDEX_KPOOL, pool_count)
    return _select_block(
        _deterministic_logits(0, num_queries, pool_count),
        pool_valid,
        select_k,
    )


def _row_blocked_selection(geometry, pool_valid, *, retain_logits: bool) -> dict:
    if geometry.query_block_count == 1:
        return _select_block(
            _deterministic_logits(
                0, geometry.num_query_rows, geometry.pool_count
            ),
            pool_valid,
            geometry.selected_pool_count,
        )
    blocks = []
    for start in range(0, geometry.num_query_rows, geometry.query_block_rows):
        end = min(start + geometry.query_block_rows, geometry.num_query_rows)
        block = _select_block(
            _deterministic_logits(start, end, geometry.pool_count),
            pool_valid,
            geometry.selected_pool_count,
        )
        mx.eval(
            block["selected_scores"],
            block["selected_pool_rows"],
            block["selected_valid"],
            *([block["logits"], block["masked_logits"]] if retain_logits else []),
        )
        blocks.append(block)
    output = {
        key: mx.concatenate([block[key] for block in blocks], axis=1)
        for key in ("selected_scores", "selected_pool_rows", "selected_valid")
    }
    if retain_logits:
        output["logits"] = mx.concatenate(
            [block["logits"] for block in blocks], axis=1
        )
        output["masked_logits"] = mx.concatenate(
            [block["masked_logits"] for block in blocks], axis=1
        )
    return output


def _expand(selection, pool_indices, pool_valid, tail_positions, tail_valid, context):
    selected_valid = selection["selected_valid"] & mx.take_along_axis(
        mx.broadcast_to(
            pool_valid[:, None],
            (1, selection["selected_pool_rows"].shape[1], pool_valid.shape[1]),
        ),
        selection["selected_pool_rows"],
        axis=-1,
    )
    return expand_selected_pools(
        selection["selected_pool_rows"],
        pool_indices,
        selected_valid,
        kv_len=context,
        index_topk=DEFAULT_INDEX_TOPK,
        index_kpool=DEFAULT_INDEX_KPOOL,
        tail_positions=tail_positions,
        tail_valid=tail_valid,
        always_select_tail=True,
    )


def _correctness_case(context: int, queries: int, *, block_rows: int) -> dict:
    base, pool_indices, pool_valid, tail_positions, tail_valid = _pool_state(context)
    budget = max(4, block_rows * base.pool_count * 4)
    geometry = plan_dsa_indexer_workspace(
        context_tokens=context,
        num_query_rows=queries,
        max_workspace_bytes=budget,
    )
    reference = _reference_selection(queries, pool_valid)
    blocked = _row_blocked_selection(geometry, pool_valid, retain_logits=True)
    reference_expanded = _expand(
        reference,
        pool_indices,
        pool_valid,
        tail_positions,
        tail_valid,
        context,
    )
    blocked_expanded = _expand(
        blocked,
        pool_indices,
        pool_valid,
        tail_positions,
        tail_valid,
        context,
    )
    arrays = (
        reference["logits"],
        blocked["logits"],
        reference["selected_scores"],
        blocked["selected_scores"],
        reference["selected_pool_rows"],
        blocked["selected_pool_rows"],
        reference_expanded[0],
        blocked_expanded[0],
        reference_expanded[1],
        blocked_expanded[1],
    )
    mx.eval(*arrays)
    indices = _array(blocked_expanded[0])
    valid = _array(blocked_expanded[1]).astype(bool, copy=False)
    return {
        "context_tokens": context,
        "query_rows": queries,
        "pool_count": geometry.pool_count,
        "query_block_rows": geometry.query_block_rows,
        "query_block_count": geometry.query_block_count,
        "workspace_bytes": geometry.fp32_logits_workspace_bytes,
        "raw_logits_byte_exact": bool(mx.array_equal(arrays[0], arrays[1]).item()),
        "topk_scores_byte_exact": bool(mx.array_equal(arrays[2], arrays[3]).item()),
        "topk_pool_indices_byte_exact": bool(
            mx.array_equal(arrays[4], arrays[5]).item()
        ),
        "expanded_indices_byte_exact": bool(
            mx.array_equal(arrays[6], arrays[7]).item()
        ),
        "sentinel_positions_byte_exact": bool(
            mx.array_equal(arrays[8], arrays[9]).item()
        ),
        "selected_width": int(indices.shape[-1]),
        "sentinel_count": int(np.count_nonzero(indices == INDEXPOOL_SENTINEL)),
        "non_sentinel_out_of_range": int(
            np.count_nonzero((indices != INDEXPOOL_SENTINEL) & ((indices < 0) | (indices >= context)))
        ),
        "valid_mask_count": int(np.count_nonzero(valid)),
        "index_hash": hashlib.sha256(indices.tobytes()).hexdigest(),
    }


def _qualification_case(context: int) -> dict:
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    active_before = int(mx.get_active_memory())
    mx.reset_peak_memory()
    started = time.perf_counter()
    result = _correctness_case(context, 256, block_rows=256)
    mx.synchronize()
    result["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
    result["active_before_bytes"] = active_before
    result["active_after_operator_bytes"] = int(mx.get_active_memory())
    result["differential_fixture_working_peak_bytes"] = (
        int(mx.get_peak_memory()) - active_before
    )
    result["differential_fixture_holds_reference_and_candidate"] = True
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    result["active_after_release_bytes"] = int(mx.get_active_memory())
    result["post_release_active_drift_bytes"] = (
        result["active_after_release_bytes"] - active_before
    )
    geometry = plan_dsa_indexer_workspace(
        context_tokens=context, num_query_rows=256
    )
    _, pool_indices, pool_valid, tail_positions, tail_valid = _pool_state(context)
    mx.reset_peak_memory()
    candidate = _row_blocked_selection(geometry, pool_valid, retain_logits=False)
    expanded = _expand(
        candidate,
        pool_indices,
        pool_valid,
        tail_positions,
        tail_valid,
        context,
    )
    mx.eval(
        candidate["selected_scores"],
        candidate["selected_pool_rows"],
        candidate["selected_valid"],
        expanded[0],
        expanded[1],
    )
    mx.synchronize()
    result["candidate_operator_working_peak_bytes"] = (
        int(mx.get_peak_memory()) - active_before
    )
    del candidate, expanded, pool_indices, pool_valid, tail_positions, tail_valid
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    result["candidate_post_release_active_drift_bytes"] = (
        int(mx.get_active_memory()) - active_before
    )
    result["geometry"] = geometry.descriptor()
    result["memory_accounting"] = account_dsa_indexer_memory(
        geometry, index_head_dim=128
    ).descriptor()
    result["logical_context_logits_bytes_forbidden"] = 256 * context * 4
    return result


def _timed_selection(
    context: int,
    *,
    blocked: bool,
    samples: int = 5,
    repetitions: int = 64,
) -> list[float]:
    geometry, _, pool_valid, _, _ = _pool_state(context)
    geometry = plan_dsa_indexer_workspace(
        context_tokens=context,
        num_query_rows=256,
    )
    timings = []
    for iteration in range(2 + samples):
        started = time.perf_counter()
        for _ in range(repetitions):
            if blocked:
                output = _row_blocked_selection(
                    geometry, pool_valid, retain_logits=False
                )
            else:
                output = _reference_selection(256, pool_valid)
            mx.eval(
                output["selected_scores"],
                output["selected_pool_rows"],
                output["selected_valid"],
            )
        mx.synchronize()
        elapsed = (time.perf_counter() - started) * 1000.0 / repetitions
        if iteration >= 2:
            timings.append(elapsed)
    return timings


def _performance_screen() -> dict:
    context = 8_192
    reference = _timed_selection(context, blocked=False)
    blocked = _timed_selection(context, blocked=True)
    reference_median = statistics.median(reference)
    blocked_median = statistics.median(blocked)
    return {
        "context_tokens": context,
        "query_rows": 256,
        "warmups": 2,
        "samples": 5,
        "repetitions_per_sample": 64,
        "reference_ms": reference,
        "blocked_ms": blocked,
        "reference_median_ms": reference_median,
        "blocked_median_ms": blocked_median,
        "blocked_over_reference": blocked_median / reference_median,
    }


def _load_probe_modules():
    scripts = str(REPOSITORY / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import probe_exact_sigmoid_gate_metal_barrier as oracle_probe
    import probe_packed_decode_runtime as packed_probe

    return oracle_probe, packed_probe


def _full_model(path: Path, report) -> dict:
    oracle_probe, packed_probe = _load_probe_modules()
    mx.set_wired_limit(int(440e9))
    mx.set_cache_limit(int(32e9))
    started = time.perf_counter()
    model, processor = load(
        path,
        experimental_packed_decode_moe=True,
        experimental_compact_nope_dsa_cache=True,
    )
    load_seconds = time.perf_counter() - started
    warm_started = time.perf_counter()
    warm_residency(model)
    warm_seconds = time.perf_counter() - warm_started
    layer3 = model.language_model.model.layers[3].self_attn.indexer
    oracle = oracle_probe._official_oracle(model, processor, report)
    vocab = int(model.language_model.lm_head.weight.shape[0])
    ram_apc = packed_probe._ram_apc(model, vocab)
    return {
        "executed": True,
        "load_seconds": load_seconds,
        "warm_residency_seconds": warm_seconds,
        "moe_backend": getattr(model, "_glm53_moe_backend", None),
        "cache_backend": getattr(model, "_glm53_cache_backend", None),
        "layer3_indexer": {
            "index_kpool": int(layer3.index_kpool),
            "index_topk": int(layer3.index_topk),
            "head_dim": int(layer3.head_dim),
        },
        "official_oracle": oracle,
        "ram_apc": ram_apc,
    }


def _all_exact(row: dict) -> bool:
    return all(
        row[key]
        for key in (
            "raw_logits_byte_exact",
            "topk_scores_byte_exact",
            "topk_pool_indices_byte_exact",
            "expanded_indices_byte_exact",
            "sentinel_positions_byte_exact",
        )
    ) and row["non_sentinel_out_of_range"] == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not mx.metal.is_available():
        raise RuntimeError("DSA workspace geometry probe requires MLX/Metal")

    report = inspect_checkpoint(args.model, require_server_ready=True)
    planner_boundaries = {
        str(context): plan_dsa_indexer_workspace(
            context_tokens=context,
            num_query_rows=256,
        ).descriptor()
        for context in (
            0,
            1,
            2,
            3,
            4,
            5,
            131_071,
            131_072,
            131_073,
            262_143,
            262_144,
            262_145,
            1_048_576,
        )
    }
    small_cases = []
    for context in CONTEXT_BOUNDARIES:
        for queries in QUERY_BOUNDARIES:
            small_cases.append(
                _correctness_case(context, queries, block_rows=min(64, queries))
            )
    qualification = {
        str(context): _qualification_case(context)
        for context in QUALIFICATION_CONTEXTS
    }
    geometry_1m = plan_dsa_indexer_workspace(
        context_tokens=1_048_576,
        num_query_rows=256,
    )
    performance = _performance_screen()
    full_model = _full_model(args.model, report)
    layer3 = full_model["layer3_indexer"]
    acceptance = {
        "official_kpool4_topk2048_confirmed": (
            layer3["index_kpool"] == 4 and layer3["index_topk"] == 2048
        ),
        "all_pool_counts_use_ceil_div": all(
            row["pool_count"] == (int(context) // 4 + int(int(context) % 4 != 0))
            for context, row in planner_boundaries.items()
        ),
        "partial_pool_boundaries_exact": (
            planner_boundaries["262144"]["pool_count"] == 65_536
            and planner_boundaries["262145"]["pool_count"] == 65_537
        ),
        "128k_q256_logits_workspace_at_most_32mib": (
            qualification["131072"]["geometry"]["fp32_logits_workspace_bytes"]
            <= 32 << 20
        ),
        "256k_q256_logits_workspace_at_most_64mib": (
            qualification["262144"]["geometry"]["fp32_logits_workspace_bytes"]
            <= DEFAULT_MAX_WORKSPACE_BYTES
        ),
        "1m_q256_uses_four_64row_blocks": (
            geometry_1m.query_block_rows == 64
            and geometry_1m.query_block_count == 4
            and geometry_1m.fp32_logits_workspace_bytes
            <= DEFAULT_MAX_WORKSPACE_BYTES
        ),
        "small_row_blocked_results_byte_exact": all(
            _all_exact(row) for row in small_cases
        ),
        "128k_256k_operator_results_byte_exact": all(
            _all_exact(row) for row in qualification.values()
        ),
        "selected_width_bounded_at_2051": all(
            row["selected_width"] == 2051
            for row in (*small_cases, *qualification.values())
        ),
        "workspace_uses_pool_not_logical_context": all(
            row["workspace_bytes"]
            <= row["query_block_rows"] * row["pool_count"] * 4
            < row["query_block_rows"] * row["context_tokens"] * 4
            for row in qualification.values()
        ),
        "transient_workspace_released": all(
            abs(row["post_release_active_drift_bytes"]) <= 64 << 20
            and abs(row["candidate_post_release_active_drift_bytes"]) <= 64 << 20
            for row in qualification.values()
        ),
        "persistent_indexpool_accounting_separate": all(
            set(row["memory_accounting"]) == {
                "transient",
                "persistent_indexpool",
                "anonymous_allocation_bytes",
            }
            and row["memory_accounting"]["anonymous_allocation_bytes"] == 0
            for row in qualification.values()
        ),
        "one_block_small_medium_regression_at_most_one_percent": (
            performance["blocked_over_reference"] <= 1.01
        ),
        "official_16_token_oracle_exact": full_model["official_oracle"][
            "first_16_match"
        ],
        "official_128_token_oracle_exact": full_model["official_oracle"][
            "full_128_match"
        ],
        "ram_apc_continuation_exact": (
            full_model["ram_apc"]["all_logits_hashes_match"]
            and full_model["ram_apc"]["post_state_exact"]
            and full_model["ram_apc"]["snapshot_immutable"]
        ),
    }
    artifact = {
        "schema": "glm53-dsa-indexer-workspace-geometry-v1",
        "date": date.today().isoformat(),
        "complete": all(acceptance.values()),
        "probe_only": True,
        "official_hf_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "contract": {
            "identity": DSA_INDEXER_WORKSPACE_CONTRACT,
            "index_kpool": DEFAULT_INDEX_KPOOL,
            "index_topk": DEFAULT_INDEX_TOPK,
            "max_fp32_logits_workspace_bytes": DEFAULT_MAX_WORKSPACE_BYTES,
            "pool_count_formula": "quotient + (remainder != 0)",
            "key_dimension": "pool_count",
            "row_block_axis": "query",
            "selected_output_width": 2051,
            "full_logical_context_logits_forbidden": True,
            "maximum_context_resident_scratch_forbidden": True,
        },
        "existing_runtime_identity": {
            "cache_identity_schema": CACHE_IDENTITY_SCHEMA,
            "direct_attention_cache_abi": NOPE_DSA_CACHE_ABI_DIRECT,
            "compact_attention_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
        },
        "planner_boundaries": planner_boundaries,
        "small_correctness_cases": small_cases,
        "qualification": qualification,
        "synthetic_1m_geometry": geometry_1m.descriptor(),
        "performance_screen": performance,
        "full_model": full_model,
        "runtime_changes": {
            "abi": False,
            "admission": False,
            "apc_namespace": False,
            "backend": False,
            "cache_implementation": False,
            "server": False,
        },
        "acceptance": acceptance,
        "decision": (
            "bounded_dsa_workspace_contract_ready_for_semantic_snapshot"
            if all(acceptance.values())
            else "stop_bounded_dsa_workspace_contract"
        ),
    }
    _atomic_write(args.output, artifact)
    print(json.dumps({"output": str(args.output), "complete": artifact["complete"]}))
    return 0 if artifact["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
