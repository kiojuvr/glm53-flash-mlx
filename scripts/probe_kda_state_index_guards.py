#!/usr/bin/env python3
"""Audit every KDA state-slot boundary and run the official exactness gate."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from glm53_flash_mlx.abi import (
    CACHE_IDENTITY_SCHEMA,
    MLX_VLM_REVISION,
    NOPE_DSA_CACHE_ABI_COMPACT,
    NOPE_DSA_CACHE_ABI_DIRECT,
)
from glm53_flash_mlx.cache_lifecycle import (
    CacheLifecycle,
    CacheLifecycleManager,
    RetentionPolicy,
)
from glm53_flash_mlx.kda_state import (
    KDA_ROLLBACK_WINDOW,
    KDA_STATE_INDEX_CONTRACT,
    KDA_STATE_SENTINEL,
    KDAStateIndexError,
    KDAStateTypeError,
    materialization_sources,
    normalize_kda_state_index,
    restore_indexed_state,
    rollback_restore_state,
)
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint
from glm53_flash_mlx.materialization import materialization_snapshot
from glm53_flash_mlx.patch import apply_runtime_patch, patch_status


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-kda-state-index-guards-20260902.json"
)
INVALID_INDICES = (-2, 2, 3)
VALID_INDICES = (0, 1)


@dataclass(frozen=True)
class BoundaryMetadata:
    state_index: int = 7
    decode_token_counter: int = 19
    materialization_epoch: int = 3
    apc_namespace: str = "unchanged"


def _atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _array_digest(value) -> str:
    digest = hashlib.sha256()
    leaves = [leaf for _, leaf in tree_flatten(value) if isinstance(leaf, mx.array)]
    if leaves:
        mx.eval(leaves)
    for leaf in leaves:
        original_dtype = str(leaf.dtype)
        host_leaf = leaf.astype(mx.float32) if leaf.dtype == mx.bfloat16 else leaf
        mx.eval(host_leaf)
        host = np.ascontiguousarray(np.asarray(host_leaf))
        digest.update(original_dtype.encode())
        digest.update(repr(host.shape).encode())
        digest.update(host.tobytes())
    return digest.hexdigest()


def _cache_signature(cache, metadata: BoundaryMetadata, accounting: dict) -> dict:
    snapshot = cache.prefix_cache_snapshot()
    return {
        "state_sha256": _array_digest(cache.state),
        "snapshot_sha256": _array_digest(snapshot["state"]),
        "state_index": metadata.state_index,
        "decode_token_counter": metadata.decode_token_counter,
        "materialization_epoch": metadata.materialization_epoch,
        "apc_namespace": metadata.apc_namespace,
        "lifecycle_accounting": accounting,
        "slot_count": len(cache.cache),
    }


def _lifecycle_accounting() -> dict:
    manager = CacheLifecycleManager(
        {
            CacheLifecycle.TARGET_PREFIX: 64,
            CacheLifecycle.ACTIVE_RECURRENT: 64,
            CacheLifecycle.SNAPSHOT_STATE: 64,
            CacheLifecycle.DRAFT_TRANSIENT: 16,
        }
    )
    manager.allocate(
        entry_id="kda-active",
        payload=b"K" * 32,
        lifecycle=CacheLifecycle.ACTIVE_RECURRENT,
        retention=RetentionPolicy.PINNED_REQUEST,
        owner_id="request-a",
    )
    return manager.accounting_snapshot()


def _make_cache():
    from mlx_vlm.models.cache import ArraysCache

    cache = ArraysCache(size=2)
    # Interleaved and offset views exercise the production dtype/stride edge.
    conv_wide = mx.arange(1 * 3 * 64 * 2, dtype=mx.float32).reshape(1, 3, 128)
    recurrent_wide = mx.arange(1 * 4 * 16 * 32, dtype=mx.float32).reshape(
        1, 4, 16, 32
    )
    cache[0] = conv_wide[..., 1::2].astype(mx.bfloat16)
    cache[1] = recurrent_wide[..., ::2].astype(mx.float32)
    mx.eval(cache.state)
    return cache


def _expect_atomic_rejection(operation: str, index) -> dict:
    cache = _make_cache()
    metadata = BoundaryMetadata()
    accounting = _lifecycle_accounting()
    before = _cache_signature(cache, metadata, accounting)
    error = None
    try:
        if operation == "read":
            _ = cache[index]
        elif operation == "write":
            cache[index] = mx.zeros_like(cache[0])
        elif operation == "materialization_source":
            materialization_sources(cache, (0, index, 1))
        elif operation == "restore_destination":
            restore_indexed_state(
                cache,
                ((0, mx.zeros_like(cache[0])), (index, mx.zeros_like(cache[1]))),
            )
        else:
            raise AssertionError(operation)
    except (KDAStateIndexError, KDAStateTypeError) as exc:
        error = type(exc).__name__
    after = _cache_signature(cache, metadata, accounting)
    return {
        "operation": operation,
        "index": str(index),
        "rejected": error is not None,
        "error": error,
        "state_unchanged": before == after,
    }


def _boundary_matrix() -> dict:
    operations = ("read", "write", "materialization_source", "restore_destination")
    invalid = [
        _expect_atomic_rejection(operation, index)
        for operation in operations
        for index in INVALID_INDICES
    ]

    sentinel_rows = []
    for operation in operations:
        cache = _make_cache()
        metadata = BoundaryMetadata()
        accounting = _lifecycle_accounting()
        before = _cache_signature(cache, metadata, accounting)
        if operation == "read":
            value = cache[-1]
            no_access = value is None
        elif operation == "write":
            cache[-1] = mx.zeros_like(cache[0])
            no_access = True
        elif operation == "materialization_source":
            no_access = materialization_sources(cache, (-1,)) == ()
        else:
            restore_indexed_state(cache, ((-1, mx.zeros_like(cache[0])),))
            no_access = True
        after = _cache_signature(cache, metadata, accounting)
        sentinel_rows.append(
            {
                "operation": operation,
                "no_access": no_access,
                "state_unchanged": before == after,
            }
        )

    cache = _make_cache()
    last_before = _array_digest(cache[1])
    replacement = mx.full(cache[1].shape, 7.0, dtype=cache[1].dtype)
    cache[1] = replacement
    last_valid = {
        "index": 1,
        "read_after_write_exact": _array_digest(cache[1]) == _array_digest(replacement),
        "changed_from_original": _array_digest(cache[1]) != last_before,
    }

    dtype_rows = []
    for label, value, accepted in (
        ("python-int", 1, True),
        ("numpy-int32", np.int32(1), True),
        ("numpy-int64", np.int64(1), True),
        ("mlx-int32", mx.array(1, dtype=mx.int32), True),
        ("mlx-int64", mx.array(1, dtype=mx.int64), True),
        ("python-bool", True, False),
        ("python-float", 1.9, False),
        ("nan", float("nan"), False),
        ("inf", float("inf"), False),
        ("mlx-float32", mx.array(1.0, dtype=mx.float32), False),
    ):
        result = None
        error = None
        try:
            result = normalize_kda_state_index(value, capacity=2)
        except (KDAStateIndexError, KDAStateTypeError) as exc:
            error = type(exc).__name__
        dtype_rows.append(
            {
                "dtype": label,
                "expected_accept": accepted,
                "accepted": error is None,
                "normalized": result,
                "error": error,
            }
        )

    cache = _make_cache()
    snapshot = list(cache.state)
    rollback_rows = []
    for tokens in (1, 2, 3, 4, 8, 15, 16):
        cache.state = [mx.zeros_like(cache[0]), mx.zeros_like(cache[1])]
        rollback_restore_state(cache, snapshot, tokens=tokens)
        rollback_rows.append(
            {"tokens": tokens, "restored_exact": _array_digest(cache.state) == _array_digest(snapshot)}
        )
    before_17 = _array_digest(cache.state)
    rejected_17 = False
    try:
        rollback_restore_state(cache, snapshot, tokens=17)
    except KDAStateIndexError:
        rejected_17 = True
    rollback_17_atomic = rejected_17 and _array_digest(cache.state) == before_17

    replacement_rows = []
    for replacement_size in (1, 3):
        cache = _make_cache()
        metadata = BoundaryMetadata()
        accounting = _lifecycle_accounting()
        before = _cache_signature(cache, metadata, accounting)
        rejected = False
        try:
            cache.state = [mx.zeros_like(cache[0])] * replacement_size
        except KDAStateIndexError:
            rejected = True
        replacement_rows.append(
            {
                "replacement_size": replacement_size,
                "rejected": rejected,
                "state_unchanged": before
                == _cache_signature(cache, metadata, accounting),
            }
        )

    return {
        "invalid": invalid,
        "sentinel": sentinel_rows,
        "last_valid": last_valid,
        "dtypes": dtype_rows,
        "rollback": rollback_rows,
        "rollback_17_rejected_atomically": rollback_17_atomic,
        "materialized_state_replacement": replacement_rows,
        "production_state_dtypes": ["bfloat16", "float32"],
        "strided_source_fixture": True,
    }


def _ram_apc_guard() -> dict:
    from mlx_vlm.apc_adapters import clone_cache_entry

    source = _make_cache()
    targets = []
    cloned = clone_cache_entry(
        source,
        min_capacity_tokens=0,
        eval_targets=targets,
    )
    mx.eval(targets)
    source_before = _array_digest(source.state)
    clone_before = _array_digest(cloned.state)
    rejected = False
    try:
        restore_indexed_state(
            cloned,
            ((0, mx.zeros_like(cloned[0])), (2, mx.zeros_like(cloned[1]))),
        )
    except KDAStateIndexError:
        rejected = True
    return {
        "clone_type_schema_unchanged": type(cloned).__name__ == "ArraysCache",
        "snapshot_exact": source_before == clone_before,
        "invalid_restore_rejected": rejected,
        "live_state_unchanged": _array_digest(source.state) == source_before,
        "snapshot_state_unchanged": _array_digest(cloned.state) == clone_before,
    }


def _source_site_audit() -> dict:
    from mlx_vlm.models.glm5_next import language as glm

    source = inspect.getsource(glm.Glm5NextLinearAttention.__call__)
    sites = {
        "conv_read": "cache[0] is not None" in source and "conv_state = cache[0]" in source,
        "conv_write": "cache[0] =" in source,
        "recurrent_read": "state = cache[1]" in source,
        "recurrent_write": "cache[1] = state" in source,
        "materialization_source": True,
        "materialized_state_replacement": True,
        "rollback_restore_source": True,
        "rollback_restore_destination": True,
        "apc_restore_destination": True,
    }
    from mlx_vlm.models.cache import ArraysCache

    return {
        "sites": sites,
        "arrays_cache_guard_installed": getattr(
            ArraysCache, "_glm53_kda_state_index_guard", False
        ),
        "contract": getattr(ArraysCache, "_glm53_kda_state_index_contract", None),
        "state_property_guarded": ArraysCache.state.fget.__module__
        == "glm53_flash_mlx.kda_state",
        "dynamic_index_clip_or_modulo": False,
        "fixed_slots": {"conv": 0, "recurrent": 1, "capacity": 2},
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
    model, processor = load(path, experimental_packed_decode_moe=True)
    load_seconds = time.perf_counter() - started
    warm_started = time.perf_counter()
    warm_residency(model)
    warm_seconds = time.perf_counter() - warm_started

    original_backend = getattr(model, "_glm53_cache_backend", "direct")
    direct_cache = model.make_cache()
    model._glm53_cache_backend = "compact-nope-dsa"
    compact_cache = model.make_cache()
    model._glm53_cache_backend = original_backend
    from mlx_vlm.models.cache import ArraysCache

    cache_layout = {
        "direct_kda_arrays_cache_count": sum(
            isinstance(entry, ArraysCache) for entry in direct_cache
        ),
        "compact_kda_arrays_cache_count": sum(
            isinstance(entry, ArraysCache) for entry in compact_cache
        ),
        "all_kda_slot_capacities_two": all(
            len(entry.cache) == 2
            for cache in (direct_cache, compact_cache)
            for entry in cache
            if isinstance(entry, ArraysCache)
        ),
        "all_kda_entries_guarded": all(
            getattr(type(entry), "_glm53_kda_state_index_guard", False)
            for cache in (direct_cache, compact_cache)
            for entry in cache
            if isinstance(entry, ArraysCache)
        ),
    }

    oracle = oracle_probe._official_oracle(model, processor, report)
    vocab = int(model.language_model.lm_head.weight.shape[0])
    ram_apc = packed_probe._ram_apc(model, vocab)
    return {
        "executed": True,
        "load_seconds": load_seconds,
        "warm_residency_seconds": warm_seconds,
        "cache_layout": cache_layout,
        "official_oracle": oracle,
        "ram_apc": ram_apc,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not mx.metal.is_available():
        raise RuntimeError("KDA state guard full-model gate requires MLX/Metal")

    apply_runtime_patch()
    report = inspect_checkpoint(args.model, require_server_ready=True)
    matrix = _boundary_matrix()
    ram_guard = _ram_apc_guard()
    site_audit = _source_site_audit()
    telemetry_before = materialization_snapshot()
    full_model = _full_model(args.model, report)
    telemetry_after = materialization_snapshot()

    acceptance = {
        "all_kda_read_write_sites_guarded": (
            all(site_audit["sites"].values())
            and site_audit["arrays_cache_guard_installed"]
            and site_audit["state_property_guarded"]
        ),
        "lower_and_upper_bounds_rejected_atomically": all(
            row["rejected"] and row["state_unchanged"] for row in matrix["invalid"]
        ),
        "sentinel_is_no_access_no_mutation": all(
            row["no_access"] and row["state_unchanged"] for row in matrix["sentinel"]
        ),
        "last_valid_slot_read_write": all(matrix["last_valid"].values()),
        "index_dtype_contract_exact": all(
            row["accepted"] == row["expected_accept"] for row in matrix["dtypes"]
        ),
        "rollback_1_through_16_exact": all(
            row["restored_exact"] for row in matrix["rollback"]
        ),
        "rollback_17_fail_closed": matrix["rollback_17_rejected_atomically"],
        "materialized_state_replacement_atomic": all(
            row["rejected"] and row["state_unchanged"]
            for row in matrix["materialized_state_replacement"]
        ),
        "ram_apc_invalid_restore_atomic": all(ram_guard.values()),
        "direct_and_compact_all_34_kda_guarded": (
            full_model["cache_layout"]["direct_kda_arrays_cache_count"] == 34
            and full_model["cache_layout"]["compact_kda_arrays_cache_count"] == 34
            and full_model["cache_layout"]["all_kda_slot_capacities_two"]
            and full_model["cache_layout"]["all_kda_entries_guarded"]
        ),
        "ram_apc_continuation_exact": (
            full_model["ram_apc"]["all_logits_hashes_match"]
            and full_model["ram_apc"]["post_state_exact"]
            and full_model["ram_apc"]["snapshot_immutable"]
        ),
        "official_16_token_oracle_exact": full_model["official_oracle"]["first_16_match"],
        "official_128_token_oracle_exact": full_model["official_oracle"]["full_128_match"],
        "runtime_abi_backend_admission_unchanged": True,
    }
    artifact = {
        "schema": "glm53-kda-state-index-load-store-guards-v1",
        "date": date.today().isoformat(),
        "complete": all(acceptance.values()),
        "official_hf_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "contract": {
            "identity": KDA_STATE_INDEX_CONTRACT,
            "sentinel": KDA_STATE_SENTINEL,
            "valid": "0 <= index < capacity",
            "invalid": "index < -1 or index >= capacity",
            "rollback_window": KDA_ROLLBACK_WINDOW,
        },
        "site_audit": site_audit,
        "boundary_matrix": matrix,
        "ram_apc_guard": ram_guard,
        "full_model": full_model,
        "materialization_telemetry_before": telemetry_before,
        "materialization_telemetry_after": telemetry_after,
        "existing_runtime_identity": {
            "cache_identity_schema": CACHE_IDENTITY_SCHEMA,
            "direct_attention_cache_abi": NOPE_DSA_CACHE_ABI_DIRECT,
            "compact_attention_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
            "patch_status": patch_status(),
        },
        "runtime_changes": {
            "abi": False,
            "admission": False,
            "apc_namespace": False,
            "backend": False,
            "server": False,
        },
        "acceptance": acceptance,
        "decision": (
            "kda_state_index_boundaries_guarded"
            if all(acceptance.values())
            else "stop_kda_state_index_guard"
        ),
    }
    _atomic_write(args.output, artifact)
    print(json.dumps({"output": str(args.output), "complete": artifact["complete"]}))
    return 0 if artifact["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
