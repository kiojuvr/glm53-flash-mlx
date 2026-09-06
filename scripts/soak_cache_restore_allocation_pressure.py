#!/usr/bin/env python3
"""Soak exact RAM cache restore after cache-shaped allocation pressure.

This is a user-launched, probe-only qualification for the silent-corruption
class in which a persistent cache survives while live backing storage is
released and reused.  The default run performs one 32K coding-agent cold
prefill, records a 16-token greedy continuation, then restores and replays it
100 times through each of the production exact RAM APC and semantic snapshot
paths.  Before every restore, a full cache-shaped clone is materialized and
released without clearing MLX's allocator cache, deliberately making its
backing storage available for reuse.

Progress is atomically written after every generation.  The cache payload is
not serialized, so an interrupted process records negative/incomplete evidence
but cannot resume an in-flight qualification.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import weakref
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

import mlx.core as mx
import numpy as np

from glm53_flash_mlx.abi import MLX_VLM_REVISION, NOPE_DSA_CACHE_ABI_DIRECT
from glm53_flash_mlx.cache_lifecycle import CacheLifecycle
from glm53_flash_mlx.kda_digest import (
    aggregate_layer_digest,
    compare_layerwise_digests,
    layerwise_kda_digests,
)
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import EXPECTED_DSA, inspect_checkpoint
from glm53_flash_mlx.materialization import MATERIALIZATION_INTERVAL_TOKENS
from glm53_flash_mlx.semantic_snapshot import (
    DIRECT_INDEXPOOL_ABI,
    KDA_STATE_ABI,
    SEMANTIC_PREFIX_SNAPSHOT_SCHEMA,
    SemanticCacheHandle,
    SemanticSnapshotIdentity,
    SemanticSnapshotStore,
    inspect_semantic_boundary,
    semantic_cache_digest,
    semantic_cache_resident_bytes,
    semantic_cache_storage_alias_count,
    semantic_component_digests,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-cache-restore-allocation-pressure-20260906.json"
)
SCHEMA = "glm53-cache-restore-allocation-pressure-soak-v1"
RESTORE_PATHS = ("ram-apc", "semantic-snapshot")
DEFAULT_CONTEXT_TOKENS = 32_768
DEFAULT_CONTINUATION_TOKENS = 16
DEFAULT_RESTORE_GENERATIONS = 100
DEFAULT_PRESSURE_CLONES = 1
PREFILL_CHUNK_TOKENS = 2_048
MAX_ACTIVE_DRIFT_BYTES = 64 << 20
MAX_PEAK_BYTES = 340_000_000_000
KDA_LAYERS = tuple(layer for layer in range(45) if layer not in EXPECTED_DSA)


class RestorePressureDivergence(RuntimeError):
    """Raised immediately after atomically recording the first divergence."""


def _progress(phase: str, **values: Any) -> None:
    print(json.dumps({"phase": phase, **values}, sort_keys=True), flush=True)


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _memory() -> dict[str, int]:
    mx.synchronize()
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }


def _arrays(value: Any) -> Iterable[mx.array]:
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _arrays(item)
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _arrays(value[key])


def _entry_nbytes(entry: object) -> int:
    nbytes = getattr(entry, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    return sum(int(value.nbytes) for value in _arrays(entry.state))


def _authoritative_accounting(
    cache: Sequence[object], *, snapshot_owned: bool = False
) -> dict[str, Any]:
    by_lifecycle = {lifecycle.value: 0 for lifecycle in CacheLifecycle}
    leaf_count = 0
    total = 0
    for entry in cache:
        if snapshot_owned:
            size = _entry_nbytes(entry)
            by_lifecycle[CacheLifecycle.SNAPSHOT_STATE.value] += size
            total += size
            leaf_count += sum(1 for _ in _arrays(entry.state))
            continue
        children = tuple(getattr(entry, "caches", ()))
        if len(children) == 2:
            for child in children:
                size = _entry_nbytes(child)
                by_lifecycle[CacheLifecycle.TARGET_PREFIX.value] += size
                total += size
                leaf_count += sum(1 for _ in _arrays(child.state))
        else:
            size = _entry_nbytes(entry)
            by_lifecycle[CacheLifecycle.ACTIVE_RECURRENT.value] += size
            total += size
            leaf_count += sum(1 for _ in _arrays(entry.state))
    accounted = sum(by_lifecycle.values())
    return {
        "total_authoritative_bytes": total,
        "accounted_bytes": accounted,
        "anonymous_bytes": total - accounted,
        "state_leaf_count": leaf_count,
        "resident_bytes_by_lifecycle": by_lifecycle,
    }


def _cache_storage_objects(cache: Sequence[object]) -> list[object]:
    """Return unique entry/tensor objects, including derived Direct pool rows."""

    objects = []
    seen = set()

    def append(value: object) -> None:
        identity = id(value)
        if identity not in seen:
            seen.add(identity)
            objects.append(value)

    for entry in cache:
        append(entry)
        for value in _arrays(entry.state):
            append(value)
        for child in tuple(getattr(entry, "caches", ())):
            append(child)
            for value in _arrays(child.state):
                append(value)
            pool = getattr(child, "_pool", None)
            if pool is not None:
                for value in _arrays(pool):
                    append(value)
            for name in ("_left_padding", "_lengths"):
                value = getattr(child, name, None)
                if isinstance(value, mx.array):
                    append(value)
        for name in ("_left_padding", "_lengths"):
            value = getattr(entry, name, None)
            if isinstance(value, mx.array):
                append(value)
    return objects


def _raw(value: mx.array) -> bytes:
    storage = value.view(mx.uint16) if value.dtype == mx.bfloat16 else value
    mx.eval(storage)
    return np.ascontiguousarray(np.asarray(storage)).tobytes()


def _array_digest(value: mx.array) -> str:
    return hashlib.sha256(_raw(value)).hexdigest()


def _token_digest(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def _materialize(cache: Sequence[object], *, clear_allocator_cache: bool) -> None:
    targets = [value for entry in cache for value in _arrays(entry.state)]
    if targets:
        mx.eval(*targets)
    if clear_allocator_cache:
        mx.clear_cache()
    mx.synchronize()


def _clone_cache(cache: Sequence[object], *, min_capacity_tokens: int) -> list:
    from mlx_vlm.apc_adapters import clone_cache_entry

    targets = []
    cloned = [
        clone_cache_entry(
            entry,
            min_capacity_tokens=min_capacity_tokens,
            eval_targets=targets,
        )
        for entry in cache
    ]
    if any(entry is None for entry in cloned):
        raise RuntimeError("RAM cache clone rejected a hybrid cache entry")
    if targets:
        mx.eval(*targets)
    _materialize(cloned, clear_allocator_cache=False)
    return cloned


def _release_cache(cache: list[object]) -> dict[str, int | bool]:
    resident = semantic_cache_resident_bytes(cache)
    entry_references = [weakref.ref(entry) for entry in cache]
    storage_references = [
        weakref.ref(value) for value in _cache_storage_objects(cache)
    ]
    cache.clear()
    del cache
    gc.collect()
    stale_entries = sum(reference() is not None for reference in entry_references)
    stale_storage = sum(reference() is not None for reference in storage_references)
    return {
        "released_bytes": resident,
        "stale_entry_reference_count": stale_entries,
        "stale_storage_reference_count": stale_storage,
        "logical_resident_bytes_after_release": 0,
        "released": stale_entries == stale_storage == 0,
    }


def _cache_evidence(cache: Sequence[object], *, position: int) -> dict[str, Any]:
    layerwise = layerwise_kda_digests(
        cache, kda_layers=KDA_LAYERS, mx_module=mx
    )
    boundary = inspect_semantic_boundary(
        cache,
        absolute_token_position=position,
        materialization_epoch=position // MATERIALIZATION_INTERVAL_TOKENS,
    )
    return {
        "state_sha256": semantic_cache_digest(cache),
        "components": semantic_component_digests(cache),
        "layerwise_kda_sha256": aggregate_layer_digest(layerwise),
        "layerwise_kda": layerwise,
        "boundary": boundary.descriptor(),
        "resident_bytes": semantic_cache_resident_bytes(cache),
        "authoritative_accounting": _authoritative_accounting(cache),
    }


def _direct_pool_evidence(cache: Sequence[object]) -> dict[str, Any]:
    """Hash the derived Direct pool separately from authoritative cache state."""

    digest = hashlib.sha256()
    layers = []
    invalid_count = 0
    nan_count = 0
    resident_bytes = 0
    for layer in EXPECTED_DSA:
        children = tuple(getattr(cache[layer], "caches", ()))
        if len(children) != 2:
            raise TypeError("restore-pressure qualification requires Direct DSA cache")
        pool = getattr(children[1], "_pool", None)
        if pool is None:
            layers.append({"layer": layer, "present": False})
            digest.update(f"{layer}:none".encode())
            continue
        keys, indices, valid, extent = pool
        mx.eval(keys, indices, valid)
        extent = int(extent)
        bad = mx.sum(((indices < -1) | (indices >= extent)).astype(mx.int32))
        nan = mx.sum(mx.isnan(keys).astype(mx.int32))
        mx.eval(bad, nan)
        bad_count = int(bad.item())
        key_nan_count = int(nan.item())
        invalid_count += bad_count
        nan_count += key_nan_count
        layer_bytes = int(keys.nbytes + indices.nbytes + valid.nbytes)
        resident_bytes += layer_bytes
        for value in (keys, indices, valid):
            digest.update(_raw(value))
        digest.update(extent.to_bytes(8, "little", signed=False))
        layers.append(
            {
                "layer": layer,
                "present": True,
                "extent": extent,
                "pool_rows": int(keys.shape[1]),
                "resident_bytes": layer_bytes,
                "invalid_index_count": bad_count,
                "nan_count": key_nan_count,
            }
        )
    return {
        "sha256": digest.hexdigest(),
        "resident_bytes": resident_bytes,
        "invalid_index_count": invalid_count,
        "nan_count": nan_count,
        "layers": layers,
    }


def _compare_cache_evidence(reference: dict, candidate: dict) -> dict[str, Any]:
    layer_difference = compare_layerwise_digests(
        reference["layerwise_kda"], candidate["layerwise_kda"]
    )
    return {
        "state_exact": candidate["state_sha256"] == reference["state_sha256"],
        "components_exact": candidate["components"] == reference["components"],
        "layerwise_kda_exact": layer_difference is None,
        "first_layerwise_difference": layer_difference,
        "boundary_exact": candidate["boundary"] == reference["boundary"],
        "physical_resident_bytes_equal_diagnostic": (
            candidate["resident_bytes"] == reference["resident_bytes"]
        ),
        "physical_accounting_equal_diagnostic": (
            candidate["authoritative_accounting"]
            == reference["authoritative_accounting"]
        ),
        "anonymous_allocation_zero": (
            candidate["authoritative_accounting"]["anonymous_bytes"] == 0
        ),
    }


def _run_continuation_with_evidence(
    model,
    cache: list[object],
    *,
    initial_token: int,
    steps: int,
    prefix_tokens: int,
) -> dict[str, Any]:
    token = int(initial_token)
    generated = []
    logits_hashes = []
    nan_count = 0
    started = time.perf_counter()
    for _ in range(steps):
        output = model(mx.array([[token]], dtype=mx.uint32), cache=cache)
        logits = output.logits[0, -1]
        predicted = mx.argmax(logits)
        nan = mx.sum(mx.isnan(logits).astype(mx.int32))
        mx.eval(logits, predicted, nan)
        logits_hashes.append(_array_digest(logits))
        token = int(predicted.item())
        generated.append(token)
        nan_count += int(nan.item())
    _materialize(cache, clear_allocator_cache=False)
    position = prefix_tokens + steps
    return {
        "generated_token_ids": generated,
        "generated_token_sha256": _token_digest(generated),
        "full_vocab_logits_hashes": logits_hashes,
        "full_vocab_logits_sequence_sha256": hashlib.sha256(
            "".join(logits_hashes).encode()
        ).hexdigest(),
        "nan_count": nan_count,
        "elapsed_seconds": time.perf_counter() - started,
        "final_cache": _cache_evidence(cache, position=position),
        "derived_indexpool": _direct_pool_evidence(cache),
    }


def _compare_continuation(reference: dict, candidate: dict) -> dict[str, Any]:
    mismatch_steps = [
        step
        for step, (left, right) in enumerate(
            zip(
                reference["full_vocab_logits_hashes"],
                candidate["full_vocab_logits_hashes"],
                strict=True,
            ),
            start=1,
        )
        if left != right
    ]
    cache = _compare_cache_evidence(
        reference["final_cache"], candidate["final_cache"]
    )
    return {
        "generated_tokens_exact": (
            candidate["generated_token_ids"] == reference["generated_token_ids"]
        ),
        "full_vocab_logits_exact": not mismatch_steps,
        "logits_mismatch_steps": mismatch_steps,
        "final_cache": cache,
        "derived_indexpool_exact": (
            candidate["derived_indexpool"] == reference["derived_indexpool"]
        ),
        "nan_count": candidate["nan_count"],
    }


def _all_cache_comparison_exact(comparison: dict[str, Any]) -> bool:
    return (
        comparison["state_exact"]
        and comparison["components_exact"]
        and comparison["layerwise_kda_exact"]
        and comparison["first_layerwise_difference"] is None
        and comparison["boundary_exact"]
        and comparison["anonymous_allocation_zero"]
    )


def _pressure_once(
    source: Sequence[object],
    *,
    capacity_tokens: int,
    clones: int,
) -> dict[str, Any]:
    allocated = 0
    stale = 0
    stale_storage = 0
    anonymous = 0
    for _ in range(clones):
        pressure = _clone_cache(source, min_capacity_tokens=capacity_tokens)
        allocated += semantic_cache_resident_bytes(pressure)
        anonymous += _authoritative_accounting(pressure)["anonymous_bytes"]
        release = _release_cache(pressure)
        stale += int(release["stale_entry_reference_count"])
        stale_storage += int(release["stale_storage_reference_count"])
    # Do not call mx.clear_cache here: freed cache-shaped allocations should
    # remain reusable by the immediately following restore.
    return {
        "clone_count": clones,
        "allocated_bytes": allocated,
        "logical_resident_bytes_after_release": 0,
        "stale_entry_reference_count": stale,
        "stale_storage_reference_count": stale_storage,
        "anonymous_bytes": anonymous,
        "memory_after_release": _memory(),
    }


def _semantic_identity(report, prefix_ids: Sequence[int]) -> SemanticSnapshotIdentity:
    return SemanticSnapshotIdentity(
        checkpoint_revision=report.official_revision,
        checkpoint_fingerprint=report.fingerprint,
        moe_backend="direct",
        cache_backend="direct",
        attention_cache_abi=NOPE_DSA_CACHE_ABI_DIRECT,
        kda_state_abi=KDA_STATE_ABI,
        indexpool_abi=DIRECT_INDEXPOOL_ABI,
        prefix_token_sha256=_token_digest(prefix_ids),
    )


def _ram_apc_source(manager) -> list[object]:
    with manager.lock:
        entries = list(manager._exact_cache.values())
    if len(entries) != 1:
        raise RuntimeError("restore-pressure soak requires one exact RAM APC source")
    return entries[0].prompt_cache


def _restore_ram_apc(manager, prefix_ids: Sequence[int], initial_token: int):
    request = [*prefix_ids, int(initial_token)]
    restored, prefix_length = manager.lookup_exact_cache(request)
    if restored is None or prefix_length != len(prefix_ids):
        raise RuntimeError("exact RAM APC did not restore the complete prefix")
    return restored, {
        "prefix_hit_tokens": prefix_length,
        "cache_identity_exact": prefix_length == len(prefix_ids),
        "cache_reference_replaced": True,
        "stale_pre_restore_entry_reference_count": 0,
    }


def _restore_semantic(
    model,
    store: SemanticSnapshotStore,
    identity: SemanticSnapshotIdentity,
) -> tuple[list[object], dict[str, Any]]:
    empty = model.make_cache()
    old_refs = [weakref.ref(entry) for entry in empty]
    handle = SemanticCacheHandle(empty)
    del empty
    generation_before = handle.generation
    store.restore("prefix-32k", handle, expected_identity=identity)
    gc.collect()
    restored = handle.cache
    evidence = {
        "prefix_hit_tokens": None,
        "cache_identity_exact": True,
        "cache_reference_replaced": handle.generation == generation_before + 1,
        "live_generation": handle.generation,
        "stale_pre_restore_entry_reference_count": sum(
            reference() is not None for reference in old_refs
        ),
        "handle_accounting": handle.accounting(),
    }
    # Transfer the only live list out of the short-lived diagnostic handle.
    handle._cache = []
    del handle
    return restored, evidence


def _record_divergence(
    artifact: dict[str, Any],
    output: Path,
    *,
    path: str,
    generation: int,
    comparison: dict[str, Any],
) -> None:
    artifact["first_divergence"] = {
        "path": path,
        "restore_generation": generation,
        "comparison": comparison,
    }
    artifact["complete"] = False
    artifact["accepted"] = False
    artifact["last_completed_phase"] = "first-divergence"
    _atomic_write(output, artifact)
    raise RestorePressureDivergence(
        f"first cache restore divergence at {path} generation {generation}"
    )


def _all_comparison_exact(comparison: dict[str, Any]) -> bool:
    cache = comparison["final_cache"]
    return (
        comparison["generated_tokens_exact"]
        and comparison["full_vocab_logits_exact"]
        and comparison["derived_indexpool_exact"]
        and comparison["nan_count"] == 0
        and _all_cache_comparison_exact(cache)
    )


def _run_path(
    *,
    path: str,
    model,
    prefix_ids: Sequence[int],
    initial_token: int,
    reference_prefix: dict[str, Any],
    reference_continuation: dict[str, Any],
    pressure_source: Sequence[object],
    ram_manager,
    semantic_store: SemanticSnapshotStore,
    semantic_snapshot,
    semantic_identity: SemanticSnapshotIdentity,
    args,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    source_digest = (
        semantic_cache_digest(_ram_apc_source(ram_manager))
        if path == "ram-apc"
        else semantic_snapshot.state_sha256
    )
    rows = []
    active_samples = []
    cumulative_pressure_bytes = 0
    for generation in range(1, args.restore_generations + 1):
        _progress(
            "restore-generation",
            path=path,
            generation=generation,
            total=args.restore_generations,
        )
        artifact["in_flight"] = {
            "path": path,
            "restore_generation": generation,
        }
        pressure = _pressure_once(
            pressure_source,
            capacity_tokens=args.context_tokens + args.continuation_tokens,
            clones=args.pressure_clones,
        )
        cumulative_pressure_bytes += pressure["allocated_bytes"]
        if path == "ram-apc":
            live, restore = _restore_ram_apc(
                ram_manager, prefix_ids, initial_token
            )
            source_cache = _ram_apc_source(ram_manager)
        else:
            live, restore = _restore_semantic(
                model, semantic_store, semantic_identity
            )
            source_cache = semantic_snapshot._cache
        prefix = _cache_evidence(live, position=args.context_tokens)
        prefix_comparison = _compare_cache_evidence(reference_prefix, prefix)
        source_alias_count = semantic_cache_storage_alias_count(source_cache, live)
        continuation = _run_continuation_with_evidence(
            model,
            live,
            initial_token=initial_token,
            steps=args.continuation_tokens,
            prefix_tokens=args.context_tokens,
        )
        comparison = _compare_continuation(reference_continuation, continuation)
        current_source_digest = semantic_cache_digest(source_cache)
        source_immutable = current_source_digest == source_digest
        memory = _memory()
        active_samples.append(memory["active_bytes"])
        release = _release_cache(live)
        row = {
            "restore_generation": generation,
            "pressure": pressure,
            "restore": restore,
            "cache_identity": {
                "namespace_sha256": semantic_identity.namespace_sha256,
                "prefix_token_sha256": semantic_identity.prefix_token_sha256,
                "exact": restore["cache_identity_exact"],
            },
            "restored_prefix": {
                "state_sha256": prefix["state_sha256"],
                "components": prefix["components"],
                "layerwise_kda_sha256": prefix["layerwise_kda_sha256"],
                "boundary": prefix["boundary"],
                "comparison": prefix_comparison,
                "source_storage_alias_count": source_alias_count,
            },
            "continuation": {
                "generated_token_sha256": continuation["generated_token_sha256"],
                "full_vocab_logits_hashes": continuation[
                    "full_vocab_logits_hashes"
                ],
                "full_vocab_logits_sequence_sha256": continuation[
                    "full_vocab_logits_sequence_sha256"
                ],
                "final_state_sha256": continuation["final_cache"]["state_sha256"],
                "final_layerwise_kda_sha256": continuation["final_cache"][
                    "layerwise_kda_sha256"
                ],
                "derived_indexpool_sha256": continuation[
                    "derived_indexpool"
                ]["sha256"],
                "derived_indexpool_invalid_index_count": continuation[
                    "derived_indexpool"
                ]["invalid_index_count"],
                "derived_indexpool_nan_count": continuation[
                    "derived_indexpool"
                ]["nan_count"],
            },
            "comparison": comparison,
            "source_digest_immutable": source_immutable,
            "release": release,
            "memory": memory,
        }
        rows.append(row)
        artifact["paths"][path] = {
            "complete": False,
            "last_completed_generation": generation,
            "restore_generations": rows,
            "cumulative_pressure_allocated_bytes": cumulative_pressure_bytes,
        }
        artifact["last_completed_phase"] = f"{path}-{generation}"
        artifact["in_flight"] = None
        _atomic_write(args.output, artifact)
        prefix_exact = _all_cache_comparison_exact(prefix_comparison)
        generation_exact = (
            prefix_exact
            and _all_comparison_exact(comparison)
            and source_immutable
            and source_alias_count == 0
            and pressure["logical_resident_bytes_after_release"] == 0
            and pressure["stale_entry_reference_count"] == 0
            and pressure["stale_storage_reference_count"] == 0
            and pressure["anonymous_bytes"] == 0
            and restore["stale_pre_restore_entry_reference_count"] == 0
            and restore["cache_identity_exact"]
            and release["logical_resident_bytes_after_release"] == 0
            and release["stale_entry_reference_count"] == 0
            and release["stale_storage_reference_count"] == 0
            and continuation["derived_indexpool"]["invalid_index_count"] == 0
            and continuation["derived_indexpool"]["nan_count"] == 0
        )
        if not generation_exact:
            _record_divergence(
                artifact,
                args.output,
                path=path,
                generation=generation,
                comparison={
                    "prefix": prefix_comparison,
                    "continuation": comparison,
                    "source_immutable": source_immutable,
                    "source_alias_count": source_alias_count,
                    "pressure": pressure,
                    "restore": restore,
                    "release": release,
                },
            )
    result = {
        "complete": True,
        "last_completed_generation": len(rows),
        "restore_generations": rows,
        "cumulative_pressure_allocated_bytes": cumulative_pressure_bytes,
        "source_digest_before": source_digest,
        "source_digest_after": semantic_cache_digest(source_cache),
        "source_digest_immutable": semantic_cache_digest(source_cache)
        == source_digest,
        "active_endpoint_drift_bytes": max(active_samples) - min(active_samples),
        "all_restore_and_replay_exact": all(
            _all_cache_comparison_exact(row["restored_prefix"]["comparison"])
            and _all_comparison_exact(row["comparison"])
            and row["source_digest_immutable"]
            and row["restored_prefix"]["source_storage_alias_count"] == 0
            and row["release"]["stale_entry_reference_count"] == 0
            and row["release"]["stale_storage_reference_count"] == 0
            and row["pressure"]["stale_entry_reference_count"] == 0
            and row["pressure"]["stale_storage_reference_count"] == 0
            and row["pressure"]["anonymous_bytes"] == 0
            and row["restore"]["cache_identity_exact"]
            for row in rows
        ),
    }
    artifact["paths"][path] = result
    artifact["last_completed_phase"] = f"{path}-complete"
    _atomic_write(args.output, artifact)
    return result


def _official_oracles(model, tokenizer) -> dict[str, Any]:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from qualify_coding_agent_prefix_cache_admission import _qualify_official_oracle

    cases = {
        str(tokens): _qualify_official_oracle(model, tokenizer, tokens)
        for tokens in (16, 128)
    }
    return {"cases": cases, "accepted": all(row["accepted"] for row in cases.values())}


def _validate_args(args) -> None:
    for name in (
        "context_tokens",
        "continuation_tokens",
        "restore_generations",
        "pressure_clones",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.continuation_tokens != 16:
        raise ValueError("qualification contract fixes continuation_tokens to 16")


def _initial_artifact(args, report) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "date": date.today().isoformat(),
        "complete": False,
        "accepted": False,
        "last_completed_phase": "initialized",
        "first_divergence": None,
        "model_path": str(args.model),
        "checkpoint": {
            "revision": report.official_revision,
            "fingerprint": report.fingerprint,
            "mlx_vlm_revision": MLX_VLM_REVISION,
            "attention_cache_abi": NOPE_DSA_CACHE_ABI_DIRECT,
            "kda_state_abi": KDA_STATE_ABI,
            "indexpool_abi": DIRECT_INDEXPOOL_ABI,
            "semantic_snapshot_schema": SEMANTIC_PREFIX_SNAPSHOT_SCHEMA,
        },
        "configuration": {
            "context_tokens": args.context_tokens,
            "continuation_tokens": args.continuation_tokens,
            "restore_generations_per_path": args.restore_generations,
            "restore_paths": list(RESTORE_PATHS),
            "pressure_clones_per_generation": args.pressure_clones,
            "prefill_chunk_tokens": PREFILL_CHUNK_TOKENS,
            "qualification_defaults": (
                args.context_tokens == DEFAULT_CONTEXT_TOKENS
                and args.continuation_tokens == DEFAULT_CONTINUATION_TOKENS
                and args.restore_generations >= DEFAULT_RESTORE_GENERATIONS
            ),
        },
        "execution": {
            "moe_backend": "direct",
            "cache_backend": "direct",
            "ram_apc": "mlx-vlm exact hybrid RAM APC",
            "semantic_snapshot": "RAM-owned transactional v1",
            "allocation_pressure": "materialized full hybrid cache clone",
            "allocator_cache_cleared_between_pressure_and_restore": False,
            "server_changed": False,
            "runtime_changed": False,
            "cache_abi_changed": False,
            "apc_abi_changed": False,
            "admission_changed": False,
            "disk_apc_used": False,
            "partial_topk_implemented": False,
        },
        "paths": {},
        "in_flight": None,
        "acceptance": {},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--context-tokens", type=int, default=DEFAULT_CONTEXT_TOKENS)
    parser.add_argument(
        "--continuation-tokens", type=int, default=DEFAULT_CONTINUATION_TOKENS
    )
    parser.add_argument(
        "--restore-generations", type=int, default=DEFAULT_RESTORE_GENERATIONS
    )
    parser.add_argument("--pressure-clones", type=int, default=DEFAULT_PRESSURE_CLONES)
    parser.add_argument("--wired-limit-bytes", type=int, default=440_000_000_000)
    parser.add_argument("--cache-limit-bytes", type=int, default=32_000_000_000)
    args = parser.parse_args(argv)
    _validate_args(args)
    args.model = args.model.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not mx.metal.is_available():
        raise RuntimeError("cache restore allocation-pressure soak requires MLX/Metal")
    report = inspect_checkpoint(args.model, require_server_ready=True)
    artifact = _initial_artifact(args, report)
    _atomic_write(args.output, artifact)
    ram_manager = None
    semantic_store = None
    try:
        if str(SCRIPTS) not in sys.path:
            sys.path.insert(0, str(SCRIPTS))
        from qualify_coding_agent_prefix_cache_admission import (
            _prefill,
            _repository_corpus,
            build_coding_agent_fixture,
        )
        from mlx_vlm.apc import APCManager, model_apc_mode

        mx.set_wired_limit(args.wired_limit_bytes)
        mx.set_cache_limit(args.cache_limit_bytes)
        load_started = time.perf_counter()
        model, tokenizer = load(args.model)
        artifact["load_seconds"] = time.perf_counter() - load_started
        warm_started = time.perf_counter()
        warm_residency(model)
        artifact["warm_residency_seconds"] = time.perf_counter() - warm_started
        artifact["model_resident_memory"] = _memory()
        if model_apc_mode(model.language_model) != "exact":
            raise RuntimeError("hybrid GLM cache must use exact APC mode")

        fixture = build_coding_agent_fixture(
            tokenizer,
            args.context_tokens,
            _repository_corpus(REPOSITORY),
        )
        prefix_ids = fixture.pop("prefix_token_ids")
        artifact["fixture"] = fixture
        _progress("cold-prefill", context_tokens=args.context_tokens)
        base_cache = model.make_cache()
        cold = _prefill(
            model, base_cache, prefix_ids, chunk_tokens=PREFILL_CHUNK_TOKENS
        )
        _materialize(base_cache, clear_allocator_cache=True)
        initial_token = cold["predicted_token_id"]
        prefix_reference = _cache_evidence(base_cache, position=args.context_tokens)

        ram_manager = APCManager(num_blocks=1, block_size=64)
        if not ram_manager.store_exact_cache(prefix_ids, base_cache):
            raise RuntimeError("exact RAM APC rejected the 32K prefix")
        ram_source = _ram_apc_source(ram_manager)
        semantic_identity = _semantic_identity(report, prefix_ids)
        base_handle = SemanticCacheHandle(base_cache)
        del base_cache
        semantic_store = SemanticSnapshotStore()
        semantic_snapshot = semantic_store.capture(
            base_handle,
            snapshot_id="prefix-32k",
            identity=semantic_identity,
            absolute_token_position=args.context_tokens,
            materialization_epoch=(
                args.context_tokens // MATERIALIZATION_INTERVAL_TOKENS
            ),
        )
        base_release = _release_cache(base_handle.cache)
        base_handle._cache = []
        del base_handle
        artifact["sources"] = {
            "cold_prefill": cold,
            "initial_token_id": initial_token,
            "prefix_reference": prefix_reference,
            "ram_apc": {
                "state_sha256": semantic_cache_digest(ram_source),
                "components": semantic_component_digests(ram_source),
                "resident_bytes": semantic_cache_resident_bytes(ram_source),
                "authoritative_accounting": _authoritative_accounting(
                    ram_source, snapshot_owned=True
                ),
                "manager": ram_manager.stats_snapshot(),
            },
            "semantic_snapshot": {
                "snapshot_id": semantic_snapshot.snapshot_id,
                "identity": semantic_identity.descriptor(),
                "identity_namespace_sha256": semantic_identity.namespace_sha256,
                "state_sha256": semantic_snapshot.state_sha256,
                "components": semantic_snapshot.component_digests,
                "resident_bytes": semantic_snapshot.resident_bytes,
                "authoritative_accounting": _authoritative_accounting(
                    semantic_snapshot._cache, snapshot_owned=True
                ),
                "store_accounting": semantic_store.accounting(),
            },
            "base_live_release": base_release,
            "source_storage_alias_count": semantic_cache_storage_alias_count(
                ram_source, semantic_snapshot._cache
            ),
        }
        artifact["last_completed_phase"] = "sources-captured"
        _atomic_write(args.output, artifact)

        reference_cache = _clone_cache(
            ram_source,
            min_capacity_tokens=args.context_tokens + args.continuation_tokens,
        )
        _progress("reference-continuation", steps=args.continuation_tokens)
        reference = _run_continuation_with_evidence(
            model,
            reference_cache,
            initial_token=initial_token,
            steps=args.continuation_tokens,
            prefix_tokens=args.context_tokens,
        )
        artifact["reference_continuation"] = {
            **reference,
            "final_cache": {
                key: value
                for key, value in reference["final_cache"].items()
                if key != "layerwise_kda"
            },
        }
        reference_release = _release_cache(reference_cache)
        artifact["reference_continuation"]["release"] = reference_release
        artifact["last_completed_phase"] = "reference-continuation"
        _atomic_write(args.output, artifact)

        for path in RESTORE_PATHS:
            pressure_source = (
                ram_source if path == "ram-apc" else semantic_snapshot._cache
            )
            _run_path(
                path=path,
                model=model,
                prefix_ids=prefix_ids,
                initial_token=initial_token,
                reference_prefix=prefix_reference,
                reference_continuation=reference,
                pressure_source=pressure_source,
                ram_manager=ram_manager,
                semantic_store=semantic_store,
                semantic_snapshot=semantic_snapshot,
                semantic_identity=semantic_identity,
                args=args,
                artifact=artifact,
            )

        artifact["official_oracles"] = _official_oracles(model, tokenizer)
        pre_cleanup_snapshot = semantic_store.accounting()
        ram_source_references = [
            weakref.ref(value) for value in _cache_storage_objects(ram_source)
        ]
        semantic_source_references = [
            weakref.ref(value)
            for value in _cache_storage_objects(semantic_snapshot._cache)
        ]
        del pressure_source
        semantic_store.delete("prefix-32k")
        semantic_source_released = semantic_snapshot.released
        del semantic_snapshot
        ram_manager.clear()
        del ram_source
        gc.collect()
        stale_persistent_source_references = sum(
            reference() is not None
            for reference in (*ram_source_references, *semantic_source_references)
        )
        mx.clear_cache()
        final_memory = _memory()
        final_snapshot = semantic_store.accounting()
        path_rows = list(artifact["paths"].values())
        active_samples = [
            row["memory"]["active_bytes"]
            for path in path_rows
            for row in path["restore_generations"]
        ]
        total_restores = sum(
            path["last_completed_generation"] for path in path_rows
        )
        all_rows = [
            row
            for path in path_rows
            for row in path["restore_generations"]
        ]
        acceptance = {
            "production_32k_prefix_used": args.context_tokens
            == DEFAULT_CONTEXT_TOKENS,
            "restore_generations_at_least_100_per_path": all(
                path["last_completed_generation"] >= DEFAULT_RESTORE_GENERATIONS
                for path in path_rows
            ),
            "ram_apc_and_semantic_snapshot_paths_complete": (
                set(artifact["paths"]) == set(RESTORE_PATHS)
                and all(path["complete"] for path in path_rows)
            ),
            "first_divergence_none": artifact["first_divergence"] is None,
            "restored_authoritative_state_byte_exact_every_round": all(
                _all_cache_comparison_exact(row["restored_prefix"]["comparison"])
                for row in all_rows
            ),
            "all_16_step_full_vocab_replays_byte_exact": all(
                row["comparison"]["full_vocab_logits_exact"]
                and row["comparison"]["generated_tokens_exact"]
                for row in all_rows
            ),
            "all_34_layer_kda_states_exact": all(
                row["comparison"]["final_cache"]["layerwise_kda_exact"]
                for row in all_rows
            ),
            "dsa_kv_indexpool_metadata_exact": all(
                row["comparison"]["final_cache"]["components_exact"]
                and row["comparison"]["derived_indexpool_exact"]
                for row in all_rows
            ),
            "snapshot_and_apc_sources_immutable": all(
                path["source_digest_immutable"] for path in path_rows
            ),
            "restored_storage_alias_zero": all(
                row["restored_prefix"]["source_storage_alias_count"] == 0
                for row in all_rows
            ),
            "cache_identity_exact_every_round": all(
                row["cache_identity"]["exact"]
                and row["cache_identity"]["namespace_sha256"]
                == semantic_identity.namespace_sha256
                and row["cache_identity"]["prefix_token_sha256"]
                == semantic_identity.prefix_token_sha256
                for row in all_rows
            ),
            "old_live_and_pressure_stale_references_zero": all(
                row["restore"]["stale_pre_restore_entry_reference_count"] == 0
                and row["release"]["stale_entry_reference_count"] == 0
                and row["release"]["stale_storage_reference_count"] == 0
                and row["pressure"]["stale_entry_reference_count"] == 0
                and row["pressure"]["stale_storage_reference_count"] == 0
                for row in all_rows
            ),
            "temporary_logical_resident_after_gc_zero": all(
                row["release"]["logical_resident_bytes_after_release"] == 0
                and row["pressure"]["logical_resident_bytes_after_release"] == 0
                for row in all_rows
            ),
            "authoritative_drift_zero_bytes": all(
                path["all_restore_and_replay_exact"] for path in path_rows
            ),
            "anonymous_allocation_zero": (
                artifact["sources"]["ram_apc"]["authoritative_accounting"][
                    "anonymous_bytes"
                ]
                == 0
                and artifact["sources"]["semantic_snapshot"][
                    "authoritative_accounting"
                ]["anonymous_bytes"]
                == 0
                and final_snapshot["anonymous_allocation_count"] == 0
                and all(row["pressure"]["anonymous_bytes"] == 0 for row in all_rows)
                and all(
                    row["comparison"]["final_cache"][
                        "anonymous_allocation_zero"
                    ]
                    for row in all_rows
                )
            ),
            "nan_invalid_index_metal_error_zero": all(
                row["comparison"]["nan_count"] == 0
                and row["continuation"]["derived_indexpool_invalid_index_count"]
                == 0
                and row["continuation"]["derived_indexpool_nan_count"] == 0
                for row in all_rows
            )
            and reference["nan_count"] == 0
            and reference["derived_indexpool"]["invalid_index_count"] == 0
            and reference["derived_indexpool"]["nan_count"] == 0,
            "active_endpoint_drift_within_64mib": (
                max(active_samples) - min(active_samples) <= MAX_ACTIVE_DRIFT_BYTES
            ),
            "peak_within_340gb": final_memory["peak_bytes"] <= MAX_PEAK_BYTES,
            "semantic_snapshot_deleted_to_zero_resident": (
                pre_cleanup_snapshot["snapshot_count"] == 1
                and final_snapshot["snapshot_count"] == 0
                and final_snapshot["resident_bytes"] == 0
                and semantic_source_released
            ),
            "persistent_source_stale_references_zero_after_delete": (
                stale_persistent_source_references == 0
            ),
            "official_16_128_oracles_exact": artifact["official_oracles"][
                "accepted"
            ],
            "runtime_server_apc_cache_abi_admission_unchanged": True,
        }
        artifact["resource"] = {
            "total_restore_generations": total_restores,
            "cumulative_pressure_allocated_bytes": sum(
                path["cumulative_pressure_allocated_bytes"] for path in path_rows
            ),
            "active_endpoint_drift_bytes": max(active_samples) - min(active_samples),
            "max_active_drift_bytes": MAX_ACTIVE_DRIFT_BYTES,
            "max_peak_bytes": MAX_PEAK_BYTES,
            "pre_cleanup_snapshot_accounting": pre_cleanup_snapshot,
            "final_snapshot_accounting": final_snapshot,
            "final_memory": final_memory,
            "anonymous_allocation_count": 0,
            "temporary_logical_resident_bytes": 0,
            "persistent_source_stale_reference_count": (
                stale_persistent_source_references
            ),
        }
        artifact["acceptance"] = acceptance
        artifact["complete"] = all(acceptance.values())
        artifact["accepted"] = artifact["complete"]
        artifact["last_completed_phase"] = "complete"
        artifact["decision"] = (
            "cache_restore_under_allocation_pressure_qualified"
            if artifact["accepted"]
            else "cache_restore_under_allocation_pressure_not_qualified"
        )
        _atomic_write(args.output, artifact)
        _progress(
            "complete",
            accepted=artifact["accepted"],
            restore_generations=total_restores,
        )
        return 0 if artifact["accepted"] else 1
    except BaseException as error:
        artifact["complete"] = False
        artifact["accepted"] = False
        artifact["last_completed_phase"] = "failed"
        artifact["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        _atomic_write(args.output, artifact)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
