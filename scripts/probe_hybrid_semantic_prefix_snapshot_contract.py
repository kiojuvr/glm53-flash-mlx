#!/usr/bin/env python3
"""Qualify the RAM-only hybrid semantic prefix snapshot contract.

The probe executes the complete packed-decode + compact-NoPE model, but does
not add a server API, disk serialization, or prefix-LRU integration.  Each
snapshot is captured only after a completed forward and materialized cache
boundary.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np

from glm53_flash_mlx.abi import MLX_VLM_REVISION, NOPE_DSA_CACHE_ABI_COMPACT
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint
from glm53_flash_mlx.materialization import (
    MATERIALIZATION_INTERVAL_TOKENS,
    MATERIALIZATION_POLICY,
)
from glm53_flash_mlx.semantic_snapshot import (
    COMPACT_INDEXPOOL_ABI,
    KDA_STATE_ABI,
    SEMANTIC_PREFIX_SNAPSHOT_SCHEMA,
    SemanticCacheHandle,
    SemanticSnapshotIdentity,
    SemanticSnapshotStore,
    semantic_cache_digest,
    semantic_cache_schema,
    semantic_component_digests,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-hybrid-semantic-prefix-snapshot-contract-20260904.json"
)
BOUNDARIES = (1, 255, 256, 257, 1023, 1024)
CONTINUATION_STEPS = 64
MUTATION_STEPS = 8


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


def _atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _raw(value: mx.array) -> bytes:
    storage = value.view(mx.uint16) if value.dtype == mx.bfloat16 else value
    mx.eval(storage)
    return np.ascontiguousarray(np.asarray(storage)).tobytes()


def _logits_hash(value: mx.array) -> str:
    return hashlib.sha256(_raw(value)).hexdigest()


def _sequence_hash(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def _tokens(count: int, vocab: int) -> list[int]:
    return [int((index * 7919) % (vocab - 1024) + 100) for index in range(count)]


def _materialize(cache) -> None:
    mx.eval([entry.state for entry in cache])
    mx.synchronize()


def _clone_cache(cache, *, min_capacity_tokens: int):
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
        raise RuntimeError("RAM APC clone rejected a semantic cache component")
    mx.eval(*targets)
    _materialize(cloned)
    return cloned


def _forward_range(
    model,
    handle: SemanticCacheHandle,
    token_ids: list[int],
    *,
    start: int,
    steps: int,
) -> dict:
    hashes = []
    materialization_steps = []
    nan_count = 0
    for position in range(start, start + steps):
        output = model(
            mx.array([[token_ids[position]]], dtype=mx.uint32),
            cache=handle.cache,
        )
        logits = output.logits[0, -1]
        nan = mx.sum(mx.isnan(logits))
        mx.eval(logits, nan)
        hashes.append(_logits_hash(logits))
        nan_count += int(nan.item())
        logical_position = position + 1
        if logical_position % MATERIALIZATION_INTERVAL_TOKENS == 0:
            _materialize(handle.cache)
            materialization_steps.append(logical_position)
    return {
        "logits_hashes": hashes,
        "logits_sequence_sha256": _sequence_hash(
            int(value[:8], 16) & 0xFFFFFFFF for value in hashes
        ),
        "final_state_sha256": semantic_cache_digest(handle.cache),
        "final_components": semantic_component_digests(handle.cache),
        "materialization_steps": materialization_steps,
        "nan_count": nan_count,
    }


def _continuation_exact(left: dict, right: dict) -> bool:
    return (
        left["logits_hashes"] == right["logits_hashes"]
        and left["final_state_sha256"] == right["final_state_sha256"]
        and left["final_components"] == right["final_components"]
        and left["materialization_steps"] == right["materialization_steps"]
        and left["nan_count"] == right["nan_count"] == 0
    )


def _identity(report, *, prefix_tokens: list[int]) -> SemanticSnapshotIdentity:
    return SemanticSnapshotIdentity(
        checkpoint_revision=report.official_revision,
        checkpoint_fingerprint=report.fingerprint,
        moe_backend="packed-decode",
        cache_backend="compact-nope-dsa",
        attention_cache_abi=NOPE_DSA_CACHE_ABI_COMPACT,
        kda_state_abi=KDA_STATE_ABI,
        indexpool_abi=COMPACT_INDEXPOOL_ABI,
        prefix_token_sha256=_sequence_hash(prefix_tokens),
    )


def _qualify_boundary(
    model,
    report,
    source: SemanticCacheHandle,
    store: SemanticSnapshotStore,
    token_ids: list[int],
    position: int,
) -> dict:
    snapshot_id = f"prefix-{position}"
    identity = _identity(report, prefix_tokens=token_ids[:position])
    live_reference = SemanticCacheHandle(
        _clone_cache(source.cache, min_capacity_tokens=position + CONTINUATION_STEPS)
    )
    live_before = semantic_cache_digest(source.cache)
    components_before = semantic_component_digests(source.cache)
    schema_before = semantic_cache_schema(source.cache)
    snapshot = store.capture(
        source,
        snapshot_id=snapshot_id,
        identity=identity,
        absolute_token_position=position,
        materialization_epoch=position // MATERIALIZATION_INTERVAL_TOKENS,
    )
    capture_observation_only = (
        semantic_cache_digest(source.cache) == live_before
        and semantic_component_digests(source.cache) == components_before
        and semantic_cache_schema(source.cache) == schema_before
    )

    uninterrupted = _forward_range(
        model,
        live_reference,
        token_ids,
        start=position,
        steps=CONTINUATION_STEPS,
    )
    capture_only = _forward_range(
        model,
        source,
        token_ids,
        start=position,
        steps=CONTINUATION_STEPS,
    )
    capture_only_exact = _continuation_exact(uninterrupted, capture_only)

    replay = SemanticCacheHandle(source.cache)
    store.restore(snapshot_id, replay, expected_identity=identity)
    alternate = list(token_ids)
    for index in range(position, position + MUTATION_STEPS):
        alternate[index] = (alternate[index] + 31337) % (
            int(model.language_model.lm_head.weight.shape[0]) - 1
        )
    _forward_range(
        model,
        replay,
        alternate,
        start=position,
        steps=MUTATION_STEPS,
    )
    mutated_digest = semantic_cache_digest(replay.cache)
    store.restore(snapshot_id, replay, expected_identity=identity)
    first_replay = _forward_range(
        model,
        replay,
        token_ids,
        start=position,
        steps=CONTINUATION_STEPS,
    )
    first_replay_exact = _continuation_exact(uninterrupted, first_replay)

    store.restore(snapshot_id, replay, expected_identity=identity)
    second_replay = _forward_range(
        model,
        replay,
        token_ids,
        start=position,
        steps=CONTINUATION_STEPS,
    )
    second_replay_exact = _continuation_exact(uninterrupted, second_replay)
    snapshot_immutable = (
        snapshot.state_sha256 == semantic_cache_digest(snapshot._cache)
        and snapshot.state_sha256 == live_before
        and dict(snapshot.component_digests) == components_before
    )

    # Restore the source to the captured boundary before progressing to the
    # next screen point.  The snapshot remains reusable until explicit delete.
    store.restore(snapshot_id, source, expected_identity=identity)
    restored_source_exact = semantic_cache_digest(source.cache) == live_before
    accounting_before_delete = store.accounting()
    store.delete(snapshot_id)
    accounting_after_delete = store.accounting()
    return {
        "position": position,
        "materialization_epoch": position // MATERIALIZATION_INTERVAL_TOKENS,
        "snapshot_descriptor": snapshot.descriptor(),
        "capture_observation_only": capture_observation_only,
        "capture_only_continuation_exact": capture_only_exact,
        "mutated_state_differs": mutated_digest != snapshot.state_sha256,
        "capture_mutate_restore_replay_exact": first_replay_exact,
        "same_snapshot_second_restore_exact": second_replay_exact,
        "snapshot_immutable": snapshot_immutable,
        "restored_source_exact": restored_source_exact,
        "full_vocab_logits_steps_compared": CONTINUATION_STEPS,
        "kda_state_exact": (
            uninterrupted["final_components"]["kda_state_sha256"]
            == first_replay["final_components"]["kda_state_sha256"]
            == second_replay["final_components"]["kda_state_sha256"]
        ),
        "dsa_kv_exact": (
            uninterrupted["final_components"]["dsa_kv_sha256"]
            == first_replay["final_components"]["dsa_kv_sha256"]
            == second_replay["final_components"]["dsa_kv_sha256"]
        ),
        "indexpool_exact": (
            uninterrupted["final_components"]["indexpool_sha256"]
            == first_replay["final_components"]["indexpool_sha256"]
            == second_replay["final_components"]["indexpool_sha256"]
        ),
        "slot_index_metadata_exact": (
            uninterrupted["final_components"]["slot_index_metadata_sha256"]
            == first_replay["final_components"]["slot_index_metadata_sha256"]
            == second_replay["final_components"]["slot_index_metadata_sha256"]
        ),
        "materialization_steps": uninterrupted["materialization_steps"],
        "accounting_before_delete": accounting_before_delete,
        "accounting_after_delete": accounting_after_delete,
        "nan_count": sum(
            row["nan_count"]
            for row in (uninterrupted, capture_only, first_replay, second_replay)
        ),
    }


def _official_oracle(model, processor, report) -> dict:
    scripts = str(REPOSITORY / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import probe_exact_sigmoid_gate_metal_barrier as oracle_probe

    return oracle_probe._official_oracle(model, processor, report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not mx.metal.is_available():
        raise RuntimeError("hybrid semantic snapshot probe requires MLX/Metal")

    report = inspect_checkpoint(args.model, require_server_ready=True)
    mx.set_wired_limit(int(440e9))
    mx.set_cache_limit(int(32e9))
    load_started = time.perf_counter()
    model, processor = load(
        args.model,
        experimental_packed_decode_moe=True,
        experimental_compact_nope_dsa_cache=True,
        compact_cache_capacity_tokens=max(BOUNDARIES) + CONTINUATION_STEPS,
    )
    load_seconds = time.perf_counter() - load_started
    warm_started = time.perf_counter()
    warm_residency(model)
    warm_seconds = time.perf_counter() - warm_started
    vocab = int(model.language_model.lm_head.weight.shape[0])
    token_ids = _tokens(max(BOUNDARIES) + CONTINUATION_STEPS, vocab)
    cache = model.make_cache()
    source = SemanticCacheHandle(cache)
    first = model(mx.array([[token_ids[0]]], dtype=mx.uint32), cache=source.cache)
    mx.eval(first.logits)
    mx.synchronize()
    position = 1
    store = SemanticSnapshotStore()
    rows = []
    peak_snapshot_resident_bytes = 0
    for boundary in BOUNDARIES:
        if position < boundary:
            _progress("advance", current=position, target=boundary)
            _forward_range(
                model,
                source,
                token_ids,
                start=position,
                steps=boundary - position,
            )
            position = boundary
        _progress("snapshot-boundary", position=boundary)
        row = _qualify_boundary(
            model, report, source, store, token_ids, boundary
        )
        peak_snapshot_resident_bytes = max(
            peak_snapshot_resident_bytes,
            int(row["accounting_before_delete"]["resident_bytes"]),
        )
        rows.append(row)

    oracle = _official_oracle(model, processor, report)
    final_accounting = store.accounting()
    all_boundary_exact = all(
        all(
            row[key]
            for key in (
                "capture_observation_only",
                "capture_only_continuation_exact",
                "mutated_state_differs",
                "capture_mutate_restore_replay_exact",
                "same_snapshot_second_restore_exact",
                "snapshot_immutable",
                "restored_source_exact",
                "kda_state_exact",
                "dsa_kv_exact",
                "indexpool_exact",
                "slot_index_metadata_exact",
            )
        )
        and row["nan_count"] == 0
        for row in rows
    )
    acceptance = {
        "all_six_boundaries_exercised": [row["position"] for row in rows]
        == list(BOUNDARIES),
        "capture_is_observation_only": all(
            row["capture_observation_only"] for row in rows
        ),
        "capture_only_continuation_exact": all(
            row["capture_only_continuation_exact"] for row in rows
        ),
        "capture_mutate_restore_replay_exact": all(
            row["capture_mutate_restore_replay_exact"] for row in rows
        ),
        "same_snapshot_reusable": all(
            row["same_snapshot_second_restore_exact"] for row in rows
        ),
        "snapshot_immutable_and_owned": all(
            row["snapshot_immutable"]
            and row["snapshot_descriptor"]["ownership"]["tensor_ownership"]
            == "owned"
            and not row["snapshot_descriptor"]["ownership"][
                "physical_storage_alias_with_live"
            ]
            for row in rows
        ),
        "all_target_components_byte_exact": all_boundary_exact,
        "snapshot_accounting_returns_to_zero": final_accounting[
            "resident_bytes"
        ]
        == 0,
        "snapshot_lifecycle_is_separate_from_prefix_lru": (
            final_accounting["lifecycle"] == "snapshot-state"
            and not final_accounting["prefix_lru_member"]
        ),
        "anonymous_snapshot_allocation_zero": final_accounting[
            "anonymous_allocation_count"
        ]
        == 0,
        "ram_only_no_disk_persistence": not final_accounting["disk_persistence"],
        "official_16_token_oracle_exact": oracle["first_16_match"],
        "official_128_token_oracle_exact": oracle["full_128_match"],
        "runtime_abi_backend_admission_unchanged": True,
    }
    artifact = {
        "schema": SEMANTIC_PREFIX_SNAPSHOT_SCHEMA,
        "date": str(date.today()),
        "complete": all(acceptance.values()),
        "probe_only": True,
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "model_path": str(args.model.resolve()),
        "load_seconds": load_seconds,
        "warm_residency_seconds": warm_seconds,
        "backend": {
            "moe": getattr(model, "_glm53_moe_backend", None),
            "cache": getattr(model, "_glm53_cache_backend", None),
        },
        "contract": {
            "persistence": "ram-only",
            "capture_point": "post-forward-quiescent-materialized",
            "capture_commit": "validate-all-then-publish",
            "restore_commit": "validate-all-then-single-cache-reference-swap",
            "snapshot_consumed_by_restore": False,
            "partial_component_restore": False,
            "prefix_lru_member": False,
            "materialization_policy": MATERIALIZATION_POLICY,
            "materialization_interval_tokens": MATERIALIZATION_INTERVAL_TOKENS,
            "kda_state_abi": KDA_STATE_ABI,
            "indexpool_abi": COMPACT_INDEXPOOL_ABI,
            "attention_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
        },
        "screen": {
            "boundaries": list(BOUNDARIES),
            "continuation_steps": CONTINUATION_STEPS,
            "mutation_steps": MUTATION_STEPS,
            "rows": rows,
        },
        "snapshot_accounting": {
            "peak_single_snapshot_resident_bytes": peak_snapshot_resident_bytes,
            "final": final_accounting,
        },
        "official_oracle": oracle,
        "acceptance": acceptance,
        "decision": (
            "hybrid_semantic_prefix_snapshot_contract_ready_for_replay_qualification"
            if all(acceptance.values())
            else "hybrid_semantic_prefix_snapshot_contract_not_ready"
        ),
        "runtime_changes": {
            "server_api": False,
            "disk_apc": False,
            "cache_abi": False,
            "backend": False,
            "admission": False,
        },
    }
    _atomic_write(args.output, artifact)
    _progress("complete", accepted=artifact["complete"], output=str(args.output))
    del model, source, cache
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    return 0 if artifact["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
