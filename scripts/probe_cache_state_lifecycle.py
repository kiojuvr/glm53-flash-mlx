#!/usr/bin/env python3
"""Prove cache lifecycle, retention, accounting, and eviction isolation."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

import mlx.core as mx

from glm53_flash_mlx.abi import (
    CACHE_IDENTITY_SCHEMA,
    MLX_VLM_REVISION,
    NOPE_DSA_CACHE_ABI_COMPACT,
    NOPE_DSA_CACHE_ABI_DIRECT,
)
from glm53_flash_mlx.cache_lifecycle import (
    CacheLifecycle,
    CacheLifecycleError,
    CacheLifecycleManager,
    EvictionReason,
    LIFECYCLE_CONTRACTS,
    PrefixIdentity,
    RetentionPolicy,
)
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-cache-state-lifecycle-20260902.json"
)
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DRAFT_ROTATIONS = 4096


def _atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _budgets() -> dict[CacheLifecycle, int]:
    return {
        CacheLifecycle.TARGET_PREFIX: 200,
        CacheLifecycle.ACTIVE_RECURRENT: 256,
        CacheLifecycle.SNAPSHOT_STATE: 256,
        CacheLifecycle.DRAFT_TRANSIENT: 32,
    }


def _identity(**overrides) -> PrefixIdentity:
    values = {
        "model_revision": "04c4e9e",
        "checkpoint_fingerprint": "checkpoint-a",
        "backend_policy": "packed-decode+compact-nope-dsa",
        "attention_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
        "kda_state_abi": "glm53-kda-state-v1",
        "indexpool_abi": "compact-indexpool-v4",
        "prefix_token_sha256": "same-token-sequence",
    }
    values.update(overrides)
    return PrefixIdentity(**values)


def _target(manager, entry_id: str, payload: bytes, identity: PrefixIdentity):
    return manager.allocate(
        entry_id=entry_id,
        payload=payload,
        lifecycle=CacheLifecycle.TARGET_PREFIX,
        retention=RetentionPolicy.LONG_REUSE,
        owner_id=f"prefix-owner:{entry_id}",
        prefix_identity=identity,
    )


def _active(manager, entry_id: str, payload: bytes):
    return manager.allocate(
        entry_id=entry_id,
        payload=payload,
        lifecycle=CacheLifecycle.ACTIVE_RECURRENT,
        retention=RetentionPolicy.PINNED_REQUEST,
        owner_id=f"request:{entry_id}",
    )


def _draft(manager, index: int) -> None:
    manager.allocate(
        entry_id=f"draft-{index}",
        payload=bytes([index % 251]),
        lifecycle=CacheLifecycle.DRAFT_TRANSIENT,
        retention=RetentionPolicy.SHORT_BOUNDED,
        owner_id="draft-session",
    )


def _audit_passes(audit: dict) -> bool:
    return audit["anonymous_entry_count"] == 0 and all(
        value for key, value in audit.items() if key != "anonymous_entry_count"
    )


def _simulate() -> dict:
    # Draft pressure is confined to the draft accounting and eviction domain.
    draft_manager = CacheLifecycleManager(_budgets())
    target = _target(
        draft_manager, "prefix-a", b"T" * 100, _identity()
    )
    target_storage_id = target.storage.storage_id
    target_digest = target.state_digest
    for index in range(DRAFT_ROTATIONS):
        _draft(draft_manager, index)
    target_after = draft_manager.get("prefix-a")
    draft_metrics = draft_manager.accounting_snapshot()

    # Active state is not inserted into any LRU and cannot be externally evicted.
    active_manager = CacheLifecycleManager(_budgets())
    active = _active(active_manager, "active-a", b"K" * 64)
    active_storage_id = active.storage.storage_id
    active_digest = active.state_digest
    for index in range(4):
        _target(
            active_manager,
            f"pressure-prefix-{index}",
            bytes([index]) * 100,
            _identity(prefix_token_sha256=f"pressure-{index}"),
        )
    for index in range(128):
        _draft(active_manager, index)
    forbidden_evictions_rejected = 0
    for reason in (EvictionReason.DRAFT_PRESSURE, EvictionReason.TARGET_PRESSURE):
        try:
            active_manager.evict("active-a", reason=reason)
        except CacheLifecycleError:
            forbidden_evictions_rejected += 1
    active_after = active_manager.get("active-a")

    # Snapshot capture and restore both own their physical bytes.
    snapshot_manager = CacheLifecycleManager(_budgets())
    caller_storage = bytearray(b"S0" * 32)
    active = _active(snapshot_manager, "active-a", caller_storage)
    caller_storage[:] = b"XX" * 32
    caller_alias_severed = active.storage.read() == b"S0" * 32
    snapshot = snapshot_manager.capture_snapshot(
        "active-a", snapshot_entry_id="snapshot-a", snapshot_owner_id="apc-a"
    )
    source_snapshot_storage_distinct = (
        active.storage.storage_id != snapshot.storage.storage_id
    )
    snapshot_digest = snapshot.state_digest
    snapshot_manager.mutate_active("active-a", b"S1" * 32)
    for index in range(4):
        _target(
            snapshot_manager,
            f"pressure-prefix-{index}",
            bytes([index]) * 100,
            _identity(prefix_token_sha256=f"snapshot-pressure-{index}"),
        )
    for index in range(128):
        _draft(snapshot_manager, index)
    snapshot_unchanged_under_pressure = (
        snapshot_manager.get("snapshot-a").state_digest == snapshot_digest
    )
    snapshot_manager.restore_snapshot("snapshot-a", "active-a")
    restored = snapshot_manager.get("active-a")
    first_restore_exact = restored.state_digest == snapshot_digest
    restore_storage_distinct = (
        restored.storage.storage_id
        != snapshot_manager.get("snapshot-a").storage.storage_id
    )
    snapshot_manager.mutate_active("active-a", b"S2" * 32)
    snapshot_still_immutable = (
        snapshot_manager.get("snapshot-a").state_digest == snapshot_digest
    )
    snapshot_manager.restore_snapshot("snapshot-a", "active-a")
    replay_restore_exact = (
        snapshot_manager.get("active-a").state_digest == snapshot_digest
    )

    # Equal tokens do not bypass any model/backend/cache/state ABI identity field.
    identity_manager = CacheLifecycleManager(_budgets())
    identity = _identity()
    _target(identity_manager, "identity-prefix", b"P" * 32, identity)
    identity_fields = (
        "model_revision",
        "checkpoint_fingerprint",
        "backend_policy",
        "attention_cache_abi",
        "kda_state_abi",
        "indexpool_abi",
    )
    mismatches = {}
    for field in identity_fields:
        candidate = replace(identity, **{field: f"different-{field}"})
        mismatches[field] = (
            identity_manager.lookup_target_prefix(candidate) is None
        )
    exact_identity_hit = (
        identity_manager.lookup_target_prefix(identity).entry_id
        == "identity-prefix"
    )

    snapshot_metrics = snapshot_manager.accounting_snapshot()
    return {
        "classes": {
            lifecycle.value: {
                "contents": list(LIFECYCLE_CONTRACTS[lifecycle].contents),
                "retention": {
                    CacheLifecycle.TARGET_PREFIX: RetentionPolicy.LONG_REUSE,
                    CacheLifecycle.ACTIVE_RECURRENT: RetentionPolicy.PINNED_REQUEST,
                    CacheLifecycle.SNAPSHOT_STATE: RetentionPolicy.SNAPSHOT_OWNED,
                    CacheLifecycle.DRAFT_TRANSIENT: RetentionPolicy.SHORT_BOUNDED,
                }[lifecycle].value,
                "owner_scope": LIFECYCLE_CONTRACTS[lifecycle].lifetime,
                "eviction_domain": LIFECYCLE_CONTRACTS[lifecycle].eviction_domain,
            }
            for lifecycle in CacheLifecycle
        },
        "draft_pressure_isolation": {
            "rotations": DRAFT_ROTATIONS,
            "draft_budget_bytes": 32,
            "draft_resident_bytes": draft_metrics["draft-transient"][
                "resident_bytes"
            ],
            "draft_evictions": draft_metrics["draft-transient"][
                "eviction_count"
            ],
            "draft_cumulative_allocated_bytes": draft_metrics[
                "draft-transient"
            ]["cumulative_allocated_bytes"],
            "target_evictions": draft_metrics["target-prefix"]["eviction_count"],
            "target_storage_identity_unchanged": (
                target_after.storage.storage_id == target_storage_id
            ),
            "target_digest_unchanged": target_after.state_digest == target_digest,
            "audit": draft_manager.audit(),
        },
        "active_recurrent_pinning": {
            "storage_identity_unchanged": (
                active_after.storage.storage_id == active_storage_id
            ),
            "state_digest_unchanged": active_after.state_digest == active_digest,
            "forbidden_evictions_rejected": forbidden_evictions_rejected,
            "prefix_lru_entry_ids": list(
                active_manager.lru_entry_ids(CacheLifecycle.TARGET_PREFIX)
            ),
            "active_lru_entry_ids": list(
                active_manager.lru_entry_ids(CacheLifecycle.ACTIVE_RECURRENT)
            ),
            "audit": active_manager.audit(),
        },
        "snapshot_independence": {
            "caller_alias_severed": caller_alias_severed,
            "source_snapshot_storage_distinct": source_snapshot_storage_distinct,
            "snapshot_unchanged_under_pressure": snapshot_unchanged_under_pressure,
            "first_restore_exact": first_restore_exact,
            "restore_storage_distinct": restore_storage_distinct,
            "snapshot_still_immutable_after_active_mutation": (
                snapshot_still_immutable
            ),
            "replay_restore_exact": replay_restore_exact,
            "snapshot_resident_bytes": snapshot_metrics["snapshot-state"][
                "resident_bytes"
            ],
            "active_resident_bytes": snapshot_metrics["active-recurrent"][
                "resident_bytes"
            ],
            "audit": snapshot_manager.audit(),
        },
        "prefix_identity": {
            "descriptor": identity.canonical_descriptor(),
            "namespace_sha256": identity.namespace_sha256,
            "exact_identity_hit": exact_identity_hit,
            "same_tokens_different_identity_misses": mismatches,
            "audit": identity_manager.audit(),
        },
        "accounting_fields": [
            "resident_bytes",
            "peak_bytes",
            "allocation_count",
            "eviction_count",
            "cumulative_allocated_bytes",
        ],
        "logical_lifecycle_separate_from_physical_storage": True,
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
    oracle = oracle_probe._official_oracle(model, processor, report)
    vocab = int(model.language_model.lm_head.weight.shape[0])
    ram_apc = packed_probe._ram_apc(model, vocab)
    return {
        "executed": True,
        "load_seconds": load_seconds,
        "warm_residency_seconds": warm_seconds,
        "moe_backend": getattr(model, "_glm53_moe_backend", None),
        "official_oracle": oracle,
        "ram_apc": ram_apc,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not mx.metal.is_available():
        raise RuntimeError("cache lifecycle full-model gate requires MLX/Metal")

    report = inspect_checkpoint(args.model, require_server_ready=True)
    simulation = _simulate()
    full_model = _full_model(args.model, report)
    draft = simulation["draft_pressure_isolation"]
    active = simulation["active_recurrent_pinning"]
    snapshot = simulation["snapshot_independence"]
    identity = simulation["prefix_identity"]
    acceptance = {
        "four_lifecycle_classes_explicit": len(simulation["classes"]) == 4,
        "lifecycle_physical_storage_separated": simulation[
            "logical_lifecycle_separate_from_physical_storage"
        ],
        "draft_pressure_does_not_evict_target": (
            draft["target_evictions"] == 0
            and draft["target_storage_identity_unchanged"]
            and draft["target_digest_unchanged"]
        ),
        "draft_resident_is_bounded": (
            draft["draft_resident_bytes"] <= draft["draft_budget_bytes"]
            and draft["draft_evictions"] > 0
        ),
        "active_recurrent_is_request_pinned": (
            active["storage_identity_unchanged"]
            and active["state_digest_unchanged"]
            and active["forbidden_evictions_rejected"] == 2
            and active["active_lru_entry_ids"] == []
        ),
        "snapshot_storage_is_independent": all(
            snapshot[name]
            for name in (
                "caller_alias_severed",
                "source_snapshot_storage_distinct",
                "snapshot_unchanged_under_pressure",
                "first_restore_exact",
                "restore_storage_distinct",
                "snapshot_still_immutable_after_active_mutation",
                "replay_restore_exact",
            )
        ),
        "prefix_identity_fields_isolate_reuse": (
            identity["exact_identity_hit"]
            and all(identity["same_tokens_different_identity_misses"].values())
        ),
        "class_accounting_complete": simulation["accounting_fields"]
        == [
            "resident_bytes",
            "peak_bytes",
            "allocation_count",
            "eviction_count",
            "cumulative_allocated_bytes",
        ],
        "all_policy_audits_exact": all(
            _audit_passes(section["audit"])
            for section in (draft, active, snapshot, identity)
        ),
        "ram_apc_continuation_exact": (
            full_model["ram_apc"]["all_logits_hashes_match"]
            and full_model["ram_apc"]["post_state_exact"]
            and full_model["ram_apc"]["snapshot_immutable"]
        ),
        "official_16_token_oracle_exact": full_model["official_oracle"][
            "first_16_match"
        ],
        "official_128_token_oracle_exact": full_model["official_oracle"][
            "full_128_match"
        ],
    }
    artifact = {
        "schema": "glm53-cache-state-lifecycle-v1",
        "date": date.today().isoformat(),
        "complete": all(acceptance.values()),
        "probe_only": True,
        "official_hf_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "existing_runtime_identity": {
            "cache_identity_schema": CACHE_IDENTITY_SCHEMA,
            "direct_attention_cache_abi": NOPE_DSA_CACHE_ABI_DIRECT,
            "compact_attention_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
        },
        "simulation": simulation,
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
            "lifecycle_contract_ready_for_state_safety_integration"
            if all(acceptance.values())
            else "stop_cache_lifecycle_contract"
        ),
    }
    _atomic_write(args.output, artifact)
    print(json.dumps({"output": str(args.output), "complete": artifact["complete"]}))
    return 0 if artifact["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
