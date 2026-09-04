#!/usr/bin/env python3
"""Stress exact cache state under dense allocation and ownership churn.

The uninterrupted arm advances once over a short deterministic token stream.
The eventful arm advances over the same stream while repeatedly cloning,
restoring, transferring, and releasing RAM APC state.  This intentionally
decouples logical decode length from cumulative physical allocation pressure.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
from collections import Counter
from dataclasses import replace
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from glm53_flash_mlx.abi import (
    KERNEL_ABI_VERSION,
    MLX_VLM_REVISION,
    NOPE_DSA_CACHE_ABI_COMPACT,
)
from glm53_flash_mlx.cache_lifecycle import CacheLifecycle, PrefixIdentity
from glm53_flash_mlx.churn import (
    BASELINE_256K_CUMULATIVE_ALLOCATION_BYTES,
    accounting_balance_errors,
    churn_tier,
    distributed_churn_schedule,
    required_churn_cycles,
    rollback_schedule,
    temporary_storage_returned,
)
from glm53_flash_mlx.kda_digest import (
    SoakLifecycleAccounting,
    aggregate_layer_digest,
    array_digest,
    compare_layerwise_digests,
    first_kda_state_difference,
    layerwise_kda_digests,
    lifecycle_accounting_delta,
    steady_active_memory_drift,
)
from glm53_flash_mlx.kda_state import (
    KDA_STATE_INDEX_CONTRACT,
    KDAStateIndexError,
    rollback_restore_state,
)
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import EXPECTED_DSA, inspect_checkpoint
from glm53_flash_mlx.materialization import (
    MATERIALIZATION_INTERVAL_TOKENS,
    MATERIALIZATION_POLICY,
)
from glm53_flash_mlx.ownership import (
    TensorLayout,
    TensorOwnershipError,
    borrowed_ephemeral_tensor,
    require_resident,
)

from soak_layerwise_kda_state_digests import (
    KDA_LAYERS,
    MAX_ACTIVE_DRIFT,
    MAX_PEAK_BYTES,
    RESERVE_TAIL,
    _atomic_write,
    _cache_binding_signature,
    _cache_groups,
    _clone_cache,
    _logits_hash,
    _memory,
    _observe,
    _refresh_live,
    _register_cache,
    _release_accounted_cache,
    _release_cache,
    _restore_from_snapshot,
    _run_forward,
    _state_leaf_count,
    _token_for_step,
    _token_sequence_digest,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-state-cumulative-allocation-churn-screen-20260904.json"
)
EXPECTED_STATE_LEAVES = 167
INVALID_OPERATIONS = (
    "wrong-cache-identity-restore",
    "invalid-kda-state-index",
    "ephemeral-resident-promotion",
    "rollback-17",
)


class ChurnDivergenceError(RuntimeError):
    pass


class SnapshotIdentityError(ValueError):
    pass


def _full_cache_digest(cache) -> str:
    digest = hashlib.sha256()
    for path, value in tree_flatten([entry.state for entry in cache]):
        digest.update(path.encode())
        if hasattr(value, "dtype") and hasattr(value, "shape"):
            digest.update(array_digest(value, mx_module=mx).encode())
        else:
            digest.update(repr(value).encode())
    for entry in cache:
        digest.update(repr(getattr(entry, "meta_state", None)).encode())
    return digest.hexdigest()


def _allocation_sequence(accounting: SoakLifecycleAccounting) -> int:
    return sum(
        int(row["allocation_count"])
        for row in accounting.snapshot()["by_lifecycle"].values()
    )


def _snapshot_identity(artifact: dict) -> PrefixIdentity:
    return PrefixIdentity(
        model_revision=artifact["checkpoint_revision"],
        checkpoint_fingerprint=artifact["checkpoint_fingerprint"],
        backend_policy=artifact["backend"],
        attention_cache_abi=artifact["cache_abi"],
        kda_state_abi=KDA_STATE_INDEX_CONTRACT,
        indexpool_abi=artifact["cache_abi"],
        prefix_token_sha256=artifact["teacher_forced_token_sha256"],
    )


def _identity_checked_restore(
    source,
    *,
    source_identity: PrefixIdentity,
    expected_identity: PrefixIdentity,
    restored_owner: str,
    live_cache,
    capacity_tokens: int,
    accounting: SoakLifecycleAccounting,
):
    if source_identity.namespace_sha256 != expected_identity.namespace_sha256:
        raise SnapshotIdentityError("RAM APC snapshot identity mismatch")
    return _restore_from_snapshot(
        source,
        restored_owner=restored_owner,
        live_cache=live_cache,
        capacity_tokens=capacity_tokens,
        accounting=accounting,
    )


def _record_failure(
    artifact: dict,
    *,
    reason: str,
    step: int,
    operation_sequence: int,
    accounting: SoakLifecycleAccounting,
    cache_a,
    cache_b,
    apc_generation: int,
    rollback_depth: int | None = None,
    resident_before: int | None = None,
    resident_after: int | None = None,
) -> None:
    artifact["first_divergence"] = {
        "reason": reason,
        "operation_sequence": operation_sequence,
        "logical_token": step,
        "allocation_sequence": _allocation_sequence(accounting),
        "lifecycle": accounting.snapshot(),
        "ownership_state": {
            "live_a": "active-recurrent+target-prefix",
            "live_b": "active-recurrent+target-prefix",
        },
        "apc_generation": apc_generation,
        "rollback_depth": rollback_depth,
        "resident_bytes_before": resident_before,
        "resident_bytes_after": resident_after,
        "cumulative_allocated_bytes": accounting.snapshot()[
            "cumulative_allocated_bytes"
        ],
        "first_state_difference": first_kda_state_difference(
            cache_a, cache_b, kda_layers=KDA_LAYERS, mx_module=mx
        ),
    }
    raise ChurnDivergenceError(reason)


def _rejected_operation(
    kind: str,
    *,
    cache_b,
    source,
    source_identity: PrefixIdentity,
    capacity_tokens: int,
    accounting: SoakLifecycleAccounting,
) -> dict:
    live_before = _full_cache_digest(cache_b)
    source_before = _full_cache_digest(source)
    accounting_before = json.dumps(accounting.snapshot(), sort_keys=True)
    binding_before = _cache_binding_signature(cache_b)
    rejected = False
    try:
        if kind == "wrong-cache-identity-restore":
            wrong = replace(source_identity, backend_policy="wrong-backend")
            _identity_checked_restore(
                source,
                source_identity=wrong,
                expected_identity=source_identity,
                restored_owner="unreachable-wrong-identity",
                live_cache=cache_b,
                capacity_tokens=capacity_tokens,
                accounting=accounting,
            )
        elif kind == "invalid-kda-state-index":
            _ = cache_b[KDA_LAYERS[0]][-2]
        elif kind == "ephemeral-resident-promotion":
            staging = mx.arange(32, dtype=mx.uint8)
            require_resident(
                borrowed_ephemeral_tensor(
                    staging,
                    layout=TensorLayout.ROW_MAJOR_CONTIGUOUS,
                )
            )
        elif kind == "rollback-17":
            kda_rejected = False
            dsa_rejected = False
            try:
                rollback_restore_state(
                    cache_b[KDA_LAYERS[0]],
                    list(cache_b[KDA_LAYERS[0]].state),
                    tokens=17,
                )
            except KDAStateIndexError:
                kda_rejected = True
            try:
                cache_b[EXPECTED_DSA[0]][1].trim(17)
            except ValueError:
                dsa_rejected = True
            if kda_rejected and dsa_rejected:
                rejected = True
            else:
                raise ChurnDivergenceError(
                    "17-token rollback was not rejected by both KDA and DSA"
                )
        else:
            raise ValueError(f"unknown rejected operation: {kind}")
    except (SnapshotIdentityError, KDAStateIndexError, TensorOwnershipError):
        rejected = True

    live_after = _full_cache_digest(cache_b)
    source_after = _full_cache_digest(source)
    return {
        "kind": kind,
        "rejected": rejected,
        "authoritative_state_unchanged": live_before == live_after,
        "snapshot_unchanged": source_before == source_after,
        "accounting_unchanged": accounting_before
        == json.dumps(accounting.snapshot(), sort_keys=True),
        "bindings_unchanged": binding_before == _cache_binding_signature(cache_b),
    }


def _apc_churn_cycle(
    cache_b,
    *,
    step: int,
    operation_sequence: int,
    apc_generation: int,
    invalid_kind: str,
    identity: PrefixIdentity,
    capacity_tokens: int,
    accounting: SoakLifecycleAccounting,
) -> tuple[list, dict]:
    accounting_before = accounting.snapshot()
    resident_before = accounting_before["resident_bytes"]
    live_before = _full_cache_digest(cache_b)
    source_owner = f"apc-{apc_generation}-source"
    restored_owner = f"apc-{apc_generation}-restored"
    source = _clone_cache(cache_b, min_capacity_tokens=capacity_tokens)
    _register_cache(accounting, source_owner, source, snapshot=True)
    source_digest = _full_cache_digest(source)

    rejected = _rejected_operation(
        invalid_kind,
        cache_b=cache_b,
        source=source,
        source_identity=identity,
        capacity_tokens=capacity_tokens,
        accounting=accounting,
    )
    if not all(
        rejected[key]
        for key in (
            "rejected",
            "authoritative_state_unchanged",
            "snapshot_unchanged",
            "accounting_unchanged",
            "bindings_unchanged",
        )
    ):
        raise ChurnDivergenceError(f"rejected operation was not atomic: {rejected}")

    cache_b = _identity_checked_restore(
        source,
        source_identity=identity,
        expected_identity=identity,
        restored_owner=restored_owner,
        live_cache=cache_b,
        capacity_tokens=capacity_tokens,
        accounting=accounting,
    )
    restored_digest = _full_cache_digest(cache_b)
    source_immutable = source_digest == _full_cache_digest(source)
    _release_accounted_cache(accounting, source_owner, source)
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    resident_after = accounting.snapshot()["resident_bytes"]
    balance_errors = accounting_balance_errors(accounting.snapshot())
    accounting_after = accounting.snapshot()
    event = {
        "kind": "apc-ownership-churn",
        "operation_sequence": operation_sequence,
        "logical_token": step,
        "allocation_sequence": _allocation_sequence(accounting),
        "apc_generation": apc_generation,
        "rejected_operation": rejected,
        "restore_exact": restored_digest == live_before,
        "snapshot_owned_storage_immutable": source_immutable,
        "resident_bytes_before": resident_before,
        "resident_bytes_after": resident_after,
        "cumulative_allocated_bytes": accounting.snapshot()[
            "cumulative_allocated_bytes"
        ],
        "lifecycle_balance_exact": not balance_errors,
        "lifecycle_balance_errors": list(balance_errors),
        "temporary_storage_returned": temporary_storage_returned(
            accounting_before, accounting_after
        ),
    }
    return cache_b, event


def _rollback_replay(
    model,
    cache_b,
    source,
    *,
    source_owner: str,
    target: int,
    depth: int,
    vocab: int,
    operation_sequence: int,
    apc_generation: int,
    capacity_tokens: int,
    accounting: SoakLifecycleAccounting,
    expected_logits,
) -> tuple[list, dict, int]:
    expected_state = _full_cache_digest(cache_b)
    expected_logits_hash = _logits_hash(expected_logits)
    source_digest = _full_cache_digest(source)
    restored_owner = f"rollback-{target}-restored"
    cache_b = _restore_from_snapshot(
        source,
        restored_owner=restored_owner,
        live_cache=cache_b,
        capacity_tokens=capacity_tokens,
        accounting=accounting,
    )
    replay_logits = None
    replay_forwards = 0
    replay_nan = 0
    for replay_step in range(target - depth + 1, target + 1):
        output, replay_logits, nans = _run_forward(
            model, cache_b, _token_for_step(replay_step, vocab)
        )
        replay_forwards += 1
        replay_nan += nans
        del output
    state_exact = _full_cache_digest(cache_b) == expected_state
    logits_exact = _logits_hash(replay_logits) == expected_logits_hash
    source_immutable = _full_cache_digest(source) == source_digest
    _release_accounted_cache(accounting, source_owner, source)
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    balance_errors = accounting_balance_errors(accounting.snapshot())
    event = {
        "kind": "rollback-replay",
        "operation_sequence": operation_sequence,
        "logical_token": target,
        "allocation_sequence": _allocation_sequence(accounting),
        "apc_generation": apc_generation,
        "rollback_depth": depth,
        "state_exact": state_exact,
        "logits_exact": logits_exact,
        "source_snapshot_immutable": source_immutable,
        "lifecycle_balance_exact": not balance_errors,
        "nan_count": replay_nan,
    }
    if not all(
        event[key]
        for key in (
            "state_exact",
            "logits_exact",
            "source_snapshot_immutable",
            "lifecycle_balance_exact",
        )
    ) or replay_nan:
        raise ChurnDivergenceError(f"rollback/replay failed: {event}")
    return cache_b, event, replay_forwards


def _materialize(cache) -> float:
    started = time.perf_counter()
    mx.eval([entry.state for entry in cache])
    mx.clear_cache()
    mx.synchronize()
    return (time.perf_counter() - started) * 1000.0


def _checkpoint(
    artifact: dict,
    *,
    step: int,
    cache_a,
    cache_b,
    accounting: SoakLifecycleAccounting,
    counters: dict,
    logits_a,
    logits_b,
) -> None:
    accounting_snapshot = accounting.snapshot()
    observed_a, elapsed_a = _observe(
        cache_a, accounting_snapshot=accounting_snapshot, counters=counters
    )
    observed_b, elapsed_b = _observe(
        cache_b, accounting_snapshot=accounting_snapshot, counters=counters
    )
    difference = compare_layerwise_digests(
        observed_a["layers"], observed_b["layers"]
    )
    if difference is not None:
        raise ChurnDivergenceError(f"KDA checkpoint divergence: {difference}")
    full_a = _full_cache_digest(cache_a)
    full_b = _full_cache_digest(cache_b)
    row = {
        "logical_token": step,
        "operation_sequence": counters["operation_sequence"],
        "allocation_sequence": _allocation_sequence(accounting),
        "apc_generation": counters["apc_generation"],
        "actual_model_forwards": dict(counters["actual_model_forwards"]),
        "materialization_count": counters["materialization_count"],
        "uninterrupted": observed_a,
        "eventful": observed_b,
        "layerwise_exact": True,
        "full_cache_digest_exact": full_a == full_b,
        "full_vocab_logits_exact": _logits_hash(logits_a) == _logits_hash(logits_b)
        if logits_a is not None
        else True,
        "state_leaf_count": {
            "uninterrupted": _state_leaf_count(cache_a),
            "eventful": _state_leaf_count(cache_b),
        },
        "authoritative_cache_bytes": {
            "uninterrupted": sum(int(entry.nbytes) for entry in cache_a),
            "eventful": sum(int(entry.nbytes) for entry in cache_b),
        },
        "observer_ms": {"uninterrupted": elapsed_a, "eventful": elapsed_b},
        "lifecycle": accounting_snapshot,
        "lifecycle_balance_exact": not accounting_balance_errors(
            accounting_snapshot
        ),
        "memory": _memory(),
    }
    if not all(
        row[key]
        for key in (
            "layerwise_exact",
            "full_cache_digest_exact",
            "full_vocab_logits_exact",
            "lifecycle_balance_exact",
        )
    ):
        raise ChurnDivergenceError(f"checkpoint invariant failed: {row}")
    artifact["checkpoints"][str(step)] = row
    artifact["observer_total_ms"] += elapsed_a + elapsed_b
    artifact["last_exact_checkpoint"] = step


def run_churn(model, artifact: dict, output_path: Path, *, vocab: int) -> None:
    tier = churn_tier(artifact["tier"])
    capacity_tokens = artifact["configured_capacity_tokens"]
    identity = _snapshot_identity(artifact)
    accounting = SoakLifecycleAccounting()
    cache_a = model.make_cache()
    cache_b = model.make_cache()
    _register_cache(accounting, "A", cache_a, snapshot=False)
    _register_cache(accounting, "B", cache_b, snapshot=False)
    counters = {
        "operation_sequence": 0,
        "apc_generation": 0,
        "materialization_count": 0,
        "actual_model_forwards": {"uninterrupted": 0, "eventful": 0},
    }
    rollback_rows = dict(rollback_schedule(tier))
    rollback_capture = {target - depth: target for target, depth in rollback_rows.items()}
    rollback_snapshots = {}
    cycle_schedule = None
    cycle_counts = Counter()
    operation_context = {
        "kind": "decode",
        "rollback_depth": None,
        "resident_bytes_before": accounting.snapshot()["resident_bytes"],
        "resident_bytes_after": accounting.snapshot()["resident_bytes"],
    }
    latencies_a = []
    latencies_b = []
    final_logits_a = final_logits_b = None
    current_step = 0
    mx.reset_peak_memory()
    started = time.perf_counter()

    try:
        _checkpoint(
            artifact,
            step=0,
            cache_a=cache_a,
            cache_b=cache_b,
            accounting=accounting,
            counters=counters,
            logits_a=None,
            logits_b=None,
        )
        artifact["lifecycle_start"] = accounting.snapshot()

        for step in range(1, tier.logical_tokens + 1):
            current_step = step
            if step - 1 in rollback_capture:
                target = rollback_capture[step - 1]
                source_owner = f"rollback-{target}-source"
                source = _clone_cache(cache_b, min_capacity_tokens=capacity_tokens)
                _register_cache(accounting, source_owner, source, snapshot=True)
                rollback_snapshots[target] = (source, source_owner)

            token = _token_for_step(step, vocab)
            forward_started = time.perf_counter()
            output_a, logits_a, nans_a = _run_forward(model, cache_a, token)
            latencies_a.append(time.perf_counter() - forward_started)
            counters["actual_model_forwards"]["uninterrupted"] += 1
            forward_started = time.perf_counter()
            output_b, logits_b, nans_b = _run_forward(model, cache_b, token)
            latencies_b.append(time.perf_counter() - forward_started)
            counters["actual_model_forwards"]["eventful"] += 1
            artifact["nan_count"] += nans_a + nans_b
            logits_equal = mx.array_equal(logits_a, logits_b)
            mx.eval(logits_equal)
            if not bool(logits_equal.item()):
                _record_failure(
                    artifact,
                    reason="A/B full-vocab logits diverged",
                    step=step,
                    operation_sequence=counters["operation_sequence"],
                    accounting=accounting,
                    cache_a=cache_a,
                    cache_b=cache_b,
                    apc_generation=counters["apc_generation"],
                )

            _refresh_live(accounting, "A", cache_a)
            _refresh_live(accounting, "B", cache_b)

            if cycle_schedule is None:
                cache_bytes = sum(int(entry.nbytes) for entry in cache_b)
                cycles = required_churn_cycles(
                    tier.cumulative_allocation_target_bytes,
                    accounting.snapshot()["cumulative_allocated_bytes"],
                    2 * cache_bytes,
                )
                cycle_schedule = distributed_churn_schedule(
                    tier.logical_tokens, cycles
                )
                cycle_counts = Counter(cycle_schedule)
                artifact["churn_plan"] = {
                    "cache_bytes_per_snapshot": cache_bytes,
                    "allocation_bytes_per_cycle": 2 * cache_bytes,
                    "planned_cycles": cycles,
                    "first_cycle_step": cycle_schedule[0] if cycle_schedule else None,
                    "last_cycle_step": cycle_schedule[-1] if cycle_schedule else None,
                }

            if step in rollback_rows:
                counters["operation_sequence"] += 1
                depth = rollback_rows[step]
                operation_context = {
                    "kind": "rollback-replay",
                    "rollback_depth": depth,
                    "resident_bytes_before": accounting.snapshot()["resident_bytes"],
                    "resident_bytes_after": None,
                }
                source, source_owner = rollback_snapshots.pop(step)
                cache_b, event, forwards = _rollback_replay(
                    model,
                    cache_b,
                    source,
                    source_owner=source_owner,
                    target=step,
                    depth=depth,
                    vocab=vocab,
                    operation_sequence=counters["operation_sequence"],
                    apc_generation=counters["apc_generation"],
                    capacity_tokens=capacity_tokens,
                    accounting=accounting,
                    expected_logits=logits_b,
                )
                counters["actual_model_forwards"]["eventful"] += forwards
                artifact["events"].append(event)
                operation_context["resident_bytes_after"] = accounting.snapshot()[
                    "resident_bytes"
                ]

            for _ in range(cycle_counts.get(step, 0)):
                counters["operation_sequence"] += 1
                counters["apc_generation"] += 1
                invalid_kind = INVALID_OPERATIONS[
                    (counters["apc_generation"] - 1) % len(INVALID_OPERATIONS)
                ]
                operation_context = {
                    "kind": f"apc-ownership-churn:{invalid_kind}",
                    "rollback_depth": 17 if invalid_kind == "rollback-17" else None,
                    "resident_bytes_before": accounting.snapshot()["resident_bytes"],
                    "resident_bytes_after": None,
                }
                cache_b, event = _apc_churn_cycle(
                    cache_b,
                    step=step,
                    operation_sequence=counters["operation_sequence"],
                    apc_generation=counters["apc_generation"],
                    invalid_kind=invalid_kind,
                    identity=identity,
                    capacity_tokens=capacity_tokens,
                    accounting=accounting,
                )
                artifact["events"].append(event)
                operation_context["resident_bytes_after"] = accounting.snapshot()[
                    "resident_bytes"
                ]
                if not all(
                    event[key]
                    for key in (
                        "restore_exact",
                        "snapshot_owned_storage_immutable",
                        "lifecycle_balance_exact",
                        "temporary_storage_returned",
                    )
                ):
                    _record_failure(
                        artifact,
                        reason=f"APC churn cycle failed: {event}",
                        step=step,
                        operation_sequence=counters["operation_sequence"],
                        accounting=accounting,
                        cache_a=cache_a,
                        cache_b=cache_b,
                        apc_generation=counters["apc_generation"],
                        rollback_depth=operation_context["rollback_depth"],
                        resident_before=operation_context["resident_bytes_before"],
                        resident_after=operation_context["resident_bytes_after"],
                    )

            if step % MATERIALIZATION_INTERVAL_TOKENS == 0:
                before_a = _full_cache_digest(cache_a)
                before_b = _full_cache_digest(cache_b)
                elapsed_a = _materialize(cache_a)
                elapsed_b = _materialize(cache_b)
                counters["materialization_count"] += 1
                artifact["materializations"].append(
                    {
                        "logical_token": step,
                        "count": counters["materialization_count"],
                        "uninterrupted_ms": elapsed_a,
                        "eventful_ms": elapsed_b,
                        "state_exact": (
                            before_a == _full_cache_digest(cache_a)
                            and before_b == _full_cache_digest(cache_b)
                        ),
                    }
                )
                _checkpoint(
                    artifact,
                    step=step,
                    cache_a=cache_a,
                    cache_b=cache_b,
                    accounting=accounting,
                    counters=counters,
                    logits_a=logits_a,
                    logits_b=logits_b,
                )
                artifact["last_completed_step"] = step
                _atomic_write(output_path, artifact)
                elapsed = time.perf_counter() - started
                rate = step / elapsed
                print(
                    json.dumps(
                        {
                            "phase": "checkpoint",
                            "step": step,
                            "churn_cycles": counters["apc_generation"],
                            "cumulative_allocated_bytes": accounting.snapshot()[
                                "cumulative_allocated_bytes"
                            ],
                            "logical_steps_per_second": rate,
                            "estimated_remaining_seconds": (
                                tier.logical_tokens - step
                            )
                            / rate,
                        }
                    ),
                    flush=True,
                )

            final_logits_a = _logits_hash(logits_a)
            final_logits_b = _logits_hash(logits_b)
            artifact["last_completed_step"] = step
            del output_a, output_b, logits_a, logits_b, logits_equal

        if rollback_snapshots:
            raise ChurnDivergenceError("rollback snapshots remained live at completion")

        end_accounting = accounting.snapshot()
        rows = [
            row for key, row in artifact["checkpoints"].items() if int(key) >= 256
        ]
        authoritative = [
            row["authoritative_cache_bytes"]["uninterrupted"]
            + row["authoritative_cache_bytes"]["eventful"]
            for row in rows
        ]
        event_counts = Counter(row["kind"] for row in artifact["events"])
        rejected_counts = Counter(
            row["rejected_operation"]["kind"]
            for row in artifact["events"]
            if row["kind"] == "apc-ownership-churn"
        )
        active_drift = steady_active_memory_drift(artifact["checkpoints"])
        elapsed = time.perf_counter() - started
        final_memory = _memory()
        acceptance = {
            "logical_tokens_at_most_16k": tier.logical_tokens <= 16_384,
            "cumulative_allocation_target_reached": end_accounting[
                "cumulative_allocated_bytes"
            ]
            >= tier.cumulative_allocation_target_bytes,
            "all_full_vocab_logits_exact": final_logits_a == final_logits_b,
            "all_34_kda_layer_digests_exact": all(
                row["layerwise_exact"] for row in rows
            ),
            "all_full_cache_digests_exact": all(
                row["full_cache_digest_exact"] for row in rows
            ),
            "apc_capture_restore_exact": all(
                row.get("restore_exact", True) for row in artifact["events"]
            ),
            "rollback_1_8_16_exact": (
                {row["rollback_depth"] for row in artifact["events"] if row["kind"] == "rollback-replay"}
                == {1, 8, 16}
                and all(
                    row.get("state_exact", False)
                    and row.get("logits_exact", False)
                    for row in artifact["events"]
                    if row["kind"] == "rollback-replay"
                )
            ),
            "all_rejected_operations_atomic": (
                set(rejected_counts) == set(INVALID_OPERATIONS)
                and all(
                    all(
                        row["rejected_operation"][key]
                        for key in (
                            "rejected",
                            "authoritative_state_unchanged",
                            "snapshot_unchanged",
                            "accounting_unchanged",
                            "bindings_unchanged",
                        )
                    )
                    for row in artifact["events"]
                    if row["kind"] == "apc-ownership-churn"
                )
            ),
            "snapshot_owned_storage_immutable": all(
                row.get("snapshot_owned_storage_immutable", True)
                and row.get("source_snapshot_immutable", True)
                for row in artifact["events"]
            ),
            "lifecycle_accounting_exact": (
                not accounting_balance_errors(end_accounting)
                and all(row["lifecycle_balance_exact"] for row in rows)
            ),
            "authoritative_state_drift_zero": max(authoritative)
            == min(authoritative),
            "anonymous_allocation_zero": end_accounting[
                "anonymous_allocation_count"
            ]
            == 0,
            "state_leaf_count_bounded": all(
                set(row["state_leaf_count"].values()) == {EXPECTED_STATE_LEAVES}
                for row in rows
            ),
            "resident_memory_bounded": active_drift <= MAX_ACTIVE_DRIFT,
            "temporary_lifecycle_storage_returned": (
                end_accounting["by_lifecycle"][CacheLifecycle.SNAPSHOT_STATE.value][
                    "resident_bytes"
                ]
                == 0
                and end_accounting["by_lifecycle"][CacheLifecycle.DRAFT_TRANSIENT.value][
                    "resident_bytes"
                ]
                == 0
            ),
            "materialization_cadence_exact": (
                counters["materialization_count"]
                == tier.logical_tokens // MATERIALIZATION_INTERVAL_TOKENS
                and all(row["state_exact"] for row in artifact["materializations"])
            ),
            "no_nan_invalid_access_or_metal_error": artifact["nan_count"] == 0,
            "peak_memory_at_most_340_gb": final_memory["peak_bytes"]
            <= MAX_PEAK_BYTES,
        }
        artifact["summary"] = {
            "logical_tokens": tier.logical_tokens,
            "actual_model_forwards": dict(counters["actual_model_forwards"]),
            "operation_count": counters["operation_sequence"],
            "apc_generation_count": counters["apc_generation"],
            "event_counts": dict(event_counts),
            "rejected_operation_counts": dict(rejected_counts),
            "checkpoint_count": len(artifact["checkpoints"]),
            "materialization_count": counters["materialization_count"],
            "final_logits_hash": final_logits_a,
            "final_layerwise_digest": aggregate_layer_digest(
                artifact["checkpoints"][str(tier.logical_tokens)]["uninterrupted"][
                    "layers"
                ]
            ),
            "authoritative_state_drift_bytes": max(authoritative)
            - min(authoritative),
            "active_memory_drift_bytes": active_drift,
            "peak_memory_bytes": final_memory["peak_bytes"],
            "elapsed_seconds": elapsed,
            "logical_steps_per_second": tier.logical_tokens / elapsed,
            "cumulative_allocation_target_bytes": tier.cumulative_allocation_target_bytes,
            "cumulative_allocated_bytes": end_accounting[
                "cumulative_allocated_bytes"
            ],
            "cumulative_allocated_tokens": end_accounting[
                "cumulative_allocated_tokens"
            ],
            "lifecycle_accounting": {
                "start": artifact["lifecycle_start"],
                "end": end_accounting,
                "delta_by_lifecycle": lifecycle_accounting_delta(
                    artifact["lifecycle_start"], end_accounting
                ),
            },
            "observer_total_ms": artifact["observer_total_ms"],
            "decode_median_ms": {
                "uninterrupted": float(np.median(latencies_a)) * 1000.0,
                "eventful": float(np.median(latencies_b)) * 1000.0,
            },
        }
        artifact["acceptance"] = {**acceptance, "accepted": all(acceptance.values())}
        artifact["complete"] = True
        artifact["final_memory"] = final_memory
        _atomic_write(output_path, artifact)
    except BaseException as exc:
        if artifact.get("first_divergence") is None:
            try:
                state_difference = first_kda_state_difference(
                    cache_a, cache_b, kda_layers=KDA_LAYERS, mx_module=mx
                )
            except BaseException as diagnostic_error:
                state_difference = {
                    "diagnostic_error": (
                        f"{type(diagnostic_error).__name__}: {diagnostic_error}"
                    )
                }
            artifact["first_divergence"] = {
                "reason": f"{type(exc).__name__}: {exc}",
                "operation_sequence": counters["operation_sequence"],
                "logical_token": current_step,
                "allocation_sequence": _allocation_sequence(accounting),
                "lifecycle": accounting.snapshot(),
                "ownership_state": {
                    "live_a": "active-recurrent+target-prefix",
                    "live_b": "active-recurrent+target-prefix",
                },
                "apc_generation": counters["apc_generation"],
                "operation_kind": operation_context["kind"],
                "rollback_depth": operation_context["rollback_depth"],
                "resident_bytes_before": operation_context[
                    "resident_bytes_before"
                ],
                "resident_bytes_after": operation_context["resident_bytes_after"],
                "cumulative_allocated_bytes": accounting.snapshot()[
                    "cumulative_allocated_bytes"
                ],
                "first_state_difference": state_difference,
            }
        artifact["complete"] = False
        artifact["metal_error"] = f"{type(exc).__name__}: {exc}"
        _atomic_write(output_path, artifact)
        raise
    finally:
        for source, owner in rollback_snapshots.values():
            _release_accounted_cache(accounting, owner, source)
        _release_cache(cache_a)
        _release_cache(cache_b)
        gc.collect()
        mx.clear_cache()
        mx.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--tier",
        choices=("developer-smoke", "screen", "qualification", "extended"),
        default="screen",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    args = parser.parse_args()
    if not mx.metal.is_available():
        raise RuntimeError("allocation churn stress requires MLX/Metal")

    tier = churn_tier(args.tier)
    report = inspect_checkpoint(args.model, require_server_ready=True)
    config = json.loads((args.model / "config.json").read_text())
    vocab = int(config["text_config"]["vocab_size"])
    artifact = {
        "schema": "glm53-state-cumulative-allocation-churn-v1",
        "date": date.today().isoformat(),
        "machine": platform.machine(),
        "complete": False,
        "probe_only": True,
        "tier": tier.name,
        "logical_tokens": tier.logical_tokens,
        "configured_capacity_tokens": tier.logical_tokens + RESERVE_TAIL,
        "cumulative_allocation_target_bytes": tier.cumulative_allocation_target_bytes,
        "baseline_256k_cumulative_allocation_bytes": (
            BASELINE_256K_CUMULATIVE_ALLOCATION_BYTES
        ),
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "kernel_abi": KERNEL_ABI_VERSION,
        "cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
        "backend": "direct-moe+compact-nope-dsa",
        "materialization_policy": MATERIALIZATION_POLICY,
        "materialization_interval_tokens": MATERIALIZATION_INTERVAL_TOKENS,
        "teacher_forced_token_sha256": _token_sequence_digest(
            tier.logical_tokens, vocab
        ),
        "rollback_plan": [
            {"target": target, "depth": depth}
            for target, depth in rollback_schedule(tier)
        ],
        "invalid_operation_rotation": list(INVALID_OPERATIONS),
        "server_admission_bypassed_inside_probe_only": True,
        "runtime_changes": {
            "abi": False,
            "backend": False,
            "server": False,
            "admission": False,
            "apc": False,
        },
        "checkpoints": {},
        "events": [],
        "materializations": [],
        "observer_total_ms": 0.0,
        "last_exact_checkpoint": 0,
        "last_completed_step": 0,
        "first_divergence": None,
        "nan_count": 0,
        "metal_error": None,
    }
    _atomic_write(args.output, artifact)
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    try:
        model, _ = load(
            args.model,
            experimental_compact_nope_dsa_cache=True,
            compact_cache_capacity_tokens=tier.logical_tokens + RESERVE_TAIL,
        )
        warm_residency(model)
        run_churn(model, artifact, args.output, vocab=vocab)
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "phase": "failure",
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "output": str(args.output),
                }
            ),
            flush=True,
        )
        return 130 if isinstance(exc, KeyboardInterrupt) else 1

    print(
        json.dumps(
            {
                "phase": "result",
                "output": str(args.output),
                "acceptance": artifact["acceptance"],
            }
        ),
        flush=True,
    )
    return 0 if artifact["acceptance"]["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
