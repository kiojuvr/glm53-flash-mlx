#!/usr/bin/env python3
"""Soak two exact cache evolutions with layerwise KDA state diagnostics.

The uninterrupted and eventful arms run in lockstep over the same deterministic
teacher-forced tokens.  The eventful arm injects RAM APC save/load, 1/8/16-token
rollback/replay, and one rejected 17-token rollback.  Only the 4,096-token
screen is intended for routine development; 100k/256k tiers are explicit
operator-run qualifications using this same script.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import tempfile
import time
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
from glm53_flash_mlx.cache_lifecycle import CacheLifecycle
from glm53_flash_mlx.kda_digest import (
    LAYERWISE_KDA_DIGEST_SCHEMA,
    SoakLifecycleAccounting,
    aggregate_layer_digest,
    apc_event_steps,
    compare_layerwise_digests,
    first_kda_state_difference,
    layerwise_kda_digests,
    observation_steps,
    rollback_events,
    steady_active_memory_drift,
)
from glm53_flash_mlx.kda_state import KDAStateIndexError, rollback_restore_state
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import EXPECTED_DSA, inspect_checkpoint
from glm53_flash_mlx.materialization import (
    MATERIALIZATION_INTERVAL_TOKENS,
    MATERIALIZATION_POLICY,
)
from glm53_flash_mlx.nope_cache import (
    CompactIndexPoolCache,
    SingleNoPELatentCache,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-layerwise-kda-state-digest-screen-20260902.json"
)
KDA_LAYERS = tuple(layer for layer in range(45) if layer not in EXPECTED_DSA)
SMOKE_STEPS = 256
SCREEN_STEPS = 4_096
QUALIFICATION_STEPS = 100_000
EXTENDED_STEPS = 256_000
MAX_ACTIVE_DRIFT = 64 * 2**20
MAX_PEAK_BYTES = 340_000_000_000
MAX_DIAGNOSTIC_REGRESSION = 0.01
EXPECTED_STATE_LEAVES = 167
RESERVE_TAIL = 16


class SoakDivergenceError(RuntimeError):
    pass


def _atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _memory() -> dict[str, int]:
    mx.synchronize()
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }


def _materialize(cache) -> float:
    started = time.perf_counter()
    mx.eval([entry.state for entry in cache])
    mx.clear_cache()
    mx.synchronize()
    return (time.perf_counter() - started) * 1000.0


def _release_cache(cache) -> None:
    if isinstance(cache, list):
        cache.clear()
    gc.collect()


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
        raise RuntimeError("RAM APC clone rejected a cache entry")
    if targets:
        mx.eval(*targets)
    mx.synchronize()
    return cloned


def _token_for_step(step: int, vocab: int) -> int:
    # Stateless schedule: replay and both arms derive exactly the same token.
    return int((int(step) * 1_103_515_245 + 12_345) % (vocab - 256) + 128)


def _token_sequence_digest(steps: int, vocab: int) -> str:
    digest = hashlib.sha256()
    for step in range(1, steps + 1):
        digest.update(_token_for_step(step, vocab).to_bytes(4, "little"))
    return digest.hexdigest()


def _logits_hash(logits) -> str:
    value = np.ascontiguousarray(np.asarray(logits.astype(mx.float32)))
    return hashlib.sha256(value.tobytes()).hexdigest()


def _state_leaf_count(cache) -> int:
    return len(tree_flatten([entry.state for entry in cache]))


def _cache_groups(cache) -> tuple[int, int, int]:
    kda_bytes = sum(int(cache[layer].nbytes) for layer in KDA_LAYERS)
    dsa_bytes = sum(int(cache[layer].nbytes) for layer in EXPECTED_DSA)
    physical = []
    for layer in EXPECTED_DSA:
        latent, pool = cache[layer]
        if not isinstance(latent, SingleNoPELatentCache) or not isinstance(
            pool, CompactIndexPoolCache
        ):
            raise TypeError("layerwise KDA soak requires compact NoPE DSA cache")
        if latent.physical_capacity_tokens:
            physical.append(int(latent.physical_capacity_tokens))
        elif latent.capacity_tokens:
            physical.append(int(latent.capacity_tokens))
    return kda_bytes, dsa_bytes, max(physical, default=0)


def _cache_binding_signature(cache) -> tuple:
    rows = []
    for layer in KDA_LAYERS:
        entry = cache[layer]
        rows.append(
            (
                layer,
                tuple(
                    None
                    if value is None
                    else (id(value), tuple(value.shape), str(value.dtype))
                    for value in entry.cache
                ),
                int(getattr(entry, "_left_padding_advance", 0)),
                int(getattr(entry, "_lengths_advance", 0)),
            )
        )
    return tuple(rows)


def _register_cache(
    accounting: SoakLifecycleAccounting,
    owner: str,
    cache,
    *,
    snapshot: bool,
) -> None:
    kda_bytes, dsa_bytes, physical_tokens = _cache_groups(cache)
    if snapshot:
        accounting.allocate(
            f"{owner}:kda",
            CacheLifecycle.SNAPSHOT_STATE,
            resident_bytes=kda_bytes,
        )
        accounting.allocate(
            f"{owner}:dsa",
            CacheLifecycle.SNAPSHOT_STATE,
            resident_bytes=dsa_bytes,
            physical_tokens=physical_tokens,
        )
    else:
        accounting.allocate(
            f"{owner}:kda",
            CacheLifecycle.ACTIVE_RECURRENT,
            resident_bytes=kda_bytes,
        )
        accounting.allocate(
            f"{owner}:dsa",
            CacheLifecycle.TARGET_PREFIX,
            resident_bytes=dsa_bytes,
            physical_tokens=physical_tokens,
        )


def _refresh_live(accounting: SoakLifecycleAccounting, owner: str, cache) -> None:
    kda_bytes, dsa_bytes, physical_tokens = _cache_groups(cache)
    accounting.resize(f"{owner}:kda", resident_bytes=kda_bytes)
    accounting.resize(f"{owner}:dsa", resident_bytes=dsa_bytes)
    accounting.update_physical_tokens(
        f"{owner}:dsa", physical_tokens=physical_tokens
    )


def _release_accounted_cache(accounting, owner: str, cache) -> None:
    accounting.release(f"{owner}:kda")
    accounting.release(f"{owner}:dsa")
    _release_cache(cache)


def _promote_snapshot_to_live(
    accounting: SoakLifecycleAccounting,
    snapshot_owner: str,
    live_owner: str,
) -> None:
    accounting.reclassify(
        f"{snapshot_owner}:kda",
        f"{live_owner}:kda",
        CacheLifecycle.ACTIVE_RECURRENT,
    )
    accounting.reclassify(
        f"{snapshot_owner}:dsa",
        f"{live_owner}:dsa",
        CacheLifecycle.TARGET_PREFIX,
    )


def _observe(cache, *, accounting_snapshot: dict, counters: dict) -> tuple[dict, float]:
    binding_before = _cache_binding_signature(cache)
    counters_before = json.dumps(counters, sort_keys=True)
    accounting_before = json.dumps(accounting_snapshot, sort_keys=True)
    started = time.perf_counter()
    rows = layerwise_kda_digests(cache, kda_layers=KDA_LAYERS, mx_module=mx)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    binding_after = _cache_binding_signature(cache)
    observation = {
        "schema": LAYERWISE_KDA_DIGEST_SCHEMA,
        "layers": rows,
        "aggregate_digest": aggregate_layer_digest(rows),
        "state_leaf_count": _state_leaf_count(cache),
        "authoritative_cache_bytes": sum(int(entry.nbytes) for entry in cache),
        "observer_state_bindings_unchanged": binding_before == binding_after,
        "observer_counters_unchanged": counters_before
        == json.dumps(counters, sort_keys=True),
        "observer_accounting_unchanged": accounting_before
        == json.dumps(accounting_snapshot, sort_keys=True),
    }
    return observation, elapsed_ms


def _detailed_divergence(left_cache, right_cache, *, step: int, previous_step: int):
    detail = first_kda_state_difference(
        left_cache, right_cache, kda_layers=KDA_LAYERS, mx_module=mx
    )
    return {
        "first_divergent_token": step,
        "previous_exact_checkpoint": previous_step,
        "localization": detail,
    }


def _record_checkpoint(
    artifact: dict,
    *,
    step: int,
    cache_a,
    cache_b,
    counters_a: dict,
    counters_b: dict,
    accounting: SoakLifecycleAccounting,
    pre_materialization: tuple[dict, dict] | None = None,
) -> None:
    accounting_snapshot = accounting.snapshot()
    observation_a, elapsed_a = _observe(
        cache_a, accounting_snapshot=accounting_snapshot, counters=counters_a
    )
    observation_b, elapsed_b = _observe(
        cache_b, accounting_snapshot=accounting_snapshot, counters=counters_b
    )
    difference = compare_layerwise_digests(
        observation_a["layers"], observation_b["layers"]
    )
    previous = int(artifact.get("last_exact_checkpoint", 0))
    if difference is not None:
        artifact["first_divergence"] = {
            **_detailed_divergence(
                cache_a, cache_b, step=step, previous_step=previous
            ),
            "digest_difference": difference,
            "materialization_count": counters_a["materialization_count"],
            "slot_index": {"conv": 0, "recurrent": 1}.get(
                difference.get("state_kind")
            ),
            "cumulative_allocated_tokens": accounting_snapshot[
                "cumulative_allocated_tokens"
            ],
            "lifecycle": accounting_snapshot,
            "memory": _memory(),
        }
        raise SoakDivergenceError(
            f"KDA state diverged at step {step}: {artifact['first_divergence']}"
        )

    pre_exact = True
    if pre_materialization is not None:
        pre_a, pre_b = pre_materialization
        pre_exact = (
            pre_a["aggregate_digest"] == observation_a["aggregate_digest"]
            and pre_b["aggregate_digest"] == observation_b["aggregate_digest"]
        )
        if not pre_exact:
            raise SoakDivergenceError(
                f"materialization changed authoritative KDA state at step {step}"
            )

    memory = _memory()
    live_authoritative_bytes = sum(int(entry.nbytes) for entry in cache_a) + sum(
        int(entry.nbytes) for entry in cache_b
    )
    row = {
        "step": step,
        "uninterrupted": observation_a,
        "eventful": observation_b,
        "layerwise_exact": True,
        "materialization_pre_post_exact": pre_exact,
        "observer_ms": {"uninterrupted": elapsed_a, "eventful": elapsed_b},
        "materialization_count": counters_a["materialization_count"],
        "actual_model_forwards": {
            "uninterrupted": counters_a["actual_model_forwards"],
            "eventful": counters_b["actual_model_forwards"],
        },
        "memory": memory,
        "lifecycle": accounting_snapshot,
        "lifecycle_accounting_exact": accounting_snapshot["resident_bytes"]
        == live_authoritative_bytes,
    }
    artifact["checkpoints"][str(step)] = row
    artifact["last_exact_checkpoint"] = step
    artifact["observer_total_ms"]["uninterrupted"] += elapsed_a
    artifact["observer_total_ms"]["eventful"] += elapsed_b


def _clone_snapshot(
    cache,
    *,
    owner: str,
    capacity_tokens: int,
    accounting: SoakLifecycleAccounting,
):
    cloned = _clone_cache(cache, min_capacity_tokens=capacity_tokens)
    _register_cache(accounting, owner, cloned, snapshot=True)
    digest, _ = _observe(
        cloned,
        accounting_snapshot=accounting.snapshot(),
        counters={"snapshot": True},
    )
    return cloned, digest


def _restore_from_snapshot(
    source,
    *,
    restored_owner: str,
    live_cache,
    capacity_tokens: int,
    accounting: SoakLifecycleAccounting,
):
    restored = _clone_cache(source, min_capacity_tokens=capacity_tokens)
    _register_cache(accounting, restored_owner, restored, snapshot=True)
    _release_accounted_cache(accounting, "B", live_cache)
    _promote_snapshot_to_live(accounting, restored_owner, "B")
    return restored


def _run_forward(model, cache, token: int):
    output = model(mx.array([[token]], dtype=mx.uint32), cache=cache)
    logits = output.logits[0, -1]
    nan = mx.sum(mx.isnan(logits))
    mx.eval(logits, nan)
    mx.synchronize()
    return output, logits, int(nan.item())


def _invalid_rollback_event(cache, artifact: dict, step: int, accounting) -> None:
    before, _ = _observe(
        cache,
        accounting_snapshot=accounting.snapshot(),
        counters={"logical_step": step},
    )
    rejected_kda = False
    rejected_dsa = False
    first_kda = cache[KDA_LAYERS[0]]
    try:
        rollback_restore_state(first_kda, list(first_kda.state), tokens=17)
    except KDAStateIndexError:
        rejected_kda = True
    try:
        cache[EXPECTED_DSA[0]].trim(17)
    except ValueError:
        rejected_dsa = True
    after, _ = _observe(
        cache,
        accounting_snapshot=accounting.snapshot(),
        counters={"logical_step": step},
    )
    row = {
        "kind": "rejected-rollback",
        "step": step,
        "tokens": 17,
        "kda_rejected": rejected_kda,
        "dsa_rejected": rejected_dsa,
        "state_unchanged": before["aggregate_digest"] == after["aggregate_digest"],
    }
    artifact["events"].append(row)
    if not all((rejected_kda, rejected_dsa, row["state_unchanged"])):
        raise SoakDivergenceError("17-token rollback was not atomic")


def run_soak(model, *, artifact: dict, output: Path, vocab: int) -> None:
    steps = artifact["steps"]
    capacity_tokens = artifact["configured_capacity_tokens"]
    checkpoints = set(observation_steps(steps))
    rollback_plan = dict(rollback_events(steps))
    rollback_capture = {target - trim: target for target, trim in rollback_plan.items()}
    apc_steps = set(apc_event_steps(steps))
    invalid_step = min(3_584, max(256, (steps // 2 // 256) * 256))
    accounting = SoakLifecycleAccounting()
    cache_a = model.make_cache()
    cache_b = model.make_cache()
    _register_cache(accounting, "A", cache_a, snapshot=False)
    _register_cache(accounting, "B", cache_b, snapshot=False)
    counters_a = {"logical_step": 0, "actual_model_forwards": 0, "materialization_count": 0}
    counters_b = {"logical_step": 0, "actual_model_forwards": 0, "materialization_count": 0}
    rollback_snapshots = {}
    pending_apc = None
    latencies_a = []
    latencies_b = []
    final_logits_a = final_logits_b = None
    mx.reset_peak_memory()

    # Empty state is the authoritative token-0 baseline.
    _record_checkpoint(
        artifact,
        step=0,
        cache_a=cache_a,
        cache_b=cache_b,
        counters_a=counters_a,
        counters_b=counters_b,
        accounting=accounting,
    )
    repeat_before = _memory()
    repeat_a, _ = _observe(
        cache_a, accounting_snapshot=accounting.snapshot(), counters=counters_a
    )
    repeat_after = _memory()
    artifact["observer_nonmutation_fixture"] = {
        "repeat_digest_exact": repeat_a["aggregate_digest"]
        == artifact["checkpoints"]["0"]["uninterrupted"]["aggregate_digest"],
        "state_bindings_unchanged": repeat_a["observer_state_bindings_unchanged"],
        "counters_unchanged": repeat_a["observer_counters_unchanged"],
        "accounting_unchanged": repeat_a["observer_accounting_unchanged"],
        "surviving_active_allocation_bytes": repeat_after["active_bytes"]
        - repeat_before["active_bytes"],
    }

    try:
        for step in range(1, steps + 1):
            if step - 1 in rollback_capture:
                target = rollback_capture[step - 1]
                owner = f"rollback-{target}-source"
                snapshot, digest = _clone_snapshot(
                    cache_b,
                    owner=owner,
                    capacity_tokens=capacity_tokens,
                    accounting=accounting,
                )
                rollback_snapshots[target] = (snapshot, digest, owner)

            token = _token_for_step(step, vocab)
            started = time.perf_counter()
            output_a, logits_a, nans_a = _run_forward(model, cache_a, token)
            latencies_a.append(time.perf_counter() - started)
            counters_a["actual_model_forwards"] += 1
            counters_a["logical_step"] = step
            started = time.perf_counter()
            output_b, logits_b, nans_b = _run_forward(model, cache_b, token)
            latencies_b.append(time.perf_counter() - started)
            counters_b["actual_model_forwards"] += 1
            counters_b["logical_step"] = step
            artifact["nan_count"] += nans_a + nans_b

            logits_equal = mx.array_equal(logits_a, logits_b)
            mx.eval(logits_equal)
            if not bool(logits_equal.item()):
                artifact["first_divergence"] = {
                    **_detailed_divergence(
                        cache_a,
                        cache_b,
                        step=step,
                        previous_step=artifact["last_exact_checkpoint"],
                    ),
                    "state_kind": "logits",
                }
                raise SoakDivergenceError(f"logits diverged at step {step}")

            if pending_apc is not None:
                source, expected, owner, event_step = pending_apc
                observed, _ = _observe(
                    source,
                    accounting_snapshot=accounting.snapshot(),
                    counters={"snapshot_step": event_step},
                )
                immutable = observed["aggregate_digest"] == expected["aggregate_digest"]
                artifact["events"].append(
                    {
                        "kind": "ram-apc-save-load",
                        "step": event_step,
                        "snapshot_immutable_after_next_decode": immutable,
                    }
                )
                _release_accounted_cache(accounting, owner, source)
                pending_apc = None
                if not immutable:
                    raise SoakDivergenceError("RAM APC source snapshot mutated")

            if step in rollback_plan:
                trim = rollback_plan[step]
                source, source_digest, source_owner = rollback_snapshots.pop(step)
                expected_rows = layerwise_kda_digests(
                    cache_b, kda_layers=KDA_LAYERS, mx_module=mx
                )
                expected_logits = _logits_hash(logits_b)
                restored_owner = f"rollback-{step}-restored"
                old_b = cache_b
                cache_b = _restore_from_snapshot(
                    source,
                    restored_owner=restored_owner,
                    live_cache=old_b,
                    capacity_tokens=capacity_tokens,
                    accounting=accounting,
                )
                replay_logits = None
                for replay_step in range(step - trim + 1, step + 1):
                    replay_output, replay_logits, replay_nans = _run_forward(
                        model,
                        cache_b,
                        _token_for_step(replay_step, vocab),
                    )
                    counters_b["actual_model_forwards"] += 1
                    artifact["nan_count"] += replay_nans
                    del replay_output
                replay_rows = layerwise_kda_digests(
                    cache_b, kda_layers=KDA_LAYERS, mx_module=mx
                )
                source_after, _ = _observe(
                    source,
                    accounting_snapshot=accounting.snapshot(),
                    counters={"snapshot_step": step - trim},
                )
                event = {
                    "kind": "rollback-replay",
                    "step": step,
                    "tokens": trim,
                    "layerwise_state_exact": compare_layerwise_digests(
                        expected_rows, replay_rows
                    )
                    is None,
                    "final_logits_exact": _logits_hash(replay_logits)
                    == expected_logits,
                    "source_snapshot_immutable": source_after["aggregate_digest"]
                    == source_digest["aggregate_digest"],
                }
                artifact["events"].append(event)
                _release_accounted_cache(accounting, source_owner, source)
                logits_b = replay_logits
                if not all(
                    event[key]
                    for key in (
                        "layerwise_state_exact",
                        "final_logits_exact",
                        "source_snapshot_immutable",
                    )
                ):
                    raise SoakDivergenceError(f"rollback replay diverged at {step}")

            _refresh_live(accounting, "A", cache_a)
            _refresh_live(accounting, "B", cache_b)

            if step % MATERIALIZATION_INTERVAL_TOKENS == 0:
                accounting_snapshot = accounting.snapshot()
                pre_a, pre_elapsed_a = _observe(
                    cache_a,
                    accounting_snapshot=accounting_snapshot,
                    counters=counters_a,
                )
                pre_b, pre_elapsed_b = _observe(
                    cache_b,
                    accounting_snapshot=accounting_snapshot,
                    counters=counters_b,
                )
                artifact["observer_total_ms"]["uninterrupted"] += pre_elapsed_a
                artifact["observer_total_ms"]["eventful"] += pre_elapsed_b
                materialization_a = _materialize(cache_a)
                materialization_b = _materialize(cache_b)
                counters_a["materialization_count"] += 1
                counters_b["materialization_count"] += 1
                artifact["materializations"].append(
                    {
                        "step": step,
                        "count": counters_a["materialization_count"],
                        "uninterrupted_ms": materialization_a,
                        "eventful_ms": materialization_b,
                    }
                )
                _record_checkpoint(
                    artifact,
                    step=step,
                    cache_a=cache_a,
                    cache_b=cache_b,
                    counters_a=counters_a,
                    counters_b=counters_b,
                    accounting=accounting,
                    pre_materialization=(pre_a, pre_b),
                )
            elif step in checkpoints:
                _record_checkpoint(
                    artifact,
                    step=step,
                    cache_a=cache_a,
                    cache_b=cache_b,
                    counters_a=counters_a,
                    counters_b=counters_b,
                    accounting=accounting,
                )

            if step == invalid_step:
                _invalid_rollback_event(cache_b, artifact, step, accounting)

            if step in apc_steps:
                source_owner = f"apc-{step}-source"
                restored_owner = f"apc-{step}-restored"
                source, source_digest = _clone_snapshot(
                    cache_b,
                    owner=source_owner,
                    capacity_tokens=capacity_tokens,
                    accounting=accounting,
                )
                restored = _restore_from_snapshot(
                    source,
                    restored_owner=restored_owner,
                    live_cache=cache_b,
                    capacity_tokens=capacity_tokens,
                    accounting=accounting,
                )
                cache_b = restored
                restored_digest, _ = _observe(
                    cache_b,
                    accounting_snapshot=accounting.snapshot(),
                    counters=counters_b,
                )
                if restored_digest["aggregate_digest"] != source_digest["aggregate_digest"]:
                    raise SoakDivergenceError("RAM APC restore was not exact")
                if step == steps:
                    source_after, _ = _observe(
                        source,
                        accounting_snapshot=accounting.snapshot(),
                        counters={"snapshot_step": step},
                    )
                    artifact["events"].append(
                        {
                            "kind": "ram-apc-save-load",
                            "step": step,
                            "snapshot_immutable_after_next_decode": (
                                source_after["aggregate_digest"]
                                == source_digest["aggregate_digest"]
                            ),
                        }
                    )
                    _release_accounted_cache(accounting, source_owner, source)
                else:
                    pending_apc = (source, source_digest, source_owner, step)

            final_logits_a = _logits_hash(logits_a)
            final_logits_b = _logits_hash(logits_b)
            artifact["last_completed_step"] = step
            artifact["actual_model_forwards"] = {
                "uninterrupted": counters_a["actual_model_forwards"],
                "eventful": counters_b["actual_model_forwards"],
            }
            if step % 256 == 0:
                print(
                    json.dumps(
                        {
                            "phase": "checkpoint",
                            "step": step,
                            "materializations": counters_a["materialization_count"],
                            "aggregate": artifact["checkpoints"][str(step)][
                                "uninterrupted"
                            ]["aggregate_digest"],
                        }
                    ),
                    flush=True,
                )
            if step % 4_096 == 0 or step == steps:
                _atomic_write(output, artifact)

            del output_a, output_b, logits_a, logits_b, logits_equal

        if pending_apc is not None:
            source, expected, owner, event_step = pending_apc
            source_after, _ = _observe(
                source,
                accounting_snapshot=accounting.snapshot(),
                counters={"snapshot_step": event_step},
            )
            immutable = source_after["aggregate_digest"] == expected["aggregate_digest"]
            artifact["events"].append(
                {
                    "kind": "ram-apc-save-load",
                    "step": event_step,
                    "snapshot_immutable_after_next_decode": immutable,
                }
            )
            _release_accounted_cache(accounting, owner, source)
            if not immutable:
                raise SoakDivergenceError("final RAM APC source snapshot mutated")

        final_memory = _memory()
        rows = [artifact["checkpoints"][step] for step in artifact["checkpoints"] if step != "0"]
        steady_rows = [row for row in rows if int(row["step"]) >= 256]
        active_memory_drift = steady_active_memory_drift(
            artifact["checkpoints"],
            first_steady_step=MATERIALIZATION_INTERVAL_TOKENS,
        )
        authoritative = [
            row["uninterrupted"]["authoritative_cache_bytes"]
            + row["eventful"]["authoritative_cache_bytes"]
            for row in steady_rows
        ]
        base_decode_ms_a = sum(latencies_a) * 1000.0
        base_decode_ms_b = sum(latencies_b) * 1000.0
        observer_a = artifact["observer_total_ms"]["uninterrupted"]
        observer_b = artifact["observer_total_ms"]["eventful"]
        overhead_a = observer_a / base_decode_ms_a
        overhead_b = observer_b / base_decode_ms_b
        event_rows = artifact["events"]
        acceptance = {
            "completed_requested_steps": artifact["last_completed_step"] == steps,
            "all_34_kda_layers_observed": all(
                len(row["uninterrupted"]["layers"]) == len(KDA_LAYERS) == 34
                for row in artifact["checkpoints"].values()
            ),
            "conv_recurrent_index_digests_independent": all(
                all(
                    all(key in layer for key in ("conv_digest", "recurrent_digest", "index_digest"))
                    for layer in row["uninterrupted"]["layers"]
                )
                for row in artifact["checkpoints"].values()
            ),
            "observer_non_mutating": all(
                all(
                    observation[key]
                    for key in (
                        "observer_state_bindings_unchanged",
                        "observer_counters_unchanged",
                        "observer_accounting_unchanged",
                    )
                )
                for row in artifact["checkpoints"].values()
                for observation in (row["uninterrupted"], row["eventful"])
            )
            and artifact["observer_nonmutation_fixture"]["repeat_digest_exact"]
            and artifact["observer_nonmutation_fixture"][
                "surviving_active_allocation_bytes"
            ]
            == 0,
            "materialization_cadence_exact": (
                len(artifact["materializations"])
                == steps // MATERIALIZATION_INTERVAL_TOKENS
                and counters_a["materialization_count"]
                == counters_b["materialization_count"]
                == steps // MATERIALIZATION_INTERVAL_TOKENS
            ),
            "materialization_pre_post_state_exact": all(
                row["materialization_pre_post_exact"]
                for step, row in artifact["checkpoints"].items()
                if int(step) % MATERIALIZATION_INTERVAL_TOKENS == 0 and int(step) > 0
            ),
            "uninterrupted_eventful_all_checkpoints_exact": artifact[
                "first_divergence"
            ]
            is None
            and all(row["layerwise_exact"] for row in artifact["checkpoints"].values()),
            "apc_and_rollback_events_exact": all(
                row.get("state_unchanged", True)
                and row.get("layerwise_state_exact", True)
                and row.get("final_logits_exact", True)
                and row.get("source_snapshot_immutable", True)
                and row.get("snapshot_immutable_after_next_decode", True)
                for row in event_rows
            ),
            "final_logits_hash_exact": final_logits_a == final_logits_b,
            "authoritative_state_drift_zero": max(authoritative) == min(authoritative),
            "state_leaf_count_constant": all(
                observation["state_leaf_count"] == EXPECTED_STATE_LEAVES
                for row in artifact["checkpoints"].values()
                if int(row["step"]) > 0
                for observation in (row["uninterrupted"], row["eventful"])
            ),
            "resident_memory_bounded": active_memory_drift <= MAX_ACTIVE_DRIFT,
            "lifecycle_anonymous_allocation_zero": accounting.snapshot()[
                "anonymous_allocation_count"
            ]
            == 0,
            "cumulative_allocation_monotonic": all(
                right["lifecycle"]["cumulative_allocated_bytes"]
                >= left["lifecycle"]["cumulative_allocated_bytes"]
                for left, right in zip(rows, rows[1:])
            )
            and all(row["lifecycle_accounting_exact"] for row in rows),
            "no_nan_invalid_index_or_metal_error": artifact["nan_count"] == 0,
            "repeat_digest_sequence_identical": all(
                row["uninterrupted"]["aggregate_digest"]
                == row["eventful"]["aggregate_digest"]
                for row in artifact["checkpoints"].values()
            ),
            "diagnostic_decode_regression_at_most_1_percent": max(overhead_a, overhead_b)
            <= MAX_DIAGNOSTIC_REGRESSION,
            "peak_memory_at_most_340_gb": final_memory["peak_bytes"] <= MAX_PEAK_BYTES,
        }
        artifact["summary"] = {
            "logical_decode_tokens": steps,
            "checkpoint_count": len(artifact["checkpoints"]),
            "periodic_256_checkpoint_count": sum(
                int(step) > 0 and int(step) % 256 == 0
                for step in artifact["checkpoints"]
            ),
            "materialization_count": counters_a["materialization_count"],
            "first_divergence": artifact["first_divergence"],
            "final_logits_hash": final_logits_a,
            "final_layerwise_digest": artifact["checkpoints"][str(steps)][
                "uninterrupted"
            ]["aggregate_digest"],
            "authoritative_state_drift_bytes": max(authoritative) - min(authoritative),
            "initialization_authoritative_growth_bytes": (
                rows[0]["uninterrupted"]["authoritative_cache_bytes"]
                + rows[0]["eventful"]["authoritative_cache_bytes"]
                - artifact["checkpoints"]["0"]["uninterrupted"][
                    "authoritative_cache_bytes"
                ]
                - artifact["checkpoints"]["0"]["eventful"][
                    "authoritative_cache_bytes"
                ]
            ),
            "active_memory_drift_bytes": active_memory_drift,
            "active_memory_drift_window": (
                "first-production-materialization-through-final-checkpoint"
            ),
            "peak_memory_bytes": final_memory["peak_bytes"],
            "state_leaf_counts": sorted(
                {
                    observation["state_leaf_count"]
                    for row in artifact["checkpoints"].values()
                    if int(row["step"]) > 0
                    for observation in (row["uninterrupted"], row["eventful"])
                }
            ),
            "observer_overhead": {
                "uninterrupted_ratio": overhead_a,
                "eventful_ratio": overhead_b,
                "uninterrupted_ms": observer_a,
                "eventful_ms": observer_b,
                "counterfactual_disabled_decode_ms": {
                    "uninterrupted": base_decode_ms_a,
                    "eventful": base_decode_ms_b,
                },
            },
            "lifecycle": accounting.snapshot(),
            "event_count": len(event_rows),
        }
        artifact["acceptance"] = {
            **acceptance,
            "accepted": all(acceptance.values()),
        }
        artifact["complete"] = True
        artifact["final_memory"] = final_memory
        _atomic_write(output, artifact)
    finally:
        _release_cache(cache_a)
        _release_cache(cache_b)
        mx.clear_cache()
        mx.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--steps",
        type=int,
        choices=(SMOKE_STEPS, SCREEN_STEPS, QUALIFICATION_STEPS, EXTENDED_STEPS),
        default=SCREEN_STEPS,
    )
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    args = parser.parse_args()
    if not mx.metal.is_available():
        raise RuntimeError("layerwise KDA soak requires MLX/Metal")

    report = inspect_checkpoint(args.model, require_server_ready=True)
    config = json.loads((args.model / "config.json").read_text())
    vocab = int(config["text_config"]["vocab_size"])
    capacity_tokens = args.steps + RESERVE_TAIL
    artifact = {
        "schema": "glm53-layerwise-kda-state-digest-soak-v1",
        "date": date.today().isoformat(),
        "machine": platform.machine(),
        "complete": False,
        "probe_only": True,
        "tier": {
            SMOKE_STEPS: "developer-smoke",
            SCREEN_STEPS: "screen",
            QUALIFICATION_STEPS: "qualification",
            EXTENDED_STEPS: "extended",
        }[args.steps],
        "steps": args.steps,
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "kernel_abi": KERNEL_ABI_VERSION,
        "cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
        "backend": "direct-moe+compact-nope-dsa",
        "configured_capacity_tokens": capacity_tokens,
        "materialization_policy": MATERIALIZATION_POLICY,
        "materialization_interval_tokens": MATERIALIZATION_INTERVAL_TOKENS,
        "allocation_accounting_contract": {
            "resident_bytes": "authoritative cache entry nbytes by lifecycle",
            "cumulative_allocated_bytes": (
                "positive authoritative growth plus every RAM cache incarnation"
            ),
            "cumulative_allocated_tokens": (
                "maximum physical DSA sequence capacity per cache incarnation"
            ),
            "anonymous_allocations": "forbidden",
        },
        "kda_layers": list(KDA_LAYERS),
        "dsa_layers": list(EXPECTED_DSA),
        "observation_steps": list(observation_steps(args.steps)),
        "rollback_plan": [
            {"target": target, "tokens": trim}
            for target, trim in rollback_events(args.steps)
        ],
        "ram_apc_steps": list(apc_event_steps(args.steps)),
        "teacher_forced_token_sha256": _token_sequence_digest(args.steps, vocab),
        "server_admission_bypassed_inside_probe_only": True,
        "disk_resume_supported": False,
        "process_resume_supported": False,
        "checkpoints": {},
        "materializations": [],
        "events": [],
        "observer_total_ms": {"uninterrupted": 0.0, "eventful": 0.0},
        "observer_nonmutation_fixture": None,
        "first_divergence": None,
        "last_exact_checkpoint": 0,
        "last_completed_step": 0,
        "actual_model_forwards": {"uninterrupted": 0, "eventful": 0},
        "nan_count": 0,
        "metal_error": None,
        "runtime_changes": {
            "abi": False,
            "apc_namespace": False,
            "backend": False,
            "admission": False,
            "server": False,
        },
    }
    _atomic_write(args.output, artifact)
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    try:
        model, _ = load(
            args.model,
            experimental_compact_nope_dsa_cache=True,
            compact_cache_capacity_tokens=capacity_tokens,
        )
        warm_residency(model)
        run_soak(model, artifact=artifact, output=args.output, vocab=vocab)
    except BaseException as exc:
        artifact["metal_error"] = f"{type(exc).__name__}: {exc}"
        artifact["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "last_completed_step": artifact["last_completed_step"],
        }
        artifact["complete"] = False
        _atomic_write(args.output, artifact)
        print(json.dumps({"phase": "failure", **artifact["failure"]}), flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1

    print(
        json.dumps(
            {
                "phase": "result",
                "output": str(args.output),
                "complete": artifact["complete"],
                "acceptance": artifact["acceptance"],
            }
        ),
        flush=True,
    )
    return 0 if artifact["acceptance"]["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
