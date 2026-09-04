#!/usr/bin/env python3
"""Qualify repeated 4K replay from an immutable hybrid semantic snapshot.

The default run performs 38,912 sequential decode forwards plus one 256-token
prefill call and is intentionally a user-launched qualification.  Progress is
atomically checkpointed after the baseline and every replay.
The cache itself is not serialized or resumable.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import weakref
from dataclasses import replace
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
    HybridSemanticPrefixSnapshot,
    SemanticCacheHandle,
    SemanticSnapshotError,
    SemanticSnapshotIdentity,
    SemanticSnapshotStore,
    semantic_cache_digest,
    semantic_component_digests,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-hybrid-semantic-prefix-snapshot-replay-20260904.json"
)
DEFAULT_PREFIX_TOKENS = 256
DEFAULT_STEPS = 4096
DEFAULT_REPLAYS = 8
DEFAULT_NESTED_STEP = 2048
CHECKPOINT_INTERVAL = MATERIALIZATION_INTERVAL_TOKENS
MAX_ACTIVE_DRIFT = 64 << 20


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


def _hash_array(value: mx.array) -> str:
    return hashlib.sha256(_raw(value)).hexdigest()


def _token_digest(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def _tokens(count: int, vocab: int) -> list[int]:
    return [int((index * 7919) % (vocab - 1024) + 100) for index in range(count)]


def _memory() -> dict[str, int]:
    mx.synchronize()
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }


def _materialize(cache) -> None:
    mx.eval([entry.state for entry in cache])
    mx.clear_cache()
    mx.synchronize()


def _identity(report, token_ids: list[int], length: int) -> SemanticSnapshotIdentity:
    return SemanticSnapshotIdentity(
        checkpoint_revision=report.official_revision,
        checkpoint_fingerprint=report.fingerprint,
        moe_backend="packed-decode",
        cache_backend="compact-nope-dsa",
        attention_cache_abi=NOPE_DSA_CACHE_ABI_COMPACT,
        kda_state_abi=KDA_STATE_ABI,
        indexpool_abi=COMPACT_INDEXPOOL_ABI,
        prefix_token_sha256=_token_digest(token_ids[:length]),
    )


def _checkpoint(cache, *, step: int, absolute_position: int) -> dict:
    components = semantic_component_digests(cache)
    return {
        "step": step,
        "absolute_position": absolute_position,
        "state_sha256": semantic_cache_digest(cache),
        "components": components,
        "memory": _memory(),
    }


def _run_trajectory(
    model,
    handle: SemanticCacheHandle,
    token_ids: list[int],
    *,
    start_position: int,
    steps: int,
    nested_step: int | None = None,
    nested_capture=None,
) -> dict:
    logits_hashes = []
    checkpoints = {}
    materializations = []
    nan_count = 0
    started = time.perf_counter()
    for step in range(1, steps + 1):
        token_position = start_position + step - 1
        output = model(
            mx.array([[token_ids[token_position]]], dtype=mx.uint32),
            cache=handle.cache,
        )
        logits = output.logits[0, -1]
        nan = mx.sum(mx.isnan(logits))
        mx.eval(logits, nan)
        logits_hashes.append(_hash_array(logits))
        nan_count += int(nan.item())
        absolute_position = start_position + step
        if absolute_position % CHECKPOINT_INTERVAL == 0:
            _materialize(handle.cache)
            materializations.append(absolute_position)
        if nested_step is not None and step == nested_step:
            _materialize(handle.cache)
            nested_capture(handle, absolute_position)
        if step % CHECKPOINT_INTERVAL == 0 or step == steps:
            checkpoints[str(absolute_position)] = _checkpoint(
                handle.cache,
                step=step,
                absolute_position=absolute_position,
            )
            _progress(
                "trajectory-checkpoint",
                step=step,
                absolute_position=absolute_position,
            )
    elapsed = time.perf_counter() - started
    return {
        "steps": steps,
        "elapsed_seconds": elapsed,
        "tokens_per_second": steps / elapsed,
        "logits_hashes": logits_hashes,
        "logits_sequence_sha256": _token_digest(
            int(value[:8], 16) & 0xFFFFFFFF for value in logits_hashes
        ),
        "checkpoints": checkpoints,
        "materialization_positions": materializations,
        "materialization_count": len(materializations),
        "final_state_sha256": semantic_cache_digest(handle.cache),
        "final_components": semantic_component_digests(handle.cache),
        "nan_count": nan_count,
        "end_memory": _memory(),
    }


def _compare_trajectory(
    reference: dict,
    candidate: dict,
    *,
    reference_logits_offset: int = 0,
) -> dict:
    expected_logits = reference["logits_hashes"][
        reference_logits_offset : reference_logits_offset + candidate["steps"]
    ]
    mismatches = [
        index + 1
        for index, (left, right) in enumerate(
            zip(expected_logits, candidate["logits_hashes"], strict=True)
        )
        if left != right
    ]
    reference_checkpoints = {
        key: value
        for key, value in reference["checkpoints"].items()
        if int(key) in {int(row) for row in candidate["checkpoints"]}
    }
    checkpoint_mismatches = []
    for position, row in candidate["checkpoints"].items():
        expected = reference_checkpoints.get(position)
        if expected is None or any(
            row[key] != expected[key]
            for key in ("state_sha256", "components")
        ):
            checkpoint_mismatches.append(int(position))
    return {
        "full_vocab_logits_exact": not mismatches,
        "logits_mismatch_steps": mismatches,
        "all_checkpoint_state_exact": not checkpoint_mismatches,
        "checkpoint_mismatch_positions": checkpoint_mismatches,
        "final_state_exact": (
            candidate["final_state_sha256"]
            == reference["final_state_sha256"]
            if reference_logits_offset == 0
            else not checkpoint_mismatches
        ),
        "kda_state_exact": not checkpoint_mismatches and all(
            row["components"]["kda_state_sha256"]
            == reference_checkpoints[position]["components"]["kda_state_sha256"]
            for position, row in candidate["checkpoints"].items()
        ),
        "dsa_kv_exact": not checkpoint_mismatches and all(
            row["components"]["dsa_kv_sha256"]
            == reference_checkpoints[position]["components"]["dsa_kv_sha256"]
            for position, row in candidate["checkpoints"].items()
        ),
        "indexpool_exact": not checkpoint_mismatches and all(
            row["components"]["indexpool_sha256"]
            == reference_checkpoints[position]["components"]["indexpool_sha256"]
            for position, row in candidate["checkpoints"].items()
        ),
        "slot_index_metadata_exact": not checkpoint_mismatches and all(
            row["components"]["slot_index_metadata_sha256"]
            == reference_checkpoints[position]["components"][
                "slot_index_metadata_sha256"
            ]
            for position, row in candidate["checkpoints"].items()
        ),
        "nan_count": candidate["nan_count"],
    }


def _snapshot_alias_count(
    left: HybridSemanticPrefixSnapshot,
    right: HybridSemanticPrefixSnapshot,
) -> int:
    def arrays(cache):
        for entry in cache:
            state = entry.state
            stack = [state]
            while stack:
                value = stack.pop()
                if isinstance(value, mx.array):
                    yield value
                elif isinstance(value, (tuple, list)):
                    stack.extend(reversed(value))
                elif isinstance(value, dict):
                    stack.extend(value[key] for key in sorted(value, reverse=True))

    left_arrays = list(arrays(left._cache))
    right_arrays = list(arrays(right._cache))
    return sum(a is b for a in left_arrays for b in right_arrays)


def _restore_and_check_stale(
    store: SemanticSnapshotStore,
    snapshot_id: str,
    handle: SemanticCacheHandle,
    identity: SemanticSnapshotIdentity,
) -> dict:
    previous_cache_id = id(handle.cache)
    old_entries = [weakref.ref(entry) for entry in handle.cache]
    generation_before = handle.generation
    snapshot_resident_before = store.accounting()["snapshot_owned_bytes"]
    store.restore(snapshot_id, handle, expected_identity=identity)
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    return {
        "previous_cache_id": previous_cache_id,
        "replacement_cache_id": id(handle.cache),
        "cache_reference_replaced": id(handle.cache) != previous_cache_id,
        "generation_before": generation_before,
        "generation_after": handle.generation,
        "stale_entry_reference_count": sum(ref() is not None for ref in old_entries),
        "snapshot_resident_before": snapshot_resident_before,
        "snapshot_resident_after": store.accounting()["snapshot_owned_bytes"],
    }


def _failure_injections(
    store: SemanticSnapshotStore,
    snapshot: HybridSemanticPrefixSnapshot,
    handle: SemanticCacheHandle,
    identity: SemanticSnapshotIdentity,
) -> dict:
    from mlx_vlm.apc_adapters import clone_cache_entry

    rows = {}

    def run(name: str, operation) -> None:
        before_ref = handle.cache
        before_generation = handle.generation
        before_digest = semantic_cache_digest(handle.cache)
        snapshot_before = snapshot.state_sha256
        rejected = False
        message = None
        try:
            operation()
        except (SemanticSnapshotError, RuntimeError) as error:
            rejected = True
            message = str(error)
        rows[name] = {
            "rejected": rejected,
            "message": message,
            "live_reference_unchanged": handle.cache is before_ref,
            "live_generation_unchanged": handle.generation == before_generation,
            "live_state_unchanged": semantic_cache_digest(handle.cache) == before_digest,
            "snapshot_unchanged": snapshot.state_sha256 == snapshot_before,
        }

    run(
        "identity_mismatch",
        lambda: store.restore(
            snapshot.snapshot_id,
            handle,
            expected_identity=replace(identity, checkpoint_revision="wrong-revision"),
        ),
    )

    calls = 0

    def corrupt_one_component(entry, **kwargs):
        nonlocal calls
        replacement = clone_cache_entry(entry, **kwargs)
        calls += 1
        if calls == 1:
            state = replacement.state
            if isinstance(state, list) and state[0] is not None:
                replacement[0] = state[0] + mx.array(1, dtype=state[0].dtype)
        return replacement

    run(
        "corrupt_component",
        lambda: store.restore(
            snapshot.snapshot_id,
            handle,
            expected_identity=identity,
            clone_entry=corrupt_one_component,
        ),
    )
    invalid_boundary = replace(
        snapshot.boundary,
        kv_logical_extents=tuple(
            (layer, extent + (1 if index == 0 else 0))
            for index, (layer, extent) in enumerate(snapshot.boundary.kv_logical_extents)
        ),
    )
    forged = replace(snapshot, boundary=invalid_boundary)
    run(
        "invalid_extent_metadata",
        lambda: handle.restore(forged, expected_identity=identity),
    )
    return rows


def _official_oracle(model, processor, report) -> dict:
    scripts = str(REPOSITORY / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import probe_exact_sigmoid_gate_metal_barrier as oracle_probe

    return oracle_probe._official_oracle(model, processor, report)


def _validate_args(args) -> None:
    for name in ("prefix_tokens", "steps", "replays", "nested_step"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.nested_step >= args.steps:
        raise ValueError("nested_step must be less than steps")
    if args.steps % CHECKPOINT_INTERVAL:
        raise ValueError("steps must be divisible by the checkpoint interval")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix-tokens", type=int, default=DEFAULT_PREFIX_TOKENS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--replays", type=int, default=DEFAULT_REPLAYS)
    parser.add_argument("--nested-step", type=int, default=DEFAULT_NESTED_STEP)
    args = parser.parse_args()
    _validate_args(args)
    if not mx.metal.is_available():
        raise RuntimeError("semantic snapshot replay qualification requires MLX/Metal")

    report = inspect_checkpoint(args.model, require_server_ready=True)
    mx.set_wired_limit(int(440e9))
    mx.set_cache_limit(int(32e9))
    capacity = args.prefix_tokens + args.steps
    load_started = time.perf_counter()
    model, processor = load(
        args.model,
        experimental_packed_decode_moe=True,
        experimental_compact_nope_dsa_cache=True,
        compact_cache_capacity_tokens=capacity,
    )
    load_seconds = time.perf_counter() - load_started
    warm_started = time.perf_counter()
    warm_residency(model)
    warm_seconds = time.perf_counter() - warm_started
    vocab = int(model.language_model.lm_head.weight.shape[0])
    token_ids = _tokens(capacity, vocab)
    cache = model.make_cache()
    handle = SemanticCacheHandle(cache)
    # Ownership has moved to the handle.  Keeping this local list alive would
    # manufacture a stale-generation reference during the first restore.
    del cache
    prefix = mx.array(token_ids[: args.prefix_tokens], dtype=mx.uint32)[None]
    prefix_output = model(prefix, cache=handle.cache)
    mx.eval(prefix_output.logits)
    del prefix_output
    _materialize(handle.cache)
    identity_s0 = _identity(report, token_ids, args.prefix_tokens)
    store = SemanticSnapshotStore()
    s0 = store.capture(
        handle,
        snapshot_id="S0",
        identity=identity_s0,
        absolute_token_position=args.prefix_tokens,
        materialization_epoch=args.prefix_tokens // CHECKPOINT_INTERVAL,
    )
    s0_initial_digest = s0.state_sha256
    s1_holder = {}

    def capture_s1(live_handle, absolute_position):
        identity = _identity(report, token_ids, absolute_position)
        s0_before = s0.state_sha256
        s1 = store.capture(
            live_handle,
            snapshot_id="S1",
            identity=identity,
            absolute_token_position=absolute_position,
            materialization_epoch=absolute_position // CHECKPOINT_INTERVAL,
        )
        s1_holder.update(
            {
                "snapshot": s1,
                "identity": identity,
                "s1_initial_digest": s1.state_sha256,
                "s0_immutable_during_s1_capture": s0.state_sha256 == s0_before,
                "s0_s1_storage_alias_count": _snapshot_alias_count(s0, s1),
            }
        )

    artifact = {
        "schema": "glm53-hybrid-semantic-prefix-snapshot-replay-v1",
        "snapshot_contract_schema": SEMANTIC_PREFIX_SNAPSHOT_SCHEMA,
        "date": str(date.today()),
        "complete": False,
        "last_completed_phase": "loaded",
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "model_path": str(args.model.resolve()),
        "load_seconds": load_seconds,
        "warm_residency_seconds": warm_seconds,
        "configuration": {
            "prefix_tokens": args.prefix_tokens,
            "steps": args.steps,
            "replays": args.replays,
            "nested_step": args.nested_step,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "expected_checkpoints_per_replay": args.steps // CHECKPOINT_INTERVAL,
            "decode_model_forward_calls": args.steps
            + args.replays * args.steps
            + (args.steps - args.nested_step),
            "prefill_model_forward_calls": 1,
            "model_forward_calls": 1
            + args.steps
            + args.replays * args.steps
            + (args.steps - args.nested_step),
            "model_token_positions": args.prefix_tokens
            + args.steps
            + args.replays * args.steps
            + (args.steps - args.nested_step),
        },
        "backend": {"moe": "packed-decode", "cache": "compact-nope-dsa"},
        "baseline": None,
        "replays": [],
        "nested_replay": None,
        "failure_injections": None,
        "official_oracle": None,
        "acceptance": {},
    }
    _atomic_write(args.output, artifact)
    _progress("baseline", steps=args.steps, nested_step=args.nested_step)
    baseline = _run_trajectory(
        model,
        handle,
        token_ids,
        start_position=args.prefix_tokens,
        steps=args.steps,
        nested_step=args.nested_step,
        nested_capture=capture_s1,
    )
    artifact["baseline"] = baseline
    artifact["snapshots_after_nested_capture"] = store.accounting()
    artifact["nested_capture"] = {
        "s0_generation": s0.snapshot_generation,
        "s1_generation": s1_holder["snapshot"].snapshot_generation,
        "s0_immutable_during_s1_capture": s1_holder[
            "s0_immutable_during_s1_capture"
        ],
        "s0_s1_storage_alias_count": s1_holder["s0_s1_storage_alias_count"],
    }
    artifact["last_completed_phase"] = "baseline"
    _atomic_write(args.output, artifact)

    replay_active = []
    snapshot_resident_reference = store.accounting()["snapshot_owned_bytes"]
    for replay_index in range(1, args.replays + 1):
        _progress("replay", replay=replay_index, total=args.replays)
        restore = _restore_and_check_stale(store, "S0", handle, identity_s0)
        candidate = _run_trajectory(
            model,
            handle,
            token_ids,
            start_position=args.prefix_tokens,
            steps=args.steps,
        )
        comparison = _compare_trajectory(baseline, candidate)
        current_snapshot_resident = store.accounting()["snapshot_owned_bytes"]
        row = {
            "replay": replay_index,
            "restore": restore,
            "comparison": comparison,
            "trajectory": {
                key: value
                for key, value in candidate.items()
                if key not in {"logits_hashes", "checkpoints"}
            },
            "logits_sequence_sha256": candidate["logits_sequence_sha256"],
            "checkpoint_count": len(candidate["checkpoints"]),
            "snapshot_resident_bytes": current_snapshot_resident,
            "snapshot_resident_constant": (
                current_snapshot_resident == snapshot_resident_reference
            ),
            "handle_accounting": handle.accounting(),
        }
        artifact["replays"].append(row)
        replay_active.append(candidate["end_memory"]["active_bytes"])
        artifact["last_completed_phase"] = f"replay-{replay_index}"
        _atomic_write(args.output, artifact)

    s1 = s1_holder["snapshot"]
    identity_s1 = s1_holder["identity"]
    _progress("nested-replay", start=args.nested_step, steps=args.steps - args.nested_step)
    nested_restore = _restore_and_check_stale(store, "S1", handle, identity_s1)
    nested = _run_trajectory(
        model,
        handle,
        token_ids,
        start_position=args.prefix_tokens + args.nested_step,
        steps=args.steps - args.nested_step,
    )
    nested_comparison = _compare_trajectory(
        baseline,
        nested,
        reference_logits_offset=args.nested_step,
    )
    artifact["nested_replay"] = {
        "restore": nested_restore,
        "comparison": nested_comparison,
        "trajectory": {
            key: value
            for key, value in nested.items()
            if key not in {"logits_hashes", "checkpoints"}
        },
        "logits_sequence_sha256": nested["logits_sequence_sha256"],
        "checkpoint_count": len(nested["checkpoints"]),
    }
    artifact["last_completed_phase"] = "nested-replay"
    _atomic_write(args.output, artifact)

    artifact["failure_injections"] = _failure_injections(
        store, s0, handle, identity_s0
    )
    s0_after_all_replays = s0.state_sha256
    s1_after_all_replays = s1.state_sha256
    store.delete("S1")
    after_s1_delete = store.accounting()
    s0_still_valid = s0.state_sha256 == s0_initial_digest and not s0.released
    final_s0_restore = _restore_and_check_stale(store, "S0", handle, identity_s0)
    final_s0_restore_exact = semantic_cache_digest(handle.cache) == s0.state_sha256
    store.delete("S0")
    final_snapshot_accounting = store.accounting()
    artifact["deletion"] = {
        "after_s1_delete": after_s1_delete,
        "s0_still_valid": s0_still_valid,
        "final_s0_restore": final_s0_restore,
        "final_s0_restore_exact": final_s0_restore_exact,
        "s0_released_after_delete": s0.released,
        "s1_released_after_delete": s1.released,
        "final_snapshot_accounting": final_snapshot_accounting,
    }
    artifact["restore_accounting"] = handle.accounting()
    artifact["resource"] = {
        "replay_endpoint_active_bytes": replay_active,
        "replay_endpoint_active_drift_bytes": (
            max(replay_active) - min(replay_active) if replay_active else 0
        ),
        "max_active_drift_bytes": MAX_ACTIVE_DRIFT,
        "snapshot_resident_reference_bytes": snapshot_resident_reference,
        "memory": _memory(),
    }
    artifact["official_oracle"] = _official_oracle(model, processor, report)

    replay_exact = all(
        all(
            row["comparison"][key]
            for key in (
                "full_vocab_logits_exact",
                "all_checkpoint_state_exact",
                "final_state_exact",
                "kda_state_exact",
                "dsa_kv_exact",
                "indexpool_exact",
                "slot_index_metadata_exact",
            )
        )
        and row["comparison"]["nan_count"] == 0
        for row in artifact["replays"]
    )
    nested_exact = all(
        nested_comparison[key]
        for key in (
            "full_vocab_logits_exact",
            "all_checkpoint_state_exact",
            "final_state_exact",
            "kda_state_exact",
            "dsa_kv_exact",
            "indexpool_exact",
            "slot_index_metadata_exact",
        )
    ) and nested_comparison["nan_count"] == 0
    replay_checkpoint_counts_exact = (
        len(baseline["checkpoints"])
        == args.steps // CHECKPOINT_INTERVAL
        and all(
            row["checkpoint_count"] == args.steps // CHECKPOINT_INTERVAL
            for row in artifact["replays"]
        )
        and artifact["nested_replay"]["checkpoint_count"]
        == (args.steps - args.nested_step) // CHECKPOINT_INTERVAL
    )
    successful_restores = [
        *(row["restore"] for row in artifact["replays"]),
        artifact["nested_replay"]["restore"],
        final_s0_restore,
    ]
    failures_atomic = all(
        row["rejected"]
        and row["live_reference_unchanged"]
        and row["live_generation_unchanged"]
        and row["live_state_unchanged"]
        and row["snapshot_unchanged"]
        for row in artifact["failure_injections"].values()
    )
    oracle = artifact["official_oracle"]
    acceptance = {
        "baseline_4096_tokens_complete": baseline["steps"] == args.steps,
        "requested_replay_count_complete": len(artifact["replays"])
        == args.replays,
        "all_full_vocab_logits_byte_exact": replay_exact and nested_exact,
        "all_256_token_checkpoint_states_exact": (
            replay_exact and nested_exact and replay_checkpoint_counts_exact
        ),
        "all_kda_dsa_indexpool_metadata_exact": replay_exact and nested_exact,
        "s0_and_s1_immutable": (
            s0_after_all_replays == s0_initial_digest
            and s1_after_all_replays == s1_holder["s1_initial_digest"]
        ),
        "s0_s1_storage_alias_zero": s1_holder["s0_s1_storage_alias_count"] == 0,
        "snapshot_resident_constant_across_restore": all(
            row["snapshot_resident_constant"] for row in artifact["replays"]
        ),
        "all_successful_restores_replace_one_cache_reference": all(
            row["cache_reference_replaced"]
            and row["generation_after"] == row["generation_before"] + 1
            for row in successful_restores
        ),
        "stale_cache_entry_references_zero": all(
            row["stale_entry_reference_count"] == 0
            for row in successful_restores
        ),
        "invalid_restores_fail_before_swap": failures_atomic,
        "delete_s1_preserves_s0": s0_still_valid and final_s0_restore_exact,
        "all_snapshots_deleted_to_zero_resident": (
            final_snapshot_accounting["resident_bytes"] == 0
            and final_snapshot_accounting["snapshot_count"] == 0
        ),
        "snapshot_allocation_release_accounting_exact": (
            final_snapshot_accounting["cumulative_allocated_bytes"]
            == final_snapshot_accounting["cumulative_released_bytes"]
            and final_snapshot_accounting["capture_count"]
            == final_snapshot_accounting["delete_count"]
            == 2
        ),
        "replacement_allocation_accounting_monotonic": (
            handle.accounting()["cumulative_replacement_allocated_bytes"]
            == handle.accounting()["cumulative_replaced_live_bytes"]
            > 0
        ),
        "replay_endpoint_active_memory_bounded": artifact["resource"][
            "replay_endpoint_active_drift_bytes"
        ]
        <= MAX_ACTIVE_DRIFT,
        "anonymous_snapshot_allocation_zero": final_snapshot_accounting[
            "anonymous_allocation_count"
        ]
        == 0,
        "nan_invalid_access_metal_error_zero": replay_exact and nested_exact,
        "official_16_token_oracle_exact": oracle["first_16_match"],
        "official_128_token_oracle_exact": oracle["full_128_match"],
        "runtime_server_disk_abi_admission_unchanged": True,
    }
    artifact["acceptance"] = acceptance
    artifact["complete"] = all(acceptance.values())
    artifact["last_completed_phase"] = "complete"
    artifact["decision"] = (
        "hybrid_semantic_prefix_snapshot_replay_qualified"
        if artifact["complete"]
        else "hybrid_semantic_prefix_snapshot_replay_not_qualified"
    )
    artifact["runtime_changes"] = {
        "server_api": False,
        "disk_serialization": False,
        "snapshot_lru": False,
        "automatic_snapshot_compression": False,
        "speculative_branch_selection": False,
        "cache_abi": False,
        "backend": False,
        "admission": False,
    }
    _atomic_write(args.output, artifact)
    _progress("complete", accepted=artifact["complete"], output=str(args.output))
    return 0 if artifact["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
