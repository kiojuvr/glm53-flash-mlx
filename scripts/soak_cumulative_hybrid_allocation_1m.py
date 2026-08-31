#!/usr/bin/env python3
"""Soak fresh hybrid-cache allocation through one million DSA token slots.

Each MoE/cache arm runs in a child process so Metal allocator state is never
shared across comparisons.  This is a probe only: it does not alter runtime
backend selection, cache ABI, APC identity, or server admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path


ARMS = (
    "direct-moe_direct-cache",
    "packed-moe_direct-cache",
    "direct-moe_compact-cache",
    "packed-moe_compact-cache",
)
MILESTONES = (0, 100_000, 500_000, 1_000_000)
TARGET_CUMULATIVE_CAPACITY = MILESTONES[-1]
COMPACT_CAPACITY_TOKENS = 4_352
MATERIALIZATION_INTERVAL = 256
EXPECTED_DSA_LAYERS = 11
EXPECTED_LEAVES = {"direct": 112, "compact-nope-dsa": 167}
MAX_MEMORY_DRIFT = 64 * 1024 * 1024
MAX_PEAK_BYTES = 340_000_000_000
MIN_LATENCY_RETENTION = 0.95
MILESTONE_SAMPLES = 5
FIXED_PROMPT_TOKENS = 16


def _write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def arm_options(arm: str) -> tuple[bool, str]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    packed = arm.startswith("packed-moe_")
    cache_backend = "compact-nope-dsa" if arm.endswith("compact-cache") else "direct"
    return packed, cache_backend


def expected_cycles(capacity_per_cycle: int) -> int:
    if capacity_per_cycle <= 0:
        raise ValueError("capacity_per_cycle must be positive")
    return (TARGET_CUMULATIVE_CAPACITY + capacity_per_cycle - 1) // capacity_per_cycle


def expected_materializations(forward_count: int) -> int:
    return int(forward_count) // MATERIALIZATION_INTERVAL


def _json_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


def _worker(args) -> int:
    # MLX is imported only in arm workers.  The parent process never creates a
    # Metal allocator, and every arm therefore begins with a fresh allocator.
    import statistics
    import time
    import weakref

    import mlx.core as mx
    import numpy as np

    sys.path.insert(0, str(Path(__file__).parent))
    import probe_long_context_first_decode_boundary as boundary

    from glm53_flash_mlx.abi import MLX_VLM_REVISION
    from glm53_flash_mlx.loader import load, warm_residency
    from glm53_flash_mlx.manifest import EXPECTED_DSA, inspect_checkpoint
    from glm53_flash_mlx.packed import PackedFP8MoE

    packed, cache_backend = arm_options(args.worker_arm)
    compact = cache_backend == "compact-nope-dsa"
    if len(EXPECTED_DSA) != EXPECTED_DSA_LAYERS:
        raise RuntimeError(f"expected {EXPECTED_DSA_LAYERS} DSA layers")
    report = inspect_checkpoint(args.model, require_server_ready=True)
    artifact = {
        "schema": "glm53-cumulative-hybrid-allocation-arm-v1",
        "date": date.today().isoformat(),
        "machine": platform.machine(),
        "arm": args.worker_arm,
        "moe_backend": "packed-decode" if packed else "direct",
        "cache_backend": cache_backend,
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "probe_only": True,
        "complete": False,
        "last_completed_cumulative_capacity": 0,
        "failure": None,
        "milestones": {},
    }

    def persist() -> None:
        _write_atomic(args.worker_output, artifact)
        if args.parent_output is not None and args.parent_output.exists():
            parent = json.loads(args.parent_output.read_text())
            parent.setdefault("arms", {})[args.worker_arm] = artifact
            _write_atomic(args.parent_output, parent)

    persist()

    def memory() -> dict[str, int]:
        mx.synchronize()
        return {
            "active_bytes": int(mx.get_active_memory()),
            "cache_bytes": int(mx.get_cache_memory()),
            "peak_bytes": int(mx.get_peak_memory()),
        }

    def arrays(value):
        if isinstance(value, mx.array):
            yield value
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from arrays(item)
        elif isinstance(value, dict):
            for key in sorted(value):
                yield from arrays(value[key])

    def entry_arrays(entry):
        yield from arrays(entry.state)

    def np_storage(value) -> np.ndarray:
        mx.eval(value)
        if value.dtype == mx.bfloat16:
            return np.ascontiguousarray(np.asarray(value.astype(mx.float32)))
        return np.ascontiguousarray(np.asarray(value))

    def hash_arrays(values) -> str:
        digest = hashlib.sha256()
        for index, value in enumerate(values):
            digest.update(str(index).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(str(value.dtype).encode())
            digest.update(np_storage(value).tobytes())
        return digest.hexdigest()

    def schema(cache) -> list[dict]:
        rows = []
        for layer, entry in enumerate(cache):
            for leaf, value in enumerate(entry_arrays(entry)):
                rows.append(
                    {
                        "layer": layer,
                        "leaf": leaf,
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "nbytes": int(value.nbytes),
                    }
                )
        return rows

    def kda_arrays(cache):
        for layer in boundary.KDA_LAYERS:
            yield from entry_arrays(cache[layer])

    def dsa_arrays(cache):
        for layer in EXPECTED_DSA:
            yield from entry_arrays(cache[layer])

    def cache_digest(cache) -> str:
        digest = hashlib.sha256()
        digest.update(
            hash_arrays(value for entry in cache for value in entry_arrays(entry)).encode()
        )
        for entry in cache:
            digest.update(repr(getattr(entry, "meta_state", None)).encode())
        return digest.hexdigest()

    def capacity(cache) -> dict:
        rows = []
        for layer in EXPECTED_DSA:
            if compact:
                latent, pool = cache[layer]
                latent_tokens = int(latent.physical_capacity_tokens)
                pool_rows = int(pool.physical_capacity_rows)
                rows.append(
                    {
                        "layer": layer,
                        "physical_capacity_tokens": latent_tokens,
                        "latent_tokens": latent_tokens,
                        "indexpool_rows": pool_rows,
                    }
                )
            else:
                latent, indexer = cache[layer]
                latent_tokens = int(latent.keys.shape[2])
                indexer_tokens = int(indexer.keys.shape[2])
                rows.append(
                    {
                        "layer": layer,
                        "physical_capacity_tokens": max(latent_tokens, indexer_tokens),
                        "latent_tokens": latent_tokens,
                        "indexer_tokens": indexer_tokens,
                    }
                )
        values = [row["physical_capacity_tokens"] for row in rows]
        return {
            "layers": rows,
            "minimum_tokens": min(values),
            "maximum_tokens": max(values),
            "all_layers_uniform": len(set(values)) == 1,
        }

    def kda_layout(cache) -> dict:
        rows = []
        for layer in boundary.KDA_LAYERS:
            leaves = list(entry_arrays(cache[layer]))
            rows.append(
                {
                    "layer": layer,
                    "leaves": [
                        {
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                            "nbytes": int(value.nbytes),
                        }
                        for value in leaves
                    ],
                    "nbytes": sum(int(value.nbytes) for value in leaves),
                }
            )
        return {
            "layers": rows,
            "total_bytes": sum(row["nbytes"] for row in rows),
            "layout_hash": _json_hash(rows),
        }

    def index_safety(cache) -> dict:
        sentinel = 0
        positive_oob = 0
        nan = 0
        for layer in EXPECTED_DSA:
            if compact:
                latent, pool = cache[layer]
                indices = pool.pool_indices[:, : pool.logical_pool_count]
                kv_len = int(latent.offset)
            else:
                latent, indexer = cache[layer]
                if getattr(indexer, "_pool", None) is None:
                    continue
                indices = indexer._pool[1]
                kv_len = int(latent.offset)
            values = np_storage(indices)
            sentinel += int((values == -1).sum())
            positive_oob += int(((values >= kv_len) & (values != -1)).sum())
            nan += int(np.isnan(values.astype(np.float64)).sum())
        return {
            "sentinel_count": sentinel,
            "positive_out_of_bounds_count": positive_oob,
            "nan_count": nan,
        }

    materialization_count = 0
    forward_count = 0
    materialization_ms = []

    def after_forward(cache) -> None:
        nonlocal materialization_count, forward_count
        forward_count += 1
        if forward_count % MATERIALIZATION_INTERVAL == 0:
            started = time.perf_counter()
            mx.eval([entry.state for entry in cache])
            mx.clear_cache()
            mx.synchronize()
            materialization_ms.append((time.perf_counter() - started) * 1000.0)
            materialization_count += 1

    def release(cache, output=None) -> int:
        refs = []
        for entry in cache:
            try:
                refs.append(weakref.ref(entry))
            except TypeError:
                pass
        if cache:
            del entry
        cache.clear()
        del cache, output
        return sum(ref() is not None for ref in refs)

    def deterministic_tokens(length: int, vocab: int) -> mx.array:
        values = ((np.arange(length, dtype=np.uint64) * 7919) % (vocab - 1024) + 100)
        return mx.array(values.astype(np.uint32)[None])

    def top_token(logits) -> int:
        token = mx.argmax(logits)
        mx.eval(token)
        return int(token.item())

    def milestone(model, vocab: int, target: int) -> dict:
        samples = []
        canonical = None
        for sample in range(MILESTONE_SAMPLES):
            cache = model.make_cache()
            prompt = deterministic_tokens(FIXED_PROMPT_TOKENS, vocab)
            prompt_output = model(prompt, cache=cache)
            prompt_logits = prompt_output.logits[0, -1]
            mx.eval(prompt_logits, [entry.state for entry in cache])
            after_forward(cache)
            prompt_hash = hashlib.sha256(np_storage(prompt_logits).tobytes()).hexdigest()
            prompt_token = top_token(prompt_logits)
            started = time.perf_counter()
            decode_output = model(mx.array([[prompt_token]], dtype=mx.uint32), cache=cache)
            decode_logits = decode_output.logits[0, -1]
            nan_value = mx.sum(mx.isnan(decode_logits))
            mx.eval(decode_logits, nan_value, [entry.state for entry in cache])
            mx.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            after_forward(cache)
            row_schema = schema(cache)
            row = {
                "prompt_final_full_vocab_logits_hash": prompt_hash,
                "prompt_generated_token": prompt_token,
                "first_decode_full_vocab_logits_hash": hashlib.sha256(
                    np_storage(decode_logits).tobytes()
                ).hexdigest(),
                "generated_token": top_token(decode_logits),
                "full_logical_cache_state_digest": cache_digest(cache),
                "state_schema_hash": _json_hash(row_schema),
                "cache_leaf_count": len(row_schema),
                "kda_state_digest": hash_arrays(kda_arrays(cache)),
                "dsa_state_digest": hash_arrays(dsa_arrays(cache)),
                "kda_layout": kda_layout(cache),
                "physical_capacity": capacity(cache),
                "index_safety": index_safety(cache),
                "first_decode_latency_ms": elapsed_ms,
                "nan_count": int(nan_value.item()),
            }
            comparable = {key: value for key, value in row.items() if key != "first_decode_latency_ms"}
            if canonical is None:
                canonical = comparable
            elif comparable != canonical:
                raise RuntimeError(f"milestone {target} repeated sample is not exact")
            samples.append(elapsed_ms)
            release(cache, prompt_output)
            del decode_output
        assert canonical is not None
        return {
            **canonical,
            "first_decode_latency_samples_ms": samples,
            "first_decode_latency_median_ms": statistics.median(samples),
        }

    try:
        _progress("load", arm=args.worker_arm)
        started = time.perf_counter()
        model, _ = load(
            args.model,
            experimental_packed_decode_moe=packed,
            experimental_compact_nope_dsa_cache=compact,
            compact_cache_capacity_tokens=COMPACT_CAPACITY_TOKENS,
        )
        warm_residency(model)
        mx.clear_cache()
        mx.synchronize()
        artifact["startup_seconds"] = time.perf_counter() - started
        artifact["packed_layer_count"] = sum(
            isinstance(layer.mlp, PackedFP8MoE)
            for layer in model.language_model.model.layers
        )
        if artifact["packed_layer_count"] != (42 if packed else 0):
            raise RuntimeError("unexpected packed MoE layer count")
        raw_config = json.loads((Path(args.model) / "config.json").read_text())
        vocab = int(raw_config["text_config"]["vocab_size"])

        # Compile the exact one-token churn and milestone paths before taking
        # the drift baseline.  These warmups are explicitly outside the soak.
        warm_cache = model.make_cache()
        warm_output = model(mx.array([[101]], dtype=mx.uint32), cache=warm_cache)
        mx.eval(warm_output.logits, [entry.state for entry in warm_cache])
        mx.synchronize()
        release(warm_cache, warm_output)
        warm_cache = None
        warm_output = None
        mx.clear_cache()
        mx.synchronize()
        baseline_milestone = milestone(model, vocab, 0)
        artifact["milestones"]["0"] = baseline_milestone
        baseline_exact = {
            key: value
            for key, value in baseline_milestone.items()
            if key not in {"first_decode_latency_samples_ms", "first_decode_latency_median_ms"}
        }

        # The hybrid-allocation invariant is checked on a real initialized
        # cache.  Compact reservation is DSA-only and must not touch KDA.
        invariant_cache = model.make_cache()
        invariant_output = model(mx.array([[103]], dtype=mx.uint32), cache=invariant_cache)
        mx.eval(invariant_output.logits, [entry.state for entry in invariant_cache])
        after_forward(invariant_cache)
        kda_before = kda_layout(invariant_cache)
        kda_digest_before = hash_arrays(kda_arrays(invariant_cache))
        if compact:
            for layer in EXPECTED_DSA:
                latent, pool = invariant_cache[layer]
                latent.reserve_until(8192)
                pool.reserve_until(8192)
            mx.eval([entry.state for entry in invariant_cache])
            mx.synchronize()
        kda_after = kda_layout(invariant_cache)
        kda_digest_after = hash_arrays(kda_arrays(invariant_cache))
        metadata_terms = ("page", "capacity", "chunk", "block", "reserve")
        kda_metadata = sorted(
            {
                key
                for layer in boundary.KDA_LAYERS
                for key in vars(invariant_cache[layer])
                if any(term in key.lower() for term in metadata_terms)
            }
        )
        dsa_metadata = sorted(
            {
                key
                for layer in EXPECTED_DSA
                for child in invariant_cache[layer]
                for key in vars(child)
                if any(term in key.lower() for term in metadata_terms)
            }
        )
        shared_page_metadata = sorted(set(kda_metadata) & set(dsa_metadata))
        artifact["hybrid_allocation_invariant"] = {
            "kda_layout_before": kda_before,
            "kda_layout_after_dsa_reserve": kda_after,
            "kda_digest_before": kda_digest_before,
            "kda_digest_after_dsa_reserve": kda_digest_after,
            "compact_reserve_until_kda_unchanged": (
                kda_before == kda_after and kda_digest_before == kda_digest_after
            ),
            "kda_capacity_metadata": kda_metadata,
            "dsa_capacity_metadata": dsa_metadata,
            "shared_capacity_metadata": shared_page_metadata,
            "kda_dsa_shared_page_size_metadata": bool(shared_page_metadata),
            "reason": (
                "ArraysCache KDA entries expose only fixed conv/recurrent state; "
                "capacity_tokens and pool_capacity belong only to DSA children"
            ),
        }
        release(invariant_cache, invariant_output)
        invariant_cache = None
        invariant_output = None
        mx.clear_cache()
        mx.synchronize()
        artifact["baseline_memory"] = memory()

        cumulative_capacity = 0
        cumulative_dsa_layer_token_slots = 0
        cumulative_kda_state_bytes = 0
        fresh_cache_count = 0
        live_cache_failures = 0
        nan_count = 0
        oob_count = 0
        capacity_per_cycle = None
        kda_layout_reference = None
        kda_state_bytes_min = None
        kda_state_bytes_max = None
        post_release_active_samples = []
        pending_milestones = list(MILESTONES[1:])

        while cumulative_capacity < TARGET_CUMULATIVE_CAPACITY:
            cycle_started = time.perf_counter()
            cache = model.make_cache()
            output = model(mx.array([[107]], dtype=mx.uint32), cache=cache)
            logits = output.logits[0, -1]
            nan_value = mx.sum(mx.isnan(logits))
            mx.eval(logits, nan_value, [entry.state for entry in cache])
            after_forward(cache)
            nan_count += int(nan_value.item())
            cap = capacity(cache)
            safety = index_safety(cache)
            oob_count += safety["positive_out_of_bounds_count"]
            cycle_capacity = int(cap["maximum_tokens"])
            if not cap["all_layers_uniform"]:
                raise RuntimeError("DSA physical capacity differs by layer")
            if capacity_per_cycle is None:
                capacity_per_cycle = cycle_capacity
            elif cycle_capacity != capacity_per_cycle:
                raise RuntimeError("fresh-cache physical capacity changed across cycles")
            layout = kda_layout(cache)
            if kda_layout_reference is None:
                kda_layout_reference = layout
            elif layout != kda_layout_reference:
                raise RuntimeError("KDA shape/dtype/nbytes changed across cycles")
            kda_bytes = int(layout["total_bytes"])
            kda_state_bytes_min = kda_bytes if kda_state_bytes_min is None else min(kda_state_bytes_min, kda_bytes)
            kda_state_bytes_max = kda_bytes if kda_state_bytes_max is None else max(kda_state_bytes_max, kda_bytes)
            cumulative_capacity += cycle_capacity
            cumulative_dsa_layer_token_slots += cycle_capacity * len(EXPECTED_DSA)
            cumulative_kda_state_bytes += kda_bytes
            fresh_cache_count += 1
            live_cache_failures += release(cache, output)
            cache = None
            output = None

            if fresh_cache_count % 128 == 0:
                _progress(
                    "churn",
                    arm=args.worker_arm,
                    cycles=fresh_cache_count,
                    cumulative_capacity=cumulative_capacity,
                    last_cycle_ms=(time.perf_counter() - cycle_started) * 1000.0,
                )

            if forward_count % MATERIALIZATION_INTERVAL == 0:
                post_release_active_samples.append(int(mx.get_active_memory()))

            while pending_milestones and cumulative_capacity >= pending_milestones[0]:
                target = pending_milestones.pop(0)
                row = milestone(model, vocab, target)
                exact = {
                    key: value
                    for key, value in row.items()
                    if key not in {"first_decode_latency_samples_ms", "first_decode_latency_median_ms"}
                }
                row["matches_arm_baseline_exactly"] = exact == baseline_exact
                row["cumulative_capacity_at_capture"] = cumulative_capacity
                row["fresh_cache_count_at_capture"] = fresh_cache_count
                artifact["milestones"][str(target)] = row
                artifact["last_completed_cumulative_capacity"] = cumulative_capacity
                persist()
                _progress(
                    "milestone",
                    arm=args.worker_arm,
                    target=target,
                    cumulative_capacity=cumulative_capacity,
                    cycles=fresh_cache_count,
                )
                if not row["matches_arm_baseline_exactly"]:
                    raise RuntimeError(f"milestone {target} differs from arm baseline")

        mx.clear_cache()
        mx.synchronize()
        final_memory = memory()
        baseline_memory = artifact["baseline_memory"]
        final_active_drift = final_memory["active_bytes"] - baseline_memory["active_bytes"]
        final_cache_drift = final_memory["cache_bytes"] - baseline_memory["cache_bytes"]
        final_latency = artifact["milestones"][str(TARGET_CUMULATIVE_CAPACITY)][
            "first_decode_latency_median_ms"
        ]
        baseline_latency = baseline_milestone["first_decode_latency_median_ms"]
        latency_retention = baseline_latency / final_latency
        artifact.update(
            {
                "complete": True,
                "cumulative_physical_sequence_capacity": cumulative_capacity,
                "cumulative_dsa_layer_token_slots": cumulative_dsa_layer_token_slots,
                "cumulative_kda_state_bytes": cumulative_kda_state_bytes,
                "fresh_cache_count": fresh_cache_count,
                "actual_model_forward_count": forward_count,
                "capacity_per_fresh_cache": capacity_per_cycle,
                "expected_fresh_cache_count": expected_cycles(capacity_per_cycle),
                "scheduled_materialization_count": materialization_count,
                "expected_scheduled_materialization_count": expected_materializations(forward_count),
                "materialization_ms": materialization_ms,
                "kda_state_bytes_min": kda_state_bytes_min,
                "kda_state_bytes_max": kda_state_bytes_max,
                "kda_layout": kda_layout_reference,
                "state_leaf_count": baseline_milestone["cache_leaf_count"],
                "live_cache_reference_failures": live_cache_failures,
                "nan_count": nan_count,
                "positive_out_of_bounds_count": oob_count,
                "metal_error": None,
                "post_release_active_samples": post_release_active_samples,
                "final_memory": final_memory,
                "final_active_memory_drift_bytes": final_active_drift,
                "final_cache_memory_drift_bytes": final_cache_drift,
                "fixed_prompt_first_decode_latency_retention": latency_retention,
            }
        )
        acceptance = {
            "cumulative_capacity_at_least_1m": cumulative_capacity >= TARGET_CUMULATIVE_CAPACITY,
            "all_milestone_logits_and_state_exact": all(
                row.get("matches_arm_baseline_exactly", True)
                for row in artifact["milestones"].values()
            ),
            "active_memory_drift_at_most_64_mib": abs(final_active_drift) <= MAX_MEMORY_DRIFT,
            "final_post_clear_drift_at_most_64_mib": abs(final_cache_drift) <= MAX_MEMORY_DRIFT,
            "peak_memory_at_most_340_gb": final_memory["peak_bytes"] <= MAX_PEAK_BYTES,
            "fixed_prompt_latency_retention_at_least_0_95": latency_retention >= MIN_LATENCY_RETENTION,
            "kda_state_size_constant_every_cycle": kda_state_bytes_min == kda_state_bytes_max,
            "state_leaf_count_expected": baseline_milestone["cache_leaf_count"] == EXPECTED_LEAVES[cache_backend],
            "no_nan_oob_or_metal_error": nan_count == 0 and oob_count == 0,
            "scheduled_materialization_count_exact": materialization_count == expected_materializations(forward_count),
            "no_live_cache_after_cycle": live_cache_failures == 0,
            "compact_reserve_does_not_change_kda": artifact["hybrid_allocation_invariant"]["compact_reserve_until_kda_unchanged"],
            "fresh_cache_count_exact": fresh_cache_count == expected_cycles(capacity_per_cycle),
        }
        acceptance["accepted"] = all(acceptance.values())
        artifact["acceptance"] = acceptance
        artifact["last_completed_cumulative_capacity"] = cumulative_capacity
        persist()
        _progress("worker_complete", arm=args.worker_arm, accepted=acceptance["accepted"])
        return 0 if acceptance["accepted"] else 1
    except Exception as error:
        artifact["failure"] = {"type": type(error).__name__, "message": str(error)}
        artifact["actual_model_forward_count"] = forward_count
        artifact["scheduled_materialization_count"] = materialization_count
        artifact["metal_error"] = str(error) if "metal" in str(error).lower() else None
        persist()
        raise


def _parent(args) -> int:
    from glm53_flash_mlx.manifest import inspect_checkpoint

    checkpoint = inspect_checkpoint(args.model, require_server_ready=True)
    existing = None
    if args.output.exists() and not args.rerun_all:
        candidate = json.loads(args.output.read_text())
        reusable_fingerprints = {
            arm.get("checkpoint_fingerprint")
            for arm in candidate.get("arms", {}).values()
            if arm.get("complete") is True
        }
        if (
            candidate.get("schema")
            == "glm53-cumulative-hybrid-allocation-1m-v1"
            and reusable_fingerprints <= {checkpoint.fingerprint}
        ):
            existing = candidate
    report = {
        "schema": "glm53-cumulative-hybrid-allocation-1m-v1",
        "date": date.today().isoformat(),
        "machine": platform.machine(),
        "checkpoint_revision": checkpoint.official_revision,
        "checkpoint_fingerprint": checkpoint.fingerprint,
        "model_path": str(Path(args.model).expanduser().resolve()),
        "probe_only": True,
        "separate_process_per_arm": True,
        "target_cumulative_physical_sequence_capacity": TARGET_CUMULATIVE_CAPACITY,
        "milestones": list(MILESTONES),
        "materialization_policy": {
            "policy": "nested-cache-eval-clear-v1",
            "interval_tokens": MATERIALIZATION_INTERVAL,
        },
        "arms": {},
        "complete": False,
        "failure": None,
        "runtime_changes": {
            "backend_default": False,
            "cache_abi": False,
            "apc_identity": False,
            "server": False,
            "admission": False,
        },
    }
    if existing is not None:
        report["arms"] = {
            arm: value
            for arm, value in existing.get("arms", {}).items()
            if arm in ARMS
            and value.get("complete") is True
            and value.get("acceptance", {}).get("accepted") is True
        }
    _write_atomic(args.output, report)
    with tempfile.TemporaryDirectory(prefix="glm53-hybrid-soak-") as directory:
        for arm in ARMS:
            if arm in report["arms"]:
                _progress("reuse_completed_worker", arm=arm)
                continue
            worker_output = Path(directory) / f"{arm}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                str(args.model),
                "--worker-arm",
                arm,
                "--worker-output",
                str(worker_output),
                "--parent-output",
                str(args.output),
            ]
            _progress("spawn_worker", arm=arm)
            completed = subprocess.run(command, check=False)
            if worker_output.exists():
                report["arms"][arm] = json.loads(worker_output.read_text())
                _write_atomic(args.output, report)
            if completed.returncode != 0:
                report["failure"] = {
                    "arm": arm,
                    "worker_returncode": completed.returncode,
                }
                _write_atomic(args.output, report)
                return completed.returncode

    arms = report["arms"]
    milestone_logits = {}
    for milestone in MILESTONES:
        milestone_logits[str(milestone)] = {
            arm: {
                "prompt": arms[arm]["milestones"][str(milestone)][
                    "prompt_final_full_vocab_logits_hash"
                ],
                "first_decode": arms[arm]["milestones"][str(milestone)][
                    "first_decode_full_vocab_logits_hash"
                ],
                "generated_token": arms[arm]["milestones"][str(milestone)][
                    "generated_token"
                ],
                "kda_state_digest": arms[arm]["milestones"][str(milestone)][
                    "kda_state_digest"
                ],
            }
            for arm in ARMS
        }
    cross_arm_logits_exact = all(
        len({row["prompt"] for row in rows.values()}) == 1
        and len({row["first_decode"] for row in rows.values()}) == 1
        and len({row["generated_token"] for row in rows.values()}) == 1
        for rows in milestone_logits.values()
    )
    cross_arm_kda_exact = all(
        len({row["kda_state_digest"] for row in rows.values()}) == 1
        for rows in milestone_logits.values()
    )
    kda_layout_hashes = {
        arm: arms[arm]["kda_layout"]["layout_hash"] for arm in ARMS
    }
    acceptance = {
        "all_arms_complete_and_accepted": all(
            arms[arm]["complete"] and arms[arm]["acceptance"]["accepted"]
            for arm in ARMS
        ),
        "all_arms_use_separate_processes": True,
        "cross_arm_logits_and_tokens_exact": cross_arm_logits_exact,
        "cross_arm_kda_state_exact": cross_arm_kda_exact,
        "direct_compact_kda_allocation_granularity_identical": len(
            set(kda_layout_hashes.values())
        )
        == 1,
        "no_shared_kda_dsa_page_size_metadata": all(
            not arms[arm]["hybrid_allocation_invariant"][
                "kda_dsa_shared_page_size_metadata"
            ]
            for arm in ARMS
        ),
        "runtime_server_apc_abi_and_admission_unchanged": True,
    }
    acceptance["accepted"] = all(acceptance.values())
    report.update(
        {
            "complete": True,
            "cross_arm_milestone_evidence": milestone_logits,
            "cross_arm_kda_layout_hashes": kda_layout_hashes,
            "acceptance": acceptance,
            "decision": {
                "packed_decode_direct_cache_default_candidate": acceptance["accepted"],
                "compact_cache_remains_single_session_opt_in": True,
            },
        }
    )
    _write_atomic(args.output, report)
    print(json.dumps({"output": str(args.output), "acceptance": acceptance}, indent=2))
    return 0 if acceptance["accepted"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "bench-results/m3ultra512-cumulative-hybrid-allocation-1m-20260831.json"
        ),
    )
    parser.add_argument("--worker-arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--parent-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--rerun-all",
        action="store_true",
        help="ignore accepted arms already present in the output artifact",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker_arm:
        if args.worker_output is None:
            raise SystemExit("--worker-output is required with --worker-arm")
        return _worker(args)
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
