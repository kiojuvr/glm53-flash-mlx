#!/usr/bin/env python3
"""Measure the full-model headroom behind the compact IndexPool update boundary.

The accepted Tier-1 native score/selection island still receives projected
query and mixture weights from MLX.  Before implementing another stateful
native plan, this probe removes only the preceding compact IndexPool update
from the timed path: an untimed oracle trajectory materializes the exact
post-update pool/raw state for every DSA layer and decode step, then the
counterfactual arm installs that state by reference and runs the unchanged
Tier-1 selection island.

This is a strict upper bound, not a backend.  Snapshot construction, copies,
and ownership transitions remain outside timing, and no replay state is ever
published to the runtime.  A wall gain below the fixed gate stops this native
boundary before production code is written.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.metadata
import json
import statistics
import sys
import tempfile
import time
import traceback
from datetime import date
from pathlib import Path

import mlx.core as mx

from glm53_flash_mlx.abi import MLX_VLM_REVISION, NOPE_DSA_CACHE_ABI_COMPACT
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import EXPECTED_DSA, inspect_checkpoint
from glm53_flash_mlx.nope_cache import CompactIndexPoolCache


ROOT = Path(__file__).resolve().parents[1]
NATIVE_PACKAGE = ROOT / "native_execution"
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-indexpool-update-boundary-headroom-20260907.json"
)
CONTEXTS = (2_048, 262_144)
MIN_256K_HEADROOM_MS = 0.75
MAX_2K_REGRESSION = 0.01
MAX_PROCESS_PEAK_BYTES = 340_000_000_000
POOL_ARRAY_FIELDS = (
    "pool_keys",
    "pool_indices",
    "pool_valid",
    "raw_keys",
    "raw_gates",
    "raw_valid",
    "raw_positions",
)


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


def _atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _load_helpers():
    for path in (str(ROOT / "scripts"), str(NATIVE_PACKAGE)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import probe_exact_sigmoid_gate_metal_barrier as oracle_probe
    import probe_long_context_first_decode_boundary as boundary
    import probe_native_dsa_score_execution_island as tier1
    import probe_native_execution_engine_feasibility as tier0

    return oracle_probe, boundary, tier1, tier0


def _eval(value) -> None:
    arrays = []

    def visit(item):
        if isinstance(item, mx.array):
            arrays.append(item)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    if arrays:
        mx.eval(*arrays)
    mx.synchronize()


def _snapshot_pool(pool: CompactIndexPoolCache) -> dict:
    arrays = {
        name: None
        if getattr(pool, name) is None
        else mx.array(getattr(pool, name))
        for name in POOL_ARRAY_FIELDS
    }
    _eval(arrays)
    return {
        "arrays": arrays,
        "total_tokens": pool.total_tokens,
        "logical_pool_count": pool.logical_pool_count,
        "pool_capacity": pool.pool_capacity,
        "capacity_tokens": pool.capacity_tokens,
    }


def _snapshot_bytes(snapshot: dict) -> int:
    return sum(
        int(value.nbytes)
        for value in snapshot["arrays"].values()
        if value is not None
    )


def _install_snapshot(pool: CompactIndexPoolCache, snapshot: dict) -> None:
    for name, value in snapshot["arrays"].items():
        setattr(pool, name, value)
    pool.total_tokens = snapshot["total_tokens"]
    pool.logical_pool_count = snapshot["logical_pool_count"]
    pool.pool_capacity = snapshot["pool_capacity"]
    pool.capacity_tokens = snapshot["capacity_tokens"]


class _UpdateRecorder:
    def __init__(self, original):
        self.original = original
        self.current: list[dict] | None = None

    def update(self, pool, indexer, x, qr, mask=None):
        result = self.original(pool, indexer, x, qr, mask=mask)
        if self.current is None:
            raise RuntimeError("IndexPool recorder has no active decode step")
        self.current.append(_snapshot_pool(pool))
        return result


@contextlib.contextmanager
def _record_updates(recorder: _UpdateRecorder, target: list[dict]):
    previous = CompactIndexPoolCache.update
    recorder.current = target

    def wrapped(pool, indexer, x, qr, mask=None):
        return recorder.update(pool, indexer, x, qr, mask=mask)

    CompactIndexPoolCache.update = wrapped
    try:
        yield
    finally:
        CompactIndexPoolCache.update = previous
        recorder.current = None


class _UpdateReplay:
    def __init__(self, snapshots: list[dict]):
        self.snapshots = snapshots
        self.cursor = 0

    def update(self, pool, indexer, x, qr, mask=None):
        if self.cursor >= len(self.snapshots):
            raise RuntimeError("precomputed IndexPool update trajectory exhausted")
        previous = pool.total_tokens
        short_bypass = pool.validate_update(
            indexer, batch=int(x.shape[0]), length=int(x.shape[1])
        )
        if short_bypass:
            raise AssertionError("headroom probe unexpectedly entered dense bypass")
        snapshot = self.snapshots[self.cursor]
        self.cursor += 1
        if snapshot["total_tokens"] != previous + int(x.shape[1]):
            raise RuntimeError("precomputed IndexPool state has a stale boundary")
        _install_snapshot(pool, snapshot)
        if mask is not None and mask.dtype == mx.bool_ and mask.shape == (1, 1):
            valid = mask
        else:
            valid = mx.ones((1, 1), dtype=mx.bool_)
        return pool._decode_selection(indexer, x, qr, valid)


@contextlib.contextmanager
def _replay_updates(snapshots: list[dict]):
    replay = _UpdateReplay(snapshots)
    previous = CompactIndexPoolCache.update

    def wrapped(pool, indexer, x, qr, mask=None):
        return replay.update(pool, indexer, x, qr, mask=mask)

    CompactIndexPoolCache.update = wrapped
    try:
        yield replay
    finally:
        CompactIndexPoolCache.update = previous
        if replay.cursor != len(snapshots):
            raise RuntimeError(
                f"precomputed update consumption {replay.cursor} != {len(snapshots)}"
            )


def _record_trajectory(model, boundary, tier1, source, steps: int):
    cache = boundary._clone_cache(source, source[EXPECTED_DSA[0]][1].total_tokens + steps + 16)
    registry = tier1._Registry(tier1._load_helpers()[0])
    original = CompactIndexPoolCache.update
    recorder = _UpdateRecorder(original)
    token = mx.array([[3000]], dtype=mx.uint32)
    trajectory = []
    logits_hashes = []
    for step in range(steps):
        _progress("record_update_state", step=step + 1, total=steps)
        snapshots = []
        with tier1._native_arm(registry, True), _record_updates(recorder, snapshots):
            output = model(token, cache=cache)
        _eval(output.logits)
        if len(snapshots) != len(EXPECTED_DSA):
            raise RuntimeError("did not record every DSA IndexPool update")
        trajectory.append(snapshots)
        logits_hashes.append(tier1._load_helpers()[3]._hash(output.logits[0, -1]))
    return trajectory, logits_hashes, cache


def _median(rows: list[dict]) -> dict:
    return {
        "median_wall_ms": statistics.median(row["wall_ms"] for row in rows),
        "median_host_submit_ms": statistics.median(
            row["host_submit_ms"] for row in rows
        ),
        "samples": rows,
    }


def _context_case(
    model, boundary, tier1, tier0, source, context: int, warmups: int, samples: int
):
    steps = warmups + samples
    trajectory, oracle_hashes, oracle_cache = _record_trajectory(
        model, boundary, tier1, source, steps
    )
    replay_bytes = sum(_snapshot_bytes(row) for step in trajectory for row in step)
    arms = ("A_tier1_normal_update", "B_precomputed_update_tier1_selection")
    caches = {
        arm: boundary._clone_cache(source, context + steps + 16) for arm in arms
    }
    registries = {
        arm: tier1._Registry(tier1._load_helpers()[0]) for arm in arms
    }
    measured = {arm: [] for arm in arms}
    hashes = {arm: [] for arm in arms}
    token = mx.array([[3000]], dtype=mx.uint32)
    for step in range(steps):
        order = arms if step % 2 == 0 else tuple(reversed(arms))
        for arm in order:
            started = time.perf_counter_ns()
            if arm == arms[0]:
                with tier1._native_arm(registries[arm], True):
                    output = model(token, cache=caches[arm])
            else:
                with tier1._native_arm(registries[arm], True), _replay_updates(
                    trajectory[step]
                ):
                    output = model(token, cache=caches[arm])
            submitted = time.perf_counter_ns()
            _eval(output.logits)
            finished = time.perf_counter_ns()
            digest = tier0._hash(output.logits[0, -1])
            hashes[arm].append(digest)
            if step >= warmups:
                measured[arm].append(
                    {
                        "host_submit_ms": (submitted - started) / 1e6,
                        "wall_ms": (finished - started) / 1e6,
                    }
                )
    timing = {arm: _median(rows) for arm, rows in measured.items()}
    saving = timing[arms[0]]["median_wall_ms"] - timing[arms[1]]["median_wall_ms"]
    exact = (
        hashes[arms[0]] == hashes[arms[1]] == oracle_hashes
        and boundary._cache_exact(caches[arms[0]], caches[arms[1]])
        and boundary._cache_exact(caches[arms[0]], oracle_cache)
    )
    result = {
        "context_tokens": context,
        "recorded_steps": steps,
        "recorded_dsa_updates": steps * len(EXPECTED_DSA),
        "precomputed_snapshot_bytes": replay_bytes,
        "timing": timing,
        "update_boundary_headroom_ms": saving,
        "all_logits_and_post_state_byte_exact": exact,
    }
    trajectory.clear()
    caches.clear()
    registries.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    return result


def _acceptance(artifact) -> dict[str, bool]:
    short = artifact.get("contexts", {}).get("2048", {})
    long = artifact.get("contexts", {}).get("262144", {})
    return {
        "all_counterfactual_logits_and_state_byte_exact": bool(short)
        and bool(long)
        and short.get("all_logits_and_post_state_byte_exact")
        and long.get("all_logits_and_post_state_byte_exact"),
        "256k_update_boundary_headroom_at_least_0_75ms": long.get(
            "update_boundary_headroom_ms", -1e9
        )
        >= MIN_256K_HEADROOM_MS,
        "2k_counterfactual_regression_at_most_1_percent": short.get(
            "timing", {}
        ).get("B_precomputed_update_tier1_selection", {}).get(
            "median_wall_ms", 1e9
        )
        <= short.get("timing", {}).get("A_tier1_normal_update", {}).get(
            "median_wall_ms", 0.0
        )
        * (1.0 + MAX_2K_REGRESSION),
        "process_peak_at_most_340GB": artifact.get(
            "process_peak_memory_bytes", 1 << 60
        )
        <= MAX_PROCESS_PEAK_BYTES,
        "official_oracle_exact": bool(
            artifact.get("official_oracle", {}).get("first_16_match")
            and artifact.get("official_oracle", {}).get("full_128_match")
            and artifact.get("official_oracle", {}).get(
                "all_full_vocab_logits_hashes_match"
            )
        ),
        "production_abi_unchanged": artifact.get("runtime_changes")
        == {
            "runtime": False,
            "server": False,
            "apc": False,
            "cache_abi": False,
            "kernel_abi": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    args = parser.parse_args()
    artifact = {
        "schema": "glm53-native-indexpool-update-boundary-headroom-v1",
        "date": date.today().isoformat(),
        "complete": False,
        "accepted": False,
        "profiling_only": True,
        "counterfactual": (
            "exact post-update state replay; capture/copy/ownership excluded "
            "from timing; unchanged Tier-1 selection retained"
        ),
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "compact_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
        "runtime_changes": {
            "runtime": False,
            "server": False,
            "apc": False,
            "cache_abi": False,
            "kernel_abi": False,
        },
    }
    try:
        oracle_probe, boundary, tier1, tier0 = _load_helpers()
        report = inspect_checkpoint(args.model, require_server_ready=True)
        artifact["checkpoint_fingerprint"] = report.fingerprint
        artifact["official_hf_revision"] = report.official_revision
        mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
        mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
        _progress("load_model")
        model, processor = load(
            args.model,
            experimental_packed_decode_moe=True,
            experimental_compact_nope_dsa_cache=True,
            compact_cache_capacity_tokens=max(CONTEXTS) + 64,
        )
        warm_residency(model)
        artifact["official_oracle"] = oracle_probe._official_oracle(
            model, processor, report
        )
        artifact["contexts"] = {}
        for context in CONTEXTS:
            _progress("context", context=context)
            source = boundary._synthetic_cache(model, context, "compact-nope-dsa")
            artifact["contexts"][str(context)] = _context_case(
                model,
                boundary,
                tier1,
                tier0,
                source,
                context,
                args.warmups,
                args.samples,
            )
            _atomic_write(args.output, artifact)
            source.clear()
            gc.collect()
            mx.clear_cache()
            mx.synchronize()
        artifact["process_peak_memory_bytes"] = int(mx.get_peak_memory())
        artifact["acceptance"] = _acceptance(artifact)
        artifact["complete"] = True
        artifact["accepted"] = all(artifact["acceptance"].values())
        artifact["decision"] = (
            "implement_native_indexpool_update_submission_island"
            if artifact["accepted"]
            else "stop_native_indexpool_update_boundary"
        )
    except Exception as error:
        artifact.update(
            complete=True,
            accepted=False,
            decision="native_indexpool_update_headroom_probe_failed",
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
    _atomic_write(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": artifact["complete"],
                "accepted": artifact["accepted"],
                "decision": artifact["decision"],
                "failed_gates": [
                    key
                    for key, value in artifact.get("acceptance", {}).items()
                    if not value
                ],
            },
            indent=2,
        )
    )
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
