#!/usr/bin/env python3
"""Measure headroom behind the four remaining Indexer input projections.

The accepted native IndexPool-update/Tier-1 island still receives projected
key, compress gate, query, and mixture weights from MLX.  This strict
counterfactual records those exact BF16 tensors on an untimed trajectory and
replays them into the unchanged native island.  Capture, copies, and storage
are excluded from timing; this is an implementation upper bound, not a
runtime backend.
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
    / "m3ultra512-native-indexer-projection-boundary-headroom-20260907.json"
)
CONTEXTS = (2_048, 262_144)
MIN_256K_HEADROOM_MS = 0.75
MAX_2K_REGRESSION = 0.01
MAX_PROCESS_PEAK_BYTES = 340_000_000_000
PROJECTION_NAMES = ("key", "gate", "query", "mixture_weights", "valid")


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
    import probe_native_execution_engine_feasibility as tier0
    import probe_native_indexpool_update_submission_island as update_island

    return oracle_probe, boundary, tier0, update_island


def _arrays(value):
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _arrays(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _arrays(child)


def _eval(value) -> None:
    arrays = list(_arrays(value))
    if arrays:
        mx.eval(*arrays)
    mx.synchronize()


def _snapshot_projection(projection: dict[str, mx.array]) -> dict[str, mx.array]:
    snapshot = {name: mx.array(projection[name]) for name in PROJECTION_NAMES}
    mx.eval(*snapshot.values())
    return snapshot


def _snapshot_bytes(snapshot: dict[str, mx.array]) -> int:
    return sum(int(value.nbytes) for value in snapshot.values())


def _project(indexer, x: mx.array, qr: mx.array) -> dict[str, mx.array]:
    return {
        "key": indexer.k_norm(indexer.wk(x)).reshape(1, 1, indexer.head_dim),
        "gate": x @ indexer.index_kpool_compress_gate.swapaxes(-1, -2),
        "query": indexer.wq_b(qr).reshape(
            1, 1, indexer.n_heads, indexer.head_dim
        ),
        "mixture_weights": indexer.weights_proj(x)
        * (indexer.n_heads**-0.5),
        "valid": mx.ones((1, 1), dtype=mx.bool_),
    }


def _execute_projection(
    update_island,
    registry,
    pool,
    indexer,
    x,
    qr,
    mask,
    projection,
):
    short_bypass = pool.validate_update(
        indexer, batch=int(x.shape[0]), length=int(x.shape[1])
    )
    if (
        short_bypass
        or int(x.shape[1]) != 1
        or pool.raw_token_count != 19
        or mask is not None
    ):
        raise RuntimeError(
            "projection headroom probe requires maskless sparse L=1 decode"
        )
    selected, _ = update_island._native_update(
        registry.get(pool, indexer),
        pool,
        projection["key"],
        projection["gate"],
        projection["valid"],
        projection["query"],
        projection["mixture_weights"],
    )
    return selected[:, None]


class _ProjectionRecorder:
    def __init__(self, update_island, registry):
        self.update_island = update_island
        self.registry = registry
        self.current: list[dict[str, mx.array]] | None = None

    def update(self, pool, indexer, x, qr, mask=None):
        if self.current is None:
            raise RuntimeError("projection recorder has no active decode step")
        projection = _project(indexer, x, qr)
        self.current.append(_snapshot_projection(projection))
        return _execute_projection(
            self.update_island,
            self.registry,
            pool,
            indexer,
            x,
            qr,
            mask,
            projection,
        )


@contextlib.contextmanager
def _record_projections(recorder, target):
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


class _ProjectionReplay:
    def __init__(self, update_island, registry, snapshots):
        self.update_island = update_island
        self.registry = registry
        self.snapshots = snapshots
        self.cursor = 0

    def update(self, pool, indexer, x, qr, mask=None):
        if self.cursor >= len(self.snapshots):
            raise RuntimeError("precomputed projection trajectory exhausted")
        projection = self.snapshots[self.cursor]
        self.cursor += 1
        expected = {
            "key": (1, 1, 128),
            "gate": (1, 1, 128),
            "query": (1, 1, 32, 128),
            "mixture_weights": (1, 1, 32),
            "valid": (1, 1),
        }
        if any(tuple(projection[name].shape) != shape for name, shape in expected.items()):
            raise RuntimeError("precomputed projection shape is stale")
        return _execute_projection(
            self.update_island,
            self.registry,
            pool,
            indexer,
            x,
            qr,
            mask,
            projection,
        )


@contextlib.contextmanager
def _replay_projections(update_island, registry, snapshots):
    replay = _ProjectionReplay(update_island, registry, snapshots)
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
                f"projection consumption {replay.cursor} != {len(snapshots)}"
            )


def _record_trajectory(model, boundary, tier0, update_island, source, steps):
    cache = boundary._clone_cache(
        source, source[EXPECTED_DSA[0]][1].total_tokens + steps + 16
    )
    registry = update_island._Registry(update_island._load_helpers()[0])
    recorder = _ProjectionRecorder(update_island, registry)
    token = mx.array([[3000]], dtype=mx.uint32)
    trajectory = []
    logits_hashes = []
    for step in range(steps):
        _progress("record_projection", step=step + 1, total=steps)
        snapshots = []
        with _record_projections(recorder, snapshots):
            output = model(token, cache=cache)
        _eval((output.logits, snapshots))
        if len(snapshots) != len(EXPECTED_DSA):
            raise RuntimeError("did not record every DSA projection boundary")
        trajectory.append(snapshots)
        logits_hashes.append(tier0._hash(output.logits[0, -1]))
    return trajectory, logits_hashes, cache


def _median(rows):
    return {
        "median_wall_ms": statistics.median(row["wall_ms"] for row in rows),
        "median_host_submit_ms": statistics.median(
            row["host_submit_ms"] for row in rows
        ),
        "samples": rows,
    }


def _context_case(
    model,
    boundary,
    tier0,
    update_island,
    source,
    context,
    warmups,
    samples,
):
    steps = warmups + samples
    trajectory, oracle_hashes, oracle_cache = _record_trajectory(
        model, boundary, tier0, update_island, source, steps
    )
    trajectory_bytes = sum(
        _snapshot_bytes(snapshot) for step in trajectory for snapshot in step
    )
    arms = (
        "A_native_update_with_mlx_projections",
        "B_precomputed_projections_native_update",
    )
    caches = {
        arm: boundary._clone_cache(source, context + steps + 16) for arm in arms
    }
    registries = {
        arm: update_island._Registry(update_island._load_helpers()[0])
        for arm in arms
    }
    measured = {arm: [] for arm in arms}
    hashes = {arm: [] for arm in arms}
    token = mx.array([[3000]], dtype=mx.uint32)
    for step in range(steps):
        order = arms if step % 2 == 0 else tuple(reversed(arms))
        for arm in order:
            started = time.perf_counter_ns()
            if arm == arms[0]:
                with update_island._native_arm(registries[arm], True):
                    output = model(token, cache=caches[arm])
            else:
                with _replay_projections(
                    update_island, registries[arm], trajectory[step]
                ):
                    output = model(token, cache=caches[arm])
            submitted = time.perf_counter_ns()
            _eval(output.logits)
            finished = time.perf_counter_ns()
            hashes[arm].append(tier0._hash(output.logits[0, -1]))
            if step >= warmups:
                measured[arm].append(
                    {
                        "host_submit_ms": (submitted - started) / 1e6,
                        "wall_ms": (finished - started) / 1e6,
                    }
                )
    timing = {arm: _median(rows) for arm, rows in measured.items()}
    result = {
        "context_tokens": context,
        "recorded_steps": steps,
        "recorded_dsa_projections": steps * len(EXPECTED_DSA),
        "precomputed_projection_bytes": trajectory_bytes,
        "timing": timing,
        "projection_boundary_headroom_ms": (
            timing[arms[0]]["median_wall_ms"]
            - timing[arms[1]]["median_wall_ms"]
        ),
        "all_logits_and_post_state_byte_exact": (
            hashes[arms[0]] == hashes[arms[1]] == oracle_hashes
            and boundary._cache_exact(caches[arms[0]], caches[arms[1]])
            and boundary._cache_exact(caches[arms[0]], oracle_cache)
        ),
        "native_plan_evidence": [
            row for registry in registries.values() for row in registry.evidence()
        ],
    }
    trajectory.clear()
    caches.clear()
    registries.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    return result


def _acceptance(artifact):
    short = artifact.get("contexts", {}).get("2048", {})
    long = artifact.get("contexts", {}).get("262144", {})
    return {
        "all_counterfactual_logits_and_state_byte_exact": bool(short)
        and bool(long)
        and short.get("all_logits_and_post_state_byte_exact")
        and long.get("all_logits_and_post_state_byte_exact"),
        "256k_projection_boundary_headroom_at_least_0_75ms": long.get(
            "projection_boundary_headroom_ms", -1e9
        )
        >= MIN_256K_HEADROOM_MS,
        "2k_counterfactual_regression_at_most_1_percent": short.get(
            "timing", {}
        ).get("B_precomputed_projections_native_update", {}).get(
            "median_wall_ms", 1e9
        )
        <= short.get("timing", {}).get(
            "A_native_update_with_mlx_projections", {}
        ).get("median_wall_ms", 0.0)
        * (1.0 + MAX_2K_REGRESSION),
        "official_oracle_exact": bool(
            artifact.get("official_oracle", {}).get(
                "all_full_vocab_logits_hashes_match"
            )
        ),
        "process_peak_at_most_340GB": artifact.get(
            "process_peak_memory_bytes", 1 << 60
        )
        <= MAX_PROCESS_PEAK_BYTES,
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
        "schema": "glm53-native-indexer-projection-boundary-headroom-v1",
        "date": date.today().isoformat(),
        "complete": False,
        "accepted": False,
        "profiling_only": True,
        "counterfactual": (
            "exact precomputed BF16 key/gate/query/mixture-weight projections; "
            "capture/copy/ownership excluded from timing; accepted native "
            "IndexPool-update/Tier-1 island unchanged"
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
        oracle_probe, boundary, tier0, update_island = _load_helpers()
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
            _progress("projection_context", context=context)
            source = boundary._synthetic_cache(model, context, "compact-nope-dsa")
            artifact["contexts"][str(context)] = _context_case(
                model,
                boundary,
                tier0,
                update_island,
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
            "implement_native_indexer_input_projections"
            if artifact["accepted"]
            else "stop_native_indexer_projection_boundary"
        )
    except Exception as error:
        artifact.update(
            complete=True,
            accepted=False,
            decision="native_indexer_projection_headroom_probe_failed",
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
