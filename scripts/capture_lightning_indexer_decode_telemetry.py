#!/usr/bin/env python3
"""Capture bounded, non-replayable Lightning Indexer System Trace telemetry.

One model process is kept resident for a context.  Each phase is prepared
outside its trace and then captured separately, so dynamic PID evidence—not
static shader labels—identifies the phase.  The traces contain timing and
allocation events, not replayable model resources.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import characterize_packed_decode_bounded_telemetry as bounded
import profile_lightning_indexer_decode_critical_path as profiler
from capture_budget import (
    atomic_write,
    path_bytes,
    remove_partial_trace,
    terminate_process_group,
)
from capture_steady_packed_decode_critical_path import _trace_identity


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
TRACE_STAGES = profiler.PHASES + ("full_model_decode",)
TRACE_TIME_LIMIT_SECONDS = 6
MAX_TRACE_BYTES = 2 << 30
MIN_FREE_BYTES = 64 << 30


def _wait_for(path: Path, timeout_s: float, process=None) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"process exited before creating {path}")
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for {path}")


def _validate_trace_dir(path: Path, context: int, completed: set[str]) -> Path:
    path = path.expanduser().resolve()
    if path.is_relative_to(REPOSITORY):
        raise ValueError("Metal System Trace directory must be outside the repository")
    path.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path).free < MIN_FREE_BYTES:
        raise RuntimeError("less than 64 GiB is free at the trace destination")
    conflicts = [path / f"indexer-{context}-{stage}.trace" for stage in TRACE_STAGES]
    existing = [
        item
        for item in conflicts
        if item.exists() and item.stem.removeprefix(f"indexer-{context}-") not in completed
    ]
    if existing:
        raise FileExistsError(f"trace path already exists: {existing[0]}")
    return path


def _clone_pool(pool):
    from glm53_flash_mlx.nope_cache import CompactIndexPoolCache

    state = tuple(mx.array(value) for value in pool.state)
    cloned = CompactIndexPoolCache.from_state(state, pool.meta_state)
    mx.eval(*cloned.state)
    mx.synchronize()
    return cloned


def _prepare_layers(model, source, boundary, operator, context: int):
    rows = []
    for layer in profiler.EXPECTED_DSA:
        attention = model.language_model.model.layers[layer].self_attn
        x = boundary._deterministic_rows(
            1,
            attention.hidden_size,
            3.75 + layer * 0.015625,
            mx.bfloat16,
        )[None]
        entry = boundary._clone_entry(source[layer], context + 16)
        latent, pool = entry
        update_pool = _clone_pool(source[layer][1])
        qr, q, latent_current = profiler._projection(attention, x)
        latent_full, _ = latent.update_and_fetch(latent_current, latent_current)
        updated = profiler._pool_update(attention.indexer, pool, x)
        query = attention.indexer.wq_b(qr).reshape(
            1,
            1,
            attention.indexer.n_heads,
            attention.indexer.head_dim,
        )
        scored = profiler._score(attention.indexer, pool, x, query)
        selected = profiler._select(attention.indexer, scored)
        indices = profiler._expand(
            attention.indexer,
            pool,
            selected[0],
            selected[1],
            updated["valid"],
        )
        gathered = profiler._gather(latent_full, indices)
        prepared_output = profiler._attention(
            operator,
            attention,
            q,
            gathered[0],
            gathered[1],
            pool,
        )
        profiler._eval(
            (
                qr,
                q,
                latent_current,
                updated,
                query,
                scored,
                selected,
                indices,
                gathered,
                prepared_output,
                update_pool.state,
            )
        )
        rows.append(
            {
                "layer": layer,
                "attention": attention,
                "x": x,
                "entry": entry,
                "update_pool": update_pool,
                "qr": qr,
                "q": q,
                "latent_current": latent_current,
                "latent_full": latent_full,
                "updated": updated,
                "query": query,
                "scored": scored,
                "selected": selected,
                "indices": indices,
                "gathered": gathered,
            }
        )
    mx.clear_cache()
    mx.synchronize()
    return rows


def _stage_outputs(stage: str, rows, operator):
    values = []
    for row in rows:
        attention = row["attention"]
        if stage == "index_query_projection":
            value = attention.indexer.wq_b(row["qr"]).reshape(
                1,
                1,
                attention.indexer.n_heads,
                attention.indexer.head_dim,
            )
        elif stage == "indexpool_update":
            value = profiler._pool_update(
                attention.indexer, row["update_pool"], row["x"]
            )
        elif stage == "pooled_key_score":
            value = profiler._score(
                attention.indexer, row["entry"][1], row["x"], row["query"]
            )
        elif stage == "topk_selection":
            value = profiler._select(attention.indexer, row["scored"])
        elif stage == "pool_token_expansion":
            value = profiler._expand(
                attention.indexer,
                row["entry"][1],
                row["selected"][0],
                row["selected"][1],
                row["updated"]["valid"],
            )
        elif stage == "sanitize_gather_preparation":
            value = profiler._gather(row["latent_full"], row["indices"])
        elif stage == "sparse_attention":
            value = profiler._attention(
                operator,
                attention,
                row["q"],
                row["gathered"][0],
                row["gathered"][1],
                row["entry"][1],
            )
        else:
            raise ValueError(f"unknown trace stage: {stage}")
        values.append(value)
    return values


def _run_stage(args, stage, model, source, rows, operator):
    wall = []
    build = []
    baseline = int(mx.get_active_memory())
    mx.reset_peak_memory()
    token = mx.array([[profiler.PROFILE_TOKEN]], dtype=mx.uint32)
    for _ in range(args.repetitions):
        started = time.perf_counter_ns()
        if stage == "full_model_decode":
            value = model(token, cache=source).logits[0, -1]
        else:
            value = _stage_outputs(stage, rows, operator)
        built = time.perf_counter_ns()
        profiler._eval(value)
        finished = time.perf_counter_ns()
        build.append((built - started) / 1e6)
        wall.append((finished - started) / 1e6)
    return {
        "stage": stage,
        "repetitions": args.repetitions,
        "cpu_graph_build_ms_median": statistics.median(build),
        "synchronized_wall_ms_median": statistics.median(wall),
        "cpu_graph_build_samples_ms": build,
        "synchronized_wall_samples_ms": wall,
        "active_memory_before_bytes": baseline,
        "active_memory_after_bytes": int(mx.get_active_memory()),
        "working_peak_bytes": max(0, int(mx.get_peak_memory()) - baseline),
    }


def _child(args) -> int:
    global mx
    import mlx.core as mx

    from glm53_flash_mlx.loader import load, warm_residency

    _, operator, boundary = profiler._load_probe_modules()
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    model, _ = load(
        args.model,
        experimental_packed_decode_moe=True,
        experimental_compact_nope_dsa_cache=True,
        compact_cache_capacity_tokens=args.context + 32,
    )
    warm_residency(model)
    source = boundary._synthetic_cache(model, args.context, "compact-nope-dsa")
    rows = _prepare_layers(model, source, boundary, operator, args.context)
    atomic_write(
        args.ready,
        {"pid": os.getpid(), "context_tokens": args.context, "stages": TRACE_STAGES},
    )
    for stage in TRACE_STAGES:
        stage_ready = args.command_dir / f"{stage}.ready"
        stage_go = args.command_dir / f"{stage}.go"
        stage_done = args.command_dir / f"{stage}.done.json"
        trace_done = args.command_dir / f"{stage}.trace-done"
        stage_ready.touch()
        _wait_for(stage_go, 120.0)
        result = _run_stage(args, stage, model, source, rows, operator)
        atomic_write(stage_done, result)
        _wait_for(trace_done, 120.0)
    profiler._release(rows, source)
    return 0


def _record_stage(args, child, pid: int, stage: str, command_dir: Path):
    ready = command_dir / f"{stage}.ready"
    go = command_dir / f"{stage}.go"
    done = command_dir / f"{stage}.done.json"
    trace_done = command_dir / f"{stage}.trace-done"
    _wait_for(ready, 900.0, child)
    trace = args.trace_dir / f"indexer-{args.context}-{stage}.trace"
    log = command_dir / f"{stage}.xctrace.log"
    with log.open("w") as handle:
        tracer = subprocess.Popen(
            [
                "xcrun",
                "xctrace",
                "record",
                "--template",
                "Metal System Trace",
                "--time-limit",
                f"{TRACE_TIME_LIMIT_SECONDS}s",
                "--output",
                str(trace),
                "--no-prompt",
                "--attach",
                str(pid),
            ],
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                text = log.read_text() if log.exists() else ""
                if "Ctrl-C to stop the recording" in text:
                    break
                if tracer.poll() is not None:
                    raise RuntimeError(f"xctrace exited early: {text}")
                time.sleep(0.2)
            else:
                raise TimeoutError("xctrace did not report recording readiness")
            go.touch()
            _wait_for(done, 120.0, child)
            if tracer.wait(timeout=120.0) != 0:
                raise RuntimeError(f"xctrace failed: {log.read_text()}")
        except Exception:
            terminate_process_group(tracer)
            remove_partial_trace(trace)
            raise
    size = path_bytes(trace)
    if size > MAX_TRACE_BYTES:
        remove_partial_trace(trace)
        raise RuntimeError(f"{stage} trace exceeded the 2 GiB budget")
    telemetry = bounded._export_telemetry(trace, pid)
    identity = _trace_identity(trace)
    identity.update(path=str(trace), stored_in_repository=False)
    trace_done.touch()
    child_row = json.loads(done.read_text())
    return {
        **child_row,
        "telemetry": telemetry,
        "trace": identity,
        "trace_time_limit_seconds": TRACE_TIME_LIMIT_SECONDS,
        "replayable": False,
        "static_kernel_labels_used_for_attribution": False,
    }


def _parent(args) -> int:
    artifact = None
    if args.output.exists():
        candidate = json.loads(args.output.read_text())
        if (
            candidate.get("schema")
            == "glm53-lightning-indexer-bounded-telemetry-v1"
            and int(candidate.get("context_tokens", -1)) == args.context
        ):
            artifact = candidate
    completed = set(artifact.get("stages", {})) if artifact else set()
    args.trace_dir = _validate_trace_dir(args.trace_dir, args.context, completed)
    with tempfile.TemporaryDirectory(prefix="glm53-indexer-telemetry-") as temporary:
        command_dir = Path(temporary)
        ready = command_dir / "child-ready.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            str(args.model),
            "--context",
            str(args.context),
            "--trace-dir",
            str(args.trace_dir),
            "--output",
            str(args.output),
            "--repetitions",
            str(args.repetitions),
            "--wired-limit-gb",
            str(args.wired_limit_gb),
            "--cache-limit-gb",
            str(args.cache_limit_gb),
            "--child",
            "--ready",
            str(ready),
            "--command-dir",
            str(command_dir),
        ]
        child = subprocess.Popen(command, start_new_session=True)
        if artifact is None:
            artifact = {
                "schema": "glm53-lightning-indexer-bounded-telemetry-v1",
                "complete": False,
                "context_tokens": args.context,
                "repetitions": args.repetitions,
                "stages": {},
                "capture_contract": {
                    "replayable": False,
                    "full_model_resources_embedded": False,
                    "dynamic_pid_attribution": True,
                    "static_kernel_labels_used": False,
                    "per_stage_trace": True,
                },
            }
        elif int(artifact["repetitions"]) != args.repetitions:
            raise ValueError("resume repetitions differ from the existing artifact")
        try:
            _wait_for(ready, 900.0, child)
            pid = int(json.loads(ready.read_text())["pid"])
            artifact["pid"] = pid
            for stage in TRACE_STAGES:
                if stage in artifact["stages"]:
                    _wait_for(command_dir / f"{stage}.ready", 900.0, child)
                    (command_dir / f"{stage}.go").touch()
                    _wait_for(command_dir / f"{stage}.done.json", 120.0, child)
                    (command_dir / f"{stage}.trace-done").touch()
                    continue
                artifact["stages"][stage] = _record_stage(
                    args, child, pid, stage, command_dir
                )
                artifact["last_completed_stage"] = stage
                atomic_write(args.output, artifact)
            if child.wait(timeout=180.0) != 0:
                raise RuntimeError("telemetry child failed")
            artifact["complete"] = set(artifact["stages"]) == set(TRACE_STAGES)
            artifact["total_trace_bytes"] = sum(
                row["trace"]["bytes"] for row in artifact["stages"].values()
            )
            atomic_write(args.output, artifact)
            print(json.dumps({"output": str(args.output), "complete": True}, indent=2))
            return 0
        except Exception:
            terminate_process_group(child)
            atomic_write(args.output, artifact)
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--context", type=int, choices=profiler.CONTEXTS, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ready", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--command-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    return _child(args) if args.child else _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
