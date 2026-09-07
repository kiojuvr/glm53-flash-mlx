#!/usr/bin/env python3
"""Qualify the exact native IndexPool update island through the real runtime.

The phases are intentionally resumable at the artifact level and execute in
fresh processes when invoked separately.  ``screen`` measures the actual
CompactIndexPoolCache integration at 2K/256K.  ``long`` performs the 4,096-step
exact differential.  ``server`` verifies the explicit production flag and
health evidence.  No phase changes the default backend.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import mlx.core as mx

from glm53_flash_mlx.abi import MLX_VLM_REVISION, NOPE_DSA_CACHE_ABI_COMPACT
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint
from glm53_flash_mlx.native_indexpool_runtime import (
    ENVIRONMENT_FLAG,
    NATIVE_INDEXPOOL_RUNTIME_ABI,
    registry_snapshot,
    require_available,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-indexpool-runtime-20260907.json"
)
SOURCE_ISLAND = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-indexpool-update-submission-island-20260907.json"
)
SOURCE_PACKED = (
    ROOT
    / "bench-results"
    / "m3ultra512-exact-fused-packed-decode-runtime-20260907.json"
)
CONTEXTS = (2_048, 262_144)
MATERIALIZATION_INTERVAL = 256
LONG_STEPS = 4_096
MAX_PROCESS_PEAK_BYTES = 340_000_000_000


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


def _atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_helpers():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import probe_exact_sigmoid_gate_metal_barrier as oracle_probe
    import probe_long_context_first_decode_boundary as boundary
    import probe_native_execution_engine_feasibility as tier0

    return oracle_probe, boundary, tier0


@contextmanager
def _native_mode(active: bool):
    previous = os.environ.get(ENVIRONMENT_FLAG)
    os.environ[ENVIRONMENT_FLAG] = "1" if active else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(ENVIRONMENT_FLAG, None)
        else:
            os.environ[ENVIRONMENT_FLAG] = previous


def _source_evidence() -> dict:
    island = json.loads(SOURCE_ISLAND.read_text())
    packed = json.loads(SOURCE_PACKED.read_text())
    if not island.get("accepted"):
        raise RuntimeError("native IndexPool source island is not accepted")
    if not packed.get("acceptance", {}).get("accepted"):
        raise RuntimeError("exact fused packed runtime correctness source failed")
    return {
        "native_indexpool_island": {
            "path": str(SOURCE_ISLAND.relative_to(ROOT)),
            "sha256": _sha256(SOURCE_ISLAND),
            "decision": island.get("decision"),
        },
        "exact_fused_packed_runtime": {
            "path": str(SOURCE_PACKED.relative_to(ROOT)),
            "sha256": _sha256(SOURCE_PACKED),
            "correctness_accepted": True,
            "release_15_tps_accepted": packed.get("accepted"),
        },
    }


def _new_artifact(model: Path) -> dict:
    report = inspect_checkpoint(model, require_server_ready=True)
    return {
        "schema": "glm53-native-indexpool-runtime-qualification-v1",
        "date": date.today().isoformat(),
        "complete": False,
        "accepted": False,
        "release_15_tps_accepted": False,
        "checkpoint_fingerprint": report.fingerprint,
        "official_hf_revision": report.official_revision,
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "compact_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
        "native_runtime_abi": NATIVE_INDEXPOOL_RUNTIME_ABI,
        "extension": require_available(),
        "source_artifacts": _source_evidence(),
        "phases": {},
        "runtime_changes": {
            "opt_in_native_indexpool_runtime": True,
            "default_backend": False,
            "packed_moe_abi": False,
            "cache_abi": False,
            "admission": False,
        },
    }


def _load_artifact(path: Path, model: Path) -> dict:
    if not path.exists():
        return _new_artifact(model)
    artifact = json.loads(path.read_text())
    fresh = _new_artifact(model)
    if artifact.get("schema") != fresh["schema"]:
        raise RuntimeError("existing output has a different schema")
    if artifact.get("checkpoint_fingerprint") != fresh["checkpoint_fingerprint"]:
        raise RuntimeError("existing output belongs to a different checkpoint")
    artifact.update(
        date=fresh["date"],
        extension=fresh["extension"],
        source_artifacts=fresh["source_artifacts"],
    )
    return artifact


def _eval_logits(output) -> None:
    mx.eval(output.logits)
    mx.synchronize()


def _median(rows: list[dict]) -> dict:
    wall = statistics.median(row["wall_ms"] for row in rows)
    return {
        "median_wall_ms": wall,
        "median_host_submit_ms": statistics.median(
            row["host_submit_ms"] for row in rows
        ),
        "tokens_per_second": 1_000.0 / wall,
        "samples": rows,
    }


def _screen_context(model, source, context, boundary, tier0, warmups, samples):
    caches = {
        "mlx": boundary._clone_cache(source, context + warmups + samples + 16),
        "native": boundary._clone_cache(source, context + warmups + samples + 16),
    }
    timing = {name: [] for name in caches}
    hashes = {name: [] for name in caches}
    tokens = {name: [] for name in caches}
    before = registry_snapshot()
    for iteration in range(warmups + samples):
        order = ("mlx", "native") if iteration % 2 == 0 else ("native", "mlx")
        token = mx.array([[3_000 + iteration]], dtype=mx.uint32)
        for arm in order:
            started = time.perf_counter_ns()
            with _native_mode(arm == "native"):
                output = model(token, cache=caches[arm])
            submitted = time.perf_counter_ns()
            _eval_logits(output)
            finished = time.perf_counter_ns()
            hashes[arm].append(tier0._hash(output.logits[0, -1]))
            tokens[arm].append(int(mx.argmax(output.logits[0, -1]).item()))
            if iteration >= warmups:
                timing[arm].append(
                    {
                        "host_submit_ms": (submitted - started) / 1e6,
                        "wall_ms": (finished - started) / 1e6,
                    }
                )
    after = registry_snapshot()
    measured = {arm: _median(rows) for arm, rows in timing.items()}
    result = {
        "context_tokens": context,
        "timing": measured,
        "native_wall_saving_ms": (
            measured["mlx"]["median_wall_ms"]
            - measured["native"]["median_wall_ms"]
        ),
        "native_speedup": (
            measured["mlx"]["median_wall_ms"]
            / measured["native"]["median_wall_ms"]
        ),
        "all_full_vocab_logits_byte_exact": hashes["mlx"] == hashes["native"],
        "all_generated_tokens_exact": tokens["mlx"] == tokens["native"],
        "post_state_byte_exact": boundary._cache_exact(
            caches["mlx"], caches["native"]
        ),
        "native_execution_count_delta": (
            after["execution_count"] - before["execution_count"]
        ),
        "expected_native_execution_count": 11 * (warmups + samples),
        "native_registry": after,
    }
    caches.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    return result


def _load_runtime_model(model_path: Path, capacity: int):
    os.environ[ENVIRONMENT_FLAG] = "0"
    model, processor = load(
        model_path,
        experimental_packed_decode_moe=True,
        experimental_compact_nope_dsa_cache=True,
        compact_cache_capacity_tokens=capacity,
    )
    warm_residency(model)
    return model, processor


def _run_screen(args, artifact):
    oracle_probe, boundary, tier0 = _load_helpers()
    _progress("load_model", phase_group="screen")
    model, processor = _load_runtime_model(args.model, max(CONTEXTS) + 64)
    report = inspect_checkpoint(args.model, require_server_ready=True)
    with _native_mode(True):
        official = oracle_probe._official_oracle(model, processor, report)
    rows = {}
    for context in CONTEXTS:
        _progress("decode_context", context=context)
        with _native_mode(False):
            source = boundary._synthetic_cache(model, context, "compact-nope-dsa")
        rows[str(context)] = _screen_context(
            model,
            source,
            context,
            boundary,
            tier0,
            args.warmups,
            args.samples,
        )
        source.clear()
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
    short = rows["2048"]
    long = rows["262144"]
    short_tps = short["timing"]["native"]["tokens_per_second"]
    gates = {
        "official_16_128_oracle_exact": official.get(
            "all_full_vocab_logits_hashes_match", False
        ),
        "both_contexts_logits_tokens_state_exact": all(
            row["all_full_vocab_logits_byte_exact"]
            and row["all_generated_tokens_exact"]
            and row["post_state_byte_exact"]
            for row in rows.values()
        ),
        "both_contexts_execute_all_11_dsa_layers": all(
            row["native_execution_count_delta"]
            == row["expected_native_execution_count"]
            for row in rows.values()
        ),
        "2k_regression_at_most_1_percent": short["native_speedup"] >= 0.99,
        "256k_wall_saving_at_least_0_75ms": long["native_wall_saving_ms"] >= 0.75,
        "process_peak_at_most_340GB": int(mx.get_peak_memory())
        <= MAX_PROCESS_PEAK_BYTES,
    }
    artifact["phases"]["screen"] = {
        "complete": True,
        "official_oracle": official,
        "contexts": rows,
        "process_peak_memory_bytes": int(mx.get_peak_memory()),
        "release_15_tps_gate": short_tps >= 15.0,
        "gates": gates,
        "accepted": all(gates.values()),
    }
    del model, processor
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def _run_long(args, artifact):
    _, boundary, tier0 = _load_helpers()
    _progress("load_model", phase_group="long")
    model, _ = _load_runtime_model(args.model, 8_256)
    with _native_mode(False):
        source = boundary._synthetic_cache(model, 2_048, "compact-nope-dsa")
    caches = {
        "mlx": boundary._clone_cache(source, 8_256),
        "native": boundary._clone_cache(source, 8_256),
    }
    source.clear()
    before = registry_snapshot()
    token = 3_000
    evidence = {}
    first_divergence = None
    nan_count = 0
    materializations = 0
    for step in range(1, LONG_STEPS + 1):
        value = mx.array([[token]], dtype=mx.uint32)
        with _native_mode(False):
            left = model(value, cache=caches["mlx"])
        with _native_mode(True):
            right = model(value, cache=caches["native"])
        equal = mx.array_equal(left.logits, right.logits)
        left_token = mx.argmax(left.logits[0, -1])
        right_token = mx.argmax(right.logits[0, -1])
        nans = mx.sum(mx.isnan(left.logits)) + mx.sum(mx.isnan(right.logits))
        mx.eval(equal, left_token, right_token, nans)
        nan_count += int(nans.item())
        if not bool(equal.item()) or int(left_token.item()) != int(right_token.item()):
            first_divergence = {
                "step": step,
                "logits_exact": bool(equal.item()),
                "mlx_token": int(left_token.item()),
                "native_token": int(right_token.item()),
            }
            break
        token = int(left_token.item())
        if step % MATERIALIZATION_INTERVAL == 0:
            boundary._materialize_cache(caches["mlx"])
            boundary._materialize_cache(caches["native"])
            materializations += 1
            state_exact = boundary._cache_exact(caches["mlx"], caches["native"])
            evidence[str(step)] = {
                "logits_hash": tier0._hash(left.logits[0, -1]),
                "state_exact": state_exact,
            }
            _progress("long_checkpoint", step=step, state_exact=state_exact)
            artifact["phases"]["long"] = {
                "complete": False,
                "steps_completed": step,
                "evidence": evidence,
            }
            _atomic_write(args.output, artifact)
            if not state_exact:
                first_divergence = {"step": step, "stage": "cache_state"}
                break
    after = registry_snapshot()
    completed = first_divergence is None and len(evidence) == 16
    final_exact = completed and boundary._cache_exact(
        caches["mlx"], caches["native"]
    )
    expected_executes = 11 * LONG_STEPS
    gates = {
        "4096_steps_completed": completed,
        "all_full_vocab_logits_and_tokens_exact": first_divergence is None,
        "all_16_materialization_checkpoints_exact": materializations == 16
        and all(row["state_exact"] for row in evidence.values()),
        "final_cache_state_exact": final_exact,
        "all_11_dsa_layers_executed_each_step": (
            after["execution_count"] - before["execution_count"]
        )
        == expected_executes,
        "nan_count_zero": nan_count == 0,
        "process_peak_at_most_340GB": int(mx.get_peak_memory())
        <= MAX_PROCESS_PEAK_BYTES,
    }
    artifact["phases"]["long"] = {
        "complete": True,
        "steps_completed": LONG_STEPS if completed else max(map(int, evidence), default=0),
        "materialization_count": materializations,
        "evidence": evidence,
        "first_divergence": first_divergence,
        "nan_count": nan_count,
        "native_execution_count_delta": after["execution_count"]
        - before["execution_count"],
        "expected_native_execution_count": expected_executes,
        "process_peak_memory_bytes": int(mx.get_peak_memory()),
        "gates": gates,
        "accepted": all(gates.values()),
    }
    caches.clear()
    del model
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def _run_server(args, artifact):
    executable = Path(sys.executable).with_name("glm53-serve")
    command = [
        str(executable),
        "--model",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.server_port),
        "--experimental-packed-decode-moe",
        "--experimental-compact-nope-dsa-cache",
        "--experimental-native-indexpool-update",
    ]
    started = time.perf_counter()
    import urllib.request

    with tempfile.NamedTemporaryFile(mode="w+", prefix="glm53-native-indexpool-") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
            start_new_session=True,
        )
        status = None
        body = None
        metrics_status = None
        metrics = None
        error = None
        ready_seconds = None
        try:
            while time.perf_counter() - started < args.server_timeout:
                if process.poll() is not None:
                    error = f"server exited with {process.returncode}"
                    break
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{args.server_port}/health", timeout=2
                    ) as response:
                        status = int(response.status)
                        body = json.loads(response.read().decode())
                    if status == 200:
                        ready_seconds = time.perf_counter() - started
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{args.server_port}/v1/metrics",
                            timeout=5,
                        ) as response:
                            metrics_status = int(response.status)
                            metrics = json.loads(response.read().decode())
                        break
                except Exception:
                    time.sleep(1.0)
            else:
                error = "server readiness timeout"
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
            log.flush()
            log.seek(0)
            lines = log.read().splitlines()
    native_health = (metrics or {}).get("server", {}).get(
        "native_indexpool_update", {}
    )
    gates = {
        "health_http_200": status == 200,
        "metrics_http_200": metrics_status == 200,
        "native_runtime_abi_reported": native_health.get("abi")
        == NATIVE_INDEXPOOL_RUNTIME_ABI,
        "ready_at_most_190_seconds": ready_seconds is not None
        and ready_seconds <= 190.0,
    }
    artifact["phases"]["server"] = {
        "complete": True,
        "command": command,
        "ready_seconds": ready_seconds,
        "health_http_status": status,
        "health": body,
        "metrics_http_status": metrics_status,
        "metrics": metrics,
        "error": error,
        "log_tail": lines[-40:],
        "gates": gates,
        "accepted": all(gates.values()),
    }


def _finalize(artifact: dict) -> None:
    required = ("screen", "long", "server")
    artifact["complete"] = all(
        artifact.get("phases", {}).get(name, {}).get("complete") for name in required
    )
    artifact["accepted"] = artifact["complete"] and all(
        artifact["phases"][name].get("accepted") for name in required
    )
    artifact["release_15_tps_accepted"] = bool(
        artifact.get("phases", {}).get("screen", {}).get("release_15_tps_gate")
    )
    if artifact["accepted"] and artifact["release_15_tps_accepted"]:
        artifact["decision"] = "keep_native_indexpool_runtime_and_pass_15_tps"
    elif artifact["accepted"]:
        artifact["decision"] = "keep_native_indexpool_runtime_opt_in_15_tps_pending"
    elif artifact["complete"]:
        artifact["decision"] = "stop_or_requalify_native_indexpool_runtime"
    else:
        artifact["decision"] = "qualification_incomplete"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--phase", choices=("screen", "long", "server", "all"), default="screen"
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    parser.add_argument("--server-port", type=int, default=18084)
    parser.add_argument("--server-timeout", type=float, default=240.0)
    args = parser.parse_args()

    artifact = _load_artifact(args.output, args.model)
    try:
        mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
        mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
        phases = ("screen", "long", "server") if args.phase == "all" else (args.phase,)
        for phase in phases:
            if phase == "screen":
                _run_screen(args, artifact)
            elif phase == "long":
                _run_long(args, artifact)
            else:
                _run_server(args, artifact)
            _finalize(artifact)
            _atomic_write(args.output, artifact)
    except Exception as error:
        artifact["last_error"] = {
            "phase": args.phase,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        _finalize(artifact)
        _atomic_write(args.output, artifact)
        raise

    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": artifact["complete"],
                "accepted": artifact["accepted"],
                "release_15_tps_accepted": artifact["release_15_tps_accepted"],
                "decision": artifact["decision"],
                "completed_phases": sorted(artifact["phases"]),
            },
            indent=2,
        )
    )
    return 0 if artifact["accepted"] or not artifact["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
