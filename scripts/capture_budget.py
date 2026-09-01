"""Resource-budget supervision shared by Metal capture probes."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CaptureBudget:
    max_elapsed_s: float = 900.0
    max_trace_bytes: int = 32 << 30
    min_free_bytes: int = 64 << 30


def atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def path_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def observation(path: Path, started: float) -> dict:
    return {
        "elapsed_seconds": time.monotonic() - started,
        "observed_trace_bytes": path_bytes(path),
        "free_bytes": shutil.disk_usage(path.parent).free,
    }


def violation(observed: dict, budget: CaptureBudget) -> str | None:
    if observed["elapsed_seconds"] > budget.max_elapsed_s:
        return "max_elapsed_s"
    if observed["observed_trace_bytes"] > budget.max_trace_bytes:
        return "max_trace_bytes"
    if observed["free_bytes"] < budget.min_free_bytes:
        return "min_free_bytes"
    return None


def terminate_process_group(process: subprocess.Popen, grace_s: float = 10.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=grace_s)


def remove_partial_trace(path: Path) -> bool:
    if path.is_dir():
        shutil.rmtree(path)
        return True
    if path.exists():
        path.unlink()
        return True
    return False


def supervise_capture(
    command: list[str],
    *,
    trace_path: Path,
    evidence_path: Path,
    budget: CaptureBudget,
    metadata: dict,
    poll_s: float = 1.0,
) -> int:
    started = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True)
    last = observation(trace_path, started)
    while process.poll() is None:
        last = observation(trace_path, started)
        reason = violation(last, budget)
        if reason is not None:
            terminate_process_group(process)
            deleted = remove_partial_trace(trace_path)
            atomic_write(
                evidence_path,
                {
                    "schema": "glm53-capture-resource-budget-negative-v1",
                    "status": "aborted_resource_budget",
                    "capture_complete": False,
                    "correctness_claim": False,
                    "budget": asdict(budget),
                    "violation": reason,
                    **last,
                    "partial_trace_deleted": deleted,
                    **metadata,
                },
            )
            return 2
        time.sleep(poll_s)
    returncode = int(process.returncode or 0)
    if returncode != 0:
        last = observation(trace_path, started)
        deleted = remove_partial_trace(trace_path)
        atomic_write(
            evidence_path,
            {
                "schema": "glm53-capture-resource-budget-negative-v1",
                "status": "capture_process_failed",
                "capture_complete": False,
                "correctness_claim": False,
                "budget": asdict(budget),
                "returncode": returncode,
                **last,
                "partial_trace_deleted": deleted,
                **metadata,
            },
        )
    return returncode
