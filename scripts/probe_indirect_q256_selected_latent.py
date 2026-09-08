#!/usr/bin/env python3
"""Qualify device-count indirect dispatch for Q256 union latent gather."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NATIVE_PACKAGE = ROOT / "native_execution"
if str(NATIVE_PACKAGE) not in sys.path:
    sys.path.insert(0, str(NATIVE_PACKAGE))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from glm53_native_execution import NativeIndirectSelectedLatentPlan
import probe_device_resident_q256_selected_union as union_probe


DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-indirect-q256-selected-latent-20260908.json"
)
CONTEXTS = (32 << 10, 128 << 10, 320 << 10)
LATENT_DIM = 512
ROWS_PER_GROUP = 8
SAMPLES = 5
MAX_320K_WALL_MS = 5.0
MAX_320K_SCRATCH_BYTES = 384 << 20


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _latent(context: int) -> mx.array:
    rows = mx.arange(context, dtype=mx.uint32)[:, None] % 251
    columns = mx.arange(LATENT_DIM, dtype=mx.uint32)[None, :] % 17
    latent = mx.contiguous(
        (rows.astype(mx.float32) + columns.astype(mx.float32) / 32.0).astype(
            mx.bfloat16
        )
    )
    mx.eval(latent)
    mx.clear_cache()
    return latent


def _bits(value: mx.array) -> np.ndarray:
    return np.asarray(value.view(mx.uint16), dtype=np.uint16)


def _execute(plan, indices, valid, latent):
    outputs = plan.execute(indices, valid, latent)
    mx.eval(*outputs)
    # Diagnostic readback only. The native plan keeps union_count and indirect
    # geometry on device and reports zero internal host synchronizations.
    mx.synchronize()
    return outputs


def _context_case(context: int) -> dict[str, object]:
    print(json.dumps({"phase": "indirect_selected_latent", "context": context}), flush=True)
    host_indices, host_valid, indices, valid = union_probe._fixture(context)
    reference_union, _ = union_probe._reference(host_indices, host_valid, context)
    latent = _latent(context)
    reference = mx.take(latent, mx.array(reference_union), axis=0)
    mx.eval(reference)

    plan = NativeIndirectSelectedLatentPlan(context)
    identities = list(plan.buffer_identities)
    union, count, gathered = _execute(plan, indices, valid, latent)
    count_value = int(np.asarray(count, dtype=np.uint32)[0])
    arguments = np.asarray(
        plan.debug_indirect_arguments, dtype=np.uint32
    ).copy()
    union_host = np.asarray(union, dtype=np.int32)[:count_value].copy()
    gathered_bits = _bits(gathered[:count_value]).copy()
    reference_bits = _bits(reference).copy()

    for _ in range(2):
        _execute(plan, indices, valid, latent)
    samples = []
    for _ in range(SAMPLES):
        started = time.perf_counter_ns()
        plan.execute(indices, valid, latent)
        mx.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)

    expected_groups = (
        reference_union.size + ROWS_PER_GROUP - 1
    ) // ROWS_PER_GROUP
    result = {
        "context_tokens": context,
        "union_count": count_value,
        "reference_union_count": int(reference_union.size),
        "union_indices_byte_exact": np.array_equal(
            union_host, reference_union
        ),
        "union_latent_byte_exact": np.array_equal(
            gathered_bits, reference_bits
        ),
        "indirect_arguments": [int(value) for value in arguments],
        "expected_indirect_arguments": [int(expected_groups), 1, 1],
        "indirect_arguments_exact": np.array_equal(
            arguments,
            np.asarray([expected_groups, 1, 1], dtype=np.uint32),
        ),
        "median_wall_ms": statistics.median(samples),
        "samples_ms": samples,
        "scratch_bytes": plan.scratch_bytes,
        "buffer_identities_stable": identities == list(plan.buffer_identities),
        "execution_count": plan.execution_count,
        "dynamic_allocation_count": plan.dynamic_allocation_count,
        "graph_node_count": plan.graph_node_count,
        "shape_discovery_count": plan.shape_discovery_count,
        "host_synchronization_count": plan.host_synchronization_count,
        "returned_intermediate_tensor_bytes": (
            plan.returned_intermediate_tensor_bytes
        ),
        "union_count_is_device_buffer": str(count.dtype) == "mlx.core.uint32",
    }
    del plan, indices, valid, latent, reference, union, count, gathered
    mx.clear_cache()
    gc.collect()
    return result


def _artifact(contexts: dict[str, object]) -> dict[str, object]:
    complete = set(map(int, contexts)) == set(CONTEXTS)
    long = contexts.get(str(320 << 10), {})
    checks = {
        "32k_128k_320k_measured": complete,
        "union_and_gather_byte_exact": complete and all(
            row["union_indices_byte_exact"]
            and row["union_latent_byte_exact"]
            for row in contexts.values()
        ),
        "device_indirect_arguments_exact": complete and all(
            row["indirect_arguments_exact"] for row in contexts.values()
        ),
        "union_count_remains_device_resident": complete and all(
            row["union_count_is_device_buffer"] for row in contexts.values()
        ),
        "fixed_execution_contract": complete and all(
            row["buffer_identities_stable"]
            and row["dynamic_allocation_count"] == 0
            and row["graph_node_count"] == 0
            and row["shape_discovery_count"] == 0
            and row["host_synchronization_count"] == 0
            and row["returned_intermediate_tensor_bytes"] == 0
            for row in contexts.values()
        ),
        "320k_wall_at_most_5ms": (
            complete and long["median_wall_ms"] <= MAX_320K_WALL_MS
        ),
        "320k_scratch_at_most_384mib": (
            complete and long["scratch_bytes"] <= MAX_320K_SCRATCH_BYTES
        ),
    }
    accepted = all(checks.values())
    return {
        "schema": "glm53-indirect-q256-selected-latent-v1",
        "date": date.today().isoformat(),
        "complete": complete,
        "accepted": accepted,
        "decision": (
            "advance_indirect_union_to_streaming_selected_kv_projection"
            if accepted
            else "stop_or_redesign_indirect_union_gather"
        ),
        "contexts": contexts,
        "checks": checks,
        "probe_only": True,
        "runtime_changes": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    contexts = {}
    artifact = _artifact(contexts)
    for context in CONTEXTS:
        contexts[str(context)] = _context_case(context)
        artifact = _artifact(contexts)
        _atomic_write(args.output, artifact)
        if not (
            contexts[str(context)]["union_indices_byte_exact"]
            and contexts[str(context)]["union_latent_byte_exact"]
            and contexts[str(context)]["indirect_arguments_exact"]
        ):
            break
    print(json.dumps({
        "output": str(args.output),
        "complete": artifact["complete"],
        "accepted": artifact["accepted"],
        "decision": artifact["decision"],
        "failed_gates": [
            name for name, passed in artifact["checks"].items() if not passed
        ],
    }))
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
