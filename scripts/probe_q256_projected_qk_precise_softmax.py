#!/usr/bin/env python3
"""Keep exact Q256 projected-QK scores native through precise softmax."""

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

from glm53_native_execution import NativeProjectedQKUnionTileLoopPlan
import probe_q256_projected_qk_union_tile_loop as q256


DEFAULT_OUTPUT = ROOT / "bench-results" / (
    "m3ultra512-q256-projected-qk-precise-softmax-20260909.json"
)
MAX_SCRATCH_BYTES = 450 << 20
MAX_INCREMENTAL_SOFTMAX_MS = 10.0


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _bits(value: mx.array) -> np.ndarray:
    return np.asarray(value.view(mx.uint16), dtype=np.uint16)


def _run_scores(plan, inputs):
    result = plan.execute(
        inputs["selected_indices"], inputs["selected_valid"], inputs["latent"],
        inputs["key_weight"], inputs["query"], q256.ATTENTION_SCALE,
    )
    mx.eval(result)
    mx.synchronize()
    return result


def _run_probabilities(plan, inputs):
    result = plan.execute_probabilities(
        inputs["selected_indices"], inputs["selected_valid"], inputs["latent"],
        inputs["key_weight"], inputs["query"], q256.ATTENTION_SCALE,
    )
    mx.eval(result)
    mx.synchronize()
    return result


def _timed(operation) -> dict[str, object]:
    for _ in range(2):
        operation()
    samples = []
    for _ in range(3):
        started = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return {"median_wall_ms": statistics.median(samples), "samples_ms": samples}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    print(json.dumps({"phase": "build_q256_fixture"}), flush=True)
    inputs, _, _ = q256._fixture()
    print(json.dumps({"phase": "direct_precise_softmax_reference"}), flush=True)
    direct = q256._direct(inputs)
    probabilities = mx.softmax(direct["scores"], axis=-1, precise=True)
    mx.eval(probabilities)
    reference_score_bits = _bits(direct["scores"]).copy()
    reference_probability_bits = _bits(probabilities).copy()

    score_plan = NativeProjectedQKUnionTileLoopPlan(
        q256.PHYSICAL_K, q256.QUERY_ROWS, q256.TILE_ROWS
    )
    softmax_plan = NativeProjectedQKUnionTileLoopPlan(
        q256.PHYSICAL_K, q256.QUERY_ROWS, q256.TILE_ROWS, True
    )
    identities = list(softmax_plan.buffer_identities)
    print(json.dumps({"phase": "native_precise_softmax"}), flush=True)
    candidate = _run_probabilities(softmax_plan, inputs)
    anchors = {
        "qk_scores": np.array_equal(
            _bits(softmax_plan.debug_attention_scores), reference_score_bits
        ),
        "precise_softmax_probabilities": np.array_equal(
            _bits(candidate), reference_probability_bits
        ),
    }

    del direct, probabilities, reference_score_bits, reference_probability_bits
    mx.clear_cache()
    gc.collect()
    print(json.dumps({"phase": "timing"}), flush=True)
    score_timing = _timed(lambda: _run_scores(score_plan, inputs))
    probability_timing = _timed(
        lambda: _run_probabilities(softmax_plan, inputs)
    )
    incremental_ms = (
        probability_timing["median_wall_ms"] - score_timing["median_wall_ms"]
    )
    checks = {
        "qk_and_precise_softmax_byte_exact": all(anchors.values()),
        "softmax_stays_in_fixed_native_execution_scope": (
            identities == list(softmax_plan.buffer_identities)
            and softmax_plan.dynamic_allocation_count == 0
            and softmax_plan.graph_node_count == 0
            and softmax_plan.shape_discovery_count == 0
            and softmax_plan.host_synchronization_count == 0
            and softmax_plan.returned_intermediate_tensor_bytes == 0
        ),
        "scratch_at_most_450mib": softmax_plan.scratch_bytes <= MAX_SCRATCH_BYTES,
        "incremental_precise_softmax_at_most_10ms": (
            incremental_ms <= MAX_INCREMENTAL_SOFTMAX_MS
        ),
    }
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-q256-projected-qk-precise-softmax-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "advance_q256_probabilities_to_tiled_value_pass"
            if accepted
            else "stop_or_relocalize_q256_native_precise_softmax"
        ),
        "geometry": {
            "physical_k": q256.PHYSICAL_K,
            "query_rows": q256.QUERY_ROWS,
            "heads": q256.HEADS,
            "selected_width": q256.SELECTED_WIDTH,
        },
        "anchors": anchors,
        "score_timing": score_timing,
        "probability_timing": probability_timing,
        "incremental_softmax_ms": incremental_ms,
        "scratch_bytes": softmax_plan.scratch_bytes,
        "checks": checks,
        "probe_only": True,
        "runtime_changes": False,
    }
    _atomic_write(args.output, artifact)
    print(json.dumps({
        "output": str(args.output),
        "complete": True,
        "accepted": accepted,
        "decision": artifact["decision"],
        "incremental_softmax_ms": incremental_ms,
        "failed_gates": [name for name, passed in checks.items() if not passed],
    }))
    del inputs, score_plan, softmax_plan, candidate
    mx.clear_cache()
    gc.collect()
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
