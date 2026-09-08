#!/usr/bin/env python3
"""Qualify an exact single-encoder projected-K/QK union tile."""

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

from glm53_native_execution import NativeProjectedQKUnionTilePlan


DEFAULT_OUTPUT = ROOT / "bench-results" / (
    "m3ultra512-exact-fused-projected-qk-union-tile-20260908.json"
)
UNION_ROWS = 4096
QUERY_ROWS = 4
HEADS = 64
LATENT_DIM = 512
SELECTED_WIDTH = 2051
ATTENTION_SCALE = LATENT_DIM**-0.5
SAMPLES = 3
MAX_SCRATCH_BYTES = 1 << 30
MIN_STRUCTURAL_SPEEDUP = 0.90


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _bits(value: mx.array) -> np.ndarray:
    return np.asarray(value.view(mx.uint16), dtype=np.uint16)


def _fixture() -> dict[str, mx.array]:
    union_latent = mx.cos(
        mx.arange(UNION_ROWS * LATENT_DIM, dtype=mx.float32).reshape(
            UNION_ROWS, LATENT_DIM
        ) * 0.000071
    ).astype(mx.bfloat16)
    key_weight = mx.sin(
        mx.arange(HEADS * LATENT_DIM * LATENT_DIM, dtype=mx.float32).reshape(
            HEADS, LATENT_DIM, LATENT_DIM
        ) * 0.000013
    ).astype(mx.bfloat16)
    query = mx.sin(
        mx.arange(HEADS * QUERY_ROWS * LATENT_DIM, dtype=mx.float32).reshape(
            1, HEADS, QUERY_ROWS, LATENT_DIM
        ) * 0.00031
    ).astype(mx.bfloat16)
    rows = []
    for row in range(QUERY_ROWS):
        selected = mx.sort(
            (mx.arange(SELECTED_WIDTH, dtype=mx.int32) * 2039 + row * 97)
            % UNION_ROWS
        )
        rows.append(selected)
    slots = mx.contiguous(mx.stack(rows))
    valid = mx.ones((QUERY_ROWS, SELECTED_WIDTH), dtype=mx.bool_)
    arrays = (union_latent, key_weight, query, slots, valid)
    mx.eval(*arrays)
    mx.clear_cache()
    return {
        "union_latent": mx.contiguous(union_latent),
        "key_weight": mx.contiguous(key_weight),
        "query": mx.contiguous(query),
        "slots": slots,
        "valid": mx.contiguous(valid),
    }


def _reference(inputs: dict[str, mx.array]) -> dict[str, mx.array]:
    projected = inputs["union_latent"] @ inputs["key_weight"]
    selected = mx.stack([
        mx.take(projected, inputs["slots"][row], axis=1)
        for row in range(QUERY_ROWS)
    ])
    scale = mx.array(ATTENTION_SCALE, dtype=mx.bfloat16)
    scaled_query = (inputs["query"] * scale).astype(mx.bfloat16)
    score_rows = []
    for row in range(QUERY_ROWS):
        score = (
            scaled_query[:, :, row : row + 1, :]
            @ selected[row].swapaxes(-1, -2)
        ).reshape(HEADS, SELECTED_WIDTH)
        score_rows.append(score)
    scores = mx.stack(score_rows)
    scores = mx.where(
        inputs["valid"][:, None, :], scores,
        mx.finfo(mx.bfloat16).min,
    )
    mx.eval(projected, selected, scaled_query, scores)
    return {
        "projected": projected,
        "selected": selected,
        "scaled_query": scaled_query,
        "scores": scores,
    }


def _native(plan, inputs):
    output = plan.execute(
        inputs["union_latent"], inputs["key_weight"], inputs["query"],
        inputs["slots"], inputs["valid"], ATTENTION_SCALE,
    )
    mx.eval(output)
    return output


def _timed(operation) -> dict[str, object]:
    for _ in range(2):
        mx.eval(operation())
        mx.synchronize()
    samples = []
    for _ in range(SAMPLES):
        started = time.perf_counter_ns()
        mx.eval(operation())
        mx.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return {"median_wall_ms": statistics.median(samples), "samples_ms": samples}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps({"phase": "build_fixture", "union_rows": UNION_ROWS}), flush=True)
    inputs = _fixture()
    print(json.dumps({"phase": "reference"}), flush=True)
    reference = _reference(inputs)
    print(json.dumps({"phase": "native"}), flush=True)
    plan = NativeProjectedQKUnionTilePlan()
    identities = list(plan.buffer_identities)
    candidate = _native(plan, inputs)
    mx.synchronize()
    anchors = {
        "projected_union_key": np.array_equal(
            _bits(plan.debug_projected_union_key), _bits(reference["projected"])
        ),
        "final_scaled_query": np.array_equal(
            _bits(plan.debug_scaled_query),
            _bits(reference["scaled_query"][0, :, QUERY_ROWS - 1]),
        ),
        "qk_scores": np.array_equal(_bits(candidate), _bits(reference["scores"])),
    }
    print(json.dumps({"phase": "timing"}), flush=True)
    direct = _timed(lambda: _reference(inputs)["scores"])
    native = _timed(lambda: _native(plan, inputs))
    speedup = direct["median_wall_ms"] / native["median_wall_ms"]
    checks = {
        "all_projection_and_qk_anchors_byte_exact": all(anchors.values()),
        "single_encoder_fixed_execution_contract": (
            identities == list(plan.buffer_identities)
            and plan.dynamic_allocation_count == 0
            and plan.graph_node_count == 0
            and plan.shape_discovery_count == 0
            and plan.host_synchronization_count == 0
            and plan.returned_intermediate_tensor_bytes == 0
        ),
        "scratch_at_most_1gib": plan.scratch_bytes <= MAX_SCRATCH_BYTES,
        "structural_speedup_at_least_0_90x": speedup >= MIN_STRUCTURAL_SPEEDUP,
    }
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-exact-fused-projected-qk-union-tile-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "advance_exact_projected_qk_to_device_union_tile_loop"
            if accepted
            else "stop_or_relocalize_projected_qk_union_tile"
        ),
        "geometry": {
            "union_rows": UNION_ROWS,
            "query_rows": QUERY_ROWS,
            "selected_width": SELECTED_WIDTH,
            "heads": HEADS,
            "latent_dim": LATENT_DIM,
        },
        "anchors": anchors,
        "direct_timing": direct,
        "native_timing": native,
        "speedup": speedup,
        "scratch_bytes": plan.scratch_bytes,
        "materialized_selected_key_bytes": 0,
        "forbidden_materialized_selected_key_bytes": int(
            QUERY_ROWS * HEADS * SELECTED_WIDTH * LATENT_DIM * 2
        ),
        "execution_count": plan.execution_count,
        "checks": checks,
        "probe_only": True,
        "runtime_changes": False,
    }
    _atomic_write(args.output, artifact)
    print(json.dumps({
        "output": str(args.output), "complete": True,
        "accepted": accepted, "decision": artifact["decision"],
        "failed_gates": [name for name, passed in checks.items() if not passed],
    }))
    del plan, inputs, reference, candidate
    mx.clear_cache()
    gc.collect()
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
