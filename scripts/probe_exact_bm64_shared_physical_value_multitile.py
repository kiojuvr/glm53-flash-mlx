#!/usr/bin/env python3
"""Prove exact FP32 BM64 accumulator continuation across physical V tiles."""

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

from glm53_native_execution import NativeSharedPhysicalValueTilePlan


DEFAULT_OUTPUT = ROOT / "bench-results" / (
    "m3ultra512-exact-bm64-shared-physical-value-multitile-20260909.json"
)
PHYSICAL_K = 16384
TILE_ROWS = 8192
QUERY_ROWS = 64
HEADS = 64
LATENT_DIM = 512
VALUE_DIM = 128
SELECTED_WIDTH = 2051
VALID_WIDTH = 2048


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _bits(value: mx.array) -> np.ndarray:
    return np.asarray(value.view(mx.uint16), dtype=np.uint16)


def _fixture() -> dict[str, mx.array]:
    host_indices = np.full((QUERY_ROWS, SELECTED_WIDTH), -1, dtype=np.int32)
    host_valid = np.zeros((QUERY_ROWS, SELECTED_WIDTH), dtype=np.bool_)
    base = np.arange(VALID_WIDTH, dtype=np.int64)
    for row in range(QUERY_ROWS):
        selected = (base * 8 + row % 8) % PHYSICAL_K
        selected.sort()
        host_indices[row, :VALID_WIDTH] = selected.astype(np.int32)
        host_valid[row, :VALID_WIDTH] = True
    scores = mx.sin(
        mx.arange(
            QUERY_ROWS * HEADS * SELECTED_WIDTH, dtype=mx.float32
        ).reshape(QUERY_ROWS, HEADS, SELECTED_WIDTH)
        * 0.00037
    )
    result = {
        "probabilities": mx.contiguous(
            mx.softmax(scores, axis=-1, precise=True).astype(mx.bfloat16)
        ),
        "indices": mx.contiguous(mx.array(host_indices)),
        "valid": mx.contiguous(mx.array(host_valid)),
        "latent": mx.contiguous(
            mx.cos(
                mx.arange(PHYSICAL_K * LATENT_DIM, dtype=mx.float32).reshape(
                    PHYSICAL_K, LATENT_DIM
                )
                * 0.000071
            ).astype(mx.bfloat16)
        ),
        "value_weight": mx.contiguous(
            mx.sin(
                mx.arange(
                    HEADS * VALUE_DIM * LATENT_DIM, dtype=mx.float32
                ).reshape(HEADS, VALUE_DIM, LATENT_DIM)
                * 0.000019
            ).astype(mx.bfloat16)
        ),
    }
    mx.eval(*result.values())
    return result


def _native(plan, inputs: dict[str, mx.array]) -> mx.array:
    output = plan.execute(
        inputs["probabilities"], inputs["indices"], inputs["valid"],
        inputs["latent"], inputs["value_weight"],
    )
    mx.eval(output)
    mx.synchronize()
    return output


def _timed(operation, samples: int = 3) -> dict[str, object]:
    for _ in range(2):
        mx.eval(operation())
        mx.synchronize()
    values = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        mx.eval(operation())
        mx.synchronize()
        values.append((time.perf_counter_ns() - started) / 1e6)
    return {"median_wall_ms": statistics.median(values), "samples_ms": values}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    print(json.dumps({"phase": "build_two_tile_fixture"}), flush=True)
    inputs = _fixture()
    single = NativeSharedPhysicalValueTilePlan(PHYSICAL_K, PHYSICAL_K)
    tiled = NativeSharedPhysicalValueTilePlan(PHYSICAL_K, TILE_ROWS)
    single_ids = list(single.buffer_identities)
    tiled_ids = list(tiled.buffer_identities)
    print(json.dumps({"phase": "single_tile_oracle"}), flush=True)
    expected = _native(single, inputs)
    print(json.dumps({"phase": "two_tile_fragment_continuation"}), flush=True)
    actual = _native(tiled, inputs)
    repeat = _native(tiled, inputs)

    anchors = {
        "two_tile_output_matches_single_continuous_bm64": np.array_equal(
            _bits(actual), _bits(expected)
        ),
        "two_tile_repeat": np.array_equal(_bits(repeat), _bits(expected)),
    }
    print(json.dumps({"phase": "timing"}), flush=True)
    single_timing = _timed(lambda: _native(single, inputs))
    tiled_timing = _timed(lambda: _native(tiled, inputs))
    overhead = tiled_timing["median_wall_ms"] / single_timing["median_wall_ms"]
    checks = {
        "fp32_mma_fragment_store_reload_byte_exact": all(anchors.values()),
        "fixed_two_tile_scope": (
            single_ids == list(single.buffer_identities)
            and tiled_ids == list(tiled.buffer_identities)
            and tiled.tile_count == 2
            and tiled.dynamic_allocation_count == 0
            and tiled.graph_node_count == 0
            and tiled.shape_discovery_count == 0
            and tiled.host_synchronization_count == 0
        ),
        "query_local_selected_value_materialization_zero": (
            tiled.materialized_query_local_selected_value_bytes == 0
        ),
        "two_tile_wall_overhead_below_25_percent": overhead <= 1.25,
    }
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-exact-bm64-shared-physical-value-multitile-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "advance_shared_physical_value_pass_to_q256_four_blocks"
            if accepted
            else "stop_or_relocalize_fp32_mma_fragment_continuation"
        ),
        "geometry": {
            "physical_k": PHYSICAL_K,
            "tile_rows": TILE_ROWS,
            "tile_count": tiled.tile_count,
            "query_rows": QUERY_ROWS,
            "heads": HEADS,
        },
        "anchors": anchors,
        "single_tile_timing": single_timing,
        "two_tile_timing": tiled_timing,
        "two_tile_overhead_ratio": overhead,
        "two_tile_scratch_bytes": tiled.scratch_bytes,
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
        "failed_gates": [name for name, passed in checks.items() if not passed],
    }))
    del single, tiled, inputs, expected, actual, repeat
    mx.clear_cache()
    gc.collect()
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
