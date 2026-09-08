#!/usr/bin/env python3
"""Compose four exact BM64 blocks over one shared multi-tile V projection."""

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
    "m3ultra512-exact-q256-shared-physical-value-pass-20260909.json"
)
PHYSICAL_K = 16384
TILE_ROWS = 8192
QUERY_ROWS = 256
QUERY_BLOCK_ROWS = 64
HEADS = 64
LATENT_DIM = 512
VALUE_DIM = 128
SELECTED_WIDTH = 2051
VALID_WIDTH = 2048
MIN_COMPOSITION_SPEEDUP = 1.20
MAX_SCRATCH = 256 << 20


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


def _execute(plan, probabilities, indices, valid, inputs) -> mx.array:
    output = plan.execute(
        probabilities, indices, valid, inputs["latent"], inputs["value_weight"]
    )
    mx.eval(output)
    mx.synchronize()
    return output


def _four_bm64_oracle(plans, inputs: dict[str, mx.array]) -> mx.array:
    blocks = []
    for block, plan in enumerate(plans):
        start = block * QUERY_BLOCK_ROWS
        stop = start + QUERY_BLOCK_ROWS
        probabilities = mx.contiguous(inputs["probabilities"][start:stop])
        indices = mx.contiguous(inputs["indices"][start:stop])
        valid = mx.contiguous(inputs["valid"][start:stop])
        mx.eval(probabilities, indices, valid)
        blocks.append(_execute(plan, probabilities, indices, valid, inputs))
    return mx.concatenate(blocks, axis=1)


def _q256(plan, inputs: dict[str, mx.array]) -> mx.array:
    return _execute(
        plan, inputs["probabilities"], inputs["indices"], inputs["valid"], inputs
    )


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

    print(json.dumps({"phase": "build_q256_fixture"}), flush=True)
    inputs = _fixture()
    oracle_plans = [
        NativeSharedPhysicalValueTilePlan(PHYSICAL_K, PHYSICAL_K, 64)
        for _ in range(4)
    ]
    candidate_plan = NativeSharedPhysicalValueTilePlan(
        PHYSICAL_K, TILE_ROWS, QUERY_ROWS
    )
    candidate_ids = list(candidate_plan.buffer_identities)
    print(json.dumps({"phase": "four_independent_bm64_oracle"}), flush=True)
    expected = _four_bm64_oracle(oracle_plans, inputs)
    mx.eval(expected)
    print(json.dumps({"phase": "shared_q256_value_pass"}), flush=True)
    actual = _q256(candidate_plan, inputs)
    repeat = _q256(candidate_plan, inputs)

    anchors = {
        "four_bm64_blocks": np.array_equal(_bits(actual), _bits(expected)),
        "repeat": np.array_equal(_bits(repeat), _bits(expected)),
    }
    print(json.dumps({"phase": "timing"}), flush=True)
    oracle_timing = _timed(lambda: _four_bm64_oracle(oracle_plans, inputs))
    candidate_timing = _timed(lambda: _q256(candidate_plan, inputs))
    speedup = oracle_timing["median_wall_ms"] / candidate_timing["median_wall_ms"]
    checks = {
        "q256_four_bm64_blocks_byte_exact": all(anchors.values()),
        "one_value_projection_shared_across_four_query_blocks": (
            candidate_plan.query_blocks == 4 and candidate_plan.tile_count == 2
        ),
        "fixed_owned_native_scope": (
            candidate_ids == list(candidate_plan.buffer_identities)
            and candidate_plan.dynamic_allocation_count == 0
            and candidate_plan.graph_node_count == 0
            and candidate_plan.shape_discovery_count == 0
            and candidate_plan.host_synchronization_count == 0
            and candidate_plan.returned_intermediate_tensor_bytes == 0
        ),
        "query_local_selected_value_materialization_zero": (
            candidate_plan.materialized_query_local_selected_value_bytes == 0
        ),
        "scratch_at_most_256mib": candidate_plan.scratch_bytes <= MAX_SCRATCH,
        "shared_projection_composition_speedup_at_least_1_20x": (
            speedup >= MIN_COMPOSITION_SPEEDUP
        ),
    }
    accepted = all(checks.values())
    artifact = {
        "schema": "glm53-exact-q256-shared-physical-value-pass-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "advance_exact_q256_shared_value_pass_to_320k_composed_prefill"
            if accepted
            else "stop_or_relocalize_q256_shared_physical_value_pass"
        ),
        "geometry": {
            "physical_k": PHYSICAL_K,
            "tile_rows": TILE_ROWS,
            "tile_count": candidate_plan.tile_count,
            "query_rows": QUERY_ROWS,
            "query_block_rows": QUERY_BLOCK_ROWS,
            "query_blocks": candidate_plan.query_blocks,
            "heads": HEADS,
        },
        "anchors": anchors,
        "four_independent_bm64_timing": oracle_timing,
        "shared_q256_timing": candidate_timing,
        "speedup": speedup,
        "scratch_bytes": candidate_plan.scratch_bytes,
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
    del oracle_plans, candidate_plan, inputs, expected, actual, repeat
    mx.clear_cache()
    gc.collect()
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
