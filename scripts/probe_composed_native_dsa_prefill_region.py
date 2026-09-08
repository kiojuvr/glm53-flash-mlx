#!/usr/bin/env python3
"""Qualify one native Q4 DSA prefill region from score through attention."""

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


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
NATIVE_PACKAGE = ROOT / "native_execution"
for path in (SCRIPTS, NATIVE_PACKAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import probe_native_execution_engine_feasibility as tier0
import probe_sparse_prefill_reordering_equivalence as equivalence
from glm53_flash_mlx.indexpool import INDEXPOOL_SENTINEL, expand_selected_pools
from glm53_native_execution import NativeSelectedKVAttentionSelectionPlan


DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-composed-native-dsa-prefill-region-20260908.json"
)
CONTEXTS = (2_048, 32_768)
INDEX_HEADS = 32
INDEX_DIM = 128
SELECTED_POOLS = 512
INDEX_SCALE = INDEX_DIM**-0.5
ATTENTION_SCALE = equivalence.LATENT_DIM**-0.5
SAMPLES = 3
MAX_SCRATCH_BYTES = 224 << 20


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _inputs(context: int) -> dict[str, mx.array | int]:
    attention = equivalence._fixture(context)
    pools = (context + 3) // 4
    index_query = mx.sin(
        mx.arange(
            equivalence.QUERY_ROWS * INDEX_HEADS * INDEX_DIM,
            dtype=mx.float32,
        ).reshape(1, equivalence.QUERY_ROWS, INDEX_HEADS, INDEX_DIM)
        * 0.00029
    ).astype(mx.bfloat16)
    mixture_weights = mx.cos(
        mx.arange(
            equivalence.QUERY_ROWS * INDEX_HEADS, dtype=mx.float32
        ).reshape(1, equivalence.QUERY_ROWS, INDEX_HEADS)
        * 0.019
    ).astype(mx.bfloat16)
    pool_keys = mx.cos(
        mx.arange(pools * INDEX_DIM, dtype=mx.float32).reshape(
            1, pools, INDEX_DIM
        )
        * 0.00011
    ).astype(mx.bfloat16)
    pool_indices = mx.arange(
        pools * 4, dtype=mx.int64
    ).reshape(1, pools, 4)
    pool_valid = mx.ones((1, pools), dtype=mx.bool_)
    raw_positions = mx.zeros((1, 4), dtype=mx.int64)
    raw_valid = mx.zeros((1, 4), dtype=mx.bool_)
    current_valid = mx.ones((equivalence.QUERY_ROWS,), dtype=mx.bool_)
    latent = mx.contiguous(attention["latent"].reshape(context, -1))
    arrays = (
        index_query,
        mixture_weights,
        pool_keys,
        pool_indices,
        pool_valid,
        raw_positions,
        raw_valid,
        current_valid,
        latent,
        attention["k_weight"],
        attention["v_weight"],
        attention["query"],
    )
    mx.eval(*arrays)
    result = {
        "context": context,
        "pools": pools,
        "index_query": mx.contiguous(index_query),
        "mixture_weights": mx.contiguous(mixture_weights),
        "pool_keys": mx.contiguous(pool_keys),
        "pool_indices": mx.contiguous(pool_indices),
        "pool_valid": mx.contiguous(pool_valid),
        "raw_positions": mx.contiguous(raw_positions),
        "raw_valid": mx.contiguous(raw_valid),
        "current_valid": mx.contiguous(current_valid),
        "latent": latent,
        "key_weight": mx.contiguous(attention["k_weight"]),
        "value_weight": mx.contiguous(attention["v_weight"]),
        "attention_query": mx.contiguous(attention["query"]),
    }
    mx.eval(*(value for value in result.values() if isinstance(value, mx.array)))
    return result


def _selection(inputs: dict[str, object]) -> tuple[mx.array, mx.array]:
    rows = equivalence.QUERY_ROWS
    pools = int(inputs["pools"])
    valid_candidates = mx.broadcast_to(
        inputs["pool_valid"][:, None], (1, rows, pools)
    )
    scores = tier0._score_expression(
        inputs["index_query"], inputs["pool_keys"],
        inputs["mixture_weights"], valid_candidates, INDEX_SCALE,
    )
    selected = mx.argsort(-scores, axis=-1)[..., :SELECTED_POOLS]
    selected_valid = mx.take_along_axis(
        valid_candidates, selected, axis=-1
    )
    indices, valid = expand_selected_pools(
        selected,
        inputs["pool_indices"],
        selected_valid,
        kv_len=int(inputs["context"]),
        index_topk=2_048,
        index_kpool=4,
        tail_positions=mx.zeros((1, 0), dtype=mx.int64),
        tail_valid=mx.zeros((1, 0), dtype=mx.bool_),
        always_select_tail=True,
    )
    valid = valid & inputs["current_valid"][..., None]
    return mx.where(valid, indices, INDEXPOOL_SENTINEL), valid


def _direct(inputs: dict[str, object]) -> mx.array:
    context = int(inputs["context"])
    indices, valid = _selection(inputs)
    key = (inputs["latent"] @ inputs["key_weight"])[None]
    value = (
        inputs["latent"] @ inputs["value_weight"].swapaxes(-1, -2)
    )[None]
    safe = mx.where(valid, indices, context)
    mask = mx.zeros(
        (1, 1, equivalence.QUERY_ROWS, context + 1), dtype=mx.bool_
    )
    mask = mx.put_along_axis(mask, safe[:, None], mx.array(True), axis=-1)[
        ..., :context
    ]
    return mx.fast.scaled_dot_product_attention(
        inputs["attention_query"], key, value,
        scale=ATTENTION_SCALE, mask=mask,
    )


def _native(plan, inputs: dict[str, object]) -> mx.array:
    output = plan.execute(
        inputs["index_query"], inputs["mixture_weights"],
        inputs["pool_keys"], inputs["pool_indices"], inputs["pool_valid"],
        inputs["raw_positions"], inputs["raw_valid"],
        inputs["current_valid"], inputs["latent"], inputs["key_weight"],
        inputs["value_weight"], inputs["attention_query"],
        int(inputs["pools"]), int(inputs["context"]), 0,
    )
    return output.transpose(2, 1, 0, 3)


def _different(left: mx.array, right: mx.array) -> int:
    mx.eval(left, right)
    return int(mx.sum(left != right).item())


def _timed(operation) -> dict[str, object]:
    samples = []
    for _ in range(SAMPLES):
        started = time.perf_counter_ns()
        output = operation()
        mx.eval(output)
        mx.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return {"median_wall_ms": statistics.median(samples), "samples_ms": samples}


def _context_case(context: int) -> dict[str, object]:
    print(json.dumps({"phase": "composed_native_dsa", "context": context}), flush=True)
    inputs = _inputs(context)
    plan = NativeSelectedKVAttentionSelectionPlan(
        int(inputs["pools"]), context, INDEX_SCALE, ATTENTION_SCALE
    )
    identities = list(plan.buffer_identities)
    score_order_indices, score_order_valid = _selection(inputs)
    sort_key = mx.where(score_order_valid, score_order_indices, context)
    physical_order = mx.argsort(sort_key, axis=-1)
    reference_indices = mx.take_along_axis(
        score_order_indices, physical_order, axis=-1
    )
    reference_valid = mx.take_along_axis(
        score_order_valid, physical_order, axis=-1
    )
    reference_latent = equivalence._gather_per_query(
        inputs["latent"].reshape(1, 1, context, equivalence.LATENT_DIM),
        reference_indices.astype(mx.int32), reference_valid,
    ).reshape(
        equivalence.QUERY_ROWS,
        equivalence.SELECTED_WIDTH,
        equivalence.LATENT_DIM,
    )
    reference_latent = mx.where(
        reference_valid.reshape(
            equivalence.QUERY_ROWS, equivalence.SELECTED_WIDTH, 1
        ),
        reference_latent,
        mx.array(0, dtype=mx.bfloat16),
    )
    reference = _direct(inputs)
    candidate = _native(plan, inputs)
    mx.eval(
        reference_indices, reference_valid, reference_latent,
        reference, candidate,
    )
    anchors = {
        "score_order_indices": _different(
            score_order_indices.astype(mx.int32),
            plan.debug_score_order_indices,
        ),
        "selected_indices": _different(
            reference_indices.astype(mx.int32), plan.debug_selected_indices
        ),
        "selected_valid": _different(
            reference_valid, plan.debug_selected_valid
        ),
        "selected_latent": _different(
            reference_latent, plan.debug_selected_latent
        ),
    }
    output_different = _different(reference, candidate)
    mx.eval(_native(plan, inputs))
    direct_timing = _timed(lambda: _direct(inputs))
    native_timing = _timed(lambda: _native(plan, inputs))
    result = {
        "context_tokens": context,
        "query_rows": equivalence.QUERY_ROWS,
        "pool_rows": int(inputs["pools"]),
        "anchor_different_elements": anchors,
        "all_anchors_byte_exact": all(value == 0 for value in anchors.values()),
        "different_output_elements": output_different,
        "byte_exact": output_different == 0,
        "direct_timing": direct_timing,
        "native_timing": native_timing,
        "speedup": direct_timing["median_wall_ms"] / native_timing["median_wall_ms"],
        "scratch_bytes": plan.scratch_bytes,
        "buffer_identities_stable": identities == list(plan.buffer_identities),
        "execution_count": plan.execution_count,
        "dynamic_allocation_count": plan.dynamic_allocation_count,
        "graph_node_count": plan.graph_node_count,
        "shape_discovery_count": plan.shape_discovery_count,
        "host_synchronization_count": plan.host_synchronization_count,
        "returned_intermediate_tensor_bytes": plan.returned_intermediate_tensor_bytes,
    }
    del (
        plan, inputs, score_order_indices, score_order_valid, sort_key,
        physical_order, reference_indices, reference_valid,
        reference_latent, reference, candidate,
    )
    mx.clear_cache()
    gc.collect()
    return result


def _artifact(contexts: dict[str, object]) -> dict[str, object]:
    complete = set(map(int, contexts)) == set(CONTEXTS)
    short = contexts.get("2048", {})
    long = contexts.get("32768", {})
    checks = {
        "2k_and_32k_measured": complete,
        "selection_gather_and_attention_byte_exact": complete and all(
            row["all_anchors_byte_exact"] and row["byte_exact"]
            for row in contexts.values()
        ),
        "scratch_at_most_224mib": complete and all(
            row["scratch_bytes"] <= MAX_SCRATCH_BYTES
            for row in contexts.values()
        ),
        "no_dynamic_allocation_graph_shape_sync_or_intermediate_return": (
            complete and all(
                row["dynamic_allocation_count"] == 0
                and row["graph_node_count"] == 0
                and row["shape_discovery_count"] == 0
                and row["host_synchronization_count"] == 0
                and row["returned_intermediate_tensor_bytes"] == 0
                for row in contexts.values()
            )
        ),
        "all_buffer_addresses_stable": complete and all(
            row["buffer_identities_stable"] for row in contexts.values()
        ),
        "32k_composed_region_speedup_at_least_2x": (
            complete and long["speedup"] >= 2.0
        ),
        "2k_requires_direct_crossover": (
            complete and short["speedup"] < 1.0
        ),
    }
    accepted = all(checks.values())
    return {
        "schema": "glm53-composed-native-dsa-prefill-region-v1",
        "date": date.today().isoformat(),
        "complete": complete,
        "accepted": accepted,
        "decision": (
            "advance_composed_dsa_region_into_native_prefill_layer_plan"
            if accepted
            else "stop_or_relocalize_composed_native_dsa_prefill_region"
        ),
        "contexts": contexts,
        "checks": checks,
        "runtime_changes": False,
        "production_admission_changed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    contexts = {}
    for context in CONTEXTS:
        contexts[str(context)] = _context_case(context)
        artifact = _artifact(contexts)
        _atomic_write(args.output, artifact)
        if not (
            contexts[str(context)]["all_anchors_byte_exact"]
            and contexts[str(context)]["byte_exact"]
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
