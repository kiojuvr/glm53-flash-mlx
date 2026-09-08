#!/usr/bin/env python3
"""Test the exact sparse-prefill reordering needed by the native engine.

Direct prefill projects the complete latent cache to per-head K/V, constructs a
full sparse mask, and calls SDPA. A useful native architecture must instead
sort selected token positions, gather latent rows, project only those rows, and
run compact attention. This probe determines whether both reorderings preserve
the pinned MLX 0.32.2 result byte-for-byte before any production Metal plan is
built.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import date
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-sparse-prefill-reordering-equivalence-20260908.json"
)
CONTEXTS = (2_048, 32_768)
QUERY_ROWS = 4
HEADS = 64
LATENT_DIM = 512
VALUE_DIM = 128
SELECTED_WIDTH = 2051


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _fixture(context: int) -> dict[str, mx.array]:
    latent = mx.cos(
        mx.arange(context * LATENT_DIM, dtype=mx.float32).reshape(
            1, 1, context, LATENT_DIM
        )
        * 0.000071
    ).astype(mx.bfloat16)
    k_weight = mx.sin(
        mx.arange(HEADS * LATENT_DIM * LATENT_DIM, dtype=mx.float32).reshape(
            HEADS, LATENT_DIM, LATENT_DIM
        )
        * 0.000013
    ).astype(mx.bfloat16)
    v_weight = mx.cos(
        mx.arange(HEADS * VALUE_DIM * LATENT_DIM, dtype=mx.float32).reshape(
            HEADS, VALUE_DIM, LATENT_DIM
        )
        * 0.000017
    ).astype(mx.bfloat16)
    query = mx.sin(
        mx.arange(HEADS * QUERY_ROWS * LATENT_DIM, dtype=mx.float32).reshape(
            1, HEADS, QUERY_ROWS, LATENT_DIM
        )
        * 0.00031
    ).astype(mx.bfloat16)

    rows = []
    valid_rows = []
    for row in range(QUERY_ROWS):
        valid_width = min(context, SELECTED_WIDTH)
        if context <= SELECTED_WIDTH:
            selected = mx.arange(valid_width, dtype=mx.int32)
        else:
            # 8191 is coprime to the power-of-two test contexts. Sort into
            # physical token order so compact attention follows dense order.
            selected = mx.sort(
                (mx.arange(valid_width, dtype=mx.int32) * 8191 + row * 97)
                % context
            )
        if valid_width < SELECTED_WIDTH:
            selected = mx.concatenate(
                [
                    selected,
                    mx.full(
                        (SELECTED_WIDTH - valid_width,), -1, dtype=mx.int32
                    ),
                ]
            )
        rows.append(selected)
        valid_rows.append(selected >= 0)
    indices = mx.stack(rows)[None]
    valid = mx.stack(valid_rows)[None]
    mx.eval(latent, k_weight, v_weight, query, indices, valid)
    return {
        "latent": latent,
        "k_weight": k_weight,
        "v_weight": v_weight,
        "query": query,
        "indices": indices,
        "valid": valid,
    }


def _gather_per_query(array: mx.array, indices: mx.array, valid: mx.array):
    safe = mx.where(valid, indices, 0)
    rows = []
    for row in range(QUERY_ROWS):
        rows.append(mx.take(array, safe[0, row], axis=2))
    return mx.concatenate(rows, axis=0)


def _arms(fixture: dict[str, mx.array], context: int):
    latent = fixture["latent"]
    k_full = latent @ fixture["k_weight"]
    v_full = latent @ fixture["v_weight"].swapaxes(-1, -2)

    # Match the production prefill sentinel contract: invalid selections write
    # only to a temporary Kv slot which is removed after scatter. Mapping them
    # to row zero would let a later invalid write erase a valid token-zero bit.
    full_mask = mx.zeros((1, 1, QUERY_ROWS, context + 1), dtype=mx.bool_)
    safe = mx.where(fixture["valid"], fixture["indices"], context)
    full_mask = mx.put_along_axis(
        full_mask,
        safe[:, None],
        mx.array(True),
        axis=-1,
    )[..., :context]
    dense_output = mx.fast.scaled_dot_product_attention(
        fixture["query"], k_full, v_full,
        scale=LATENT_DIM**-0.5, mask=full_mask,
    )

    gathered_projected_k = _gather_per_query(
        k_full, fixture["indices"], fixture["valid"]
    )
    gathered_projected_v = _gather_per_query(
        v_full, fixture["indices"], fixture["valid"]
    )
    gathered_latent = _gather_per_query(
        latent, fixture["indices"], fixture["valid"]
    )
    projected_gathered_k = gathered_latent @ fixture["k_weight"]
    projected_gathered_v = (
        gathered_latent @ fixture["v_weight"].swapaxes(-1, -2)
    )
    compact_mask = fixture["valid"].reshape(
        QUERY_ROWS, 1, 1, SELECTED_WIDTH
    )
    compact_query = fixture["query"].transpose(2, 1, 0, 3)
    compact_after_full_projection = mx.fast.scaled_dot_product_attention(
        compact_query, gathered_projected_k, gathered_projected_v,
        scale=LATENT_DIM**-0.5, mask=compact_mask,
    ).transpose(2, 1, 0, 3)
    compact_after_latent_gather = mx.fast.scaled_dot_product_attention(
        compact_query, projected_gathered_k, projected_gathered_v,
        scale=LATENT_DIM**-0.5, mask=compact_mask,
    ).transpose(2, 1, 0, 3)
    return {
        "dense_output": dense_output,
        "gathered_projected_k": gathered_projected_k,
        "gathered_projected_v": gathered_projected_v,
        "projected_gathered_k": projected_gathered_k,
        "projected_gathered_v": projected_gathered_v,
        "compact_after_full_projection": compact_after_full_projection,
        "compact_after_latent_gather": compact_after_latent_gather,
    }


def _exact(left: mx.array, right: mx.array) -> bool:
    return bool(mx.array_equal(left, right).item())


def _diff(left: mx.array, right: mx.array) -> dict[str, object]:
    left_fp32 = left.astype(mx.float32)
    right_fp32 = right.astype(mx.float32)
    different = left != right
    return {
        "different_elements": int(mx.sum(different).item()),
        "max_absolute_difference": float(
            mx.max(mx.abs(left_fp32 - right_fp32)).item()
        ),
    }


def _context_case(context: int, samples: int) -> dict[str, object]:
    _progress("sparse_prefill_equivalence", context=context)
    fixture = _fixture(context)
    arms = _arms(fixture, context)
    mx.eval(*arms.values())
    checks = {
        "gather_after_full_k_equals_project_after_latent_gather": _exact(
            arms["gathered_projected_k"], arms["projected_gathered_k"]
        ),
        "gather_after_full_v_equals_project_after_latent_gather": _exact(
            arms["gathered_projected_v"], arms["projected_gathered_v"]
        ),
        "dense_mask_attention_equals_sorted_compact_attention": _exact(
            arms["dense_output"], arms["compact_after_full_projection"]
        ),
        "dense_mask_attention_equals_latent_gather_compact_attention": _exact(
            arms["dense_output"], arms["compact_after_latent_gather"]
        ),
    }
    diagnostics = {}
    comparisons = {
        "k_projection": (
            arms["gathered_projected_k"], arms["projected_gathered_k"]
        ),
        "v_projection": (
            arms["gathered_projected_v"], arms["projected_gathered_v"]
        ),
        "compact_attention": (
            arms["dense_output"], arms["compact_after_full_projection"]
        ),
        "end_to_end": (
            arms["dense_output"], arms["compact_after_latent_gather"]
        ),
    }
    for name, pair in comparisons.items():
        diagnostics[name] = _diff(*pair)

    timings = {"direct_dense_ms": [], "sparse_reordered_ms": []}
    for _ in range(samples):
        started = time.perf_counter_ns()
        direct = _arms(fixture, context)
        mx.eval(direct["dense_output"])
        timings["direct_dense_ms"].append(
            (time.perf_counter_ns() - started) / 1e6
        )
        started = time.perf_counter_ns()
        sparse = _arms(fixture, context)
        mx.eval(sparse["compact_after_latent_gather"])
        timings["sparse_reordered_ms"].append(
            (time.perf_counter_ns() - started) / 1e6
        )
    return {
        "context_tokens": context,
        "query_rows": QUERY_ROWS,
        "selected_width": SELECTED_WIDTH,
        "checks": checks,
        "all_exact": all(checks.values()),
        "diagnostics": diagnostics,
        "timing_is_architecture_headroom_only": True,
        "median_direct_dense_ms": statistics.median(timings["direct_dense_ms"]),
        "median_sparse_reordered_ms": statistics.median(
            timings["sparse_reordered_ms"]
        ),
        "samples": timings,
    }


def _artifact(contexts: dict[str, object]) -> dict[str, object]:
    rows = list(contexts.values())
    checks = {
        "2k_and_32k_measured": set(map(int, contexts)) == set(CONTEXTS),
        "selected_k_projection_reordering_byte_exact": (
            len(rows) == len(CONTEXTS)
            and all(
                row["checks"][
                    "gather_after_full_k_equals_project_after_latent_gather"
                ]
                for row in rows
            )
        ),
        "selected_v_projection_reordering_byte_exact": (
            len(rows) == len(CONTEXTS)
            and all(
                row["checks"][
                    "gather_after_full_v_equals_project_after_latent_gather"
                ]
                for row in rows
            )
        ),
        "sorted_compact_attention_matches_dense_sparse_mask": (
            len(rows) == len(CONTEXTS)
            and all(
                row["checks"][
                    "dense_mask_attention_equals_sorted_compact_attention"
                ]
                for row in rows
            )
        ),
        "end_to_end_reordered_prefill_dsa_byte_exact": (
            len(rows) == len(CONTEXTS)
            and all(row["all_exact"] for row in rows)
        ),
    }
    accepted = all(checks.values())
    return {
        "schema": "glm53-sparse-prefill-reordering-equivalence-v1",
        "date": date.today().isoformat(),
        "complete": len(contexts) == len(CONTEXTS),
        "accepted": accepted,
        "decision": (
            "implement_sorted_gather_project_compact_attention_native_region"
            if accepted
            else "preserve_direct_prefill_order_and_redesign_native_dsa"
        ),
        "contexts": contexts,
        "checks": checks,
        "promotion": {
            "runtime": False,
            "production_prefill": False,
            "requires_real_checkpoint_confirmation": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args(argv)
    contexts = {}
    for context in CONTEXTS:
        contexts[str(context)] = _context_case(context, args.samples)
        artifact = _artifact(contexts)
        _atomic_write(args.output, artifact)
        if not contexts[str(context)]["all_exact"]:
            break
    print(json.dumps({
        "output": str(args.output), "complete": artifact["complete"],
        "accepted": artifact["accepted"], "decision": artifact["decision"]
    }))
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
