#!/usr/bin/env python3
"""Test a bounded virtual-physical-BK16 representation for exact prefill AV.

Direct and compact AV use the same pinned MLX Steel BM64/BN64/BK16 pipeline.
The compact path differs because densely packing 2,051 selected tokens changes
their BK16 lane grouping.  This probe preserves each selected token's original
``physical_position % 16`` lane, keeps occupied physical blocks ordered, and
removes only wholly empty physical blocks.  Multiplication by those removed
zero blocks cannot change the FP32 accumulator.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import localize_sparse_prefill_attention_reduction as localization
import probe_sparse_prefill_reordering_equivalence as equivalence


DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-virtual-physical-bk16-sparse-prefill-av-20260908.json"
)
CONTEXTS = (2_048, 32_768)
BK = 16


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _row_case(
    row: int,
    indices: mx.array,
    valid: mx.array,
    probabilities: mx.array,
    values: mx.array,
    direct: mx.array,
) -> dict[str, object]:
    valid_np = np.asarray(valid[0, row], dtype=np.bool_)
    physical = np.asarray(indices[0, row], dtype=np.int32)[valid_np]
    blocks = np.unique(physical // BK)
    ranks = np.searchsorted(blocks, physical // BK)
    packed_positions_np = ranks * BK + physical % BK
    packed_k = int(len(blocks) * BK)
    fixed_capacity_k = min(
        int(probabilities.shape[-1]), equivalence.SELECTED_WIDTH * BK
    )
    packed_positions = mx.array(packed_positions_np, dtype=mx.int32)

    selected_probabilities = localization._gather_query_axis(
        probabilities, mx.where(valid, indices, probabilities.shape[-1])
    )[row : row + 1, :, :, : len(physical)]
    selected_values = equivalence._gather_per_query(
        values, indices, valid
    )[row : row + 1, :, : len(physical), :]
    packed_probabilities = mx.zeros(
        (1, equivalence.HEADS, 1, packed_k), dtype=mx.bfloat16
    ).at[:, :, :, packed_positions].add(selected_probabilities)
    packed_values = mx.zeros(
        (1, equivalence.HEADS, packed_k, equivalence.VALUE_DIM),
        dtype=mx.bfloat16,
    ).at[:, :, packed_positions, :].add(selected_values)
    candidate = packed_probabilities @ packed_values
    fixed_probabilities = mx.pad(
        packed_probabilities,
        [(0, 0), (0, 0), (0, 0), (0, fixed_capacity_k - packed_k)],
    )
    fixed_values = mx.pad(
        packed_values,
        [(0, 0), (0, 0), (0, fixed_capacity_k - packed_k), (0, 0)],
    )
    fixed_candidate = fixed_probabilities @ fixed_values
    reference = direct[:, :, row : row + 1, :]
    mx.eval(
        candidate, fixed_candidate, reference,
        packed_probabilities, packed_values,
    )
    different = int(mx.sum(candidate != reference).item())
    fixed_different = int(mx.sum(fixed_candidate != reference).item())
    row_result = {
        "row": row,
        "selected_tokens": int(len(physical)),
        "occupied_physical_bk16_blocks": int(len(blocks)),
        "packed_k": packed_k,
        "fixed_capacity_k": fixed_capacity_k,
        "full_physical_k": int(probabilities.shape[-1]),
        "removed_empty_bk16_blocks": (
            int(probabilities.shape[-1] // BK) - int(len(blocks))
        ),
        "physical_lane_preserved": bool(
            np.array_equal(packed_positions_np % BK, physical % BK)
        ),
        "physical_block_order_preserved": bool(np.all(np.diff(blocks) > 0)),
        "different_output_elements": different,
        "byte_exact": different == 0,
        "fixed_capacity_different_output_elements": fixed_different,
        "fixed_capacity_byte_exact": fixed_different == 0,
    }
    del (
        candidate,
        fixed_candidate,
        reference,
        packed_probabilities,
        packed_values,
        fixed_probabilities,
        fixed_values,
    )
    mx.clear_cache()
    gc.collect()
    return row_result


def _context_case(context: int) -> dict[str, object]:
    print(json.dumps({"phase": "build_virtual_bk16_fixture", "context": context}), flush=True)
    fixture = equivalence._fixture(context)
    latent = fixture["latent"]
    key = latent @ fixture["k_weight"]
    value = latent @ fixture["v_weight"].swapaxes(-1, -2)
    safe = mx.where(fixture["valid"], fixture["indices"], context)
    mask = mx.zeros(
        (1, 1, equivalence.QUERY_ROWS, context + 1), dtype=mx.bool_
    )
    mask = mx.put_along_axis(mask, safe[:, None], mx.array(True), axis=-1)[
        ..., :context
    ]
    scale = mx.array(equivalence.LATENT_DIM**-0.5, dtype=mx.bfloat16)
    scores = (fixture["query"] * scale).astype(mx.bfloat16) @ key.swapaxes(-1, -2)
    scores = mx.where(mask, scores, mx.finfo(mx.bfloat16).min)
    probabilities = mx.softmax(scores, axis=-1, precise=True)
    direct = probabilities @ value
    mx.eval(probabilities, value, direct)
    rows = []
    for row in range(equivalence.QUERY_ROWS):
        print(json.dumps({"phase": "virtual_bk16_row", "context": context, "row": row}), flush=True)
        rows.append(
            _row_case(
                row,
                fixture["indices"],
                fixture["valid"],
                probabilities,
                value,
                direct,
            )
        )
    return {
        "context_tokens": context,
        "rows": rows,
        "all_rows_byte_exact": all(
            item["byte_exact"] and item["fixed_capacity_byte_exact"]
            for item in rows
        ),
        "max_packed_k": max(item["packed_k"] for item in rows),
        "max_packed_value_scratch_bytes": max(
            equivalence.HEADS
            * item["packed_k"]
            * equivalence.VALUE_DIM
            * 2
            for item in rows
        ),
    }


def _artifact(contexts: dict[str, object]) -> dict[str, object]:
    complete = set(map(int, contexts)) == set(CONTEXTS)
    checks = {
        "2k_and_32k_measured": complete,
        "all_query_rows_byte_exact": complete and all(
            row["all_rows_byte_exact"] for row in contexts.values()
        ),
        "physical_bk16_lane_preserved": complete and all(
            item["physical_lane_preserved"]
            for row in contexts.values()
            for item in row["rows"]
        ),
        "physical_bk16_block_order_preserved": complete and all(
            item["physical_block_order_preserved"]
            for row in contexts.values()
            for item in row["rows"]
        ),
        "packed_k_bounded_by_selected_width_times_bk": complete and all(
            row["max_packed_k"] <= equivalence.SELECTED_WIDTH * BK
            for row in contexts.values()
        ),
        "fixed_capacity_trailing_zero_blocks_byte_exact": complete and all(
            item["fixed_capacity_byte_exact"]
            for row in contexts.values()
            for item in row["rows"]
        ),
    }
    accepted = all(checks.values())
    return {
        "schema": "glm53-virtual-physical-bk16-sparse-prefill-av-v1",
        "date": date.today().isoformat(),
        "complete": complete,
        "accepted": accepted,
        "decision": (
            "implement_native_virtual_physical_bk16_av"
            if accepted
            else "capture_deeper_steel_av_reduction_geometry"
        ),
        "bk": BK,
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
        if not contexts[str(context)]["all_rows_byte_exact"]:
            break
    print(json.dumps({
        "output": str(args.output),
        "complete": artifact["complete"],
        "accepted": artifact["accepted"],
        "decision": artifact["decision"],
    }))
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
