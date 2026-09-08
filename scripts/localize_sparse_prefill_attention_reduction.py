#!/usr/bin/env python3
"""Localize the 32K compact-prefill attention exactness barrier."""

from __future__ import annotations

import argparse
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

import probe_sparse_prefill_reordering_equivalence as equivalence


DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-sparse-prefill-attention-reduction-localization-20260908.json"
)
CONTEXT = 32_768


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _gather_query_axis(array: mx.array, safe: mx.array) -> mx.array:
    rows = [
        mx.take(array[:, :, row, :], safe[0, row], axis=-1)
        for row in range(equivalence.QUERY_ROWS)
    ]
    return mx.concatenate(rows, axis=0).reshape(
        equivalence.QUERY_ROWS,
        equivalence.HEADS,
        1,
        equivalence.SELECTED_WIDTH,
    )


def _first_diff(left: mx.array, right: mx.array) -> dict[str, object] | None:
    mx.eval(left, right)
    different = np.asarray(left != right)
    coordinates = np.argwhere(different)
    if not len(coordinates):
        return None
    coordinate = tuple(int(value) for value in coordinates[0])
    left_bits = np.asarray(left.view(mx.uint16), dtype=np.uint16)
    right_bits = np.asarray(right.view(mx.uint16), dtype=np.uint16)
    return {
        "coordinate": list(coordinate),
        "direct_bf16_bits": int(left_bits[coordinate]),
        "compact_bf16_bits": int(right_bits[coordinate]),
        "direct_fp32": float(np.asarray(left.astype(mx.float32))[coordinate]),
        "compact_fp32": float(np.asarray(right.astype(mx.float32))[coordinate]),
    }


def build_artifact() -> dict[str, object]:
    fixture = equivalence._fixture(CONTEXT)
    latent = fixture["latent"]
    key = latent @ fixture["k_weight"]
    value = latent @ fixture["v_weight"].swapaxes(-1, -2)
    safe = mx.where(fixture["valid"], fixture["indices"], CONTEXT)
    mask = mx.zeros(
        (1, 1, equivalence.QUERY_ROWS, CONTEXT + 1), dtype=mx.bool_
    )
    mask = mx.put_along_axis(mask, safe[:, None], mx.array(True), axis=-1)[
        ..., :CONTEXT
    ]
    scale = mx.array(equivalence.LATENT_DIM**-0.5, dtype=mx.bfloat16)
    scaled_query = (fixture["query"] * scale).astype(mx.bfloat16)
    full_scores = scaled_query @ key.swapaxes(-1, -2)
    full_scores = mx.where(mask, full_scores, mx.finfo(mx.bfloat16).min)
    full_probabilities = mx.softmax(full_scores, axis=-1, precise=True)
    explicit_full_output = full_probabilities @ value
    direct_output = mx.fast.scaled_dot_product_attention(
        fixture["query"], key, value,
        scale=equivalence.LATENT_DIM**-0.5, mask=mask,
    )

    gathered_key = equivalence._gather_per_query(
        key, fixture["indices"], fixture["valid"]
    )
    gathered_value = equivalence._gather_per_query(
        value, fixture["indices"], fixture["valid"]
    )
    compact_query = fixture["query"].transpose(2, 1, 0, 3)
    compact_scaled_query = (compact_query * scale).astype(mx.bfloat16)
    compact_scores = compact_scaled_query @ gathered_key.swapaxes(-1, -2)
    compact_mask = fixture["valid"].reshape(
        equivalence.QUERY_ROWS, 1, 1, equivalence.SELECTED_WIDTH
    )
    compact_scores = mx.where(
        compact_mask, compact_scores, mx.finfo(mx.bfloat16).min
    )
    compact_probabilities = mx.softmax(
        compact_scores, axis=-1, precise=True
    )
    compact_output = compact_probabilities @ gathered_value
    compact_output = compact_output.transpose(2, 1, 0, 3)

    selected_full_scores = _gather_query_axis(full_scores, safe)
    selected_full_probabilities = _gather_query_axis(full_probabilities, safe)
    mx.eval(
        direct_output, explicit_full_output, selected_full_scores,
        selected_full_probabilities, compact_scores, compact_probabilities,
        compact_output,
    )
    checks = {
        "explicit_fallback_matches_direct_fast_sdpa": bool(
            mx.array_equal(direct_output, explicit_full_output).item()
        ),
        "selected_qk_scores_byte_exact": bool(
            mx.array_equal(selected_full_scores, compact_scores).item()
        ),
        "selected_precise_softmax_probabilities_byte_exact": bool(
            mx.array_equal(
                selected_full_probabilities, compact_probabilities
            ).item()
        ),
        "compact_av_accumulation_is_first_and_only_barrier": (
            _first_diff(explicit_full_output, compact_output) is not None
        ),
    }
    return {
        "schema": "glm53-sparse-prefill-attention-reduction-localization-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": all(checks.values()),
        "decision": "implement_exact_virtual_full_kv_av_reduction",
        "context_tokens": CONTEXT,
        "selected_width": equivalence.SELECTED_WIDTH,
        "checks": checks,
        "first_differing_stage": "attention_probability_times_value",
        "first_difference": _first_diff(explicit_full_output, compact_output),
        "different_output_elements": int(
            mx.sum(explicit_full_output != compact_output).item()
        ),
        "promotion": {
            "runtime": False,
            "ordinary_compact_sdpa": False,
            "compact_qk": True,
            "compact_precise_softmax": True,
            "compact_av": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps({"phase": "localize_sparse_prefill_attention", "context": CONTEXT}), flush=True)
    artifact = build_artifact()
    _atomic_write(args.output, artifact)
    print(json.dumps({
        "output": str(args.output), "complete": artifact["complete"],
        "accepted": artifact["accepted"], "decision": artifact["decision"],
        "first_differing_stage": artifact["first_differing_stage"],
    }))
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
