#!/usr/bin/env python3
"""Measure the row-contiguous Metal input ABI on the official M3 Ultra runtime."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import time
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np

import glm53_flash_mlx.fp8 as fp8
import glm53_flash_mlx.grouped_fp8 as grouped_fp8
import glm53_flash_mlx.packed as packed
from glm53_flash_mlx.abi import KERNEL_ABI_VERSION, MLX_VLM_REVISION
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint

WARMUPS = 3
SAMPLES = 11
MAX_REGRESSION = 0.02
MAX_WORKING_PEAK_INCREASE = 64 * 2**20


def _hash_logits(logits) -> str:
    mx.eval(logits)
    values = np.ascontiguousarray(np.asarray(logits.astype(mx.float32)))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _set_contract(enabled: bool, originals) -> None:
    replacement = originals[0] if enabled else (lambda value: value)
    fp8._metal_input = replacement
    packed._metal_input = originals[1] if enabled else replacement
    grouped_fp8._metal_input = originals[2] if enabled else replacement


def _run_one(model) -> dict:
    gc.collect()
    mx.synchronize()
    active_before = int(mx.get_active_memory())
    mx.reset_peak_memory()
    cache = model.make_cache()
    started = time.perf_counter()
    output = model(mx.array([[1]], dtype=mx.uint32), cache=cache)
    logits_hash = _hash_logits(output.logits[0, -1])
    mx.synchronize()
    elapsed = time.perf_counter() - started
    peak = int(mx.get_peak_memory())
    del output, cache
    gc.collect()
    mx.synchronize()
    return {
        "seconds": elapsed,
        "logits_hash": logits_hash,
        "active_before_bytes": active_before,
        "peak_bytes": peak,
        "working_peak_bytes": max(0, peak - active_before),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    args = parser.parse_args()

    mx.set_wired_limit(int(440e9))
    mx.set_cache_limit(int(32e9))
    report = inspect_checkpoint(args.model, require_server_ready=True)
    model, _ = load(args.model)
    warm_residency(model)
    originals = (fp8._metal_input, packed._metal_input, grouped_fp8._metal_input)

    for enabled in (False, True):
        _set_contract(enabled, originals)
        for _ in range(args.warmups):
            _run_one(model)

    arms = {"raw_contiguous_baseline": [], "enforced_row_contiguous": []}
    for sample in range(args.samples):
        order = (False, True) if sample % 2 == 0 else (True, False)
        for enabled in order:
            _set_contract(enabled, originals)
            key = "enforced_row_contiguous" if enabled else "raw_contiguous_baseline"
            row = _run_one(model)
            row["sample"] = sample
            arms[key].append(row)
            print(json.dumps({"phase": key, **row}), flush=True)
    _set_contract(True, originals)

    summarized = {}
    for key, rows in arms.items():
        seconds = [row["seconds"] for row in rows]
        summarized[key] = {
            "samples_seconds": seconds,
            "median_seconds": statistics.median(seconds),
            "median_tok_s": 1.0 / statistics.median(seconds),
            "working_peak_bytes": max(row["working_peak_bytes"] for row in rows),
            "logits_hashes": [row["logits_hash"] for row in rows],
        }

    raw = summarized["raw_contiguous_baseline"]
    enforced = summarized["enforced_row_contiguous"]
    regression = enforced["median_seconds"] / raw["median_seconds"] - 1.0
    peak_increase = enforced["working_peak_bytes"] - raw["working_peak_bytes"]
    hashes = raw["logits_hashes"] + enforced["logits_hashes"]
    acceptance = {
        "contiguous_warm_regression_at_most_2_percent": regression <= MAX_REGRESSION,
        "working_peak_increase_at_most_64_mib": (
            peak_increase <= MAX_WORKING_PEAK_INCREASE
        ),
        "all_logits_hashes_identical": len(set(hashes)) == 1,
    }
    acceptance["accepted"] = all(acceptance.values())
    artifact = {
        "schema": "glm53-metal-input-layout-abi-v1",
        "date": date.today().isoformat(),
        "machine": platform.machine(),
        "checkpoint_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "kernel_abi": KERNEL_ABI_VERSION,
        "method": {
            "backend": "default-direct",
            "input": "single token with fresh cache",
            "warmups_per_arm": args.warmups,
            "paired_alternating_samples": args.samples,
        },
        "arms": summarized,
        "contiguous_warm_regression": regression,
        "working_peak_increase_bytes": peak_increase,
        "acceptance": acceptance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"phase": "result", **acceptance}), flush=True)
    return 0 if acceptance["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
