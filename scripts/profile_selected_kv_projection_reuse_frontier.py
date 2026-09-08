#!/usr/bin/env python3
"""Measure selected-K/V projection reuse on real Q256 DSA prefills.

The profile reuses the accepted synthetic long-context cache fixture and the
same deterministic token chunk as the authoritative prefill profile.  It
captures every DSA layer's real Indexer result without changing its graph,
then compares full-history, per-query, and block-union projection row counts.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import statistics
import sys
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np

from glm53_flash_mlx.abi import MLX_VLM_REVISION, NOPE_DSA_CACHE_ABI_COMPACT
from glm53_flash_mlx.context_capacity import (
    DESIGN_PREFILL_PROFILE_CONTEXTS,
    DESIGN_PREFILL_QUERY_ROWS,
    DESIGN_TOTAL_CONTEXT_TOKENS,
    NATIVE_CONTEXT_CAPACITY_CONTRACT,
)
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import EXPECTED_DSA, inspect_checkpoint


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-selected-kv-projection-reuse-frontier-20260908.json"
)
REFERENCE_PROFILE = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-prefill-critical-path-20260908.json"
)
CONTEXTS = DESIGN_PREFILL_PROFILE_CONTEXTS
ROWS = DESIGN_PREFILL_QUERY_ROWS
BLOCK_ROWS = (1, 4, 8, 16, 32, 64, 128, 256)
EXPECTED_DSA_LAYERS = 11


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class _CaptureIndexer:
    def __init__(self, delegate, layer: int):
        self._delegate = delegate
        self.layer = layer
        self.output = None

    def __call__(self, *args, **kwargs):
        self.output = self._delegate(*args, **kwargs)
        return self.output

    def __getattr__(self, name):
        return getattr(self._delegate, name)


@contextlib.contextmanager
def _capture_indexers(model):
    installed = []
    captures = []
    try:
        for layer in EXPECTED_DSA:
            attention = model.language_model.model.layers[layer].self_attn
            original = attention.indexer
            capture = _CaptureIndexer(original, layer)
            attention.indexer = capture
            installed.append((attention, original))
            captures.append(capture)
        yield captures
    finally:
        for attention, original in installed:
            attention.indexer = original


def _load_helpers():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import profile_native_prefill_critical_path as profile

    return profile, profile._load_boundary_probe()


def _layer_frontier(indices: mx.array, context: int, layer: int) -> dict[str, object]:
    mx.eval(indices)
    host = np.asarray(indices, dtype=np.int32).reshape(ROWS, -1)
    valid = (host >= 0) & (host < context)
    rows = {}
    for block in BLOCK_ROWS:
        total_unique = 0
        block_counts = []
        for start in range(0, ROWS, block):
            values = host[start : start + block]
            mask = valid[start : start + block]
            count = int(np.unique(values[mask]).size)
            total_unique += count
            block_counts.append(count)
        rows[str(block)] = {
            "projection_rows": total_unique,
            "block_count": len(block_counts),
            "mean_unique_rows_per_block": statistics.mean(block_counts),
            "max_unique_rows_per_block": max(block_counts),
            "reuse_ratio": int(valid.sum()) / total_unique,
        }
    strategies = {
        "full_history": context,
        **{f"union_q{block}": rows[str(block)]["projection_rows"] for block in BLOCK_ROWS},
    }
    best = min(strategies, key=strategies.get)
    return {
        "layer": layer,
        "shape": list(indices.shape),
        "valid_selected_entries": int(valid.sum()),
        "sentinel_entries": int((~valid).sum()),
        "block_union": rows,
        "projection_rows_by_strategy": strategies,
        "minimum_projection_strategy": best,
        "minimum_projection_rows": strategies[best],
    }


def _context_case(model, boundary_probe, profile, reference: dict, context: int):
    prefix = context - ROWS
    _progress("build_synthetic_prefix", context=context, prefix=prefix)
    source = boundary_probe._synthetic_cache(model, prefix, "compact-nope-dsa")
    cache = boundary_probe._clone_cache(source, context)
    tokens = profile._tokens(
        ROWS, int(reference["vocab_size"]), context
    )
    _progress("capture_real_indexer_selections", context=context)
    with _capture_indexers(model) as captures:
        output = model(tokens, cache=cache)
    logits = output.logits[0, -1]
    captured = [item.output for item in captures]
    if any(value is None for value in captured):
        raise RuntimeError("one or more DSA Indexers did not execute")
    mx.eval(logits, *captured)
    state = profile._state_signature(boundary_probe, cache)
    reference_row = reference["contexts"][str(context)]["uninstrumented"]["samples"][0]
    layers = [
        _layer_frontier(value, context, capture.layer)
        for capture, value in zip(captures, captured)
    ]
    totals = {
        "full_history": context * len(layers),
        **{
            f"union_q{block}": sum(
                row["block_union"][str(block)]["projection_rows"]
                for row in layers
            )
            for block in BLOCK_ROWS
        },
    }
    best = min(totals, key=totals.get)
    return {
        "context_tokens_after_chunk": context,
        "synthetic_prefix_tokens": prefix,
        "query_rows": ROWS,
        "captured_dsa_layers": len(layers),
        "layers": layers,
        "aggregate_projection_rows_by_strategy": totals,
        "aggregate_minimum_strategy": best,
        "aggregate_minimum_projection_rows": totals[best],
        "aggregate_minimum_vs_full_history": totals[best] / totals["full_history"],
        "exactness": {
            "final_logits_matches_authoritative_profile": (
                profile._hash_array(logits) == reference_row["final_logits_sha256"]
            ),
            "state_matches_authoritative_profile": (
                state["diagnostic_state_sha256"]
                == reference_row["state"]["diagnostic_state_sha256"]
            ),
            "dsa_offsets_match_authoritative_profile": (
                state["dsa_offsets"] == reference_row["state"]["dsa_offsets"]
            ),
        },
    }


def _checks(artifact: dict[str, object]) -> dict[str, bool]:
    contexts = artifact.get("contexts", {})
    complete = set(map(int, contexts)) == set(CONTEXTS)
    return {
        "32k_128k_320k_real_q256_selections_captured": complete,
        "all_11_dsa_layers_captured_per_context": complete and all(
            row["captured_dsa_layers"] == EXPECTED_DSA_LAYERS
            for row in contexts.values()
        ),
        "capture_preserves_logits_state_and_offsets": complete and all(
            all(row["exactness"].values()) for row in contexts.values()
        ),
        "all_projection_strategies_have_positive_bounded_rows": complete and all(
            all(
                0 < value <= ROWS * 2051 * EXPECTED_DSA_LAYERS
                for value in row["aggregate_projection_rows_by_strategy"].values()
            )
            for row in contexts.values()
        ),
        "runtime_server_cache_kernel_abi_admission_unchanged": (
            artifact.get("runtime_changes")
            == {
                "admission": False,
                "backend": False,
                "cache_abi": False,
                "kernel_abi": False,
                "server": False,
            }
        ),
    }


def _decision(artifact: dict[str, object]) -> str:
    if not all(artifact.get("checks", {}).values()):
        return "projection_reuse_profile_incomplete_or_invalid"
    long = artifact["contexts"][str(320 << 10)]
    best = long["aggregate_minimum_strategy"]
    if best == "full_history":
        return "compose_full_history_projection_with_native_sparse_attention"
    return f"implement_{best}_selected_projection_reuse_plan"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-profile", type=Path, default=REFERENCE_PROFILE)
    parser.add_argument("--contexts", type=int, nargs="+", default=list(CONTEXTS))
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    args = parser.parse_args(argv)
    if any(context not in CONTEXTS for context in args.contexts):
        raise ValueError("contexts must be selected from 32K, 128K, and 320K")
    reference = json.loads(args.reference_profile.read_text())
    reference_contexts = reference.get("contexts", {})
    reference_exact = (
        set(map(int, reference_contexts)) == set(CONTEXTS)
        and all(
            row["uninstrumented"]["repeat_logits_exact"]
            and row["uninstrumented"]["repeat_state_exact"]
            and all(row["exactness"].values())
            for row in reference_contexts.values()
        )
    )
    if not reference_exact:
        raise RuntimeError(
            "complete exact authoritative prefill reference is required"
        )
    report = inspect_checkpoint(args.model, require_server_ready=True)
    profile, boundary_probe = _load_helpers()
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    os.environ["GLM53_EXPERIMENTAL_NATIVE_INDEXPOOL_UPDATE"] = "0"
    _progress("load_model")
    model, _ = load(
        args.model,
        experimental_packed_decode_moe=True,
        experimental_compact_nope_dsa_cache=True,
        compact_cache_capacity_tokens=DESIGN_TOTAL_CONTEXT_TOKENS,
    )
    warm_residency(model)
    artifact = {
        "schema": "glm53-selected-kv-projection-reuse-frontier-v1",
        "date": date.today().isoformat(),
        "complete": False,
        "accepted": False,
        "decision": "projection_reuse_profile_incomplete_or_invalid",
        "profiling_only": True,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "capacity_contract": NATIVE_CONTEXT_CAPACITY_CONTRACT,
        "compact_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
        "query_rows": ROWS,
        "projection_block_rows": list(BLOCK_ROWS),
        "reference_profile": str(args.reference_profile),
        "contexts": {},
        "runtime_changes": {
            "admission": False,
            "backend": False,
            "cache_abi": False,
            "kernel_abi": False,
            "server": False,
        },
    }
    if args.output.exists() and not args.rerun:
        existing = json.loads(args.output.read_text())
        if existing.get("schema") == artifact["schema"]:
            if existing.get("checkpoint_fingerprint") != report.fingerprint:
                raise ValueError("existing artifact uses another checkpoint")
            artifact = existing
    for context in args.contexts:
        key = str(context)
        if key in artifact["contexts"] and not args.rerun:
            _progress("skip_completed_context", context=context)
            continue
        artifact["contexts"][key] = _context_case(
            model, boundary_probe, profile, reference, context
        )
        artifact["checks"] = _checks(artifact)
        artifact["complete"] = all(artifact["checks"].values())
        artifact["accepted"] = artifact["complete"]
        artifact["decision"] = _decision(artifact)
        _atomic_write(args.output, artifact)
    print(json.dumps({
        "output": str(args.output),
        "complete": artifact["complete"],
        "accepted": artifact["accepted"],
        "decision": artifact["decision"],
        "completed_contexts": sorted(map(int, artifact["contexts"])),
    }))
    return 0 if artifact["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
