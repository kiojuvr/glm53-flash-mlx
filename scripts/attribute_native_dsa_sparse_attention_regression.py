#!/usr/bin/env python3
"""Attribute the exact Tier-2 native sparse-attention operator regression.

The accepted Tier-1 score/selection island remains the baseline.  This probe
splits only the rejected Tier-2 widening at its durable numerical boundary:

  selected-row gather + BF16 query scale | QK + mask + softmax + V readout

Both native halves reuse the Tier-2 fixed arena and exact kernels.  Each half
is compared with the pinned MLX 0.32.2 fallback using already-materialized
inputs, so no per-suboperator synchronization is inserted into a full model
execution.  The artificial fixture runs before the optional 320 GB model
load.  This is attribution evidence, not a production runtime backend.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import statistics
import sys
import tempfile
import time
import traceback
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
NATIVE_PACKAGE = ROOT / "native_execution"
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-dsa-sparse-attention-attribution-20260907.json"
)
CONTEXT = 262_144


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


def _atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _load_helpers():
    for path in (str(ROOT / "scripts"), str(NATIVE_PACKAGE)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from glm53_native_execution import NativeDSASparseAttentionPlan
    import probe_long_context_first_decode_boundary as boundary
    import probe_native_dsa_sparse_attention_island as tier2
    import probe_native_execution_engine_feasibility as tier0

    return NativeDSASparseAttentionPlan, boundary, tier2, tier0


def _arrays(value):
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _arrays(value[key])
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _arrays(item)


def _eval(value) -> None:
    arrays = list(_arrays(value))
    if arrays:
        mx.eval(*arrays)
    mx.synchronize()


def _exact(left: mx.array, right: mx.array) -> bool:
    mx.eval(left, right)
    return bool(mx.array_equal(left, right).item())


def _mlx_prepare(fixture, indices, valid):
    safe = mx.where(valid, indices, 0)
    gathered = mx.take_along_axis(
        fixture["latent_physical"],
        mx.broadcast_to(
            safe[..., None],
            safe.shape + (fixture["latent_physical"].shape[-1],),
        ),
        axis=2,
    ).reshape(2051, 512)
    scale = mx.array(fixture["attention"].scale, dtype=mx.bfloat16)
    scaled = (fixture["attention_query"].reshape(64, 512) * scale).astype(
        mx.bfloat16
    )
    return scaled, gathered


def _mlx_math(scaled_query, gathered, valid):
    return mx.fast.scaled_dot_product_attention(
        scaled_query.reshape(1, 64, 1, 512),
        gathered.reshape(1, 1, 2051, 512),
        gathered.reshape(1, 1, 2051, 512),
        scale=1.0,
        mask=valid.reshape(1, 1, 1, 2051),
    )


def _artificial_fixture(tier0):
    pools, kv_rows, kv_len = 512, 2_112, 2_051
    index_axis = mx.arange(32 * 128, dtype=mx.float32).reshape(1, 1, 32, 128)
    index_query = mx.sin(index_axis * 0.001).astype(mx.bfloat16)
    weights = mx.cos(mx.arange(32, dtype=mx.float32) * 0.03).reshape(
        1, 1, 32
    ).astype(mx.bfloat16)
    pool_keys = mx.cos(
        mx.arange(pools * 128, dtype=mx.float32).reshape(1, pools, 128)
        * 0.0001
    ).astype(mx.bfloat16)
    pool_indices = mx.arange(pools * 4, dtype=mx.int64).reshape(1, pools, 4)
    pool_valid = mx.arange(pools)[None] < (pools - 1)
    raw_positions = mx.array([[0, 2048, 2049, 2050]], dtype=mx.int64)
    raw_valid = mx.array([[False, True, True, True]], dtype=mx.bool_)
    current_valid = mx.ones((1,), dtype=mx.bool_)
    attention_query = mx.sin(
        mx.arange(64 * 512, dtype=mx.float32).reshape(1, 64, 1, 512)
        * 0.0003
    ).astype(mx.bfloat16)
    latent = mx.cos(
        mx.arange(kv_rows * 512, dtype=mx.float32).reshape(1, 1, kv_rows, 512)
        * 0.00007
    ).astype(mx.bfloat16)
    valid_candidates = mx.broadcast_to(pool_valid[:, None], (1, 1, pools))
    scores = tier0._score_expression(
        index_query, pool_keys, weights, valid_candidates, 128**-0.5
    )
    pool = SimpleNamespace(
        pool_keys=pool_keys,
        pool_indices=pool_indices,
        pool_valid=pool_valid,
        raw_positions=raw_positions,
        raw_valid=raw_valid,
        logical_pool_count=pools,
        total_tokens=kv_len,
        active_tail_count=3,
    )
    fixture = {
        "attention": SimpleNamespace(
            scale=512**-0.5,
            indexer=SimpleNamespace(
                softmax_scale=128**-0.5,
                index_topk=2048,
                index_kpool=4,
                index_kpool_always_select_tail=True,
            ),
        ),
        "pool": pool,
        "index_query": index_query,
        "weights": weights,
        "scores": scores,
        "valid_candidates": valid_candidates,
        "current_valid": current_valid,
        "attention_query": attention_query,
        "latent_physical": latent,
    }
    _eval(fixture)
    return fixture


def _selection(fixture, tier0):
    return tier0._reference_selection(
        fixture["attention"].indexer,
        fixture["pool"],
        fixture["scores"],
        fixture["valid_candidates"],
        fixture["current_valid"],
    )


def _exact_case(fixture, plan, tier0) -> tuple[dict, tuple]:
    indices, valid = _selection(fixture, tier0)
    native_indices = indices.astype(mx.int32)
    reference_prepared = _mlx_prepare(fixture, indices, valid)
    # Native submission accepts only already-scheduled buffers.  Materialize
    # the attribution boundary itself before either candidate is encoded.
    _eval((native_indices, valid, reference_prepared))
    native_prepared = plan.debug_prepare_inputs(
        native_indices,
        valid,
        fixture["attention_query"],
        fixture["latent_physical"],
        fixture["pool"].total_tokens,
    )
    _eval((reference_prepared, native_prepared))
    reference_math = _mlx_math(*reference_prepared, valid)
    native_math = plan.debug_attention_math(
        reference_prepared[0], reference_prepared[1], valid
    )
    _eval((reference_math, native_math))
    row = {
        "scaled_query_byte_exact": _exact(
            reference_prepared[0], native_prepared[0]
        ),
        "gathered_latent_byte_exact": _exact(
            reference_prepared[1], native_prepared[1]
        ),
        "attention_math_byte_exact": _exact(reference_math, native_math),
        "selected_width": int(valid.size),
        "valid_selected": int(mx.sum(valid).item()),
    }
    row["all_exact"] = all(
        row[key]
        for key in (
            "scaled_query_byte_exact",
            "gathered_latent_byte_exact",
            "attention_math_byte_exact",
        )
    )
    return row, (native_indices, valid, reference_prepared)


def _median_timing(functions: dict[str, object], warmups: int, samples: int):
    measured = {name: [] for name in functions}
    names = tuple(functions)
    for iteration in range(warmups + samples):
        order = names if iteration % 2 == 0 else tuple(reversed(names))
        for name in order:
            started = time.perf_counter_ns()
            output = functions[name]()
            submitted = time.perf_counter_ns()
            _eval(output)
            finished = time.perf_counter_ns()
            if iteration >= warmups:
                measured[name].append(
                    {
                        "host_submit_ms": (submitted - started) / 1e6,
                        "wall_ms": (finished - started) / 1e6,
                    }
                )
    return {
        name: {
            "median_wall_ms": statistics.median(row["wall_ms"] for row in rows),
            "median_host_submit_ms": statistics.median(
                row["host_submit_ms"] for row in rows
            ),
            "samples": rows,
        }
        for name, rows in measured.items()
    }


def _timing(fixtures, cases, plans, warmups, samples):
    def mlx_prepare():
        return [
            _mlx_prepare(fixture, indices, valid)
            for fixture, (indices, valid, _) in zip(fixtures, cases)
        ]

    def native_prepare():
        return [
            plan.debug_prepare_inputs(
                indices,
                valid,
                fixture["attention_query"],
                fixture["latent_physical"],
                fixture["pool"].total_tokens,
            )
            for fixture, (indices, valid, _), plan in zip(fixtures, cases, plans)
        ]

    def mlx_math():
        return [
            _mlx_math(*prepared, valid)
            for (_, valid, prepared) in cases
        ]

    def native_math():
        return [
            plan.debug_attention_math(
                prepared[0], prepared[1], valid
            )
            for (_, valid, prepared), plan in zip(cases, plans)
        ]

    prepare = _median_timing(
        {"mlx": mlx_prepare, "native": native_prepare}, warmups, samples
    )
    math = _median_timing(
        {"mlx": mlx_math, "native": native_math}, warmups, samples
    )
    for stage in (prepare, math):
        stage["native_minus_mlx_ms"] = (
            stage["native"]["median_wall_ms"]
            - stage["mlx"]["median_wall_ms"]
        )
        stage["native_minus_mlx_ms_per_layer"] = stage[
            "native_minus_mlx_ms"
        ] / len(fixtures)
    return {"prepare": prepare, "attention_math": math}


def _decision(timing: dict) -> str:
    prepare = timing["prepare"]["native_minus_mlx_ms"]
    math = timing["attention_math"]["native_minus_mlx_ms"]
    if prepare > 0.0 and prepare >= math:
        return "prototype_exact_indirect_latent_loaders"
    if math > 0.0:
        return "stop_native_attention_keep_tier1_score_island"
    return "tier2_regression_not_reproduced_repeat_bounded_trace"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--artificial-only", action="store_true")
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    args = parser.parse_args()
    artifact = {
        "schema": "glm53-native-dsa-sparse-attention-attribution-v1",
        "date": date.today().isoformat(),
        "complete": False,
        "accepted": False,
        "probe_only": True,
        "mlx_version": importlib.metadata.version("mlx"),
        "runtime_changes": {
            "runtime": False,
            "server": False,
            "apc": False,
            "cache_abi": False,
            "kernel_abi": False,
        },
    }
    try:
        Plan, boundary, tier2, tier0 = _load_helpers()
        _progress("artificial_attribution_contract")
        fixture = _artificial_fixture(tier0)
        plan = Plan(512, 2112, 128**-0.5, 512**-0.5)
        exact, case = _exact_case(fixture, plan, tier0)
        artifact["artificial"] = {
            "exactness": exact,
            "timing": _timing(
                [fixture], [case], [plan], args.warmups, args.samples
            ),
        }
        _atomic_write(args.output, artifact)
        if not exact["all_exact"]:
            artifact.update(
                complete=True,
                decision="reject_attribution_boundary_numerical_mismatch",
            )
            _atomic_write(args.output, artifact)
            return 1
        if args.artificial_only:
            artifact.update(
                complete=True,
                decision="artificial_attribution_passed_real_layers_required",
            )
            _atomic_write(args.output, artifact)
            print(json.dumps({"output": str(args.output), **{
                key: artifact[key] for key in ("complete", "accepted", "decision")
            }}, indent=2))
            return 0

        from glm53_flash_mlx.loader import load, warm_residency
        from glm53_flash_mlx.manifest import EXPECTED_DSA, inspect_checkpoint

        inspect_checkpoint(args.model, require_server_ready=True)
        mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
        mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
        _progress("load_model")
        model, _ = load(
            args.model,
            experimental_packed_decode_moe=True,
            experimental_compact_nope_dsa_cache=True,
            compact_cache_capacity_tokens=CONTEXT + 64,
        )
        warm_residency(model)
        _progress("build_synthetic_context", context=CONTEXT)
        source = boundary._synthetic_cache(model, CONTEXT, "compact-nope-dsa")
        fixtures = [
            tier2._prepared_fixture(model, boundary, source, CONTEXT, layer)
            for layer in EXPECTED_DSA
        ]
        plans = [
            Plan(
                int(row["pool"].pool_keys.shape[1]),
                int(row["latent_physical"].shape[2]),
                float(row["attention"].indexer.softmax_scale),
                float(row["attention"].scale),
            )
            for row in fixtures
        ]
        exact_rows = []
        cases = []
        for fixture, plan in zip(fixtures, plans):
            _progress("exact_layer", layer=fixture["layer"])
            exact, case = _exact_case(fixture, plan, tier0)
            exact["layer"] = fixture["layer"]
            exact_rows.append(exact)
            cases.append(case)
        all_exact = all(row["all_exact"] for row in exact_rows)
        artifact["real_layers"] = {
            "context_tokens": CONTEXT,
            "layers": exact_rows,
            "all_exact": all_exact,
        }
        if all_exact:
            _progress("time_attribution", context=CONTEXT)
            artifact["real_layers"]["timing"] = _timing(
                fixtures, cases, plans, args.warmups, args.samples
            )
            artifact["decision"] = _decision(
                artifact["real_layers"]["timing"]
            )
            artifact["accepted"] = True
        else:
            artifact["decision"] = "reject_real_layer_attribution_mismatch"
        artifact["complete"] = True
        fixtures.clear()
        cases.clear()
        plans.clear()
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
    except Exception as error:
        artifact.update(
            complete=True,
            accepted=False,
            decision="native_sparse_attention_attribution_failed",
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
    _atomic_write(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": artifact["complete"],
                "accepted": artifact["accepted"],
                "decision": artifact["decision"],
            },
            indent=2,
        )
    )
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
