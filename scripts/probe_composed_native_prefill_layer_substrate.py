#!/usr/bin/env python3
"""Prove the fixed-arena substrate for a composed native prefill layer.

This probe is intentionally arithmetic-free. Its three pass-through stages
reserve the final 512K DSA and 256-row MoE geometry and share one native Metal
encoder scope. Later probes replace the middle stages with exact DSA and MoE
math without changing buffer ownership, scratch capacity, or the Python ABI.
No throughput or production-runtime claim is made here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np

from glm53_flash_mlx.native_prefill_plan import NATIVE_PREFILL_PLAN_ABI

ROOT = Path(__file__).resolve().parents[1]
NATIVE_PACKAGE = ROOT / "native_execution"
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-composed-native-prefill-layer-substrate-20260908.json"
)
EXPECTED_TOPOLOGY = (
    "ingress_to_hidden_ping",
    "dsa_region_hidden_ping_to_pong",
    "moe_region_hidden_pong_to_layer_output",
)
EXPECTED_DSA_SCRATCH_BYTES = 64 << 20
MAX_SUBSTRATE_ARENA_BYTES = 128 << 20


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_plan_type():
    native = str(NATIVE_PACKAGE)
    if native not in sys.path:
        sys.path.insert(0, native)
    from glm53_native_execution import NativePrefillLayerSubstrate

    return NativePrefillLayerSubstrate


def _hidden(iteration: int) -> mx.array:
    host = (
        (np.arange(256 * 4096, dtype=np.uint32) * 7_919 + iteration * 101)
        % 65_521
    ).astype(np.float32)
    host = ((host - 32_760.0) / 4_096.0).reshape(256, 4096)
    value = mx.array(host).astype(mx.bfloat16)
    mx.eval(value)
    return value


def _hash(value: mx.array) -> str:
    mx.eval(value)
    host = np.ascontiguousarray(np.asarray(value.astype(mx.float32)))
    return hashlib.sha256(host.tobytes()).hexdigest()


def _execute_exact(plan, iteration: int) -> tuple[bool, float, float, str]:
    hidden = _hidden(iteration)
    started = time.perf_counter_ns()
    output = plan.execute(hidden)[0]
    submitted = time.perf_counter_ns()
    mx.eval(output)
    finished = time.perf_counter_ns()
    exact = bool(mx.array_equal(hidden, output).item())
    return (
        exact,
        (submitted - started) / 1e6,
        (finished - started) / 1e6,
        _hash(output),
    )


def _rejected_without_output_mutation(plan, output_hash: str) -> dict[str, object]:
    cases = {
        "wrong_dtype": mx.zeros((256, 4096), dtype=mx.float32),
        "wrong_rows": mx.zeros((128, 4096), dtype=mx.bfloat16),
        "wrong_hidden": mx.zeros((256, 2048), dtype=mx.bfloat16),
    }
    rows = {}
    for name, value in cases.items():
        mx.eval(value)
        try:
            plan.execute(value)
        except (RuntimeError, ValueError) as error:
            rows[name] = {
                "rejected": True,
                "error": str(error),
                "output_unchanged": _hash(plan.output) == output_hash,
            }
        else:
            rows[name] = {"rejected": False, "output_unchanged": False}
    return rows


def build_artifact(*, repeats: int) -> dict[str, object]:
    if repeats < 8:
        raise ValueError("at least eight substrate repeats are required")
    plan_type = _load_plan_type()
    active_before = int(mx.get_active_memory())
    plan = plan_type()
    active_after_plan = int(mx.get_active_memory())
    initial_identities = list(plan.buffer_identities)

    for iteration in range(4):
        exact, _, _, _ = _execute_exact(plan, iteration)
        if not exact:
            raise RuntimeError("native prefill substrate warmup changed hidden bytes")

    host_ms = []
    wall_ms = []
    hashes = []
    all_exact = True
    for iteration in range(repeats):
        exact, host, wall, digest = _execute_exact(plan, iteration + 10)
        all_exact = all_exact and exact
        host_ms.append(host)
        wall_ms.append(wall)
        hashes.append(digest)

    valid_hash = _hash(plan.execute(_hidden(999))[0])
    invalid = _rejected_without_output_mutation(plan, valid_hash)
    final_identities = list(plan.buffer_identities)
    topology = tuple(plan.fixed_topology)
    structure = {
        "query_rows": plan.query_rows,
        "query_block_rows": plan.query_block_rows,
        "query_block_count": plan.query_block_count,
        "hidden_size": plan.hidden_size,
        "physical_pool_rows": plan.physical_pool_rows,
        "selected_width": plan.selected_width,
        "route_rows": plan.route_rows,
        "fixed_topology": list(topology),
        "native_encoder_scopes_per_execute": (
            plan.native_encoder_scopes_per_execute
        ),
        "startup_pipeline_lookup_count": plan.startup_pipeline_lookup_count,
        "pipeline_lookup_count_per_execute": (
            plan.pipeline_lookup_count_per_execute
        ),
        "dynamic_allocation_count": plan.dynamic_allocation_count,
        "graph_node_count": plan.graph_node_count,
        "shape_discovery_count": plan.shape_discovery_count,
        "host_synchronization_count": plan.host_synchronization_count,
        "returned_intermediate_tensor_bytes": (
            plan.returned_intermediate_tensor_bytes
        ),
        "dsa_score_scratch_bytes": plan.dsa_score_scratch_bytes,
        "scratch_bytes": plan.scratch_bytes,
        "arena_bytes": plan.arena_bytes,
        "active_delta_after_plan_bytes": active_after_plan - active_before,
        "initial_buffer_identities": initial_identities,
        "final_buffer_identities": final_identities,
        "buffer_identities_stable": bool(plan.buffer_identities_stable)
        and initial_identities == final_identities,
    }
    measurements = {
        "repeat_count": repeats,
        "all_outputs_byte_exact": all_exact,
        "output_hashes_are_input_specific": len(set(hashes)) == repeats,
        "median_native_host_call_ms": statistics.median(host_ms),
        "median_synchronized_wall_ms": statistics.median(wall_ms),
        "note": "pass-through substrate timing is not a prefill performance claim",
    }
    checks = {
        "512k_q256_geometry_is_fixed": (
            plan.query_rows == 256
            and plan.query_block_rows == 128
            and plan.query_block_count == 2
            and plan.physical_pool_rows == 131_072
        ),
        "dsa_score_scratch_is_exactly_64mib": (
            plan.dsa_score_scratch_bytes == EXPECTED_DSA_SCRATCH_BYTES
        ),
        "complete_substrate_arena_is_at_most_128mib": (
            plan.arena_bytes <= MAX_SUBSTRATE_ARENA_BYTES
        ),
        "hidden_ping_pong_output_is_byte_exact": all_exact,
        "fixed_topology_has_dsa_and_moe_regions": topology == EXPECTED_TOPOLOGY,
        "one_native_encoder_scope_per_execute": (
            plan.native_encoder_scopes_per_execute == 1
        ),
        "pipelines_are_prebound_before_execute": (
            plan.startup_pipeline_lookup_count == 1
            and plan.pipeline_lookup_count_per_execute == 0
        ),
        "no_dynamic_allocation_graph_shape_or_host_sync": (
            plan.dynamic_allocation_count == 0
            and plan.graph_node_count == 0
            and plan.shape_discovery_count == 0
            and plan.host_synchronization_count == 0
        ),
        "no_intermediate_tensor_returns_to_mlx": (
            plan.returned_intermediate_tensor_bytes == 0
        ),
        "all_native_buffer_addresses_are_stable": structure[
            "buffer_identities_stable"
        ],
        "invalid_inputs_fail_before_output_mutation": all(
            row["rejected"] and row["output_unchanged"]
            for row in invalid.values()
        ),
    }
    accepted = all(checks.values())
    return {
        "schema": "glm53-composed-native-prefill-layer-substrate-v1",
        "date": date.today().isoformat(),
        "complete": True,
        "accepted": accepted,
        "decision": (
            "advance_substrate_to_exact_dsa_and_moe_arithmetic"
            if accepted
            else "stop_composed_native_prefill_layer_substrate"
        ),
        "probe_only": True,
        "native_prefill_plan_abi": NATIVE_PREFILL_PLAN_ABI,
        "arithmetic_coverage": {
            "dsa": False,
            "moe": False,
            "residual_norm": False,
            "pass_through_only": True,
        },
        "structure": structure,
        "measurements": measurements,
        "invalid_inputs": invalid,
        "checks": checks,
        "runtime_changes": {
            "server": False,
            "admission": False,
            "cache_abi": False,
            "kernel_abi": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=32)
    args = parser.parse_args(argv)
    artifact = build_artifact(repeats=args.repeats)
    _atomic_write(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": artifact["complete"],
                "accepted": artifact["accepted"],
                "decision": artifact["decision"],
                "arena_bytes": artifact["structure"]["arena_bytes"],
            }
        )
    )
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
