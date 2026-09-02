#!/usr/bin/env python3
"""Validate resident tensor lifetime contracts against reusable staging storage."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import tempfile
import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from glm53_flash_mlx.abi import MLX_VLM_REVISION
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint
from glm53_flash_mlx.ownership import (
    TensorLayout,
    borrowed_ephemeral_tensor,
    materialize_owned,
    owned_tensor,
    resident_concatenate,
)
from glm53_flash_mlx.packed import PackedFP8ExpertBank, PackedFP8MoE


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-resident-tensor-ownership-20260902.json"
)
MODEL_DEFAULT = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
STARTUP_PEAK_LIMIT = 340_000_000_000
READY_SECONDS_LIMIT = 190.0
REFERENCE_PACKED_ARTIFACT = (
    REPOSITORY / "bench-results/m3ultra512-packed-decode-runtime-20260831.json"
)


def _atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha(value: mx.array) -> str:
    mx.eval(value)
    if value.dtype == mx.bfloat16:
        raw = np.ascontiguousarray(np.asarray(value.view(mx.uint16)))
    else:
        raw = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(raw.tobytes()).hexdigest()


def _resident_staging_fixture() -> dict:
    staging = np.empty((3, 32), dtype=np.float32)
    q_expected = np.arange(staging.size, dtype=np.float32).reshape(staging.shape)
    staging[...] = q_expected
    unsafe = mx.asarray(staging)
    mx.eval(unsafe)
    q = materialize_owned(
        borrowed_ephemeral_tensor(
            unsafe,
            owner=staging,
            layout=TensorLayout.ROW_MAJOR_CONTIGUOUS,
        )
    )
    q_hash = _sha(q.value)

    kv_expected = q_expected * -0.25 + 17.0
    staging[...] = kv_expected
    alias_mutation_reproduced = np.array_equal(np.asarray(unsafe), kv_expected)
    kv = materialize_owned(
        borrowed_ephemeral_tensor(
            mx.asarray(staging),
            owner=staging,
            layout=TensorLayout.ROW_MAJOR_CONTIGUOUS,
        )
    )
    fused = resident_concatenate(
        [owned_tensor(q.value, layout=q.layout), owned_tensor(kv.value, layout=kv.layout)],
        axis=0,
    )
    fused_expected = np.concatenate([q_expected, kv_expected], axis=0)
    staging.fill(12345.0)
    del unsafe, staging
    gc.collect()

    return {
        "unsafe_alias_mutation_reproduced": bool(alias_mutation_reproduced),
        "q_source_overwrite_invariant": _sha(q.value) == q_hash,
        "fused_projection_byte_exact": bool(
            np.array_equal(np.asarray(fused.value), fused_expected)
        ),
        "source_lifetime_end_safe": bool(
            np.array_equal(np.asarray(q.value), q_expected)
        ),
        "resident_contract": {
            "ownership": fused.ownership.value,
            "layout": fused.layout.value,
        },
    }


def _packed_staging_fixture() -> dict:
    experts = 8
    hidden = 128
    intermediate = 128
    weight_staging = np.empty((experts, 2 * intermediate, hidden), np.uint8)
    gate_up_expected = (
        np.arange(weight_staging.size, dtype=np.uint32).reshape(weight_staging.shape)
        % 120
    ).astype(np.uint8)
    weight_staging[...] = gate_up_expected
    gate_up = materialize_owned(
        borrowed_ephemeral_tensor(
            mx.asarray(weight_staging),
            owner=weight_staging,
            layout=TensorLayout.ROW_MAJOR_CONTIGUOUS,
        )
    )
    down_expected = (
        np.arange(experts * hidden * intermediate, dtype=np.uint32).reshape(
            experts, hidden, intermediate
        )
        * 7
        % 120
    ).astype(np.uint8)
    weight_staging[:, :hidden, :] = down_expected
    down = materialize_owned(
        borrowed_ephemeral_tensor(
            mx.asarray(weight_staging[:, :hidden, :]),
            owner=weight_staging,
            layout=TensorLayout.ROW_MAJOR_CONTIGUOUS,
        )
    )

    scale_staging = np.empty((experts, 2, 1), np.float32)
    gate_up_scale_expected = np.linspace(
        0.001, 0.008, scale_staging.size, dtype=np.float32
    ).reshape(scale_staging.shape)
    scale_staging[...] = gate_up_scale_expected
    gate_up_scale = materialize_owned(
        borrowed_ephemeral_tensor(
            mx.asarray(scale_staging),
            owner=scale_staging,
            layout=TensorLayout.ROW_MAJOR_CONTIGUOUS,
        )
    )
    down_scale_expected = np.linspace(
        0.002, 0.009, experts, dtype=np.float32
    ).reshape(experts, 1, 1)
    scale_staging[:, :1, :] = down_scale_expected
    down_scale = materialize_owned(
        borrowed_ephemeral_tensor(
            mx.asarray(scale_staging[:, :1, :]),
            owner=scale_staging,
            layout=TensorLayout.ROW_MAJOR_CONTIGUOUS,
        )
    )
    bank = PackedFP8ExpertBank(
        gate_up,
        gate_up_scale,
        down,
        down_scale,
        intermediate_size=intermediate,
    )

    routes = mx.arange(experts, dtype=mx.uint32).reshape(1, 1, experts)
    scores = mx.full((1, 1, experts), 1.0 / experts, dtype=mx.float32)

    class FixedGate(nn.Module):
        def __call__(self, x):
            return routes, scores.astype(x.dtype)

    config = SimpleNamespace(
        hidden_size=hidden,
        moe_intermediate_size=intermediate,
        swiglu_limit=3.0,
        n_routed_experts=experts,
        num_experts_per_tok=experts,
    )
    moe = PackedFP8MoE(bank, config, FixedGate(), None)
    x = mx.linspace(-0.25, 0.25, hidden).reshape(1, 1, hidden).astype(mx.bfloat16)
    before = moe(x)
    mx.eval(before)
    before_hash = _sha(before)
    weight_staging.fill(255)
    scale_staging.fill(-7.0)
    del weight_staging, scale_staging
    gc.collect()
    after = moe(x)
    mx.eval(after)

    return {
        "storage_contracts": bank.storage_contracts,
        "weight_dtype": str(bank.gate_up_weight.dtype).rsplit(".", 1)[-1],
        "scale_dtype": str(bank.gate_up_scale_inv.dtype).rsplit(".", 1)[-1],
        "gate_up_weight_byte_exact": bool(
            np.array_equal(np.asarray(bank.gate_up_weight), gate_up_expected)
        ),
        "gate_up_scale_byte_exact": bool(
            np.array_equal(np.asarray(bank.gate_up_scale_inv), gate_up_scale_expected)
        ),
        "down_weight_byte_exact": bool(
            np.array_equal(np.asarray(bank.down_weight), down_expected)
        ),
        "down_scale_byte_exact": bool(
            np.array_equal(np.asarray(bank.down_scale_inv), down_scale_expected)
        ),
        "selected_output_byte_exact": bool(mx.array_equal(before, after).item()),
        "selected_output_sha256": before_hash,
    }


def _official_oracle(model, processor, report) -> dict:
    scripts = str(REPOSITORY / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import probe_exact_sigmoid_gate_metal_barrier as oracle_probe

    return oracle_probe._official_oracle(model, processor, report)


def _full_model(path: Path, report) -> dict:
    mx.set_wired_limit(int(440e9))
    mx.set_cache_limit(int(32e9))
    mx.reset_peak_memory()
    started = time.perf_counter()
    model, processor = load(path, experimental_packed_decode_moe=True)
    load_seconds = time.perf_counter() - started
    warm_started = time.perf_counter()
    resident_bytes = warm_residency(model)
    warm_seconds = time.perf_counter() - warm_started
    install_report = model._glm53_packed_decode_report
    install_peak = max(
        (row["peak_bytes"] for row in install_report["layers"]), default=0
    )
    startup_peak = max(mx.get_peak_memory(), install_peak)
    startup_active = mx.get_active_memory()

    banks = [
        layer.mlp.bank
        for layer in model.language_model.model.layers
        if hasattr(layer.mlp, "bank")
    ]
    contracts = [bank.storage_contracts for bank in banks]
    all_contracts_owned = all(
        row == {"ownership": "owned", "layout": "row-major-contiguous"}
        for bank in contracts
        for row in bank.values()
    )
    oracle = _official_oracle(model, processor, report)
    return {
        "executed": True,
        "backend": getattr(model, "_glm53_moe_backend", None),
        "load_seconds": load_seconds,
        "warm_residency_seconds": warm_seconds,
        "ready_seconds": load_seconds + warm_seconds,
        "resident_bytes": resident_bytes,
        "startup_active_bytes": startup_active,
        "startup_peak_bytes": startup_peak,
        "layerwise_install_peak_bytes": install_peak,
        "packed_bank_count": len(banks),
        "all_packed_bank_contracts_owned_row_major": all_contracts_owned,
        "model_storage_contract": model._glm53_tensor_storage_contract,
        "official_oracle": oracle,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path, default=MODEL_DEFAULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not mx.metal.is_available():
        raise RuntimeError("resident tensor ownership probe requires MLX/Metal")

    report = inspect_checkpoint(args.model, require_server_ready=True)
    staging = _resident_staging_fixture()
    packed = _packed_staging_fixture()
    full_model = _full_model(args.model, report)
    reference = json.loads(REFERENCE_PACKED_ARTIFACT.read_text())
    reference_peak = reference["packed_decode"]["install_peak_bytes"]
    peak_delta = full_model["startup_peak_bytes"] - reference_peak
    acceptance = {
        "reusable_staging_corruption_reproduced": staging[
            "unsafe_alias_mutation_reproduced"
        ],
        "ephemeral_source_overwrite_invariant": staging[
            "q_source_overwrite_invariant"
        ],
        "source_lifetime_end_safe": staging["source_lifetime_end_safe"],
        "fused_projection_byte_exact": staging["fused_projection_byte_exact"],
        "packed_fp8_weight_and_scale_byte_exact": all(
            packed[name]
            for name in (
                "gate_up_weight_byte_exact",
                "gate_up_scale_byte_exact",
                "down_weight_byte_exact",
                "down_scale_byte_exact",
            )
        ),
        "packed_selected_output_byte_exact": packed["selected_output_byte_exact"],
        "all_42_packed_banks_owned_row_major": (
            full_model["packed_bank_count"] == 42
            and full_model["all_packed_bank_contracts_owned_row_major"]
        ),
        "official_16_token_oracle_exact": full_model["official_oracle"][
            "first_16_match"
        ],
        "official_128_token_oracle_exact": full_model["official_oracle"][
            "full_128_match"
        ],
        "startup_peak_at_most_340gb": (
            full_model["startup_peak_bytes"] <= STARTUP_PEAK_LIMIT
        ),
        "startup_peak_regression_at_most_64mib": peak_delta <= 64 * 2**20,
        "ready_time_at_most_190_seconds": (
            full_model["ready_seconds"] <= READY_SECONDS_LIMIT
        ),
    }
    artifact = {
        "schema": "glm53-resident-tensor-ownership-v1",
        "date": date.today().isoformat(),
        "complete": all(acceptance.values()),
        "official_hf_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "ownership_layout_independent": True,
        "runtime_changes": {
            "abi": False,
            "admission": False,
            "backend_policy": False,
            "cache": False,
            "server": False,
        },
        "staging_fixture": staging,
        "packed_fp8_fixture": packed,
        "full_model": full_model,
        "reference_packed_startup_peak_bytes": reference_peak,
        "startup_peak_delta_bytes": peak_delta,
        "acceptance": acceptance,
    }
    _atomic_write(args.output, artifact)
    print(json.dumps({"output": str(args.output), "complete": artifact["complete"]}))
    return 0 if artifact["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
