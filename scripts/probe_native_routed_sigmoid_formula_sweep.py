#!/usr/bin/env python3
"""Sweep AOT sigmoid dtype/intrinsic order over all finite BF16 gate values."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import tempfile
import traceback
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
NATIVE_PACKAGE = ROOT / "native_execution"
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-routed-sigmoid-formula-sweep-20260907.json"
)
SOURCE_ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-routed-hidden-numerical-localization-20260907.json"
)
FORMULAS = (
    "standard_bf16",
    "precise_bf16",
    "standard_f32",
    "precise_f32",
    "fast_bf16",
    "fast_f32",
)


def _atomic_write(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _bf16_domain() -> tuple[mx.array, np.ndarray]:
    bits = np.arange(1 << 16, dtype=np.uint16)
    values = (bits.astype(np.uint32) << 16).view(np.float32)
    keep = np.isfinite(values) & (np.abs(values) <= np.float32(10.0))
    selected_bits = bits[keep]
    # Preserve subnormal and signed-zero payloads exactly.  Constructing via
    # FP32 would flush the 254 BF16 subnormal encodings before the probe runs.
    gate = mx.array(selected_bits).view(mx.bfloat16)
    mx.eval(gate)
    observed_bits = np.ascontiguousarray(
        np.asarray(gate.view(mx.uint16)), dtype=np.uint16
    )
    if not np.array_equal(observed_bits, selected_bits):
        raise RuntimeError("BF16 formula domain did not preserve source bits")
    return gate, selected_bits


def _metrics(reference: mx.array, actual: mx.array) -> dict:
    mx.eval(reference, actual)
    reference_bits = np.ascontiguousarray(
        np.asarray(reference.view(mx.uint16)), dtype=np.uint16
    ).reshape(-1)
    actual_bits = np.ascontiguousarray(
        np.asarray(actual.view(mx.uint16)), dtype=np.uint16
    ).reshape(-1)
    different = np.flatnonzero(reference_bits != actual_bits)
    row = {
        "byte_identical": not bool(different.size),
        "different_elements": int(different.size),
        "reference_hash": hashlib.sha256(reference_bits.tobytes()).hexdigest(),
        "actual_hash": hashlib.sha256(actual_bits.tobytes()).hexdigest(),
        "first_difference": None,
    }
    if different.size:
        index = int(different[0])
        row["first_difference"] = {
            "index": index,
            "reference_bits": f"0x{int(reference_bits[index]):04x}",
            "actual_bits": f"0x{int(actual_bits[index]):04x}",
        }
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = {
        "schema": "glm53-native-routed-sigmoid-formula-sweep-v1",
        "date": date.today().isoformat(),
        "complete": False,
        "accepted": False,
        "probe_only": True,
        "source_artifact": str(SOURCE_ARTIFACT.relative_to(ROOT)),
        "formulas": list(FORMULAS),
        "mlx_version": importlib.metadata.version("mlx"),
        "runtime_changes": {
            "runtime": False,
            "server": False,
            "apc": False,
            "cache_abi": False,
            "production_kernel_abi": False,
        },
    }
    try:
        source = json.loads(SOURCE_ARTIFACT.read_text())
        if not source["accepted"] or source["evidence"][
            "first_differing_stage"
        ] != "sigmoid_bf16":
            raise RuntimeError("source does not establish a routed sigmoid failure")
        for path in (str(SCRIPTS), str(NATIVE_PACKAGE)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from glm53_native_execution import NativeRoutedSigmoidFormulaSweep
        import localize_native_routed_hidden_numerics as localizer

        gate, domain_bits = _bf16_domain()
        ones = mx.ones(gate.shape, dtype=mx.bfloat16)
        reference, reference_silu, reference_hidden = localizer._activation_diagnostics(
            gate, ones
        )
        plan = NativeRoutedSigmoidFormulaSweep(int(gate.size))
        mx.async_eval(gate)
        plan.execute(gate)
        candidates = {name: getattr(plan, name) for name in FORMULAS}
        mx.eval(reference, reference_hidden, *candidates.values())
        rows = {name: _metrics(reference, value) for name, value in candidates.items()}
        exact = [name for name, row in rows.items() if row["byte_identical"]]
        artifact["domain"] = {
            "definition": "all finite BF16 values in closed interval [-10, 10]",
            "elements": int(domain_bits.size),
            "bits_sha256": hashlib.sha256(domain_bits.tobytes()).hexdigest(),
            "minimum_bits": f"0x{int(domain_bits.min()):04x}",
            "maximum_bits": f"0x{int(domain_bits.max()):04x}",
        }
        artifact["reference"] = {
            "execution": "mx.fast JIT literal routed activation ordering",
            "sigmoid_hash": _metrics(reference, reference)["reference_hash"],
            "unit_up_hidden_equals_silu": bool(
                mx.array_equal(reference_silu, reference_hidden).item()
            ),
        }
        artifact["results"] = rows
        artifact["exact_formulas"] = exact
        artifact["selected_formula"] = next(
            (name for name in ("fast_bf16", "fast_f32", "precise_f32") if name in exact),
            exact[0] if exact else None,
        )
        artifact["complete"] = True
        artifact["accepted"] = bool(exact) and artifact["reference"][
            "unit_up_hidden_equals_silu"
        ]
        artifact["decision"] = (
            f"repair_native_routed_sigmoid_with_{artifact['selected_formula']}"
            if artifact["accepted"]
            else "stop_aot_sigmoid_formula_repair"
        )
    except Exception as error:
        artifact.update(
            complete=True,
            accepted=False,
            decision="native_routed_sigmoid_formula_sweep_failed",
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
                "exact_formulas": artifact.get("exact_formulas", []),
            },
            indent=2,
        )
    )
    return 0 if artifact["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
