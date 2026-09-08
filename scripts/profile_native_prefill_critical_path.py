#!/usr/bin/env python3
"""Profile the whole exact prefill dataflow at practical context positions.

The uninstrumented pass is authoritative for wall time and host graph-build
time.  A separate boundary-synchronized pass attributes the same 256-row
chunk across all 45 layers.  Synchronized stage times are diagnostic and are
never added to or substituted for the uninstrumented wall measurement.

Contexts are represented by deterministic compact-cache state, so 32K, 128K,
and 320K positions can be compared without first executing hours of cold
prefill.  This is a profiling artifact, not a production-capacity claim.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

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
    / "m3ultra512-native-prefill-critical-path-20260908.json"
)
PROFILE_CONTEXTS = DESIGN_PREFILL_PROFILE_CONTEXTS
PROFILE_ROWS = DESIGN_PREFILL_QUERY_ROWS
EXPECTED_LAYER_COUNT = 45
EXPECTED_KDA_LAYER_COUNT = 34
EXPECTED_DSA_LAYER_COUNT = 11
EXPECTED_DENSE_FFN_LAYER_COUNT = 3
EXPECTED_ROUTED_MOE_LAYER_COUNT = 42
STAGE_NAMES = (
    "layer_handoff_attention_shell",
    "kda_attention",
    "dsa_attention",
    "post_attention_ffn_shell",
    "dense_ffn",
    "routed_moe",
    "final_norm_lm_head",
)


def _progress(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_boundary_probe():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import probe_long_context_first_decode_boundary as boundary_probe

    return boundary_probe


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


def _hash_array(value: mx.array) -> str:
    mx.eval(value)
    host = np.ascontiguousarray(
        np.asarray(value.astype(mx.float32) if value.dtype == mx.bfloat16 else value)
    )
    return hashlib.sha256(host.tobytes()).hexdigest()


def _temporary_bytes(value) -> int:
    return sum(int(array.nbytes) for array in _arrays(value))


def _module_storage_bytes(module) -> int:
    seen = set()
    total = 0
    for _, value in tree_flatten(module.parameters()):
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        total += int(value.nbytes)
    return total


def _model_storage_inventory(model) -> dict[str, object]:
    stages = defaultdict(int)
    layers = model.language_model.model.layers
    dsa_layers = set(EXPECTED_DSA)
    for layer_index, layer in enumerate(layers):
        stages[
            "dsa_attention" if layer_index in dsa_layers else "kda_attention"
        ] += _module_storage_bytes(layer.self_attn)
        stages["dense_ffn" if layer_index < 3 else "routed_moe"] += (
            _module_storage_bytes(layer.mlp)
        )
    stages["embedding_final_norm_lm_head"] += _module_storage_bytes(
        model.language_model.model.embed_tokens
    )
    stages["embedding_final_norm_lm_head"] += _module_storage_bytes(
        model.language_model.model.norm
    )
    if hasattr(model.language_model, "lm_head"):
        stages["embedding_final_norm_lm_head"] += _module_storage_bytes(
            model.language_model.lm_head
        )
    return {
        "resident_parameter_bytes_by_stage": dict(stages),
        "categorized_parameter_bytes": sum(stages.values()),
        "note": (
            "resident capacity by execution stage, not a claim that every byte "
            "is read for every chunk"
        ),
    }


def _release(*values) -> None:
    for value in values:
        if isinstance(value, (list, dict)):
            value.clear()
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


class _StageRecorder:
    def __init__(self):
        self.samples: dict[str, list[dict[str, object]]] = defaultdict(list)
        self.maximum_absolute_peak_bytes = int(mx.get_peak_memory())

    def _record(
        self,
        stage: str,
        *,
        layer: int | None,
        synchronized_wall_ms: float,
        cpu_graph_build_ms: float,
        output_temporary_bytes: int,
        working_peak_bytes: int,
    ) -> None:
        self.samples[stage].append(
            {
                "layer": layer,
                "synchronized_wall_ms": synchronized_wall_ms,
                "cpu_graph_build_ms": cpu_graph_build_ms,
                "output_temporary_bytes": output_temporary_bytes,
                "working_peak_bytes": working_peak_bytes,
            }
        )

    def boundary(self, stage: str, value, *, layer: int) -> None:
        active = int(mx.get_active_memory())
        mx.reset_peak_memory()
        started = time.perf_counter_ns()
        _eval(value)
        finished = time.perf_counter_ns()
        peak = int(mx.get_peak_memory())
        self.maximum_absolute_peak_bytes = max(
            self.maximum_absolute_peak_bytes, peak
        )
        self._record(
            stage,
            layer=layer,
            synchronized_wall_ms=(finished - started) / 1e6,
            cpu_graph_build_ms=0.0,
            output_temporary_bytes=_temporary_bytes(value),
            working_peak_bytes=max(0, peak - active),
        )

    def call(self, stage: str, delegate, args, kwargs, *, layer: int):
        active = int(mx.get_active_memory())
        mx.reset_peak_memory()
        started = time.perf_counter_ns()
        value = delegate(*args, **kwargs)
        built = time.perf_counter_ns()
        _eval(value)
        finished = time.perf_counter_ns()
        peak = int(mx.get_peak_memory())
        self.maximum_absolute_peak_bytes = max(
            self.maximum_absolute_peak_bytes, peak
        )
        self._record(
            stage,
            layer=layer,
            synchronized_wall_ms=(finished - started) / 1e6,
            cpu_graph_build_ms=(built - started) / 1e6,
            output_temporary_bytes=_temporary_bytes(value),
            working_peak_bytes=max(0, peak - active),
        )
        return value

    def final(self, value) -> None:
        active = int(mx.get_active_memory())
        mx.reset_peak_memory()
        started = time.perf_counter_ns()
        _eval(value)
        finished = time.perf_counter_ns()
        peak = int(mx.get_peak_memory())
        self.maximum_absolute_peak_bytes = max(
            self.maximum_absolute_peak_bytes, peak
        )
        self._record(
            "final_norm_lm_head",
            layer=None,
            synchronized_wall_ms=(finished - started) / 1e6,
            cpu_graph_build_ms=0.0,
            output_temporary_bytes=_temporary_bytes(value),
            working_peak_bytes=max(0, peak - active),
        )

    def summary(self) -> dict[str, object]:
        stages = {}
        diagnostic_total = 0.0
        for stage in STAGE_NAMES:
            samples = self.samples.get(stage, [])
            wall = sum(float(row["synchronized_wall_ms"]) for row in samples)
            diagnostic_total += wall
            stages[stage] = {
                "call_count": len(samples),
                "synchronized_wall_sum_ms": wall,
                "synchronized_wall_median_ms": (
                    statistics.median(
                        float(row["synchronized_wall_ms"]) for row in samples
                    )
                    if samples
                    else 0.0
                ),
                "cpu_graph_build_sum_ms": sum(
                    float(row["cpu_graph_build_ms"]) for row in samples
                ),
                "maximum_working_peak_bytes": max(
                    (int(row["working_peak_bytes"]) for row in samples),
                    default=0,
                ),
                "samples": samples,
            }
        for row in stages.values():
            row["diagnostic_share"] = (
                row["synchronized_wall_sum_ms"] / diagnostic_total
                if diagnostic_total
                else 0.0
            )
        return {
            "stages": stages,
            "synchronized_stage_sum_ms": diagnostic_total,
            "stage_sum_is_uninstrumented_wall": False,
            "timing_note": (
                "each boundary is synchronized independently; use only for "
                "attribution, never as additive production wall"
            ),
            "maximum_absolute_peak_bytes": self.maximum_absolute_peak_bytes,
        }


class _TimedModule:
    def __init__(
        self,
        delegate,
        recorder: _StageRecorder,
        *,
        layer: int,
        boundary_stage: str,
        operator_stage: str,
    ):
        self._delegate = delegate
        self._recorder = recorder
        self._layer = layer
        self._boundary_stage = boundary_stage
        self._operator_stage = operator_stage

    def __call__(self, *args, **kwargs):
        if args:
            self._recorder.boundary(
                self._boundary_stage, args[0], layer=self._layer
            )
        return self._recorder.call(
            self._operator_stage,
            self._delegate,
            args,
            kwargs,
            layer=self._layer,
        )

    def __getattr__(self, name):
        return getattr(self._delegate, name)


@contextlib.contextmanager
def _instrument_prefill(model, recorder: _StageRecorder):
    originals = []
    dsa_layers = set(EXPECTED_DSA)
    layers = model.language_model.model.layers
    try:
        for layer_index, layer in enumerate(layers):
            attention = layer.self_attn
            mlp = layer.mlp
            originals.append((layer, attention, mlp))
            layer.self_attn = _TimedModule(
                attention,
                recorder,
                layer=layer_index,
                boundary_stage="layer_handoff_attention_shell",
                operator_stage=(
                    "dsa_attention" if layer_index in dsa_layers else "kda_attention"
                ),
            )
            layer.mlp = _TimedModule(
                mlp,
                recorder,
                layer=layer_index,
                boundary_stage="post_attention_ffn_shell",
                operator_stage=("dense_ffn" if layer_index < 3 else "routed_moe"),
            )
        yield
    finally:
        for layer, attention, mlp in originals:
            layer.self_attn = attention
            layer.mlp = mlp


def _tokens(rows: int, vocab_size: int, context: int) -> mx.array:
    values = (
        mx.arange(rows, dtype=mx.uint32) * 7_919 + context + 101
    ) % (vocab_size - 1_024)
    return (values + 128)[None]


def _state_signature(boundary_probe, cache) -> dict[str, object]:
    return {
        "diagnostic_state_sha256": boundary_probe._post_state_hash(
            cache, "compact-nope-dsa"
        ),
        "diagnostic_state_coverage": (
            "all KDA and IndexPool values, all offsets, and first/last 64 "
            "latent rows per DSA layer"
        ),
        "dsa_offsets": boundary_probe._dsa_offsets(
            cache, "compact-nope-dsa"
        ),
        "cache_leaf_count": boundary_probe._cache_leaf_count(cache),
    }


def _run_uninstrumented(model, cache, token_ids, boundary_probe) -> dict[str, object]:
    active = int(mx.get_active_memory())
    mx.reset_peak_memory()
    started = time.perf_counter_ns()
    output = model(token_ids, cache=cache)
    built = time.perf_counter_ns()
    final_logits = output.logits[0, -1]
    _eval(final_logits)
    finished = time.perf_counter_ns()
    result = {
        "host_graph_build_ms": (built - started) / 1e6,
        "wall_ms": (finished - started) / 1e6,
        "tokens_per_second": (
            int(token_ids.shape[1]) * 1e9 / (finished - started)
        ),
        "working_peak_bytes": max(0, int(mx.get_peak_memory()) - active),
        "absolute_peak_bytes": int(mx.get_peak_memory()),
        "final_logits_sha256": _hash_array(final_logits),
        "state": _state_signature(boundary_probe, cache),
    }
    del output, final_logits
    return result


def _run_instrumented(model, cache, token_ids, boundary_probe) -> dict[str, object]:
    recorder = _StageRecorder()
    started = time.perf_counter_ns()
    with _instrument_prefill(model, recorder):
        output = model(token_ids, cache=cache)
    final_logits = output.logits[0, -1]
    recorder.final(final_logits)
    finished = time.perf_counter_ns()
    result = {
        "diagnostic_wall_ms": (finished - started) / 1e6,
        "final_logits_sha256": _hash_array(final_logits),
        "state": _state_signature(boundary_probe, cache),
        "attribution": recorder.summary(),
    }
    del output, final_logits
    return result


def _summarize_uninstrumented(samples: list[dict[str, object]]) -> dict[str, object]:
    return {
        "sample_count": len(samples),
        "median_wall_ms": statistics.median(float(row["wall_ms"]) for row in samples),
        "median_host_graph_build_ms": statistics.median(
            float(row["host_graph_build_ms"]) for row in samples
        ),
        "median_tokens_per_second": statistics.median(
            float(row["tokens_per_second"]) for row in samples
        ),
        "maximum_working_peak_bytes": max(
            int(row["working_peak_bytes"]) for row in samples
        ),
        "maximum_absolute_peak_bytes": max(
            int(row["absolute_peak_bytes"]) for row in samples
        ),
        "repeat_logits_exact": len(
            {str(row["final_logits_sha256"]) for row in samples}
        )
        == 1,
        "repeat_state_exact": len(
            {str(row["state"]["diagnostic_state_sha256"]) for row in samples}
        )
        == 1,
        "samples": samples,
    }


def _context_case(model, boundary_probe, context: int, rows: int, samples: int):
    prefix = context - rows
    if prefix <= 0:
        raise ValueError("profile context must exceed the representative chunk")
    _progress("build_synthetic_prefix", context=context, prefix=prefix)
    mx.reset_peak_memory()
    source = boundary_probe._synthetic_cache(
        model, prefix, "compact-nope-dsa"
    )
    source_build_peak = int(mx.get_peak_memory())
    token_ids = _tokens(rows, model.language_model.vocab_size, context)
    uninstrumented_samples = []
    for sample in range(samples):
        _progress("uninstrumented_chunk", context=context, sample=sample + 1)
        cache = boundary_probe._clone_cache(source, context)
        row = _run_uninstrumented(model, cache, token_ids, boundary_probe)
        uninstrumented_samples.append(row)
        _release(cache)
    baseline = _summarize_uninstrumented(uninstrumented_samples)

    _progress("synchronized_stage_attribution", context=context)
    profiled_cache = boundary_probe._clone_cache(source, context)
    profiled = _run_instrumented(
        model, profiled_cache, token_ids, boundary_probe
    )
    _release(profiled_cache, source)
    baseline_signature = uninstrumented_samples[0]
    exactness = {
        "instrumentation_final_logits_exact": (
            profiled["final_logits_sha256"]
            == baseline_signature["final_logits_sha256"]
        ),
        "instrumentation_diagnostic_state_exact": (
            profiled["state"]["diagnostic_state_sha256"]
            == baseline_signature["state"]["diagnostic_state_sha256"]
        ),
        "instrumentation_dsa_offsets_exact": (
            profiled["state"]["dsa_offsets"]
            == baseline_signature["state"]["dsa_offsets"]
        ),
    }
    stages = profiled["attribution"]["stages"]
    dominant = max(
        STAGE_NAMES,
        key=lambda name: float(stages[name]["synchronized_wall_sum_ms"]),
    )
    return {
        "context_tokens_after_chunk": context,
        "synthetic_prefix_tokens": prefix,
        "representative_chunk_tokens": rows,
        "synthetic_source_peak_bytes": source_build_peak,
        "uninstrumented": baseline,
        "instrumented": profiled,
        "exactness": exactness,
        "dominant_synchronized_stage": dominant,
    }


def _acceptance(artifact: dict[str, object]) -> dict[str, bool]:
    contexts = artifact.get("contexts", {})
    complete = set(map(int, contexts)) == set(PROFILE_CONTEXTS)
    rows = list(contexts.values())
    expected_calls = {
        "layer_handoff_attention_shell": EXPECTED_LAYER_COUNT,
        "kda_attention": EXPECTED_KDA_LAYER_COUNT,
        "dsa_attention": EXPECTED_DSA_LAYER_COUNT,
        "post_attention_ffn_shell": EXPECTED_LAYER_COUNT,
        "dense_ffn": EXPECTED_DENSE_FFN_LAYER_COUNT,
        "routed_moe": EXPECTED_ROUTED_MOE_LAYER_COUNT,
        "final_norm_lm_head": 1,
    }
    return {
        "32k_128k_320k_representative_chunks_measured": complete,
        "uninstrumented_wall_and_host_build_recorded": complete
        and all(
            row["uninstrumented"]["median_wall_ms"] > 0
            and row["uninstrumented"]["median_host_graph_build_ms"] > 0
            for row in rows
        ),
        "profile_repeats_are_exact": complete
        and all(
            row["uninstrumented"]["repeat_logits_exact"]
            and row["uninstrumented"]["repeat_state_exact"]
            for row in rows
        ),
        "instrumentation_logits_state_offsets_exact": complete
        and all(all(row["exactness"].values()) for row in rows),
        "all_45_layers_and_final_boundary_attributed": complete
        and all(
            all(
                row["instrumented"]["attribution"]["stages"][stage][
                    "call_count"
                ]
                == count
                for stage, count in expected_calls.items()
            )
            for row in rows
        ),
        "synchronized_stage_sum_not_claimed_as_production_wall": complete
        and all(
            row["instrumented"]["attribution"][
                "stage_sum_is_uninstrumented_wall"
            ]
            is False
            for row in rows
        ),
        "process_peak_at_most_340gb": int(
            artifact.get("process_peak_memory_bytes", 1 << 62)
        )
        <= 340_000_000_000,
        "runtime_server_cache_abi_admission_unchanged": artifact.get(
            "runtime_changes"
        )
        == {
            "admission": False,
            "backend": False,
            "cache_abi": False,
            "kernel_abi": False,
            "server": False,
        },
    }


def _decision(artifact: dict[str, object]) -> str:
    if not all(artifact.get("acceptance", {}).values()):
        return "profiling_incomplete_or_invalid"
    long = artifact["contexts"][str(320 << 10)]
    dominant = long["dominant_synchronized_stage"]
    return f"design_native_prefill_plan_around_{dominant}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=list(PROFILE_CONTEXTS)
    )
    parser.add_argument("--rows", type=int, default=PROFILE_ROWS)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--wired-limit-gb", type=float, default=440.0)
    parser.add_argument("--cache-limit-gb", type=float, default=32.0)
    args = parser.parse_args(argv)
    if args.rows != PROFILE_ROWS:
        raise ValueError("the authoritative profile chunk is fixed at 256 rows")
    if any(context not in PROFILE_CONTEXTS for context in args.contexts):
        raise ValueError("contexts must be selected from 32K, 128K, and 320K")
    if args.samples < 2:
        raise ValueError("at least two uninstrumented samples are required")

    report = inspect_checkpoint(args.model, require_server_ready=True)
    boundary_probe = _load_boundary_probe()
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
    storage_inventory = _model_storage_inventory(model)
    artifact = None
    if args.output.exists():
        candidate = json.loads(args.output.read_text())
        if candidate.get("schema") == "glm53-native-prefill-critical-path-v1":
            if candidate.get("checkpoint_fingerprint") != report.fingerprint:
                raise ValueError("existing artifact uses another checkpoint")
            artifact = candidate
    if artifact is None:
        artifact = {
            "schema": "glm53-native-prefill-critical-path-v1",
            "date": date.today().isoformat(),
            "complete": False,
            "accepted": False,
            "decision": "profiling_incomplete_or_invalid",
            "profiling_only": True,
            "checkpoint_fingerprint": report.fingerprint,
            "official_hf_revision": report.official_revision,
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_vlm_revision": MLX_VLM_REVISION,
            "capacity_contract": NATIVE_CONTEXT_CAPACITY_CONTRACT,
            "compact_cache_abi": NOPE_DSA_CACHE_ABI_COMPACT,
            "contexts_requested": list(PROFILE_CONTEXTS),
            "representative_chunk_tokens": PROFILE_ROWS,
            "measurement_contract": {
                "uninstrumented_wall_is_authoritative": True,
                "stage_attribution_is_separate_synchronized_pass": True,
                "stage_sum_is_production_wall": False,
                "synthetic_prefix_state_avoids_hours_of_cold_prefill": True,
                "profile_goal_tokens_per_second": None,
                "100_tps_is_a_checkpoint_not_a_stop_condition": True,
            },
            "model_storage_inventory": storage_inventory,
            "contexts": {},
            "process_peak_memory_bytes": 0,
            "runtime_changes": {
                "admission": False,
                "backend": False,
                "cache_abi": False,
                "kernel_abi": False,
                "server": False,
            },
        }
    else:
        artifact["model_storage_inventory"] = storage_inventory

    for context in args.contexts:
        key = str(context)
        if key in artifact["contexts"] and not args.rerun:
            _progress("skip_completed_context", context=context)
            continue
        try:
            artifact["contexts"][key] = _context_case(
                model, boundary_probe, context, args.rows, args.samples
            )
        except Exception as error:
            artifact["failure"] = {
                "context": context,
                "type": type(error).__name__,
                "message": str(error),
            }
            artifact["acceptance"] = _acceptance(artifact)
            artifact["complete"] = False
            artifact["accepted"] = False
            artifact["decision"] = "profiling_incomplete_or_invalid"
            _atomic_write(args.output, artifact)
            raise
        artifact.pop("failure", None)
        context_peak = max(
            artifact["contexts"][key]["synthetic_source_peak_bytes"],
            artifact["contexts"][key]["uninstrumented"][
                "maximum_absolute_peak_bytes"
            ],
            artifact["contexts"][key]["instrumented"]["attribution"][
                "maximum_absolute_peak_bytes"
            ],
        )
        artifact["process_peak_memory_bytes"] = max(
            int(artifact["process_peak_memory_bytes"]), int(context_peak)
        )
        artifact["acceptance"] = _acceptance(artifact)
        artifact["complete"] = all(artifact["acceptance"].values())
        artifact["accepted"] = artifact["complete"]
        artifact["decision"] = _decision(artifact)
        _atomic_write(args.output, artifact)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": artifact["complete"],
                "accepted": artifact["accepted"],
                "decision": artifact["decision"],
                "completed_contexts": sorted(map(int, artifact["contexts"])),
            }
        )
    )
    return 0 if artifact["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
