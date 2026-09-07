#!/usr/bin/env python3
"""Qualify the native IndexPool backend on a real 32K coding-agent HTTP flow.

Run ``baseline`` against packed-decode + compact-cache + APC, then restart the
server with the additional native IndexPool flag and run ``native``.  The two
phases share one artifact but never keep two 320 GB model processes resident.
Each arm performs one cold 32K request, a tool-result suffix request which must
reuse the guarded base prefix, and an exact warm repeat of that extended turn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
import urllib.error
from datetime import date
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import qualify_coding_agent_prefix_cache_admission as fixture_helpers  # noqa: E402
import qualify_production_coding_agent_admission as http_helpers  # noqa: E402

from glm53_flash_mlx.native_indexpool_runtime import (  # noqa: E402
    NATIVE_INDEXPOOL_RUNTIME_ABI,
)
from glm53_flash_mlx.server import (  # noqa: E402
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_GENERATION_TOKENS,
    EXACT_APC_PREFIX_GUARD_TOKENS,
    admission_snapshot,
)


DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-coding-agent-http-20260908.json"
)
NATIVE_RUNTIME_SOURCE = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-indexpool-runtime-20260907.json"
)
SCHEMA = "glm53-native-coding-agent-http-v1"
CONTEXT_TOKENS = 32_768
DECODE_TOKENS = 256
PREFILL_ALIGNMENT = 2_048
EXPECTED_DSA_LAYERS = 11
MAX_PEAK_BYTES = 340_000_000_000
REQUEST_NAMES = ("cold_base", "tool_suffix", "tool_suffix_warm")


class QualificationPreconditionError(RuntimeError):
    """Raised before starting a long HTTP request against a wrong server."""


def _progress(phase: str, **values: Any) -> None:
    print(json.dumps({"phase": phase, **values}, sort_keys=True), flush=True)


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token_sha256(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _source_evidence() -> dict[str, Any]:
    source = json.loads(NATIVE_RUNTIME_SOURCE.read_text())
    if not source.get("complete") or not source.get("accepted"):
        raise QualificationPreconditionError(
            "native IndexPool runtime source qualification is not accepted"
        )
    if not source.get("release_15_tps_accepted"):
        raise QualificationPreconditionError(
            "native IndexPool runtime source did not pass 15 tok/s"
        )
    if source.get("native_runtime_abi") != NATIVE_INDEXPOOL_RUNTIME_ABI:
        raise QualificationPreconditionError(
            "native IndexPool source qualification has a different runtime ABI"
        )
    return {
        "path": str(NATIVE_RUNTIME_SOURCE.relative_to(ROOT)),
        "sha256": _sha256(NATIVE_RUNTIME_SOURCE),
        "decision": source.get("decision"),
        "native_runtime_abi": source.get("native_runtime_abi"),
    }


def _checkpoint_identity(model: Path) -> dict[str, str]:
    from glm53_flash_mlx.manifest import inspect_checkpoint

    report = inspect_checkpoint(model, require_server_ready=True)
    return {
        "fingerprint": report.fingerprint,
        "official_revision": report.official_revision,
    }


def _initial_artifact(model: Path) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "date": date.today().isoformat(),
        "complete": False,
        "accepted": False,
        "decision": "qualification_incomplete",
        "model_path": str(model),
        "checkpoint": _checkpoint_identity(model),
        "source_native_runtime": _source_evidence(),
        "configuration": {
            "context_tokens": CONTEXT_TOKENS,
            "decode_tokens": DECODE_TOKENS,
            "prefill_alignment_tokens": PREFILL_ALIGNMENT,
            "expected_dsa_layers": EXPECTED_DSA_LAYERS,
            "server_processes_are_separate": True,
        },
        "phases": {},
        "runtime_changes": {
            "inference_execution": False,
            "runtime_telemetry": True,
            "server_telemetry": True,
            "cache_abi": False,
            "admission": False,
            "default_backend": False,
        },
    }


def _load_artifact(path: Path, model: Path) -> dict[str, Any]:
    fresh = _initial_artifact(model)
    if not path.exists():
        return fresh
    artifact = json.loads(path.read_text())
    if artifact.get("schema") != SCHEMA:
        raise QualificationPreconditionError(
            "existing output has a different qualification schema"
        )
    if artifact.get("model_path") != str(model):
        raise QualificationPreconditionError(
            "existing output belongs to a different model"
        )
    if artifact.get("checkpoint") != fresh["checkpoint"]:
        raise QualificationPreconditionError(
            "existing output belongs to a different checkpoint identity"
        )
    artifact.update(
        date=fresh["date"], source_native_runtime=fresh["source_native_runtime"]
    )
    return artifact


def _build_fixture(model: Path):
    from mlx_vlm.tokenizer_utils import load_tokenizer

    tokenizer = load_tokenizer(model)._tokenizer
    corpus = fixture_helpers._repository_corpus(ROOT)
    base_messages, base_count = http_helpers.build_exact_http_messages(
        tokenizer, corpus, CONTEXT_TOKENS
    )
    extended_messages = http_helpers._extended_messages(base_messages)
    base_ids = fixture_helpers._render_ids(tokenizer, base_messages)
    extended_ids = fixture_helpers._render_ids(tokenizer, extended_messages)
    if base_count != CONTEXT_TOKENS or len(base_ids) != CONTEXT_TOKENS:
        raise RuntimeError("coding-agent HTTP fixture is not exactly 32K tokens")
    return base_messages, extended_messages, {
        "base_prompt_tokens": len(base_ids),
        "base_prompt_token_sha256": _token_sha256(base_ids),
        "extended_prompt_tokens": len(extended_ids),
        "extended_prompt_token_sha256": _token_sha256(extended_ids),
        "repository_corpus_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
        "has_system_prompt": True,
        "has_tool_definitions": True,
        "has_repository_context": True,
        "has_conversation_history": True,
        "has_tool_call_and_result_suffix": True,
    }


def _post(args, path: str, payload: dict[str, Any] | None = None):
    return http_helpers._http_json(
        f"{args.server_url}{path}", payload=payload, timeout=args.http_timeout
    )


def _metrics(args) -> dict[str, Any]:
    status, value = _post(args, "/v1/metrics")
    if status != 200:
        raise QualificationPreconditionError(f"metrics endpoint returned HTTP {status}")
    return value


def _expected_cached_tokens(prompt_tokens: int) -> int:
    safe = int(prompt_tokens) - EXACT_APC_PREFIX_GUARD_TOKENS
    return max(0, safe // PREFILL_ALIGNMENT) * PREFILL_ALIGNMENT


def _preflight(args, phase: str) -> dict[str, Any]:
    health_status, health = _post(args, "/health")
    metrics = _metrics(args)
    if health_status != 200 or health.get("apc_enabled") is not True:
        raise QualificationPreconditionError(
            "qualification requires a healthy server started with --apc"
        )
    runtime = metrics.get("server", {}).get("glm53_runtime")
    expected = {
        "moe_backend": "packed-decode",
        "cache_backend": "compact-nope-dsa",
        "native_indexpool_update": phase == "native",
    }
    if runtime != expected:
        raise QualificationPreconditionError(
            f"wrong server backend for {phase}: expected {expected}, got {runtime}"
        )
    if metrics.get("server", {}).get("admission") != admission_snapshot(
        max_generation_tokens=DEFAULT_MAX_GENERATION_TOKENS,
        max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
    ):
        raise QualificationPreconditionError(
            "server is not using the qualified prompt+generation admission"
        )
    native = metrics.get("server", {}).get("native_indexpool_update")
    if phase == "native":
        if not isinstance(native, dict) or native.get("abi") != NATIVE_INDEXPOOL_RUNTIME_ABI:
            raise QualificationPreconditionError(
                "native server does not report the qualified runtime ABI"
            )
    elif native is not None:
        raise QualificationPreconditionError(
            "baseline server unexpectedly has native IndexPool enabled"
        )
    return {"health": health, "metrics_server": metrics.get("server", {})}


def _payload(messages: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": "glm-5.3-flash",
        "messages": list(messages),
        "tools": fixture_helpers._tools(),
        "temperature": 0,
        "seed": 0,
        "max_tokens": DECODE_TOKENS,
        "logprobs": True,
        "top_logprobs": 5,
        "stream": False,
    }


def _request(args, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    _progress("http_request", arm=args.phase, request=name)
    before = _metrics(args)
    started = time.perf_counter()
    status, response = _post(args, "/v1/chat/completions", payload)
    elapsed = time.perf_counter() - started
    if status != 200:
        raise RuntimeError(f"{name} returned HTTP {status}: {response}")
    after = _metrics(args)
    signature = http_helpers._choice_signature(response)
    before_native = before.get("server", {}).get("native_indexpool_update", {})
    after_native = after.get("server", {}).get("native_indexpool_update", {})
    before_exec = int(before_native.get("cumulative_execution_count", 0))
    after_exec = int(after_native.get("cumulative_execution_count", 0))
    return {
        "elapsed_seconds": elapsed,
        "usage": response.get("usage", {}),
        "cached_tokens": http_helpers._cached_tokens(response),
        "choice_sha256": _json_sha256(signature),
        "choice_signature": signature,
        "metrics": after.get("latest"),
        "native_execution_count_before": before_exec,
        "native_execution_count_after": after_exec,
        "native_execution_count_delta": after_exec - before_exec,
        "native_registry": after_native or None,
    }


def _phase_local_checks(phase: str, row: dict[str, Any]) -> dict[str, bool]:
    requests = row["requests"]
    cold = requests["cold_base"]
    suffix = requests["tool_suffix"]
    warm = requests["tool_suffix_warm"]
    extended_tokens = int(suffix["usage"].get("prompt_tokens", 0))
    full_decode = all(
        int(requests[name]["usage"].get("completion_tokens", 0)) == DECODE_TOKENS
        for name in REQUEST_NAMES
    )
    expected_executes = sum(
        max(0, int(requests[name]["usage"].get("completion_tokens", 0)) - 1)
        for name in REQUEST_NAMES
    ) * EXPECTED_DSA_LAYERS
    observed_executes = sum(
        requests[name]["native_execution_count_delta"] for name in REQUEST_NAMES
    )
    peak_gb = max(
        float((requests[name].get("metrics") or {}).get("peak_memory_gb") or 0.0)
        for name in REQUEST_NAMES
    )
    cache_stats = row["cache_stats"]
    checks = {
        "cold_prompt_is_exactly_32k": int(cold["usage"].get("prompt_tokens", 0))
        == CONTEXT_TOKENS,
        "cold_request_has_no_prefix_hit": cold["cached_tokens"] == 0,
        "tool_suffix_reuses_guarded_32k_prefix": suffix["cached_tokens"]
        == _expected_cached_tokens(CONTEXT_TOKENS),
        "warm_tool_suffix_hit_is_exact": warm["cached_tokens"]
        == _expected_cached_tokens(extended_tokens),
        "tool_suffix_warm_output_is_exact": suffix["choice_sha256"]
        == warm["choice_sha256"],
        "all_requests_complete_full_256_decode": full_decode,
        "all_requests_stay_within_context_capacity": all(
            int(requests[name]["usage"].get("total_tokens", 0))
            <= DEFAULT_MAX_CONTEXT_TOKENS
            for name in REQUEST_NAMES
        ),
        "request_metrics_are_present": all(
            isinstance(requests[name].get("metrics"), dict)
            and float(requests[name]["metrics"].get("decode_tok_s") or 0.0) > 0
            for name in REQUEST_NAMES
        ),
        "peak_is_at_most_340gb": peak_gb * 1e9 <= MAX_PEAK_BYTES,
        "apc_reports_both_exact_prefix_hits": int(
            cache_stats.get("exact_hits", 0)
        )
        >= 2,
        "apc_has_no_reject_or_eviction": (
            int(cache_stats.get("rejects", 0)) == 0
            and int(cache_stats.get("evictions", 0)) == 0
        ),
        "native_execution_count_matches_decode_for_all_11_dsa_layers": (
            observed_executes == expected_executes
            if phase == "native"
            else observed_executes == 0
        ),
    }
    row["resource"] = {
        "peak_memory_gb": peak_gb,
        "expected_native_execution_count": expected_executes if phase == "native" else 0,
        "observed_native_execution_count": observed_executes,
    }
    return checks


def _cross_arm_checks(artifact: dict[str, Any]) -> dict[str, bool]:
    baseline = artifact["phases"]["baseline"]
    native = artifact["phases"]["native"]
    base_requests = baseline["requests"]
    native_requests = native["requests"]
    native_warm_metrics = native_requests["tool_suffix_warm"]["metrics"] or {}
    return {
        "fixture_identity_exact": baseline["fixture"] == native["fixture"],
        "all_http_choice_signatures_exact_across_backends": all(
            base_requests[name]["choice_sha256"]
            == native_requests[name]["choice_sha256"]
            for name in REQUEST_NAMES
        ),
        "all_usage_and_prefix_hits_exact_across_backends": all(
            base_requests[name]["usage"] == native_requests[name]["usage"]
            and base_requests[name]["cached_tokens"]
            == native_requests[name]["cached_tokens"]
            for name in REQUEST_NAMES
        ),
        "native_warm_tool_suffix_decode_at_least_15_tps": float(
            native_warm_metrics.get("decode_tok_s") or 0.0
        )
        >= 15.0,
        "native_runtime_source_remains_accepted": artifact[
            "source_native_runtime"
        ]["decision"]
        == "keep_native_indexpool_runtime_and_pass_15_tps",
    }


def _finish(artifact: dict[str, Any]) -> None:
    phases = artifact.get("phases", {})
    phase_complete = all(
        phases.get(name, {}).get("accepted") is True for name in ("baseline", "native")
    )
    if phase_complete:
        artifact["cross_arm_checks"] = _cross_arm_checks(artifact)
    artifact["complete"] = phase_complete
    artifact["accepted"] = phase_complete and all(
        artifact.get("cross_arm_checks", {}).values()
    )
    artifact["decision"] = (
        "qualify_native_backend_for_real_coding_agent_http"
        if artifact["accepted"]
        else "stop_or_requalify_native_coding_agent_http"
        if artifact["complete"]
        else "qualification_incomplete"
    )


def _run_phase(args, artifact: dict[str, Any]) -> None:
    if args.phase == "native" and artifact.get("phases", {}).get(
        "baseline", {}
    ).get("accepted") is not True:
        raise QualificationPreconditionError(
            "native phase requires an accepted baseline phase first"
        )
    if (
        not args.force_phase
        and artifact.get("phases", {}).get(args.phase, {}).get("accepted") is True
    ):
        _progress("skip_accepted_phase", arm=args.phase)
        return

    # A newly measured baseline must never be paired with a native result from
    # an older server/process run.  The native phase remains independently
    # resumable, but it must follow the currently accepted baseline.
    if args.phase == "baseline":
        artifact.setdefault("phases", {}).pop("native", None)

    preflight = _preflight(args, args.phase)
    base_messages, extended_messages, fixture = _build_fixture(args.model)
    phase_row = {
        "complete": False,
        "accepted": False,
        "started_unix": time.time(),
        "preflight": preflight,
        "fixture": fixture,
        "requests": {},
    }
    artifact.setdefault("phases", {})[args.phase] = phase_row
    _atomic_write(args.output, artifact)

    reset_status, reset = _post(args, "/v1/cache/reset", {})
    if reset_status != 200:
        raise RuntimeError(f"cache reset returned HTTP {reset_status}: {reset}")
    payloads = {
        "cold_base": _payload(base_messages),
        "tool_suffix": _payload(extended_messages),
        "tool_suffix_warm": _payload(extended_messages),
    }
    for name in REQUEST_NAMES:
        phase_row["requests"][name] = _request(args, name, payloads[name])
        _atomic_write(args.output, artifact)

    stats_status, stats = _post(args, "/v1/cache/stats")
    if stats_status != 200:
        raise RuntimeError(f"cache stats returned HTTP {stats_status}")
    phase_row["cache_stats"] = stats
    phase_row["checks"] = _phase_local_checks(args.phase, phase_row)
    phase_row["finished_unix"] = time.time()
    phase_row["complete"] = True
    phase_row["accepted"] = all(phase_row["checks"].values())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=("baseline", "native"), required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--http-timeout", type=float, default=1800.0)
    parser.add_argument("--force-phase", action="store_true")
    args = parser.parse_args(argv)
    args.model = args.model.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    try:
        artifact = _load_artifact(args.output, args.model)
        _run_phase(args, artifact)
        artifact.pop("last_error", None)
        _finish(artifact)
        _atomic_write(args.output, artifact)
    except QualificationPreconditionError as error:
        print(f"qualification precondition: {error}", file=sys.stderr)
        return 2
    except urllib.error.URLError as error:
        print(
            f"qualification precondition: cannot reach {args.server_url}: "
            f"{error.reason}",
            file=sys.stderr,
        )
        return 2
    except Exception as error:
        if "artifact" in locals():
            failure = {
                "phase": args.phase,
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            artifact["last_error"] = failure
            artifact.setdefault("phase_failures", []).append(failure)
            _finish(artifact)
            _atomic_write(args.output, artifact)
        raise

    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": artifact["complete"],
                "accepted": artifact["accepted"],
                "decision": artifact["decision"],
                "completed_phases": sorted(
                    name
                    for name, row in artifact["phases"].items()
                    if row.get("complete")
                ),
            },
            indent=2,
        )
    )
    return 0 if artifact["accepted"] or not artifact["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
