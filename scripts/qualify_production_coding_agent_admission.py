#!/usr/bin/env python3
"""Qualify promoted coding-agent admission through model and HTTP paths.

Run the model phase and HTTP phase in separate processes so the 320 GB model
is never resident twice.  Progress is atomically merged into one artifact.
The model phase proves a 32K prefix plus 4,096 exact continuation.  The HTTP
phase proves cold/warm OpenAI-compatible coding-agent turns at 8K/16K/32K and
the production total-context rejection boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import qualify_coding_agent_prefix_cache_admission as prior  # noqa: E402

from glm53_flash_mlx.server import (  # noqa: E402
    ADMISSION_POLICY,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_FIRST_TOKEN_TIMEOUT_SECONDS,
    DEFAULT_MAX_GENERATION_TOKENS,
    EXACT_APC_PREFIX_GUARD_TOKENS,
    EXACT_APC_STORE_POLICY,
    QUALIFIED_PROMPT_TOKENS,
    admission_snapshot,
    validate_admission,
)


DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-production-coding-agent-admission-20260905.json"
)
HTTP_CONTEXTS = (8192, 16384, 32768)
HTTP_DECODE_TOKENS = 256
HTTP_PREFILL_ALIGNMENT_TOKENS = 2048
MODEL_PREFIX_TOKENS = 32768
MODEL_CONTINUATION_TOKENS = 4096
MATERIALIZATION_INTERVAL = 256
MAX_PEAK_BYTES = 340_000_000_000
SCHEMA = "glm53-production-coding-agent-admission-v1"


class QualificationPreconditionError(RuntimeError):
    """Raised before expensive work when the phase prerequisites are absent."""


def _progress(phase: str, **values: Any) -> None:
    print(json.dumps({"phase": phase, **values}, sort_keys=True), flush=True)


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _initial_artifact(model: Path) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "date": date.today().isoformat(),
        "complete": False,
        "accepted": False,
        "model_path": str(model),
        "admission": admission_snapshot(
            max_generation_tokens=DEFAULT_MAX_GENERATION_TOKENS,
            max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
        ),
        "model_32k_4096": {"completed": False},
        "http": {"completed": False, "cases": {}},
    }


def _load_artifact(path: Path, model: Path) -> dict[str, Any]:
    if not path.exists():
        return _initial_artifact(model)
    artifact = json.loads(path.read_text())
    if artifact.get("schema") != SCHEMA:
        raise ValueError("existing artifact has a different schema")
    if artifact.get("model_path") != str(model):
        raise ValueError("existing artifact belongs to a different model")
    return artifact


def _finish(artifact: dict[str, Any]) -> None:
    model_ok = artifact.get("model_32k_4096", {}).get("accepted") is True
    http_ok = artifact.get("http", {}).get("accepted") is True
    artifact["acceptance"] = {
        "production_policy_is_prompt_plus_generation": (
            artifact["admission"]["policy"] == ADMISSION_POLICY
        ),
        "context_capacity_is_36864": (
            artifact["admission"]["max_context_tokens"] == 36864
        ),
        "generation_cap_is_4096": (
            artifact["admission"]["max_generation_tokens"] == 4096
        ),
        "32k_plus_4096_model_exact": model_ok,
        "http_8k_16k_32k_exact": http_ok,
    }
    artifact["complete"] = model_ok and http_ok
    artifact["accepted"] = artifact["complete"] and all(
        artifact["acceptance"].values()
    )


def _rolling_update(digest, value: str) -> None:
    digest.update(bytes.fromhex(value))


def _run_continuation(model, cache, first_token: int, steps: int) -> dict[str, Any]:
    import mlx.core as mx

    token = int(first_token)
    token_digest = hashlib.sha256()
    logits_digest = hashlib.sha256()
    evidence: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    materializations = 0
    nan_count = 0
    for step in range(1, steps + 1):
        output = model(mx.array([[token]], dtype=mx.uint32), cache=cache)
        logits = output.logits[0, -1]
        predicted = mx.argmax(logits)
        step_nan = mx.sum(mx.isnan(logits))
        mx.eval(logits, predicted, step_nan)
        logits_hash = prior._array_digest(logits)
        token = int(predicted.item())
        nan_count += int(step_nan.item())
        token_digest.update(token.to_bytes(4, "little", signed=False))
        _rolling_update(logits_digest, logits_hash)
        if step % MATERIALIZATION_INTERVAL == 0:
            mx.eval([entry.state for entry in cache])
            mx.clear_cache()
            mx.synchronize()
            materializations += 1
            evidence[str(step)] = {
                "token_id": token,
                "logits_sha256": logits_hash,
                "state": prior._snapshot_digest(cache),
            }
            _progress(
                "model_continuation",
                step=step,
                steps=steps,
                materializations=materializations,
            )
    mx.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "steps": steps,
        "elapsed_seconds": elapsed,
        "tokens_per_second": steps / elapsed,
        "token_sha256": token_digest.hexdigest(),
        "all_step_logits_sha256": logits_digest.hexdigest(),
        "evidence": evidence,
        "materializations": materializations,
        "nan_count": nan_count,
        "final_state": prior._snapshot_digest(cache),
        "final_accounting": prior._cache_accounting(cache, snapshot=False),
    }


def _run_model_phase(args, artifact: dict[str, Any]) -> None:
    if artifact.get("model_32k_4096", {}).get("accepted") is True:
        _progress("skip_model_32k_4096")
        return

    import mlx.core as mx
    from mlx_vlm.apc import APCManager, model_apc_mode

    from glm53_flash_mlx.loader import load, warm_residency
    from glm53_flash_mlx.manifest import inspect_checkpoint

    mx.set_wired_limit(args.wired_limit_bytes)
    mx.set_cache_limit(args.cache_limit_bytes)
    report = inspect_checkpoint(args.model, require_server_ready=True)
    model, tokenizer = load(args.model)
    warm_residency(model)
    if model_apc_mode(model.language_model) != "exact":
        raise RuntimeError("GLM hybrid cache must use exact APC mode")
    corpus = prior._repository_corpus(REPOSITORY)
    fixture = prior.build_coding_agent_fixture(
        tokenizer, MODEL_PREFIX_TOKENS, corpus
    )
    prefix_ids = fixture.pop("prefix_token_ids")
    cache = model.make_cache()
    _progress("model_32k_prefill")
    cold = prior._prefill(
        model,
        cache,
        prefix_ids,
        chunk_tokens=prior.PREFILL_CHUNK_TOKENS,
    )
    prefix_state = prior._snapshot_digest(cache)
    manager = APCManager(num_blocks=1, block_size=64)
    if not manager.store_exact_cache(prefix_ids, cache):
        raise RuntimeError("failed to store exact 32K prefix")
    baseline = _run_continuation(
        model,
        cache,
        cold["predicted_token_id"],
        MODEL_CONTINUATION_TOKENS,
    )

    restored, prefix_hit = manager.lookup_exact_cache(
        prefix_ids + [cold["predicted_token_id"]]
    )
    if restored is None or prefix_hit != MODEL_PREFIX_TOKENS:
        raise RuntimeError("failed to restore exact 32K prefix")
    restored_prefix_state = prior._snapshot_digest(restored)
    replay = _run_continuation(
        model,
        restored,
        cold["predicted_token_id"],
        MODEL_CONTINUATION_TOKENS,
    )
    memory = prior._memory(mx)
    checks = {
        "prefix_is_exactly_32768": len(prefix_ids) == MODEL_PREFIX_TOKENS,
        "prefix_hit_is_exactly_32768": prefix_hit == MODEL_PREFIX_TOKENS,
        "restored_prefix_state_exact": restored_prefix_state == prefix_state,
        "all_4096_tokens_exact": replay["token_sha256"] == baseline["token_sha256"],
        "all_4096_full_vocab_logits_exact": (
            replay["all_step_logits_sha256"]
            == baseline["all_step_logits_sha256"]
        ),
        "all_256_boundaries_exact": replay["evidence"] == baseline["evidence"],
        "final_hybrid_state_exact": replay["final_state"] == baseline["final_state"],
        "kda_state_exact": (
            replay["final_state"]["components"]["kda_state_sha256"]
            == baseline["final_state"]["components"]["kda_state_sha256"]
        ),
        "dsa_kv_exact": (
            replay["final_state"]["components"]["dsa_kv_sha256"]
            == baseline["final_state"]["components"]["dsa_kv_sha256"]
        ),
        "indexpool_exact": (
            replay["final_state"]["components"]["indexpool_sha256"]
            == baseline["final_state"]["components"]["indexpool_sha256"]
        ),
        "slot_metadata_exact": (
            replay["final_state"]["components"]["slot_index_metadata_sha256"]
            == baseline["final_state"]["components"]["slot_index_metadata_sha256"]
        ),
        "materialization_count_16_each": (
            baseline["materializations"] == 16
            and replay["materializations"] == 16
        ),
        "nan_free": baseline["nan_count"] == replay["nan_count"] == 0,
        "anonymous_allocation_zero": (
            baseline["final_accounting"]["anonymous_bytes"] == 0
            and replay["final_accounting"]["anonymous_bytes"] == 0
        ),
        "peak_within_340gb": memory["peak_bytes"] <= MAX_PEAK_BYTES,
    }
    artifact["checkpoint"] = {
        "revision": report.official_revision,
        "fingerprint": report.fingerprint,
        "chat_template_revision": report.chat_template_revision,
        "chat_template_digest": report.chat_template_digest,
    }
    artifact["model_32k_4096"] = {
        "completed": True,
        "fixture": fixture,
        "cold": cold,
        "prefix_state": prefix_state,
        "baseline": baseline,
        "replay": replay,
        "memory": memory,
        "checks": checks,
        "accepted": all(checks.values()),
    }
    _finish(artifact)
    _atomic_write(args.output, artifact)
    manager.clear()


def _http_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = error.read()
        return error.code, json.loads(body) if body else {}


def _candidate_messages(tokenizer, payload: str) -> tuple[list[dict[str, Any]], int]:
    messages = [dict(message) for message in prior._agent_messages()]
    messages[-1]["content"] = str(messages[-1]["content"]).replace(
        prior.REPOSITORY_MARKER,
        payload,
    )
    messages[-1]["content"] += (
        "\nProduce at least 256 tokens before finishing; inspect the supplied "
        "repository evidence and do not end early."
    )
    return messages, len(prior._render_ids(tokenizer, messages))


def build_exact_http_messages(
    tokenizer, corpus: str, target_tokens: int
) -> tuple[list[dict[str, Any]], int]:
    """Find a fully rendered chat fixture with an exact token length."""
    separator = "\n# repeated coding-agent repository context\n"
    repeated = corpus
    while _candidate_messages(tokenizer, repeated)[1] < target_tokens:
        repeated += separator + corpus
    low, high = 0, len(repeated)
    best: tuple[list[dict[str, Any]], int, int] | None = None
    while low <= high:
        middle = (low + high) // 2
        messages, count = _candidate_messages(tokenizer, repeated[:middle])
        if count <= target_tokens:
            best = messages, count, middle
            low = middle + 1
        else:
            high = middle - 1
    if best is None:
        raise ValueError("target is shorter than the fixed chat template")
    messages, count, middle = best
    if count == target_tokens:
        return messages, count
    start = max(0, middle - 32)
    stop = min(len(repeated), middle + 96)
    for length in range(start, stop + 1):
        candidate, candidate_count = _candidate_messages(
            tokenizer, repeated[:length]
        )
        if candidate_count == target_tokens:
            return candidate, candidate_count
    raise ValueError(
        f"could not build exact {target_tokens}-token HTTP fixture; nearest={count}"
    )


def _choice_signature(response: dict[str, Any]) -> dict[str, Any]:
    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {})
    tool_calls = []
    for call in message.get("tool_calls") or []:
        # OpenAI-compatible tool-call ids are transport identifiers and may be
        # freshly allocated for an otherwise byte-identical token trajectory.
        # Compare the model-produced function/name/arguments, not that UUID.
        tool_calls.append(
            {
                "type": call.get("type"),
                "function": call.get("function"),
            }
        )
    return {
        "finish_reason": choice.get("finish_reason"),
        "content": message.get("content"),
        "reasoning_content": message.get("reasoning_content"),
        "tool_calls": tool_calls,
        "logprobs": choice.get("logprobs"),
        "completion_tokens": response.get("usage", {}).get("completion_tokens"),
    }


def _cached_tokens(response: dict[str, Any]) -> int:
    return int(
        response.get("usage", {})
        .get("prompt_tokens_details", {})
        .get("cached_tokens", 0)
    )


def _expected_exact_apc_prefix(prompt_tokens: int) -> int:
    safe = int(prompt_tokens) - EXACT_APC_PREFIX_GUARD_TOKENS
    return max(0, safe // HTTP_PREFILL_ALIGNMENT_TOKENS) * (
        HTTP_PREFILL_ALIGNMENT_TOKENS
    )


def _post(args, path: str, payload: dict[str, Any] | None = None):
    return _http_json(
        f"{args.server_url}{path}",
        payload=payload,
        timeout=args.http_timeout,
    )


def _request(args, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    status, response = _post(args, "/v1/chat/completions", payload)
    elapsed = time.perf_counter() - started
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {response}")
    return response, elapsed


def _extended_messages(base: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(base) + [
        {
            "role": "assistant",
            "content": "I will inspect the server admission contract.",
            "tool_calls": [
                {
                    "id": "call_admission_contract",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "glm53_flash_mlx/server.py"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_admission_contract",
            "content": (
                "The authoritative policy is prompt tokens plus requested "
                "generation tokens within the configured context capacity."
            ),
        },
        {
            "role": "user",
            "content": (
                "Use the retained repository prefix. Explain the invariant in "
                "at least 256 tokens without ending early."
            ),
        },
    ]


def _run_http_case(args, tokenizer, corpus: str, context: int) -> dict[str, Any]:
    messages, prompt_tokens = build_exact_http_messages(tokenizer, corpus, context)
    payload = {
        "model": "glm-5.3-flash",
        "messages": messages,
        "tools": prior._tools(),
        "temperature": 0,
        "seed": 0,
        "max_tokens": HTTP_DECODE_TOKENS,
        "logprobs": True,
        "top_logprobs": 5,
        "stream": False,
    }
    _post(args, "/v1/cache/reset", {})
    cold, cold_seconds = _request(args, payload)
    warm, warm_seconds = _request(args, payload)

    extended_payload = {
        **payload,
        "messages": _extended_messages(messages),
    }
    reused, reused_seconds = _request(args, extended_payload)
    reused_cached = _cached_tokens(reused)

    # Remove every APC entry, then compute the same extended request as an
    # uncached Direct HTTP reference and repeat it through APC once more.
    _post(args, "/v1/cache/reset", {})
    direct, direct_seconds = _request(args, extended_payload)
    direct_warm, direct_warm_seconds = _request(args, extended_payload)
    stats_status, stats = _post(args, "/v1/cache/stats")
    metrics_status, metrics = _post(args, "/v1/metrics")
    if stats_status != 200 or metrics_status != 200:
        raise RuntimeError("failed to read server telemetry")

    signatures = {
        "cold": _choice_signature(cold),
        "warm": _choice_signature(warm),
        "reused_tool_suffix": _choice_signature(reused),
        "uncached_tool_suffix_reference": _choice_signature(direct),
        "warm_tool_suffix_reference": _choice_signature(direct_warm),
    }
    usage_rows = {
        name: response.get("usage", {})
        for name, response in (
            ("cold", cold),
            ("warm", warm),
            ("reused_tool_suffix", reused),
            ("uncached_tool_suffix_reference", direct),
            ("warm_tool_suffix_reference", direct_warm),
        )
    }
    server_admission = metrics.get("server", {}).get("admission", {})
    peak_gb = float(metrics.get("latest", {}).get("peak_memory_gb") or 0.0)
    checks = {
        "prompt_token_count_exact": prompt_tokens == context,
        "cold_warm_output_exact": signatures["cold"] == signatures["warm"],
        "base_prefix_hit_exact": _cached_tokens(warm)
        == _expected_exact_apc_prefix(context),
        "tool_suffix_reuses_guarded_base_prefix": reused_cached
        == _expected_exact_apc_prefix(context),
        "tool_suffix_matches_uncached_direct": (
            signatures["reused_tool_suffix"]
            == signatures["uncached_tool_suffix_reference"]
        ),
        "tool_suffix_warm_repeat_exact": (
            signatures["uncached_tool_suffix_reference"]
            == signatures["warm_tool_suffix_reference"]
        ),
        "tool_suffix_warm_hit": _cached_tokens(direct_warm) > 0,
        "decode_budget_respected": all(
            0 < int(row.get("completion_tokens", 0)) <= HTTP_DECODE_TOKENS
            for row in usage_rows.values()
        ),
        "full_256_decode_exercised": all(
            int(usage_rows[name].get("completion_tokens", 0))
            == HTTP_DECODE_TOKENS
            for name in (
                "reused_tool_suffix",
                "uncached_tool_suffix_reference",
                "warm_tool_suffix_reference",
            )
        ),
        "total_context_accounting_within_capacity": all(
            int(row.get("total_tokens", 0)) <= DEFAULT_MAX_CONTEXT_TOKENS
            for row in usage_rows.values()
        ),
        "server_admission_policy_exact": server_admission
        == admission_snapshot(
            max_generation_tokens=DEFAULT_MAX_GENERATION_TOKENS,
            max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
        ),
        "peak_within_340gb": peak_gb * 1e9 <= MAX_PEAK_BYTES,
    }
    return {
        "context_tokens": context,
        "prompt_tokens": prompt_tokens,
        "latency_seconds": {
            "cold": cold_seconds,
            "warm": warm_seconds,
            "reused_tool_suffix": reused_seconds,
            "uncached_tool_suffix_reference": direct_seconds,
            "warm_tool_suffix_reference": direct_warm_seconds,
        },
        "cached_tokens": {
            name: _cached_tokens(response)
            for name, response in (
                ("cold", cold),
                ("warm", warm),
                ("reused_tool_suffix", reused),
                ("uncached_tool_suffix_reference", direct),
                ("warm_tool_suffix_reference", direct_warm),
            )
        },
        "usage": usage_rows,
        "choice_sha256": {
            name: hashlib.sha256(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for name, value in signatures.items()
        },
        "choice_signatures": signatures,
        "cache_stats": stats,
        "server_admission": server_admission,
        "latest_peak_memory_gb": peak_gb,
        "checks": checks,
        "accepted": all(checks.values()),
    }


def _run_http_phase(args, artifact: dict[str, Any]) -> None:
    if artifact.get("model_32k_4096", {}).get("accepted") is not True:
        raise QualificationPreconditionError(
            "HTTP phase requires an accepted model phase first; run this script "
            "with --phase model while the server is stopped"
        )
    from mlx_vlm.tokenizer_utils import load_tokenizer

    tokenizer = load_tokenizer(args.model)._tokenizer
    corpus = prior._repository_corpus(REPOSITORY)
    health_status, health = _post(args, "/health")
    metrics_status, metrics = _post(args, "/v1/metrics")
    if health_status != 200 or metrics_status != 200:
        raise QualificationPreconditionError(
            f"server health or metrics endpoint failed at {args.server_url}"
        )
    expected_admission = admission_snapshot(
        max_generation_tokens=DEFAULT_MAX_GENERATION_TOKENS,
        max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
    )
    if metrics.get("server", {}).get("admission") != expected_admission:
        raise QualificationPreconditionError(
            "server is not running the promoted admission policy"
        )
    if metrics.get("server", {}).get("exact_apc") != {
        "store_policy": EXACT_APC_STORE_POLICY,
        "prefix_guard_tokens": EXACT_APC_PREFIX_GUARD_TOKENS,
        "checkpoint_alignment_tokens": HTTP_PREFILL_ALIGNMENT_TOKENS,
    }:
        raise QualificationPreconditionError(
            "server is not running the guarded exact APC store policy; restart it"
        )
    if metrics.get("server", {}).get("long_prefill") != {
        "token_queue_timeout_seconds": DEFAULT_FIRST_TOKEN_TIMEOUT_SECONDS,
    }:
        raise QualificationPreconditionError(
            "server is not running the qualified long-prefill timeout; restart it"
        )
    if health.get("apc_enabled") is not True:
        raise QualificationPreconditionError(
            "qualification server must be started with --apc"
        )

    rows = artifact.setdefault("http", {}).setdefault("cases", {})
    for context in args.http_contexts:
        if rows.get(str(context), {}).get("accepted") is True:
            _progress("skip_http_context", context=context)
            continue
        _progress("http_context", context=context)
        rows[str(context)] = _run_http_case(args, tokenizer, corpus, context)
        _atomic_write(args.output, artifact)
        if not rows[str(context)]["accepted"]:
            break

    # The accepted side of the exact boundary is covered by the 32K+4096
    # model arm.  Exercise its +1 rejection through the real HTTP preflight.
    boundary_messages, boundary_prompt = build_exact_http_messages(
        tokenizer, corpus, QUALIFIED_PROMPT_TOKENS + 1
    )
    reject_payload = {
        "model": "glm-5.3-flash",
        "messages": boundary_messages,
        "tools": prior._tools(),
        "temperature": 0,
        "max_tokens": DEFAULT_MAX_GENERATION_TOKENS,
        "stream": False,
    }
    before_status, before_stats = _post(args, "/v1/cache/stats")
    reject_status, reject_body = _post(
        args, "/v1/chat/completions", reject_payload
    )
    after_status, after_stats = _post(args, "/v1/cache/stats")
    validate_admission(
        QUALIFIED_PROMPT_TOKENS,
        DEFAULT_MAX_GENERATION_TOKENS,
        max_generation_tokens=DEFAULT_MAX_GENERATION_TOKENS,
        max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
    )
    boundary_checks = {
        "accepted_boundary_validates": True,
        "rejected_prompt_is_32769": boundary_prompt == 32769,
        "plus_one_rejected_http_400": reject_status == 400,
        "rejection_mentions_total_context": "context" in json.dumps(reject_body).lower(),
        "cache_stats_available": before_status == after_status == 200,
        "rejected_request_does_not_store_cache": (
            before_stats.get("exact_stores") == after_stats.get("exact_stores")
        ),
    }
    all_cases = all(
        rows.get(str(context), {}).get("accepted") is True
        for context in HTTP_CONTEXTS
    )
    artifact["http"].update(
        {
            "completed": all_cases,
            "health": health,
            "initial_metrics_server": metrics.get("server"),
            "boundary": {
                "prompt_tokens": boundary_prompt,
                "requested_generation_tokens": DEFAULT_MAX_GENERATION_TOKENS,
                "status": reject_status,
                "body": reject_body,
                "checks": boundary_checks,
            },
            "accepted": all_cases and all(boundary_checks.values()),
        }
    )
    _finish(artifact)
    _atomic_write(args.output, artifact)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=("model", "http"), required=True)
    parser.add_argument(
        "--http-contexts",
        nargs="+",
        type=int,
        choices=HTTP_CONTEXTS,
        default=list(HTTP_CONTEXTS),
        help="HTTP contexts to run; accepted existing rows are reused",
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--http-timeout", type=float, default=1800.0)
    parser.add_argument("--wired-limit-bytes", type=int, default=440_000_000_000)
    parser.add_argument("--cache-limit-bytes", type=int, default=32_000_000_000)
    args = parser.parse_args(argv)
    args.model = args.model.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    artifact = _load_artifact(args.output, args.model)
    _atomic_write(args.output, artifact)
    try:
        if args.phase == "model":
            _run_model_phase(args, artifact)
        else:
            _run_http_phase(args, artifact)
    except QualificationPreconditionError as error:
        print(f"qualification precondition: {error}", file=sys.stderr)
        return 2
    except urllib.error.URLError as error:
        print(
            f"qualification precondition: cannot reach {args.server_url}: "
            f"{error.reason}; start `uv run glm53 serve --apc` first",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
