#!/usr/bin/env python3
"""Qualify long coding-agent prompts and exact hybrid APC reuse.

This is deliberately a user-launched, probe-only qualification.  It bypasses
the production prompt admission limit without changing it, uses the default
Direct MoE/cache backend, and atomically saves progress after every phase.  A
completed context can be skipped on a later invocation, but an in-flight model
cache is never serialized or advertised as resumable.

The uninterrupted and APC-restored arms consume the *same suffix in one call*.
That detail matters: comparing a long-Q cold call with a short-Q warm call would
also compare different attention reduction geometries, not just cache state.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_OUTPUT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-coding-agent-prefix-cache-admission-20260905.json"
)
CONTEXTS = (4096, 8192, 16384, 32768)
PREFILL_CHUNK_TOKENS = 2048
REPEATED_WARM_TURNS = 3
MAX_PEAK_BYTES = 340_000_000_000
MAX_STEADY_ACTIVE_DRIFT_BYTES = 64 << 20
REPOSITORY_MARKER = "__GLM53_REPOSITORY_CONTEXT_SLOT__"
SCHEMA = "glm53-coding-agent-prefix-cache-admission-v1"


def _progress(phase: str, **values: Any) -> None:
    print(json.dumps({"phase": phase, **values}, sort_keys=True), flush=True)


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256_token_ids(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 repository file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_repository",
                "description": "Search repository text with a regular expression.",
                "parameters": {
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}},
                    "required": ["pattern"],
                },
            },
        },
    ]


def _agent_messages() -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a coding agent. Inspect repository evidence before "
                "answering, use tools when needed, and preserve exact runtime "
                "contracts."
            ),
        },
        {
            "role": "user",
            "content": "Locate the cache admission and state-safety invariants.",
        },
        {
            "role": "assistant",
            "reasoning_content": "I should inspect the repository first.",
            "content": "I will inspect the relevant runtime and tests.",
        },
        {
            "role": "user",
            "content": (
                "Repository context follows. Explain whether long coding-agent "
                "prefix reuse preserves every authoritative state component.\n\n"
                + REPOSITORY_MARKER
            ),
        },
    ]


def _repository_corpus(root: Path) -> str:
    """Return deterministic real repository text used to fill long prompts."""

    candidates = (
        "README.md",
        "glm53_flash_mlx/server.py",
        "glm53_flash_mlx/semantic_snapshot.py",
        "glm53_flash_mlx/cache_lifecycle.py",
        "glm53_flash_mlx/nope_cache.py",
        "tests/test_semantic_snapshot.py",
        "tests/test_cache_lifecycle.py",
    )
    sections = []
    for relative in candidates:
        path = root / relative
        sections.append(f"\n===== {relative} =====\n{path.read_text()}\n")
    return "".join(sections)


def _render_ids(tokenizer, messages: list[dict[str, Any]]) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=_tools(),
        add_generation_prompt=True,
        tokenize=True,
    )
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("coding-agent fixture requires one token sequence")
        encoded = encoded[0]
    return [int(value) for value in encoded]


def _subsequence_positions(values: Sequence[int], needle: Sequence[int]) -> list[int]:
    if not needle:
        return []
    return [
        offset
        for offset in range(len(values) - len(needle) + 1)
        if list(values[offset : offset + len(needle)]) == list(needle)
    ]


def build_coding_agent_fixture(tokenizer, target_tokens: int, corpus: str) -> dict[str, Any]:
    """Build an exact-length, coding-agent-shaped prefix and tool-result suffix.

    The official template renders the surrounding system/tool/conversation
    structure.  Only the repository marker is replaced at token level, which
    keeps the assistant generation boundary intact while making all requested
    context lengths exact and deterministic.
    """

    messages = _agent_messages()
    marker_ids = tokenizer.encode(REPOSITORY_MARKER, add_special_tokens=False)
    skeleton = _render_ids(tokenizer, messages)
    positions = _subsequence_positions(skeleton, marker_ids)
    if len(positions) != 1:
        raise ValueError("repository marker must tokenize to one unique span")
    marker_start = positions[0]
    fixed_tokens = len(skeleton) - len(marker_ids)
    payload_tokens = int(target_tokens) - fixed_tokens
    if payload_tokens <= 0:
        raise ValueError(
            f"target {target_tokens} is too short for the coding-agent fixture "
            f"({fixed_tokens} fixed tokens)"
        )

    corpus_ids = tokenizer.encode(corpus, add_special_tokens=False)
    separator_ids = tokenizer.encode("\n# repository corpus repeats\n", add_special_tokens=False)
    if not corpus_ids:
        raise ValueError("repository corpus tokenized to an empty sequence")
    expanded: list[int] = []
    while len(expanded) < payload_tokens:
        expanded.extend(corpus_ids)
        expanded.extend(separator_ids)
    payload = expanded[:payload_tokens]
    prefix = skeleton[:marker_start] + payload + skeleton[marker_start + len(marker_ids) :]
    if len(prefix) != target_tokens:
        raise AssertionError("coding-agent fixture did not reach its exact target")

    # The suffix is the next agent/tool turn after the open generation prompt.
    # It is deliberately fixed except for the first actually predicted token,
    # which is prepended by the model qualification after cold prefill.
    suffix_text = (
        "</think><tool_call>read_file<arg_key>path</arg_key>"
        "<arg_value>glm53_flash_mlx/server.py</arg_value></tool_call>"
        "<|observation|><tool_response>Admission defaults and cache policy read."
        "</tool_response><|user|>Now compare the warm prefix state and continue."
        "<|assistant|><think>"
    )
    suffix = [int(value) for value in tokenizer.encode(suffix_text, add_special_tokens=False)]
    return {
        "context_tokens": int(target_tokens),
        "prefix_token_ids": prefix,
        "prefix_token_sha256": _sha256_token_ids(prefix),
        "repository_payload_tokens": payload_tokens,
        "repository_payload_sha256": _sha256_token_ids(payload),
        "fixed_template_tokens": fixed_tokens,
        "suffix_after_predicted_token_ids": suffix,
        "suffix_after_predicted_token_sha256": _sha256_token_ids(suffix),
        "has_system_prompt": True,
        "has_tool_definitions": True,
        "has_repository_context": True,
        "has_conversation_history": True,
        "has_assistant_generation_boundary": True,
        "has_tool_call_and_result_suffix": True,
    }


def build_server_smoke_messages(
    tokenizer, corpus: str, *, target_tokens: int = 4096
) -> tuple[list[dict[str, Any]], int]:
    """Build a fully template-rendered HTTP fixture at or below the target."""

    messages = _agent_messages()

    def candidate(characters: int) -> tuple[list[dict[str, Any]], int]:
        replacement = corpus[:characters]
        current = [dict(message) for message in messages]
        current[-1]["content"] = str(current[-1]["content"]).replace(
            REPOSITORY_MARKER, replacement
        )
        return current, len(_render_ids(tokenizer, current))

    low = 0
    high = len(corpus)
    best_messages, best_tokens = candidate(0)
    while low <= high:
        middle = (low + high) // 2
        current, count = candidate(middle)
        if count <= target_tokens:
            best_messages, best_tokens = current, count
            low = middle + 1
        else:
            high = middle - 1
    if best_tokens < target_tokens - 64:
        raise ValueError("could not build a sufficiently full server smoke prompt")
    return best_messages, best_tokens


def _memory(mx) -> dict[str, int]:
    mx.synchronize()
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }


def _array_digest(value) -> str:
    import mlx.core as mx
    import numpy as np

    storage = value.view(mx.uint16) if value.dtype == mx.bfloat16 else value
    mx.eval(storage)
    return hashlib.sha256(np.ascontiguousarray(np.asarray(storage)).tobytes()).hexdigest()


def _array_f32(value):
    import mlx.core as mx
    import numpy as np

    converted = value.astype(mx.float32)
    mx.eval(converted)
    return np.ascontiguousarray(np.asarray(converted), dtype=np.float32)


def _arrays(value: Any) -> Iterable[Any]:
    import mlx.core as mx

    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _arrays(item)
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _arrays(value[key])


def _entry_nbytes(entry: Any) -> int:
    nbytes = getattr(entry, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    return sum(int(value.nbytes) for value in _arrays(entry.state))


def _cache_accounting(cache: Sequence[Any], *, snapshot: bool) -> dict[str, Any]:
    """Classify every authoritative cache byte without inferring from dtype."""

    from glm53_flash_mlx.cache_lifecycle import CacheLifecycle

    by_class = {lifecycle.value: 0 for lifecycle in CacheLifecycle}
    leaf_count = 0
    total = 0
    for entry in cache:
        children = tuple(getattr(entry, "caches", ()))
        if snapshot:
            size = _entry_nbytes(entry)
            by_class[CacheLifecycle.SNAPSHOT_STATE.value] += size
            total += size
            leaf_count += sum(1 for _ in _arrays(entry.state))
        elif len(children) == 2:
            for child in children:
                size = _entry_nbytes(child)
                by_class[CacheLifecycle.TARGET_PREFIX.value] += size
                total += size
                leaf_count += sum(1 for _ in _arrays(child.state))
        else:
            size = _entry_nbytes(entry)
            by_class[CacheLifecycle.ACTIVE_RECURRENT.value] += size
            total += size
            leaf_count += sum(1 for _ in _arrays(entry.state))
    accounted = sum(by_class.values())
    return {
        "total_authoritative_bytes": total,
        "resident_bytes_by_lifecycle": by_class,
        "accounted_bytes": accounted,
        "anonymous_bytes": total - accounted,
        "state_leaf_count": leaf_count,
    }


def _prefill(model, cache, token_ids: Sequence[int], *, chunk_tokens: int) -> dict[str, Any]:
    import mlx.core as mx

    started = time.perf_counter()
    chunks = []
    output = None
    for start in range(0, len(token_ids), chunk_tokens):
        stop = min(start + chunk_tokens, len(token_ids))
        chunk_started = time.perf_counter()
        output = model(
            mx.array([token_ids[start:stop]], dtype=mx.uint32),
            cache=cache,
        )
        mx.eval(output.logits, [entry.state for entry in cache])
        mx.synchronize()
        chunks.append(
            {
                "start": start,
                "stop": stop,
                "tokens": stop - start,
                "latency_seconds": time.perf_counter() - chunk_started,
            }
        )
        _progress("prefill_chunk", start=start, stop=stop, total=len(token_ids))
    if output is None:
        raise ValueError("prefill token sequence must be non-empty")
    logits = output.logits[0, -1]
    nan_count = mx.sum(mx.isnan(logits))
    predicted = mx.argmax(logits)
    mx.eval(logits, nan_count, predicted)
    return {
        "latency_seconds": time.perf_counter() - started,
        "chunks": chunks,
        "final_logits_sha256": _array_digest(logits),
        "predicted_token_id": int(predicted.item()),
        "nan_count": int(nan_count.item()),
    }


def _snapshot_digest(cache: Sequence[Any]) -> dict[str, Any]:
    from glm53_flash_mlx.semantic_snapshot import (
        semantic_cache_digest,
        semantic_cache_schema,
        semantic_component_digests,
    )

    return {
        "state_sha256": semantic_cache_digest(cache),
        "schema_sha256": hashlib.sha256(
            json.dumps(
                semantic_cache_schema(cache),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "components": semantic_component_digests(cache),
    }


def _exact_entries(manager) -> list[Any]:
    with manager.lock:
        return list(manager._exact_cache.values())


def _exact_cache_accounting(manager) -> dict[str, Any]:
    rows = [_cache_accounting(entry.prompt_cache, snapshot=True) for entry in _exact_entries(manager)]
    return {
        "entry_count": len(rows),
        "resident_bytes": sum(row["total_authoritative_bytes"] for row in rows),
        "anonymous_bytes": sum(row["anonymous_bytes"] for row in rows),
        "entries": rows,
    }


def _release_transient(mx, *values: Any) -> None:
    del values
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def _qualify_context(model, tokenizer, corpus: str, context_tokens: int) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_vlm.apc import APCManager, model_apc_mode

    from glm53_flash_mlx.semantic_snapshot import (
        semantic_cache_storage_alias_count,
    )

    fixture = build_coding_agent_fixture(tokenizer, context_tokens, corpus)
    prefix_ids = fixture.pop("prefix_token_ids")
    suffix_tail = fixture.pop("suffix_after_predicted_token_ids")
    manager = APCManager(num_blocks=1, block_size=64)
    if model_apc_mode(model.language_model) != "exact":
        raise RuntimeError("hybrid GLM cache must use exact APC mode")

    before = _memory(mx)
    cold_cache = model.make_cache()
    _progress("cold_prefill", context_tokens=context_tokens)
    cold = _prefill(model, cold_cache, prefix_ids, chunk_tokens=PREFILL_CHUNK_TOKENS)
    cold_state = _snapshot_digest(cold_cache)
    cold_accounting = _cache_accounting(cold_cache, snapshot=False)
    predicted = cold["predicted_token_id"]
    suffix_ids = [predicted] + suffix_tail
    full_ids = prefix_ids + suffix_ids

    stored = manager.store_exact_cache(prefix_ids, cold_cache)
    if not stored:
        raise RuntimeError("exact APC rejected the coding-agent prefix")
    stored_entries = _exact_entries(manager)
    if len(stored_entries) != 1:
        raise AssertionError("one stored prefix must produce one exact APC entry")
    stored_cache = stored_entries[0].prompt_cache
    stored_digest_before = _snapshot_digest(stored_cache)
    stored_accounting_before = _exact_cache_accounting(manager)
    live_snapshot_aliases = semantic_cache_storage_alias_count(cold_cache, stored_cache)

    # Uninterrupted Direct reference: same live prefix, same suffix geometry.
    _progress("uncached_suffix_reference", context_tokens=context_tokens)
    reference = _prefill(model, cold_cache, suffix_ids, chunk_tokens=len(suffix_ids))
    reference_state = _snapshot_digest(cold_cache)
    reference_accounting = _cache_accounting(cold_cache, snapshot=False)

    warm_rows = []
    warm_active_samples = []
    for turn in range(REPEATED_WARM_TURNS):
        lookup_started = time.perf_counter()
        warm_cache, prefix_len = manager.lookup_exact_cache(full_ids)
        lookup_seconds = time.perf_counter() - lookup_started
        if warm_cache is None or prefix_len != context_tokens:
            raise AssertionError("APC did not return the exact full coding-agent prefix")
        aliases = semantic_cache_storage_alias_count(stored_cache, warm_cache)
        restored_state = _snapshot_digest(warm_cache)
        _progress(
            "warm_suffix",
            context_tokens=context_tokens,
            turn=turn + 1,
            prefix_hit_tokens=prefix_len,
        )
        warm = _prefill(model, warm_cache, suffix_ids, chunk_tokens=len(suffix_ids))
        warm_state = _snapshot_digest(warm_cache)
        warm_accounting = _cache_accounting(warm_cache, snapshot=False)
        warm_memory = _memory(mx)
        warm_active_samples.append(warm_memory["active_bytes"])
        row = {
            "turn": turn + 1,
            "lookup_seconds": lookup_seconds,
            "prefix_hit_tokens": prefix_len,
            "prefill_tokens_actually_computed": len(suffix_ids),
            "restored_prefix_state_exact": restored_state == cold_state,
            "stored_to_restored_alias_count": aliases,
            "suffix": warm,
            "final_state": warm_state,
            "accounting": warm_accounting,
            "memory": warm_memory,
            "final_logits_exact": (
                warm["final_logits_sha256"] == reference["final_logits_sha256"]
            ),
            "final_state_exact": warm_state == reference_state,
            "component_state_exact": (
                warm_state["components"] == reference_state["components"]
            ),
        }
        warm_rows.append(row)
        del warm_cache
        _release_transient(mx)

    stored_digest_after = _snapshot_digest(stored_cache)
    stored_accounting_after = _exact_cache_accounting(manager)
    after = _memory(mx)
    steady_drift = max(warm_active_samples) - min(warm_active_samples)
    stats = manager.stats_snapshot()
    all_warm_exact = all(
        row["restored_prefix_state_exact"]
        and row["final_logits_exact"]
        and row["final_state_exact"]
        and row["component_state_exact"]
        for row in warm_rows
    )
    result = {
        "context_tokens": context_tokens,
        "fixture": fixture,
        "suffix_tokens": len(suffix_ids),
        "suffix_token_sha256": _sha256_token_ids(suffix_ids),
        "full_request_tokens": len(full_ids),
        "cold": {
            **cold,
            "state": cold_state,
            "accounting": cold_accounting,
        },
        "uncached_direct_suffix_reference": {
            **reference,
            "state": reference_state,
            "accounting": reference_accounting,
        },
        "warm_turns": warm_rows,
        "apc": {
            "mode": "exact",
            "stored": stored,
            "stats": stats,
            "stored_digest_immutable": stored_digest_before == stored_digest_after,
            "stored_accounting_before": stored_accounting_before,
            "stored_accounting_after": stored_accounting_after,
            "stored_resident_bounded": (
                stored_accounting_before == stored_accounting_after
                and stored_accounting_after["entry_count"] == 1
            ),
            "live_to_stored_alias_count": live_snapshot_aliases,
        },
        "resource": {
            "before": before,
            "after": after,
            "warm_active_samples": warm_active_samples,
            "warm_active_drift_bytes": steady_drift,
            "peak_limit_bytes": MAX_PEAK_BYTES,
        },
        "acceptance": {
            "cold_prefill_succeeded": cold["nan_count"] == 0,
            "exact_prefix_hit_every_turn": all(
                row["prefix_hit_tokens"] == context_tokens for row in warm_rows
            ),
            "suffix_only_prefill_every_turn": all(
                row["prefill_tokens_actually_computed"] == len(suffix_ids)
                for row in warm_rows
            ),
            "restored_prefix_state_exact": all(
                row["restored_prefix_state_exact"] for row in warm_rows
            ),
            "cold_restore_suffix_logits_exact": all_warm_exact,
            "kda_state_exact": all(
                row["final_state"]["components"]["kda_state_sha256"]
                == reference_state["components"]["kda_state_sha256"]
                for row in warm_rows
            ),
            "dsa_kv_exact": all(
                row["final_state"]["components"]["dsa_kv_sha256"]
                == reference_state["components"]["dsa_kv_sha256"]
                for row in warm_rows
            ),
            "indexpool_exact": all(
                row["final_state"]["components"]["indexpool_sha256"]
                == reference_state["components"]["indexpool_sha256"]
                for row in warm_rows
            ),
            "slot_index_metadata_exact": all(
                row["final_state"]["components"]["slot_index_metadata_sha256"]
                == reference_state["components"]["slot_index_metadata_sha256"]
                for row in warm_rows
            ),
            "snapshot_immutable": stored_digest_before == stored_digest_after,
            "snapshot_live_alias_free": live_snapshot_aliases == 0,
            "restored_cache_alias_free": all(
                row["stored_to_restored_alias_count"] == 0 for row in warm_rows
            ),
            "lifecycle_accounting_exact": (
                cold_accounting["anonymous_bytes"] == 0
                and reference_accounting["anonymous_bytes"] == 0
                and stored_accounting_after["anonymous_bytes"] == 0
                and all(row["accounting"]["anonymous_bytes"] == 0 for row in warm_rows)
            ),
            "snapshot_resident_bounded": stored_accounting_before
            == stored_accounting_after,
            "warm_active_drift_within_limit": steady_drift
            <= MAX_STEADY_ACTIVE_DRIFT_BYTES,
            "peak_within_limit": after["peak_bytes"] <= MAX_PEAK_BYTES,
            "nan_free": cold["nan_count"] == 0
            and reference["nan_count"] == 0
            and all(row["suffix"]["nan_count"] == 0 for row in warm_rows),
        },
    }
    result["accepted"] = all(result["acceptance"].values())
    manager.clear()
    del cold_cache, stored_cache, stored_entries
    _release_transient(mx)
    return result


def _qualify_official_oracle(model, tokenizer, generation_tokens: int) -> dict[str, Any]:
    """Reuse the already-resident model for the accepted 16/128-token oracle."""

    import mlx.core as mx
    import numpy as np

    expected_path = REPOSITORY / f"oracles/glm53-official-greedy-{generation_tokens}.json"
    expected = json.loads(expected_path.read_text())
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": expected["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(formatted, return_tensors="np", add_special_tokens=True)
    prompt_ids = np.asarray(encoded["input_ids"], dtype=np.int32).reshape(1, -1)
    cache = model.make_cache()
    output = model(mx.array(prompt_ids), cache=cache)
    generated = []
    hashes = []
    for step in range(generation_tokens):
        logits = output.logits[0, -1]
        logits_f32 = _array_f32(logits)
        hashes.append(hashlib.sha256(logits_f32.tobytes()).hexdigest())
        top2 = np.argpartition(logits_f32, -2)[-2:]
        top2 = top2[np.argsort(logits_f32[top2])[::-1]]
        generated.append(int(top2[0]))
        if step + 1 < generation_tokens:
            output = model(mx.array([[generated[-1]]], dtype=mx.uint32), cache=cache)
    expected_hashes = [row["logits_f32_sha256"] for row in expected["steps"]]
    result = {
        "generation_tokens": generation_tokens,
        "generated_token_ids": generated,
        "generated_token_ids_exact": generated == expected["generated_token_ids"],
        "full_vocab_logits_hashes_exact": hashes == expected_hashes,
        "prompt_token_count": int(prompt_ids.size),
        "prompt_render_sha256": hashlib.sha256(formatted.encode()).hexdigest(),
    }
    result["accepted"] = (
        result["generated_token_ids_exact"]
        and result["full_vocab_logits_hashes_exact"]
    )
    del cache, output
    _release_transient(mx)
    return result


def _initial_artifact(model_path: Path, contexts: Sequence[int]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "date": date.today().isoformat(),
        "complete": False,
        "accepted": False,
        "last_completed_phase": "initialized",
        "last_completed_context": None,
        "model_path": str(model_path),
        "contexts": list(contexts),
        "prefill_chunk_tokens": PREFILL_CHUNK_TOKENS,
        "repeated_warm_turns": REPEATED_WARM_TURNS,
        "execution": {
            "moe_backend": "direct",
            "cache_backend": "direct",
            "apc_mode": "exact-hybrid-ram",
            "server_admission_bypassed_by_probe_only": True,
            "production_admission_changed": False,
            "runtime_backend_changed": False,
            "disk_apc_used": False,
        },
        "cases": {},
        "server_smoke": {
            "completed": False,
            "note": (
                "Run separately against an explicitly long-admission server; "
                "the model qualification itself does not mutate server defaults."
            ),
        },
    }


def _load_existing_or_initialize(
    output: Path, model_path: Path, contexts: Sequence[int]
) -> dict[str, Any]:
    if not output.exists():
        return _initial_artifact(model_path, contexts)
    artifact = json.loads(output.read_text())
    if artifact.get("schema") != SCHEMA:
        raise ValueError("existing output uses a different schema")
    if artifact.get("model_path") != str(model_path):
        raise ValueError("existing output belongs to a different model path")
    if artifact.get("contexts") != list(contexts):
        raise ValueError("existing output belongs to a different context plan")
    return artifact


def _run_model_qualification(args, artifact: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx

    from glm53_flash_mlx.abi import MLX_VLM_REVISION, NOPE_DSA_CACHE_ABI_DIRECT
    from glm53_flash_mlx.loader import load, warm_residency
    from glm53_flash_mlx.manifest import inspect_checkpoint

    report = inspect_checkpoint(args.model, require_server_ready=True)
    artifact["checkpoint"] = {
        "revision": report.official_revision,
        "fingerprint": report.fingerprint,
        "tokenizer_revision": report.tokenizer_revision,
        "tokenizer_digest": report.tokenizer_digest,
        "chat_template_revision": report.chat_template_revision,
        "chat_template_digest": report.chat_template_digest,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "attention_cache_abi": NOPE_DSA_CACHE_ABI_DIRECT,
    }
    mx.set_wired_limit(args.wired_limit_bytes)
    mx.set_cache_limit(args.cache_limit_bytes)
    model, tokenizer = load(args.model)
    warm_residency(model)
    corpus = _repository_corpus(REPOSITORY)
    artifact["last_completed_phase"] = "model_loaded"
    artifact["model_resident_memory"] = _memory(mx)
    _atomic_write(args.output, artifact)
    for context_tokens in args.contexts:
        key = str(context_tokens)
        if artifact["cases"].get(key, {}).get("accepted") is True:
            _progress("skip_completed_context", context_tokens=context_tokens)
            continue
        try:
            case = _qualify_context(model, tokenizer, corpus, context_tokens)
        except BaseException as error:
            artifact["complete"] = False
            artifact["accepted"] = False
            artifact["last_completed_phase"] = "failed"
            artifact["failure"] = {
                "context_tokens": context_tokens,
                "type": type(error).__name__,
                "message": str(error),
            }
            _atomic_write(args.output, artifact)
            raise
        artifact["cases"][key] = case
        artifact["last_completed_context"] = context_tokens
        artifact["last_completed_phase"] = f"context-{context_tokens}"
        _atomic_write(args.output, artifact)
        _progress("context_complete", context_tokens=context_tokens, accepted=case["accepted"])
        if not case["accepted"]:
            break

    contexts_passed = all(
        artifact["cases"].get(str(context), {}).get("accepted") is True
        for context in args.contexts
    )
    if contexts_passed and not artifact.get("official_oracles", {}).get("accepted"):
        _progress("official_oracles", tokens=[16, 128])
        oracle_rows = {
            str(tokens): _qualify_official_oracle(model, tokenizer, tokens)
            for tokens in (16, 128)
        }
        artifact["official_oracles"] = {
            "cases": oracle_rows,
            "accepted": all(row["accepted"] for row in oracle_rows.values()),
        }
        artifact["last_completed_phase"] = "official-oracles"
        _atomic_write(args.output, artifact)

    all_contexts = all(
        artifact["cases"].get(str(context), {}).get("accepted") is True
        for context in args.contexts
    )
    artifact["acceptance"] = {
        "all_contexts_completed": all_contexts,
        "all_contexts_accepted": all_contexts,
        "contexts_are_4k_8k_16k_32k": tuple(args.contexts) == CONTEXTS,
        "default_direct_backend_preserved": True,
        "production_admission_unchanged": True,
        "disk_apc_unused": True,
        "latest_template_not_promoted_by_this_probe": True,
        "official_16_128_oracles_exact": artifact.get("official_oracles", {}).get(
            "accepted"
        )
        is True,
        "openai_server_smoke_completed": artifact.get("server_smoke", {}).get(
            "accepted"
        )
        is True,
    }
    artifact["accepted"] = all(artifact["acceptance"].values())
    artifact["complete"] = all_contexts and artifact["acceptance"][
        "openai_server_smoke_completed"
    ]
    if artifact["complete"]:
        artifact["last_completed_phase"] = "complete"
    artifact["final_memory"] = _memory(mx)
    _atomic_write(args.output, artifact)
    return artifact


def _http_json(url: str, *, payload: dict[str, Any] | None = None, timeout: float) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _run_server_smoke(args, artifact: dict[str, Any]) -> dict[str, Any]:
    """Optional real OpenAI-compatible smoke, intentionally a separate process/run."""

    from mlx_vlm.tokenizer_utils import load_tokenizer

    tokenizer = load_tokenizer(args.model)._tokenizer
    corpus = _repository_corpus(REPOSITORY)
    if not all(
        artifact.get("cases", {}).get(str(context), {}).get("accepted") is True
        for context in artifact["contexts"]
    ):
        raise ValueError("run and pass the model qualification before server smoke")
    # Keep this smoke bounded: 4K proves the actual HTTP/APC path while the
    # in-process qualification covers exact internal state through 32K.
    messages, prompt_tokens = build_server_smoke_messages(tokenizer, corpus)
    first_payload = {
        "model": str(args.model),
        "messages": messages,
        "tools": _tools(),
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
    }
    health_before = _http_json(f"{args.server_url}/health", timeout=args.http_timeout)
    cache_before = _http_json(
        f"{args.server_url}/v1/cache/stats", timeout=args.http_timeout
    )
    timings = []
    responses = []
    started = time.perf_counter()
    first = _http_json(
        f"{args.server_url}/v1/chat/completions",
        payload=first_payload,
        timeout=args.http_timeout,
    )
    timings.append(time.perf_counter() - started)
    first_message = first.get("choices", [{}])[0].get("message", {})
    assistant_message = {
        key: value
        for key, value in first_message.items()
        if key in {"role", "content", "reasoning_content", "tool_calls"}
        and value is not None
    }
    assistant_message.setdefault("role", "assistant")
    second_messages = list(messages) + [
        assistant_message,
        {
            "role": "user",
            "content": "Use the retained repository prefix and report the next invariant.",
        },
    ]
    second_payload = {**first_payload, "messages": second_messages}
    started = time.perf_counter()
    second = _http_json(
        f"{args.server_url}/v1/chat/completions",
        payload=second_payload,
        timeout=args.http_timeout,
    )
    timings.append(time.perf_counter() - started)
    for turn, response in enumerate((first, second), start=1):
        responses.append(
            {
                "turn": turn,
                "id": response.get("id"),
                "usage": response.get("usage"),
                "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
            }
        )
    health_after = _http_json(f"{args.server_url}/health", timeout=args.http_timeout)
    cache_after = _http_json(
        f"{args.server_url}/v1/cache/stats", timeout=args.http_timeout
    )
    matched_delta = int(cache_after.get("matched_tokens", 0)) - int(
        cache_before.get("matched_tokens", 0)
    )
    artifact["server_smoke"] = {
        "completed": True,
        "server_url": args.server_url,
        "prompt_tokens": prompt_tokens,
        "turn_latency_seconds": timings,
        "responses": responses,
        "health_before": health_before,
        "health_after": health_after,
        "cache_before": cache_before,
        "cache_after": cache_after,
        "matched_tokens_delta": matched_delta,
        "openai_compatible_requests_succeeded": len(responses) == 2,
        "exact_prefix_hit_observed": matched_delta > 0,
    }
    artifact["server_smoke"]["accepted"] = (
        artifact["server_smoke"]["openai_compatible_requests_succeeded"]
        and artifact["server_smoke"]["exact_prefix_hit_observed"]
        and health_after.get("status") == "healthy"
        and health_after.get("apc_enabled") is True
    )
    if "acceptance" in artifact:
        artifact["acceptance"]["openai_server_smoke_completed"] = artifact[
            "server_smoke"
        ]["accepted"]
        artifact["accepted"] = all(artifact["acceptance"].values())
        artifact["complete"] = artifact["accepted"]
        if artifact["complete"]:
            artifact["last_completed_phase"] = "complete"
    _atomic_write(args.output, artifact)
    return artifact


def _contexts(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part) for part in value.split(",") if part.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("contexts must be positive comma-separated integers")
    if tuple(sorted(set(parsed))) != parsed:
        raise argparse.ArgumentTypeError("contexts must be unique and increasing")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contexts", type=_contexts, default=CONTEXTS)
    parser.add_argument(
        "--phase",
        choices=("model", "server-smoke"),
        default="model",
        help="run model qualification or merge a separately launched HTTP smoke",
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--http-timeout", type=float, default=1800.0)
    parser.add_argument("--wired-limit-bytes", type=int, default=440_000_000_000)
    parser.add_argument("--cache-limit-bytes", type=int, default=32_000_000_000)
    args = parser.parse_args(argv)
    args.model = args.model.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    artifact = _load_existing_or_initialize(args.output, args.model, args.contexts)
    _atomic_write(args.output, artifact)
    if args.phase == "server-smoke":
        try:
            _run_server_smoke(args, artifact)
        except (OSError, urllib.error.URLError) as error:
            artifact["server_smoke"] = {
                "completed": False,
                "error": f"{type(error).__name__}: {error}",
            }
            _atomic_write(args.output, artifact)
            raise
        return 0
    _run_model_qualification(args, artifact)
    return 0


if __name__ == "__main__":
    sys.exit(main())
