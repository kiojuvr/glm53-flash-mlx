#!/usr/bin/env python3
"""Reproducible full-model greedy trace for the out-of-CI M3 Ultra gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from glm53_flash_mlx.abi import KERNEL_ABI_VERSION
from glm53_flash_mlx.loader import load, warm_residency
from glm53_flash_mlx.manifest import OFFICIAL_HF_REVISION, inspect_checkpoint

DEFAULT_PROMPT = "Reply with exactly: OK"


def _sha256_array(value: mx.array) -> tuple[str, np.ndarray]:
    value = value.astype(mx.float32)
    mx.eval(value)
    array = np.ascontiguousarray(np.asarray(value), dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest(), array


def build_trace(
    model_path: str,
    *,
    prompt: str,
    tokens: int,
    warm: bool,
    experimental_packed_grouped_moe: bool = False,
    experimental_compact_nope_dsa_cache: bool = False,
) -> dict:
    report = inspect_checkpoint(model_path, require_server_ready=True)
    mx.set_wired_limit(int(440e9))
    mx.set_cache_limit(int(32e9))
    model, processor = load(
        model_path,
        experimental_packed_grouped_moe=experimental_packed_grouped_moe,
        experimental_compact_nope_dsa_cache=experimental_compact_nope_dsa_cache,
    )
    if warm:
        warm_residency(model)
    messages = [{"role": "user", "content": prompt}]
    formatted = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = processor(formatted, return_tensors="np", add_special_tokens=True)
    prompt_ids = np.asarray(encoded["input_ids"], dtype=np.int32).reshape(1, -1)
    cache = model.make_cache()
    started = time.perf_counter()
    output = model(mx.array(prompt_ids), cache=cache)
    steps = []
    generated = []
    for step in range(tokens):
        digest, logits = _sha256_array(output.logits[0, -1])
        top2 = np.argpartition(logits, -2)[-2:]
        top2 = top2[np.argsort(logits[top2])[::-1]]
        token = int(top2[0])
        generated.append(token)
        steps.append(
            {
                "step": step,
                "token": token,
                "logits_f32_sha256": digest,
                "top1_logit": float(logits[top2[0]]),
                "top2_token": int(top2[1]),
                "top1_margin": float(logits[top2[0]] - logits[top2[1]]),
            }
        )
        if step + 1 < tokens:
            output = model(mx.array([[token]], dtype=mx.uint32), cache=cache)
    elapsed = time.perf_counter() - started
    return {
        "schema": "glm53-greedy-oracle-v1",
        "official_hf_revision": OFFICIAL_HF_REVISION,
        "checkpoint_fingerprint": report.fingerprint,
        "checkpoint_layout_digest": report.layout_digest,
        "kernel_abi": KERNEL_ABI_VERSION,
        "moe_backend": getattr(model, "_glm53_moe_backend", "direct"),
        "cache_backend": getattr(model, "_glm53_cache_backend", "direct"),
        "prompt": prompt,
        "formatted_prompt_sha256": hashlib.sha256(formatted.encode()).hexdigest(),
        "prompt_token_count": int(prompt_ids.size),
        "prompt_token_ids_sha256": hashlib.sha256(prompt_ids.tobytes()).hexdigest(),
        "generation_tokens": tokens,
        "generated_token_ids": generated,
        "decoded_text": processor.decode(generated, skip_special_tokens=False),
        "elapsed_seconds": elapsed,
        "steps": steps,
    }


def compare_trace(actual: dict, expected: dict) -> list[str]:
    failures = []
    for key in (
        "schema", "official_hf_revision", "checkpoint_fingerprint",
        "checkpoint_layout_digest", "kernel_abi",
        "prompt", "formatted_prompt_sha256", "prompt_token_count",
        "prompt_token_ids_sha256", "generation_tokens", "generated_token_ids",
    ):
        if actual.get(key) != expected.get(key):
            failures.append(f"{key} mismatch")
    expected_steps = expected.get("steps") or []
    actual_steps = actual.get("steps") or []
    if len(actual_steps) != len(expected_steps):
        failures.append("step count mismatch")
    for index, (got, want) in enumerate(zip(actual_steps, expected_steps)):
        if got.get("token") != want.get("token"):
            failures.append(f"step {index} token mismatch")
        if got.get("logits_f32_sha256") != want.get("logits_f32_sha256"):
            failures.append(f"step {index} logits hash mismatch")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tokens", type=int, choices=(16, 128), default=16)
    parser.add_argument("--warm-residency", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expect", type=Path)
    parser.add_argument("--experimental-packed-grouped-moe", action="store_true")
    parser.add_argument("--experimental-compact-nope-dsa-cache", action="store_true")
    args = parser.parse_args()
    trace = build_trace(
        args.model,
        prompt=args.prompt,
        tokens=args.tokens,
        warm=args.warm_residency,
        experimental_packed_grouped_moe=args.experimental_packed_grouped_moe,
        experimental_compact_nope_dsa_cache=(
            args.experimental_compact_nope_dsa_cache
        ),
    )
    print(json.dumps(trace, indent=2, ensure_ascii=False), flush=True)
    if args.expect:
        failures = compare_trace(trace, json.loads(args.expect.read_text()))
        if failures:
            for failure in failures:
                print(f"oracle mismatch: {failure}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
