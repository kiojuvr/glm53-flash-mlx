#!/usr/bin/env python3
"""Deterministic admission-limit probe for the bounded prefill path."""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import mlx.core as mx
import numpy as np

from glm53_flash_mlx.loader import load_model, warm_residency
from glm53_flash_mlx.manifest import inspect_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--warm-residency", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--experimental-packed-grouped-moe", action="store_true")
    args = parser.parse_args()
    report = inspect_checkpoint(args.model, require_server_ready=True)
    mx.set_wired_limit(int(440e9))
    mx.set_cache_limit(int(32e9))
    model, config = load_model(
        args.model,
        strict=True,
        experimental_packed_grouped_moe=args.experimental_packed_grouped_moe,
    )
    if args.warm_residency:
        warm_residency(model)
    vocab = int(config["text_config"]["vocab_size"])
    token_ids = ((np.arange(args.tokens, dtype=np.uint64) * 7919) % (vocab - 1024) + 100).astype(np.uint32)
    mx.reset_peak_memory()
    started = time.perf_counter()
    output = model(mx.array(token_ids[None, :]), cache=model.make_cache())
    logits = output.logits[0, -1].astype(mx.float32)
    mx.eval(logits)
    elapsed = time.perf_counter() - started
    logits_np = np.ascontiguousarray(np.asarray(logits), dtype=np.float32)
    result = {
        "schema": "glm53-prefill-probe-v1",
        "official_hf_revision": report.official_revision,
        "checkpoint_fingerprint": report.fingerprint,
        "prompt_tokens": args.tokens,
        "token_ids_sha256": hashlib.sha256(token_ids.tobytes()).hexdigest(),
        "last_logits_f32_sha256": hashlib.sha256(logits_np.tobytes()).hexdigest(),
        "elapsed_seconds": elapsed,
        "prompt_tok_s": args.tokens / elapsed,
        "active_gb": mx.get_active_memory() / 1e9,
        "peak_gb": mx.get_peak_memory() / 1e9,
        "moe_backend": getattr(model, "_glm53_moe_backend", "direct"),
        "implementation": (
            "packed sorted grouped FP8 MoE; full-KV DSA prefill"
            if args.experimental_packed_grouped_moe
            else "tiled-8 FP8 projections; CPU expert bucket; full-KV DSA prefill"
        ),
    }
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
