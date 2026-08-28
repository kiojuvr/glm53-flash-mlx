#!/usr/bin/env python3
"""Cold/warm official-FP8 decode probe for the M3 Ultra runtime."""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx

from glm53_flash_mlx.loader import load_model, prefault_checkpoint, warm_residency


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--tokens", type=int, default=3)
    p.add_argument("--prefault", action="store_true")
    p.add_argument("--warm-residency", action="store_true")
    p.add_argument("--experimental-packed-grouped-moe", action="store_true")
    args = p.parse_args()
    mx.set_wired_limit(int(440e9))
    mx.set_cache_limit(int(32e9))
    if args.prefault:
        t = time.time()
        count = prefault_checkpoint(args.model)
        print(json.dumps({"phase": "prefault", "bytes": count, "seconds": time.time() - t}), flush=True)
    t = time.time()
    model, _ = load_model(
        args.model,
        strict=True,
        experimental_packed_grouped_moe=args.experimental_packed_grouped_moe,
    )
    print(json.dumps({"phase": "load", "seconds": time.time() - t}), flush=True)
    if args.warm_residency:
        t = time.time()
        resident_bytes = warm_residency(model)
        print(
            json.dumps(
                {
                    "phase": "warm_residency",
                    "bytes": resident_bytes,
                    "seconds": time.time() - t,
                    "active_gb": mx.get_active_memory() / 1e9,
                }
            ),
            flush=True,
        )
    cache = model.make_cache()
    token = 1
    for step in range(args.tokens):
        t = time.time()
        output = model(mx.array([[token]], dtype=mx.uint32), cache=cache)
        token = int(mx.argmax(output.logits[0, -1]).item())
        elapsed = time.time() - t
        print(
            json.dumps(
                {
                    "phase": "decode",
                    "step": step,
                    "token": token,
                    "seconds": elapsed,
                    "tok_s": 1.0 / elapsed,
                    "peak_gb": mx.get_peak_memory() / 1e9,
                    "active_gb": mx.get_active_memory() / 1e9,
                    "cache_gb": mx.get_cache_memory() / 1e9,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
