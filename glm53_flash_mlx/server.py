"""M3 Ultra tuned launcher for mlx-vlm's OpenAI-compatible server."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import sys
from pathlib import Path

from .abi import CACHE_IDENTITY_SCHEMA, KERNEL_ABI_VERSION, MLX_VLM_REVISION
from .manifest import ManifestError, checkpoint_content_digest, inspect_checkpoint
from .patch import apply_runtime_patch, patch_status

DEFAULT_SOURCE = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_MODEL = DEFAULT_SOURCE
DEFAULT_MAX_PROMPT_TOKENS = 256
DEFAULT_MAX_CONTEXT_TOKENS = 16_384


def validate_admission(
    prompt_tokens: int,
    max_generation_tokens: int,
    *,
    max_prompt_tokens: int,
    max_context_tokens: int,
) -> None:
    """Reject expensive prefill separately from total cache capacity."""
    if prompt_tokens > max_prompt_tokens:
        raise ValueError(
            f"prompt has {prompt_tokens} tokens, but this bounded-prefill runtime "
            f"is limited to {max_prompt_tokens}; GPU expert bucketing and sparse "
            "DSA prefill are not implemented"
        )
    requested = prompt_tokens + max(0, max_generation_tokens)
    if requested > max_context_tokens:
        raise ValueError(
            f"request needs {requested} total tokens "
            f"({prompt_tokens} prompt + {max_generation_tokens} generation), "
            f"but the configured total context limit is {max_context_tokens}"
        )


def _disk_cache_identity(model: Path) -> str:
    """Content-bound identity for the explicitly experimental disk APC."""
    content = checkpoint_content_digest(model)
    descriptor = {
        "schema": CACHE_IDENTITY_SCHEMA,
        "checkpoint_content_sha256": content,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "metal_kernel_abi": KERNEL_ABI_VERSION,
        "apc_hash": "sha256",
        "apc_block_size": 64,
        "kv_bits": os.environ.get("KV_BITS"),
        "kv_key_bits": os.environ.get("KV_KEY_BITS"),
        "kv_value_bits": os.environ.get("KV_VALUE_BITS"),
        "kv_group_size": os.environ.get("KV_GROUP_SIZE"),
        "kv_quant_scheme": os.environ.get("KV_QUANT_SCHEME", "uniform"),
        "quantized_kv_start": os.environ.get("QUANTIZED_KV_START"),
    }
    return hashlib.sha256(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def configure_m3_ultra(
    *,
    model: Path,
    prefill_step_size: int,
    max_tokens: int,
    api_key: str | None,
    apc: bool,
    apc_blocks: int,
    apc_disk_path: Path | None,
    warm_residency: bool,
    max_prompt_tokens: int,
    max_context_tokens: int,
) -> None:
    """Set runtime knobs before mlx-vlm imports its server configuration."""
    os.environ["MLX_VLM_PRELOAD_MODEL"] = str(model)
    os.environ["MLX_VLM_MAX_NUM_SEQS"] = "1"
    os.environ["PREFILL_STEP_SIZE"] = str(prefill_step_size)
    os.environ["MLX_VLM_MAX_TOKENS"] = str(max_tokens)
    os.environ["GLM53_MAX_PROMPT_TOKENS"] = str(max_prompt_tokens)
    os.environ["MAX_KV_SIZE"] = str(max_context_tokens)
    os.environ.setdefault("MLX_VLM_LOG_PROGRESS_INTERVAL", "16")
    os.environ.setdefault("MLX_VLM_ENABLE_THINKING", "1")
    os.environ.setdefault("MLX_VLM_VISION_CACHE_SIZE", "8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["GLM53_WARM_RESIDENCY"] = "1" if warm_residency else "0"
    if api_key:
        os.environ["MLX_VLM_SERVER_API_KEY"] = api_key
    if apc:
        os.environ["APC_ENABLED"] = "1"
        os.environ["APC_BLOCK_SIZE"] = "64"
        os.environ["APC_NUM_BLOCKS"] = str(apc_blocks)
        os.environ["APC_HASH"] = "sha256"
        if apc_disk_path is not None:
            apc_disk_path.mkdir(parents=True, exist_ok=True)
            os.environ["APC_DISK_PATH"] = str(apc_disk_path)
    else:
        os.environ["APC_ENABLED"] = "0"


def _install_server_loader() -> None:
    """Make every server load pass through the audited GLM-5.3 patch."""
    apply_runtime_patch()
    from mlx_vlm.server import generation, openai
    from mlx_vlm import apc as mlx_apc
    server_app = importlib.import_module("mlx_vlm.server.app")

    from .loader import load as direct_load, warm_residency

    def load_patched(path, adapter_path=None, **kwargs):
        inspect_checkpoint(path, require_server_ready=True)
        loaded = direct_load(path, adapter_path=adapter_path, **kwargs)
        if os.environ.get("GLM53_WARM_RESIDENCY", "1") == "1":
            warm_residency(loaded[0])
        return loaded

    generation.load = load_patched

    stock_budget_check = generation._check_configured_context_budget

    def check_glm53_budget(prompt_tokens, max_tokens):
        prompt_limit = int(os.environ["GLM53_MAX_PROMPT_TOKENS"])
        context_limit = int(os.environ["MAX_KV_SIZE"])
        try:
            validate_admission(
                int(prompt_tokens), int(max_tokens or 0),
                max_prompt_tokens=prompt_limit,
                max_context_tokens=context_limit,
            )
        except ValueError as exc:
            raise generation.PromptTooLongError(str(exc)) from exc
        return stock_budget_check(prompt_tokens, max_tokens)

    generation._check_configured_context_budget = check_glm53_budget

    stock_disk_namespace = mlx_apc.apc_disk_namespace

    def content_bound_disk_namespace(model_path, **kwargs):
        identity = os.environ.get("GLM53_DISK_APC_IDENTITY")
        if os.environ.get("APC_DISK_PATH") and not identity:
            raise RuntimeError("disk APC requires a content-bound GLM53 cache identity")
        if identity:
            kwargs["weights_fingerprint"] = identity
        return stock_disk_namespace(model_path, **kwargs)

    mlx_apc.apc_disk_namespace = content_bound_disk_namespace

    # Clients should not need to send an absolute local filesystem path as the
    # OpenAI model id. Resolve a stable alias while retaining mlx-vlm's cache.
    stock_get_cached_model = server_app.get_cached_model
    configured_model = os.environ["MLX_VLM_PRELOAD_MODEL"]
    aliases = {"glm-5.3-flash", "GLM-5.3-Flash", configured_model}

    def get_cached_model_aliased(model_path, *args, **kwargs):
        if model_path in aliases:
            model_path = configured_model
        return stock_get_cached_model(model_path, *args, **kwargs)

    server_app.get_cached_model = get_cached_model_aliased
    openai.get_cached_model = get_cached_model_aliased


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="glm53-serve",
        description="Serve GLM-5.3-Flash through OpenAI-compatible endpoints on M3 Ultra.",
    )
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--api-key", default=os.environ.get("GLM53_API_KEY"))
    p.add_argument("--prefill-step-size", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    p.add_argument("--max-context-tokens", type=int, default=DEFAULT_MAX_CONTEXT_TOKENS)
    p.add_argument("--wired-limit-gb", type=float, default=440.0)
    p.add_argument("--cache-limit-gb", type=float, default=32.0)
    p.add_argument("--apc", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--apc-blocks", type=int, default=256)
    p.add_argument("--apc-disk-path", type=Path)
    p.add_argument(
        "--experimental-disk-apc",
        action="store_true",
        help="allow disk APC after hashing every checkpoint shard for cache identity",
    )
    p.add_argument(
        "--warm-residency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="materialize all canonical FP8 text weights before accepting requests",
    )
    p.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        report = inspect_checkpoint(args.model, require_server_ready=True)
    except ManifestError as exc:
        print(f"glm53-serve: {exc}", file=sys.stderr)
        return 2
    if args.max_prompt_tokens <= 0 or args.max_context_tokens <= 0 or args.max_tokens <= 0:
        print("glm53-serve: token limits must be positive", file=sys.stderr)
        return 2
    if args.max_prompt_tokens > args.max_context_tokens:
        print("glm53-serve: max prompt tokens cannot exceed total context", file=sys.stderr)
        return 2
    if args.apc_disk_path is not None:
        if not args.apc or not args.experimental_disk_apc:
            print(
                "glm53-serve: disk APC is experimental; require both --apc and "
                "--experimental-disk-apc",
                file=sys.stderr,
            )
            return 2

    configure_m3_ultra(
        model=args.model,
        prefill_step_size=args.prefill_step_size,
        max_tokens=args.max_tokens,
        api_key=args.api_key,
        apc=args.apc,
        apc_blocks=args.apc_blocks,
        apc_disk_path=args.apc_disk_path,
        warm_residency=args.warm_residency,
        max_prompt_tokens=args.max_prompt_tokens,
        max_context_tokens=args.max_context_tokens,
    )
    if args.apc_disk_path is not None:
        logging.getLogger(__name__).warning(
            "Experimental disk APC: hashing all checkpoint and tokenizer bytes"
        )
        os.environ["GLM53_DISK_APC_IDENTITY"] = _disk_cache_identity(args.model)
    apply_runtime_patch()

    import mlx.core as mx

    try:
        mx.set_wired_limit(int(args.wired_limit_gb * 1_000_000_000))
        mx.set_cache_limit(int(args.cache_limit_gb * 1_000_000_000))
    except Exception as exc:
        logging.getLogger(__name__).warning("Could not apply MLX memory limits: %s", exc)

    _install_server_loader()
    logging.getLogger(__name__).info(
        "checkpoint=%s fingerprint=%s patch=%s",
        report.path,
        report.fingerprint[:16],
        patch_status(),
    )

    import uvicorn
    from mlx_vlm.server import app

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        server_header=False,
        log_level=args.log_level.lower(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
