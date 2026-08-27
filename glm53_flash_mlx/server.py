"""M3 Ultra tuned launcher for mlx-vlm's OpenAI-compatible server."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from pathlib import Path

from .manifest import ManifestError, inspect_checkpoint
from .patch import apply_runtime_patch, patch_status

DEFAULT_SOURCE = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_MODEL = DEFAULT_SOURCE


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
) -> None:
    """Set runtime knobs before mlx-vlm imports its server configuration."""
    os.environ["MLX_VLM_PRELOAD_MODEL"] = str(model)
    os.environ["MLX_VLM_MAX_NUM_SEQS"] = "1"
    os.environ["PREFILL_STEP_SIZE"] = str(prefill_step_size)
    os.environ["MLX_VLM_MAX_TOKENS"] = str(max_tokens)
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
    server_app = importlib.import_module("mlx_vlm.server.app")

    from .loader import load as direct_load, warm_residency

    def load_patched(path, adapter_path=None, **kwargs):
        inspect_checkpoint(path, require_server_ready=True)
        loaded = direct_load(path, adapter_path=adapter_path, **kwargs)
        if os.environ.get("GLM53_WARM_RESIDENCY", "1") == "1":
            warm_residency(loaded[0])
        return loaded

    generation.load = load_patched

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
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--wired-limit-gb", type=float, default=440.0)
    p.add_argument("--cache-limit-gb", type=float, default=32.0)
    p.add_argument("--apc", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--apc-blocks", type=int, default=256)
    p.add_argument("--apc-disk-path", type=Path)
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

    configure_m3_ultra(
        model=args.model,
        prefill_step_size=args.prefill_step_size,
        max_tokens=args.max_tokens,
        api_key=args.api_key,
        apc=args.apc,
        apc_blocks=args.apc_blocks,
        apc_disk_path=args.apc_disk_path,
        warm_residency=args.warm_residency,
    )
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
