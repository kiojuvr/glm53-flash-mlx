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

from .abi import (
    CACHE_IDENTITY_SCHEMA,
    GROUPED_KERNEL_ABI,
    GROUPED_MIN_ROUTES,
    KERNEL_ABI_VERSION,
    MLX_VLM_REVISION,
    NOPE_DSA_CACHE_ABI_COMPACT,
    NOPE_DSA_CACHE_ABI_DIRECT,
    PACKED_DECODE_KERNEL_ABI,
    PACKED_EXPERT_BANK_ABI,
)
from .manifest import (
    ManifestError,
    OfficialRevisionIdentity,
    attest_revision_identity,
    inspect_checkpoint,
)
from .materialization import (
    MATERIALIZATION_INTERVAL_TOKENS,
    MATERIALIZATION_POLICY,
    install_bounded_recurrent_materialization_policy,
    materialization_snapshot,
)
from .patch import apply_runtime_patch, patch_status

DEFAULT_SOURCE = Path("/Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash")
DEFAULT_MODEL = DEFAULT_SOURCE
# Archived probes import this value to describe the historical safety gate
# they measured.  It is not part of production admission.
LEGACY_PROBE_MAX_PROMPT_TOKENS = 256
DEFAULT_MAX_GENERATION_TOKENS = 4_096
DEFAULT_MAX_CONTEXT_TOKENS = 36_864
DEFAULT_FIRST_TOKEN_TIMEOUT_SECONDS = 1_800.0
QUALIFIED_PROMPT_TOKENS = 32_768
ADMISSION_POLICY = "prompt-plus-generation-v1"
EXACT_APC_STORE_POLICY = "aligned-guarded-prefill-checkpoint-v1"
EXACT_APC_PREFIX_GUARD_TOKENS = 16


def admission_snapshot(
    *, max_generation_tokens: int, max_context_tokens: int
) -> dict[str, int | str]:
    """Return the auditable production admission contract."""
    return {
        "policy": ADMISSION_POLICY,
        "max_context_tokens": int(max_context_tokens),
        "max_generation_tokens": int(max_generation_tokens),
        "max_prompt_tokens_at_max_generation": (
            int(max_context_tokens) - int(max_generation_tokens)
        ),
    }


def _disable_unsafe_full_prompt_exact_harvest(prompt_batch) -> bool:
    """Keep exact APC stores on the authoritative guarded prefill boundary.

    ``PromptProcessingBatch.prompt_step()`` publishes the checkpoint before
    the final guarded suffix.  Its later ``generate()`` harvest runs after the
    final prefill forward has also produced the first generated token.  That
    state cannot be persisted under the full-prompt identity.  Block APC is
    unaffected.
    """
    if (
        getattr(prompt_batch, "_apc_manager", None) is None
        or getattr(prompt_batch, "_apc_mode", None) != "exact"
    ):
        return False
    prompt_batch._apc_harvest_enabled = False
    return True


def _align_exact_apc_checkpoint(
    safe_checkpoint_tokens: int,
    *,
    alignment_tokens: int,
) -> int:
    """Align an upstream-safe checkpoint to the absolute prefill geometry."""
    safe = max(0, int(safe_checkpoint_tokens))
    alignment = int(alignment_tokens)
    if alignment <= 0:
        raise ValueError("exact APC checkpoint alignment must be positive")
    return (safe // alignment) * alignment


def validate_cache_apc_policy(
    *,
    apc: bool,
    apc_disk_path: Path | None,
    experimental_disk_apc: bool,
    experimental_compact_nope_dsa_cache: bool,
) -> None:
    """Fail closed before loading weights for unsupported disk cache layouts."""
    if apc_disk_path is None:
        return
    if not apc or not experimental_disk_apc:
        raise ValueError(
            "disk APC is experimental; require both --apc and "
            "--experimental-disk-apc"
        )
    if experimental_compact_nope_dsa_cache:
        raise ValueError(
            "compact NoPE DSA disk APC is not implemented; "
            "use RAM APC or disable the compact cache"
        )


def validate_admission(
    prompt_tokens: int,
    requested_generation_tokens: int,
    *,
    max_generation_tokens: int,
    max_context_tokens: int,
    max_prompt_tokens: int | None = None,
) -> None:
    """Validate generation and the authoritative total-context budget.

    ``max_prompt_tokens`` is a compatibility-only explicit override for old
    probe fixtures.  The production server omits it and admits prompts solely
    through ``prompt + requested generation <= context capacity``.
    """
    if prompt_tokens < 0:
        raise ValueError("prompt tokens must be non-negative")
    if max_prompt_tokens is not None and prompt_tokens > max_prompt_tokens:
        raise ValueError(
            f"prompt has {prompt_tokens} tokens, but the explicit prompt override "
            f"is limited to {max_prompt_tokens}"
        )
    if requested_generation_tokens < 0:
        raise ValueError("requested generation tokens must be non-negative")
    if requested_generation_tokens > max_generation_tokens:
        raise ValueError(
            f"request asks for {requested_generation_tokens} generation tokens, "
            f"but the configured generation limit is {max_generation_tokens}"
        )
    requested = prompt_tokens + requested_generation_tokens
    if requested > max_context_tokens:
        raise ValueError(
            f"request needs {requested} total tokens "
            f"({prompt_tokens} prompt + {requested_generation_tokens} generation), "
            f"but the configured total context limit is {max_context_tokens}"
        )


def _disk_cache_descriptor(
    revision_identity: str | OfficialRevisionIdentity,
) -> dict:
    """Build the auditable descriptor behind the disk APC namespace hash."""
    cache_backend = os.environ.get("GLM53_CACHE_BACKEND", "direct")
    if isinstance(revision_identity, OfficialRevisionIdentity):
        revision_descriptor = revision_identity.descriptor()
        checkpoint_namespace = revision_identity.namespace_sha256
    else:
        # Preserve the narrow helper contract used by existing unit fixtures.
        # Production startup always supplies the fully split identity above.
        revision_descriptor = {}
        checkpoint_namespace = revision_identity
    descriptor = {
        "schema": CACHE_IDENTITY_SCHEMA,
        "checkpoint_content_sha256": checkpoint_namespace,
        **revision_descriptor,
        "mlx_vlm_revision": MLX_VLM_REVISION,
        "metal_kernel_abi": KERNEL_ABI_VERSION,
        "attention_cache_abi": (
            NOPE_DSA_CACHE_ABI_COMPACT
            if cache_backend == "compact-nope-dsa"
            else NOPE_DSA_CACHE_ABI_DIRECT
        ),
        "cache_backend": cache_backend,
        "apc_hash": "sha256",
        "apc_block_size": 64,
        "exact_apc_store_policy": EXACT_APC_STORE_POLICY,
        "exact_apc_prefix_guard_tokens": EXACT_APC_PREFIX_GUARD_TOKENS,
        "exact_apc_checkpoint_alignment_tokens": int(
            os.environ.get("PREFILL_STEP_SIZE", "2048")
        ),
        "kv_bits": os.environ.get("KV_BITS"),
        "kv_key_bits": os.environ.get("KV_KEY_BITS"),
        "kv_value_bits": os.environ.get("KV_VALUE_BITS"),
        "kv_group_size": os.environ.get("KV_GROUP_SIZE"),
        "kv_quant_scheme": os.environ.get("KV_QUANT_SCHEME", "uniform"),
        "quantized_kv_start": os.environ.get("QUANTIZED_KV_START"),
        "moe_backend": os.environ.get("GLM53_MOE_BACKEND", "direct"),
    }
    if descriptor["moe_backend"] in {"packed-decode", "packed-grouped"}:
        descriptor.update(
            {
                "packed_bank_abi": PACKED_EXPERT_BANK_ABI,
                "packed_decode_kernel_abi": PACKED_DECODE_KERNEL_ABI,
            }
        )
    if descriptor["moe_backend"] == "packed-grouped":
        descriptor.update(
            {
                "grouped_kernel_abi": GROUPED_KERNEL_ABI,
                "grouped_min_routes": GROUPED_MIN_ROUTES,
            }
        )
    return descriptor


def _disk_cache_identity(
    revision_identity: str | OfficialRevisionIdentity,
) -> str:
    """Content-bound identity for the explicitly experimental disk APC."""
    return hashlib.sha256(
        json.dumps(
            _disk_cache_descriptor(revision_identity),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
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
    experimental_packed_decode_moe: bool,
    experimental_packed_grouped_moe: bool,
    experimental_compact_nope_dsa_cache: bool,
    max_context_tokens: int,
) -> None:
    """Set runtime knobs before mlx-vlm imports its server configuration."""
    if experimental_packed_decode_moe and experimental_packed_grouped_moe:
        raise ValueError(
            "--experimental-packed-decode-moe and "
            "--experimental-packed-grouped-moe are mutually exclusive"
        )
    os.environ["MLX_VLM_PRELOAD_MODEL"] = str(model)
    os.environ["MLX_VLM_MAX_NUM_SEQS"] = "1"
    # Production policy: do not allow an inherited 0 or oversized value to
    # disable recurrent-state graph bounding.
    os.environ["MLX_VLM_BATCH_CACHE_EVAL_INTERVAL"] = str(
        MATERIALIZATION_INTERVAL_TOKENS
    )
    os.environ["PREFILL_STEP_SIZE"] = str(prefill_step_size)
    os.environ["MLX_VLM_MAX_TOKENS"] = str(max_tokens)
    os.environ.pop("GLM53_MAX_PROMPT_TOKENS", None)
    os.environ["GLM53_ADMISSION_POLICY"] = ADMISSION_POLICY
    os.environ["GLM53_MAX_GENERATION_TOKENS"] = str(max_tokens)
    os.environ["GLM53_COMPACT_CACHE_CAPACITY_TOKENS"] = str(
        max_context_tokens
    )
    os.environ["MAX_KV_SIZE"] = str(max_context_tokens)
    # A qualified 32K cold prefill takes about 750 seconds on M3 Ultra.
    # Production admission must not be defeated by mlx-vlm's 600s default or
    # by a shorter inherited user environment value.
    os.environ["MLX_VLM_TOKEN_QUEUE_TIMEOUT"] = str(
        DEFAULT_FIRST_TOKEN_TIMEOUT_SECONDS
    )
    os.environ.setdefault("MLX_VLM_LOG_PROGRESS_INTERVAL", "16")
    os.environ.setdefault("MLX_VLM_ENABLE_THINKING", "1")
    os.environ.setdefault("MLX_VLM_VISION_CACHE_SIZE", "8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["GLM53_WARM_RESIDENCY"] = "1" if warm_residency else "0"
    os.environ["GLM53_EXPERIMENTAL_PACKED_GROUPED_MOE"] = (
        "1" if experimental_packed_grouped_moe else "0"
    )
    os.environ["GLM53_EXPERIMENTAL_PACKED_DECODE_MOE"] = (
        "1" if experimental_packed_decode_moe else "0"
    )
    os.environ["GLM53_MOE_BACKEND"] = (
        "packed-decode"
        if experimental_packed_decode_moe
        else "packed-grouped"
        if experimental_packed_grouped_moe
        else "direct"
    )
    os.environ["GLM53_EXPERIMENTAL_COMPACT_NOPE_DSA_CACHE"] = (
        "1" if experimental_compact_nope_dsa_cache else "0"
    )
    os.environ["GLM53_CACHE_BACKEND"] = (
        "compact-nope-dsa" if experimental_compact_nope_dsa_cache else "direct"
    )
    if api_key:
        os.environ["MLX_VLM_SERVER_API_KEY"] = api_key
    if apc:
        os.environ["APC_ENABLED"] = "1"
        os.environ["APC_BLOCK_SIZE"] = "64"
        os.environ["APC_NUM_BLOCKS"] = str(apc_blocks)
        os.environ["APC_HASH"] = "sha256"
        os.environ["APC_EXACT_PREFIX_GUARD_TOKENS"] = str(
            EXACT_APC_PREFIX_GUARD_TOKENS
        )
        if apc_disk_path is not None:
            apc_disk_path.mkdir(parents=True, exist_ok=True)
            os.environ["APC_DISK_PATH"] = str(apc_disk_path)
    else:
        os.environ["APC_ENABLED"] = "0"


def _install_server_loader() -> None:
    """Make every server load pass through the audited GLM-5.3 patch."""
    apply_runtime_patch()
    from mlx_vlm.generate import ar as mlx_ar
    from mlx_vlm.server import generation, openai
    from mlx_vlm import apc as mlx_apc
    server_app = importlib.import_module("mlx_vlm.server.app")

    install_bounded_recurrent_materialization_policy()

    from .loader import load as direct_load, warm_residency

    def load_patched(path, adapter_path=None, **kwargs):
        inspect_checkpoint(path, require_server_ready=True)
        kwargs["experimental_packed_grouped_moe"] = (
            os.environ.get("GLM53_EXPERIMENTAL_PACKED_GROUPED_MOE") == "1"
        )
        kwargs["experimental_packed_decode_moe"] = (
            os.environ.get("GLM53_EXPERIMENTAL_PACKED_DECODE_MOE") == "1"
        )
        kwargs["experimental_compact_nope_dsa_cache"] = (
            os.environ.get("GLM53_EXPERIMENTAL_COMPACT_NOPE_DSA_CACHE") == "1"
        )
        kwargs["compact_cache_capacity_tokens"] = int(
            os.environ.get("GLM53_COMPACT_CACHE_CAPACITY_TOKENS", "4352")
        )
        loaded = direct_load(path, adapter_path=adapter_path, **kwargs)
        if os.environ.get("GLM53_WARM_RESIDENCY", "1") == "1":
            warm_residency(loaded[0])
        return loaded

    generation.load = load_patched

    batch_generator_type = mlx_ar.BatchGenerator
    if not getattr(
        batch_generator_type._apc_exact_checkpoint_len,
        "_glm53_aligned_exact_apc_checkpoint",
        False,
    ):
        stock_exact_checkpoint_len = batch_generator_type._apc_exact_checkpoint_len

        def aligned_exact_checkpoint_len(batch_generator, token_ids):
            safe_checkpoint = stock_exact_checkpoint_len(batch_generator, token_ids)
            return _align_exact_apc_checkpoint(
                safe_checkpoint,
                alignment_tokens=int(os.environ["PREFILL_STEP_SIZE"]),
            )

        aligned_exact_checkpoint_len._glm53_aligned_exact_apc_checkpoint = True
        batch_generator_type._apc_exact_checkpoint_len = aligned_exact_checkpoint_len

    prompt_batch_type = mlx_ar.PromptProcessingBatch
    if not getattr(
        prompt_batch_type.generate,
        "_glm53_guarded_exact_apc_store",
        False,
    ):
        stock_prompt_generate = prompt_batch_type.generate

        def generate_with_guarded_exact_apc_store(prompt_batch, *args, **kwargs):
            if _disable_unsafe_full_prompt_exact_harvest(prompt_batch):
                logging.getLogger(__name__).debug(
                    "Disabled post-final-prefill exact APC harvest; retaining "
                    "the authoritative %d-token guarded checkpoint",
                    EXACT_APC_PREFIX_GUARD_TOKENS,
                )
            return stock_prompt_generate(prompt_batch, *args, **kwargs)

        generate_with_guarded_exact_apc_store._glm53_guarded_exact_apc_store = True
        prompt_batch_type.generate = generate_with_guarded_exact_apc_store

    stock_budget_check = generation._check_configured_context_budget

    def check_glm53_budget(prompt_tokens, max_tokens):
        generation_limit = int(os.environ["GLM53_MAX_GENERATION_TOKENS"])
        context_limit = int(os.environ["MAX_KV_SIZE"])
        try:
            validate_admission(
                int(prompt_tokens), int(max_tokens or 0),
                max_generation_tokens=generation_limit,
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

    stock_runtime_snapshot = server_app._server_runtime_snapshot

    def runtime_snapshot_with_materialization():
        snapshot = stock_runtime_snapshot()
        snapshot["recurrent_state_materialization"] = materialization_snapshot()
        generation_limit = int(os.environ["GLM53_MAX_GENERATION_TOKENS"])
        context_limit = int(os.environ["MAX_KV_SIZE"])
        snapshot["admission"] = admission_snapshot(
            max_generation_tokens=generation_limit,
            max_context_tokens=context_limit,
        )
        snapshot["exact_apc"] = {
            "store_policy": EXACT_APC_STORE_POLICY,
            "prefix_guard_tokens": EXACT_APC_PREFIX_GUARD_TOKENS,
            "checkpoint_alignment_tokens": int(os.environ["PREFILL_STEP_SIZE"]),
        }
        snapshot["long_prefill"] = {
            "token_queue_timeout_seconds": float(
                os.environ["MLX_VLM_TOKEN_QUEUE_TIMEOUT"]
            ),
        }
        return snapshot

    server_app._server_runtime_snapshot = runtime_snapshot_with_materialization


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
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_GENERATION_TOKENS)
    p.add_argument("--max-context-tokens", type=int, default=DEFAULT_MAX_CONTEXT_TOKENS)
    p.add_argument("--wired-limit-gb", type=float, default=440.0)
    p.add_argument("--cache-limit-gb", type=float, default=32.0)
    p.add_argument("--apc", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--apc-blocks", type=int, default=256)
    p.add_argument("--apc-disk-path", type=Path)
    moe_group = p.add_mutually_exclusive_group()
    moe_group.add_argument(
        "--experimental-packed-decode-moe",
        action="store_true",
        help="pack routed experts and use selected top-8 decode without grouped prefill",
    )
    moe_group.add_argument(
        "--experimental-packed-grouped-moe",
        action="store_true",
        help="pack all routed experts and enable the grouped prefill kernel",
    )
    p.add_argument(
        "--experimental-compact-nope-dsa-cache",
        action="store_true",
        help=(
            "use single-latent and compact authoritative IndexPool caches; "
            "single-sequence and RAM-APC only"
        ),
    )
    p.add_argument(
        "--experimental-disk-apc",
        action="store_true",
        help="allow disk APC using the mandatory attested checkpoint identity",
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
        validate_cache_apc_policy(
            apc=args.apc,
            apc_disk_path=args.apc_disk_path,
            experimental_disk_apc=args.experimental_disk_apc,
            experimental_compact_nope_dsa_cache=(
                args.experimental_compact_nope_dsa_cache
            ),
        )
        report = inspect_checkpoint(args.model, require_server_ready=True)
    except (ManifestError, ValueError) as exc:
        print(f"glm53-serve: {exc}", file=sys.stderr)
        return 2
    if args.max_context_tokens <= 0 or args.max_tokens <= 0:
        print("glm53-serve: token limits must be positive", file=sys.stderr)
        return 2
    if args.max_tokens > args.max_context_tokens:
        print("glm53-serve: max generation tokens cannot exceed total context", file=sys.stderr)
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
        experimental_packed_decode_moe=args.experimental_packed_decode_moe,
        experimental_packed_grouped_moe=args.experimental_packed_grouped_moe,
        experimental_compact_nope_dsa_cache=(
            args.experimental_compact_nope_dsa_cache
        ),
        max_context_tokens=args.max_context_tokens,
    )
    logging.getLogger(__name__).warning(
        "Attesting every checkpoint shard against the pinned official revision"
    )
    try:
        revision_identity = attest_revision_identity(args.model)
    except ManifestError as exc:
        print(f"glm53-serve: {exc}", file=sys.stderr)
        return 2
    if args.apc_disk_path is not None:
        os.environ["GLM53_DISK_APC_IDENTITY"] = _disk_cache_identity(
            revision_identity
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
    logging.getLogger(__name__).info(
        "revision_identity checkpoint=%s tokenizer=%s chat_template=%s",
        report.official_revision,
        report.tokenizer_revision,
        report.chat_template_revision,
    )
    logging.getLogger(__name__).info(
        "moe_backend=%s cache_backend=%s admission=%s context_limit=%d "
        "generation_limit=%d prompt_at_max_generation=%d "
        "first_token_timeout_seconds=%.0f",
        os.environ["GLM53_MOE_BACKEND"],
        os.environ["GLM53_CACHE_BACKEND"],
        ADMISSION_POLICY,
        args.max_context_tokens,
        args.max_tokens,
        args.max_context_tokens - args.max_tokens,
        DEFAULT_FIRST_TOKEN_TIMEOUT_SECONDS,
    )
    logging.getLogger(__name__).info(
        "materialization_policy=%s interval_tokens=%d",
        MATERIALIZATION_POLICY,
        MATERIALIZATION_INTERVAL_TOKENS,
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
