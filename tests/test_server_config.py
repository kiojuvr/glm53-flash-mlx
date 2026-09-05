from pathlib import Path

import pytest

from glm53_flash_mlx.abi import (
    KERNEL_ABI_VERSION,
    NOPE_DSA_CACHE_ABI,
    NOPE_DSA_CACHE_ABI_COMPACT,
    NOPE_DSA_CACHE_ABI_DIRECT,
)
from glm53_flash_mlx.manifest import (
    APPROVED_CHAT_TEMPLATE_REVISIONS,
    EXPECTED_CHECKPOINT_DIGEST,
    EXPECTED_TOKENIZER_DIGEST,
    revision_identity_from_digests,
)
from glm53_flash_mlx.server import (
    ADMISSION_POLICY,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_FIRST_TOKEN_TIMEOUT_SECONDS,
    DEFAULT_MAX_GENERATION_TOKENS,
    EXACT_APC_PREFIX_GUARD_TOKENS,
    EXACT_APC_STORE_POLICY,
    QUALIFIED_PROMPT_TOKENS,
    _align_exact_apc_checkpoint,
    _disk_cache_descriptor,
    _disk_cache_identity,
    _disable_unsafe_full_prompt_exact_harvest,
    admission_snapshot,
    build_parser,
    configure_m3_ultra,
    validate_admission,
    validate_cache_apc_policy,
)


def test_m3_defaults(monkeypatch, tmp_path):
    for key in (
        "MLX_VLM_PRELOAD_MODEL",
        "MLX_VLM_MAX_NUM_SEQS",
        "PREFILL_STEP_SIZE",
        "APC_ENABLED",
    ):
        monkeypatch.delenv(key, raising=False)
    configure_m3_ultra(
        model=Path("/model"), prefill_step_size=2048, max_tokens=4096,
        api_key=None, apc=True, apc_blocks=64, apc_disk_path=tmp_path / "apc",
        warm_residency=True,
        experimental_packed_decode_moe=False,
        experimental_packed_grouped_moe=False,
        experimental_compact_nope_dsa_cache=False,
        max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
    )
    import os
    assert os.environ["MLX_VLM_MAX_NUM_SEQS"] == "1"
    assert os.environ["MLX_VLM_BATCH_CACHE_EVAL_INTERVAL"] == "256"
    assert os.environ["PREFILL_STEP_SIZE"] == "2048"
    assert os.environ["APC_BLOCK_SIZE"] == "64"
    assert os.environ["APC_NUM_BLOCKS"] == "64"
    assert os.environ["APC_EXACT_PREFIX_GUARD_TOKENS"] == "16"
    assert os.environ["GLM53_WARM_RESIDENCY"] == "1"
    assert "GLM53_MAX_PROMPT_TOKENS" not in os.environ
    assert os.environ["GLM53_ADMISSION_POLICY"] == ADMISSION_POLICY
    assert os.environ["GLM53_MAX_GENERATION_TOKENS"] == "4096"
    assert os.environ["GLM53_COMPACT_CACHE_CAPACITY_TOKENS"] == "36864"
    assert os.environ["MAX_KV_SIZE"] == "36864"
    assert os.environ["MLX_VLM_TOKEN_QUEUE_TIMEOUT"] == "1800.0"
    assert os.environ["GLM53_MOE_BACKEND"] == "direct"
    assert os.environ["GLM53_EXPERIMENTAL_PACKED_DECODE_MOE"] == "0"
    assert os.environ["GLM53_EXPERIMENTAL_PACKED_GROUPED_MOE"] == "0"
    assert os.environ["GLM53_EXPERIMENTAL_COMPACT_NOPE_DSA_CACHE"] == "0"
    assert os.environ["GLM53_CACHE_BACKEND"] == "direct"


def test_production_materialization_interval_overwrites_user_environment(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MLX_VLM_BATCH_CACHE_EVAL_INTERVAL", "0")
    monkeypatch.setenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", "1")
    configure_m3_ultra(
        model=Path("/model"), prefill_step_size=2048, max_tokens=4096,
        api_key=None, apc=False, apc_blocks=64, apc_disk_path=None,
        warm_residency=False,
        experimental_packed_decode_moe=False,
        experimental_packed_grouped_moe=False,
        experimental_compact_nope_dsa_cache=False,
        max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
    )
    import os
    assert os.environ["MLX_VLM_BATCH_CACHE_EVAL_INTERVAL"] == "256"
    assert float(os.environ["MLX_VLM_TOKEN_QUEUE_TIMEOUT"]) == (
        DEFAULT_FIRST_TOKEN_TIMEOUT_SECONDS
    )


def test_packed_grouped_server_backend_is_explicitly_opt_in():
    assert not build_parser().parse_args([]).experimental_packed_grouped_moe
    assert build_parser().parse_args(
        ["--experimental-packed-grouped-moe"]
    ).experimental_packed_grouped_moe


def test_packed_decode_server_backend_is_explicitly_opt_in_and_exclusive():
    parser = build_parser()
    assert not parser.parse_args([]).experimental_packed_decode_moe
    assert parser.parse_args(
        ["--experimental-packed-decode-moe"]
    ).experimental_packed_decode_moe
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--experimental-packed-decode-moe",
                "--experimental-packed-grouped-moe",
            ]
        )


def test_compact_nope_dsa_cache_is_explicitly_opt_in():
    assert not build_parser().parse_args([]).experimental_compact_nope_dsa_cache
    assert build_parser().parse_args(
        ["--experimental-compact-nope-dsa-cache"]
    ).experimental_compact_nope_dsa_cache


def test_disk_cache_identity_separates_direct_and_grouped_moe(monkeypatch):
    assert NOPE_DSA_CACHE_ABI == (
        "glm53-nope-dsa-v1-kv-latent512-sentinel-minus1"
    )
    assert NOPE_DSA_CACHE_ABI == NOPE_DSA_CACHE_ABI_DIRECT
    assert "shared-row-plan" not in NOPE_DSA_CACHE_ABI
    monkeypatch.setenv("GLM53_MOE_BACKEND", "direct")
    direct = _disk_cache_identity("checkpoint-digest")
    monkeypatch.setenv("GLM53_MOE_BACKEND", "packed-grouped")
    grouped = _disk_cache_identity("checkpoint-digest")
    assert direct != grouped
    assert grouped == _disk_cache_identity("checkpoint-digest")
    descriptor = _disk_cache_descriptor("checkpoint-digest")
    assert descriptor["moe_backend"] == "packed-grouped"
    assert descriptor["grouped_kernel_abi"].startswith("glm53-grouped-fp8-")
    assert descriptor["grouped_min_routes"] == 256
    assert descriptor["packed_bank_abi"].startswith("glm53-packed-expert-bank-")
    assert descriptor["packed_decode_kernel_abi"].startswith(
        "glm53-packed-selected8-"
    )
    assert descriptor["attention_cache_abi"] == NOPE_DSA_CACHE_ABI
    assert descriptor["exact_apc_store_policy"] == EXACT_APC_STORE_POLICY
    assert descriptor["exact_apc_prefix_guard_tokens"] == 16
    assert descriptor["exact_apc_checkpoint_alignment_tokens"] == 2048

    monkeypatch.setenv("GLM53_MOE_BACKEND", "direct")
    direct_descriptor = _disk_cache_descriptor("checkpoint-digest")
    assert direct_descriptor["metal_kernel_abi"] == KERNEL_ABI_VERSION
    assert "v4-row-contiguous" in KERNEL_ABI_VERSION
    assert direct_descriptor["attention_cache_abi"] == descriptor[
        "attention_cache_abi"
    ]
    assert direct_descriptor["cache_backend"] == "direct"


def test_disk_cache_descriptor_audits_split_official_revision_identity():
    templates = list(APPROVED_CHAT_TEMPLATE_REVISIONS)
    baseline = revision_identity_from_digests(
        checkpoint_digest=EXPECTED_CHECKPOINT_DIGEST,
        tokenizer_digest=EXPECTED_TOKENIZER_DIGEST,
        chat_template_digest=templates[0],
    )
    candidate = revision_identity_from_digests(
        checkpoint_digest=EXPECTED_CHECKPOINT_DIGEST,
        tokenizer_digest=EXPECTED_TOKENIZER_DIGEST,
        chat_template_digest=templates[1],
    )

    descriptor = _disk_cache_descriptor(candidate)
    assert descriptor["checkpoint_revision"] == candidate.checkpoint_revision
    assert descriptor["checkpoint_digest"] == EXPECTED_CHECKPOINT_DIGEST
    assert descriptor["tokenizer_revision"] == candidate.tokenizer_revision
    assert descriptor["tokenizer_digest"] == EXPECTED_TOKENIZER_DIGEST
    assert descriptor["chat_template_revision"] == candidate.chat_template_revision
    assert descriptor["chat_template_digest"] == templates[1]
    assert descriptor["checkpoint_content_sha256"] == candidate.namespace_sha256
    assert _disk_cache_identity(baseline) != _disk_cache_identity(candidate)


def test_disk_cache_identity_separates_packed_decode_without_grouped_abi(monkeypatch):
    monkeypatch.setenv("GLM53_MOE_BACKEND", "direct")
    direct = _disk_cache_identity("checkpoint-digest")
    monkeypatch.setenv("GLM53_MOE_BACKEND", "packed-decode")
    packed = _disk_cache_identity("checkpoint-digest")
    descriptor = _disk_cache_descriptor("checkpoint-digest")
    assert packed != direct
    assert descriptor["moe_backend"] == "packed-decode"
    assert descriptor["packed_bank_abi"].startswith("glm53-packed-expert-bank-")
    assert descriptor["packed_decode_kernel_abi"].startswith(
        "glm53-packed-selected8-"
    )
    assert "grouped_kernel_abi" not in descriptor
    assert "grouped_min_routes" not in descriptor


def test_disk_cache_identity_separates_compact_cache_and_moe_combinations(monkeypatch):
    monkeypatch.setenv("GLM53_CACHE_BACKEND", "direct")
    monkeypatch.setenv("GLM53_MOE_BACKEND", "direct")
    direct = _disk_cache_identity("checkpoint-digest")

    monkeypatch.setenv("GLM53_CACHE_BACKEND", "compact-nope-dsa")
    compact_direct_moe = _disk_cache_identity("checkpoint-digest")
    compact_descriptor = _disk_cache_descriptor("checkpoint-digest")
    assert compact_descriptor["attention_cache_abi"] == NOPE_DSA_CACHE_ABI_COMPACT
    assert NOPE_DSA_CACHE_ABI_COMPACT.startswith("glm53-nope-dsa-v4-")
    assert "compact-indexpool-v4" in NOPE_DSA_CACHE_ABI_COMPACT
    assert "fixed-absolute-capacity" in NOPE_DSA_CACHE_ABI_COMPACT
    assert "self-contained-ape" in NOPE_DSA_CACHE_ABI_COMPACT
    assert compact_descriptor["cache_backend"] == "compact-nope-dsa"
    assert compact_direct_moe != direct

    monkeypatch.setenv("GLM53_MOE_BACKEND", "packed-grouped")
    compact_grouped_moe = _disk_cache_identity("checkpoint-digest")
    assert compact_grouped_moe != compact_direct_moe
    assert _disk_cache_descriptor("checkpoint-digest")["packed_bank_abi"].startswith(
        "glm53-packed-expert-bank-"
    )


def test_compact_cache_disk_apc_fails_closed_before_model_load(tmp_path):
    with pytest.raises(ValueError, match="compact NoPE DSA disk APC is not implemented"):
        validate_cache_apc_policy(
            apc=True,
            apc_disk_path=tmp_path / "apc",
            experimental_disk_apc=True,
            experimental_compact_nope_dsa_cache=True,
        )
    validate_cache_apc_policy(
        apc=True,
        apc_disk_path=None,
        experimental_disk_apc=False,
        experimental_compact_nope_dsa_cache=True,
    )


def test_production_admission_is_dynamic_within_total_context():
    validate_admission(
        QUALIFIED_PROMPT_TOKENS,
        DEFAULT_MAX_GENERATION_TOKENS,
        max_generation_tokens=DEFAULT_MAX_GENERATION_TOKENS,
        max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
    )
    validate_admission(
        DEFAULT_MAX_CONTEXT_TOKENS - 1,
        1,
        max_generation_tokens=DEFAULT_MAX_GENERATION_TOKENS,
        max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
    )
    with pytest.raises(ValueError, match="total context"):
        validate_admission(
            QUALIFIED_PROMPT_TOKENS + 1,
            DEFAULT_MAX_GENERATION_TOKENS,
            max_generation_tokens=DEFAULT_MAX_GENERATION_TOKENS,
            max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
        )
    with pytest.raises(ValueError, match="total context"):
        validate_admission(
            DEFAULT_MAX_CONTEXT_TOKENS,
            1,
            max_generation_tokens=DEFAULT_MAX_GENERATION_TOKENS,
            max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
        )


def test_legacy_explicit_prompt_override_remains_probe_only():
    validate_admission(
        256,
        1,
        max_generation_tokens=4096,
        max_context_tokens=16384,
        max_prompt_tokens=256,
    )
    with pytest.raises(ValueError, match="explicit prompt override"):
        validate_admission(
            257,
            1,
            max_generation_tokens=4096,
            max_context_tokens=16384,
            max_prompt_tokens=256,
        )


def test_admission_snapshot_and_parser_publish_qualified_defaults():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.max_tokens == 4096
    assert args.max_context_tokens == 36864
    assert not hasattr(args, "max_prompt_tokens")
    assert admission_snapshot(
        max_generation_tokens=args.max_tokens,
        max_context_tokens=args.max_context_tokens,
    ) == {
        "policy": "prompt-plus-generation-v1",
        "max_context_tokens": 36864,
        "max_generation_tokens": 4096,
        "max_prompt_tokens_at_max_generation": 32768,
    }


def test_generation_limit_has_an_independent_boundary():
    validate_admission(
        32768, 4096, max_generation_tokens=4096,
        max_context_tokens=36864,
    )
    with pytest.raises(ValueError, match="generation limit is 4096"):
        validate_admission(
            32768, 4097, max_generation_tokens=4096,
            max_context_tokens=36864,
        )
    with pytest.raises(ValueError, match="generation limit is 4096"):
        validate_admission(
            256, 16000, max_generation_tokens=4096,
            max_context_tokens=36864,
        )


def test_full_prompt_exact_harvest_cannot_replace_guarded_checkpoint():
    from types import SimpleNamespace

    exact = SimpleNamespace(
        _apc_manager=object(),
        _apc_mode="exact",
        _apc_harvest_enabled=True,
    )
    assert _disable_unsafe_full_prompt_exact_harvest(exact)
    assert exact._apc_harvest_enabled is False

    block = SimpleNamespace(
        _apc_manager=object(),
        _apc_mode="block",
        _apc_harvest_enabled=True,
    )
    assert not _disable_unsafe_full_prompt_exact_harvest(block)
    assert block._apc_harvest_enabled is True

    disabled = SimpleNamespace(
        _apc_manager=None,
        _apc_mode="exact",
        _apc_harvest_enabled=True,
    )
    assert not _disable_unsafe_full_prompt_exact_harvest(disabled)
    assert disabled._apc_harvest_enabled is True


@pytest.mark.parametrize(
    ("safe_checkpoint", "expected"),
    [
        (0, 0),
        (15, 0),
        (2047, 0),
        (2048, 2048),
        (8176, 6144),
        (8242, 8192),
        (32752, 30720),
    ],
)
def test_exact_apc_checkpoint_is_aligned_to_prefill_geometry(
    safe_checkpoint, expected
):
    assert _align_exact_apc_checkpoint(
        safe_checkpoint,
        alignment_tokens=2048,
    ) == expected


def test_exact_apc_checkpoint_rejects_invalid_alignment():
    with pytest.raises(ValueError, match="alignment must be positive"):
        _align_exact_apc_checkpoint(8192, alignment_tokens=0)
