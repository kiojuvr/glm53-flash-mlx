from pathlib import Path

import pytest

from glm53_flash_mlx.abi import (
    NOPE_DSA_CACHE_ABI,
    NOPE_DSA_CACHE_ABI_COMPACT,
    NOPE_DSA_CACHE_ABI_DIRECT,
)
from glm53_flash_mlx.server import (
    _disk_cache_descriptor,
    _disk_cache_identity,
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
        experimental_packed_grouped_moe=False,
        experimental_compact_nope_dsa_cache=False,
        max_prompt_tokens=256, max_context_tokens=16384,
    )
    import os
    assert os.environ["MLX_VLM_MAX_NUM_SEQS"] == "1"
    assert os.environ["PREFILL_STEP_SIZE"] == "2048"
    assert os.environ["APC_BLOCK_SIZE"] == "64"
    assert os.environ["APC_NUM_BLOCKS"] == "64"
    assert os.environ["GLM53_WARM_RESIDENCY"] == "1"
    assert os.environ["GLM53_MAX_PROMPT_TOKENS"] == "256"
    assert os.environ["GLM53_MAX_GENERATION_TOKENS"] == "4096"
    assert os.environ["MAX_KV_SIZE"] == "16384"
    assert os.environ["GLM53_MOE_BACKEND"] == "direct"
    assert os.environ["GLM53_EXPERIMENTAL_PACKED_GROUPED_MOE"] == "0"
    assert os.environ["GLM53_EXPERIMENTAL_COMPACT_NOPE_DSA_CACHE"] == "0"
    assert os.environ["GLM53_CACHE_BACKEND"] == "direct"


def test_packed_grouped_server_backend_is_explicitly_opt_in():
    assert not build_parser().parse_args([]).experimental_packed_grouped_moe
    assert build_parser().parse_args(
        ["--experimental-packed-grouped-moe"]
    ).experimental_packed_grouped_moe


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

    monkeypatch.setenv("GLM53_MOE_BACKEND", "direct")
    direct_descriptor = _disk_cache_descriptor("checkpoint-digest")
    assert direct_descriptor["attention_cache_abi"] == descriptor[
        "attention_cache_abi"
    ]
    assert direct_descriptor["cache_backend"] == "direct"


def test_disk_cache_identity_separates_compact_cache_and_moe_combinations(monkeypatch):
    monkeypatch.setenv("GLM53_CACHE_BACKEND", "direct")
    monkeypatch.setenv("GLM53_MOE_BACKEND", "direct")
    direct = _disk_cache_identity("checkpoint-digest")

    monkeypatch.setenv("GLM53_CACHE_BACKEND", "compact-nope-dsa")
    compact_direct_moe = _disk_cache_identity("checkpoint-digest")
    compact_descriptor = _disk_cache_descriptor("checkpoint-digest")
    assert compact_descriptor["attention_cache_abi"] == NOPE_DSA_CACHE_ABI_COMPACT
    assert NOPE_DSA_CACHE_ABI_COMPACT.startswith("glm53-nope-dsa-v3-")
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


def test_prompt_and_total_context_admission_are_independent():
    validate_admission(
        256, 4096, max_prompt_tokens=256, max_generation_tokens=4096,
        max_context_tokens=16384,
    )
    with pytest.raises(ValueError, match="bounded-prefill"):
        validate_admission(
            257, 1, max_prompt_tokens=256, max_generation_tokens=4096,
            max_context_tokens=16384,
        )
    with pytest.raises(ValueError, match="total context"):
        validate_admission(
            256, 4096, max_prompt_tokens=256, max_generation_tokens=4096,
            max_context_tokens=4095,
        )


def test_generation_limit_has_an_independent_boundary():
    validate_admission(
        256, 4096, max_prompt_tokens=256, max_generation_tokens=4096,
        max_context_tokens=16384,
    )
    with pytest.raises(ValueError, match="generation limit is 4096"):
        validate_admission(
            256, 4097, max_prompt_tokens=256, max_generation_tokens=4096,
            max_context_tokens=16384,
        )
    with pytest.raises(ValueError, match="generation limit is 4096"):
        validate_admission(
            256, 16000, max_prompt_tokens=256, max_generation_tokens=4096,
            max_context_tokens=16384,
        )
