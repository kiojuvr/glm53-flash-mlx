from pathlib import Path

import pytest

from glm53_flash_mlx.server import configure_m3_ultra, validate_admission


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
        max_prompt_tokens=256, max_context_tokens=16384,
    )
    import os
    assert os.environ["MLX_VLM_MAX_NUM_SEQS"] == "1"
    assert os.environ["PREFILL_STEP_SIZE"] == "2048"
    assert os.environ["APC_BLOCK_SIZE"] == "64"
    assert os.environ["APC_NUM_BLOCKS"] == "64"
    assert os.environ["GLM53_WARM_RESIDENCY"] == "1"
    assert os.environ["GLM53_MAX_PROMPT_TOKENS"] == "256"
    assert os.environ["MAX_KV_SIZE"] == "16384"


def test_prompt_and_total_context_admission_are_independent():
    validate_admission(256, 4096, max_prompt_tokens=256, max_context_tokens=16384)
    with pytest.raises(ValueError, match="bounded-prefill"):
        validate_admission(257, 1, max_prompt_tokens=256, max_context_tokens=16384)
    with pytest.raises(ValueError, match="total context"):
        validate_admission(256, 16129, max_prompt_tokens=256, max_context_tokens=16384)
