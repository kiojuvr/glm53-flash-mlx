from pathlib import Path

from glm53_flash_mlx.server import configure_m3_ultra


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
    )
    import os
    assert os.environ["MLX_VLM_MAX_NUM_SEQS"] == "1"
    assert os.environ["PREFILL_STEP_SIZE"] == "2048"
    assert os.environ["APC_BLOCK_SIZE"] == "64"
    assert os.environ["APC_NUM_BLOCKS"] == "64"
    assert os.environ["GLM53_WARM_RESIDENCY"] == "1"
