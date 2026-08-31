import importlib.util
from pathlib import Path

import pytest

try:
    import mlx.core  # noqa: F401
except ImportError:
    pytest.skip("MLX/Metal is unavailable", allow_module_level=True)


def _load_probe():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "probe_kpool4_kv_dtype_separation.py"
    )
    spec = importlib.util.spec_from_file_location("kpool_dtype_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_keeps_fp8_latent_probe_only_and_indexpool_bf16():
    probe = _load_probe()
    source = Path(probe.__file__).read_text()
    assert probe.INDEX_TOPK == 2048
    assert probe.INDEX_KPOOL == 4
    assert probe.ARMS == ("bf16", "fp8_per_token_head", "fp8_group64")
    assert '"fp8_scope": "probe-only; no runtime cache backend or ABI"' in source
    assert '"pool_keys": "bfloat16"' in source
    assert "NOPE_DSA_CACHE_ABI_DIRECT" in source
    assert "NOPE_DSA_CACHE_ABI_COMPACT" in source
    assert '"fp8_latent_backend_registered": False' in source


def test_context_and_tail_frontiers_cover_required_boundaries():
    probe = _load_probe()
    assert probe.BYPASS_CONTEXTS == (2047, 2048, 2049)
    assert set(probe.TAIL_CONTEXTS) == {4349, 4350, 4351, 4352}
    assert set(probe.CONTEXTS) >= {
        2049, 4351, 4352, 16384, 65536, 131072, 262144
    }


def test_quantized_storage_is_e4m3_codes_with_independent_scales():
    probe = _load_probe()
    import mlx.core as mx

    latent = mx.sin(mx.arange(2 * 512, dtype=mx.float32)).reshape(
        1, 1, 2, 512
    ).astype(mx.bfloat16)
    token_codes, token_scale = probe._quantize_per_token_head(latent)
    group_codes, group_scale = probe._quantize_group64(latent)
    mx.eval(token_codes, token_scale, group_codes, group_scale)
    assert token_codes.dtype == group_codes.dtype == mx.uint8
    assert token_scale.dtype == group_scale.dtype == mx.float32
    assert token_scale.shape == (1, 1, 2, 1)
    assert group_scale.shape == (1, 1, 2, 8)
