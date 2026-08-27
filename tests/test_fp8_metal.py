import math

import pytest


def test_block_fp8_kernel_matches_dequantized_matmul():
    try:
        import mlx.core as mx
    except ImportError as exc:
        pytest.skip(f"MLX/Metal is unavailable in this session: {exc}")
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    from glm53_flash_mlx.fp8 import block_fp8_linear

    # Dimensions cross both output and reduction block boundaries.
    m, k = 129, 257
    codes = mx.random.randint(0, 247, shape=(m, k)).astype(mx.uint8)
    scales = mx.random.uniform(shape=(math.ceil(m / 128), math.ceil(k / 128))).astype(mx.float32) * 0.02
    x = mx.random.normal(shape=(2, k)).astype(mx.bfloat16)
    actual = block_fp8_linear(x, codes, scales)

    dense = mx.from_fp8(codes, dtype=mx.float32)
    expanded = mx.repeat(mx.repeat(scales, 128, axis=0), 128, axis=1)[:m, :k]
    expected = (x.astype(mx.float32) @ (dense * expanded).T).astype(mx.bfloat16)
    mx.eval(actual, expected)
    error = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()
    assert error <= 0.03125
