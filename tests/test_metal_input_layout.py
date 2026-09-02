import math

import pytest

try:
    import mlx.core as mx
except ImportError:
    pytest.skip("MLX/Metal is unavailable", allow_module_level=True)

if not mx.metal.is_available():
    pytest.skip("Metal is unavailable", allow_module_level=True)

from glm53_flash_mlx.fp8 import (
    DirectFP8Expert,
    _metal_input,
    _selected_down,
    _selected_projection,
    block_fp8_linear,
)
from glm53_flash_mlx.grouped_fp8 import grouped_fp8_linear
from glm53_flash_mlx.packed import (
    PackedFP8ExpertBank,
    _packed_selected_down,
    _packed_selected_projection,
)

TOKEN_COUNTS = (1, 7, 8, 9, 127, 128, 129, 255, 256, 1023, 1024)
WIDTH = 128


def _feature_interleaved(rows: int, width: int, *, dtype=mx.bfloat16):
    wide = mx.arange(rows * width * 2, dtype=mx.float32).reshape(
        rows, width * 2
    )
    return (wide[:, 1::2] * 0.0009765625).astype(dtype)


def _token_interleaved(rows: int, width: int):
    wide = mx.arange(rows * width * 2, dtype=mx.float32).reshape(
        rows * 2, width
    )
    return (wide[::2] * 0.0009765625).astype(mx.bfloat16)


def _offset_rows(rows: int, width: int):
    storage = mx.arange((rows + 1) * width, dtype=mx.float32).reshape(
        rows + 1, width
    )
    return (storage[1:] * 0.0009765625).astype(mx.bfloat16)


def _strided_weight(*shape):
    wide_shape = (*shape[:-1], shape[-1] * 2)
    wide = (mx.arange(math.prod(wide_shape), dtype=mx.uint32) % 247).reshape(
        wide_shape
    )
    return wide[..., 1::2].astype(mx.uint8)


def _strided_scale(*shape):
    wide_shape = (*shape[:-1], shape[-1] * 2)
    wide = mx.arange(math.prod(wide_shape), dtype=mx.float32).reshape(wide_shape)
    return (wide[..., 1::2] * 0.00001 + 0.001).astype(mx.float32)


def _strided_descriptor(values):
    values = mx.array(values, dtype=mx.uint32)
    wide = mx.zeros((values.shape[0] * 2,), dtype=mx.uint32)
    wide[1::2] = values
    return wide[1::2]


def _assert_byte_identical(actual, expected):
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
@pytest.mark.parametrize(
    "activation_factory",
    (_feature_interleaved, _token_interleaved, _offset_rows),
)
def test_block_fp8_all_strided_inputs_match_row_contiguous(
    tokens, activation_factory
):
    activation = activation_factory(tokens, WIDTH)
    weight = _strided_weight(WIDTH, WIDTH)
    scale = _strided_scale(1, 1)
    expected = block_fp8_linear(
        _metal_input(activation),
        _metal_input(weight),
        _metal_input(scale),
    )
    actual = block_fp8_linear(activation, weight, scale)
    _assert_byte_identical(actual, expected)


def _strided_experts():
    actual = []
    expected = []
    for _ in range(8):
        strided = DirectFP8Expert(WIDTH, WIDTH, 3.0)
        contiguous = DirectFP8Expert(WIDTH, WIDTH, 3.0)
        for name in ("gate_proj", "up_proj", "down_proj"):
            left = getattr(strided, name)
            right = getattr(contiguous, name)
            left.weight = _strided_weight(WIDTH, WIDTH)
            left.weight_scale_inv = _strided_scale(1, 1)
            right.weight = _metal_input(left.weight)
            right.weight_scale_inv = _metal_input(left.weight_scale_inv)
        actual.append(strided)
        expected.append(contiguous)
    return actual, expected


def test_selected_top8_projection_and_down_accept_strided_buffers():
    actual_experts, expected_experts = _strided_experts()
    x = _feature_interleaved(1, WIDTH)[0]
    expected_projection = _selected_projection(
        _metal_input(x), expected_experts, "gate_proj"
    )
    actual_projection = _selected_projection(x, actual_experts, "gate_proj")
    _assert_byte_identical(actual_projection, expected_projection)

    hidden = _token_interleaved(8, WIDTH)
    scores = _feature_interleaved(1, 8, dtype=mx.float32)[0]
    expected_down = _selected_down(
        _metal_input(hidden), scores, expected_experts
    )
    actual_down = _selected_down(hidden, scores, actual_experts)
    _assert_byte_identical(actual_down, expected_down)


def _packed_bank_from_strided_inputs():
    from glm53_flash_mlx.ownership import owned_tensor

    return PackedFP8ExpertBank(
        owned_tensor(_strided_weight(8, WIDTH * 2, WIDTH)),
        owned_tensor(_strided_scale(8, 2, 1)),
        owned_tensor(_strided_weight(8, WIDTH, WIDTH)),
        owned_tensor(_strided_scale(8, 1, 1)),
        intermediate_size=WIDTH,
    )


def test_packed_selected_top8_normalizes_bank_activation_and_expert_ids():
    from glm53_flash_mlx.ownership import TensorLayout, owned_tensor

    bank = _packed_bank_from_strided_inputs()
    contiguous_bank = PackedFP8ExpertBank(
        owned_tensor(
            _metal_input(bank.gate_up_weight),
            layout=TensorLayout.ROW_MAJOR_CONTIGUOUS,
        ),
        owned_tensor(
            _metal_input(bank.gate_up_scale_inv),
            layout=TensorLayout.ROW_MAJOR_CONTIGUOUS,
        ),
        owned_tensor(
            _metal_input(bank.down_weight),
            layout=TensorLayout.ROW_MAJOR_CONTIGUOUS,
        ),
        owned_tensor(
            _metal_input(bank.down_scale_inv),
            layout=TensorLayout.ROW_MAJOR_CONTIGUOUS,
        ),
        intermediate_size=WIDTH,
    )
    expert_ids = _strided_descriptor([7, 1, 5, 0, 3, 6, 2, 4])
    x = _feature_interleaved(1, WIDTH)[0]
    for row_offset in (0, WIDTH):
        actual = _packed_selected_projection(
            x, expert_ids, bank, row_offset=row_offset
        )
        expected = _packed_selected_projection(
            _metal_input(x),
            _metal_input(expert_ids),
            contiguous_bank,
            row_offset=row_offset,
        )
        _assert_byte_identical(actual, expected)

    hidden = _token_interleaved(8, WIDTH)
    scores = _feature_interleaved(1, 8, dtype=mx.float32)[0]
    actual_down = _packed_selected_down(hidden, scores, expert_ids, bank)
    expected_down = _packed_selected_down(
        _metal_input(hidden),
        scores,
        _metal_input(expert_ids),
        contiguous_bank,
    )
    _assert_byte_identical(actual_down, expected_down)


@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
@pytest.mark.parametrize(
    "activation_factory",
    (_feature_interleaved, _token_interleaved, _offset_rows),
)
def test_grouped_fp8_strided_activation_weight_scale_and_descriptors(
    tokens, activation_factory
):
    activation = activation_factory(tokens, WIDTH)
    weight = _strided_weight(1, WIDTH, WIDTH)
    scale = _strided_scale(1, 1, 1)
    tiles = math.ceil(tokens / 32)
    experts = _strided_descriptor([0] * tiles)
    starts = _strided_descriptor([tile * 32 for tile in range(tiles)])
    lengths = _strided_descriptor(
        [min(32, tokens - tile * 32) for tile in range(tiles)]
    )
    plan = (experts, starts, lengths)
    expected = grouped_fp8_linear(
        _metal_input(activation),
        tuple(_metal_input(value) for value in plan),
        _metal_input(weight),
        _metal_input(scale),
    )
    actual = grouped_fp8_linear(activation, plan, weight, scale)
    _assert_byte_identical(actual, expected)
