import importlib.util
from pathlib import Path

import pytest

try:
    import mlx.core as mx
except ImportError:
    pytest.skip("MLX/Metal is unavailable", allow_module_level=True)

if not mx.metal.is_available():
    pytest.skip("Metal is unavailable", allow_module_level=True)


def _load_probe():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "probe_recurrent_state_materialization_frontier.py"
    )
    spec = importlib.util.spec_from_file_location("materialization_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hash_schedule_contains_all_requested_boundaries():
    probe = _load_probe()
    steps = set(probe.hash_steps(8192))
    required = {
        1,
        49,
        50,
        51,
        127,
        128,
        129,
        255,
        256,
        257,
        511,
        512,
        513,
        8191,
        8192,
    }
    required.update(range(256, 8193, 256))
    assert required <= steps


def test_materialization_counts_match_floor_contract():
    probe = _load_probe()
    assert probe.expected_materialization_count(8192, 0) == 0
    assert probe.expected_materialization_count(8192, 50) == 163
    assert probe.expected_materialization_count(8192, 128) == 64
    assert probe.expected_materialization_count(8192, 256) == 32
    assert probe.expected_materialization_count(8192, 512) == 16


def test_materialization_uses_nested_cache_state_without_changing_it():
    probe = _load_probe()

    class Entry:
        def __init__(self, value):
            self.value = value

        @property
        def state(self):
            return (self.value,)

    cache = [Entry(mx.arange(8)), Entry(mx.arange(4))]
    before = [mx.array(entry.value) for entry in cache]
    probe.materialize_cache(cache)
    for entry, expected in zip(cache, before, strict=True):
        assert mx.array_equal(entry.value, expected).item()


def test_probe_is_explicitly_non_runtime_and_has_no_buffer_count_inference():
    probe = _load_probe()
    source = Path(probe.__file__).read_text()
    assert "metal_buffer_count_api_available" in source
    assert "buffer_count_estimate" not in source
    assert "experimental_compact_nope_dsa_cache=True" in source
    assert "compact_cache_reserve_tokens=args.steps + RESERVE_TAIL" in source
    assert "MLX_VLM_BATCH_CACHE_EVAL_INTERVAL" not in source
