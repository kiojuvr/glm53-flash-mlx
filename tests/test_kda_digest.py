from __future__ import annotations

import numpy as np
import pytest

from glm53_flash_mlx.cache_lifecycle import CacheLifecycle
from glm53_flash_mlx.kda_digest import (
    SoakLifecycleAccounting,
    aggregate_layer_digest,
    apc_event_steps,
    compare_layerwise_digests,
    first_kda_state_difference,
    layerwise_kda_digests,
    observation_steps,
    rollback_events,
)


class FakeArraysCache:
    def __init__(self, conv, recurrent):
        self.cache = [conv, recurrent]
        self._left_padding = None
        self._left_padding_advance = 0
        self._lengths = None
        self._lengths_advance = 0

    def __getitem__(self, index):
        return self.cache[index]


def _cache(delta=0):
    caches = [None] * 4
    caches[0] = FakeArraysCache(
        np.arange(12, dtype=np.uint16).reshape(1, 3, 4),
        np.arange(16, dtype=np.float32).reshape(1, 2, 2, 4) + delta,
    )
    caches[2] = FakeArraysCache(
        np.arange(12, dtype=np.uint16).reshape(1, 3, 4),
        np.arange(16, dtype=np.float32).reshape(1, 2, 2, 4),
    )
    return caches


def test_observation_schedule_covers_materialization_and_extra_boundaries():
    assert observation_steps(4_096) == (
        0,
        1,
        255,
        256,
        257,
        *tuple(range(512, 3_841, 256)),
        4_095,
        4_096,
    )
    assert len(
        [step for step in observation_steps(100_000) if step > 0 and step % 256 == 0]
    ) == 390
    assert apc_event_steps(4_096) == (2_048,)
    assert rollback_events(4_096) == ((1_024, 1), (2_048, 8), (3_072, 16))


def test_layerwise_conv_recurrent_and_index_digests_are_independent():
    base = layerwise_kda_digests(_cache(), kda_layers=(0, 2))
    same = layerwise_kda_digests(_cache(), kda_layers=(0, 2))
    changed = layerwise_kda_digests(_cache(delta=1), kda_layers=(0, 2))
    assert base == same
    assert aggregate_layer_digest(base) == aggregate_layer_digest(same)
    difference = compare_layerwise_digests(base, changed)
    assert difference["layer"] == 0
    assert difference["state_kind"] == "recurrent"
    assert base[0]["conv_digest"] == changed[0]["conv_digest"]
    assert base[0]["index_digest"] == changed[0]["index_digest"]


def test_first_difference_reports_coordinate_dtype_and_raw_bits():
    left = _cache()
    right = _cache()
    right[2].cache[0] = right[2].cache[0].copy()
    right[2].cache[0][0, 1, 2] += 1
    difference = first_kda_state_difference(left, right, kda_layers=(0, 2))
    assert difference == {
        "layer": 2,
        "state_kind": "conv",
        "coordinate": [0, 1, 2],
        "dtype": "uint16",
        "raw_dtype": "uint16",
        "left_bits": 6,
        "right_bits": 7,
    }


def test_float_signed_zero_is_localized_as_a_raw_bit_difference():
    left = _cache()
    right = _cache()
    left[0].cache[1][0, 0, 0, 0] = np.float32(0.0)
    right[0].cache[1][0, 0, 0, 0] = np.float32(-0.0)
    difference = first_kda_state_difference(left, right, kda_layers=(0, 2))
    assert difference["coordinate"] == [0, 0, 0, 0]
    assert difference["left_bits"] == 0x00000000
    assert difference["right_bits"] == 0x80000000


def test_lifecycle_accounting_is_explicit_monotonic_and_reclassifiable():
    accounting = SoakLifecycleAccounting()
    accounting.allocate(
        "a-kda",
        CacheLifecycle.ACTIVE_RECURRENT,
        resident_bytes=100,
    )
    accounting.allocate(
        "a-dsa",
        CacheLifecycle.TARGET_PREFIX,
        resident_bytes=200,
        physical_tokens=4_352,
    )
    accounting.allocate(
        "snapshot-kda",
        CacheLifecycle.SNAPSHOT_STATE,
        resident_bytes=100,
    )
    accounting.resize("a-dsa", resident_bytes=220)
    before = accounting.snapshot()
    accounting.release("a-kda")
    accounting.reclassify(
        "snapshot-kda", "a-kda-restored", CacheLifecycle.ACTIVE_RECURRENT
    )
    after = accounting.snapshot()
    assert after["resident_bytes"] == 320
    assert after["anonymous_allocation_count"] == 0
    assert after["cumulative_allocated_bytes"] == before["cumulative_allocated_bytes"]
    assert after["cumulative_allocated_tokens"] == 4_352
    assert after["by_lifecycle"]["active-recurrent"]["resident_bytes"] == 100
    assert after["by_lifecycle"]["snapshot-state"]["resident_bytes"] == 0


def test_accounting_rejects_anonymous_or_implicit_lifecycle():
    accounting = SoakLifecycleAccounting()
    with pytest.raises(ValueError):
        accounting.allocate(
            "", CacheLifecycle.ACTIVE_RECURRENT, resident_bytes=1
        )
    with pytest.raises(TypeError):
        accounting.allocate("x", "active-recurrent", resident_bytes=1)
