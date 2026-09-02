"""Layerwise, bit-exact KDA state observation for long-running diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .cache_lifecycle import CacheLifecycle
from .kda_state import (
    KDA_CONV_STATE_SLOT,
    KDA_RECURRENT_STATE_SLOT,
    KDA_STATE_INDEX_CONTRACT,
    KDA_STATE_SENTINEL,
)


LAYERWISE_KDA_DIGEST_SCHEMA = "glm53-layerwise-kda-state-digest-v1"


def steady_active_memory_drift(
    checkpoints: Mapping[str, Mapping],
    *,
    first_steady_step: int = 256,
) -> int:
    """Return the active-memory range after residency initialization.

    Token 1 is an intentional observation boundary, but it precedes the first
    production materialization and may still create lazy graph/residency state.
    Long-run boundedness is therefore measured from the first materialization
    onward.  Raw initialization observations remain in the artifact.
    """

    values = [
        int(row["memory"]["active_bytes"])
        for row in checkpoints.values()
        if int(row["step"]) >= first_steady_step
    ]
    if not values:
        raise ValueError("no steady-state memory checkpoints")
    return max(values) - min(values)


def observation_steps(steps: int, *, interval: int = 256) -> tuple[int, ...]:
    if steps < 1 or interval < 1:
        raise ValueError("steps and interval must be positive")
    values = {0, 1, steps}
    values.update(step for step in (255, 256, 257, 4095, 4096) if step <= steps)
    values.update(range(interval, steps + 1, interval))
    return tuple(sorted(values))


def rollback_events(steps: int) -> tuple[tuple[int, int], ...]:
    """Deterministic 1/8/16-token rollback events for each soak tier."""

    if steps < 256:
        return ()
    if steps <= 4_096:
        candidates = ((1_024, 1), (2_048, 8), (3_072, 16))
    elif steps <= 100_000:
        candidates = (
            (1_024, 1),
            (2_048, 8),
            (3_072, 16),
            (32_768, 1),
            (65_536, 8),
            (98_304, 16),
        )
    else:
        candidates = (
            (1_024, 1),
            (2_048, 8),
            (3_072, 16),
            (65_536, 1),
            (131_072, 8),
            (196_608, 16),
        )
    return tuple((target, trim) for target, trim in candidates if target <= steps)


def apc_event_steps(steps: int) -> tuple[int, ...]:
    # A save/load at the terminal step has no continuation with which to prove
    # snapshot immutability, so cadence events stop strictly before the end.
    values = set(range(4_096, steps, 4_096))
    if steps <= 4_096 and steps >= 2_048:
        values.add(2_048)
    return tuple(sorted(values))


def _canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _raw_storage(value, mx_module=None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    if mx_module is None:
        import mlx.core as mx_module
    if value.dtype == mx_module.bfloat16:
        value = value.view(mx_module.uint16)
    mx_module.eval(value)
    return np.ascontiguousarray(np.asarray(value))


def array_digest(value, mx_module=None) -> str:
    if value is None:
        return _sha256(b"none")
    raw = _raw_storage(value, mx_module)
    descriptor = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "raw_dtype": str(raw.dtype),
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json(descriptor))
    digest.update(raw.tobytes())
    return digest.hexdigest()


def _metadata_array_digest(value, mx_module=None) -> str | None:
    return None if value is None else array_digest(value, mx_module)


def kda_index_metadata(cache, *, layer: int, mx_module=None) -> dict:
    return {
        "layer": int(layer),
        "contract": KDA_STATE_INDEX_CONTRACT,
        "sentinel": KDA_STATE_SENTINEL,
        "capacity": len(cache.cache),
        "conv_slot": KDA_CONV_STATE_SLOT,
        "recurrent_slot": KDA_RECURRENT_STATE_SLOT,
        "slot_present": [value is not None for value in cache.cache],
        "left_padding_advance": int(getattr(cache, "_left_padding_advance", 0)),
        "lengths_advance": int(getattr(cache, "_lengths_advance", 0)),
        "left_padding_digest": _metadata_array_digest(
            getattr(cache, "_left_padding", None), mx_module
        ),
        "lengths_digest": _metadata_array_digest(
            getattr(cache, "_lengths", None), mx_module
        ),
    }


def layerwise_kda_digests(
    cache,
    *,
    kda_layers: Sequence[int],
    mx_module=None,
) -> list[dict]:
    rows = []
    for layer in kda_layers:
        entry = cache[layer]
        conv = entry[KDA_CONV_STATE_SLOT]
        recurrent = entry[KDA_RECURRENT_STATE_SLOT]
        metadata = kda_index_metadata(entry, layer=layer, mx_module=mx_module)
        row = {
            "layer": int(layer),
            "conv_digest": array_digest(conv, mx_module),
            "recurrent_digest": array_digest(recurrent, mx_module),
            "index_digest": _sha256(_canonical_json(metadata)),
        }
        row["layer_digest"] = _sha256(_canonical_json(row))
        rows.append(row)
    return rows


def aggregate_layer_digest(rows: Sequence[Mapping[str, object]]) -> str:
    return _sha256(_canonical_json(list(rows)))


def compare_layerwise_digests(left: Sequence[dict], right: Sequence[dict]) -> dict | None:
    if len(left) != len(right):
        return {
            "state_kind": "layer-count",
            "left": len(left),
            "right": len(right),
        }
    for left_row, right_row in zip(left, right, strict=True):
        if left_row["layer"] != right_row["layer"]:
            return {
                "state_kind": "layer-order",
                "left": left_row["layer"],
                "right": right_row["layer"],
            }
        for kind in ("conv", "recurrent", "index"):
            key = f"{kind}_digest"
            if left_row[key] != right_row[key]:
                return {
                    "layer": left_row["layer"],
                    "state_kind": kind,
                    "left_digest": left_row[key],
                    "right_digest": right_row[key],
                }
    return None


def first_array_difference(left, right, *, mx_module=None) -> dict | None:
    if left is None or right is None:
        return None if left is right else {"coordinate": None, "left": None, "right": None}
    if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
        return {
            "coordinate": None,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "left_dtype": str(left.dtype),
            "right_dtype": str(right.dtype),
        }
    left_raw = _raw_storage(left, mx_module)
    right_raw = _raw_storage(right, mx_module)
    unsigned = {
        1: np.uint8,
        2: np.uint16,
        4: np.uint32,
        8: np.uint64,
    }.get(left_raw.dtype.itemsize)
    if unsigned is None:
        raise TypeError(f"unsupported KDA state itemsize: {left_raw.dtype.itemsize}")
    left_bits = left_raw.view(unsigned)
    right_bits = right_raw.view(unsigned)
    different = np.argwhere(left_bits != right_bits)
    if different.size == 0:
        return None
    coordinate = tuple(int(value) for value in different[0])
    return {
        "coordinate": list(coordinate),
        "dtype": str(left.dtype),
        "raw_dtype": str(left_raw.dtype),
        "left_bits": int(left_bits[coordinate]),
        "right_bits": int(right_bits[coordinate]),
    }


def first_kda_state_difference(
    left_cache,
    right_cache,
    *,
    kda_layers: Sequence[int],
    mx_module=None,
) -> dict | None:
    for layer in kda_layers:
        left = left_cache[layer]
        right = right_cache[layer]
        for kind, slot in (
            ("conv", KDA_CONV_STATE_SLOT),
            ("recurrent", KDA_RECURRENT_STATE_SLOT),
        ):
            difference = first_array_difference(
                left[slot], right[slot], mx_module=mx_module
            )
            if difference is not None:
                return {"layer": int(layer), "state_kind": kind, **difference}
        left_metadata = kda_index_metadata(left, layer=layer, mx_module=mx_module)
        right_metadata = kda_index_metadata(right, layer=layer, mx_module=mx_module)
        if left_metadata != right_metadata:
            differing = next(
                key for key in left_metadata if left_metadata[key] != right_metadata[key]
            )
            return {
                "layer": int(layer),
                "state_kind": "index",
                "metadata_field": differing,
                "left": left_metadata[differing],
                "right": right_metadata[differing],
            }
    return None


@dataclass
class _AccountingEntry:
    lifecycle: CacheLifecycle
    resident_bytes: int
    physical_tokens: int


class SoakLifecycleAccounting:
    """Explicit, payload-free accounting for real soak cache incarnations."""

    def __init__(self):
        self._entries: dict[str, _AccountingEntry] = {}
        self._stats = {
            lifecycle: {
                "resident_bytes": 0,
                "peak_bytes": 0,
                "allocation_count": 0,
                "eviction_count": 0,
                "cumulative_allocated_bytes": 0,
                "cumulative_allocated_tokens": 0,
            }
            for lifecycle in CacheLifecycle
        }

    def allocate(
        self,
        owner: str,
        lifecycle: CacheLifecycle,
        *,
        resident_bytes: int,
        physical_tokens: int = 0,
    ) -> None:
        if not owner or owner in self._entries:
            raise ValueError("soak allocations require a unique explicit owner")
        if not isinstance(lifecycle, CacheLifecycle):
            raise TypeError("soak lifecycle must be explicit")
        resident_bytes = int(resident_bytes)
        physical_tokens = int(physical_tokens)
        if resident_bytes < 0 or physical_tokens < 0:
            raise ValueError("soak allocation values must be non-negative")
        self._entries[owner] = _AccountingEntry(
            lifecycle, resident_bytes, physical_tokens
        )
        row = self._stats[lifecycle]
        row["resident_bytes"] += resident_bytes
        row["peak_bytes"] = max(row["peak_bytes"], row["resident_bytes"])
        row["allocation_count"] += 1
        row["cumulative_allocated_bytes"] += resident_bytes
        row["cumulative_allocated_tokens"] += physical_tokens

    def resize(self, owner: str, *, resident_bytes: int) -> None:
        entry = self._entries[owner]
        resident_bytes = int(resident_bytes)
        if resident_bytes < 0:
            raise ValueError("soak resident bytes must be non-negative")
        delta = resident_bytes - entry.resident_bytes
        row = self._stats[entry.lifecycle]
        row["resident_bytes"] += delta
        row["peak_bytes"] = max(row["peak_bytes"], row["resident_bytes"])
        if delta > 0:
            row["allocation_count"] += 1
            row["cumulative_allocated_bytes"] += delta
        entry.resident_bytes = resident_bytes

    def update_physical_tokens(self, owner: str, *, physical_tokens: int) -> None:
        entry = self._entries[owner]
        physical_tokens = int(physical_tokens)
        if physical_tokens < entry.physical_tokens:
            raise ValueError("soak physical token capacity cannot shrink")
        delta = physical_tokens - entry.physical_tokens
        self._stats[entry.lifecycle]["cumulative_allocated_tokens"] += delta
        entry.physical_tokens = physical_tokens

    def release(self, owner: str) -> None:
        entry = self._entries.pop(owner)
        row = self._stats[entry.lifecycle]
        row["resident_bytes"] -= entry.resident_bytes
        row["eviction_count"] += 1

    def reclassify(self, owner: str, new_owner: str, lifecycle: CacheLifecycle) -> None:
        if not new_owner or new_owner in self._entries:
            raise ValueError("reclassified soak owner must be unique")
        entry = self._entries.pop(owner)
        previous = self._stats[entry.lifecycle]
        current = self._stats[lifecycle]
        previous["resident_bytes"] -= entry.resident_bytes
        current["resident_bytes"] += entry.resident_bytes
        current["peak_bytes"] = max(current["peak_bytes"], current["resident_bytes"])
        entry.lifecycle = lifecycle
        self._entries[new_owner] = entry

    def snapshot(self) -> dict:
        rows = {
            lifecycle.value: dict(self._stats[lifecycle])
            for lifecycle in CacheLifecycle
        }
        return {
            "by_lifecycle": rows,
            "live_owner_count": len(self._entries),
            "anonymous_allocation_count": sum(not owner for owner in self._entries),
            "resident_bytes": sum(row["resident_bytes"] for row in rows.values()),
            "cumulative_allocated_bytes": sum(
                row["cumulative_allocated_bytes"] for row in rows.values()
            ),
            "cumulative_allocated_tokens": sum(
                row["cumulative_allocated_tokens"] for row in rows.values()
            ),
        }
