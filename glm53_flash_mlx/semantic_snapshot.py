"""Transactional RAM snapshots for a complete GLM-5.3 semantic prefix.

The snapshot contract is deliberately separate from the opportunistic prefix
cache.  A snapshot is an immutable, explicitly owned cache incarnation which
can recreate the semantic boundary immediately after a prefix.  Capture and
restore prepare every component before publishing a single Python reference;
there is no API which installs only KDA, DSA latent, or IndexPool state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Sequence

from .cache_lifecycle import CacheLifecycle, RetentionPolicy
from .materialization import (
    MATERIALIZATION_INTERVAL_TOKENS,
    MATERIALIZATION_POLICY,
)


SEMANTIC_PREFIX_SNAPSHOT_SCHEMA = (
    "glm53-hybrid-semantic-prefix-snapshot-v1-ram-owned-transactional"
)
KDA_STATE_ABI = "glm53-kda-arrays-cache-v1-conv0-recurrent1"
DIRECT_INDEXPOOL_ABI = "glm53-indexpool-direct-v1-kpool4-int64"
COMPACT_INDEXPOOL_ABI = "glm53-compact-indexpool-v4-kpool4-int64-raw19"


class SemanticSnapshotError(ValueError):
    """Raised before a partial semantic snapshot can be published or restored."""


def _nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticSnapshotError(f"{name} must be a non-empty string")
    return value


def _nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SemanticSnapshotError(f"{name} must be a non-negative Python int")
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class SemanticSnapshotIdentity:
    checkpoint_revision: str
    checkpoint_fingerprint: str
    moe_backend: str
    cache_backend: str
    attention_cache_abi: str
    kda_state_abi: str
    indexpool_abi: str
    prefix_token_sha256: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _nonempty_string(name, value)

    def descriptor(self) -> dict[str, str]:
        return asdict(self)

    @property
    def namespace_sha256(self) -> str:
        return _sha256(_canonical_json(self.descriptor()))


@dataclass(frozen=True)
class SemanticSnapshotBoundary:
    absolute_token_position: int
    logical_prefix_length: int
    materialization_epoch: int
    rollback_epoch: int
    active_state_slots: tuple[tuple[int, tuple[int, ...]], ...]
    kv_logical_extents: tuple[tuple[int, int], ...]
    indexpool_logical_extents: tuple[tuple[int, int], ...]
    materialization_policy: str = MATERIALIZATION_POLICY
    materialization_interval_tokens: int = MATERIALIZATION_INTERVAL_TOKENS

    def __post_init__(self) -> None:
        for name in (
            "absolute_token_position",
            "logical_prefix_length",
            "materialization_epoch",
            "rollback_epoch",
        ):
            _nonnegative_integer(name, getattr(self, name))
        if self.absolute_token_position != self.logical_prefix_length:
            raise SemanticSnapshotError(
                "RAM snapshot v1 requires absolute position == logical prefix length"
            )
        _nonempty_string("materialization_policy", self.materialization_policy)
        interval = _nonnegative_integer(
            "materialization_interval_tokens", self.materialization_interval_tokens
        )
        if interval == 0:
            raise SemanticSnapshotError("materialization interval must be positive")
        expected_epoch = self.absolute_token_position // interval
        if self.materialization_epoch != expected_epoch:
            raise SemanticSnapshotError(
                "materialization epoch does not match the semantic token boundary"
            )

    def descriptor(self) -> dict[str, object]:
        return {
            **asdict(self),
            "active_state_slots": [
                [layer, list(slots)] for layer, slots in self.active_state_slots
            ],
            "kv_logical_extents": [list(row) for row in self.kv_logical_extents],
            "indexpool_logical_extents": [
                list(row) for row in self.indexpool_logical_extents
            ],
        }


@dataclass(frozen=True)
class SemanticSnapshotOwnership:
    lifecycle: CacheLifecycle = CacheLifecycle.SNAPSHOT_STATE
    retention: RetentionPolicy = RetentionPolicy.SNAPSHOT_OWNED
    tensor_ownership: str = "owned"
    physical_storage_alias_with_live: bool = False
    prefix_lru_member: bool = False
    persistence: str = "ram-only"

    def __post_init__(self) -> None:
        if self.lifecycle is not CacheLifecycle.SNAPSHOT_STATE:
            raise SemanticSnapshotError("semantic snapshots require SNAPSHOT_STATE")
        if self.retention is not RetentionPolicy.SNAPSHOT_OWNED:
            raise SemanticSnapshotError("semantic snapshots require SNAPSHOT_OWNED")
        if self.tensor_ownership != "owned":
            raise SemanticSnapshotError("semantic snapshot tensors must be owned")
        if self.physical_storage_alias_with_live or self.prefix_lru_member:
            raise SemanticSnapshotError("semantic snapshot storage must be isolated")
        if self.persistence != "ram-only":
            raise SemanticSnapshotError("semantic snapshot v1 is RAM-only")

    def descriptor(self) -> dict[str, object]:
        return {
            "lifecycle": self.lifecycle.value,
            "retention": self.retention.value,
            "tensor_ownership": self.tensor_ownership,
            "physical_storage_alias_with_live": self.physical_storage_alias_with_live,
            "prefix_lru_member": self.prefix_lru_member,
            "persistence": self.persistence,
        }


def _arrays(value) -> Iterable:
    try:
        import mlx.core as mx
    except ImportError:
        return
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _arrays(item)
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _arrays(value[key])


def _entry_state(entry):
    return getattr(entry, "state", ())


_AUX_ARRAY_ATTRIBUTES = ("_left_padding", "_lengths")
_AUX_SCALAR_ATTRIBUTES = ("_left_padding_advance", "_lengths_advance")


def _entry_aux_arrays(entry) -> Iterable:
    for name in _AUX_ARRAY_ATTRIBUTES:
        value = getattr(entry, name, None)
        if value is not None:
            yield name, value


def _entry_metadata(entry) -> dict[str, object]:
    return {
        "meta_state": repr(getattr(entry, "meta_state", None)),
        "aux_scalars": {
            name: int(getattr(entry, name))
            for name in _AUX_SCALAR_ATTRIBUTES
            if hasattr(entry, name)
        },
        "aux_arrays": [
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _sha256(_raw_bytes(value)),
            }
            for name, value in _entry_aux_arrays(entry)
        ],
    }


def _cache_arrays(cache: Sequence[object]) -> Iterable:
    for entry in cache:
        yield from _arrays(_entry_state(entry))


def _raw_bytes(value) -> bytes:
    import mlx.core as mx
    import numpy as np

    materialized = value.view(mx.uint16) if value.dtype == mx.bfloat16 else value
    mx.eval(materialized)
    return np.ascontiguousarray(np.asarray(materialized)).tobytes()


def semantic_cache_schema(cache: Sequence[object]) -> tuple[dict[str, object], ...]:
    rows = []
    for layer, entry in enumerate(cache):
        leaves = []
        for leaf, value in enumerate(_arrays(_entry_state(entry))):
            leaves.append(
                {
                    "leaf": leaf,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            )
        rows.append(
            {
                "layer": layer,
                "entry_type": f"{type(entry).__module__}.{type(entry).__qualname__}",
                "leaves": leaves,
                "metadata": _entry_metadata(entry),
            }
        )
    return tuple(rows)


def semantic_cache_digest(cache: Sequence[object]) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_json(semantic_cache_schema(cache)))
    for leaf, value in enumerate(_cache_arrays(cache)):
        digest.update(str(leaf).encode())
        digest.update(_raw_bytes(value))
    for layer, entry in enumerate(cache):
        digest.update(str(layer).encode())
        digest.update(_canonical_json(_entry_metadata(entry)))
    return digest.hexdigest()


def semantic_component_digests(cache: Sequence[object]) -> dict[str, str]:
    kda = hashlib.sha256()
    latent = hashlib.sha256()
    indexpool = hashlib.sha256()
    metadata = hashlib.sha256()
    for layer, entry in enumerate(cache):
        children = tuple(getattr(entry, "caches", ()))
        targets = (
            (("dsa-kv", children[0]), ("indexpool", children[1]))
            if len(children) == 2
            else (("kda", entry),)
        )
        for kind, target in targets:
            selected = {"kda": kda, "dsa-kv": latent, "indexpool": indexpool}[kind]
            selected.update(str(layer).encode())
            for value in _arrays(_entry_state(target)):
                selected.update(_raw_bytes(value))
            metadata.update(str(layer).encode())
            metadata.update(_canonical_json(_entry_metadata(target)))
    return {
        "kda_state_sha256": kda.hexdigest(),
        "dsa_kv_sha256": latent.hexdigest(),
        "indexpool_sha256": indexpool.hexdigest(),
        "slot_index_metadata_sha256": metadata.hexdigest(),
    }


def _entry_nbytes(entry: object) -> int:
    nbytes = getattr(entry, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    return sum(int(value.nbytes) for value in _arrays(_entry_state(entry)))


def _logical_size(entry: object) -> int:
    size = getattr(entry, "size", None)
    if callable(size):
        return int(size())
    if hasattr(entry, "offset"):
        return int(entry.offset)
    raise SemanticSnapshotError(
        f"cache component {type(entry).__name__} has no logical extent"
    )


def inspect_semantic_boundary(
    cache: Sequence[object],
    *,
    absolute_token_position: int,
    materialization_epoch: int,
    rollback_epoch: int = 0,
) -> SemanticSnapshotBoundary:
    position = _nonnegative_integer(
        "absolute_token_position", absolute_token_position
    )
    active_slots = []
    kv_extents = []
    pool_extents = []
    for layer, entry in enumerate(cache):
        children = tuple(getattr(entry, "caches", ()))
        if len(children) == 2:
            kv_extents.append((layer, _logical_size(children[0])))
            pool_extents.append((layer, _logical_size(children[1])))
            continue
        state = _entry_state(entry)
        if not isinstance(state, (tuple, list)):
            raise SemanticSnapshotError("KDA cache state must expose explicit slots")
        active_slots.append(
            (layer, tuple(index for index, value in enumerate(state) if value is not None))
        )
    for kind, rows in (("DSA/KV", kv_extents), ("IndexPool", pool_extents)):
        if any(extent != position for _, extent in rows):
            raise SemanticSnapshotError(
                f"{kind} logical extent does not match semantic token boundary"
            )
    if not kv_extents or len(kv_extents) != len(pool_extents):
        raise SemanticSnapshotError("semantic snapshot requires paired DSA/KV and IndexPool")
    return SemanticSnapshotBoundary(
        absolute_token_position=position,
        logical_prefix_length=position,
        materialization_epoch=materialization_epoch,
        rollback_epoch=rollback_epoch,
        active_state_slots=tuple(active_slots),
        kv_logical_extents=tuple(kv_extents),
        indexpool_logical_extents=tuple(pool_extents),
    )


def _materialize_cache(cache: Sequence[object]) -> None:
    import mlx.core as mx

    arrays = list(_cache_arrays(cache))
    if arrays:
        mx.eval(*arrays)
    mx.synchronize()


def _clone_cache_entries(
    cache: Sequence[object],
    *,
    min_capacity_tokens: int,
    clone_entry: Callable | None = None,
) -> list[object]:
    import mlx.core as mx
    from mlx_vlm.apc_adapters import clone_cache_entry

    clone_entry = clone_cache_entry if clone_entry is None else clone_entry
    targets = []
    cloned = []
    for entry in cache:
        replacement = clone_entry(
            entry,
            min_capacity_tokens=min_capacity_tokens,
            eval_targets=targets,
        )
        if replacement is None:
            raise SemanticSnapshotError(
                f"RAM snapshot cannot clone cache entry {type(entry).__name__}"
            )
        cloned.append(replacement)
        # ArraysCache's upstream APC adapter preserves the semantic padding
        # value but normalizes its internal advance counter.  Semantic
        # snapshots preserve both the owned raw tensor and representation
        # epoch so the complete slot metadata remains reproducible.
        for name, source in _entry_aux_arrays(entry):
            copied = mx.contiguous(mx.array(source, dtype=source.dtype))
            setattr(replacement, name, copied)
            targets.append(copied)
        for name in _AUX_SCALAR_ATTRIBUTES:
            if hasattr(entry, name):
                setattr(replacement, name, int(getattr(entry, name)))
    if targets:
        mx.eval(*targets)
    _materialize_cache(cloned)
    source_arrays = list(_cache_arrays(cache))
    cloned_arrays = list(_cache_arrays(cloned))
    if len(source_arrays) != len(cloned_arrays):
        raise SemanticSnapshotError("RAM snapshot clone changed cache leaf count")
    if any(left is right for left, right in zip(source_arrays, cloned_arrays, strict=True)):
        raise SemanticSnapshotError("RAM snapshot retained a live tensor alias")
    return cloned


@dataclass(frozen=True)
class HybridSemanticPrefixSnapshot:
    snapshot_id: str
    identity: SemanticSnapshotIdentity
    boundary: SemanticSnapshotBoundary
    schema_sha256: str
    state_sha256: str
    component_digests: Mapping[str, str]
    cache_leaf_count: int
    resident_bytes: int
    ownership: SemanticSnapshotOwnership = field(
        default_factory=SemanticSnapshotOwnership
    )
    _cache: tuple[object, ...] = field(repr=False, compare=False, default=())

    def __post_init__(self) -> None:
        _nonempty_string("snapshot_id", self.snapshot_id)
        _nonempty_string("schema_sha256", self.schema_sha256)
        _nonempty_string("state_sha256", self.state_sha256)
        _nonnegative_integer("cache_leaf_count", self.cache_leaf_count)
        _nonnegative_integer("resident_bytes", self.resident_bytes)
        if not self._cache:
            raise SemanticSnapshotError("published snapshot must contain complete cache state")
        object.__setattr__(
            self, "component_digests", MappingProxyType(dict(self.component_digests))
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": SEMANTIC_PREFIX_SNAPSHOT_SCHEMA,
            "snapshot_id": self.snapshot_id,
            "identity": self.identity.descriptor(),
            "identity_namespace_sha256": self.identity.namespace_sha256,
            "boundary": self.boundary.descriptor(),
            "schema_sha256": self.schema_sha256,
            "state_sha256": self.state_sha256,
            "component_digests": dict(self.component_digests),
            "cache_leaf_count": self.cache_leaf_count,
            "resident_bytes": self.resident_bytes,
            "ownership": self.ownership.descriptor(),
        }

    @property
    def released(self) -> bool:
        return not self._cache


def prepare_semantic_snapshot(
    *,
    snapshot_id: str,
    live_cache: Sequence[object],
    identity: SemanticSnapshotIdentity,
    absolute_token_position: int,
    materialization_epoch: int,
    rollback_epoch: int = 0,
    clone_entry: Callable | None = None,
) -> HybridSemanticPrefixSnapshot:
    """Prepare a complete owned snapshot without publishing partial state."""

    _nonempty_string("snapshot_id", snapshot_id)
    if not isinstance(identity, SemanticSnapshotIdentity):
        raise SemanticSnapshotError("snapshot identity must be explicit")
    _materialize_cache(live_cache)
    source_digest = semantic_cache_digest(live_cache)
    source_boundary = inspect_semantic_boundary(
        live_cache,
        absolute_token_position=absolute_token_position,
        materialization_epoch=materialization_epoch,
        rollback_epoch=rollback_epoch,
    )
    cloned = _clone_cache_entries(
        live_cache,
        min_capacity_tokens=absolute_token_position,
        clone_entry=clone_entry,
    )
    cloned_boundary = inspect_semantic_boundary(
        cloned,
        absolute_token_position=absolute_token_position,
        materialization_epoch=materialization_epoch,
        rollback_epoch=rollback_epoch,
    )
    cloned_digest = semantic_cache_digest(cloned)
    source_digest_after = semantic_cache_digest(live_cache)
    source_boundary_after = inspect_semantic_boundary(
        live_cache,
        absolute_token_position=absolute_token_position,
        materialization_epoch=materialization_epoch,
        rollback_epoch=rollback_epoch,
    )
    if source_digest_after != source_digest or source_boundary_after != source_boundary:
        raise SemanticSnapshotError("live cache changed during snapshot capture")
    if cloned_boundary != source_boundary or cloned_digest != source_digest:
        raise SemanticSnapshotError("prepared snapshot is not byte-exact")
    schema = semantic_cache_schema(cloned)
    return HybridSemanticPrefixSnapshot(
        snapshot_id=snapshot_id,
        identity=identity,
        boundary=cloned_boundary,
        schema_sha256=_sha256(_canonical_json(schema)),
        state_sha256=cloned_digest,
        component_digests=semantic_component_digests(cloned),
        cache_leaf_count=sum(1 for _ in _cache_arrays(cloned)),
        resident_bytes=sum(_entry_nbytes(entry) for entry in cloned),
        _cache=tuple(cloned),
    )


def prepare_snapshot_restore(
    snapshot: HybridSemanticPrefixSnapshot,
    *,
    expected_identity: SemanticSnapshotIdentity,
    clone_entry: Callable | None = None,
) -> list[object]:
    """Validate and clone every component before returning a replacement cache."""

    if not isinstance(snapshot, HybridSemanticPrefixSnapshot):
        raise SemanticSnapshotError("restore source is not a semantic snapshot")
    if expected_identity != snapshot.identity:
        raise SemanticSnapshotError("semantic snapshot identity mismatch")
    cloned = _clone_cache_entries(
        snapshot._cache,
        min_capacity_tokens=snapshot.boundary.absolute_token_position,
        clone_entry=clone_entry,
    )
    boundary = inspect_semantic_boundary(
        cloned,
        absolute_token_position=snapshot.boundary.absolute_token_position,
        materialization_epoch=snapshot.boundary.materialization_epoch,
        rollback_epoch=snapshot.boundary.rollback_epoch,
    )
    if boundary != snapshot.boundary:
        raise SemanticSnapshotError("restored cache boundary does not match snapshot")
    if semantic_cache_digest(cloned) != snapshot.state_sha256:
        raise SemanticSnapshotError("restored cache digest does not match snapshot")
    schema_hash = _sha256(_canonical_json(semantic_cache_schema(cloned)))
    if schema_hash != snapshot.schema_sha256:
        raise SemanticSnapshotError("restored cache schema does not match snapshot")
    return cloned


class SemanticCacheHandle:
    """A live cache reference with a single atomic replacement point."""

    def __init__(self, cache: Sequence[object]):
        if not isinstance(cache, (tuple, list)) or not cache:
            raise SemanticSnapshotError("semantic cache handle requires a complete cache")
        self._cache = list(cache)

    @property
    def cache(self) -> list[object]:
        return self._cache

    def restore(
        self,
        snapshot: HybridSemanticPrefixSnapshot,
        *,
        expected_identity: SemanticSnapshotIdentity,
        clone_entry: Callable | None = None,
    ) -> None:
        replacement = prepare_snapshot_restore(
            snapshot,
            expected_identity=expected_identity,
            clone_entry=clone_entry,
        )
        # The only mutation in restore: every component is already validated.
        self._cache = replacement


@dataclass
class _SnapshotAccounting:
    resident_bytes: int = 0
    peak_bytes: int = 0
    cumulative_allocated_bytes: int = 0
    cumulative_released_bytes: int = 0
    allocation_count: int = 0
    release_count: int = 0

    def allocate(self, nbytes: int) -> None:
        self.resident_bytes += nbytes
        self.peak_bytes = max(self.peak_bytes, self.resident_bytes)
        self.cumulative_allocated_bytes += nbytes
        self.allocation_count += 1

    def release(self, nbytes: int) -> None:
        self.resident_bytes -= nbytes
        self.cumulative_released_bytes += nbytes
        self.release_count += 1
        if self.resident_bytes < 0:
            raise AssertionError("semantic snapshot accounting became negative")

    def descriptor(self) -> dict[str, int]:
        return {**asdict(self), "anonymous_allocation_count": 0}


class SemanticSnapshotStore:
    """Explicit RAM ownership domain; not a prefix LRU and not disk APC."""

    def __init__(self):
        self._snapshots: dict[str, HybridSemanticPrefixSnapshot] = {}
        self._accounting = _SnapshotAccounting()

    def capture(
        self,
        handle: SemanticCacheHandle,
        *,
        snapshot_id: str,
        identity: SemanticSnapshotIdentity,
        absolute_token_position: int,
        materialization_epoch: int,
        rollback_epoch: int = 0,
        clone_entry: Callable | None = None,
    ) -> HybridSemanticPrefixSnapshot:
        if snapshot_id in self._snapshots:
            raise SemanticSnapshotError("duplicate semantic snapshot id")
        prepared = prepare_semantic_snapshot(
            snapshot_id=snapshot_id,
            live_cache=handle.cache,
            identity=identity,
            absolute_token_position=absolute_token_position,
            materialization_epoch=materialization_epoch,
            rollback_epoch=rollback_epoch,
            clone_entry=clone_entry,
        )
        # Publish only after every component and invariant has passed.
        self._snapshots[snapshot_id] = prepared
        self._accounting.allocate(prepared.resident_bytes)
        return prepared

    def get(self, snapshot_id: str) -> HybridSemanticPrefixSnapshot:
        try:
            return self._snapshots[snapshot_id]
        except KeyError as error:
            raise SemanticSnapshotError("unknown semantic snapshot") from error

    def restore(
        self,
        snapshot_id: str,
        handle: SemanticCacheHandle,
        *,
        expected_identity: SemanticSnapshotIdentity,
        clone_entry: Callable | None = None,
    ) -> None:
        handle.restore(
            self.get(snapshot_id),
            expected_identity=expected_identity,
            clone_entry=clone_entry,
        )

    def delete(self, snapshot_id: str) -> None:
        snapshot = self.get(snapshot_id)
        del self._snapshots[snapshot_id]
        self._accounting.release(snapshot.resident_bytes)
        # Explicit deletion invalidates the private payload even when callers
        # retain the immutable descriptor object.  External Python references
        # therefore cannot keep unaccounted snapshot tensors resident.
        object.__setattr__(snapshot, "_cache", ())

    def accounting(self) -> dict[str, object]:
        return {
            "lifecycle": CacheLifecycle.SNAPSHOT_STATE.value,
            "retention": RetentionPolicy.SNAPSHOT_OWNED.value,
            "prefix_lru_member": False,
            "disk_persistence": False,
            **self._accounting.descriptor(),
        }

    @property
    def snapshot_ids(self) -> tuple[str, ...]:
        return tuple(self._snapshots)
