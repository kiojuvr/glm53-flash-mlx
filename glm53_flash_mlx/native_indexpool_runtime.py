"""Opt-in production bridge for the qualified native IndexPool update island."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
import threading
import weakref
from pathlib import Path

import mlx.core as mx

from .native_execution import NATIVE_INDEXPOOL_UPDATE_ISLAND_ABI

NATIVE_INDEXPOOL_RUNTIME_ABI = (
    NATIVE_INDEXPOOL_UPDATE_ISLAND_ABI + "-mlx0322-explicit-opt-in-v1"
)
QUALIFIED_MLX_VERSION = "0.32.2"
ENVIRONMENT_FLAG = "GLM53_EXPERIMENTAL_NATIVE_INDEXPOOL_UPDATE"

_NOT_USED = object()
_PLAN_LOCK = threading.Lock()
_PLANS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

_QUALIFIED_HEAD_DIM = 128
_QUALIFIED_INDEX_HEADS = 32
_QUALIFIED_KPOOL = 4
_QUALIFIED_TOPK = 2_048
_QUALIFIED_RAW_WINDOW = 19
_MIN_PHYSICAL_POOL_ROWS = 512
_MAX_PHYSICAL_POOL_ROWS = 65_600
_PHYSICAL_POOL_ALIGNMENT = 64


def enabled() -> bool:
    return os.environ.get(ENVIRONMENT_FLAG) == "1"


def _extension_package():
    """Load the separately built M3-Ultra extension or fail closed."""
    if importlib.metadata.version("mlx") != QUALIFIED_MLX_VERSION:
        raise RuntimeError(
            "native IndexPool update is qualified only for "
            f"mlx=={QUALIFIED_MLX_VERSION}"
        )
    try:
        return importlib.import_module("glm53_native_execution")
    except ImportError as first_error:
        source_tree = Path(__file__).resolve().parents[1] / "native_execution"
        if source_tree.is_dir() and str(source_tree) not in sys.path:
            sys.path.insert(0, str(source_tree))
        try:
            return importlib.import_module("glm53_native_execution")
        except ImportError as error:
            raise RuntimeError(
                "native IndexPool update extension is unavailable; run "
                "`uv run python scripts/build_native_execution_engine.py`"
            ) from error


def require_available() -> dict[str, str]:
    package = _extension_package()
    if not hasattr(package, "NativeIndexPoolUpdateSelectionPlan"):
        raise RuntimeError("native extension lacks the qualified IndexPool plan")
    return {
        "abi": NATIVE_INDEXPOOL_RUNTIME_ABI,
        "mlx_version": QUALIFIED_MLX_VERSION,
        "extension": str(Path(package.__file__).resolve()),
    }


def _plan_for(cache, indexer):
    capacity = int(cache.pool_keys.shape[1])
    with _PLAN_LOCK:
        existing = _PLANS.get(cache)
        if existing is not None and existing[0] == capacity:
            return existing[1]
        plan_type = _extension_package().NativeIndexPoolUpdateSelectionPlan
        plan = plan_type(capacity, float(indexer.softmax_scale))
        _PLANS[cache] = (capacity, plan)
        return plan


def _has_qualified_geometry(cache, indexer) -> bool:
    """Keep the native ABI confined to its measured GLM-5.3 shape domain."""
    capacity = int(cache.pool_keys.shape[1])
    return (
        int(cache.index_kpool) == _QUALIFIED_KPOOL
        and int(cache.index_topk) == _QUALIFIED_TOPK
        and int(cache.head_dim) == _QUALIFIED_HEAD_DIM
        and int(cache.raw_state_window) == _QUALIFIED_RAW_WINDOW
        and int(indexer.n_heads) == _QUALIFIED_INDEX_HEADS
        and int(indexer.head_dim) == _QUALIFIED_HEAD_DIM
        and _MIN_PHYSICAL_POOL_ROWS <= capacity <= _MAX_PHYSICAL_POOL_ROWS
        and capacity % _PHYSICAL_POOL_ALIGNMENT == 0
    )


def _has_writable_pool_row(cache) -> bool:
    """The native fixed arena cannot grow the persistent pool in execute()."""
    capacity = int(cache.pool_keys.shape[1])
    return int(cache.total_tokens) // int(cache.index_kpool) < capacity


def try_native_update(cache, indexer, x, qr, *, mask, short_bypass):
    """Execute the qualified L=1/raw19 path or return the private sentinel."""
    if not enabled():
        return _NOT_USED
    if short_bypass or int(x.shape[1]) != 1 or mask is not None:
        return _NOT_USED
    if cache.raw_token_count != cache.raw_state_window:
        return _NOT_USED
    if cache.pool_keys is None or cache.pool_indices is None or cache.pool_valid is None:
        raise RuntimeError("native IndexPool update requires allocated pool storage")
    if not _has_qualified_geometry(cache, indexer):
        return _NOT_USED
    # A manually constructed or minimally restored cache may reach its current
    # physical edge.  Let the eager path perform its normal aligned growth;
    # the following token will create a new native plan for the larger arena.
    if not _has_writable_pool_row(cache):
        return _NOT_USED

    key = indexer.k_norm(indexer.wk(x)).reshape(1, 1, cache.head_dim)
    gate = x @ indexer.index_kpool_compress_gate.swapaxes(-1, -2)
    valid = mx.ones((1, 1), dtype=mx.bool_)
    query = indexer.wq_b(qr).reshape(
        1, 1, indexer.n_heads, indexer.head_dim
    )
    weights = indexer.weights_proj(x) * (indexer.n_heads**-0.5)
    dependencies = (
        key,
        gate,
        valid,
        query,
        weights,
        cache.pool_keys,
        cache.pool_indices,
        cache.pool_valid,
        cache.raw_keys,
        cache.raw_gates,
        cache.raw_valid,
        cache.raw_positions,
        cache.compress_ape,
    )
    mx.async_eval(*dependencies)
    previous = cache.total_tokens
    plan = _plan_for(cache, indexer)
    selected, _, raw_keys, raw_gates, raw_valid, raw_positions = plan.execute(
        key,
        gate,
        valid,
        query,
        weights,
        cache.pool_keys,
        cache.pool_indices,
        cache.pool_valid,
        cache.raw_keys,
        cache.raw_gates,
        cache.raw_valid,
        cache.raw_positions,
        cache.compress_ape,
        previous,
    )
    cache.raw_keys = raw_keys
    cache.raw_gates = raw_gates
    cache.raw_valid = raw_valid
    cache.raw_positions = raw_positions
    cache.total_tokens = previous + 1
    cache.logical_pool_count = (
        cache.total_tokens + cache.index_kpool - 1
    ) // cache.index_kpool
    return selected[:, None]


def is_not_used(value) -> bool:
    return value is _NOT_USED


def registry_snapshot() -> dict[str, int | str]:
    with _PLAN_LOCK:
        plans = [entry[1] for entry in _PLANS.values()]
    return {
        "abi": NATIVE_INDEXPOOL_RUNTIME_ABI,
        "live_plan_count": len(plans),
        "execution_count": sum(int(plan.execution_count) for plan in plans),
        "scratch_bytes": sum(int(plan.scratch_bytes) for plan in plans),
    }
