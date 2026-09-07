from pathlib import Path
from types import SimpleNamespace

import pytest

from glm53_flash_mlx.native_execution import NATIVE_INDEXPOOL_UPDATE_ISLAND_ABI
from glm53_flash_mlx.native_indexpool_runtime import (
    ENVIRONMENT_FLAG,
    NATIVE_INDEXPOOL_RUNTIME_ABI,
)


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "glm53_flash_mlx" / "native_indexpool_runtime.py"
CACHE = ROOT / "glm53_flash_mlx" / "nope_cache.py"


def test_runtime_abi_is_derived_from_the_accepted_native_island():
    assert NATIVE_INDEXPOOL_RUNTIME_ABI.startswith(NATIVE_INDEXPOOL_UPDATE_ISLAND_ABI)
    assert "mlx0322" in NATIVE_INDEXPOOL_RUNTIME_ABI
    assert "explicit-opt-in" in NATIVE_INDEXPOOL_RUNTIME_ABI


def test_runtime_is_disabled_without_the_explicit_environment_flag(monkeypatch):
    from glm53_flash_mlx import native_indexpool_runtime as runtime

    monkeypatch.delenv(ENVIRONMENT_FLAG, raising=False)
    assert runtime.enabled() is False
    monkeypatch.setenv(ENVIRONMENT_FLAG, "0")
    assert runtime.enabled() is False
    monkeypatch.setenv(ENVIRONMENT_FLAG, "1")
    assert runtime.enabled() is True


def test_cache_calls_native_before_constructing_the_eager_update_graph():
    source = CACHE.read_text()
    update = source[source.index("    def update(self, indexer") :]
    native = update.index("try_native_update(")
    eager = update.index("keys = indexer.k_norm")
    assert native < eager
    assert "if not is_not_used(native):\n            return native" in update


def test_native_path_has_fixed_qualified_geometry_and_no_silent_error_fallback():
    source = RUNTIME.read_text()
    assert "cache.raw_token_count != cache.raw_state_window" in source
    assert "int(x.shape[1]) != 1" in source
    assert "mask is not None" in source
    assert "mx.async_eval(*dependencies)" in source
    assert "plan.execute(" in source
    assert "_has_qualified_geometry(cache, indexer)" in source
    assert "capacity % _PHYSICAL_POOL_ALIGNMENT == 0" in source
    assert "except Exception" not in source
    assert "cache.total_tokens = previous + 1" in source


def test_qualified_geometry_is_explicit_and_rejects_unmeasured_shapes():
    from glm53_flash_mlx.native_indexpool_runtime import (
        _has_qualified_geometry,
        _has_writable_pool_row,
    )

    cache = SimpleNamespace(
        index_kpool=4,
        index_topk=2_048,
        head_dim=128,
        raw_state_window=19,
        total_tokens=2_048,
        pool_keys=SimpleNamespace(shape=(1, 576, 128)),
    )
    indexer = SimpleNamespace(n_heads=32, head_dim=128)
    assert _has_qualified_geometry(cache, indexer)
    cache.pool_keys = SimpleNamespace(shape=(1, 577, 128))
    assert not _has_qualified_geometry(cache, indexer)
    cache.pool_keys = SimpleNamespace(shape=(1, 576, 128))
    indexer.n_heads = 2
    assert not _has_qualified_geometry(cache, indexer)
    cache.total_tokens = 576 * 4
    assert not _has_writable_pool_row(cache)


def test_server_flag_requires_packed_decode_and_compact_cache(monkeypatch):
    from glm53_flash_mlx.server import build_parser, configure_m3_ultra

    parser = build_parser()
    monkeypatch.setenv(ENVIRONMENT_FLAG, "0")
    assert not parser.parse_args([]).experimental_native_indexpool_update
    assert parser.parse_args(
        ["--experimental-native-indexpool-update"]
    ).experimental_native_indexpool_update
    common = dict(
        model=Path("/tmp/model"),
        prefill_step_size=2048,
        max_tokens=4096,
        api_key=None,
        apc=False,
        apc_blocks=1,
        apc_disk_path=None,
        warm_residency=False,
        experimental_packed_grouped_moe=False,
        max_context_tokens=36_864,
        experimental_native_indexpool_update=True,
    )
    with pytest.raises(ValueError, match="requires both"):
        configure_m3_ultra(
            **common,
            experimental_packed_decode_moe=False,
            experimental_compact_nope_dsa_cache=True,
        )
    with pytest.raises(ValueError, match="requires both"):
        configure_m3_ultra(
            **common,
            experimental_packed_decode_moe=True,
            experimental_compact_nope_dsa_cache=False,
        )
    configure_m3_ultra(
        **common,
        experimental_packed_decode_moe=True,
        experimental_compact_nope_dsa_cache=True,
    )
    assert __import__("os").environ[ENVIRONMENT_FLAG] == "1"


def test_disk_apc_identity_separates_native_execution_policy(monkeypatch):
    from glm53_flash_mlx.server import _disk_cache_identity

    monkeypatch.setenv("GLM53_MOE_BACKEND", "packed-decode")
    monkeypatch.setenv("GLM53_CACHE_BACKEND", "compact-nope-dsa")
    monkeypatch.setenv(ENVIRONMENT_FLAG, "0")
    mlx_identity = _disk_cache_identity("checkpoint")
    monkeypatch.setenv(ENVIRONMENT_FLAG, "1")
    native_identity = _disk_cache_identity("checkpoint")
    assert native_identity != mlx_identity


def test_native_extension_is_available_after_the_required_build_step():
    from glm53_flash_mlx.native_indexpool_runtime import require_available

    status = require_available()
    assert status["abi"] == NATIVE_INDEXPOOL_RUNTIME_ABI
    assert status["mlx_version"] == "0.32.2"
    assert status["extension"].endswith((".py", ".pyc"))
