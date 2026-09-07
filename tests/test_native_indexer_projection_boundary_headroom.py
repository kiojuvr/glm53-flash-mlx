import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_native_indexer_projection_boundary_headroom.py"


def test_projection_headroom_probe_is_a_counterfactual_not_a_backend():
    source = PROBE.read_text()
    ast.parse(source)
    assert "exact precomputed BF16 key/gate/query/mixture-weight projections" in source
    assert "_ProjectionRecorder" in source
    assert "_ProjectionReplay" in source
    assert "_snapshot_projection" in source
    assert "capture/copy/ownership excluded" in source
    assert '"profiling_only": True' in source
    for component in ("runtime", "server", "apc", "cache_abi", "kernel_abi"):
        assert f'"{component}": False' in source


def test_projection_headroom_probe_keeps_native_update_island_unchanged():
    source = PROBE.read_text()
    assert "update_island._native_update(" in source
    assert "registry.get(pool, indexer)" in source
    assert "A_native_update_with_mlx_projections" in source
    assert "B_precomputed_projections_native_update" in source
    assert "CONTEXTS = (2_048, 262_144)" in source
    assert "MIN_256K_HEADROOM_MS = 0.75" in source


def test_projection_replay_validates_before_native_state_update():
    source = PROBE.read_text()
    replay = source[source.index("class _ProjectionReplay") :]
    exhausted = replay.index("precomputed projection trajectory exhausted")
    stale = replay.index("precomputed projection shape is stale")
    execute = replay.index("_execute_projection(")
    assert exhausted < stale < execute


def test_projection_headroom_requires_exact_logits_state_and_official_oracle():
    source = PROBE.read_text()
    assert '"all_logits_and_post_state_byte_exact"' in source
    assert "boundary._cache_exact(caches[arms[0]], caches[arms[1]])" in source
    assert "boundary._cache_exact(caches[arms[0]], oracle_cache)" in source
    assert '"official_oracle_exact"' in source
    assert '"implement_native_indexer_input_projections"' in source
    assert '"stop_native_indexer_projection_boundary"' in source
