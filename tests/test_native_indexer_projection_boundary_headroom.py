import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_native_indexer_projection_boundary_headroom.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-indexer-projection-boundary-headroom-20260907.json"
)


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


def test_projection_headroom_qualification_records_exact_negative_evidence():
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is False
    assert artifact["decision"] == "stop_native_indexer_projection_boundary"
    assert artifact["runtime_changes"] == {
        "runtime": False,
        "server": False,
        "apc": False,
        "cache_abi": False,
        "kernel_abi": False,
    }
    assert artifact["official_oracle"]["all_full_vocab_logits_hashes_match"]
    assert artifact["process_peak_memory_bytes"] <= 340_000_000_000

    for context in ("2048", "262144"):
        row = artifact["contexts"][context]
        baseline = row["timing"]["A_native_update_with_mlx_projections"]
        replay = row["timing"]["B_precomputed_projections_native_update"]
        assert row["all_logits_and_post_state_byte_exact"]
        assert row["projection_boundary_headroom_ms"] < 0.0
        assert replay["median_host_submit_ms"] < baseline["median_host_submit_ms"]
        assert replay["median_wall_ms"] > baseline["median_wall_ms"] + 4.0

    assert artifact["acceptance"] == {
        "256k_projection_boundary_headroom_at_least_0_75ms": False,
        "2k_counterfactual_regression_at_most_1_percent": False,
        "all_counterfactual_logits_and_state_byte_exact": True,
        "official_oracle_exact": True,
        "process_peak_at_most_340GB": True,
        "production_abi_unchanged": True,
    }
