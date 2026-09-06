from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "soak_cache_restore_allocation_pressure.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-cache-restore-allocation-pressure-20260906.json"
)


def test_soak_is_user_launched_atomic_and_fixes_the_qualification_geometry():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert "DEFAULT_CONTEXT_TOKENS = 32_768" in source
    assert "DEFAULT_CONTINUATION_TOKENS = 16" in source
    assert "DEFAULT_RESTORE_GENERATIONS = 100" in source
    assert 'RESTORE_PATHS = ("ram-apc", "semantic-snapshot")' in source
    assert "temporary.replace(path)" in source
    assert "cache payload is\nnot serialized" in source
    assert '"complete": False' in source
    assert '"first_divergence": None' in source
    assert "default=_json_default" in source
    assert "dict(semantic_snapshot.component_digests)" in source


def test_atomic_artifact_encoder_accepts_immutable_snapshot_mappings():
    tree = ast.parse(SCRIPT.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_json_default"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"Mapping": Mapping, "Path": Path, "np": np}
    exec(compile(module, str(SCRIPT), "exec"), namespace)
    encoded = json.dumps(
        {"components": MappingProxyType({"kda": "exact"})},
        default=namespace["_json_default"],
    )
    assert json.loads(encoded) == {"components": {"kda": "exact"}}


def test_pressure_is_full_cache_shaped_and_released_before_restore():
    source = SCRIPT.read_text()
    pressure = source.index("pressure = _pressure_once(")
    restore = source.index('if path == "ram-apc":', pressure)
    assert pressure < restore
    assert "_clone_cache(source, min_capacity_tokens=capacity_tokens)" in source
    assert "_materialize(cloned, clear_allocator_cache=False)" in source
    assert "Do not call mx.clear_cache here" in source
    assert '"logical_resident_bytes_after_release": 0' in source
    assert "stale_entry_reference_count" in source
    assert "stale_storage_reference_count" in source
    assert "_authoritative_accounting" in source
    assert '"anonymous_bytes": total - accounted' in source
    assert "persistent_source_stale_reference_count" in source


def test_every_generation_records_full_hybrid_exactness_and_source_immutability():
    source = SCRIPT.read_text()
    for required in (
        "semantic_cache_digest",
        "semantic_component_digests",
        "layerwise_kda_digests",
        "compare_layerwise_digests",
        "full_vocab_logits_hashes",
        "derived_indexpool",
        "invalid_index_count",
        "source_digest_immutable",
        "source_storage_alias_count",
        "cache_identity_exact",
        "first_divergence",
    ):
        assert required in source
    assert "for generation in range(1, args.restore_generations + 1)" in source
    assert "_atomic_write(args.output, artifact)" in source
    assert "raise RestorePressureDivergence" in source


def test_restore_paths_are_actual_ram_apc_and_transactional_semantic_snapshot():
    source = SCRIPT.read_text()
    for required in (
        "APCManager",
        "store_exact_cache",
        "lookup_exact_cache",
        "SemanticSnapshotStore",
        'semantic_store.capture(',
        'store.restore("prefix-32k"',
        "cache_reference_replaced",
    ):
        assert required in source
    assert '"moe_backend": "direct"' in source
    assert '"cache_backend": "direct"' in source
    assert '"disk_apc_used": False' in source
    assert '"partial_topk_implemented": False' in source


def test_completed_m3_artifact_passes_all_restore_pressure_gates_when_present():
    if not ARTIFACT.exists():
        pytest.skip("M3 Ultra cache restore pressure soak has not been run yet")
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "glm53-cache-restore-allocation-pressure-soak-v1"
    if not artifact.get("complete"):
        pytest.skip("cache restore pressure soak is incomplete")
    assert artifact["accepted"] is True
    assert artifact["first_divergence"] is None
    assert all(artifact["acceptance"].values())
    assert set(artifact["paths"]) == {"ram-apc", "semantic-snapshot"}
    for path in artifact["paths"].values():
        assert path["complete"] is True
        assert path["last_completed_generation"] >= 100
        assert path["all_restore_and_replay_exact"] is True
        assert path["source_digest_immutable"] is True
        assert len(path["restore_generations"]) >= 100
        for row in path["restore_generations"]:
            assert row["source_digest_immutable"] is True
            assert row["cache_identity"]["exact"] is True
            assert row["restored_prefix"]["source_storage_alias_count"] == 0
            assert row["comparison"]["full_vocab_logits_exact"] is True
            assert row["comparison"]["final_cache"]["layerwise_kda_exact"] is True
            assert row["comparison"]["derived_indexpool_exact"] is True
            assert row["release"]["stale_entry_reference_count"] == 0
            assert row["release"]["stale_storage_reference_count"] == 0
            assert row["pressure"]["stale_entry_reference_count"] == 0
            assert row["pressure"]["stale_storage_reference_count"] == 0
            assert row["pressure"]["anonymous_bytes"] == 0
