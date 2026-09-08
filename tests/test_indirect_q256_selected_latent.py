import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_indirect_q256_selected_latent.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-indirect-q256-selected-latent-20260908.json"
)


def test_native_plan_uses_device_count_and_indirect_dispatch():
    header = (
        ROOT
        / "native_execution"
        / "native_indirect_selected_latent_plan.h"
    ).read_text()
    source = (
        ROOT
        / "native_execution"
        / "native_indirect_selected_latent_plan.cpp"
    ).read_text()
    metal = (ROOT / "native_execution" / "native_indexer_plan.metal").read_text()
    assert "NativeSelectedUnionPlan union_plan_" in header
    assert "indirect_arguments_" in header
    assert "union_latent_" in header
    assert "dispatchThreadgroups(" in source
    assert "metal_buffer(indirect_arguments_)" in source
    assert "build_selected_union_gather_arguments" in metal
    assert "gather_selected_union_latent_bfloat16" in metal
    assert "host_synchronization_count() const { return 0; }" in header


def test_probe_requires_exact_indirect_geometry_and_bounded_320k_arena():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert '"device_indirect_arguments_exact"' in source
    assert '"union_and_gather_byte_exact"' in source
    assert '"320k_wall_at_most_5ms"' in source
    assert '"320k_scratch_at_most_384mib"' in source


def test_artifact_is_accepted_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert all(artifact["checks"].values())
