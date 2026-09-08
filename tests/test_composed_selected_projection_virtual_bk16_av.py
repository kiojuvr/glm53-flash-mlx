import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_composed_selected_projection_virtual_bk16_av.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-composed-selected-projection-virtual-bk16-av-20260908.json"
)


def test_probe_composes_selected_projection_and_exact_virtual_av():
    source = PROBE.read_text()
    assert "equivalence._gather_per_query" in source
    assert "mx.softmax(scores, axis=-1, precise=True)" in source
    assert "NativeSparsePrefillAVPlan" in source
    assert '"candidate_has_explicit_mlx_native_boundary": True' in source


def test_probe_uses_region_wall_as_hard_gate():
    source = PROBE.read_text()
    assert '"all_outputs_byte_exact"' in source
    assert '"32k_composed_region_speedup_at_least_1_20x"' in source
    assert '"runtime_changes": False' in source


def test_artifact_is_accepted_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert all(artifact["checks"].values())
    assert artifact["contexts"]["32768"]["byte_exact"] is True
