import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_virtual_physical_bk16_sparse_prefill_av.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-virtual-physical-bk16-sparse-prefill-av-20260908.json"
)


def test_probe_preserves_physical_bk16_lanes_and_removes_only_empty_blocks():
    source = PROBE.read_text()
    assert "physical // BK" in source
    assert "physical % BK" in source
    assert "np.unique" in source
    assert '"removed_empty_bk16_blocks"' in source
    assert '"packed_k_bounded_by_selected_width_times_bk"' in source
    assert '"fixed_capacity_trailing_zero_blocks_byte_exact"' in source


def test_probe_requires_every_query_row_to_be_byte_exact():
    source = PROBE.read_text()
    assert '"all_query_rows_byte_exact"' in source
    assert '"different_output_elements"' in source
    assert '"implement_native_virtual_physical_bk16_av"' in source
    assert '"runtime_changes": False' in source


def test_probe_artifact_is_exact_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert all(artifact["checks"].values())
    assert artifact["bk"] == 16
    for context in artifact["contexts"].values():
        assert context["max_packed_k"] <= 2051 * 16
        assert all(
            row["byte_exact"] and row["fixed_capacity_byte_exact"]
            for row in context["rows"]
        )
