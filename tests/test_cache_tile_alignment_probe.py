from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_kpool_cache_tile_alignment.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-kpool-cache-tile-alignment-20260905.json"
)


def test_probe_covers_local_geometry_not_foreign_page_constants():
    source = SCRIPT.read_text()
    assert "DEFAULT_NOPE_CACHE_TILE_ALIGNMENT" in source
    assert "ALIGNMENT_BOUNDARIES = (255, 256, 257, 511, 512, 513)" in source
    assert "LONG_BOUNDARIES = (262_143, 262_144, 262_145)" in source
    assert '"custom_metal_indexer_kernel": False' in source
    assert '"virtual_page_or_tile_split": False' in source
    assert "1152" not in source
    assert "1280" not in source


def test_probe_covers_padding_restore_and_full_model_differentials():
    source = SCRIPT.read_text()
    for text in (
        "padding_indices_all_sentinel",
        "padding_valid_all_false",
        "padding_keys_all_zero",
        "restore_trim_replay_state_exact",
        "_tier2_context(model, SYNTHETIC_CONTEXT)",
        "direct_compact_dsa_indices_byte_identical",
        "compact_resident_restore_post_state_exact",
        "first_16_match",
        "full_128_match",
    ):
        assert text in source


def test_artifact_passes_alignment_safety_when_present():
    if not ARTIFACT.exists():
        pytest.skip("M3 Ultra tile-alignment artifact has not been generated yet")
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "glm53-kpool-cache-tile-alignment-v1"
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert all(artifact["acceptance"].values())
    assert artifact["decision"] == "cache_tile_alignment_safety_closed"
    assert artifact["contract"]["allocation_alignment_tokens"] == 256
    assert artifact["contract"]["allocation_alignment_pool_rows"] == 64
    assert artifact["contract"]["virtual_tile_split_allowed"] is False
    assert artifact["claims"] == {
        "validated": (
            "256k resident/restore to first decode with aligned compact cache"
        ),
        "unsupported_unvalidated": "256k cold prefill to first decode",
    }
