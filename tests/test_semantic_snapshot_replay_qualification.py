from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualify_hybrid_semantic_prefix_snapshot_replay.py"


def test_qualification_script_is_syntactically_valid_and_has_full_defaults():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert "DEFAULT_PREFIX_TOKENS = 256" in source
    assert "DEFAULT_STEPS = 4096" in source
    assert "DEFAULT_REPLAYS = 8" in source
    assert "DEFAULT_NESTED_STEP = 2048" in source
    assert "CHECKPOINT_INTERVAL = MATERIALIZATION_INTERVAL_TOKENS" in source


def test_qualification_records_every_semantic_component_and_checkpoint():
    source = SCRIPT.read_text()
    for name in (
        "full_vocab_logits_exact",
        "all_checkpoint_state_exact",
        "kda_state_exact",
        "dsa_kv_exact",
        "indexpool_exact",
        "slot_index_metadata_exact",
        "final_state_sha256",
    ):
        assert name in source
    assert "step % CHECKPOINT_INTERVAL == 0" in source
    assert "expected_checkpoints_per_replay" in source


def test_qualification_has_nested_snapshots_generation_and_stale_reference_evidence():
    source = SCRIPT.read_text()
    assert 'snapshot_id="S0"' in source
    assert 'snapshot_id="S1"' in source
    assert "s0_s1_storage_alias_count" in source
    assert "snapshot_generation" in source
    assert "live_generation" in source
    assert "stale_entry_reference_count" in source
    assert "cumulative_replacement_allocated_bytes" in source
    assert "snapshot_resident_constant" in source


def test_qualification_injects_three_atomic_restore_failures():
    source = SCRIPT.read_text()
    assert '"identity_mismatch"' in source
    assert '"corrupt_component"' in source
    assert '"invalid_extent_metadata"' in source
    assert "live_reference_unchanged" in source
    assert "live_generation_unchanged" in source
    assert "live_state_unchanged" in source
    assert "snapshot_unchanged" in source


def test_qualification_artifact_is_atomic_and_failure_resumability_is_not_claimed():
    source = SCRIPT.read_text()
    assert "temporary.replace(path)" in source
    assert '"complete": False' in source
    assert "last_completed_phase" in source
    assert "cache itself is not serialized or resumable" in source


def test_qualification_does_not_expand_runtime_scope():
    source = SCRIPT.read_text()
    for key in (
        "server_api",
        "disk_serialization",
        "snapshot_lru",
        "automatic_snapshot_compression",
        "speculative_branch_selection",
        "cache_abi",
        "backend",
        "admission",
    ):
        assert f'"{key}": False' in source
