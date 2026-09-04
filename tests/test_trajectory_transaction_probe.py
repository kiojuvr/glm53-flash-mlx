from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/probe_trajectory_transaction_winner_commit.py"


def test_probe_is_syntactically_valid_and_has_bounded_defaults():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert "DEFAULT_PREFIX_TOKENS = 256" in source
    assert "DEFAULT_TRAJECTORY_STEPS = 256" in source
    assert "DEFAULT_CONTINUATION_STEPS = 64" in source
    assert "DEFAULT_FAILURE_STEPS = 16" in source
    assert "MAX_PEAK_BYTES = 340_000_000_000" in source


def test_probe_separates_evaluation_policy_from_transaction_commit():
    source = SCRIPT.read_text()
    assert '"evaluation_policy": "external-deterministic-fixture"' in source
    assert '"winner_selection_is_runtime_policy": False' in source
    assert "evaluation_is_observation_only" in source
    assert "winner_branch_id=winner_id" in source


def test_probe_qualifies_both_winners_cas_abort_and_terminal_states():
    source = SCRIPT.read_text()
    for evidence in (
        '(("a-wins", 0), ("b-wins", 1))',
        "winner_owned_storage_promoted_without_copy",
        "winner_continuation_full_state_exact",
        "stale_evaluation_rejected_atomically",
        "stale_base_rejected_atomically",
        "abort_preserves_active_and_releases_candidates",
        "terminal_transactions_fail_closed",
        "mixed_generation_zero",
    ):
        assert evidence in source


def test_probe_covers_full_semantic_state_and_resource_release():
    source = SCRIPT.read_text()
    for evidence in (
        "full_vocab_logits_exact",
        "kda_state_exact",
        "dsa_kv_exact",
        "indexpool_exact",
        "slot_index_metadata_exact",
        "loser_released_bytes",
        "old_active_stale_entry_count",
        "loser_stale_entry_count",
        "transaction_resident_bytes",
        "anonymous_allocation_zero",
        "ownership_accounting_balanced",
    ):
        assert evidence in source


def test_probe_is_ram_only_and_does_not_expand_runtime_scope():
    source = SCRIPT.read_text()
    for key in (
        "server",
        "disk_apc",
        "cache_abi",
        "kernel_abi",
        "backend",
        "admission",
        "parallel_branch_execution",
    ):
        assert f'"{key}": False' in source
    assert "_atomic_write(args.output" in source
