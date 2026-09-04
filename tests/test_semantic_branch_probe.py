from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/probe_semantic_branch_isolation.py"


def test_branch_probe_is_syntactically_valid_and_has_bounded_defaults():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert "DEFAULT_PREFIX_TOKENS = 256" in source
    assert "DEFAULT_TWIN_STEPS = 1024" in source
    assert "DEFAULT_DIVERGENT_STEPS = 512" in source
    assert "DEFAULT_REPLAY_STEPS = 64" in source
    assert "ROLLBACK_TOKENS = 16" in source
    assert "MAX_PEAK_BYTES = 340_000_000_000" in source


def test_branch_probe_covers_twin_divergent_and_local_mutation_isolation():
    source = SCRIPT.read_text()
    for evidence in (
        "baseline_vs_a",
        "baseline_vs_b",
        "a_b_logits_sequence_exact",
        "b_unchanged_while_a_advanced",
        "a_unchanged_while_b_advanced",
        "first_input_tokens_differ",
        "final_states_differ",
        "rollback_replay_exact",
        "reference_baseline_cache_released",
    ):
        assert evidence in source
    assert "full_vocab_logits_exact" in source
    assert "kda_state_exact" in source
    assert "dsa_kv_exact" in source
    assert "indexpool_exact" in source
    assert "slot_index_metadata_exact" in source


def test_branch_probe_records_lineage_alias_generations_and_resource_accounting():
    source = SCRIPT.read_text()
    for evidence in (
        "snapshot_lineage",
        "storage_alias_count",
        "generation_advanced_once",
        "stale_entry_reference_count",
        "branch_accounting_before_delete",
        "snapshot_accounting_before_delete",
        "cumulative_allocated_bytes",
        "cumulative_released_bytes",
        "anonymous_allocation_count",
    ):
        assert evidence in source


def test_branch_probe_injects_atomic_activation_rollback_and_identity_failures():
    source = SCRIPT.read_text()
    for name in (
        "activation_validator_failure",
        "unknown_branch_activation",
        "rollback_17",
        "snapshot_identity_mismatch",
    ):
        assert name in source
    for evidence in (
        "active_branch_unchanged",
        "active_cache_reference_unchanged",
        "all_branch_state_and_accounting_unchanged",
        "parent_snapshot_unchanged",
        "manager_accounting_unchanged",
    ):
        assert evidence in source


def test_branch_probe_is_sequential_ram_only_and_does_not_expand_runtime_scope():
    source = SCRIPT.read_text()
    for key in (
        "parallel_execution",
        "scheduler",
        "continuous_batching",
        "kernel_abi",
        "cache_abi",
        "server",
        "disk_apc",
        "admission",
    ):
        assert f'"{key}": False' in source
    assert "live branch state itself is RAM-only and not resumable" in source
    assert "_atomic_write(args.output" in source
