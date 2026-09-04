from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-trajectory-transaction-winner-commit-20260905.json"
)
EXACT_KEYS = (
    "full_vocab_logits_exact",
    "all_checkpoint_state_exact",
    "final_state_exact",
    "kda_state_exact",
    "dsa_kv_exact",
    "indexpool_exact",
    "slot_index_metadata_exact",
)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def test_artifact_is_complete_and_evaluation_policy_is_external():
    artifact = _artifact()
    assert artifact["complete"] is True
    assert len(artifact["acceptance"]) == 25
    assert all(artifact["acceptance"].values())
    configuration = artifact["configuration"]
    assert configuration == {
        "candidate_count": 2,
        "capacity_tokens": 592,
        "continuation_steps": 64,
        "evaluation_policy": "external-deterministic-fixture",
        "failure_steps": 16,
        "prefix_tokens": 256,
        "trajectory_steps": 256,
        "winner_selection_is_runtime_policy": False,
    }


def test_a_and_b_commit_promote_exact_owned_winner_state():
    artifact = _artifact()
    expected = {
        "a_wins": (
            0,
            "36801a94e20f6613c63ac67a11acbcf70203799f89235f9aec21cb362b17fb73",
        ),
        "b_wins": (
            1,
            "6a5ced01ee0bc947fa93b84c7089b72cce2f4117c04b4fb2293209e33165c5ca",
        ),
    }
    for name, (winner_offset, digest) in expected.items():
        row = artifact[name]
        commit = row["commit"]
        assert row["winner_offset"] == winner_offset
        assert row["evaluation_observation_only"] is True
        assert row["transaction_state"] == "committed"
        assert commit["winner_semantic_digest"] == digest
        assert commit["active_semantic_digest"] == digest
        assert row["post_commit"]["state_sha256"] == digest
        assert commit["winner_storage_promoted_without_copy"] is True
        assert row["winner_cache_id_before"] == row["winner_cache_id_after"]
        assert commit["winner_promoted_bytes"] == 157_001_113
        assert commit["loser_released_bytes"] == 157_001_113
        assert commit["mixed_generation_count"] == 0
        assert row["old_active_stale_entry_count"] == 0
        assert row["loser_stale_entry_count"] == 0
        assert all(row["sibling_isolation"].values())
        assert all(row["terminal_rejections"].values())


def test_committed_active_continuation_matches_winner_oracle_exactly():
    artifact = _artifact()
    for name in ("a_wins", "b_wins"):
        row = artifact[name]
        assert row["committed_continuation"]["steps"] == 64
        assert row["oracle_continuation"]["steps"] == 64
        assert all(row["continuation_comparison"][key] for key in EXACT_KEYS)
        assert row["continuation_comparison"]["logits_mismatch_steps"] == []
        assert row["continuation_final_state_exact"] is True
        assert row["s0_immutable"] is True


def test_stale_evaluation_stale_base_and_abort_are_atomic():
    artifact = _artifact()
    for name in ("stale_evaluation", "stale_base"):
        row = artifact[name]
        assert row["rejected"] is True
        assert row["all_state_and_references_unchanged"] is True
        assert row["transaction_state_after_abort"] == "aborted"
        assert row["resident_after_abort"] == 0
    abort = artifact["abort"]
    assert abort == {
        "active_unchanged": True,
        "resident_after_abort": 0,
        "state": "aborted",
        "terminal_commit_rejected": True,
    }


def test_transaction_ownership_accounting_balances_and_releases_everything():
    artifact = _artifact()
    accounting = artifact["accounting_before_release"]
    assert accounting["transaction_count"] == 5
    assert accounting["commit_count"] == 2
    assert accounting["abort_count"] == 3
    assert accounting["stale_commit_reject_count"] == 2
    assert accounting["winner_promoted_bytes"] == 314_002_226
    assert accounting["loser_released_bytes"] == 314_002_226
    assert accounting["transaction_resident_bytes"] == 0
    assert accounting["ownership_balance_exact"] is True
    branch = accounting["branch"]
    assert branch["branch_create_count"] == 12
    assert branch["branch_delete_count"] == 10
    assert branch["branch_promotion_count"] == 2
    assert branch["peak_bytes"] == 314_002_226
    assert branch["ownership_balance_bytes"] == 0
    assert accounting["active"]["ownership_balance_bytes"] == 0

    final = artifact["final_accounting"]
    assert final["transaction"]["transaction_resident_bytes"] == 0
    assert final["transaction"]["branch"]["resident_bytes"] == 0
    assert final["transaction"]["active"]["resident_bytes"] == 0
    assert final["snapshot"]["resident_bytes"] == 0
    assert final["transaction"]["anonymous_allocation_count"] == 0
    assert final["snapshot"]["anonymous_allocation_count"] == 0
    assert final["transaction"]["ownership_balance_exact"] is True
    assert final["memory"]["peak_bytes"] <= 340_000_000_000


def test_official_oracle_and_runtime_scope_remain_exact():
    artifact = _artifact()
    oracle = artifact["official_oracle"]
    assert oracle["first_16_match"] is True
    assert oracle["full_128_match"] is True
    assert oracle["all_full_vocab_logits_hashes_match"] is True
    assert oracle["failures_16"] == []
    assert oracle["failures_128"] == []
    assert all(value is False for value in artifact["runtime_changes"].values())
