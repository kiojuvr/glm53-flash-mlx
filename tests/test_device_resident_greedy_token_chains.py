from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts" / "probe_device_resident_greedy_token_chains.py"
ARTIFACT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-device-resident-greedy-token-chains-20260901.json"
)


def _module():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("device_chain_probe_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_chain_width_is_bounded_by_every_state_transition():
    probe = _module()
    limit = probe.ChainLimit(
        configured_width=16,
        remaining_generation=7,
        tokens_until_materialization=3,
        tokens_until_capacity=5,
    )
    assert probe.bounded_chain_width(limit) == 3


def test_chain_width_fail_closes_for_invalid_contract():
    probe = _module()
    with pytest.raises(ValueError):
        probe.bounded_chain_width(probe.ChainLimit(0, 1, 1, 1))
    assert probe.bounded_chain_width(probe.ChainLimit(2, 0, 2, 2)) == 0


def test_materialization_boundary_never_gets_crossed():
    probe = _module()
    assert probe.tokens_until_materialization(0) == 256
    assert probe.tokens_until_materialization(254) == 2
    assert probe.tokens_until_materialization(255) == 1
    assert probe.tokens_until_materialization(256) == 256
    assert probe.bounded_chain_width(probe.ChainLimit(4, 3, 2, 100)) == 2


def test_stop_fixture_covers_atomic_exit_reasons_and_all_positions():
    probe = _module()
    for width in (2, 4):
        rows = {
            accepted: probe.stop_reasons(width, accepted)
            for accepted in range(width + 1)
        }
        assert set(rows) == set(range(width + 1))
        assert "client_cancellation" in rows[0]
        assert "eos" in rows[1]
        assert "stop_token" in rows[1]
        assert any("multi_token_stop_sequence" in value for value in rows.values())
        assert any("generation_cap" in value for value in rows.values())
        assert "total_context_cap" in rows[width]


def test_hot_chain_does_not_materialize_device_tokens_as_python_scalars():
    source = SCRIPT.read_text()
    interval = source[source.index("def _chain(") : source.index("def _finish_chain_evidence")]
    assert ".item()" not in interval
    assert "int(predicted" not in interval
    assert "current = predicted.reshape(1, 1)" in interval
    assert "mx.stack(device_tokens)" in interval


def test_n16_has_a_parent_side_resource_budget_and_no_correctness_claim():
    source = SCRIPT.read_text()
    assert "SCREEN_TIME_LIMIT_SECONDS = 30.0" in source
    assert '"status": "aborted_resource_frontier"' in source
    assert '"correctness_claim": False' in source
    assert "remove_partial_trace(trace)" in source


def test_committed_device_chain_frontier_is_exact_but_misses_15_tps():
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"]
    assert artifact["screening_passed_arms"] == []
    assert artifact["n2_qualification_required"] is False
    baseline = artifact["arms"]["A"]["child"]["screen"]
    for arm, width in zip("ABCD", (1, 2, 4, 8), strict=True):
        row = artifact["arms"][arm]
        screen = row["child"]["screen"]
        comparison = artifact["comparisons_to_A"][arm]
        dynamic = row["dynamic_gap_attribution"]
        assert row["configuration"]["chain_width"] == width
        assert comparison["screening"]["all_exact"]
        assert screen["generated_token_sha256"] == baseline[
            "generated_token_sha256"
        ]
        assert screen["full_vocab_logits_hashes"] == baseline[
            "full_vocab_logits_hashes"
        ]
        assert dynamic["long_application_gap_count_matches_readbacks"]
        assert dynamic["application_starvation_long_gap_count"] == 64 // width
    assert max(
        artifact["comparisons_to_A"][arm]["performance"]["tokens_per_second"]
        for arm in "BCD"
    ) < 15.0


def test_committed_stop_and_materialization_transactions_are_exact():
    artifact = json.loads(ARTIFACT.read_text())
    for arm in ("B", "C"):
        child = artifact["arms"][arm]["child"]
        assert child["rollback_transactions"]["all_exact"]
        assert all(
            row["generated_prefix_exact"]
            and row["restored_replay_state_exact"]
            and row["snapshot_immutable"]
            for row in child["rollback_transactions"]["cases"]
        )
        boundary = child["materialization_boundary"]
        assert boundary["tokens_exact"]
        assert boundary["logits_exact"]
        assert boundary["cache_state_exact"]
        assert boundary["candidate"]["materialization_steps"] == [256]


def test_committed_n16_frontier_is_negative_evidence_only():
    artifact = json.loads(ARTIFACT.read_text())
    row = artifact["arms"]["E"]
    assert row["status"] == "aborted_resource_frontier"
    assert row["elapsed_lower_bound_seconds"] >= 300.0
    assert row["correctness_claim"] is False
    assert row["trace_complete"] is False
    assert row["partial_trace_deleted"] is True
