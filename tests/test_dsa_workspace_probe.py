from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/probe_dsa_indexer_workspace_geometry.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-dsa-indexer-workspace-geometry-20260904.json"
)


def test_probe_scope_is_geometry_and_operator_only():
    source = SCRIPT.read_text()
    assert "plan_dsa_indexer_workspace" in source
    assert "_row_blocked_selection" in source
    assert "expand_selected_pools" in source
    assert "logical_context_logits_bytes_forbidden" in source
    for key in (
        "abi",
        "admission",
        "apc_namespace",
        "backend",
        "cache_implementation",
        "server",
    ):
        assert f'"{key}": False' in source


def test_artifact_passes_every_bounded_workspace_gate():
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "glm53-dsa-indexer-workspace-geometry-v1"
    assert artifact["complete"] is True
    assert artifact["probe_only"] is True
    assert all(artifact["acceptance"].values())
    assert artifact["decision"] == (
        "bounded_dsa_workspace_contract_ready_for_semantic_snapshot"
    )
    assert artifact["contract"] == {
        "identity": (
            "glm53-dsa-indexer-workspace-v1"
            "-ceil-div-kpool4"
            "-query-row-blocked"
            "-fp32-logits-64mib"
            "-transient-separate-from-indexpool"
        ),
        "index_kpool": 4,
        "index_topk": 2048,
        "max_fp32_logits_workspace_bytes": 64 << 20,
        "pool_count_formula": "quotient + (remainder != 0)",
        "key_dimension": "pool_count",
        "row_block_axis": "query",
        "selected_output_width": 2051,
        "full_logical_context_logits_forbidden": True,
        "maximum_context_resident_scratch_forbidden": True,
    }


def test_pool_tail_and_1m_geometry_are_exact():
    artifact = json.loads(ARTIFACT.read_text())
    boundaries = artifact["planner_boundaries"]
    expected = {
        "0": 0,
        "1": 1,
        "2": 1,
        "3": 1,
        "4": 1,
        "5": 2,
        "131071": 32_768,
        "131072": 32_768,
        "131073": 32_769,
        "262143": 65_536,
        "262144": 65_536,
        "262145": 65_537,
        "1048576": 262_144,
    }
    assert {key: row["pool_count"] for key, row in boundaries.items()} == expected
    geometry = artifact["synthetic_1m_geometry"]
    assert geometry == {
        "context_tokens": 1_048_576,
        "fp32_logits_workspace_bytes": 64 << 20,
        "index_kpool": 4,
        "max_workspace_bytes": 64 << 20,
        "num_query_rows": 256,
        "pool_count": 262_144,
        "query_block_count": 4,
        "query_block_rows": 64,
        "selected_pool_count": 512,
        "selected_token_width": 2051,
    }


def test_all_88_row_block_boundaries_are_byte_exact_and_range_safe():
    rows = json.loads(ARTIFACT.read_text())["small_correctness_cases"]
    assert len(rows) == 88
    assert {row["query_rows"] for row in rows} == {1, 63, 64, 65, 127, 128, 255, 256}
    assert {row["context_tokens"] for row in rows} == {
        1,
        2,
        3,
        4,
        5,
        2_047,
        2_048,
        2_049,
        4_095,
        4_096,
        4_097,
    }
    assert any(row["query_block_count"] > 1 for row in rows)
    assert all(
        row["raw_logits_byte_exact"]
        and row["topk_scores_byte_exact"]
        and row["topk_pool_indices_byte_exact"]
        and row["expanded_indices_byte_exact"]
        and row["sentinel_positions_byte_exact"]
        and row["selected_width"] == 2051
        and row["non_sentinel_out_of_range"] == 0
        for row in rows
    )


def test_128k_and_256k_use_pooled_logits_and_release_transients():
    rows = json.loads(ARTIFACT.read_text())["qualification"]
    expected = {
        "131072": (32_768, 32 << 20, 128 << 20, 33_554_432),
        "262144": (65_536, 64 << 20, 256 << 20, 67_108_864),
    }
    for key, (pools, logits, forbidden, argsort_order) in expected.items():
        row = rows[key]
        assert row["pool_count"] == pools
        assert row["workspace_bytes"] == logits
        assert row["logical_context_logits_bytes_forbidden"] == forbidden
        assert row["memory_accounting"]["transient"][
            "topk_index_scratch_bytes"
        ] == argsort_order
        assert row["memory_accounting"]["anonymous_allocation_bytes"] == 0
        assert row["candidate_post_release_active_drift_bytes"] == 0
        assert row["post_release_active_drift_bytes"] == 0
        assert row["differential_fixture_holds_reference_and_candidate"] is True
        assert row["candidate_operator_working_peak_bytes"] > logits
        assert row["differential_fixture_working_peak_bytes"] > row[
            "candidate_operator_working_peak_bytes"
        ]
        assert all(
            row[field]
            for field in (
                "raw_logits_byte_exact",
                "topk_scores_byte_exact",
                "topk_pool_indices_byte_exact",
                "expanded_indices_byte_exact",
                "sentinel_positions_byte_exact",
            )
        )
        assert row["non_sentinel_out_of_range"] == 0


def test_one_block_fast_path_and_full_model_regressions_pass():
    artifact = json.loads(ARTIFACT.read_text())
    performance = artifact["performance_screen"]
    assert performance["warmups"] == 2
    assert performance["samples"] == 5
    assert performance["repetitions_per_sample"] == 64
    assert performance["blocked_over_reference"] <= 1.01

    full_model = artifact["full_model"]
    assert full_model["layer3_indexer"] == {
        "head_dim": 128,
        "index_kpool": 4,
        "index_topk": 2048,
    }
    assert full_model["official_oracle"]["first_16_match"] is True
    assert full_model["official_oracle"]["full_128_match"] is True
    assert full_model["ram_apc"] == {
        "all_logits_hashes_match": True,
        "post_state_exact": True,
        "snapshot_immutable": True,
        "steps": 16,
    }
    assert artifact["runtime_changes"] == {
        "abi": False,
        "admission": False,
        "apc_namespace": False,
        "backend": False,
        "cache_implementation": False,
        "server": False,
    }
