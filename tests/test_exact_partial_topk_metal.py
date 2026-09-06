import ast
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_exact_partial_topk_metal.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-exact-partial-topk-metal-20260906.json"
)


def test_probe_is_score_only_and_keeps_runtime_unchanged():
    source = PROBE.read_text()
    ast.parse(source)
    assert "existing finite BF16 or FP32 Indexer score tensor" in source
    assert '"score_generation_changed": False' in source
    assert '"score_topk_fused": False' in source
    assert '"full_sort_materialized": False' in source
    assert '"probe_only": True' in source
    for component in ("runtime", "server", "apc", "cache_abi", "kernel_abi"):
        assert f'"{component}": False' in source


def test_kernel_encodes_stable_ties_and_bounded_partial_selection():
    source = PROBE.read_text()
    assert "score descending; equal scores preserve ascending source index" in source
    assert "0x1ffffu - index" in source
    assert "for (int shift = 44; shift >= 0; shift -= 4)" in source
    assert "threadgroup ulong candidates[K]" in source
    assert "mx.argsort(-scores, axis=-1)[..., :SELECT_K]" in source
    assert "MAX_POOL_COUNT = 65_600" in source
    assert '"persistent_allocation_bytes": 0' in source
    assert '"command_buffer_count_inferred": False' in source


def test_requested_artificial_and_real_frontiers_are_present():
    source = PROBE.read_text()
    assert "CONTEXTS = (2_048, 32_768, 131_072, 262_144)" in source
    for fixture in (
        "strictly_ascending",
        "strictly_descending",
        "all_equal",
        "large_tie_groups",
        "tie_at_kth_boundary",
        "positive_negative_zero",
        "very_small_fp32_differences",
        "bf16_boundary_derived_fp32",
        "production_bfloat16_ties",
        "production_sentinel_ties",
        "first_post_256k_partial_pool",
        "aligned_256k_pool_capacity",
    ):
        assert f'"{fixture}"' in source
    for evidence in (
        "topk_values_byte_exact",
        "topk_indices_byte_exact",
        "expanded_indices_byte_exact",
        "sentinel_positions_exact",
        "gather_byte_exact",
        "attention_output_byte_exact",
        "post_state_byte_exact",
        "all_full_vocab_logits_exact",
    ):
        assert f'"{evidence}"' in source
    assert 'candidate["scores"].astype(mx.float32)' in source


def test_fixed_performance_gates_are_not_relaxed():
    source = PROBE.read_text()
    assert "TOPK_KEEP_MS = 1.15" in source
    assert "TOPK_REJECT_MS = 1.50" in source
    assert "MIN_TOPK_SAVING_MS = 0.75" in source
    assert "MAX_FULL_MODEL_MS = 81.2" in source
    assert "MAX_WORKING_PEAK_DELTA = 32 << 20" in source


def test_artifact_is_exact_and_meets_keep_gate_when_present():
    if not ARTIFACT.exists():
        pytest.skip("M3 Ultra exact partial top-k artifact has not been generated")
    artifact = json.loads(ARTIFACT.read_text())
    if artifact.get("schema") != "glm53-exact-partial-topk-metal-v2":
        pytest.skip("stale partial top-k artifact predates the 65,537-pool fix")
    if not artifact["complete"]:
        pytest.skip("exact partial top-k model qualification is still resumable")
    assert artifact["accepted"] is True
    assert all(artifact["acceptance"].values())
    assert artifact["decision"] == "keep_exact_partial_topk_candidate"
    assert set(map(int, artifact["contexts"])) == {2048, 32768, 131072, 262144}
    assert (
        artifact["contexts"]["262144"]["topk"]["candidate"]["median_wall_ms"]
        <= 1.15
    )
