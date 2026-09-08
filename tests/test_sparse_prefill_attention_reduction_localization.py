import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "localize_sparse_prefill_attention_reduction.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-sparse-prefill-attention-reduction-localization-20260908.json"
)


def test_localizer_checks_qk_softmax_and_av_separately():
    source = PROBE.read_text()
    assert '"explicit_fallback_matches_direct_fast_sdpa"' in source
    assert '"selected_qk_scores_byte_exact"' in source
    assert '"selected_precise_softmax_probabilities_byte_exact"' in source
    assert '"compact_av_accumulation_is_first_and_only_barrier"' in source
    assert '"attention_probability_times_value"' in source


def test_localizer_forbids_compact_av_runtime_promotion():
    source = PROBE.read_text()
    assert '"ordinary_compact_sdpa": False' in source
    assert '"compact_qk": True' in source
    assert '"compact_precise_softmax": True' in source
    assert '"compact_av": False' in source
    assert '"runtime": False' in source


def test_localization_artifact_is_complete_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert all(artifact["checks"].values())
    assert artifact["first_differing_stage"] == (
        "attention_probability_times_value"
    )
