import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_sparse_prefill_reordering_equivalence.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-sparse-prefill-reordering-equivalence-20260908.json"
)


def test_probe_compares_direct_dense_and_reordered_sparse_prefill():
    source = PROBE.read_text()
    assert "CONTEXTS = (2_048, 32_768)" in source
    assert '"gather_after_full_k_equals_project_after_latent_gather"' in source
    assert '"gather_after_full_v_equals_project_after_latent_gather"' in source
    assert '"dense_mask_attention_equals_sorted_compact_attention"' in source
    assert '"dense_mask_attention_equals_latent_gather_compact_attention"' in source
    assert '"implement_sorted_gather_project_compact_attention_native_region"' in source


def test_probe_does_not_promote_model_free_equivalence_to_runtime():
    source = PROBE.read_text()
    assert '"runtime": False' in source
    assert '"production_prefill": False' in source
    assert '"requires_real_checkpoint_confirmation": True' in source


def test_artifact_is_self_consistent_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    if artifact["accepted"]:
        assert all(artifact["checks"].values())
    else:
        assert not all(artifact["checks"].values())
