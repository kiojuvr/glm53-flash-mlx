import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qualify_exact_fused_packed_decode_runtime.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-exact-fused-packed-decode-runtime-20260907.json"
)


def test_qualification_reuses_the_full_direct_vs_packed_runtime_suite():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert "import probe_packed_decode_runtime as qualification" in source
    assert "qualification.main()" in source
    assert "TARGET_2K_TPS = 15.0" in source
    assert '"2k_decode_at_least_15_tps"' in source
    assert '"delegated_runtime_qualification_accepted"' in source
    assert 'parser.add_argument("--server-port", type=int, default=18083)' in source


def test_qualification_requires_both_prequalified_fused_sources():
    source = SCRIPT.read_text()
    assert "m3ultra512-residual-packed-decode-moe-fusion-20260901.json" in source
    assert "m3ultra512-native-fused-decode-composition-20260907.json" in source
    assert 'residual.get("selected_aggregation") != "B1"' in source
    assert 'residual.get("aggregation_exact", {}).get("B1")' in source
    assert 'residual.get("shared_exact")' in source
    assert 'composition.get("accepted")' in source


def test_qualification_keeps_prefill_grouped_and_defaults_unchanged():
    source = SCRIPT.read_text()
    assert '"prefill_semantics": "Direct packed-bank semantics"' in source
    assert '"grouped_kernel_calls": 0' in source
    assert '"default_backend": False' in source
    assert '"prompt_admission": False' in source
    assert '"grouped_backend": False' in source


def test_qualification_artifact_records_exact_keep_and_15_tps_stop_when_present():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "glm53-exact-fused-packed-decode-runtime-v2"
    assert artifact["complete"] is True
    assert artifact["accepted"] is False
    assert artifact["decision"] == "stop_or_requalify_exact_fused_packed_decode_runtime"
    assert artifact["promotion_acceptance"] == {
        "2k_decode_at_least_15_tps": False,
        "delegated_runtime_qualification_accepted": True,
        "source_fused_composition_prequalified": True,
        "v2_exact_fused_kernel_abi": True,
    }
    assert artifact["acceptance"]["accepted"] is True
    assert all(
        value
        for name, value in artifact["acceptance"].items()
        if name != "accepted"
    )
    assert artifact["comparisons"]["decode_2k_speedup"] >= 1.12
    assert artifact["comparisons"]["decode_256k_speedup"] >= 1.10
    assert artifact["comparisons"]["decode_4096_speedup"] >= 1.10
    assert artifact["comparisons"]["packed_compact_2k_to_256k_retention"] >= 0.90
    assert artifact["packed_decode"]["frontier"]["direct:2049"][
        "tokens_per_second"
    ] < 15.0
    assert artifact["promotion"]["grouped_kernel_calls"] == 0
