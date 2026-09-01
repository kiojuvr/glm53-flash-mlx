from pathlib import Path

import json


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_functional_stateful_decode_executable.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-functional-stateful-decode-executable-20260902.json"
)


def _source() -> str:
    return SCRIPT.read_text()


def test_probe_is_tier_gated_and_probe_only():
    source = _source()
    assert '"probe_only": True' in source
    assert '"runtime_changes": False' in source
    assert '"tier0_state_schema"' in source
    assert '"tier1_kda"' in source
    assert '"tier2_dsa"' in source
    assert '"tier3_complete_layer"' in source
    assert '"tier4_full_executable"' in source
    assert '"Tier 1 KDA gate failed; DSA was not loaded"' in source


def test_kda_state_is_explicit_and_range_checked_before_execution():
    source = _source()
    assert "def validate_kda_state" in source
    assert "exactly conv and recurrent" in source
    assert "KDA state schema mismatch" in source
    assert "functional_kda_decode(" in source
    assert "position + mx.array(1, mx.int32)" in source


def test_kda_compile_signature_and_offset_fixtures_are_first_class():
    source = _source()
    assert "KDA_OFFSET_FIXTURES = (0, 1, 255, 256, 2048)" in source
    assert "trace_counter" in source
    assert "fixed_signature" in source
    assert "host_build_reduction" in source
    assert "strided_contiguous_byte_identical" in source


def test_full_compile_has_a_fifteen_minute_budget():
    source = _source()
    assert "FULL_COMPILE_BUDGET_SECONDS = 15 * 60" in source
    assert '"--cold-compile-budget-seconds"' in source


def test_selective_checkpoint_loading_precedes_expensive_tiers():
    source = _source()
    assert "def _checkpoint_tensors" in source
    assert "_load_kda_attention" in source
    assert "checkpoint prefix has no tensors" in source


def test_committed_negative_evidence_is_complete_and_tier_gated():
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["evidence_complete"] is True
    assert artifact["runtime_changes"] is False
    assert artifact["decision"] == (
        "reject_functional_stateful_decode_at_kda_exactness"
    )
    tier0 = artifact["tiers"]["tier0_state_schema"]
    tier1 = artifact["tiers"]["tier1_kda"]
    assert tier0["passed"] is True
    assert tier1["fixed_signature"] is True
    assert tier1["offset_updates_exact"] is True
    assert tier1["final_state_byte_identical"] is True
    assert tier1["all_step_outputs_byte_identical"] is False
    assert tier1["first_divergent_output_step"] == 7
    assert tier1["eager_repeated_execution_byte_identical"] is True
    assert tier1["compiled_repeated_execution_byte_identical"] is True
    assert artifact["tiers"]["tier2_dsa"]["executed"] is False
