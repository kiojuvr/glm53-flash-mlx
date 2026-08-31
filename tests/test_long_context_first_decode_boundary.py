import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "probe_long_context_first_decode_boundary.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-long-context-first-decode-boundary-20260831.json"
)


def _source() -> str:
    return SCRIPT.read_text()


def test_probe_scope_keeps_long_cold_prefill_fail_closed():
    source = _source()
    assert "TIER1_PROMPTS = (16, 128, 255, 256)" in source
    assert (
        "TIER2_CONTEXTS = (16384, 65536, 131072, 262143, 262144, 262145, 262146, 262147)"
        in source
    )
    assert '"256k cold prefill remains unsupported and unvalidated"' in source
    assert "validate_admission(" in source
    assert '"cache_abi": False' in source
    assert '"admission": False' in source


def test_probe_constructs_all_layer_state_and_preallocates_first_decode():
    source = _source()
    assert "KDA_LAYERS = tuple(" in source
    assert "for layer in EXPECTED_DSA" in source
    assert "capacity = _round_up(context + 1)" in source
    assert "mx.contiguous(conv" in source
    assert "mx.contiguous(recurrent" in source
    assert "EXPECTED_DIRECT_LEAVES" in source
    assert "EXPECTED_COMPACT_LEAVES" in source


def test_probe_records_ram_apc_materialization_and_exact_state_checks():
    source = _source()
    assert "clone_cache_entry" in source
    assert "_cache_exact(compact, restored)" in source
    assert "boundary_materialization_state_exact" in source
    assert "MATERIALIZATION_INTERVAL_TOKENS" in source
    assert "first_logits_hash" in source
    assert "post_state_hash" in source


def test_artifact_is_complete_when_present():
    if not ARTIFACT.exists():
        pytest.skip("M3 Ultra artifact has not been generated yet")
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["acceptance"]["accepted"] is True
    assert artifact["tier1_prompts"] == [16, 128, 255, 256]
    assert artifact["tier2_contexts"] == [
        16384,
        65536,
        131072,
        262143,
        262144,
        262145,
        262146,
        262147,
    ]
    assert artifact["claims"]["validated"] == "256k resident/restore to first decode"
    assert (
        artifact["claims"]["unsupported_unvalidated"]
        == "256k cold prefill to first decode"
    )


def test_signature_dataclass_and_stride_contract_on_metal():
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        pytest.skip("MLX/Metal is unavailable")
    spec = importlib.util.spec_from_file_location("first_decode_boundary_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    signature = module.FirstDecodeBoundarySignature(
        context_tokens=262144,
        construction_mode="fixture",
        cache_leaf_count=167,
        cache_schema_hash="schema",
        kda_state_hash="kda",
        indexpool_hash="pool",
        active_tail_count=1,
        selected_width=2051,
        physical_capacity_tokens=262400,
        first_token_id=1,
        first_logits_hash="logits",
        post_state_hash="post",
    )
    assert signature.selected_width == 2051
    assert module._canonical_row_major_strides((2, 3, 4), 2) == (24, 8, 2)
    assert module.EXPECTED_DIRECT_LEAVES == 112
    assert module.EXPECTED_COMPACT_LEAVES == 167
