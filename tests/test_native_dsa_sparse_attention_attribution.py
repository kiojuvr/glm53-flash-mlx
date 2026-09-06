from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / "native_execution" / "native_dsa_attention_plan.cpp"
HEADER = ROOT / "native_execution" / "native_dsa_attention_plan.h"
BINDINGS = ROOT / "native_execution" / "bindings.cpp"
PROBE = ROOT / "scripts" / "attribute_native_dsa_sparse_attention_regression.py"


def test_attribution_uses_probe_only_durable_numerical_boundary():
    header = HEADER.read_text()
    bindings = BINDINGS.read_text()
    source = PROBE.read_text()
    assert "debug_prepare_inputs" in header
    assert "debug_attention_math" in header
    assert "debug_prepare_inputs" in bindings
    assert "debug_attention_math" in bindings
    assert "_mlx_prepare" in source
    assert "_mlx_math" in source
    assert '"artificial_attribution_contract"' in source
    assert '"exact_layer"' in source


def test_attribution_does_not_relax_tier2_gate_or_change_production_abi():
    source = PROBE.read_text()
    assert "MIN_256K_INCREMENTAL_SAVING_MS" not in source
    assert '"runtime": False' in source
    assert '"server": False' in source
    assert '"apc": False' in source
    assert '"cache_abi": False' in source
    assert '"kernel_abi": False' in source
    assert "prototype_exact_indirect_latent_loaders" in source
    assert "stop_native_attention_keep_tier1_score_island" in source


def test_debug_entry_points_reuse_fixed_arena_without_allocation_or_wait():
    cpp = CPP.read_text()
    start = cpp.index("debug_prepare_inputs(")
    body = cpp[start:]
    assert "scaled_query_" in body
    assert "gathered_latent_" in body
    assert "attention_scores_" in body
    assert "splitk_accum_" in body
    assert "attention_output_" in body
    assert "mx::allocator::malloc" not in body
    assert "mx::eval(" not in body
    assert "mx::synchronize(" not in body
