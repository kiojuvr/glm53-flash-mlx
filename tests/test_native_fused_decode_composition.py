import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_native_fused_decode_composition.py"


def test_composition_reuses_only_prequalified_exact_components():
    source = PROBE.read_text()
    ast.parse(source)
    assert "NativeIndexPoolUpdateSelectionPlan" in source
    assert "m3ultra512-residual-packed-decode-moe-fusion-20260901.json" in source
    assert 'aggregation != "B1"' in source
    assert 'source["aggregation_exact"].get(aggregation)' in source
    assert 'source["shared_exact"]' in source
    assert 'all(source["correctness"].values())' in source
    assert "residual.Arm(aggregation, True)" in source


def test_composition_has_no_new_execution_boundary_or_production_change():
    source = PROBE.read_text()
    assert "This probe does not add another native boundary" in source
    assert '"exact_nonproduction_composition": True' in source
    for component in ("runtime", "server", "apc", "cache_abi", "kernel_abi"):
        assert f'"{component}": False' in source


def test_composition_uses_full_model_exactness_and_fixed_performance_gates():
    source = PROBE.read_text()
    assert "CONTEXTS = (2_048, 262_144)" in source
    assert "TARGET_2K_TPS = 15.0" in source
    assert "MIN_256K_SAVING_MS = 5.0" in source
    assert "MIN_CONTEXT_RETENTION = 0.90" in source
    assert '"all_full_vocab_logits_byte_exact"' in source
    assert '"all_generated_tokens_exact"' in source
    assert '"post_state_byte_exact"' in source
    assert '"all_42_sparse_layers_use_fused_path"' in source
    assert '"official_oracle_exact_both_arms"' in source


def test_composition_keeps_native_update_for_both_arms():
    source = PROBE.read_text()
    assert "update_island._native_arm(registry, True)" in source
    assert "residual._runtime(fused_arm)" in source
    assert "A_native_indexpool_production_packed" in source
    assert "B_native_indexpool_exact_fused_packed" in source
    assert "advance_fused_moe_topology_into_native_executor" in source
