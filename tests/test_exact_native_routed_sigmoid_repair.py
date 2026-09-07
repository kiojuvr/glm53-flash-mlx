import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_exact_native_routed_sigmoid_repair.py"
METAL = ROOT / "native_execution" / "native_indexer_plan.metal"
HEADER = ROOT / "native_execution" / "native_packed_moe_plan.h"


def test_repair_changes_only_the_specialized_routed_sigmoid_intrinsic():
    source = PROBE.read_text()
    metal = METAL.read_text()
    ast.parse(source)
    assert '"metal::exp to metal::precise::exp"' in source
    assert '"generic_kernel_changed": False' in source
    assert '"shared_expert_kernel_changed": False' in source
    assert '"projection_reduction_changed": False' in source
    assert metal.count("metal::precise::exp") == 1
    precise = metal.index("metal::precise::exp")
    specialized = metal.index(
        "glm53_native_glm53_packed_selected8_gate_up_swiglu("
    )
    diagnostic = metal.index(
        "glm53_native_glm53_packed_selected8_gate_up_swiglu_diagnostic("
    )
    assert specialized < precise < diagnostic


def test_repair_requires_localized_and_prior_exact_sigmoid_evidence():
    source = PROBE.read_text()
    assert "first_differing_stage" in source
    assert '!= "sigmoid_bf16"' in source
    assert '"gate_projection_bf16"' in source
    assert '"up_projection_bf16"' in source
    assert '["selected_mode"] != 7' in source
    assert "actual_and_synthetic_sigmoid_bits_exact" in source


def test_repair_replays_all_layers_before_official_oracle_and_performance():
    source = PROBE.read_text()
    assert "42 * TARGET_STEP" in source
    assert "all_2856_owned_layer_steps_byte_exact" in source
    assert "official_16_128_oracle_exact_both_arms" in source
    assert "advance_exact_native_moe_to_performance_requalification" in source
    assert "uses_precise_routed_sigmoid" in HEADER.read_text()
    assert '"probe_only": True' in source
    for component in (
        "runtime",
        "server",
        "apc",
        "cache_abi",
        "production_kernel_abi",
    ):
        assert f'"{component}": False' in source
