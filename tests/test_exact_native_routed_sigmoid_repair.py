import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_exact_native_routed_sigmoid_repair.py"
METAL = ROOT / "native_execution" / "native_indexer_plan.metal"
HEADER = ROOT / "native_execution" / "native_packed_moe_plan.h"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-exact-native-routed-sigmoid-repair-20260907.json"
)
FAST_ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-exact-native-routed-fast-sigmoid-repair-20260907.json"
)


def test_repair_changes_only_the_specialized_routed_sigmoid_intrinsic():
    source = PROBE.read_text()
    metal = METAL.read_text()
    ast.parse(source)
    assert '"metal::exp to metal::fast::exp"' in source
    assert '"generic_kernel_changed": False' in source
    assert '"shared_expert_kernel_changed": False' in source
    assert '"projection_reduction_changed": False' in source
    fast = metal.index("metal::fast::exp")
    specialized = metal.index(
        "glm53_native_glm53_packed_selected8_gate_up_swiglu("
    )
    diagnostic = metal.index(
        "glm53_native_glm53_packed_selected8_gate_up_swiglu_diagnostic("
    )
    assert specialized < fast < diagnostic


def test_repair_requires_localized_and_prior_exact_sigmoid_evidence():
    source = PROBE.read_text()
    assert "first_differing_stage" in source
    assert '!= "sigmoid_bf16"' in source
    assert '"gate_projection_bf16"' in source
    assert '"up_projection_bf16"' in source
    assert '["selected_formula"] != "fast_bf16"' in source
    assert '["exact_formulas"] != ["fast_bf16"]' in source


def test_repair_replays_all_layers_before_official_oracle_and_performance():
    source = PROBE.read_text()
    assert "42 * TARGET_STEP" in source
    assert "all_2856_owned_layer_steps_byte_exact" in source
    assert "official_16_128_oracle_exact_both_arms" in source
    assert "advance_exact_native_moe_to_performance_requalification" in source
    assert "uses_fast_bf16_routed_sigmoid" in HEADER.read_text()
    assert '"probe_only": True' in source
    for component in (
        "runtime",
        "server",
        "apc",
        "cache_abi",
        "production_kernel_abi",
    ):
        assert f'"{component}": False' in source


def test_precise_bf16_exp_repair_is_archived_as_a_negative_result():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is False
    assert artifact["decision"] == "stop_or_relocalize_precise_native_sigmoid_repair"
    replay = artifact["owned_activation_replay"]
    assert replay["exact_layer_step_comparisons"] == 1214
    assert replay["plan_evidence"]["all_use_precise_routed_sigmoid"]
    first = replay["first_divergence"]
    assert (first["step"], first["layer"]) == (29, 41)
    assert first["stage_localization"]["first_differing_stage"] == "routed_hidden"


def test_fast_bf16_exp_repair_closes_the_full_correctness_gate():
    artifact = json.loads(FAST_ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert artifact["decision"] == (
        "advance_exact_native_moe_to_performance_requalification"
    )
    assert all(artifact["acceptance"].values())
    replay = artifact["owned_activation_replay"]
    assert replay["exact_layer_step_comparisons"] == 2_856
    assert replay["expected_comparisons"] == 2_856
    assert replay["first_divergence"] is None
    assert replay["plan_evidence"] == {
        "all_fixed_arena_invariants": True,
        "all_use_fast_bf16_routed_sigmoid": True,
        "all_use_shape_specialized_kernel": True,
        "plan_count": 42,
    }
    oracle = artifact["official_oracle"]
    assert oracle["A_exact_composition"]["all_full_vocab_logits_hashes_match"]
    assert oracle["B_fast_bf16_sigmoid_native"][
        "all_full_vocab_logits_hashes_match"
    ]
