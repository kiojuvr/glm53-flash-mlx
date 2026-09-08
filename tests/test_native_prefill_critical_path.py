from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "profile_native_prefill_critical_path.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-prefill-critical-path-20260908.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("native_prefill_profile", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _context_row(module, dominant="routed_moe"):
    expected = {
        "layer_handoff_attention_shell": 45,
        "kda_attention": 34,
        "dsa_attention": 11,
        "post_attention_ffn_shell": 45,
        "dense_ffn": 3,
        "routed_moe": 42,
        "final_norm_lm_head": 1,
    }
    return {
        "uninstrumented": {
            "median_wall_ms": 1_000.0,
            "median_host_graph_build_ms": 10.0,
            "repeat_logits_exact": True,
            "repeat_state_exact": True,
        },
        "instrumented": {
            "attribution": {
                "stage_sum_is_uninstrumented_wall": False,
                "stages": {
                    stage: {"call_count": count}
                    for stage, count in expected.items()
                },
            }
        },
        "exactness": {
            "instrumentation_final_logits_exact": True,
            "instrumentation_diagnostic_state_exact": True,
            "instrumentation_dsa_offsets_exact": True,
        },
        "dominant_synchronized_stage": dominant,
    }


def _complete_artifact(module):
    return {
        "contexts": {
            str(context): _context_row(module)
            for context in module.PROFILE_CONTEXTS
        },
        "process_peak_memory_bytes": 339_000_000_000,
        "runtime_changes": {
            "admission": False,
            "backend": False,
            "cache_abi": False,
            "kernel_abi": False,
            "server": False,
        },
    }


def test_profile_is_whole_model_structure_not_an_operator_speed_gate():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert "PROFILE_CONTEXTS = DESIGN_PREFILL_PROFILE_CONTEXTS" in source
    assert "STAGE_NAMES = (" in source
    assert '"kda_attention"' in source
    assert '"dsa_attention"' in source
    assert '"routed_moe"' in source
    assert '"final_norm_lm_head"' in source
    assert '"profile_goal_tokens_per_second": None' in source
    assert '"100_tps_is_a_checkpoint_not_a_stop_condition": True' in source


def test_authoritative_wall_and_synchronized_attribution_are_separate():
    source = SCRIPT.read_text()
    assert "_run_uninstrumented" in source
    assert "_run_instrumented" in source
    assert '"stage_sum_is_uninstrumented_wall": False' in source
    assert "each boundary is synchronized independently" in source
    assert "synthetic_prefix_state_avoids_hours_of_cold_prefill" in source


def test_profile_contexts_are_32k_128k_and_320k_with_256_rows():
    module = _module()
    assert module.PROFILE_CONTEXTS == (32_768, 131_072, 327_680)
    assert module.PROFILE_ROWS == 256
    assert module.EXPECTED_LAYER_COUNT == 45
    assert module.EXPECTED_KDA_LAYER_COUNT == 34
    assert module.EXPECTED_DSA_LAYER_COUNT == 11
    assert module.EXPECTED_DENSE_FFN_LAYER_COUNT == 3
    assert module.EXPECTED_ROUTED_MOE_LAYER_COUNT == 42


def test_vocab_size_comes_from_checkpoint_config_not_runtime_model_shape(tmp_path):
    module = _module()
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"vocab_size": 154_880}})
    )
    assert module._checkpoint_vocab_size(tmp_path) == 154_880
    source = SCRIPT.read_text()
    assert "model.language_model.vocab_size" not in source


def test_acceptance_requires_all_contexts_layers_exactness_and_bounded_peak():
    module = _module()
    artifact = _complete_artifact(module)
    checks = module._acceptance(artifact)
    assert all(checks.values())
    assert module._decision({**artifact, "acceptance": checks}) == (
        "design_native_prefill_plan_around_routed_moe"
    )

    artifact["contexts"]["327680"]["instrumented"]["attribution"][
        "stages"
    ]["dsa_attention"]["call_count"] = 10
    assert not module._acceptance(artifact)[
        "all_45_layers_and_final_boundary_attributed"
    ]


def test_incomplete_profile_never_selects_a_native_architecture():
    module = _module()
    artifact = _complete_artifact(module)
    artifact["contexts"].pop("327680")
    artifact["acceptance"] = module._acceptance(artifact)
    assert module._decision(artifact) == "profiling_incomplete_or_invalid"


def test_instrumentation_restores_original_layer_modules_on_failure():
    module = _module()
    layers = [
        SimpleNamespace(self_attn=object(), mlp=object())
        for _ in range(module.EXPECTED_LAYER_COUNT)
    ]
    model = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=layers))
    )
    originals = [(layer.self_attn, layer.mlp) for layer in layers]
    with pytest.raises(RuntimeError, match="fixture"):
        with module._instrument_prefill(model, module._StageRecorder()):
            assert all(
                layer.self_attn is not attention and layer.mlp is not mlp
                for layer, (attention, mlp) in zip(layers, originals, strict=True)
            )
            raise RuntimeError("fixture")
    assert all(
        layer.self_attn is attention and layer.mlp is mlp
        for layer, (attention, mlp) in zip(layers, originals, strict=True)
    )


def test_profile_artifact_when_present_is_authoritative():
    if not ARTIFACT.exists():
        pytest.skip("user-launched native prefill profile is pending")
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "glm53-native-prefill-critical-path-v1"
    if not artifact["complete"]:
        pytest.skip("artifact is an atomic progress record")
    assert artifact["accepted"] is True
    assert all(artifact["acceptance"].values())
    assert set(map(int, artifact["contexts"])) == {
        32_768,
        131_072,
        327_680,
    }
