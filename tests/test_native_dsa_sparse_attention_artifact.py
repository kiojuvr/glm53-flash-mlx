import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-dsa-sparse-attention-island-20260907.json"
)


def test_tier2_artifact_records_exact_but_fixed_gate_rejection():
    result = json.loads(ARTIFACT.read_text())
    assert result["complete"] is True
    assert result["accepted"] is False
    assert result["decision"] == "stop_or_redesign_native_sparse_attention_island"
    assert result["artificial_native_contract"]["all_exact"] is True
    assert result["official_oracle"]["first_16_match"] is True
    assert result["official_oracle"]["full_128_match"] is True
    assert result["acceptance"]["artificial_and_all_dsa_layers_byte_exact"] is True
    assert (
        result["acceptance"][
            "256k_incremental_full_model_saving_at_least_0_75ms"
        ]
        is False
    )


def test_tier2_artifact_keeps_exactness_and_resources_bounded_at_both_contexts():
    result = json.loads(ARTIFACT.read_text())
    for context in ("2048", "262144"):
        row = result["contexts"][context]
        assert row["all_layers_exact"] is True
        assert all(
            layer["selected_indices_byte_exact"]
            and layer["selected_valid_byte_exact"]
            and layer["gathered_latent_byte_exact"]
            and layer["attention_output_byte_exact"]
            and layer["buffer_identities_stable"]
            for layer in row["layers"]
        )
        assert row["full_model"]["all_logits_byte_exact"] is True
        assert row["full_model"]["post_state_byte_exact"] is True
    assert result["maximum_concurrent_native_plan_scratch_bytes"] <= 128 << 20
    assert result["process_peak_memory_bytes"] <= 340_000_000_000


def test_tier2_wall_gain_is_positive_but_below_the_predeclared_gate():
    result = json.loads(ARTIFACT.read_text())
    short = result["contexts"]["2048"]
    long = result["contexts"]["262144"]
    assert short["full_model"]["incremental_native_saving_ms"] > 0
    assert 0 < long["full_model"]["incremental_native_saving_ms"] < 0.75
    assert long["operator_timing"]["incremental_native_saving_ms"] < 0
    assert result["runtime_changes"] == {
        "runtime": False,
        "server": False,
        "apc": False,
        "cache_abi": False,
        "kernel_abi": False,
    }
