import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "localize_compiled_kda_numerical_barriers.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-compiled-kda-numerical-barriers-20260902.json"
)


def _source() -> str:
    return SCRIPT.read_text()


def test_anchor_arms_are_orthogonal_and_probe_only():
    source = _source()
    assert '"A": {"gated_rmsnorm": "eager", "final_projection": "eager"}' in source
    assert '"B": {"gated_rmsnorm": "compiled", "final_projection": "compiled"}' in source
    assert '"C": {"gated_rmsnorm": "eager-materialized", "final_projection": "compiled"}' in source
    assert '"D": {"gated_rmsnorm": "compiled-materialized", "final_projection": "eager"}' in source
    assert '"E": {"gated_rmsnorm": "eager-materialized", "final_projection": "eager"}' in source
    assert '"probe_only": True' in source
    assert '"runtime_changes": False' in source


def test_norm_ladder_and_bit_pattern_fixtures_are_explicit():
    source = _source()
    assert "BIT_PATTERN_STEPS = (0, 1, 6, 7, 8, 63)" in source
    for stage in (
        "square",
        "mean_square",
        "inverse_rms",
        "normalized",
        "weighted",
        "sigmoid_gate",
        "gated_f32",
        "dtype_rounding",
    ):
        assert f'"{stage}"' in source
    assert "ulp_distance" in source
    assert "reference_bits" in source
    assert "actual_bits" in source


def test_decision_requires_eager_norm_anchor_to_fix_projection():
    source = _source()
    assert '"gated_rmsnorm_is_only_observed_blocker"' in source
    assert '"final_projection_is_independent_blocker"' in source
    assert (
        '"implement_probe_exact_gated_rmsnorm_primitive_with_eager_sigmoid_order"'
        in source
    )
    assert '"stop_numerical_barrier_path"' in source


def test_committed_artifact_localizes_the_blocker():
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["runtime_changes"] is False
    assert artifact["steps"] == 64
    arms = artifact["localization"]["arms"]
    assert arms["A"]["all_final_projection_byte_identical"] is True
    assert arms["B"]["all_final_projection_byte_identical"] is False
    assert arms["C"]["all_final_projection_byte_identical"] is True
    assert arms["D"]["all_final_projection_byte_identical"] is False
    assert arms["E"]["all_final_projection_byte_identical"] is True
    assert artifact["decision"]["gated_rmsnorm_is_only_observed_blocker"] is True
    assert artifact["decision"]["final_projection_is_independent_blocker"] is False
    assert artifact["decision"]["first_numerical_barrier"] == {
        "stage": "sigmoid_gate",
        "step": 0,
    }
    assert artifact["acceptance"]["full_compiled_state_exact"] is True
