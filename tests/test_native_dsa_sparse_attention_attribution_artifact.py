import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-dsa-sparse-attention-attribution-20260907.json"
)


def test_native_sparse_attention_regression_is_exactly_attributed():
    result = json.loads(ARTIFACT.read_text())
    assert result["complete"] is True
    assert result["accepted"] is True
    assert result["decision"] == "stop_native_attention_keep_tier1_score_island"
    assert result["artificial"]["exactness"]["all_exact"] is True
    assert result["real_layers"]["all_exact"] is True
    assert len(result["real_layers"]["layers"]) == 11
    assert all(row["all_exact"] for row in result["real_layers"]["layers"])


def test_neither_isolated_attention_half_explains_tier2_regression():
    result = json.loads(ARTIFACT.read_text())
    timing = result["real_layers"]["timing"]
    # Tier 2 regressed by about 0.308 ms in the qualification.  At the durable
    # split, prepare is faster and attention math is effectively neutral.
    assert timing["prepare"]["native_minus_mlx_ms"] < 0.0
    assert 0.0 <= timing["attention_math"]["native_minus_mlx_ms"] < 0.05
    assert result["runtime_changes"] == {
        "runtime": False,
        "server": False,
        "apc": False,
        "cache_abi": False,
        "kernel_abi": False,
    }
