from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts" / "probe_asynchronous_autoregressive_submission.py"
ARTIFACT = (
    REPOSITORY
    / "bench-results"
    / "m3ultra512-asynchronous-autoregressive-submission-20260901.json"
)


def _module():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("async_submission_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("a_only", "pair_a", "pair_b", "expected"),
    [
        (6.0, 6.2, 5.8, "event_scoped"),
        (6.0, 9.2, 0.1, "stream_wide"),
        (6.0, 8.0, 2.0, "inconclusive"),
    ],
)
def test_readback_scope_classifier(a_only, pair_a, pair_b, expected):
    probe = _module()
    assert (
        probe.classify_readback_scope(
            a_only_wait_ms=a_only,
            pair_a_wait_ms=pair_a,
            pair_b_remaining_wait_ms=pair_b,
        )
        == expected
    )


def test_readback_scope_classifier_rejects_invalid_control():
    probe = _module()
    with pytest.raises(ValueError):
        probe.classify_readback_scope(
            a_only_wait_ms=0.0,
            pair_a_wait_ms=1.0,
            pair_b_remaining_wait_ms=1.0,
        )


def test_tier2_is_guarded_by_event_scoped_tier1():
    source = SCRIPT.read_text()
    assert '"proceed_to_full_model_lookahead"' in source
    assert '"reject_MLX_Python_async_lookahead"' in source
    assert '"executed": False' in source
    assert "scope == \"event_scoped\"" in source


def test_committed_async_scope_is_stream_wide_and_skips_full_model():
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"]
    tier1 = artifact["tier1"]
    medians = tier1["child"]["medians"]
    derived = tier1["derived"]
    assert tier1["child"]["readback_scope"] == "stream_wide"
    assert tier1["decision"] == "reject_MLX_Python_async_lookahead"
    assert medians["pair_a_readback_wait_ms"] > (
        medians["a_only_wait_ms"] * 1.5
    )
    assert medians["pair_b_remaining_wait_ms"] < (
        medians["a_only_wait_ms"] * 0.25
    )
    assert derived["a_then_b_gpu_busy_ratio"] > 1.7
    assert derived["b_frame_end_after_a_median_ms"] > 0
    assert derived["a_item_waits_until_b_completion"]
    assert tier1["gpu_timeline"]["observed_burst_count"] == 10
    assert tier1["acceptance"]["stream_wide_readback_observed"]
    assert artifact["tier2"]["executed"] is False
    assert artifact["tier2"]["correctness_claim"] is False
    assert artifact["tier2"]["performance_claim"] is False
