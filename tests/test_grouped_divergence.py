import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import mlx.core as mx
except ImportError:
    pytest.skip("MLX/Metal is unavailable", allow_module_level=True)

_SPEC = importlib.util.spec_from_file_location(
    "localize_grouped_fp8_divergence",
    Path(__file__).parents[1] / "scripts" / "localize_grouped_fp8_divergence.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_TRACE_SPEC = importlib.util.spec_from_file_location(
    "trace_grouped_fp8_route_amplification",
    Path(__file__).parents[1]
    / "scripts"
    / "trace_grouped_fp8_route_amplification.py",
)
assert _TRACE_SPEC is not None and _TRACE_SPEC.loader is not None
_TRACE_MODULE = importlib.util.module_from_spec(_TRACE_SPEC)
sys.modules[_TRACE_SPEC.name] = _TRACE_MODULE
_TRACE_SPEC.loader.exec_module(_TRACE_MODULE)

_SUFFIX_SPEC = importlib.util.spec_from_file_location(
    "sweep_grouped_fp8_suffix",
    Path(__file__).parents[1] / "scripts" / "sweep_grouped_fp8_suffix.py",
)
assert _SUFFIX_SPEC is not None and _SUFFIX_SPEC.loader is not None
_SUFFIX_MODULE = importlib.util.module_from_spec(_SUFFIX_SPEC)
sys.modules[_SUFFIX_SPEC.name] = _SUFFIX_MODULE
_SUFFIX_SPEC.loader.exec_module(_SUFFIX_MODULE)

DISABLED_MIN_ROUTES = _MODULE.DISABLED_MIN_ROUTES
_logits_metrics = _MODULE._logits_metrics
_set_grouped_layers = _MODULE._set_grouped_layers
_first_route_divergence = _TRACE_MODULE._first_route_divergence
_first_route_divergence_context = _TRACE_MODULE._first_route_divergence_context
_route_metrics = _TRACE_MODULE._route_metrics
_route_layer_metrics = _SUFFIX_MODULE._route_layer_metrics
_screen = _SUFFIX_MODULE._screen
_RecordedIndicesCurrentScoresGate = (
    _SUFFIX_MODULE._RecordedIndicesCurrentScoresGate
)


def test_set_grouped_layers_enables_only_requested_layers():
    grouped = {layer: SimpleNamespace(min_routes=None) for layer in range(3, 6)}

    _set_grouped_layers(grouped, {3, 5}, grouped_min_routes=256)

    assert grouped[3].min_routes == 256
    assert grouped[4].min_routes == DISABLED_MIN_ROUTES
    assert grouped[5].min_routes == 256


def test_logits_metrics_distinguish_exact_and_rank_changes():
    reference = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    exact = _logits_metrics(reference, reference, top_k=3)
    changed = _logits_metrics(
        reference, np.array([0.0, 1.0, 3.5, 3.0], dtype=np.float32), top_k=3
    )

    assert exact["array_equal"]
    assert exact["relative_l2"] == 0.0
    assert exact["kl_reference_to_actual"] == 0.0
    assert not changed["array_equal"]
    assert not changed["argmax_match"]
    assert changed["relative_l2"] > 0.0


def test_route_metrics_and_first_divergence_context():
    reference = {
        "indices": np.array([[0, 1], [2, 3]], dtype=np.int32),
        "scores": np.array([[0.8, 0.2], [0.7, 0.3]], dtype=np.float32),
    }
    exact = _route_metrics(reference, reference)
    changed = _route_metrics(
        reference,
        {
            "indices": np.array([[0, 1], [2, 4]], dtype=np.int32),
            "scores": np.array([[0.8, 0.2], [0.6, 0.4]], dtype=np.float32),
        },
    )
    rows = {3: exact, 4: exact, 5: changed}
    boundaries = {
        "5": {
            name: {"relative_l2": float(index)}
            for index, name in enumerate(
                (
                    "layer_input",
                    "attention_hc_collapse",
                    "attention_output",
                    "post_attention_hc_expand",
                    "ffn_hc_collapse",
                    "normalized_router_input",
                )
            )
        }
    }

    assert exact["slot_agreement"] == 1.0
    assert changed["changed_route_slots"] == 1
    assert changed["tokens_with_identical_top8_set"] == 1
    assert changed["tokens_with_changed_top8_set"] == 1
    assert changed["tokens_with_order_only_change"] == 0
    assert changed["expert_membership_replacements"] == 1
    assert _first_route_divergence(rows, 3) == 5
    context = _first_route_divergence_context(boundaries, rows, 3)
    assert context["layer"] == 5
    assert context["router"] == changed
    assert context["precursor_boundaries"]["normalized_router_input"] == {
        "relative_l2": 5.0
    }


def test_suffix_router_metrics_separate_order_membership_and_aligned_scores():
    reference = {
        "indices": np.array([[0, 1]], dtype=np.int32),
        "scores": np.array([[0.8, 0.2]], dtype=np.float32),
    }
    order_only, differences = _route_layer_metrics(
        reference,
        {
            "indices": np.array([[1, 0]], dtype=np.int32),
            "scores": np.array([[0.2, 0.8]], dtype=np.float32),
        },
    )
    membership, _ = _route_layer_metrics(
        reference,
        {
            "indices": np.array([[0, 2]], dtype=np.int32),
            "scores": np.array([[0.75, 0.25]], dtype=np.float32),
        },
    )

    assert order_only["slot_position_mismatches"] == 2
    assert order_only["tokens_with_changed_top8_set"] == 0
    assert order_only["tokens_with_order_only_change"] == 1
    assert order_only["expert_membership_replacements"] == 0
    assert np.array_equal(differences, np.zeros(2))
    assert membership["tokens_with_changed_top8_set"] == 1
    assert membership["expert_membership_replacements"] == 1
    assert membership["score_difference_by_expert_id"]["matched_memberships"] == 1


def test_suffix_screen_is_strictly_a_pareto_filter():
    passing = {
        "argmax_match": True,
        "top_k_set_match": True,
        "relative_l2": 0.02,
        "kl_reference_to_actual": 5e-4,
    }
    assert _screen(passing)
    assert not _screen({**passing, "relative_l2": 0.02001})
    assert not _screen({**passing, "top_k_set_match": False})


def test_indices_only_gate_recomputes_scores_for_direct_expert_ids():
    delegate = SimpleNamespace(
        weight=mx.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=mx.float32
        ),
        top_k=2,
        norm_topk_prob=True,
        routed_scaling_factor=2.0,
    )
    gate = _RecordedIndicesCurrentScoresGate(
        delegate, np.array([[2, 0]], dtype=np.int32)
    )
    indices, scores = gate(mx.array([[1.0, 2.0]], dtype=mx.float32))
    mx.eval(indices, scores)
    expected_raw = 1.0 / (1.0 + np.exp(-np.array([3.0, 1.0])))
    expected = expected_raw / expected_raw.sum() * 2.0

    assert np.array_equal(np.asarray(indices), np.array([[2, 0]]))
    assert np.allclose(np.asarray(scores), expected, rtol=1e-6, atol=1e-6)
