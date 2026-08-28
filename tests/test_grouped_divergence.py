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

DISABLED_MIN_ROUTES = _MODULE.DISABLED_MIN_ROUTES
_logits_metrics = _MODULE._logits_metrics
_set_grouped_layers = _MODULE._set_grouped_layers
_first_route_divergence = _TRACE_MODULE._first_route_divergence
_first_route_divergence_context = _TRACE_MODULE._first_route_divergence_context
_route_metrics = _TRACE_MODULE._route_metrics


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
    assert _first_route_divergence(rows, 3) == 5
    context = _first_route_divergence_context(boundaries, rows, 3)
    assert context["layer"] == 5
    assert context["router"] == changed
    assert context["precursor_boundaries"]["normalized_router_input"] == {
        "relative_l2": 5.0
    }
