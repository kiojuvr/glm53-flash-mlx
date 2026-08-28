import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")

_SPEC = importlib.util.spec_from_file_location(
    "localize_grouped_fp8_divergence",
    Path(__file__).parents[1] / "scripts" / "localize_grouped_fp8_divergence.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

DISABLED_MIN_ROUTES = _MODULE.DISABLED_MIN_ROUTES
_logits_metrics = _MODULE._logits_metrics
_set_grouped_layers = _MODULE._set_grouped_layers


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
