"""Native execution bridge for GLM-5.3 on the qualified MLX ABI.

The IndexPool update/Tier-1 plan is available to the explicit production
opt-in. Other plans remain probe-only. The binary is built separately, so an
unavailable or incompatible extension fails closed without changing the MLX
fallback runtime.
"""

from ._ext import (
    NativeDSAScoreSelectionPlan,
    NativeDSASparseAttentionPlan,
    NativeIndexSelectionPlan,
    NativeIndexPoolUpdateSelectionPlan,
    NativePackedMoEDecodePlan,
    NativePackedMoERoutedDiagnostic,
    NativeRoutedSigmoidFormulaSweep,
)

__all__ = [
    "NativeDSAScoreSelectionPlan",
    "NativeDSASparseAttentionPlan",
    "NativeIndexSelectionPlan",
    "NativeIndexPoolUpdateSelectionPlan",
    "NativePackedMoEDecodePlan",
    "NativePackedMoERoutedDiagnostic",
    "NativeRoutedSigmoidFormulaSweep",
]
