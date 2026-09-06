"""Probe-only native execution bridge.

The binary extension is intentionally built separately from the production
package.  Import failure therefore cannot change the supported MLX runtime.
"""

from ._ext import NativeDSAScoreSelectionPlan, NativeIndexSelectionPlan

__all__ = ["NativeDSAScoreSelectionPlan", "NativeIndexSelectionPlan"]
