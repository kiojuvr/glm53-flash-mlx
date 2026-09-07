import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_native_routed_sigmoid_formula_sweep.py"
METAL = ROOT / "native_execution" / "native_indexer_plan.metal"
HEADER = ROOT / "native_execution" / "native_packed_moe_plan.h"
PACKAGE = ROOT / "native_execution" / "glm53_native_execution" / "__init__.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-routed-sigmoid-formula-sweep-20260907.json"
)


def test_formula_sweep_covers_dtype_promotion_and_exp_intrinsic_independently():
    source = PROBE.read_text()
    ast.parse(source)
    for formula in (
        "standard_bf16",
        "precise_bf16",
        "standard_f32",
        "precise_f32",
        "fast_bf16",
        "fast_f32",
    ):
        assert f'"{formula}"' in source
        assert formula in METAL.read_text()


def test_formula_sweep_uses_the_complete_production_bf16_clamp_domain():
    source = PROBE.read_text()
    assert "np.arange(1 << 16, dtype=np.uint16)" in source
    assert "np.isfinite(values)" in source
    assert "np.abs(values) <= np.float32(10.0)" in source
    assert "mx.array(selected_bits).view(mx.bfloat16)" in source
    assert "254 BF16 subnormal encodings" in source
    assert "BF16 formula domain did not preserve source bits" in source
    assert '"unit_up_hidden_equals_silu"' in source


def test_aot_formula_sweep_is_native_bound_and_probe_only():
    source = PROBE.read_text()
    assert "class NativeRoutedSigmoidFormulaSweep" in HEADER.read_text()
    assert "NativeRoutedSigmoidFormulaSweep" in PACKAGE.read_text()
    assert "glm53_native_routed_sigmoid_formula_sweep" in METAL.read_text()
    assert '"probe_only": True' in source
    for component in (
        "runtime",
        "server",
        "apc",
        "cache_abi",
        "production_kernel_abi",
    ):
        assert f'"{component}": False' in source


def test_formula_sweep_selects_only_fast_bf16_over_the_complete_domain():
    if not ARTIFACT.exists():
        return
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert artifact["decision"] == "repair_native_routed_sigmoid_with_fast_bf16"
    assert artifact["domain"]["elements"] == 33346
    assert artifact["reference"]["unit_up_hidden_equals_silu"]
    assert artifact["exact_formulas"] == ["fast_bf16"]
    assert artifact["results"]["fast_bf16"]["different_elements"] == 0
    assert artifact["results"]["standard_bf16"]["different_elements"] == 1
    assert artifact["results"]["precise_bf16"]["different_elements"] == 1
    assert artifact["results"]["standard_f32"]["different_elements"] == 990
    assert artifact["results"]["precise_f32"]["different_elements"] == 990
