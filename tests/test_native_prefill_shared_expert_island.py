import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "native_execution" / "native_prefill_shared_expert_plan.h"
SOURCE = ROOT / "native_execution" / "native_prefill_shared_expert_plan.cpp"
METAL = ROOT / "native_execution" / "native_indexer_plan.metal"
SCRIPT = ROOT / "scripts" / "probe_native_prefill_shared_expert_island.py"
ARTIFACT = ROOT / "bench-results" / (
    "m3ultra512-native-prefill-shared-expert-island-20260909.json"
)


def test_native_shared_expert_uses_one_fixed_bm8_arena():
    header = HEADER.read_text(); source = SOURCE.read_text(); metal = METAL.read_text()
    assert "kTileRows = 8" in header
    assert "returned_intermediate_tensor_bytes() const { return 0; }" in header
    assert "glm53_native_prefill_shared_bm8_gate_up_swiglu" in source
    assert "glm53_native_prefill_shared_bm8_down" in source
    assert "bfloat16_t(bfloat16_t(gv * sig) * uv)" in metal


def test_native_shared_probe_contract_and_artifact():
    ast.parse(SCRIPT.read_text())
    text = SCRIPT.read_text()
    assert "real_checkpoint_shared_output_byte_exact" in text
    assert "advance_shared_expert_into_full_native_prefill_moe" in text
    if ARTIFACT.exists():
        artifact = json.loads(ARTIFACT.read_text())
        assert artifact["complete"] is True
        assert artifact["accepted"] is True
        assert all(artifact["checks"].values())
