"""Measured architecture contract for the native GLM-5.3 prefill engine.

This module turns the whole-model prefill profile into an execution-plan
contract.  It does not promote 512K admission or install a runtime backend.
The measured 340 GB profiling failure remains visible: native execution must
replace the source-plus-clone diagnostic topology with one live cache and a
bounded, reusable arena.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .context_capacity import (
    DESIGN_PREFILL_QUERY_ROWS,
    DESIGN_TOTAL_CONTEXT_TOKENS,
    plan_native_context_capacity,
)

NATIVE_PREFILL_PLAN_ABI = (
    "glm53-native-prefill-plan-v1"
    "-tile256-persistent-layer-dataflow"
    "-dsa-score-select-attention"
    "-exact-grouped-moe"
)
PROFILE_CONTEXTS = (32 << 10, 128 << 10, 320 << 10)
TARGET_PREFILL_TOKENS_PER_SECOND = (100, 200, 300)
PRODUCTION_PEAK_BUDGET_BYTES = 340_000_000_000


class NativePrefillPlanError(ValueError):
    """Raised when a profile cannot support a native prefill plan."""


@dataclass(frozen=True)
class PrefillThroughputTarget:
    tokens_per_second: int
    chunk_budget_ms: float
    required_speedup_from_320k: float


@dataclass(frozen=True)
class NativePrefillExecutionPlan:
    abi: str
    tile_rows: int
    measured_320k_wall_ms: float
    measured_320k_tokens_per_second: float
    dsa_diagnostic_share_320k: float
    routed_moe_diagnostic_share_320k: float
    combined_structural_share_320k: float
    dsa_32k_to_320k_scaling: float
    routed_moe_32k_to_320k_scaling: float
    measured_profile_peak_bytes: int
    measured_profile_peak_over_budget_bytes: int
    model_parameter_bytes: int
    planned_512k_persistent_state_bytes: int
    planned_native_arena_budget_bytes: int
    dsa_logits_workspace_bytes: int
    selected_token_width: int
    throughput_targets: tuple[PrefillThroughputTarget, ...]
    execution_regions: tuple[str, ...]
    invariants: tuple[str, ...]
    production_admission_changed: bool

    def descriptor(self) -> dict[str, object]:
        return asdict(self)


def _context(profile: dict[str, object], context: int) -> dict[str, object]:
    try:
        return profile["contexts"][str(context)]
    except (KeyError, TypeError) as error:
        raise NativePrefillPlanError(
            f"profile is missing context {context}"
        ) from error


def _stage_ms(row: dict[str, object], name: str) -> float:
    return float(
        row["instrumented"]["attribution"]["stages"][name][
            "synchronized_wall_sum_ms"
        ]
    )


def _validate_profile(profile: dict[str, object]) -> None:
    if profile.get("schema") != "glm53-native-prefill-critical-path-v1":
        raise NativePrefillPlanError("unexpected native prefill profile schema")
    acceptance = profile.get("acceptance", {})
    allowed_negative = {"process_peak_at_most_340gb"}
    failed = {name for name, passed in acceptance.items() if not passed}
    if failed - allowed_negative:
        raise NativePrefillPlanError(
            f"profile has non-resource failures: {sorted(failed)}"
        )
    if failed != allowed_negative:
        raise NativePrefillPlanError(
            "the measured source-plus-clone peak failure must remain explicit"
        )
    if set(map(int, profile.get("contexts", {}))) != set(PROFILE_CONTEXTS):
        raise NativePrefillPlanError("profile context set is incomplete")
    for context in PROFILE_CONTEXTS:
        row = _context(profile, context)
        if not row["uninstrumented"]["repeat_logits_exact"]:
            raise NativePrefillPlanError(f"logits repeat failed at {context}")
        if not row["uninstrumented"]["repeat_state_exact"]:
            raise NativePrefillPlanError(f"state repeat failed at {context}")
        if not all(row["exactness"].values()):
            raise NativePrefillPlanError(
                f"instrumentation changed outputs at {context}"
            )


def build_native_prefill_execution_plan(
    profile: dict[str, object],
) -> NativePrefillExecutionPlan:
    """Derive a whole-dataflow native plan from the measured profile."""

    _validate_profile(profile)
    short = _context(profile, 32 << 10)
    long = _context(profile, 320 << 10)
    stage_total = float(
        long["instrumented"]["attribution"]["synchronized_stage_sum_ms"]
    )
    if stage_total <= 0:
        raise NativePrefillPlanError("diagnostic stage total must be positive")
    dsa_short = _stage_ms(short, "dsa_attention")
    dsa_long = _stage_ms(long, "dsa_attention")
    moe_short = _stage_ms(short, "routed_moe")
    moe_long = _stage_ms(long, "routed_moe")
    wall = float(long["uninstrumented"]["median_wall_ms"])
    if min(dsa_short, dsa_long, moe_short, moe_long, wall) <= 0:
        raise NativePrefillPlanError("measured stage and wall times must be positive")

    capacity = plan_native_context_capacity()
    model_bytes = int(profile["model_storage_inventory"]["categorized_parameter_bytes"])
    persistent_state = (
        capacity.persistent_nope_latent_bytes_all_dsa_layers
        + capacity.persistent_indexpool_bytes_all_dsa_layers
    )
    arena_budget = PRODUCTION_PEAK_BUDGET_BYTES - model_bytes - persistent_state
    if arena_budget <= 0:
        raise NativePrefillPlanError("512K state leaves no native arena budget")

    targets = tuple(
        PrefillThroughputTarget(
            tokens_per_second=rate,
            chunk_budget_ms=DESIGN_PREFILL_QUERY_ROWS * 1_000.0 / rate,
            required_speedup_from_320k=(
                wall / (DESIGN_PREFILL_QUERY_ROWS * 1_000.0 / rate)
            ),
        )
        for rate in TARGET_PREFILL_TOKENS_PER_SECOND
    )
    peak = int(profile["process_peak_memory_bytes"])
    return NativePrefillExecutionPlan(
        abi=NATIVE_PREFILL_PLAN_ABI,
        tile_rows=DESIGN_PREFILL_QUERY_ROWS,
        measured_320k_wall_ms=wall,
        measured_320k_tokens_per_second=(
            DESIGN_PREFILL_QUERY_ROWS * 1_000.0 / wall
        ),
        dsa_diagnostic_share_320k=dsa_long / stage_total,
        routed_moe_diagnostic_share_320k=moe_long / stage_total,
        combined_structural_share_320k=(dsa_long + moe_long) / stage_total,
        dsa_32k_to_320k_scaling=dsa_long / dsa_short,
        routed_moe_32k_to_320k_scaling=moe_long / moe_short,
        measured_profile_peak_bytes=peak,
        measured_profile_peak_over_budget_bytes=max(
            0, peak - PRODUCTION_PEAK_BUDGET_BYTES
        ),
        model_parameter_bytes=model_bytes,
        planned_512k_persistent_state_bytes=persistent_state,
        planned_native_arena_budget_bytes=arena_budget,
        dsa_logits_workspace_bytes=capacity.fp32_logits_workspace_bytes,
        selected_token_width=capacity.selected_token_width,
        throughput_targets=targets,
        execution_regions=(
            "native_kda_layer",
            "native_dsa_score_select_gather_attention_layer",
            "native_exact_grouped_routed_and_shared_moe_layer",
            "native_dense_layer",
            "native_final_norm_lm_head",
        ),
        invariants=(
            "one live cache; diagnostic source and execution clone never coexist",
            "two stable hidden-state buffers ping-pong across all 45 layers",
            "one bounded scratch arena is reused across layers and tiles",
            "DSA score/top-k/expansion/gather/attention does not return intermediates to MLX",
            "MoE route/group/gate-up/SwiGLU/down/reduce/shared does not dispatch per expert",
            "packed weights are canonical and are not duplicated by prefill/decode views",
            "every Direct BF16/FP32 rounding boundary remains byte exact",
            "partial kernels cannot be promoted outside the composed native plan",
            "dynamic allocation and shape discovery are zero in steady chunks",
            "100 tok/s is a checkpoint; 200 and 300 tok/s remain measured targets",
        ),
        production_admission_changed=False,
    )

