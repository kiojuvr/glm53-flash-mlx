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
DSA_INDEXER_HEADS = 32
DSA_INDEXER_HEAD_DIM = 128
DSA_STREAMING_QUERY_ROWS = 4
DSA_STREAMING_MATRIX_ROWS = DSA_STREAMING_QUERY_ROWS * DSA_INDEXER_HEADS
DSA_STREAMING_MICROTILES_PER_CHUNK = (
    DESIGN_PREFILL_QUERY_ROWS // DSA_STREAMING_QUERY_ROWS
)


class NativePrefillPlanError(ValueError):
    """Raised when a profile cannot support a native prefill plan."""


@dataclass(frozen=True)
class PrefillThroughputTarget:
    tokens_per_second: int
    chunk_budget_ms: float
    required_speedup_from_320k: float
    required_structural_region_speedup_if_other_fixed: float
    fixed_other_region_must_also_improve: bool


@dataclass(frozen=True)
class NativePrefillExecutionPlan:
    abi: str
    tile_rows: int
    measured_320k_wall_ms: float
    measured_320k_tokens_per_second: float
    measured_320k_ms_per_token: float
    projected_dsa_ms_per_token_320k: float
    projected_routed_moe_ms_per_token_320k: float
    projected_other_ms_per_token_320k: float
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

    wall_per_token = wall / DESIGN_PREFILL_QUERY_ROWS
    dsa_share = dsa_long / stage_total
    moe_share = moe_long / stage_total
    structural_per_token = wall_per_token * (dsa_share + moe_share)
    other_per_token = wall_per_token - structural_per_token
    targets = []
    for rate in TARGET_PREFILL_TOKENS_PER_SECOND:
        chunk_budget = DESIGN_PREFILL_QUERY_ROWS * 1_000.0 / rate
        token_budget = 1_000.0 / rate
        remaining_structural_budget = token_budget - other_per_token
        structural_speedup = (
            structural_per_token / remaining_structural_budget
            if remaining_structural_budget > 0
            else float("inf")
        )
        targets.append(
            PrefillThroughputTarget(
                tokens_per_second=rate,
                chunk_budget_ms=chunk_budget,
                required_speedup_from_320k=wall / chunk_budget,
                required_structural_region_speedup_if_other_fixed=(
                    structural_speedup
                ),
                fixed_other_region_must_also_improve=(
                    structural_speedup > 20.0
                ),
            )
        )
    peak = int(profile["process_peak_memory_bytes"])
    return NativePrefillExecutionPlan(
        abi=NATIVE_PREFILL_PLAN_ABI,
        tile_rows=DESIGN_PREFILL_QUERY_ROWS,
        measured_320k_wall_ms=wall,
        measured_320k_tokens_per_second=(
            DESIGN_PREFILL_QUERY_ROWS * 1_000.0 / wall
        ),
        measured_320k_ms_per_token=wall_per_token,
        projected_dsa_ms_per_token_320k=wall_per_token * dsa_share,
        projected_routed_moe_ms_per_token_320k=wall_per_token * moe_share,
        projected_other_ms_per_token_320k=other_per_token,
        dsa_diagnostic_share_320k=dsa_share,
        routed_moe_diagnostic_share_320k=moe_share,
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
        throughput_targets=tuple(targets),
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


@dataclass(frozen=True)
class NativeDSAPrefillStreamingGeometry:
    query_rows_per_microtile: int
    indexer_heads: int
    matrix_rows: int
    microtiles_per_256_row_chunk: int
    physical_pool_rows: int
    bf16_head_score_scratch_bytes: int
    bf16_index_score_scratch_bytes: int
    selected_pool_scratch_bytes: int
    total_score_selection_scratch_bytes: int
    max_score_selection_scratch_bytes: int
    steel_bm: int
    steel_matrix_rows_aligned: bool
    full_q256_head_score_bytes_avoided: int

    def descriptor(self) -> dict[str, object]:
        return asdict(self)


def plan_native_dsa_prefill_streaming_geometry(
    *, physical_pool_rows: int = 131_072
) -> NativeDSAPrefillStreamingGeometry:
    """Plan exact BF16 score materialization in bounded Q=4 microtiles."""

    if (
        isinstance(physical_pool_rows, bool)
        or not isinstance(physical_pool_rows, int)
        or physical_pool_rows < 512
        or physical_pool_rows > 131_072
        or physical_pool_rows % 64
    ):
        raise NativePrefillPlanError(
            "physical_pool_rows must be 64-aligned in [512, 131072]"
        )
    bf16_bytes = 2
    uint32_bytes = 4
    head = (
        DSA_STREAMING_MATRIX_ROWS * physical_pool_rows * bf16_bytes
    )
    index = (
        DSA_STREAMING_QUERY_ROWS * physical_pool_rows * bf16_bytes
    )
    selected = DSA_STREAMING_QUERY_ROWS * 512 * uint32_bytes
    return NativeDSAPrefillStreamingGeometry(
        query_rows_per_microtile=DSA_STREAMING_QUERY_ROWS,
        indexer_heads=DSA_INDEXER_HEADS,
        matrix_rows=DSA_STREAMING_MATRIX_ROWS,
        microtiles_per_256_row_chunk=DSA_STREAMING_MICROTILES_PER_CHUNK,
        physical_pool_rows=physical_pool_rows,
        bf16_head_score_scratch_bytes=head,
        bf16_index_score_scratch_bytes=index,
        selected_pool_scratch_bytes=selected,
        total_score_selection_scratch_bytes=head + index + selected,
        max_score_selection_scratch_bytes=64 << 20,
        steel_bm=64,
        steel_matrix_rows_aligned=(DSA_STREAMING_MATRIX_ROWS % 64 == 0),
        full_q256_head_score_bytes_avoided=(
            DESIGN_PREFILL_QUERY_ROWS
            * DSA_INDEXER_HEADS
            * physical_pool_rows
            * bf16_bytes
        ),
    )


@dataclass(frozen=True)
class NativeSparsePrefillAttentionPlan:
    logical_context_tokens: int
    score_microtile_rows: int
    attention_microtile_rows: int
    selected_width: int
    selected_latent_bytes: int
    projected_key_bytes: int
    projected_value_bytes: int
    attention_score_bytes: int
    attention_accumulator_bytes: int
    score_selection_scratch_bytes: int
    total_scratch_bytes: int
    max_scratch_bytes: int
    full_projected_key_value_bytes_avoided: int
    full_sparse_mask_bytes_avoided: int
    selected_projection_reordering_exact: bool
    ordinary_compact_sdpa_allowed: bool
    requires_virtual_full_kv_reduction_topology: bool
    invariants: tuple[str, ...]
    production_admission_changed: bool

    def descriptor(self) -> dict[str, object]:
        return asdict(self)


def build_native_sparse_prefill_attention_plan(
    reordering_probe: dict[str, object],
    microtile_probe: dict[str, object],
    *,
    logical_context_tokens: int = DESIGN_TOTAL_CONTEXT_TOKENS,
) -> NativeSparsePrefillAttentionPlan:
    """Build the exact sparse-attention plan from positive/negative evidence."""

    if logical_context_tokens != DESIGN_TOTAL_CONTEXT_TOKENS:
        raise NativePrefillPlanError("native sparse prefill plan is fixed at 512K")
    if reordering_probe.get("schema") != (
        "glm53-sparse-prefill-reordering-equivalence-v1"
    ):
        raise NativePrefillPlanError("unexpected sparse reordering probe schema")
    if not reordering_probe.get("complete"):
        raise NativePrefillPlanError("sparse reordering probe is incomplete")
    checks = reordering_probe.get("checks", {})
    if not checks.get("selected_k_projection_reordering_byte_exact") or not checks.get(
        "selected_v_projection_reordering_byte_exact"
    ):
        raise NativePrefillPlanError("selected K/V projection reordering is not exact")
    if checks.get("sorted_compact_attention_matches_dense_sparse_mask"):
        raise NativePrefillPlanError(
            "the measured compact-attention numerical barrier must remain explicit"
        )
    long = reordering_probe.get("contexts", {}).get(str(32 << 10), {})
    compact_diff = long.get("diagnostics", {}).get("compact_attention", {})
    if int(compact_diff.get("different_elements", 0)) <= 0:
        raise NativePrefillPlanError("missing measured compact-attention divergence")
    if microtile_probe.get("schema") != (
        "glm53-native-dsa-prefill-streaming-microtile-v1"
    ) or not microtile_probe.get("accepted"):
        raise NativePrefillPlanError("exact Q4 score-selection microtile is unavailable")

    geometry = plan_native_dsa_prefill_streaming_geometry()
    bf16_bytes = 2
    selected_latent = 2_051 * 512 * bf16_bytes
    projected_key = 64 * 2_051 * 512 * bf16_bytes
    projected_value = 64 * 2_051 * 128 * bf16_bytes
    attention_score = 64 * 2_051 * bf16_bytes
    attention_accumulator = 4 * 64 * 128 * 4
    total = (
        geometry.total_score_selection_scratch_bytes
        + selected_latent
        + projected_key
        + projected_value
        + attention_score
        + attention_accumulator
    )
    full_key = 64 * logical_context_tokens * 512 * bf16_bytes
    full_value = 64 * logical_context_tokens * 128 * bf16_bytes
    full_mask = DESIGN_PREFILL_QUERY_ROWS * logical_context_tokens
    return NativeSparsePrefillAttentionPlan(
        logical_context_tokens=logical_context_tokens,
        score_microtile_rows=DSA_STREAMING_QUERY_ROWS,
        attention_microtile_rows=1,
        selected_width=2_051,
        selected_latent_bytes=selected_latent,
        projected_key_bytes=projected_key,
        projected_value_bytes=projected_value,
        attention_score_bytes=attention_score,
        attention_accumulator_bytes=attention_accumulator,
        score_selection_scratch_bytes=(
            geometry.total_score_selection_scratch_bytes
        ),
        total_scratch_bytes=total,
        max_scratch_bytes=256 << 20,
        full_projected_key_value_bytes_avoided=full_key + full_value,
        full_sparse_mask_bytes_avoided=full_mask,
        selected_projection_reordering_exact=True,
        ordinary_compact_sdpa_allowed=False,
        requires_virtual_full_kv_reduction_topology=True,
        invariants=(
            "Q4 score/top-k/expansion executes inside the composed native region",
            "selected valid token indices are sorted in physical token order",
            "attention consumes one query row at a time from reusable scratch",
            "selected latent is projected with the Direct BF16 K/V dot order",
            "unselected logical positions behave as masked finite-min scores",
            "softmax max/sum follows the Direct logical full-Kv reduction tree",
            "value accumulation follows physical token order with virtual zero lanes",
            "ordinary compact SDPA is forbidden by measured 32K byte divergence",
            "no full-context projected K/V or Q256-by-Kv mask is materialized",
            "only final attention output may cross the native execution boundary",
        ),
        production_admission_changed=False,
    )
