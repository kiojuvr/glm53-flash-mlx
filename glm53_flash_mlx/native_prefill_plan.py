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
    compact_qk_exact: bool
    compact_precise_softmax_exact: bool
    ordinary_compact_sdpa_allowed: bool
    compact_av_allowed: bool
    requires_virtual_full_kv_av_reduction_topology: bool
    invariants: tuple[str, ...]
    production_admission_changed: bool

    def descriptor(self) -> dict[str, object]:
        return asdict(self)


def build_native_sparse_prefill_attention_plan(
    reordering_probe: dict[str, object],
    microtile_probe: dict[str, object],
    reduction_localization: dict[str, object],
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
    if reduction_localization.get("schema") != (
        "glm53-sparse-prefill-attention-reduction-localization-v1"
    ) or not reduction_localization.get("accepted"):
        raise NativePrefillPlanError("attention reduction localization is unavailable")
    localized = reduction_localization.get("checks", {})
    if not (
        localized.get("explicit_fallback_matches_direct_fast_sdpa")
        and localized.get("selected_qk_scores_byte_exact")
        and localized.get("selected_precise_softmax_probabilities_byte_exact")
        and localized.get("compact_av_accumulation_is_first_and_only_barrier")
        and reduction_localization.get("first_differing_stage")
        == "attention_probability_times_value"
    ):
        raise NativePrefillPlanError("prefill attention barrier is not isolated to AV")

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
        compact_qk_exact=True,
        compact_precise_softmax_exact=True,
        ordinary_compact_sdpa_allowed=False,
        compact_av_allowed=False,
        requires_virtual_full_kv_av_reduction_topology=True,
        invariants=(
            "Q4 score/top-k/expansion executes inside the composed native region",
            "selected valid token indices are sorted in physical token order",
            "attention consumes one query row at a time from reusable scratch",
            "selected latent is projected with the Direct BF16 K/V dot order",
            "compact QK and precise softmax are byte exact at the selected width",
            "value accumulation follows the Direct full-Kv split-K topology",
            "unselected physical positions enter AV reduction as virtual zero lanes",
            "ordinary compact AV is forbidden by measured 32K byte divergence",
            "no full-context projected K/V or Q256-by-Kv mask is materialized",
            "only final attention output may cross the native execution boundary",
        ),
        production_admission_changed=False,
    )


@dataclass(frozen=True)
class NativeQ256UnionAttentionPlan:
    abi: str
    query_rows: int
    selected_width: int
    maximum_selected_edges: int
    projection_tile_rows: int
    maximum_union_tiles_512k: int
    qk_score_bytes: int
    qk_probability_bytes: int
    key_phase_projected_tile_bytes: int
    value_phase_projected_tile_bytes: int
    selected_value_edge_bytes: int
    maximum_phase_arena_bytes: int
    available_native_arena_bytes: int
    full_512k_projected_key_value_bytes_forbidden: int
    per_query_selected_key_value_bytes_forbidden: int
    measured_320k_union_rows_all_dsa_layers: int
    measured_320k_union_vs_full_history: float
    measured_320k_union_build_ms: float
    measured_320k_indirect_gather_ms: float
    execution_phases: tuple[str, ...]
    invariants: tuple[str, ...]
    production_admission_changed: bool

    def descriptor(self) -> dict[str, object]:
        return asdict(self)


def build_native_q256_union_attention_plan(
    reuse_probe: dict[str, object],
    union_probe: dict[str, object],
    indirect_probe: dict[str, object],
    native_prefill_plan: dict[str, object],
) -> NativeQ256UnionAttentionPlan:
    """Fix the bounded two-pass Q256 union attention architecture."""

    expected = (
        (reuse_probe, "glm53-selected-kv-projection-reuse-frontier-v1"),
        (union_probe, "glm53-device-resident-q256-selected-union-v1"),
        (indirect_probe, "glm53-indirect-q256-selected-latent-v1"),
        (native_prefill_plan, "glm53-native-prefill-execution-plan-v1"),
    )
    for artifact, schema in expected:
        if artifact.get("schema") != schema or not artifact.get("accepted"):
            raise NativePrefillPlanError(
                f"accepted source evidence is required for {schema}"
            )
    long_reuse = reuse_probe["contexts"][str(320 << 10)]
    if long_reuse["aggregate_minimum_strategy"] != "union_q256":
        raise NativePrefillPlanError("Q256 union is not the measured minimum")
    if indirect_probe["decision"] != (
        "advance_indirect_union_to_streaming_selected_kv_projection"
    ):
        raise NativePrefillPlanError("indirect union substrate is not qualified")

    query_rows = DESIGN_PREFILL_QUERY_ROWS
    selected_width = 2_051
    heads = 64
    latent_dim = 512
    value_dim = 128
    bf16_bytes = 2
    float_bytes = 4
    tile_rows = 65_536
    edges = query_rows * selected_width
    scores = edges * heads * bf16_bytes
    probabilities = scores
    key_tile = heads * tile_rows * latent_dim * bf16_bytes
    value_tile = heads * tile_rows * value_dim * bf16_bytes
    selected_values = edges * heads * value_dim * bf16_bytes
    tile_latent = tile_rows * latent_dim * bf16_bytes
    # The K projection workspace and selected-V storage occupy disjoint phases
    # and must alias the same native arena.  QK scores survive into the V/AV
    # phase; probabilities and an FP32 output accumulator are bounded extras.
    k_phase = key_tile + tile_latent + scores
    v_phase = (
        selected_values
        + value_tile
        + tile_latent
        + scores
        + probabilities
        + query_rows * heads * value_dim * float_bytes
    )
    maximum_phase = max(k_phase, v_phase)
    available = int(
        native_prefill_plan["plan"]["planned_native_arena_budget_bytes"]
    )
    full_key = heads * DESIGN_TOTAL_CONTEXT_TOKENS * latent_dim * bf16_bytes
    full_value = heads * DESIGN_TOTAL_CONTEXT_TOKENS * value_dim * bf16_bytes
    per_query_key = edges * heads * latent_dim * bf16_bytes
    per_query_value = selected_values
    return NativeQ256UnionAttentionPlan(
        abi="glm53-native-q256-union-attention-plan-v1",
        query_rows=query_rows,
        selected_width=selected_width,
        maximum_selected_edges=edges,
        projection_tile_rows=tile_rows,
        maximum_union_tiles_512k=(
            DESIGN_TOTAL_CONTEXT_TOKENS + tile_rows - 1
        ) // tile_rows,
        qk_score_bytes=scores,
        qk_probability_bytes=probabilities,
        key_phase_projected_tile_bytes=key_tile,
        value_phase_projected_tile_bytes=value_tile,
        selected_value_edge_bytes=selected_values,
        maximum_phase_arena_bytes=maximum_phase,
        available_native_arena_bytes=available,
        full_512k_projected_key_value_bytes_forbidden=full_key + full_value,
        per_query_selected_key_value_bytes_forbidden=(
            per_query_key + per_query_value
        ),
        measured_320k_union_rows_all_dsa_layers=int(
            long_reuse["aggregate_minimum_projection_rows"]
        ),
        measured_320k_union_vs_full_history=float(
            long_reuse["aggregate_minimum_vs_full_history"]
        ),
        measured_320k_union_build_ms=float(
            union_probe["contexts"][str(320 << 10)]["median_wall_ms"]
        ),
        measured_320k_indirect_gather_ms=float(
            indirect_probe["contexts"][str(320 << 10)]["median_wall_ms"]
        ),
        execution_phases=(
            "score_select_and_build_device_q256_union",
            "key_tiles_project_once_and_fuse_into_query_edge_qk",
            "query_head_precise_softmax",
            "value_tiles_project_once_and_scatter_selected_value_edges",
            "direct_order_virtual_bk16_av",
        ),
        invariants=(
            "union count and tile dispatch remain device resident",
            "one union row is projected at most once per K pass and V pass",
            "K projection is consumed by query-edge QK before tile reuse",
            "fused projected-QK preserves the Direct BF16 projection boundary",
            "selected V edges remain in query-local physical token order",
            "virtual BK16 AV preserves the Direct full-Kv reduction topology",
            "K-phase projection storage aliases V-phase selected-value storage",
            "full-context projected K/V is never materialized",
            "per-query selected K is never materialized",
            "only the final Q256 attention output crosses the native boundary",
        ),
        production_admission_changed=False,
    )


@dataclass(frozen=True)
class NativeQ256SharedValuePassPlan:
    abi: str
    query_rows: int
    query_block_rows: int
    query_blocks: int
    selected_width: int
    physical_value_tile_rows: int
    maximum_physical_tiles_512k: int
    qk_score_bytes: int
    probability_bytes: int
    projected_value_tile_bytes: int
    fp32_attention_accumulator_bytes: int
    mapping_bytes: int
    rejected_query_local_selected_value_bytes: int
    planned_value_phase_bytes: int
    planned_key_phase_bytes: int
    maximum_phase_arena_bytes: int
    available_native_arena_bytes: int
    measured_q4_direct_ms: float
    measured_q4_candidate_ms: float
    measured_q256_direct_ms: float
    measured_q256_rejected_ms: float
    execution_phases: tuple[str, ...]
    invariants: tuple[str, ...]
    production_admission_changed: bool

    def descriptor(self) -> dict[str, object]:
        return asdict(self)


def build_native_q256_shared_value_pass_plan(
    q4_attention_probe: dict[str, object],
    q256_rejected_probe: dict[str, object],
    projected_qk_320k_probe: dict[str, object],
    native_prefill_plan: dict[str, object],
) -> NativeQ256SharedValuePassPlan:
    """Replace rejected per-query selected V with shared physical BK16 tiles."""

    if (
        q4_attention_probe.get("schema")
        != "glm53-q4-union-tiled-full-attention-v1"
        or not q4_attention_probe.get("accepted")
        or not all(q4_attention_probe.get("anchors", {}).values())
    ):
        raise NativePrefillPlanError("accepted exact Q4 value evidence is required")
    if (
        q256_rejected_probe.get("schema")
        != "glm53-q256-union-tiled-full-attention-v1"
        or q256_rejected_probe.get("accepted")
        or q256_rejected_probe.get("decision")
        != "stop_or_redesign_q256_union_tiled_value_pass"
    ):
        raise NativePrefillPlanError(
            "the measured Q256 query-local value rejection must remain explicit"
        )
    rejected_anchors = q256_rejected_probe.get("anchors", {})
    if (
        not rejected_anchors.get("selected_value_samples")
        or rejected_anchors.get("q256_attention_output")
    ):
        raise NativePrefillPlanError(
            "Q256 evidence must isolate the failure to AV topology"
        )
    if (
        projected_qk_320k_probe.get("schema")
        != "glm53-q256-projected-qk-320k-tiles-v1"
        or not projected_qk_320k_probe.get("accepted")
        or projected_qk_320k_probe.get("best_tile_rows") != 65_536
    ):
        raise NativePrefillPlanError("accepted 320K projected-QK evidence is required")
    if (
        native_prefill_plan.get("schema")
        != "glm53-native-prefill-execution-plan-v1"
        or not native_prefill_plan.get("accepted")
    ):
        raise NativePrefillPlanError("accepted native prefill plan is required")

    query_rows = DESIGN_PREFILL_QUERY_ROWS
    query_block_rows = 64
    selected_width = 2_051
    heads = 64
    latent_dim = 512
    value_dim = 128
    tile_rows = 65_536
    bf16_bytes = 2
    fp32_bytes = 4
    edges = query_rows * selected_width
    scores = edges * heads * bf16_bytes
    probabilities = scores
    value_tile = heads * tile_rows * value_dim * bf16_bytes
    accumulator = query_rows * heads * value_dim * fp32_bytes
    mapping = edges * (4 + 4 + 1)
    rejected_selected_values = edges * heads * value_dim * bf16_bytes
    tile_latent = tile_rows * latent_dim * bf16_bytes
    key_tile = heads * tile_rows * latent_dim * bf16_bytes
    key_phase = key_tile + tile_latent + scores + probabilities
    value_phase = (
        value_tile + tile_latent + scores + probabilities + accumulator + mapping
    )
    maximum_phase = max(key_phase, value_phase)
    available = int(
        native_prefill_plan["plan"]["planned_native_arena_budget_bytes"]
    )
    if maximum_phase > available:
        raise NativePrefillPlanError("shared physical value plan exceeds arena")

    return NativeQ256SharedValuePassPlan(
        abi="glm53-native-q256-shared-physical-bk16-value-pass-v1",
        query_rows=query_rows,
        query_block_rows=query_block_rows,
        query_blocks=query_rows // query_block_rows,
        selected_width=selected_width,
        physical_value_tile_rows=tile_rows,
        maximum_physical_tiles_512k=DESIGN_TOTAL_CONTEXT_TOKENS // tile_rows,
        qk_score_bytes=scores,
        probability_bytes=probabilities,
        projected_value_tile_bytes=value_tile,
        fp32_attention_accumulator_bytes=accumulator,
        mapping_bytes=mapping,
        rejected_query_local_selected_value_bytes=rejected_selected_values,
        planned_value_phase_bytes=value_phase,
        planned_key_phase_bytes=key_phase,
        maximum_phase_arena_bytes=maximum_phase,
        available_native_arena_bytes=available,
        measured_q4_direct_ms=float(
            q4_attention_probe["direct_timing"]["median_wall_ms"]
        ),
        measured_q4_candidate_ms=float(
            q4_attention_probe["native_timing"]["median_wall_ms"]
        ),
        measured_q256_direct_ms=float(
            q256_rejected_probe["direct_timing"]["median_wall_ms"]
        ),
        measured_q256_rejected_ms=float(
            q256_rejected_probe["native_timing"]["median_wall_ms"]
        ),
        execution_phases=(
            "consume_owned_q256_precise_probabilities",
            "iterate_physical_k_tiles_in_ascending_order",
            "project_one_shared_physical_bk16_value_tile",
            "execute_four_bm64_query_blocks_against_shared_value_tile",
            "carry_fp32_attention_accumulator_to_next_physical_tile",
            "round_bfloat16_once_after_final_physical_tile",
        ),
        invariants=(
            "query-local selected V materialization is forbidden",
            "one projected physical value tile is shared by all 256 queries",
            "queries execute as four Direct-compatible BM64 blocks",
            "physical BK16 traversal order is strictly increasing across tiles",
            "unselected token lanes contribute exact zero probabilities",
            "FP32 accumulators are stored and reloaded without arithmetic",
            "BF16 output rounding occurs only after the final physical tile",
            "K-phase and V-phase arenas alias and never coexist at peak",
            "dynamic allocation and host-visible tile counts remain zero",
            "only final Q256 attention output crosses the native boundary",
        ),
        production_admission_changed=False,
    )
