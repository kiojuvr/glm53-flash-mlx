"""Design contract for the next practical GLM-5.3 context capacity.

This module is planning-only.  It deliberately does not change the production
server defaults until native prefill and the larger native IndexPool decode
geometry have both passed their independent qualifications.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .cache_geometry import plan_nope_cache_capacity
from .dsa_workspace import (
    BF16_BYTES,
    DEFAULT_MAX_WORKSPACE_BYTES,
    account_dsa_indexer_memory,
    plan_dsa_indexer_workspace,
)
from .materialization import MATERIALIZATION_INTERVAL_TOKENS
from .native_execution import NATIVE_INDEXPOOL_QUALIFIED_MAX_PHYSICAL_POOL_ROWS

MODEL_NATIVE_CONTEXT_TOKENS = 1_048_576
DESIGN_TOTAL_CONTEXT_TOKENS = 512 << 10
DESIGN_CODING_AGENT_PROMPT_TOKENS = 320 << 10
DESIGN_MAX_GENERATION_TOKENS = 128 << 10
DESIGN_PREFILL_QUERY_ROWS = 256
DESIGN_PREFILL_PROFILE_CONTEXTS = (32 << 10, 128 << 10, 320 << 10)
DSA_LAYER_COUNT = 11
NOPE_LATENT_WIDTH = 512
NATIVE_CONTEXT_CAPACITY_CONTRACT = (
    "glm53-native-context-capacity-v1"
    "-total512k-prompt320k-generation128k"
    "-compact-kpool4-workspace64mib"
    "-no-production-promotion-before-native-qualification"
)


class NativeContextCapacityError(ValueError):
    """Raised before an invalid target capacity can enter qualification."""


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NativeContextCapacityError(f"{name} must be a positive Python int")
    return value


@dataclass(frozen=True)
class NativeContextCapacityPlan:
    contract: str
    total_context_tokens: int
    coding_agent_prompt_tokens: int
    max_generation_tokens: int
    target_headroom_tokens: int
    max_prompt_at_max_generation: int
    materialization_interval_tokens: int
    generation_materialization_count: int
    logical_pool_rows: int
    physical_capacity_tokens: int
    physical_pool_rows: int
    cache_padding_tokens: int
    prefill_query_rows: int
    prefill_query_block_rows: int
    prefill_query_block_count: int
    fp32_logits_workspace_bytes: int
    max_workspace_bytes: int
    selected_token_width: int
    persistent_nope_latent_bytes_all_dsa_layers: int
    persistent_indexpool_bytes_all_dsa_layers: int
    currently_qualified_native_pool_rows: int
    native_pool_row_deficit: int
    native_decode_geometry_qualified: bool
    production_default_changed: bool

    def descriptor(self) -> dict[str, object]:
        return asdict(self)


def plan_native_context_capacity(
    *,
    total_context_tokens: int = DESIGN_TOTAL_CONTEXT_TOKENS,
    coding_agent_prompt_tokens: int = DESIGN_CODING_AGENT_PROMPT_TOKENS,
    max_generation_tokens: int = DESIGN_MAX_GENERATION_TOKENS,
    prefill_query_rows: int = DESIGN_PREFILL_QUERY_ROWS,
    max_workspace_bytes: int = DEFAULT_MAX_WORKSPACE_BYTES,
) -> NativeContextCapacityPlan:
    """Plan the 512K target without implying production admission."""

    total = _positive_integer("total_context_tokens", total_context_tokens)
    prompt = _positive_integer(
        "coding_agent_prompt_tokens", coding_agent_prompt_tokens
    )
    generation = _positive_integer("max_generation_tokens", max_generation_tokens)
    query_rows = _positive_integer("prefill_query_rows", prefill_query_rows)
    workspace_budget = _positive_integer(
        "max_workspace_bytes", max_workspace_bytes
    )
    if total > MODEL_NATIVE_CONTEXT_TOKENS:
        raise NativeContextCapacityError(
            "target context exceeds the model-native context capacity"
        )
    if generation > total:
        raise NativeContextCapacityError(
            "generation target exceeds the total context capacity"
        )
    if prompt + generation > total:
        raise NativeContextCapacityError(
            "prompt plus generation exceeds the total context capacity"
        )

    cache = plan_nope_cache_capacity(total)
    workspace = plan_dsa_indexer_workspace(
        context_tokens=total,
        num_query_rows=query_rows,
        max_workspace_bytes=workspace_budget,
    )
    accounting = account_dsa_indexer_memory(workspace, index_head_dim=128)
    persistent_indexpool_per_layer = (
        accounting.persistent_pool_keys_bytes
        + accounting.persistent_pool_indices_bytes
        + accounting.persistent_pool_validity_bytes
    )
    latent_bytes_per_layer = (
        cache.physical_capacity_tokens * NOPE_LATENT_WIDTH * BF16_BYTES
    )
    required_rows = cache.physical_pool_rows
    native_deficit = max(
        0, required_rows - NATIVE_INDEXPOOL_QUALIFIED_MAX_PHYSICAL_POOL_ROWS
    )
    return NativeContextCapacityPlan(
        contract=NATIVE_CONTEXT_CAPACITY_CONTRACT,
        total_context_tokens=total,
        coding_agent_prompt_tokens=prompt,
        max_generation_tokens=generation,
        target_headroom_tokens=total - prompt - generation,
        max_prompt_at_max_generation=total - generation,
        materialization_interval_tokens=MATERIALIZATION_INTERVAL_TOKENS,
        generation_materialization_count=(
            generation // MATERIALIZATION_INTERVAL_TOKENS
        ),
        logical_pool_rows=cache.logical_pool_rows,
        physical_capacity_tokens=cache.physical_capacity_tokens,
        physical_pool_rows=required_rows,
        cache_padding_tokens=cache.padding_tokens,
        prefill_query_rows=query_rows,
        prefill_query_block_rows=workspace.query_block_rows,
        prefill_query_block_count=workspace.query_block_count,
        fp32_logits_workspace_bytes=workspace.fp32_logits_workspace_bytes,
        max_workspace_bytes=workspace.max_workspace_bytes,
        selected_token_width=workspace.selected_token_width,
        persistent_nope_latent_bytes_all_dsa_layers=(
            latent_bytes_per_layer * DSA_LAYER_COUNT
        ),
        persistent_indexpool_bytes_all_dsa_layers=(
            persistent_indexpool_per_layer * DSA_LAYER_COUNT
        ),
        currently_qualified_native_pool_rows=(
            NATIVE_INDEXPOOL_QUALIFIED_MAX_PHYSICAL_POOL_ROWS
        ),
        native_pool_row_deficit=native_deficit,
        native_decode_geometry_qualified=(native_deficit == 0),
        production_default_changed=False,
    )


def native_prefill_profile_geometries() -> tuple[dict[str, int], ...]:
    """Return deterministic 32K/128K/320K planning rows for profiling."""

    rows = []
    for context_tokens in DESIGN_PREFILL_PROFILE_CONTEXTS:
        workspace = plan_dsa_indexer_workspace(
            context_tokens=context_tokens,
            num_query_rows=DESIGN_PREFILL_QUERY_ROWS,
        )
        rows.append(
            {
                "context_tokens": context_tokens,
                "pool_count": workspace.pool_count,
                "query_block_rows": workspace.query_block_rows,
                "query_block_count": workspace.query_block_count,
                "fp32_logits_workspace_bytes": (
                    workspace.fp32_logits_workspace_bytes
                ),
            }
        )
    return tuple(rows)
