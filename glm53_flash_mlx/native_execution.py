"""Host contract for the probe-only native prefill/decode execution engine.

The first feasibility tier owns only the Indexer selection/expansion island.
It deliberately keeps score production outside the island: the purpose of
this tier is to prove that a native C++ submission bridge can consume an
already-scheduled MLX buffer and encode a fixed Metal topology without making
another MLX graph or allocating per invocation.  Later tiers may widen the
same ABI to projection, score production, KDA/DSA, and MoE.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from .cache_geometry import plan_nope_cache_capacity

NATIVE_EXECUTION_ENGINE_ABI = (
    "glm53-native-execution-engine-v1"
    "-shared-prefill-decode-plan"
    "-fixed-address-arena"
    "-direct-metal-submission"
)
NATIVE_DSA_SCORE_ISLAND_ABI = (
    "glm53-native-dsa-score-island-v2"
    "-bf16-eager-rounding"
    "-coalesced-pool32-head32"
    "-exact-topk-expand"
)
NATIVE_DSA_SPARSE_ATTENTION_ISLAND_ABI = (
    "glm53-native-dsa-sparse-attention-island-v1"
    "-decode-width2051-d512"
    "-mlx0322-steel-precise-softmax"
    "-fixed-arena"
)
NATIVE_INDEXPOOL_UPDATE_ISLAND_ABI = (
    "glm53-native-indexpool-update-island-v1"
    "-decode-raw19-kpool4"
    "-exact-bf16-softmax-reduction"
    "-tier1-score-selection"
)
NATIVE_INDEXPOOL_QUALIFIED_MAX_PHYSICAL_POOL_ROWS = 65_600


class NativeExecutionContractError(ValueError):
    """Raised before an incompatible native execution plan can be used."""


class NativeExecutionMode(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class NativeBufferRole(str, Enum):
    INPUT = "input"
    IMMUTABLE_WEIGHT = "immutable-weight"
    MUTABLE_STATE = "mutable-state"
    EXTERNAL_MUTABLE_STATE = "external-mutable-state"
    SCRATCH = "scratch"
    OUTPUT = "output"


@dataclass(frozen=True)
class NativeBufferSpec:
    name: str
    role: NativeBufferRole
    dtype: str
    shape: tuple[int, ...]
    row_major: bool
    owned_by_plan: bool
    stable_address: bool
    returned_to_mlx: bool

    @property
    def elements(self) -> int:
        result = 1
        for extent in self.shape:
            result *= extent
        return result

    def descriptor(self) -> dict[str, object]:
        value = asdict(self)
        value["role"] = self.role.value
        value["elements"] = self.elements
        return value


@dataclass(frozen=True)
class NativeStageSpec:
    name: str
    pipeline: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]

    def descriptor(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NativeExecutionPlan:
    mode: NativeExecutionMode
    query_rows: int
    logical_capacity_tokens: int
    physical_pool_rows: int
    selected_pool_rows: int
    selected_token_width: int
    buffers: tuple[NativeBufferSpec, ...]
    stages: tuple[NativeStageSpec, ...]
    dynamic_allocations_per_execute: int = 0
    python_graph_nodes_per_execute: int = 0
    shape_discovery_per_execute: int = 0
    host_synchronizations_per_execute: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.query_rows, bool) or self.query_rows <= 0:
            raise NativeExecutionContractError("query_rows must be positive")
        if self.mode is NativeExecutionMode.DECODE and self.query_rows != 1:
            raise NativeExecutionContractError("decode plan requires one query row")
        if self.logical_capacity_tokens <= 0:
            raise NativeExecutionContractError(
                "logical_capacity_tokens must be positive"
            )
        if self.selected_pool_rows != 512 or self.selected_token_width != 2051:
            raise NativeExecutionContractError(
                "GLM-5.3 Indexer plan requires 512 pools and width 2051"
            )
        names = [buffer.name for buffer in self.buffers]
        if len(names) != len(set(names)):
            raise NativeExecutionContractError("native buffer names must be unique")
        known = set(names)
        for stage in self.stages:
            unknown = (set(stage.reads) | set(stage.writes)) - known
            if unknown:
                raise NativeExecutionContractError(
                    f"stage {stage.name} references unknown buffers: {sorted(unknown)}"
                )
        for field in (
            "dynamic_allocations_per_execute",
            "python_graph_nodes_per_execute",
            "shape_discovery_per_execute",
            "host_synchronizations_per_execute",
        ):
            if getattr(self, field) != 0:
                raise NativeExecutionContractError(f"{field} must be zero")
        for buffer in self.buffers:
            if not buffer.row_major:
                raise NativeExecutionContractError(
                    f"native buffer {buffer.name} must be row-major"
                )
            if buffer.role in {
                NativeBufferRole.SCRATCH,
                NativeBufferRole.MUTABLE_STATE,
                NativeBufferRole.OUTPUT,
            } and (not buffer.owned_by_plan or not buffer.stable_address):
                raise NativeExecutionContractError(
                    f"mutable native buffer {buffer.name} must be owned and stable"
                )
            if (
                buffer.role is NativeBufferRole.EXTERNAL_MUTABLE_STATE
                and (buffer.owned_by_plan or not buffer.stable_address)
            ):
                raise NativeExecutionContractError(
                    f"external mutable buffer {buffer.name} must be cache-owned and stable"
                )
            if buffer.role is NativeBufferRole.SCRATCH and buffer.returned_to_mlx:
                raise NativeExecutionContractError(
                    f"scratch buffer {buffer.name} cannot escape the native island"
                )

    @property
    def fixed_command_topology(self) -> tuple[str, ...]:
        return tuple(stage.pipeline for stage in self.stages)

    def descriptor(self) -> dict[str, object]:
        return {
            "abi": NATIVE_EXECUTION_ENGINE_ABI,
            "mode": self.mode.value,
            "query_rows": self.query_rows,
            "logical_capacity_tokens": self.logical_capacity_tokens,
            "physical_pool_rows": self.physical_pool_rows,
            "selected_pool_rows": self.selected_pool_rows,
            "selected_token_width": self.selected_token_width,
            "buffers": [buffer.descriptor() for buffer in self.buffers],
            "stages": [stage.descriptor() for stage in self.stages],
            "fixed_command_topology": list(self.fixed_command_topology),
            "dynamic_allocations_per_execute": self.dynamic_allocations_per_execute,
            "python_graph_nodes_per_execute": self.python_graph_nodes_per_execute,
            "shape_discovery_per_execute": self.shape_discovery_per_execute,
            "host_synchronizations_per_execute": (
                self.host_synchronizations_per_execute
            ),
            "coverage": "tier0-indexer-selection-expansion-submission-bridge",
            "score_producer_native": False,
            "kda_dsa_native": False,
            "moe_native": False,
        }


def plan_native_indexer_island(
    mode: NativeExecutionMode | str,
    *,
    query_rows: int,
    logical_capacity_tokens: int,
    score_dtype: str = "bfloat16",
) -> NativeExecutionPlan:
    """Build one fixed-address selection/expansion plan for either mode."""

    try:
        mode = NativeExecutionMode(mode)
    except ValueError as error:
        raise NativeExecutionContractError(f"unknown native mode: {mode}") from error
    if score_dtype not in {"bfloat16", "float32"}:
        raise NativeExecutionContractError(
            "native Indexer scores must be bfloat16 or float32"
        )
    capacity = plan_nope_cache_capacity(logical_capacity_tokens)
    rows = int(query_rows)
    if rows <= 0:
        raise NativeExecutionContractError("query_rows must be positive")
    pool_rows = capacity.physical_pool_rows
    buffers = (
        NativeBufferSpec(
            "scores",
            NativeBufferRole.INPUT,
            score_dtype,
            (1, rows, pool_rows),
            True,
            False,
            False,
            False,
        ),
        NativeBufferSpec(
            "pool_indices",
            NativeBufferRole.INPUT,
            "int64",
            (1, pool_rows, 4),
            True,
            False,
            False,
            False,
        ),
        NativeBufferSpec(
            "pool_valid",
            NativeBufferRole.INPUT,
            "bool",
            (1, pool_rows),
            True,
            False,
            False,
            False,
        ),
        NativeBufferSpec(
            "selected_pool_scratch",
            NativeBufferRole.SCRATCH,
            "uint32",
            (1, rows, 512),
            True,
            True,
            True,
            False,
        ),
        NativeBufferSpec(
            "selected_token_indices",
            NativeBufferRole.OUTPUT,
            "int32",
            (1, rows, 2051),
            True,
            True,
            True,
            True,
        ),
        NativeBufferSpec(
            "selected_token_valid",
            NativeBufferRole.OUTPUT,
            "bool",
            (1, rows, 2051),
            True,
            True,
            True,
            True,
        ),
    )
    stages = (
        NativeStageSpec(
            "exact-partial-topk",
            "glm53_native_exact_partial_topk_512",
            ("scores",),
            ("selected_pool_scratch",),
        ),
        NativeStageSpec(
            "pool-token-expansion",
            "glm53_native_expand_selected_pools",
            ("selected_pool_scratch", "pool_indices", "pool_valid"),
            ("selected_token_indices", "selected_token_valid"),
        ),
    )
    return NativeExecutionPlan(
        mode=mode,
        query_rows=rows,
        logical_capacity_tokens=logical_capacity_tokens,
        physical_pool_rows=pool_rows,
        selected_pool_rows=512,
        selected_token_width=2051,
        buffers=buffers,
        stages=stages,
    )


def plan_native_dsa_score_island(
    mode: NativeExecutionMode | str,
    *,
    query_rows: int,
    logical_capacity_tokens: int,
) -> NativeExecutionPlan:
    """Plan the Tier-1 pooled-score -> selected-token execution island.

    Query and mixture-weight projections remain explicit MLX inputs.  The
    potentially context-sized score tensor is plan-owned scratch and cannot
    escape back into the MLX graph.
    """

    try:
        mode = NativeExecutionMode(mode)
    except ValueError as error:
        raise NativeExecutionContractError(f"unknown native mode: {mode}") from error
    rows = int(query_rows)
    if rows <= 0:
        raise NativeExecutionContractError("query_rows must be positive")
    capacity = plan_nope_cache_capacity(logical_capacity_tokens)
    pool_rows = capacity.physical_pool_rows
    buffers = (
        NativeBufferSpec(
            "query",
            NativeBufferRole.INPUT,
            "bfloat16",
            (1, rows, 32, 128),
            True,
            False,
            False,
            False,
        ),
        NativeBufferSpec(
            "mixture_weights",
            NativeBufferRole.INPUT,
            "bfloat16",
            (1, rows, 32),
            True,
            False,
            False,
            False,
        ),
        NativeBufferSpec(
            "pool_keys",
            NativeBufferRole.INPUT,
            "bfloat16",
            (1, pool_rows, 128),
            True,
            False,
            False,
            False,
        ),
        NativeBufferSpec(
            "pool_indices",
            NativeBufferRole.INPUT,
            "int64",
            (1, pool_rows, 4),
            True,
            False,
            False,
            False,
        ),
        NativeBufferSpec(
            "pool_valid",
            NativeBufferRole.INPUT,
            "bool",
            (1, pool_rows),
            True,
            False,
            False,
            False,
        ),
        NativeBufferSpec(
            "head_scores",
            NativeBufferRole.SCRATCH,
            "bfloat16",
            (rows * 32, pool_rows),
            True,
            True,
            True,
            False,
        ),
        NativeBufferSpec(
            "index_scores",
            NativeBufferRole.SCRATCH,
            "bfloat16",
            (1, rows, pool_rows),
            True,
            True,
            True,
            False,
        ),
        NativeBufferSpec(
            "selected_pool_scratch",
            NativeBufferRole.SCRATCH,
            "uint32",
            (1, rows, 512),
            True,
            True,
            True,
            False,
        ),
        NativeBufferSpec(
            "selected_token_indices",
            NativeBufferRole.OUTPUT,
            "int32",
            (1, rows, 2051),
            True,
            True,
            True,
            True,
        ),
        NativeBufferSpec(
            "selected_token_valid",
            NativeBufferRole.OUTPUT,
            "bool",
            (1, rows, 2051),
            True,
            True,
            True,
            True,
        ),
    )
    stages = (
        NativeStageSpec(
            "pooled-head-score",
            "glm53_native_steel_gemm_nt_bfloat16",
            ("query", "pool_keys"),
            ("head_scores",),
        ),
        NativeStageSpec(
            "score-scale-weight-reduce",
            "glm53_native_finish_pooled_score_bfloat16_pool32",
            ("head_scores", "mixture_weights", "pool_valid"),
            ("index_scores",),
        ),
        NativeStageSpec(
            "exact-partial-topk",
            "glm53_native_exact_partial_topk_512_bfloat16",
            ("index_scores",),
            ("selected_pool_scratch",),
        ),
        NativeStageSpec(
            "pool-token-expansion",
            "glm53_native_expand_selected_pools",
            ("selected_pool_scratch", "pool_indices", "pool_valid"),
            ("selected_token_indices", "selected_token_valid"),
        ),
    )
    return NativeExecutionPlan(
        mode=mode,
        query_rows=rows,
        logical_capacity_tokens=logical_capacity_tokens,
        physical_pool_rows=pool_rows,
        selected_pool_rows=512,
        selected_token_width=2051,
        buffers=buffers,
        stages=stages,
    )


def plan_native_dsa_sparse_attention_island(
    *, logical_capacity_tokens: int
) -> NativeExecutionPlan:
    """Plan the Tier-2 decode score -> sparse-attention execution island.

    The pinned MLX runtime has no fused SDPA implementation for GLM's D512
    latent decode geometry.  The native island therefore preserves its exact
    fallback sequence: BF16 NT GEMM, bool-mask materialization, precise BF16
    softmax, and BF16/FP32 split-K NN GEMM.  Context-sized score, selected
    indices, gathered latent, softmax, and split-K accumulation remain private
    plan scratch; only the D512 attention output crosses back to MLX.
    """

    capacity = plan_nope_cache_capacity(logical_capacity_tokens)
    pool_rows = capacity.physical_pool_rows
    kv_rows = capacity.physical_capacity_tokens
    buffers = (
        NativeBufferSpec(
            "index_query",
            NativeBufferRole.INPUT,
            "bfloat16",
            (1, 1, 32, 128),
            True,
            False,
            False,
            False,
        ),
        NativeBufferSpec(
            "mixture_weights",
            NativeBufferRole.INPUT,
            "bfloat16",
            (1, 1, 32),
            True,
            False,
            False,
            False,
        ),
        NativeBufferSpec(
            "pool_keys",
            NativeBufferRole.INPUT,
            "bfloat16",
            (1, pool_rows, 128),
            True,
            False,
            False,
            False,
        ),
        NativeBufferSpec(
            "latent",
            NativeBufferRole.INPUT,
            "bfloat16",
            (1, 1, kv_rows, 512),
            True,
            False,
            False,
            False,
        ),
        NativeBufferSpec(
            "attention_query",
            NativeBufferRole.INPUT,
            "bfloat16",
            (1, 64, 1, 512),
            True,
            False,
            False,
            False,
        ),
        NativeBufferSpec(
            "head_scores",
            NativeBufferRole.SCRATCH,
            "bfloat16",
            (32, pool_rows),
            True,
            True,
            True,
            False,
        ),
        NativeBufferSpec(
            "index_scores",
            NativeBufferRole.SCRATCH,
            "bfloat16",
            (1, 1, pool_rows),
            True,
            True,
            True,
            False,
        ),
        NativeBufferSpec(
            "selected_indices",
            NativeBufferRole.SCRATCH,
            "int32",
            (1, 1, 2051),
            True,
            True,
            True,
            False,
        ),
        NativeBufferSpec(
            "selected_valid",
            NativeBufferRole.SCRATCH,
            "bool",
            (1, 1, 2051),
            True,
            True,
            True,
            False,
        ),
        NativeBufferSpec(
            "scaled_query",
            NativeBufferRole.SCRATCH,
            "bfloat16",
            (64, 512),
            True,
            True,
            True,
            False,
        ),
        NativeBufferSpec(
            "gathered_latent",
            NativeBufferRole.SCRATCH,
            "bfloat16",
            (2051, 512),
            True,
            True,
            True,
            False,
        ),
        NativeBufferSpec(
            "attention_scores",
            NativeBufferRole.SCRATCH,
            "bfloat16",
            (64, 2051),
            True,
            True,
            True,
            False,
        ),
        NativeBufferSpec(
            "splitk_accum",
            NativeBufferRole.SCRATCH,
            "float32",
            (4, 64, 512),
            True,
            True,
            True,
            False,
        ),
        NativeBufferSpec(
            "attention_output",
            NativeBufferRole.OUTPUT,
            "bfloat16",
            (1, 64, 1, 512),
            True,
            True,
            True,
            True,
        ),
    )
    stages = (
        NativeStageSpec(
            "pooled-score-selection-expansion",
            NATIVE_DSA_SCORE_ISLAND_ABI,
            ("index_query", "mixture_weights", "pool_keys"),
            ("selected_indices", "selected_valid"),
        ),
        NativeStageSpec(
            "gather-and-query-scale",
            "glm53_native_prepare_sparse_attention_bfloat16",
            ("selected_indices", "selected_valid", "attention_query", "latent"),
            ("scaled_query", "gathered_latent"),
        ),
        NativeStageSpec(
            "query-key-gemm",
            "mlx0322-steel-nt-bf16",
            ("scaled_query", "gathered_latent"),
            ("attention_scores",),
        ),
        NativeStageSpec(
            "selection-mask",
            "glm53_native_mask_sparse_attention_scores_bfloat16",
            ("selected_valid", "attention_scores"),
            ("attention_scores",),
        ),
        NativeStageSpec(
            "precise-softmax",
            "mlx0322-block-softmax-precise-bf16",
            ("attention_scores",),
            ("attention_scores",),
        ),
        NativeStageSpec(
            "weighted-value-splitk",
            "mlx0322-steel-splitk-nn-bf16-fp32",
            ("attention_scores", "gathered_latent"),
            ("splitk_accum", "attention_output"),
        ),
    )
    return NativeExecutionPlan(
        mode=NativeExecutionMode.DECODE,
        query_rows=1,
        logical_capacity_tokens=logical_capacity_tokens,
        physical_pool_rows=pool_rows,
        selected_pool_rows=512,
        selected_token_width=2051,
        buffers=buffers,
        stages=stages,
    )


def plan_native_indexpool_update_island(
    *, logical_capacity_tokens: int
) -> NativeExecutionPlan:
    """Plan decode raw19/kpool4 update through accepted Tier-1 selection."""

    capacity = plan_nope_cache_capacity(logical_capacity_tokens)
    pool_rows = capacity.physical_pool_rows

    def input_buffer(name: str, dtype: str, shape: tuple[int, ...]):
        return NativeBufferSpec(
            name,
            NativeBufferRole.INPUT,
            dtype,
            shape,
            True,
            False,
            False,
            False,
        )

    def external_state(name: str, dtype: str, shape: tuple[int, ...]):
        return NativeBufferSpec(
            name,
            NativeBufferRole.EXTERNAL_MUTABLE_STATE,
            dtype,
            shape,
            True,
            False,
            True,
            False,
        )

    def scratch(name: str, dtype: str, shape: tuple[int, ...]):
        return NativeBufferSpec(
            name,
            NativeBufferRole.SCRATCH,
            dtype,
            shape,
            True,
            True,
            True,
            False,
        )

    buffers = (
        input_buffer("key", "bfloat16", (1, 1, 128)),
        input_buffer("gate", "bfloat16", (1, 1, 128)),
        input_buffer("current_valid", "bool", (1, 1)),
        input_buffer("query", "bfloat16", (1, 1, 32, 128)),
        input_buffer("mixture_weights", "bfloat16", (1, 1, 32)),
        input_buffer("compress_ape", "bfloat16", (4, 128)),
        external_state("pool_keys", "bfloat16", (1, pool_rows, 128)),
        external_state("pool_indices", "int64", (1, pool_rows, 4)),
        external_state("pool_valid", "bool", (1, pool_rows)),
        scratch("raw_keys_a", "bfloat16", (1, 19, 128)),
        scratch("raw_keys_b", "bfloat16", (1, 19, 128)),
        scratch("raw_gates_a", "bfloat16", (1, 19, 128)),
        scratch("raw_gates_b", "bfloat16", (1, 19, 128)),
        scratch("raw_valid_a", "bool", (1, 19)),
        scratch("raw_valid_b", "bool", (1, 19)),
        scratch("raw_positions_a", "int64", (1, 19)),
        scratch("raw_positions_b", "int64", (1, 19)),
        scratch("pool_logits", "bfloat16", (128, 4)),
        scratch("pool_probabilities", "bfloat16", (128, 4)),
        scratch("head_scores", "bfloat16", (32, pool_rows)),
        scratch("index_scores", "bfloat16", (1, 1, pool_rows)),
        scratch("selected_pool_scratch", "uint32", (1, 1, 512)),
        NativeBufferSpec(
            "selected_token_indices",
            NativeBufferRole.OUTPUT,
            "int32",
            (1, 1, 2051),
            True,
            True,
            True,
            True,
        ),
        NativeBufferSpec(
            "selected_token_valid",
            NativeBufferRole.OUTPUT,
            "bool",
            (1, 1, 2051),
            True,
            True,
            True,
            True,
        ),
    )
    stages = (
        NativeStageSpec(
            "advance-raw19",
            "glm53_native_advance_indexpool_raw19",
            ("key", "gate", "current_valid"),
            (
                "raw_keys_a",
                "raw_keys_b",
                "raw_gates_a",
                "raw_gates_b",
                "raw_valid_a",
                "raw_valid_b",
                "raw_positions_a",
                "raw_positions_b",
            ),
        ),
        NativeStageSpec(
            "update-kpool4-row",
            "glm53_native_update_indexpool_row_bfloat16",
            (
                "raw_keys_a",
                "raw_keys_b",
                "raw_gates_a",
                "raw_gates_b",
                "raw_valid_a",
                "raw_valid_b",
                "compress_ape",
            ),
            ("pool_logits", "pool_probabilities", "pool_keys", "pool_indices", "pool_valid"),
        ),
        NativeStageSpec(
            "tier1-score-selection",
            NATIVE_DSA_SCORE_ISLAND_ABI,
            ("query", "mixture_weights", "pool_keys", "pool_indices", "pool_valid"),
            (
                "head_scores",
                "index_scores",
                "selected_pool_scratch",
                "selected_token_indices",
                "selected_token_valid",
            ),
        ),
    )
    return NativeExecutionPlan(
        mode=NativeExecutionMode.DECODE,
        query_rows=1,
        logical_capacity_tokens=logical_capacity_tokens,
        physical_pool_rows=pool_rows,
        selected_pool_rows=512,
        selected_token_width=2051,
        buffers=buffers,
        stages=stages,
    )
