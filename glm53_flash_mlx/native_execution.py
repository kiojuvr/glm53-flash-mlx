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


class NativeExecutionContractError(ValueError):
    """Raised before an incompatible native execution plan can be used."""


class NativeExecutionMode(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class NativeBufferRole(str, Enum):
    INPUT = "input"
    IMMUTABLE_WEIGHT = "immutable-weight"
    MUTABLE_STATE = "mutable-state"
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
