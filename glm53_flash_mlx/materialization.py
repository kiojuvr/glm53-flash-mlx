"""Bounded recurrent-state materialization policy and telemetry."""

from __future__ import annotations

from threading import Lock

MATERIALIZATION_POLICY = "nested-cache-eval-clear-v1"
MATERIALIZATION_INTERVAL_TOKENS = 256


class RecurrentMaterializationTelemetry:
    """Thread-safe observations from the production decode generator."""

    def __init__(self, interval_tokens: int = MATERIALIZATION_INTERVAL_TOKENS):
        self._lock = Lock()
        self.configured_interval_tokens = int(interval_tokens)
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.completed_materializations = 0
            self.decode_steps_since_materialization = 0
            self.last_materialization_step = None
            self.last_boundary_active_bytes = None
            self.last_boundary_cache_bytes = None
            self.last_boundary_peak_bytes = None

    def start_generator(self, interval_tokens: int) -> None:
        with self._lock:
            self.configured_interval_tokens = int(interval_tokens)
            self.completed_materializations = 0
            self.decode_steps_since_materialization = 0
            self.last_materialization_step = None
            self.last_boundary_active_bytes = None
            self.last_boundary_cache_bytes = None
            self.last_boundary_peak_bytes = None

    def observe_decode_step(
        self,
        *,
        step: int,
        materialized: bool,
        memory: dict[str, int] | None = None,
    ) -> None:
        with self._lock:
            if materialized:
                self.completed_materializations += 1
                self.decode_steps_since_materialization = 0
                self.last_materialization_step = int(step)
                if memory is not None:
                    self.last_boundary_active_bytes = int(memory["active_memory_bytes"])
                    self.last_boundary_cache_bytes = int(memory["cache_memory_bytes"])
                    self.last_boundary_peak_bytes = int(memory["peak_memory_bytes"])
            else:
                last = self.last_materialization_step or 0
                self.decode_steps_since_materialization = int(step) - last

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "policy": MATERIALIZATION_POLICY,
                "configured_interval_tokens": self.configured_interval_tokens,
                "completed_materializations": self.completed_materializations,
                "decode_steps_since_materialization": (
                    self.decode_steps_since_materialization
                ),
                "last_materialization_step": self.last_materialization_step,
                "last_boundary_active_bytes": self.last_boundary_active_bytes,
                "last_boundary_cache_bytes": self.last_boundary_cache_bytes,
                "last_boundary_peak_bytes": self.last_boundary_peak_bytes,
                "metal_buffer_count_api_available": False,
            }


MATERIALIZATION_TELEMETRY = RecurrentMaterializationTelemetry()


def _memory_snapshot(mx_module) -> dict[str, int]:
    return {
        "active_memory_bytes": int(mx_module.get_active_memory()),
        "cache_memory_bytes": int(mx_module.get_cache_memory()),
        "peak_memory_bytes": int(mx_module.get_peak_memory()),
    }


def install_bounded_recurrent_materialization_policy(
    *,
    batch_generator_cls=None,
    mx_module=None,
    telemetry: RecurrentMaterializationTelemetry = MATERIALIZATION_TELEMETRY,
) -> None:
    """Pin the stock nested-cache materialization and observe completed boundaries."""
    if batch_generator_cls is None:
        from mlx_vlm.generate.ar import BatchGenerator

        batch_generator_cls = BatchGenerator
    if mx_module is None:
        import mlx.core as mx

        mx_module = mx
    if getattr(batch_generator_cls, "_glm53_bounded_materialization", False):
        return

    original_init = batch_generator_cls.__init__
    original_next = batch_generator_cls._next

    def init_pinned(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._cache_eval_interval = MATERIALIZATION_INTERVAL_TOKENS
        telemetry.start_generator(MATERIALIZATION_INTERVAL_TOKENS)

    def next_observed(self, *args, **kwargs):
        before = int(getattr(self, "_steps_counter", 0))
        result = original_next(self, *args, **kwargs)
        after = int(getattr(self, "_steps_counter", before))
        if after > before:
            materialized = (
                after % MATERIALIZATION_INTERVAL_TOKENS == 0
            )
            telemetry.observe_decode_step(
                step=after,
                materialized=materialized,
                memory=_memory_snapshot(mx_module) if materialized else None,
            )
        return result

    batch_generator_cls.__init__ = init_pinned
    batch_generator_cls._next = next_observed
    batch_generator_cls._glm53_bounded_materialization = True


def materialization_snapshot() -> dict:
    return MATERIALIZATION_TELEMETRY.snapshot()
