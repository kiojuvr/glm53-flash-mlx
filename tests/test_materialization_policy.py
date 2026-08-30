from glm53_flash_mlx.materialization import (
    MATERIALIZATION_INTERVAL_TOKENS,
    MATERIALIZATION_POLICY,
    RecurrentMaterializationTelemetry,
    install_bounded_recurrent_materialization_policy,
)


class _FakeMX:
    @staticmethod
    def get_active_memory():
        return 101

    @staticmethod
    def get_cache_memory():
        return 202

    @staticmethod
    def get_peak_memory():
        return 303


def _generator():
    class FakeBatchGenerator:
        def __init__(self):
            self._steps_counter = 0
            self._cache_eval_interval = 0
            self.advance = True
            self.state = ["unchanged"]

        def _next(self):
            if self.advance:
                self._steps_counter += 1
            return self._steps_counter

    telemetry = RecurrentMaterializationTelemetry()
    install_bounded_recurrent_materialization_policy(
        batch_generator_cls=FakeBatchGenerator,
        mx_module=_FakeMX,
        telemetry=telemetry,
    )
    generator = FakeBatchGenerator()
    return generator, telemetry, FakeBatchGenerator


def test_policy_identity_and_interval_are_fixed():
    generator, telemetry, _ = _generator()
    assert MATERIALIZATION_POLICY == "nested-cache-eval-clear-v1"
    assert MATERIALIZATION_INTERVAL_TOKENS == 256
    assert generator._cache_eval_interval == 256
    assert telemetry.snapshot()["configured_interval_tokens"] == 256


def test_completed_boundaries_are_observed_at_exact_decode_steps():
    generator, telemetry, _ = _generator()
    expected = {
        255: (0, 255, None),
        256: (1, 0, 256),
        257: (1, 1, 256),
        511: (1, 255, 256),
        512: (2, 0, 512),
        4095: (15, 255, 3840),
        4096: (16, 0, 4096),
    }
    for step in range(1, 4097):
        generator._next()
        if step in expected:
            snapshot = telemetry.snapshot()
            assert (
                snapshot["completed_materializations"],
                snapshot["decode_steps_since_materialization"],
                snapshot["last_materialization_step"],
            ) == expected[step]
    snapshot = telemetry.snapshot()
    assert snapshot["last_boundary_active_bytes"] == 101
    assert snapshot["last_boundary_cache_bytes"] == 202
    assert snapshot["last_boundary_peak_bytes"] == 303
    assert snapshot["metal_buffer_count_api_available"] is False


def test_idle_and_dummy_work_do_not_advance_counter_or_state():
    generator, telemetry, _ = _generator()
    generator.advance = False
    before_state = list(generator.state)
    before = telemetry.snapshot()
    for _ in range(8):
        generator._next()
    assert telemetry.snapshot() == before
    assert generator.state == before_state


def test_install_is_idempotent():
    generator, telemetry, generator_cls = _generator()
    install_bounded_recurrent_materialization_policy(
        batch_generator_cls=generator_cls,
        mx_module=_FakeMX,
        telemetry=telemetry,
    )
    generator._next()
    assert generator._steps_counter == 1
