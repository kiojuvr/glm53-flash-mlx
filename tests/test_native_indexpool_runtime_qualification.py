import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qualify_native_indexpool_runtime.py"
ARTIFACT = (
    ROOT
    / "bench-results"
    / "m3ultra512-native-indexpool-runtime-20260907.json"
)


def test_qualification_script_parses_and_uses_the_real_runtime_flag():
    source = SCRIPT.read_text()
    ast.parse(source)
    assert "ENVIRONMENT_FLAG" in source
    assert "try_native_update" not in source
    assert 'with _native_mode(arm == "native")' in source
    assert "--experimental-native-indexpool-update" in source
    assert "/v1/metrics" in source
    assert '(metrics or {}).get("server", {}).get(' in source


def test_qualification_is_split_into_resumable_external_phases():
    source = SCRIPT.read_text()
    assert '("screen", "long", "server", "all")' in source
    assert 'default="screen"' in source
    assert 'artifact["phases"]["long"]' in source
    assert "_atomic_write(args.output, artifact)" in source
    assert "start_new_session=True" in source
    assert "os.killpg(process.pid, signal.SIGTERM)" in source


def test_qualification_keeps_component_and_release_gates_distinct():
    source = SCRIPT.read_text()
    assert '"256k_wall_saving_at_least_0_75ms"' in source
    assert '"2k_regression_at_most_1_percent"' in source
    assert 'short_tps >= 15.0' in source
    assert '"release_15_tps_accepted"' in source
    assert '"keep_native_indexpool_runtime_opt_in_15_tps_pending"' in source


def test_long_phase_checks_every_materialization_boundary():
    source = SCRIPT.read_text()
    assert "LONG_STEPS = 4_096" in source
    assert "MATERIALIZATION_INTERVAL = 256" in source
    assert '"all_16_materialization_checkpoints_exact"' in source
    assert "11 * LONG_STEPS" in source
    assert "_compact_capacity_evidence(cache, 8_256)" in source


def test_qualification_artifact_passes_component_and_release_gates():
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "glm53-native-indexpool-runtime-qualification-v1"
    assert artifact["complete"] is True
    assert artifact["accepted"] is True
    assert artifact["release_15_tps_accepted"] is True
    assert artifact["decision"] == "keep_native_indexpool_runtime_and_pass_15_tps"
    assert set(artifact["phases"]) == {"screen", "long", "server"}
    assert all(phase["accepted"] for phase in artifact["phases"].values())
    assert artifact["runtime_changes"] == {
        "admission": False,
        "cache_abi": False,
        "default_backend": False,
        "opt_in_native_indexpool_runtime": True,
        "packed_moe_abi": False,
    }


def test_screen_reaches_15_tps_and_is_exact_at_both_contexts():
    screen = json.loads(ARTIFACT.read_text())["phases"]["screen"]
    assert all(screen["gates"].values())
    short = screen["contexts"]["2048"]
    long = screen["contexts"]["262144"]
    assert short["timing"]["native"]["tokens_per_second"] >= 15.0
    assert short["native_wall_saving_ms"] >= 0.0
    assert long["native_wall_saving_ms"] >= 0.75
    assert (
        long["timing"]["native"]["tokens_per_second"]
        / short["timing"]["native"]["tokens_per_second"]
        >= 0.90
    )
    for row in (short, long):
        assert row["all_full_vocab_logits_byte_exact"] is True
        assert row["all_generated_tokens_exact"] is True
        assert row["post_state_byte_exact"] is True
        assert row["native_execution_count_delta"] == row[
            "expected_native_execution_count"
        ] == 77


def test_long_qualification_is_exact_and_capacity_is_preallocated():
    long = json.loads(ARTIFACT.read_text())["phases"]["long"]
    assert all(long["gates"].values())
    assert long["steps_completed"] == 4_096
    assert long["materialization_count"] == 16
    assert long["first_divergence"] is None
    assert long["nan_count"] == 0
    assert long["native_execution_count_delta"] == long[
        "expected_native_execution_count"
    ] == 45_056
    assert len(long["evidence"]) == 16
    assert all(row["state_exact"] for row in long["evidence"].values())
    for arm in ("mlx", "native"):
        assert len(long["restored_capacity"][arm]) == 11
        assert all(row["reserved"] for row in long["restored_capacity"][arm])


def test_fresh_server_reports_the_native_runtime_abi():
    server = json.loads(ARTIFACT.read_text())["phases"]["server"]
    assert all(server["gates"].values())
    assert server["health_http_status"] == 200
    assert server["metrics_http_status"] == 200
    assert server["ready_seconds"] <= 190.0
    native = server["metrics"]["server"]["native_indexpool_update"]
    assert native["abi"].startswith("glm53-native-indexpool-update-island-v1-")
    assert native["live_plan_count"] == native["execution_count"] == 0
    assert native["scratch_bytes"] == 0
