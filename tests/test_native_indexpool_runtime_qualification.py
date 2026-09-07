import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qualify_native_indexpool_runtime.py"


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
