"""Lifecycle tests use a deterministic bridge protocol fixture, never a second runtime."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from triad_rl.directional_inputs import BOSON_PLUS_640_18MM
from triad_rl.training_algorithms import load_algorithm
from triad_rl.training_environment import NATIVE_STEP_BATCH
from triad_rl.training_manager import TrainingManager


def native_context():
    examples = Path(__file__).resolve().parents[2] / "Examples"
    state = json.loads((examples / "public-snapshot.json").read_text())
    state.update(timestamp=0, placements=[], done=False, budget_total=2., budget_remaining=2.,
                 max_sites=3, sites=[[60., 0.], [0., 60.], [-60., 0.]], blocked_sites=[],
                 available_sensor_ids=["thermal"])
    return {"completedSteps": 0, "committed": False, "coordinateSystem": "unreal_xy_relative_m_z_up",
            "catalogue": [deepcopy(BOSON_PLUS_640_18MM)], "publicSnapshot": state,
            "temporalConfig": json.loads((examples / "temporal-config.json").read_text()),
            "sensorModel": "native_test_directional_contract", "trainingConfigurationVersion": 1}


class ProtocolFixture:
    """Canned native observations for testing transport orchestration only."""
    def __init__(self, *, block_after=None, fail_after=None):
        self.context = native_context()
        self.resets, self.layouts, self.seeds = [], [], []
        self.timeouts, self.step_batches = [], []
        self.completed = 0
        self.cancelled = False
        self.block_after, self.fail_after = block_after, fail_after
        self.blocked, self.release = threading.Event(), threading.Event()

    def __call__(self, port, timeout):
        assert port == 8765 and timeout > 0
        self.timeouts.append(timeout)
        return self

    def __enter__(self): return self
    def __exit__(self, *args): pass

    def request(self, op):
        assert op == "blue_context"
        return {"context": deepcopy(self.context)}

    def reset(self, seed, *, blue_configuration):
        self.resets.append(deepcopy(blue_configuration))
        self.seeds.append(seed)
        self.context["publicSnapshot"].update(budget_total=blue_configuration["budget"],
            budget_remaining=blue_configuration["budget"], available_sensor_ids=blue_configuration["availableSensorIds"])
        return {"runId": f"fixture-{len(self.resets)}", "minRadiusCm": 3000., "maxRadiusCm": 10000.,
                "objectiveWorldCm": {"x": 0., "y": 0., "z": 0.}, "groupCount": 2, "heightOffsetCm": 4000.}

    def get_blue_context(self): return deepcopy(self.context)

    def deploy(self, placements):
        self.layouts.append(deepcopy(placements))

    def place_red(self, centers):
        assert len(centers) == 2

    def step(self, count):
        assert count == NATIVE_STEP_BATCH
        self.step_batches.append(count)
        if self.fail_after is not None and self.completed >= self.fail_after:
            raise OSError("Native bridge disconnected during episode")
        if self.block_after is not None and self.completed >= self.block_after:
            self.blocked.set()
            assert self.release.wait(10.), "test must release the simulated native request"
            return {"blueObservation": {"terminated": False, "truncated": False}}
        self.completed += 1
        cost = len(self.layouts[-1])
        rows = [{"droneId": 1, "firstDetectionSeconds": 2., "firstConfirmationSeconds": 4.,
                 "zoneEntrySeconds": 20., "warningSeconds": 18.},
                {"droneId": 2, "firstDetectionSeconds": None, "firstConfirmationSeconds": None,
                 "zoneEntrySeconds": 15., "warningSeconds": 0.}]
        return {"blueObservation": {"terminated": True, "truncated": False, "metricsAvailable": True,
            "warningEvidenceForEvaluationOnly": rows, "reward": 2.2, "elapsedSeconds": 20.,
            "metrics": {"mean_drone_warning_seconds_lower_bound": 9., "team_warning_seconds_lower_bound": 13.,
                        "cost": cost, "timely_fraction": .5}}}

    def cancel(self): self.cancelled = True


def configuration(name="trial", algorithm="ppo", episodes=3):
    return {"runName": name, "algorithm": algorithm, "episodes": episodes, "budget": 2.,
            "enabledSensorIds": ["thermal"], "seed": 55, "checkpointFrequency": 2}


def finish(manager):
    manager.worker.join(20.)
    assert not manager.worker.is_alive(), "training worker failed to finish"
    return manager.detail(manager.current["id"])


@pytest.mark.parametrize("algorithm", ["reinforce", "ppo"])
def test_full_worker_native_forwarding_metrics_artifacts_reload_and_history(tmp_path, algorithm):
    bridge, owners = ProtocolFixture(), []
    manager = TrainingManager(tmp_path, client_factory=bridge,
                              acquire=lambda owner: owners.append(("acquire", owner)),
                              release=lambda owner: owners.append(("release", owner)))
    options = manager.options()
    assert options["environment"]["available"]
    assert options["catalogue"][0]["directional"]
    manager.start(configuration(algorithm=algorithm))
    run = finish(manager)
    assert run["status"] == "completed", run.get("error")
    assert bridge.timeouts == [3., 120.]  # read-only probe is short; native physics may take longer
    assert NATIVE_STEP_BATCH == 20
    assert bridge.step_batches == [20] * 12
    assert run["episode"] == 3
    assert run["summary"]["averageWarningTime"] == 9.
    assert run["summary"]["successRate"] == .5
    assert run["summary"]["averageReward"] == pytest.approx(2.2)
    assert all(row == {"budget": 2., "availableSensorIds": ["thermal"]} for row in bridge.resets)
    assert len(bridge.resets) == 13  # setup + 3 training + initial/2 periodic checkpoints × 3 held-out
    rows = run["metrics"]
    assert [row["episode"] for row in rows] == [1, 2, 3]
    assert rows[0]["policyLoss"] is None
    assert rows[1]["policyLoss"] is not None
    if algorithm == "reinforce":
        assert rows[1]["valueLoss"] is None
        assert rows[1]["optimizerUpdate"]["mean_training_warning_s"] == 9
    else:
        assert rows[1]["valueLoss"] >= 0
    for row in rows:
        assert row["firstDetectionTime"] == 2.
        assert row["confirmationTime"] == 4.
        assert row["confirmationWarningTime"] == 8.
        assert row["threatsDetected"] == row["threatsMissed"] == 1
        assert row["budgetUsed"] <= 2.
    assert run["checkpoints"]["best"]["episode"] == 0  # same scores retain the untrained baseline
    assert run["checkpoints"]["final"]["episode"] == 3
    directory = tmp_path / "trial"
    for name in ("config.json", "algorithm_config.json", "native_context.json", "sensor_config.json", "reproducibility.json",
                 "metrics.jsonl", "metrics.csv", "summary.json", "metrics.json", "graph_data.json", "evidence.jsonl"):
        assert (directory / name).is_file(), name
    saved_config = json.loads((directory / "config.json").read_text())
    assert saved_config["algorithmConfig"]["learning_rate"] > 0
    assert not set(saved_config["trainingSeeds"]) & set(saved_config["evaluationSeeds"])
    assert [row["seed"] for row in rows] == saved_config["trainingSeeds"]
    assert "trainingSeeds" not in run["config"]
    model = manager.model_spec("trial", "final")
    policy = load_algorithm(model["path"], bridge.context)
    assert policy.algorithm_id == algorithm
    assert manager.load({"runId": "trial", "checkpoint": "final"})["model"]["id"] == "trained:trial:final"
    reopened = TrainingManager(tmp_path, client_factory=bridge)
    assert reopened.detail("trial")["metrics"] == run["metrics"]
    assert reopened.runs()["runs"][0]["budget"] == 2.
    assert owners[-1] == ("release", "trial")
    manager.start(configuration(algorithm=algorithm, episodes=1))
    duplicate = finish(manager)
    assert duplicate["id"] == "trial-2"


def test_stop_discards_unfinished_episode_and_saves_completed_partial_learning(tmp_path):
    bridge = ProtocolFixture(block_after=5)  # 3 baseline evaluations, then 2 training episodes
    manager = TrainingManager(tmp_path, client_factory=bridge)
    config = configuration(episodes=50)
    config["checkpointFrequency"] = 10
    manager.start(config)
    assert bridge.blocked.wait(10.)
    with pytest.raises(ValueError, match="active"):
        manager.start(configuration("other"))
    with pytest.raises(ValueError, match="Stop training"):
        manager.model_spec("trial", "latest")
    manager.stop("trial")
    bridge.release.set()
    run = finish(manager)
    assert run["status"] == "stopped"
    assert run["episode"] == 2
    assert bridge.cancelled
    assert bridge.timeouts == [120.]
    assert bridge.step_batches == [20] * 6  # no further request after the in-flight call returns
    assert (tmp_path / "trial" / "final_partial_update.json").is_file()
    policy = load_algorithm(manager.model_spec("trial", "final")["path"], bridge.context)
    assert policy.generation == 1
    assert "final" in run["checkpoints"]


def test_native_disconnect_retains_complete_episodes_and_reports_failure(tmp_path):
    bridge = ProtocolFixture(fail_after=4)  # 3 baseline evaluations, then 1 training episode
    manager = TrainingManager(tmp_path, client_factory=bridge)
    manager.start(configuration(episodes=5))
    run = finish(manager)
    assert run["status"] == "failed"
    assert "disconnected" in run["error"]
    assert run["episode"] == 1
    assert "final" in run["checkpoints"]


def test_unavailable_native_does_not_fabricate_results(tmp_path):
    released = []
    def unavailable(*args, **kwargs): raise ConnectionRefusedError("Native bridge is not running")
    manager = TrainingManager(tmp_path, client_factory=unavailable, release=released.append)
    assert not manager.options()["environment"]["available"]
    manager.start(configuration())
    run = finish(manager)
    assert run["status"] == "failed"
    assert "not running" in run["error"]
    assert run["episode"] == 0 and not run["metrics"] and not run["checkpoints"]
    assert released[-1] == "trial"


def test_old_native_is_rejected_before_reset(tmp_path):
    bridge = ProtocolFixture()
    bridge.context.pop("trainingConfigurationVersion")
    manager = TrainingManager(tmp_path, client_factory=bridge)
    manager.start(configuration())
    run = finish(manager)
    assert run["status"] == "failed"
    assert "Build this branch" in run["error"]
    assert bridge.resets == []


def test_history_downsamples_graph_but_preserves_latest_and_full_artifacts(tmp_path):
    bridge = ProtocolFixture()
    manager = TrainingManager(tmp_path, client_factory=bridge)
    manager.start(configuration(episodes=1))
    run = finish(manager)
    original = run["metrics"][0]
    rows = [{**original, "episode": i, "reward": float(i), "warningTime": float(i)} for i in range(1, 5001)]
    manager.current["metrics"] = rows
    manager.current["episode"] = len(rows)
    manager.current["summary"] = manager._summary(rows)
    detail = manager.detail("trial")
    assert detail["metricsSampled"]
    assert len(detail["metrics"]) == 1000
    assert detail["metrics"][0]["episode"] == 1
    assert detail["metrics"][-1]["episode"] == 5000
    assert detail["latestMetric"]["episode"] == 5000
    assert detail["summary"]["averageReward"] == 2500.5
    assert len(manager.current["metrics"]) == 5000


def test_recover_marks_interrupted_run_and_retains_only_complete_jsonl(tmp_path):
    bridge = ProtocolFixture()
    manager = TrainingManager(tmp_path, client_factory=bridge)
    manager.start(configuration(episodes=1))
    finish(manager)
    directory = tmp_path / "trial"
    run = json.loads((directory / "run.json").read_text())
    run["status"] = "running"
    (directory / "run.json").write_text(json.dumps(run))
    with (directory / "metrics.jsonl").open("a") as stream:
        stream.write('{"episode":2,')
    restarted = TrainingManager(tmp_path, client_factory=bridge)
    detail = restarted.detail("trial")
    assert detail["status"] == "interrupted"
    assert detail["episode"] == 1
    assert "checkpoints" in detail


@pytest.mark.parametrize("name", ["../outside", "a/b", "NUL", "CON", "has spaces", "", "x" * 65])
def test_run_names_cannot_escape_directory_or_use_windows_reserved_paths(tmp_path, name):
    manager = TrainingManager(tmp_path, client_factory=ProtocolFixture())
    with pytest.raises(ValueError, match="name"):
        manager.start(configuration(name))
    assert not [path for path in tmp_path.iterdir() if path.name != ".training.lock"]


def test_tampered_model_cannot_be_selected_for_live(tmp_path):
    manager = TrainingManager(tmp_path, client_factory=ProtocolFixture())
    manager.start(configuration(episodes=1))
    finish(manager)
    path = manager.model_spec("trial", "final")["path"]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="integrity"):
        manager.model_spec("trial", "final")


def test_initial_disk_failure_releases_native_ownership(tmp_path, monkeypatch):
    owners = []
    manager = TrainingManager(tmp_path, client_factory=ProtocolFixture(),
                              acquire=lambda owner: owners.append(("acquire", owner)),
                              release=lambda owner: owners.append(("release", owner)))
    def fail_write(*args): raise OSError("Disk full")
    monkeypatch.setattr("triad_rl.training_manager.atomic_json", fail_write)
    with pytest.raises(OSError, match="Disk full"):
        manager.start(configuration())
    assert owners == [("acquire", "trial"), ("release", "trial")]
    assert manager.current is None
    monkeypatch.undo()
    following = TrainingManager(tmp_path, client_factory=ProtocolFixture())
    following.start(configuration("after-failure", episodes=1))
    assert finish(following)["status"] == "completed"  # filesystem ownership was released too


def test_second_manager_preserves_active_run_and_cannot_start_until_owner_finishes(tmp_path):
    bridge = ProtocolFixture(block_after=3)  # completed initial evaluation, first training step in flight
    owner = TrainingManager(tmp_path, client_factory=bridge)
    owner.start(configuration(episodes=10))
    try:
        assert bridge.blocked.wait(10.)
        path = tmp_path / "trial" / "run.json"
        before = path.read_bytes()
        assert json.loads(before)["status"] == "running"
        other_bridge = ProtocolFixture()
        observer = TrainingManager(tmp_path, client_factory=other_bridge)
        assert path.read_bytes() == before  # constructor must not label another worker interrupted
        assert observer.detail("trial")["status"] == "running"
        assert not observer.options()["environment"]["available"]
        assert other_bridge.timeouts == []  # a second console must not open even a probe socket
        with pytest.raises(ValueError, match="[Aa]ctive|[Oo]wn|[Aa]nother|[Uu]se|[Rr]eserv"):
            observer.start(configuration("second", episodes=1))
        assert not other_bridge.resets
        assert path.read_bytes() == before
    finally:
        owner.stop("trial")
        bridge.release.set()
        owner_run = finish(owner)
    assert owner_run["status"] == "stopped"
    observer.start(configuration("second", episodes=1))
    assert finish(observer)["status"] == "completed"


def test_native_ownership_failure_also_releases_training_store(tmp_path):
    def reject(owner):
        raise ValueError("Native simulator already reserved")
    manager = TrainingManager(tmp_path, client_factory=ProtocolFixture(), acquire=reject)
    with pytest.raises(ValueError, match="already reserved"):
        manager.start(configuration())
    following = TrainingManager(tmp_path, client_factory=ProtocolFixture())
    following.start(configuration("after-rejection", episodes=1))
    assert finish(following)["status"] == "completed"


def test_store_lease_excludes_other_process_and_exit_allows_interrupted_recovery(tmp_path):
    from triad_rl.training_store import TrainingStoreLease

    directory = tmp_path / "abandoned"
    directory.mkdir()
    path = directory / "run.json"
    path.write_text(json.dumps({"id": "abandoned", "status": "running"}))
    child_code = """
import sys
sys.path.insert(0, sys.argv[2])
from triad_rl.training_store import TrainingStoreLease
lease = TrainingStoreLease(sys.argv[1])
assert lease.acquire()
print('lease held', flush=True)
sys.stdin.readline()
# Intentionally do not call release(): process exit must release the OS lock.
"""
    process = subprocess.Popen([sys.executable, "-c", child_code, str(tmp_path),
                                str(Path(__file__).resolve().parents[1])],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "lease held"
        competing = TrainingStoreLease(tmp_path)
        assert not competing.acquire()
        before = path.read_bytes()
        TrainingManager(tmp_path, client_factory=ProtocolFixture())
        assert path.read_bytes() == before
        _, errors = process.communicate("\n", timeout=10.)
        assert process.returncode == 0, errors
        assert competing.acquire()
        competing.release()
        TrainingManager(tmp_path, client_factory=ProtocolFixture())
        recovered = json.loads(path.read_text())
        assert recovered["status"] == "interrupted"
        assert "saved checkpoints" in recovered["error"]
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10.)
