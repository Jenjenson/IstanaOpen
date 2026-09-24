from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import pytest

from triad_rl.adaptive_inputs import INPUT_SCHEMA
from triad_rl.directional_inputs import BOSON_PLUS_640_18MM
from triad_rl.temporal_inputs import TemporalConfig
from triad_rl.training_workbench import (
    BALANCED_LANE_YAWS, TrainingManager, _red_centers, _reward_breakdown,
    _trajectory_digest, _viewer,
    common_sense_start)
from triad_rl.warning_algorithms import MaskedA2CPolicy, MaskedPPOPolicy, TRAINING_ALGORITHMS
from triad_rl.warning_policy import WarningPolicy


def native_context(max_sites=8, budget=8.):
    sites = []
    for radius in (30., 45.):
        for index in range(16):
            angle = index * 2 * math.pi / 16
            sites.append([radius * math.cos(angle), radius * math.sin(angle)])
    state = {
        "schema": INPUT_SCHEMA, "timestamp": 0., "sites": sites, "placements": [],
        "budget_total": budget, "budget_remaining": budget, "max_sites": max_sites,
        "min_separation": 20., "weather": {}, "tracks": [], "blocked_sites": [],
        "available_sensor_ids": ["thermal"],
        "forecast": {"approach_weights": [1 / 8] * 8, "altitude": 45.,
                     "target_size_m": .4, "angular_uncertainty": .1},
    }
    return {"runId": "training-test", "revision": 1, "completedSteps": 0,
            "committed": False, "coordinateSystem": "unreal_xy_relative_m_z_up",
            "worldOriginCm": {"x": 0., "y": 0., "z": 0.},
            "catalogue": [deepcopy(BOSON_PLUS_640_18MM)], "publicSnapshot": state,
            "temporalConfig": asdict(TemporalConfig())}


@pytest.mark.parametrize("preset", ["directional_balanced_8", "directional_public_8"])
def test_common_sense_start_is_eight_legal_limited_fov_sensors(preset):
    context = native_context()
    before = deepcopy(context)
    result = common_sense_start(context, preset)
    assert context == before
    assert result["public_only"] and len(result["placements"]) == 8
    assert len({row["siteId"] for row in result["placements"]}) == 8
    assert all(row["profileId"] == "thermal" for row in result["placements"])
    assert all(row["yawDeg"] in BOSON_PLUS_640_18MM["yaw_bins_deg"] for row in result["placements"])
    assert all(row["pitchDeg"] in BOSON_PLUS_640_18MM["pitch_bins_deg"] for row in result["placements"])
    assert "not a guarantee" in result["coverage_intent"]
    if preset == "directional_balanced_8":
        for lane, row in zip(BALANCED_LANE_YAWS, result["placements"]):
            x, y = context["publicSnapshot"]["sites"][row["siteId"]]
            assert abs((math.degrees(math.atan2(y, x)) - lane + 180) % 360 - 180) < 1e-6
            assert row["yawDeg"] == lane and row["pitchDeg"] == 10.


@pytest.mark.parametrize("preset", ["directional_balanced_8", "directional_public_8"])
@pytest.mark.parametrize("sensor_count", [1, 3, 8])
def test_common_sense_start_uses_selected_sensor_count(preset, sensor_count):
    result = common_sense_start(native_context(), preset, count=sensor_count)
    assert len(result["placements"]) == sensor_count
    assert result["sensor_count"] == sensor_count
    assert result["label"].startswith(f"{sensor_count} ·")


def test_seeded_red_centers_use_five_distinct_jittered_sectors_at_one_fixed_radius():
    red = {"minRadiusCm": 56000., "maxRadiusCm": 58000., "groupCount": 5,
           "spreadRadiusCm": 1000., "movement": {"spacingCm": 250.},
           "objectiveWorldCm": {"x": 100., "y": -200., "z": 300.}, "heightOffsetCm": 0.}
    a, b = _red_centers(red, 19), _red_centers(red, 20)
    assert a != b and a == _red_centers(red, 19)
    for index, (x, y, z) in enumerate(a):
        assert math.isclose(math.hypot(x - 100., y + 200.), 57000.) and z == 300.
        for other_x, other_y, _ in a[:index]:
            assert math.hypot(x - other_x, y - other_y) >= 2250.
    bearings = [math.degrees(math.atan2(y + 200., x - 100.)) % 360. for x, y, _ in a]
    nearest_sectors = [min(BALANCED_LANE_YAWS,
                           key=lambda lane: abs((angle - lane + 180) % 360 - 180))
                       for angle in bearings]
    assert len(set(nearest_sectors)) == 5
    assert all(abs((angle - sector + 180) % 360 - 180) <= 4
               for angle, sector in zip(bearings, nearest_sectors))


def test_trajectory_digest_excludes_blue_detection_annotations():
    frame = {"time": 1., "completedSteps": 20, "detections": [], "tracks": [],
             "threats": [{"id": "drone-1", "position": [1., 2., 3.], "active": True,
                          "observer_truth": True, "detected": False, "confirmed": False}]}
    changed = deepcopy(frame)
    changed["threats"][0].update(detected=True, confirmed=True)
    assert _trajectory_digest([frame]) == _trajectory_digest([changed])


def test_native_reward_breakdown_matches_unreal_formula():
    context = native_context()
    run = {"context": context, "reward": -1.375,
        "metrics": {"targets": 5},
        "native_metrics": {"detected_fraction": .4, "confirmed_fraction": .4,
            "timely_fraction": .2, "breached_fraction": .8, "cost": 3.}}
    result = _reward_breakdown(run)
    assert result["total"] == pytest.approx(-1.375)
    assert [row["value"] for row in result["components"]] == pytest.approx([
        .8, 1.2, 1., -4., -.375])
    assert result["components"][0]["label"] == "2 / 5 drones detected"


def test_eight_sensor_start_requires_explicit_native_workbench_allowance():
    with pytest.raises(ValueError, match="-TrainingWorkbench"):
        common_sense_start(native_context(max_sites=3, budget=3), "directional_balanced_8")


def test_policy_warm_start_biases_exact_layout_but_remains_trainable():
    context = native_context()
    layout = common_sense_start(context, "directional_balanced_8")["placements"]
    policy = WarningPolicy(context, seed=11)
    report = policy.initialize_from_placements(layout)
    assert report["trainable"] and report["placements"] == 8
    assert np.count_nonzero(policy.logits()) == 8


def test_policy_warm_start_deterministically_replays_all_eight_choices():
    context = native_context()
    layout = common_sense_start(context, "directional_balanced_8")["placements"]
    policy = WarningPolicy(context, seed=11)
    policy.initialize_from_placements(layout)
    planned, _ = policy.plan(context, deterministic=True)
    key = lambda row: (row["profileId"], row["siteId"], row["yawDeg"], row["pitchDeg"])
    assert {key(row) for row in planned} == {key(row) for row in layout}
    policy.update([([], 1.), ([], 0.)])
    with pytest.raises(ValueError, match="fresh"):
        policy.initialize_from_placements(layout)


def test_completed_training_view_retains_native_warning_metrics():
    context = native_context()
    placements = [{"sensor_id": "thermal", "sensor_index": 0, "position": [45., 0.],
                   "cost": 1., "yaw_deg": 0., "pitch_deg": 10.}]
    frames = [{"time": 0., "completedSteps": 0, "threats": [{"id": "drone-1",
        "position": [570., 0., 120.], "active": True, "observer_truth": True}],
        "detections": [], "tracks": []},
        {"time": 5., "completedSteps": 250, "threats": [{"id": "drone-1",
        "position": [520., 0., 110.], "active": True, "observer_truth": True}],
        "detections": [], "tracks": []}]
    view = _viewer({"context": context, "public_placements": placements,
        "metrics": {"detected_fraction": .8, "mean_drone_warning_s": 41.25,
                    "team_warning_s": 48.5, "cost": 1., "targets": 5},
        "frames": frames}, 7, 32)
    assert view["label"] == "RL training · episode 7 / 32"
    assert view["placements"] == placements
    assert view["metrics"]["mean_drone_warning_seconds_lower_bound"] == 41.25
    assert view["metrics"]["team_warning_seconds_lower_bound"] == 48.5
    assert view["frames"] == frames and view["frames"][0]["threats"][0]["position"] != \
        view["frames"][1]["threats"][0]["position"]


def test_background_manager_retains_progress_checkpoints_and_summary(tmp_path):
    context = native_context()

    class Client:
        def __init__(self, **_): self.closed = False
        def reset(self, _): return {}
        def get_blue_context(self): return deepcopy(context)
        def close(self): self.closed = True

    class Policy:
        def __init__(self, *_args, sensor_count=None, **_kwargs):
            self.updates = 0
            self.sensor_count = sensor_count
        def save(self, path): Path(path).write_text("{}", encoding="utf-8"); return "sha"
        def update(self, _): self.updates += 1; return {"update": self.updates}

    calls = []
    def episode(_client, policy, seed, **kwargs):
        number = len(calls) + 1
        calls.append((seed, kwargs))
        placements = [{"sensor_id": "thermal", "sensor_index": 0, "position": [45., 0.],
                       "cost": 1., "yaw_deg": 180., "pitch_deg": 10.}]
        evidence = [{"droneId": index, "firstDetectionSeconds": 10.,
                     "firstConfirmationSeconds": 12., "zoneEntrySeconds": 50.,
                     "warningSeconds": 40.} for index in range(5)]
        targets = [{"id": f"drone-{index}", "first_detection": 10.,
                    "first_confirmation": 12., "time_to_zone": 50.,
                    "warning_time": 40., "timely_confirmed": True,
                    "unresolved": False} for index in range(5)]
        return {"context": deepcopy(context), "public_placements": placements,
                "placements": [{"profileId": "thermal", "siteId": 16,
                                 "yawDeg": 180., "pitchDeg": 10.}],
                "metrics": {"mean_drone_warning_s": float(number), "team_warning_s": float(number),
                            "first_detection_s": 10., "detected_fraction": 1.,
                            "targets": 5, "cost": 1.},
                "native_metrics": {"mean_drone_warning_seconds_lower_bound": float(number),
                    "team_warning_seconds_lower_bound": float(number), "detected_fraction": 1.,
                    "confirmed_fraction": 1., "timely_fraction": 1.,
                    "breached_fraction": 0., "cost": 1.},
                "reward": 9.875, "red_native_reward": -9.875,
                "red_policy": "fixed_radius_random_sectors",
                "red_decision": {"formation": "randomized_eight_sector_fixed_radius",
                    "radius_cm": 57000., "angles_degrees": [5., 50., 95., 185., 275.],
                    "sector_centers_degrees": [0., 45., 90., 180., 270.]},
                "elapsed_seconds": 50.,
                "warning_evidence": evidence, "target_results": targets,
                "frames": [{"time": 0., "completedSteps": 0, "threats": [],
                            "detections": [], "tracks": []}],
                "trajectory_sha256": "matched",
                "run_id": f"run-{number}", "steps": 1000, "seed": seed,
                "sensor_count": getattr(policy, "sensor_count", 1)}, []

    manager = TrainingManager(8765, tmp_path, client_factory=Client,
                              policy_factory=Policy, episode_runner=episode)
    manager.start({"name": "Perimeter watcher", "algorithm": "reinforce",
                   "episodes": 4, "batchSize": 2,
                   "validationCases": 2, "headroomProbes": 0, "checkpointInterval": 2,
                   "sensorCount": 3, "seed": 917, "initialization": "untrained"})
    manager.thread.join(timeout=5)
    status = manager.status()
    assert status["phase"] == "complete" and not status["running"], status
    assert status["episode"] == 4 and len(status["history"]) == 4
    assert all(row["trainingReward"] == row["meanWarningSeconds"] for row in status["history"])
    assert all(row["nativeReward"] == pytest.approx(9.875) for row in status["history"])
    assert all(row["blueNativeReward"] == pytest.approx(9.875) for row in status["history"])
    assert all(row["redNativeReward"] == pytest.approx(-9.875) for row in status["history"])
    assert all(row["redPolicy"] == "fixed_radius_random_sectors" for row in status["history"])
    assert status["history"][0]["redScenario"]["spawnRadiusM"] == 570.
    assert all(len(row["rewardBreakdown"]["components"]) == 5 for row in status["history"])
    assert status["history"][-1]["validationChanges"][0]["value"] == pytest.approx(4.)
    assert status["registeredModel"]["name"] == "Perimeter watcher"
    assert status["sensorCount"] == 3
    assert status["bestEpisode"] == 4
    assert status["checkpoints"] == [0, 2, 4]
    output = Path(status["outputDirectory"])
    assert (output / "configuration.json").exists() and (output / "summary.json").exists()
    logged = [json.loads(line) for line in
              (output / "training.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(logged) == 4 and logged[-1]["validationChanges"]
    assert len(calls) == 17
    assert all(kwargs.get("deterministic") for seed, kwargs in calls if seed >= 2700000)
    validation_seeds = {seed for seed, _ in calls if 2700000 <= seed < 2800000}
    test_seeds = {seed for seed, _ in calls if 3800000 <= seed < 3900000}
    assert len(validation_seeds) == len(test_seeds) == 2
    assert validation_seeds.isdisjoint(test_seeds)
    assert manager.registry.list()[0]["name"] == "Perimeter watcher"
    assert status["registeredModel"]["bestObservedEpisode"] == 4
    assert status["registeredModel"]["bestObservedWarningSeconds"] == pytest.approx(10.)
    assert (output / "best-observed-episode.json").exists()
    comparison = manager.registry.comparison(status["registeredModel"]["id"])
    assert comparison["rl"]["maxSensors"] == comparison["baseline"]["maxSensors"] == 3
    assert comparison["rl"]["metrics"]["target_results"]
    assert comparison["audit"]["sameTrajectories"]
    observed = manager.registry.observed_comparison(status["registeredModel"]["id"])
    assert observed["audit"]["exactTrainingEpisode"] == 4
    assert observed["audit"]["generalPolicyClaim"] is False
    json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert manager.registry.list()[0]["sensorCount"] == 3
    assert len(list((output / "episodes").glob("episode-*.json"))) == 4
    assert manager.replay("baseline")["episode"] == 0
    assert manager.replay("latest")["episode"] == 4
    assert manager.replay("checkpoint-2")["episode"] == 2
    assert manager.replay("best")["episode"] == 4
    assert len(manager.export()["history"]) == 4
    assert manager.status()["viewerSelection"] == "latest"


@pytest.mark.parametrize("name", ["", "   ", "bad\nname", "x" * 65])
def test_training_requires_a_safe_nonempty_model_name(name):
    with pytest.raises(ValueError, match="Model name"):
        TrainingManager.validate({"name": name, "algorithm": "reinforce",
                                  "episodes": 4, "batchSize": 2,
                                  "sensorCount": 5, "seed": 1,
                                  "initialization": "untrained"})


def test_training_rejects_unknown_algorithm():
    with pytest.raises(ValueError, match="algorithm"):
        TrainingManager.validate({"name": "test", "algorithm": "ddpg",
                                  "episodes": 4, "batchSize": 2,
                                  "sensorCount": 5, "seed": 1,
                                  "initialization": "untrained"})


def test_training_episode_ceiling_is_a_console_guard_not_512():
    config = TrainingManager.validate({"name": "long run", "algorithm": "reinforce",
        "episodes": 1024, "batchSize": 32, "sensorCount": 3,
        "seed": 1, "initialization": "untrained"})
    assert config["episodes"] == 1024 and config["sensorCount"] == 3
    with pytest.raises(ValueError, match="10000"):
        TrainingManager.validate({**config, "episodes": 10016})


@pytest.mark.parametrize("sensor_count", [0, 9, 2.5, True])
def test_training_rejects_invalid_sensor_count(sensor_count):
    with pytest.raises(ValueError, match="Sensor count"):
        TrainingManager.validate({"name": "test", "algorithm": "reinforce",
            "episodes": 4, "batchSize": 2, "sensorCount": sensor_count,
            "seed": 1, "initialization": "untrained"})


@pytest.mark.parametrize("policy_type", [WarningPolicy, MaskedPPOPolicy, MaskedA2CPolicy])
@pytest.mark.parametrize("sensor_count", [1, 3, 8])
def test_training_policy_enforces_selected_exact_sensor_count(policy_type, sensor_count):
    context = native_context()
    policy = policy_type(context, seed=31, sensor_count=sensor_count)
    for seed in range(3):
        placements, records = policy.plan(context, rng=np.random.default_rng(seed))
        assert len(placements) == sensor_count
        assert len(records) == sensor_count + 1


@pytest.mark.parametrize("policy_type,algorithm", [
    (MaskedPPOPolicy, "ppo"), (MaskedA2CPolicy, "a2c")])
def test_masked_actor_critic_trains_saves_and_reloads(policy_type, algorithm, tmp_path):
    context = native_context()
    policy = policy_type(context, seed=31, sensor_count=3)
    episodes = []
    for seed, reward in ((4, 5.), (5, 40.), (6, 15.), (7, 32.)):
        placements, records = policy.plan(context, rng=np.random.default_rng(seed))
        assert len(placements) == 3 and records
        assert all(record["mask"][record["action"]] for record in records)
        episodes.append((records, reward))
    before = policy.logits()
    report = policy.update(episodes)
    assert report["algorithm"] == algorithm and report["update"] == 1
    assert np.isfinite(policy.logits()).all() and not np.array_equal(before, policy.logits())
    assert np.isfinite(policy.values).all() and np.any(policy.value_updates)
    checkpoint = tmp_path / f"{algorithm}.json"
    policy.save(checkpoint)
    restored = policy_type.load(checkpoint, context)
    np.testing.assert_allclose(restored.logits(), policy.logits())
    np.testing.assert_allclose(restored.values, policy.values)
    assert restored.updates == policy.updates and restored.optimizer_steps == policy.optimizer_steps
    assert restored.sensor_count == 3
    assert len(restored.plan(context, deterministic=True)[0]) == 3


def test_training_algorithm_catalogue_has_the_four_supported_choices():
    assert list(TRAINING_ALGORITHMS) == ["local_ppo", "reinforce", "ppo", "a2c"]
    assert {row["label"] for row in TRAINING_ALGORITHMS.values()} == {
        "Local refinement PPO", "REINFORCE", "Masked PPO", "Masked A2C"}
