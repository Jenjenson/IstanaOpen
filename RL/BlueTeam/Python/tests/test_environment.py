"""Transport-double tests of v4 AEC semantics; these do not prove Unreal learning."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from triad_rl.environment import TRIADRedBlueEnv


CONFIG = Path(__file__).resolve().parents[2] / "DefaultTrainingConfig.json"


def blue(index, position=(0.0, 0.6)):
    return {
        "catalogue_index": int(index),
        "position": np.asarray(position, dtype=np.float32),
    }


class ScriptedClient:
    def __init__(self, cfg, fingerprint):
        self.cfg = cfg
        self.fingerprint = fingerprint
        self.calls = []
        self.seeds = []

    @staticmethod
    def modality(candidate):
        sensor = candidate["Sensor"]
        return (
            int(bool(sensor.get("DetectionRangeMeters", 0) > 0 and sensor.get("SupportedFrequenciesGHz")))
            | (2 if sensor.get("bEnableSearchRadar") else 0)
            | (4 if sensor.get("bEnableEOPTZ") else 0)
            | (8 if sensor.get("bEnableThermalPTZ") else 0)
        )

    @staticmethod
    def sensor_range(candidate):
        sensor = candidate["Sensor"]
        return max(
            float(sensor.get("DetectionRangeMeters", 0)),
            float(sensor.get("SearchRadarRangeMeters", 0)) if sensor.get("bEnableSearchRadar") else 0,
            float(sensor.get("EOPTZConfirmationRangeMeters", 0)) if sensor.get("bEnableEOPTZ") else 0,
            float(sensor.get("ThermalPTZConfirmationRangeMeters", 0)) if sensor.get("bEnableThermalPTZ") else 0,
        )

    def reset(self, seed):
        self.seeds.append(seed)
        self.s = {
            "SchemaVersion": "triad.rl_step.v4",
            "ScenarioId": self.cfg["ScenarioId"],
            "ConfigFingerprint": self.fingerprint,
            "Phase": "BluePlacement",
            "TransitionIndex": 0,
            "StepIndex": 0,
            "EpisodeSeed": seed,
            "RemainingBudgetUnits": float(self.cfg["SensorBudgetUnits"]),
            "RemainingSiteCount": int(self.cfg["MaximumSensorSites"]),
            "BlueReward": 0.0,
            "RedReward": 0.0,
            "ActiveTargetCount": 0,
            "DetectedTargetCount": 0,
            "TargetObservations": [],
            "PlacedCatalogueIndices": [],
            "SensorObservations": [],
            "CurrentlyDetectedTargetCount": 0,
            "TrackedTargetCount": 0,
            "AllTargetsTrackedSteps": 0,
            "InvalidActionCount": 0,
            "bTerminal": False,
            "TerminationReason": "None",
            "MountObservations": [],
        }
        for index, candidate in enumerate(self.cfg["SensorCandidates"]):
            self.s["MountObservations"].append({
                "CatalogueIndex": index,
                "CandidateId": candidate["CandidateId"],
                "SensorProfileId": candidate["SensorProfileId"],
                "MountId": candidate["MountId"],
                "OffsetEnuMeters": candidate["OffsetEnuMeters"],
                "SurfaceNormal": {"X": 0, "Y": 0, "Z": 1},
                "YawDegrees": candidate["YawDegrees"],
                "PitchDegrees": candidate["PitchDegrees"],
                "HorizontalFovDegrees": candidate["HorizontalFovDegrees"],
                "VerticalFovDegrees": candidate["VerticalFovDegrees"],
                "RangeMeters": self.sensor_range(candidate),
                "CostUnits": candidate["CostUnits"],
                "SensorModalityMask": self.modality(candidate),
                "bAllowDynamicPosition": candidate["bAllowDynamicPosition"],
                "bSurfaceResolved": True,
                "bOccupied": False,
                "bEnabled": True,
                "CurrentlyDetectingTargetCount": 0,
            })
        self.mask()
        return copy.deepcopy(self.s)

    def mask(self):
        blue_phase = self.s["Phase"] == "BluePlacement"
        sites = self.s["RemainingSiteCount"] > 0
        self.s["BlueActionMask"] = [
            bool(blue_phase and sites and mount["bSurfaceResolved"] and mount["bEnabled"]
                 and mount["CostUnits"] <= self.s["RemainingBudgetUnits"] + 1e-9
                 and (mount["bAllowDynamicPosition"] or not mount["bOccupied"]))
            for mount in self.s["MountObservations"]
        ] + [bool(blue_phase)]

    def result(self):
        self.s["TransitionIndex"] += 1
        self.mask()
        return copy.deepcopy(self.s)

    def blue_action(self, index, position=(0.0, 0.0), stop=False):
        position = [float(value) for value in position]
        self.calls.append(("blue", index, position, stop))
        if stop:
            self.s["Phase"] = "RedDeployment"
        else:
            candidate = self.cfg["SensorCandidates"][index]
            self.s["PlacedCatalogueIndices"].append(index)
            self.s["SensorObservations"].append({"X": position[0], "Y": position[1]})
            self.s["RemainingBudgetUnits"] -= float(candidate["CostUnits"])
            self.s["RemainingSiteCount"] -= 1
            self.s["BlueReward"] -= 0.05
            if not candidate["bAllowDynamicPosition"]:
                mount = self.s["MountObservations"][index]["MountId"]
                for item in self.s["MountObservations"]:
                    item["bOccupied"] |= (
                        not item["bAllowDynamicPosition"] and item["MountId"] == mount
                    )
            if self.s["RemainingSiteCount"] == 0:
                self.s["Phase"] = "RedDeployment"
        return self.result()

    def red_deployment(self, values):
        self.calls.append(("deploy", values))
        self.s.update(Phase="RedMovement", ActiveTargetCount=1)
        self.s["TargetObservations"] = [{
            "NormalizedZoneRelativeEnu": {"X": 0, "Y": 0.7, "Z": 0.1},
            "NormalizedVelocityEnu": {"X": 0, "Y": 0, "Z": 0},
            "bCurrentlyDetected": False,
            "bTracked": False,
            "bConfirmedDetected": False,
            "ConsecutiveDetectionSteps": 0,
            "FirstDetectionStep": -1,
            "LastDetectionStep": -1,
            "TrackedStepCount": 0,
        }]
        return self.result()

    def red_actions(self, values):
        self.calls.append(("move", values))
        self.s["StepIndex"] += 1
        if self.s["StepIndex"] == 1:
            self.s["BlueReward"] += 0.5
            self.s["RedReward"] -= 0.5
        else:
            self.s["BlueReward"] += 10
            self.s["RedReward"] -= 10
            self.s.update(Phase="Terminal", bTerminal=True, TerminationReason="AllTargetsConfirmed")
        return self.result()


@pytest.fixture
def env(tmp_path):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    cfg.update(MaximumSensorSites=3, SensorBudgetUnits=4.0)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    fingerprint = "md5:" + hashlib.md5(path.read_bytes()).hexdigest()
    result = TRIADRedBlueEnv(ScriptedClient(cfg, fingerprint), path)
    result.reset(seed=5)
    return result


def advance_to_movement(env):
    env.step(blue(0))
    env.step(blue(env.option_count))
    assert env.agent_selection == "red_deployment"
    env.step(np.zeros(5, dtype=np.float32))


def test_repeated_dynamic_profile_turns_deliver_each_increment_once(env):
    assert env.last()[1] == 0
    env.step(blue(0))
    assert env.agent_selection == "blue_placement"
    assert env.last()[1] == pytest.approx(-0.05)
    assert env.last()[1] == pytest.approx(-0.05)
    env.step(blue(0, (0.6, 0.0)))  # duplicate dynamic profile is intentional
    assert env.last()[1] == pytest.approx(-0.05)
    env.step(blue(env.option_count))
    assert env._cumulative_rewards["blue_placement"] == 0
    assert env.snapshot["PlacedCatalogueIndices"] == [0, 0]


def test_layout_has_named_modalities_and_fixed_site_history(env):
    before = env.observe("blue_placement")
    env.step(blue(1, (0.3, 0.4)))
    after = env.observe("blue_placement")
    decoded = env.decode_observation("blue_placement", after)
    assert env.observation_fields["blue_placement"]["mounts"]["columns"] == list(env.mount_feature_names)
    assert decoded["mounts"].shape == (5, 22)
    radar = env.mount_feature_names.index("search_radar")
    assert decoded["mounts"][1, radar] == 1
    assert decoded["placements"].shape == (3, 9)
    assert decoded["placements"][0].tolist() == pytest.approx([
        0.3, 0.4, 1 / 4, 0, 1, 0, 0, 1.2 / 4.0, 1,
    ])
    assert decoded["placements"][1:].tolist() == [[0] * 9, [0] * 9]
    assert decoded["budget"].tolist() == pytest.approx([2.8 / 4.0, 2 / 3])
    assert env.decode_observation("blue_placement", before)["placements"].sum() == 0
    for agent in env.possible_agents:
        assert env.observation_space(agent).contains(env.observe(agent))


def test_invalid_and_masked_hybrid_actions_never_reach_transport(env):
    env.snapshot["BlueActionMask"][2] = False
    calls = len(env.client.calls)
    invalid = [
        blue(2),
        blue(-1),
        blue(env.option_count + 1),
        {"catalogue_index": 0, "position": np.asarray([np.nan, 0], dtype=np.float32)},
        {"catalogue_index": 0},
        0,
    ]
    for action in invalid:
        with pytest.raises(ValueError):
            env.step(action)
    assert len(env.client.calls) == calls


def test_zero_and_conflicting_positions_are_deterministically_legalized(env):
    env.step(blue(0, (0, 0)))
    first = np.asarray(env.client.calls[-1][2])
    assert first.tolist() == pytest.approx([0, (30 + 0.5) / 150])
    env.step(blue(0, (0, 0)))
    second = np.asarray(env.client.calls[-1][2])
    assert np.linalg.norm(second) <= (150 - 0.5) / 150 + 1e-6
    assert np.linalg.norm(second) >= (30 + 0.5) / 150 - 1e-6
    assert np.linalg.norm(second - first) >= 20.5 / 150 - 1e-6
    saved = second.copy()
    env.reset(seed=5)
    env.step(blue(0, (0, 0)))
    env.step(blue(0, (0, 0)))
    assert env.client.calls[-1][2] == pytest.approx(saved.tolist())


@pytest.mark.parametrize("controls", [(0, 0), (0, 0.2), (1, 1), (-1, -1)])
def test_projection_reserves_half_metre_at_both_native_boundaries(env, controls):
    projected = env._project_blue_position(controls)
    metres = float(np.linalg.norm(projected)) * 150.0
    assert metres >= 30.5 - 1e-4
    assert metres <= 149.5 + 1e-4


def test_red_heads_only_dispatch_relevant_fields(env):
    advance_to_movement(env)
    assert env.action_space("red_deployment").shape == (5,)
    assert env.action_space("red_movement").shape == (1, 3)
    env.step(np.asarray([[0, -1, 0]], dtype=np.float32))
    assert [call[0] for call in env.client.calls] == ["blue", "blue", "deploy", "move"]
    assert env.client.calls[-1][1] == [[0, -1, 0]]


def test_delayed_terminal_credit_is_drained_once_without_summing_red_heads(env):
    advance_to_movement(env)
    env.step(np.zeros((1, 3), dtype=np.float32))
    assert env.last()[1] == pytest.approx(-0.5)
    env.step(np.zeros((1, 3), dtype=np.float32))
    received = {}
    while env.agents:
        agent = env.agent_selection
        _, reward, terminated, truncated, _ = env.last()
        assert terminated and not truncated and agent not in received
        received[agent] = reward
        env.step(None)
    assert received == pytest.approx({
        "blue_placement": 10.5, "red_deployment": -10.5, "red_movement": -10,
    })
    assert env.snapshot["BlueReward"] == pytest.approx(10.45)


def test_reset_seed_sequence_reproducible_but_not_constant(env):
    env.reset(seed=123)
    env.reset()
    env.reset()
    sequence = env.client.seeds[-3:]
    env.reset(seed=123)
    env.reset()
    env.reset()
    assert env.client.seeds[-3:] == sequence
    assert len(set(sequence)) == 3
    assert all(value == 0 for value in env._cumulative_rewards.values())


def test_stale_transition_wrong_catalogue_and_fingerprint_fail_closed(env):
    original = env.client.blue_action

    def stale(*args, **kwargs):
        state = original(*args, **kwargs)
        state["TransitionIndex"] = 0
        return state

    env.client.blue_action = stale
    with pytest.raises(RuntimeError, match="Stale"):
        env.step(blue(0))
    calls = len(env.client.calls)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(blue(2))
    assert len(env.client.calls) == calls
    env.client.blue_action = original
    env.reset(seed=8)
    state = copy.deepcopy(env.snapshot)
    state["MountObservations"][0]["SensorProfileId"] = "wrong"
    with pytest.raises(RuntimeError, match="identity"):
        env._validate_snapshot(state)
    state = copy.deepcopy(env.snapshot)
    state["ConfigFingerprint"] = "md5:" + "0" * 32
    with pytest.raises(RuntimeError, match="fingerprint"):
        env._validate_snapshot(state)


def test_pettingzoo_api_contract(env):
    from pettingzoo.test import api_test
    api_test(env, num_cycles=40, verbose_progress=False)


def test_flat_policy_schema_retains_large_values_without_clipping(env):
    env.snapshot["MountObservations"][0]["RangeMeters"] = 10000
    for agent in env.possible_agents:
        observation = env.observe(agent)
        assert observation["observation"].dtype == np.float32
        fields = env.decode_observation(agent, observation)
        assert fields["mounts"][0, env.mount_feature_names.index("range")] == pytest.approx(10000 / 300)
        assert fields["mounts"][0, 3:6].tolist() == [0, 0, 1]
        assert env.observation_space(agent).contains(observation)
        if agent != "blue_placement":
            assert fields["targets"].shape == (1, 13)


def test_transport_timeout_cannot_replay_an_uncertain_action(env):
    original = env.client.blue_action

    def accepted_but_reply_lost(*args, **kwargs):
        original(*args, **kwargs)
        raise TimeoutError("Reply lost")

    env.client.blue_action = accepted_but_reply_lost
    with pytest.raises(TimeoutError):
        env.step(blue(0))
    assert len(env.client.calls) == 1
    with pytest.raises(RuntimeError, match="reset"):
        env.step(blue(0))
    assert len(env.client.calls) == 1
    env.client.blue_action = original
    env.reset(seed=99)
    env.step(blue(0))
    assert env.last()[1] == pytest.approx(-0.05)


@pytest.mark.parametrize("field,value", [
    ("RemainingBudgetUnits", float("nan")),
    ("ActiveTargetCount", 2),
    ("RemainingSiteCount", 9),
    ("PlacedCatalogueIndices", [99]),
    ("SensorObservations", [{"X": 2, "Y": 0}]),
])
def test_malformed_state_is_rejected(env, field, value):
    state = copy.deepcopy(env.snapshot)
    state[field] = value
    with pytest.raises(RuntimeError):
        env._validate_snapshot(state)


def test_toy_inactive_red_axes_are_not_sent(env):
    assert env.action_dimension_mask("red_deployment").tolist() == [0, 0, 0, 0, 0]
    advance_to_movement(env)
    assert env.action_dimension_mask("red_movement").tolist() == [[0, 1, 0]]
    env.step(np.asarray([[0.75, -0.5, 0.8]], dtype=np.float32))
    assert env.client.calls[-1] == ("move", [[0, -0.5, 0]])
