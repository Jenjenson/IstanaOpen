"""Contract, physics, leakage and backend equivalence for the adaptive benchmark."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from triad_rl.adaptive_env import AdaptivePlacementEnv, generate_scenario
from triad_rl.adaptive_inputs import (
    DEFAULT_CATALOGUE, FEATURE_NAMES, FEATURE_SCHEMA, INPUT_SCHEMA,
    LiveObservationAdapter, apply_placement, build_observation,
    sensing_probabilities, validate_catalogue,
)


def greedy_episode(env):
    observation = env.observe()
    rewards = []
    while not env.done:
        gain = observation["option_features"][:, FEATURE_NAMES.index("marginal_coverage")]
        action = int(np.argmax(np.where(observation["action_mask"], gain, -np.inf)))
        observation, reward, _, info = env.step(action)
        rewards.append(reward)
    return rewards, info


def test_seeded_replay_is_exact_and_every_artifact_is_json_safe():
    a, b = AdaptivePlacementEnv(seed=17), AdaptivePlacementEnv(seed=17)
    assert a.scenario == b.scenario
    rewards_a, info_a = greedy_episode(a)
    rewards_b, info_b = greedy_episode(b)
    assert rewards_a == rewards_b
    assert info_a == info_b
    json.dumps(a.scenario, allow_nan=False)
    json.dumps(info_a, allow_nan=False)
    json.dumps(a.observe()["state"], allow_nan=False)


def test_randomized_family_has_all_requested_variations_and_stress_shift():
    scenarios = [generate_scenario(seed) for seed in range(50)]
    targets = [t for s in scenarios for t in s["targets"]]
    assert {t["path"] for t in targets} == {"direct", "curved", "weaving"}
    assert {len(s["targets"]) for s in scenarios} == {1, 2, 3, 4, 5}
    assert np.ptp([t["altitude"] for t in targets]) > 70
    assert np.ptp([t["speed"] for t in targets]) > 15
    assert np.ptp([t["bearing"] for t in targets]) > 5
    assert np.ptp([t["emitter_duty"] for t in targets]) > .95
    assert len({s["public"]["weather"]["illumination"] for s in scenarios}) == 3
    assert len({s["public"]["budget_total"] for s in scenarios}) > 3
    assert len({tuple(s["public"]["available_sensor_ids"]) for s in scenarios}) > 4
    assert scenarios[0]["public"]["sites"] != scenarios[1]["public"]["sites"]
    stress = [generate_scenario(seed, "stress") for seed in range(30)]
    assert max(len(s["targets"]) for s in stress) == 8
    assert np.mean([s["public"]["forecast"]["altitude"] for s in stress]) > np.mean([s["public"]["forecast"]["altitude"] for s in scenarios]) + 40
    assert np.mean([s["public"]["weather"]["visibility"] for s in stress]) < .4


def test_no_hidden_truth_in_observation_or_forecast_features():
    env = AdaptivePlacementEnv(seed=19)
    original = env.observe()
    changed = deepcopy(env.scenario)
    changed["targets"][0].update(bearing=2.1, altitude=160., speed=30., path="weaving",
                                  emitter_duty=0., emitter_phase=.9, curvature=.7)
    changed["seed"] = 90123  # Also changes common sensing draws, not public features.
    other = env.reset(scenario=changed)
    assert original["state"] == other["state"]
    np.testing.assert_array_equal(original["option_features"], other["option_features"])
    np.testing.assert_array_equal(original["action_mask"], other["action_mask"])
    assert "targets" not in original["state"]
    assert not any("phase" in feature or "ground" in feature for feature in FEATURE_NAMES)
    assert env.info == {}


def test_hidden_future_changes_do_not_leak_during_sequential_placement():
    a, b = AdaptivePlacementEnv(seed=21), AdaptivePlacementEnv(seed=21)
    scenario = deepcopy(b.scenario)
    for target in scenario["targets"]:
        target.update(curvature=-target["curvature"], emitter_phase=.987, emitter_duty=0.)
    b.reset(scenario=scenario)
    for _ in range(2):
        obs_a, obs_b = a.observe(), b.observe()
        np.testing.assert_array_equal(obs_a["option_features"], obs_b["option_features"])
        legal = np.flatnonzero(obs_a["action_mask"][:-1])
        if not len(legal):
            break
        action = int(legal[0])
        next_a, reward_a, done_a, info_a = a.step(action)
        next_b, reward_b, done_b, info_b = b.step(action)
        np.testing.assert_array_equal(next_a["option_features"], next_b["option_features"])
        assert done_a == done_b
        if not done_a:
            assert reward_a == reward_b and info_a == info_b == {}
        else:
            break


def test_policy_consumer_cannot_mutate_environment_legality():
    env = AdaptivePlacementEnv(seed=4)
    obs = env.observe()
    obs["action_mask"][:] = False
    obs["options"][0]["position"][0] = 9999
    obs["state"]["budget_remaining"] = 0
    assert env.observe()["action_mask"].any()
    assert env.observe()["options"][0]["position"][0] != 9999
    assert env.public_state["budget_remaining"] > 0


def test_feature_contract_fixed_across_catalogue_and_offered_site_count():
    normal = AdaptivePlacementEnv(seed=0).observe()
    catalogue = deepcopy(DEFAULT_CATALOGUE[:2])
    catalogue[0]["id"] = "field-rf-replacement"
    env = AdaptivePlacementEnv(seed=0, catalogue=catalogue)
    scenario = deepcopy(env.scenario)
    scenario["public"]["sites"] = [[50., 10.], [-60., -20.], [0., 110.]]
    changed = env.reset(scenario=scenario)
    assert normal["feature_names"] == changed["feature_names"] == FEATURE_NAMES
    assert changed["feature_schema"] == FEATURE_SCHEMA
    assert changed["option_features"].shape == (7, len(FEATURE_NAMES))
    assert changed["options"][-1]["stop"]
    assert np.isfinite(changed["option_features"]).all()
    assert changed["options"][0]["sensor_id"] == "field-rf-replacement"


def test_mask_budget_separation_annulus_blocking_availability_and_site_limit():
    env = AdaptivePlacementEnv(seed=1)
    state = deepcopy(env.public_state)
    state.update(sites=[[0., 0.], [40., 0.], [45., 0.], [160., 0.], [-90., 0.]],
                 budget_total=2., budget_remaining=2., max_sites=2,
                 available_sensor_ids=["rf", "radar"], blocked_sites=[4])
    obs = build_observation(state)
    assert not obs["action_mask"][0]  # Objective standoff.
    assert not obs["action_mask"][3]  # Outside annulus.
    assert not obs["action_mask"][4]  # Explicit blocked site.
    assert not obs["action_mask"][10:25].any()  # Missing capabilities.
    updated = apply_placement(state, 1)
    obs = build_observation(updated)
    assert not obs["action_mask"][1]  # Site occupied.
    assert not obs["action_mask"][2]  # Too close to placement.
    assert updated["budget_remaining"] == pytest.approx(1.2)
    # A costly fused sensor is masked regardless of availability.
    updated["available_sensor_ids"] = [s["id"] for s in DEFAULT_CATALOGUE]
    updated["blocked_sites"] = []
    obs = build_observation(updated)
    assert not obs["action_mask"][20:25].any()
    updated = apply_placement(updated, 4)
    assert build_observation(updated)["action_mask"].tolist() == [False] * 25 + [True]


@pytest.mark.parametrize("action", [-1, 999999, .5, True, "stop"])
def test_invalid_attempts_are_penalized_and_bounded(action):
    env = AdaptivePlacementEnv(seed=3)
    total = 0.
    for index in range(3):
        obs, reward, done, info = env.step(action)
        total += reward
        assert done == (index == 2)
    assert total == pytest.approx(info["return"])
    assert info["invalid_actions"] == 3
    assert info["reward_components"]["invalid_actions"] == -3
    assert total == pytest.approx(sum(info["reward_components"].values()))
    assert obs["action_mask"][-1] and not obs["action_mask"][:-1].any()
    with pytest.raises(RuntimeError, match="done"):
        env.step(0)


def test_stop_without_sensors_is_a_breach_not_free_success():
    env = AdaptivePlacementEnv(seed=9)
    _, reward, done, info = env.step(len(env.observe()["options"]) - 1)
    assert done and not info["success"]
    assert info["breached_fraction"] == 1
    assert info["detection_rate"] == 0
    assert info["confirmed_fraction"] == 0
    assert info["cost"] == 0
    assert reward == pytest.approx(sum(info["reward_components"].values()))


def test_reward_components_telescope_and_include_cost_blindspots_coverage_and_early():
    env = AdaptivePlacementEnv(seed=23)
    _, invalid_reward, _, _ = env.step(-1)
    rewards, info = greedy_episode(env)
    assert invalid_reward + sum(rewards) == pytest.approx(info["return"])
    assert sum(info["reward_components"].values()) == pytest.approx(info["return"])
    assert info["reward_components"]["sensor_cost"] < 0
    assert info["reward_components"]["blind_spots"] < 0
    assert 0 <= info["coverage"] <= 1
    assert 0 <= info["early_detection"] <= 1
    assert info["confirmed_fraction"] <= info["detected_fraction"]


def test_synthetic_modalities_respond_to_emission_weather_altitude_and_fusion():
    catalogue = validate_catalogue()
    positions = np.zeros((5, 2))
    points = np.array([[20., 0., 10.], [20., 0., 190.]])
    clear = {"visibility": 1., "rain": 0., "illumination": 1., "humidity": 0., "rf_noise": 0.}
    good = sensing_probabilities(positions, catalogue, points, clear, 1.)
    silent = sensing_probabilities(positions, catalogue, points, clear, 0.)
    dark = sensing_probabilities(positions, catalogue, points, {**clear, "illumination": .02}, 1.)
    fog = sensing_probabilities(positions, catalogue, points, {**clear, "visibility": .1, "rain": .9}, 1.)
    assert good[0, 0, 0] > 0 and silent[0, 0, 0] == 0
    assert dark[2, 0, 2] < good[2, 0, 2] * .2
    assert dark[3, 0, 3] == good[3, 0, 3]
    assert fog[2, 0, 2] < good[2, 0, 2] * .1
    assert fog[1, 0, 1] < good[1, 0, 1]
    assert not good[:, 1, :].any()  # Slant altitude exceeds every range.
    fused = 1 - np.prod(1 - good[4, 0])
    assert fused > good[1, 0, 1] and fused > good[3, 0, 3]


def test_silent_target_rf_deployment_fails_while_radar_can_confirm_it():
    scenario = generate_scenario(55)
    target = scenario["targets"][0]
    target.update(bearing=0., altitude=15., altitude_amplitude=0., speed=10.,
                  spawn_radius=300., path="direct", emitter_duty=0.)
    scenario["targets"] = [target]
    scenario["public"].update(sites=[[120., 0.]], budget_total=2.4, budget_remaining=2.4,
                              max_sites=1, available_sensor_ids=[s["id"] for s in DEFAULT_CATALOGUE])
    scenario["public"]["weather"].update(rain=0., visibility=1., illumination=1.)
    rf, radar = AdaptivePlacementEnv(), AdaptivePlacementEnv()
    rf.reset(scenario=scenario)
    radar.reset(scenario=scenario)
    _, _, _, rf_info = rf.step(0)
    _, _, _, radar_info = radar.step(1)
    assert rf_info["detected_fraction"] == 0 and not rf_info["success"]
    assert radar_info["confirmed_fraction"] == 1 and radar_info["success"]


def test_public_forecasts_and_catalogue_changes_affect_candidate_features():
    state = deepcopy(AdaptivePlacementEnv(seed=8).public_state)
    state["tracks"] = []
    state["forecast"]["approach_weights"] = [1., 0., 0., 0., 0., 0., 0., 0.]
    east = build_observation(state)
    state["forecast"]["approach_weights"] = [0., 0., 0., 0., 1., 0., 0., 0.]
    west = build_observation(state)
    column = FEATURE_NAMES.index("marginal_coverage")
    assert not np.array_equal(east["option_features"][:, column], west["option_features"][:, column])
    east_best = int(np.argmax(east["option_features"][:, column]))
    west_best = int(np.argmax(west["option_features"][:, column]))
    assert east["options"][east_best]["position"][0] > 0
    assert west["options"][west_best]["position"][0] < 0
    state["available_sensor_ids"] = ["eo"]
    changed = build_observation(state)
    assert not changed["action_mask"][:64].any()
    assert not changed["action_mask"][96:-1].any()
    assert changed["action_mask"][64:96].any()


def test_common_sensing_draws_do_not_depend_on_placement_order():
    a = AdaptivePlacementEnv(seed=31)
    b = AdaptivePlacementEnv(seed=31)
    scenario = deepcopy(a.scenario)
    scenario["public"].update(budget_total=4., budget_remaining=4., available_sensor_ids=[s["id"] for s in DEFAULT_CATALOGUE])
    a.reset(scenario=scenario)
    b.reset(scenario=scenario)
    # RF at site0 and radar at the opposite site8; same set, swapped order.
    for env, actions in ((a, [0, 32 + 8]), (b, [32 + 8, 0])):
        for action in actions:
            env.step(action)
        if not env.done:
            env.step(len(env.observe()["options"]) - 1)
    assert a.info["target_results"] == b.info["target_results"]
    assert a.info["detected_fraction"] == b.info["detected_fraction"]
    assert a.info["early_detection"] == b.info["early_detection"]
    assert [[t for t in f["threats"]] for f in a.info["frames"]] == [[t for t in f["threats"]] for f in b.info["frames"]]


def test_real_adapter_and_simulator_build_identical_features_and_plan_transitions():
    env = AdaptivePlacementEnv(seed=43)
    adapter = LiveObservationAdapter(env.catalogue)
    live = adapter.observe(json.dumps(env.public_state), now=0)
    sim = env.observe()
    np.testing.assert_array_equal(live["option_features"], sim["option_features"])
    np.testing.assert_array_equal(live["action_mask"], sim["action_mask"])
    action = int(np.flatnonzero(live["action_mask"][:-1])[0])
    recommendation = adapter.recommendation(live, action)
    assert recommendation == sim["options"][action]
    planned = apply_placement(env.public_state, action, env.catalogue)
    simulated, _, _, _ = env.step(action)
    np.testing.assert_array_equal(build_observation(planned, env.catalogue)["option_features"], simulated["option_features"])
    stopped = apply_placement(planned, len(sim["options"]) - 1, env.catalogue)
    assert stopped["done"]
    with pytest.raises(ValueError, match="done"):
        apply_placement(stopped, action, env.catalogue)


def test_stale_and_missing_tracks_degrade_to_priors_without_becoming_fresh_again():
    state = AdaptivePlacementEnv(seed=1).public_state
    state = deepcopy(state)
    state["timestamp"] = 100.
    state["tracks"] = [
        {"id": "fresh", "position": [220, 0, 30], "timestamp": 99., "confidence": .8},
        {"id": "old", "position": [0, 220, 30], "timestamp": 70., "confidence": .9},
    ]
    adapter = LiveObservationAdapter()
    obs = adapter.observe(state, now=100.)
    assert [t["id"] for t in obs["state"]["tracks"]] == ["fresh"]
    column = FEATURE_NAMES.index("fresh_track_fraction")
    assert obs["option_features"][-1, column] == .5
    rebuilt = build_observation(obs["state"])
    np.testing.assert_array_equal(obs["option_features"], rebuilt["option_features"])
    no_tracks = adapter.observe(state, now=120.)
    assert no_tracks["state"]["tracks"] == []
    assert no_tracks["option_features"][-1, column] == 0.
    assert no_tracks["action_mask"].any()
    del state["tracks"]
    assert adapter.observe(state)["state"]["tracks"] == []


@pytest.mark.parametrize("mutation", ["schema", "future", "nan", "negative_budget", "overspent", "unknown_sensor", "future_track"])
def test_real_provider_validation_rejects_invalid_or_hidden_payloads(mutation):
    env = AdaptivePlacementEnv(seed=2)
    state = deepcopy(env.public_state)
    if mutation == "schema":
        state["schema"] = "arbitrary"
    elif mutation == "future":
        state["ground_truth_paths"] = []
    elif mutation == "nan":
        state["weather"]["rain"] = float("nan")
    elif mutation == "negative_budget":
        state["budget_remaining"] = -.1
    elif mutation == "overspent":
        state["placements"] = [{"sensor_id": "rf", "position": [60., 0.]}]
    elif mutation == "unknown_sensor":
        state["available_sensor_ids"] = ["nonexistent"]
    elif mutation == "future_track":
        state["tracks"] = [{"id": "t", "position": [100, 0, 20], "timestamp": 100.}]
    with pytest.raises((ValueError, TypeError)):
        LiveObservationAdapter().observe(state)


def test_replay_reports_current_hits_and_arc_length_speed():
    env = AdaptivePlacementEnv(seed=42)
    _, info = greedy_episode(env)
    previous = None
    speeds = {t["id"]: t["speed"] for t in env.scenario["targets"]}
    for frame in info["frames"]:
        hit_ids = {hit["target_id"] for hit in frame["detections"]}
        for target in frame["threats"]:
            assert target["detected"] == (target["id"] in hit_ids)
            if previous is not None and target["active"]:
                before = next(t for t in previous["threats"] if t["id"] == target["id"])
                distance = np.linalg.norm(np.array(target["position"]) - before["position"])
                assert distance <= speeds[target["id"]] * (frame["time"] - previous["time"]) + 1e-6
        previous = frame


def test_published_provider_examples_validate_without_simulator_truth():
    examples = Path(__file__).resolve().parents[2] / "Examples"
    state = json.loads((examples / "public-snapshot.json").read_text(encoding="utf-8"))
    catalogue = json.loads((examples / "sensor-catalogue.json").read_text(encoding="utf-8"))
    assert state["schema"] == INPUT_SCHEMA and state["source"] == "example-provider"
    assert "targets" not in state and "seed" not in state and "emitter_phase" not in json.dumps(state)
    adapter = LiveObservationAdapter(catalogue)
    observation = adapter.observe(state, now=0)
    assert observation["option_features"].shape[1] == len(FEATURE_NAMES)
    assert observation["action_mask"][:-1].any()
    assert {s["id"] for s in observation["catalogue"]} == {"rf", "radar", "eo", "thermal", "fused"}
