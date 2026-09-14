"""Exact scoring parity using handcrafted cases; never generate new scenarios."""
from copy import deepcopy
import json

import numpy as np
import pytest

from triad_rl import adaptive_env as frozen
from triad_rl import adaptive_inputs as legacy
from triad_rl import temporal_env as wrapper
from triad_rl.temporal_inputs import FEATURE_NAMES, build_observation
from test_temporal_inputs import catalogue, config, public_state, assert_observations_equal


def scenario():
    return {"schema": frozen.SCENARIO_SCHEMA, "seed": 499, "split": "handcrafted",
            "public": public_state(), "objective_radius": 20., "dt": 1.,
            "required_confirmations": 2, "confirmation_window": 3,
            "defence_lead_time": 4., "max_invalid_actions": 3,
            "targets": [{"id": "private-target-not-in-public-inputs", "bearing": 0.,
                         "spawn_radius": 220., "altitude": 50., "speed": 20.,
                         "path": "direct", "curvature": .2, "weave_amplitude": .1,
                         "weave_phase": .4, "altitude_amplitude": 0.,
                         "emitter_duty": .7, "emitter_period": 7, "emitter_phase": .2}]}


def forbidden(*args, **kwargs):
    raise AssertionError("No scenario generation or ghost constructor is allowed")


@pytest.fixture(autouse=True)
def no_new_generated_scenarios(monkeypatch):
    monkeypatch.setattr(frozen, "generate_scenario", forbidden)
    monkeypatch.setattr(wrapper.robust, "make_case", forbidden)
    monkeypatch.setattr(frozen.AdaptivePlacementEnv, "__init__", forbidden)


def core(case, sensors):
    # Independent controlled initialization of the exact frozen reference.
    env = frozen.AdaptivePlacementEnv.__new__(frozen.AdaptivePlacementEnv)
    env.catalogue = legacy.validate_catalogue(sensors)
    env.split, env._custom_catalogue = case["split"], True
    env._rng = np.random.default_rng(499)
    env.reset(scenario=case)
    return env


def create(case=None, sensors=None, **kwargs):
    return wrapper.TemporalPlacementEnv(scenario=scenario() if case is None else case,
        catalogue=catalogue() if sensors is None else sensors, config=kwargs.pop("config", config()), **kwargs)


def test_supplied_constructor_has_no_hidden_generator_or_base_constructor_calls():
    case, sensors = scenario(), catalogue()
    expected = deepcopy((case, sensors))
    env = create(case, sensors)
    assert wrapper.TemporalRobustPlacementEnv is wrapper.TemporalPlacementEnv
    assert env.feature_names == FEATURE_NAMES
    assert env.case_metadata["profile"] == "supplied"
    assert env.case_metadata["scenario_filtered"] is False
    assert env.info == {} and not env.done and env.placements == []
    assert (case, sensors) == expected
    assert env.scenario == core(case, sensors).scenario
    assert isinstance(env._env, frozen.AdaptivePlacementEnv)
    with pytest.raises(AttributeError):
        _ = env.nonexistent_public_attribute


def test_seed_profile_constructor_calls_case_factory_exactly_once(monkeypatch):
    calls = []
    case, sensors = scenario(), catalogue()
    metadata = {"scenario_seed": 499, "profile": "stress", "scenario_filtered": False}
    def make_case(seed, profile):
        calls.append((seed, profile))
        return deepcopy(case), deepcopy(sensors), deepcopy(metadata)
    monkeypatch.setattr(wrapper.robust, "make_case", make_case)
    env = wrapper.TemporalPlacementEnv(seed=499, profile="stress", config=config())
    assert calls == [(499, "stress")]
    assert env.case_metadata == metadata
    assert env.catalogue == legacy.validate_catalogue(sensors)
    assert env.scenario == core(case, sensors).scenario


@pytest.mark.parametrize("actions", [[0, 1, 6], [3, 1, 6], [6], [-1, 0, 6], [-1, -1, -1], [0, 1, 2]])
def test_same_actions_have_exact_frozen_rewards_done_and_complete_terminal_metrics(actions):
    env, reference = create(), core(scenario(), catalogue())
    rewards = []
    for action in actions:
        temporal, reward, done, info = env.step(action)
        old, expected_reward, expected_done, expected_info = reference.step(action)
        assert reward == expected_reward and done == expected_done and info == expected_info
        np.testing.assert_array_equal(temporal["action_mask"], old["action_mask"])
        np.testing.assert_array_equal(temporal["option_features"][:, :len(legacy.FEATURE_NAMES)], old["option_features"])
        rewards.append(reward)
    assert env.done and env.info == reference.info
    assert sum(rewards) == env.total_reward == env.info["return"]
    assert sum(env.info["reward_components"].values()) == pytest.approx(env.info["return"], abs=2e-14)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_external_public_builder_matches_every_environment_observation():
    env = create()
    for action in (0, 1, 6):
        assert_observations_equal(env.observe(), build_observation(env.public_state, env.catalogue, config=env.config))
        env.step(action)
    assert_observations_equal(env.observe(), build_observation(json.dumps(env.public_state), env.catalogue, config=env.config))


def test_builder_only_receives_public_state_and_catalogue_not_private_case(monkeypatch):
    original, calls = wrapper.TemporalObservationBuilder.observe, []
    def observe(builder, payload, sensors=None, **kwargs):
        assert payload["schema"] == legacy.INPUT_SCHEMA
        assert not {"targets", "seed", "scenario_seed", "defence_lead_time", "dt"} & set(payload)
        calls.append(deepcopy(payload))
        return original(builder, payload, sensors, **kwargs)
    monkeypatch.setattr(wrapper.TemporalObservationBuilder, "observe", observe)
    env = create()
    env.step(0)
    env.step(6)
    assert len(calls) >= 3
    assert all("private-target-not-in-public-inputs" not in json.dumps(row) for row in calls)
    assert "targets" in env.scenario  # Reporting access is explicitly separate.


def test_private_future_changes_never_change_public_features_even_after_scoring():
    changed = scenario()
    changed["targets"][0].update(bearing=1.5, altitude=180., speed=30., path="weaving", emitter_duty=0.)
    left, right = create(), create(changed)
    for action in (0, 6):
        assert_observations_equal(left.observe(), right.observe())
        left.step(action)
        right.step(action)
    assert_observations_equal(left.observe(), right.observe())
    assert left.info != right.info


@pytest.mark.parametrize("key,value", [("objective_radius", 21.), ("dt", .5), ("required_confirmations", 1),
                                       ("confirmation_window", 4), ("defence_lead_time", 3.)])
def test_mission_mismatch_fails_without_resetting_existing_environment(key, value):
    env = create()
    env.step(0)
    before, original_core = env.observe(), env._env
    altered = scenario()
    altered[key] = value
    with pytest.raises(ValueError, match="configuration"):
        env.reset(scenario=altered, catalogue=catalogue())
    assert env._env is original_core
    assert_observations_equal(env.observe(), before)


def test_explicit_nondefault_public_mission_matches_core_exactly():
    case = scenario()
    case.update(objective_radius=25., dt=.5, required_confirmations=3, confirmation_window=4, defence_lead_time=2.)
    mission = config(objective_radius_m=25., look_interval_s=.5, required_confirmations=3,
                     confirmation_window=4, lead_time_s=2.)
    env, reference = create(case, config=mission), core(case, catalogue())
    for action in (0, 1, 6):
        _, reward, done, info = env.step(action)
        _, expected_reward, expected_done, expected_info = reference.step(action)
        assert (reward, done, info) == (expected_reward, expected_done, expected_info)


def test_reset_discards_cache_old_placements_and_catalogue_and_reuses_no_generated_case():
    env = create()
    env.step(0)
    old_builder = env._builder
    altered = scenario()
    altered["public"]["weather"]["rain"] = .9
    sensors = catalogue()
    sensors.reverse()
    sensors[0]["height_m"] = 15.
    observation = env.reset(scenario=altered, catalogue=sensors)
    assert env._builder is not old_builder
    assert env.placements == [] and not env.done and env.total_reward == 0. and env.info == {}
    assert_observations_equal(observation, build_observation(altered["public"], sensors, config=env.config))
    assert env.catalogue == legacy.validate_catalogue(sensors)
    # Omitting a supplied-case catalogue retains this explicit catalogue.
    again = env.reset(scenario=altered)
    assert_observations_equal(again, observation)


def test_observation_or_constructor_input_mutation_does_not_change_core():
    case, sensors = scenario(), catalogue()
    env = create(case, sensors)
    expected = env.observe()
    case["targets"][0]["altitude"] = 180.
    case["public"]["budget_remaining"] = 0.
    sensors[0]["ranges"]["rf"] = 1.
    observed = env.observe()
    observed["action_mask"][:] = False
    observed["state"]["weather"]["rain"] = 1.
    observed["option_features"][:] = -9
    assert_observations_equal(env.observe(), expected)


def test_impossible_empty_available_catalogue_case_is_retained_unfiltered():
    case = scenario()
    case["public"]["available_sensor_ids"] = []
    env = create(case)
    assert env.case_metadata["scenario_filtered"] is False
    assert not env.observe()["action_mask"][:-1].any()
    _, reward, done, info = env.step(6)
    assert done and not info["success"]
    assert info["detected_fraction"] == 0. and info["breached_fraction"] == 1. and info["cost"] == 0.
    assert reward == info["return"]


def test_explicit_resets_do_not_advance_unseeded_sequence_and_unseeded_reset_calls_once(monkeypatch):
    env = create()
    calls = []
    class Sequence:
        def integers(self, low, high):
            calls.append((low, high))
            return 499
    env._sequence_rng = Sequence()
    env.reset(seed=499, scenario=scenario())
    assert calls == []
    factories = []
    def make_case(seed, profile):
        factories.append((seed, profile))
        return scenario(), catalogue(), {"scenario_seed": 499, "profile": profile, "scenario_filtered": False}
    monkeypatch.setattr(wrapper.robust, "make_case", make_case)
    env.reset(seed=499, profile="capability")
    assert calls == [] and factories == [(499, "capability")]
    env.reset()
    assert calls == [(0, 2 ** 62)] and factories == [(499, "capability"), (499, "capability")]


def test_invalid_initialization_and_conflicting_seed_fail_without_generation():
    for kwargs in ({}, {"seed": True}, {"seed": -1}, {"seed": 499, "profile": "unknown"},
                   {"seed": 499, "catalogue": catalogue()}, {"seed": 500, "scenario": scenario()},
                   {"scenario": {**scenario(), "seed": True}}, {"scenario": []}):
        with pytest.raises(ValueError):
            wrapper.TemporalPlacementEnv(**kwargs)
    env = create()
    with pytest.raises(ValueError):
        env.reset(scenario=[])  # No attempted generation or old-state mutation.
