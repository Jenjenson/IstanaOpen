"""Paired STOP baselines preserve the actor, live trajectory and shaped RTG."""
from copy import deepcopy
import json

import numpy as np
import pytest

from triad_rl.adaptive_evaluation import canonical_hash
from triad_rl.balanced_policy import BalancedPolicy
from triad_rl import counterfactual_rollout as paired
from triad_rl.robust_scenarios import RobustPlacementEnv
from triad_rl.train_robust import rollout


def best_rf(observation):
    column = observation["feature_names"].index("marginal_coverage")
    candidates = [i for i, option in enumerate(observation["options"])
                  if option["sensor_id"] == "rf" and observation["action_mask"][i]]
    return max(candidates, key=lambda i: observation["option_features"][i, column])


class ScriptedSampler:
    def __init__(self, actions):
        self.actions, self.observations = list(actions), []

    def sample(self, observation):
        self.observations.append(deepcopy(observation))
        action = self.actions.pop(0)
        if action == "stop":
            action = next(i for i, option in enumerate(observation["options"]) if option["stop"])
        return action, {"features": observation["option_features"].copy(),
                        "mask": observation["action_mask"].copy(), "action": action,
                        "value": 777., "reward": 0.}


@pytest.mark.parametrize("profile", ["normal", "stress", "capability", "mixed"])
def test_initial_stop_is_constant_and_not_an_optimality_label(profile):
    assert paired.stop_return_to_go(seed=700, profile=profile) == pytest.approx(-10.5)


def test_prefix_baseline_is_immediate_shaped_stop_reward_not_full_episode_return():
    env = RobustPlacementEnv(seed=0, profile="normal")
    initial = env.reset(seed=0)
    action = best_rf(initial)
    current, paid_reward, done, _ = env.step(action)
    assert not done and abs(paid_reward) > 0.01
    snapshot, total = deepcopy(current), env.total_reward
    baseline = paired.stop_return_to_go(seed=0, profile="normal", prior_actions=[action],
                                       expected_observation=current)
    assert paired._same_public(env.observe(), snapshot)
    assert env.total_reward == total
    _, actual_stop, done, info = env.step(len(current["options"]) - 1)
    assert done
    assert baseline == actual_stop
    assert baseline == pytest.approx(info["return"] - paid_reward)
    assert baseline != pytest.approx(info["return"])


@pytest.mark.parametrize("prefix", [[-1], [999999], [True], [0.5], "0"])
def test_illegal_prefixes_rejected(prefix):
    with pytest.raises(ValueError, match="prefix|prior_actions"):
        paired.stop_return_to_go(seed=0, profile="normal", prior_actions=prefix)


def test_masked_repeated_site_and_terminal_stop_prefix_rejected():
    env = RobustPlacementEnv(seed=0, profile="normal")
    obs = env.observe()
    action = best_rf(obs)
    with pytest.raises(ValueError, match="legal integer"):
        paired.stop_return_to_go(seed=0, profile="normal", prior_actions=[action, action])
    with pytest.raises(ValueError, match="terminal STOP"):
        paired.stop_return_to_go(seed=0, profile="normal", prior_actions=[len(obs["options"]) - 1])


def test_prefix_which_exhausted_resources_has_no_next_baseline():
    env = RobustPlacementEnv(seed=0, profile="normal")
    obs = env.observe()
    actions = []
    while True:
        action = next(i for i, option in enumerate(obs["options"])
                      if not option["stop"] and obs["action_mask"][i])
        actions.append(action)
        obs, _, done, _ = env.step(action)
        if done:
            break
    with pytest.raises(ValueError, match="already terminated"):
        paired.stop_return_to_go(seed=0, profile="normal", prior_actions=actions)


def test_wrong_seed_profile_or_changed_public_state_fails_closed():
    env = RobustPlacementEnv(seed=0, profile="normal")
    expected = env.observe()
    with pytest.raises(ValueError, match="does not match"):
        paired.stop_return_to_go(seed=1, profile="normal", expected_observation=expected)
    with pytest.raises(ValueError, match="does not match"):
        paired.stop_return_to_go(seed=0, profile="stress", expected_observation=expected)
    expected["state"]["budget_remaining"] -= .1
    with pytest.raises(ValueError, match="does not match"):
        paired.stop_return_to_go(seed=0, profile="normal", expected_observation=expected)


@pytest.mark.parametrize("seed", [True, -1, .5, "1", 10**15, 2 * 10**15])
def test_validation_final_and_invalid_seeds_rejected_before_any_environment(seed, monkeypatch):
    monkeypatch.setattr(paired, "RobustPlacementEnv", lambda **kw: pytest.fail("No scenario may be generated"))
    with pytest.raises(ValueError, match="training seeds"):
        paired.stop_return_to_go(seed=seed, profile="normal")


@pytest.mark.parametrize("gamma", [True, .99, 0, float("nan"), float("inf"), "1"])
def test_non_unit_gamma_rejected(gamma):
    with pytest.raises(ValueError, match="gamma=1"):
        paired.stop_return_to_go(seed=0, profile="normal", gamma=gamma)


@pytest.mark.parametrize("profile", ["normal", "stress", "capability", "mixed"])
def test_same_sampled_trajectory_rng_weights_and_reward_as_original_rollout(profile):
    seed = 701
    ordinary_env, paired_env = RobustPlacementEnv(profile=profile), RobustPlacementEnv(profile=profile)
    names = ordinary_env.reset(seed=seed)["feature_names"]
    ordinary_actor, paired_actor = BalancedPolicy(names, seed=83), BalancedPolicy(names, seed=83)
    before = paired_actor.weights_fingerprint()
    expected, expected_info = rollout(ordinary_env, ordinary_actor, seed, training=True)
    actual, actual_info = paired.counterfactual_rollout(paired_env, paired_actor, seed)
    assert len(expected) == len(actual)
    for left, right in zip(expected, actual):
        assert set(left) == set(right) == {"features", "mask", "action", "value", "reward"}
        for key in ("features", "mask"):
            np.testing.assert_array_equal(left[key], right[key])
        assert left["action"] == right["action"] and left["reward"] == right["reward"]
    assert {k: v for k, v in actual_info.items() if k != "counterfactual_baseline"} == expected_info
    assert actual_info["counterfactual_baseline"]["branch_rollouts"] == len(actual)
    assert actual_info["counterfactual_baseline"]["optimal_stop_label"] is False
    assert paired_actor.weights_fingerprint() == ordinary_actor.weights_fingerprint() == before
    assert paired_actor.update_count == ordinary_actor.update_count == 0
    assert paired_actor.rng.bit_generator.state == ordinary_actor.rng.bit_generator.state
    assert paired_env._sequence_rng.bit_generator.state == ordinary_env._sequence_rng.bit_generator.state
    assert paired_env._env._rng.bit_generator.state == ordinary_env._env._rng.bit_generator.state
    assert paired._same_public(paired_env.observe(), ordinary_env.observe())


def test_baseline_attached_after_sampling_and_stop_advantage_is_exactly_zero():
    env = RobustPlacementEnv(seed=0, profile="normal")
    first = env.observe()
    action = best_rf(first)
    actor = ScriptedSampler([action, "stop"])
    records, info = paired.counterfactual_rollout(env, actor, seed=0)
    assert len(records) == 2
    returns = np.cumsum([row["reward"] for row in records][::-1])[::-1]
    advantages = returns - np.array([row["value"] for row in records])
    assert records[0]["value"] == -10.5
    assert advantages[0] == pytest.approx(info["episode_return"] + 10.5)
    assert records[1]["value"] == records[1]["reward"]
    assert advantages[1] == 0
    ordinary = RobustPlacementEnv(seed=0, profile="normal")
    assert paired._same_public(actor.observations[0], ordinary.observe())
    second, _, _, _ = ordinary.step(action)
    assert paired._same_public(actor.observations[1], second)
    assert not any(key in actor.observations[0] for key in ("counterfactual_baseline", "value", "scenario", "targets", "case_metadata"))


def test_forced_stop_record_has_finite_baseline_and_zero_advantage(monkeypatch):
    class ForcedStopEnv(RobustPlacementEnv):
        def reset(self, seed=None, scenario=None, catalogue=None):
            super().reset(seed=seed, scenario=scenario, catalogue=catalogue)
            case = deepcopy(self.scenario)
            case["public"]["available_sensor_ids"] = []
            return super().reset(seed=seed, scenario=case, catalogue=self.catalogue)

    monkeypatch.setattr(paired, "RobustPlacementEnv", ForcedStopEnv)
    env = ForcedStopEnv(seed=10, profile="capability")
    assert not env.observe()["action_mask"][:-1].any()
    records, info = paired.counterfactual_rollout(env, ScriptedSampler(["stop"]), seed=10)
    assert len(records) == 1 and records[0]["value"] == records[0]["reward"] == -10.5
    assert info["counterfactual_baseline"]["branch_rollouts"] == 1


def test_branch_does_not_consume_global_numpy_rng_or_mutate_expected_observation():
    env = RobustPlacementEnv(seed=0, profile="normal")
    observation = env.observe()
    before = canonical_hash(observation)
    global_before = np.random.get_state()
    paired.stop_return_to_go(seed=0, profile="normal", expected_observation=observation)
    assert canonical_hash(observation) == before
    global_after = np.random.get_state()
    assert global_before[0] == global_after[0] and global_before[2:] == global_after[2:]
    np.testing.assert_array_equal(global_before[1], global_after[1])


def test_baseline_does_not_depend_on_next_sampled_action():
    env = RobustPlacementEnv(seed=0, profile="normal")
    action = best_rf(env.observe())
    stop_records, _ = paired.counterfactual_rollout(env, ScriptedSampler(["stop"]), seed=0)
    deploy_records, _ = paired.counterfactual_rollout(env, ScriptedSampler([action, "stop"]), seed=0)
    assert stop_records[0]["value"] == deploy_records[0]["value"] == -10.5


def test_private_case_metadata_only_in_returned_audit_not_record_features():
    env = RobustPlacementEnv(seed=700, profile="mixed")
    actor = ScriptedSampler(["stop"])
    records, info = paired.counterfactual_rollout(env, actor, seed=700)
    assert "case_metadata" in info and "counterfactual_baseline" in info
    assert set(records[0]) == {"features", "mask", "action", "value", "reward"}
    assert "case_metadata" not in json.dumps(actor.observations[0]["state"])


def test_cached_sampler_record_never_receives_private_baseline_or_reward():
    class CachingSampler(ScriptedSampler):
        def __init__(self, actions):
            super().__init__(actions)
            self.retained = []

        def sample(self, observation):
            # Check on the next actor call, not only after rollout completion.
            assert all(row["value"] == 777. and row["reward"] == 0. for row in self.retained)
            action, record = super().sample(observation)
            self.retained.append(record)
            return action, record

    env = RobustPlacementEnv(seed=0, profile="normal")
    actor = CachingSampler([best_rf(env.observe()), "stop"])
    records, _ = paired.counterfactual_rollout(env, actor, seed=0)
    assert len(actor.retained) == len(records) == 2
    for annotated, retained in zip(records, actor.retained):
        assert annotated is not retained
        assert retained["value"] == 777. and retained["reward"] == 0.
        assert annotated["value"] != 777.
        for key in ("features", "mask"):
            assert not np.shares_memory(annotated[key], retained[key])
            np.testing.assert_array_equal(annotated[key], retained[key])
