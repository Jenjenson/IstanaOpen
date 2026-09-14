"""Anchored credit arithmetic and fixture-only rollout/update parity."""
from copy import deepcopy

import numpy as np
import pytest

from triad_rl import anchored_credit as anchored
from triad_rl.balanced_policy import BalancedPolicy
from triad_rl.counterfactual_rollout import _same_public
from triad_rl.robust_scenarios import RobustPlacementEnv
from triad_rl.train_robust import rollout as ordinary_rollout


NAMES = ("x", "stop", "y")
SEED = 499_000_000


def observation(mask=(True, False, True)):
    return {"feature_names": NAMES, "option_features": np.array([[.7, 0., -.4], [.2, 0., .5], [0., 1., .3]]),
            "action_mask": np.array(mask, dtype=bool)}


def record(reward=1., value=.2, *, action=0, baseline=None):
    obs = observation()
    row = {"features": obs["option_features"], "mask": obs["action_mask"],
           "action": action, "value": value, "reward": reward}
    if baseline is not None:
        row[anchored.CREDIT_FIELD] = baseline
    return row


def state(policy):
    return {"count": policy.update_count, "rng": deepcopy(policy.rng.bit_generator.state),
            "parameters": {key: value.tobytes() for key, value in policy.parameters.items()},
            "adam_m": {key: value.tobytes() for key, value in policy.adam_m.items()},
            "adam_v": {key: value.tobytes() for key, value in policy.adam_v.items()},
            "metadata": deepcopy(policy.metadata), "training_state": deepcopy(policy.training_state)}


def batch():
    return [[record(2., 1.), record(3., 4., action=2)], [record(4., 2.)],
            [record(-2., 1.), record(1., -.5), record(-.5, -1., action=2)]]


@pytest.mark.parametrize("warm_optimizer", [False, True])
@pytest.mark.parametrize("entropy,clip", [(0., 1.), (.015, .001), (.1, 100.)])
def test_no_substitution_exact_frozen_update_all_parameters_adam_rng_and_metrics(warm_optimizer, entropy, clip):
    frozen = BalancedPolicy(NAMES, seed=499, hidden_size=5)
    episodes = batch()
    if warm_optimizer:
        frozen.update(episodes, gamma=1., value_coef=.5)
        assert frozen.adam_m["wv"].any()
    policy = deepcopy(frozen)
    records_before = deepcopy(episodes)
    for _ in range(3):
        expected = frozen.update(episodes, learning_rate=.002, entropy_coef=entropy,
                                 gamma=1., value_coef=0., max_grad_norm=clip)
        actual = anchored.update(policy, episodes, learning_rate=.002, entropy_coef=entropy,
                                 gamma=1., value_coef=0., max_grad_norm=clip)
        assert actual == expected
        assert state(policy) == state(frozen)
    assert _same_public(episodes, records_before)


def test_reference_anchor_calculated_once_and_first_coefficients_bit_exact():
    policy = BalancedPolicy(NAMES, seed=499)
    episodes = [[record(2., 1.), record(3., 4., baseline=5., action=2)], [record(4., 2.)]]
    prepared = anchored.prepare_batch(policy, episodes)
    returns = np.array([5., 3., 4.])
    reference_raw, applied_raw = np.array([4., -1., 2.]), np.array([4., -2., 2.])
    denominator = reference_raw.std() + 1e-8
    np.testing.assert_array_equal(prepared["returns"], returns)
    np.testing.assert_array_equal(prepared["reference_raw_advantages"], reference_raw)
    np.testing.assert_array_equal(prepared["credit_raw_advantages"], applied_raw)
    np.testing.assert_array_equal(prepared["reference_advantages"], (reference_raw-reference_raw.mean()) / denominator)
    np.testing.assert_array_equal(prepared["advantages"], (applied_raw-reference_raw.mean()) / denominator)
    first = prepared["first_mask"]
    assert first.tolist() == [True, False, True]
    assert prepared["advantages"][first].tobytes() == prepared["reference_advantages"][first].tobytes()
    assert abs(prepared["advantages"].mean()) > .1  # No second recentering.
    assert prepared["normalizer"]["reference_mean"] == reference_raw.mean()


@pytest.mark.parametrize("small", [0., 1e-10])
def test_degenerate_reference_keeps_applied_raw_credit_without_new_normalization(small):
    policy = BalancedPolicy(NAMES, seed=499)
    episodes = [[record(0., 0.), record(2., small, baseline=-3.)]]
    result = anchored.prepare_batch(policy, episodes)
    assert result["normalizer"]["applied"] is False
    np.testing.assert_array_equal(result["advantages"], [2., 5.])
    np.testing.assert_array_equal(result["reference_advantages"], [2., 2.-small])
    assert result["normalizer"]["center"] == 0. and result["normalizer"]["denominator"] == 1.


def test_single_transition_control_exact_and_no_forced_centering():
    first = BalancedPolicy(NAMES, seed=499)
    second = deepcopy(first)
    episodes = [[record(2., -3., action=2)]]
    assert anchored.prepare_batch(first, episodes)["advantages"].tolist() == [5.]
    assert anchored.update(first, episodes) == second.update(episodes, learning_rate=.002, gamma=1., value_coef=0.)
    assert state(first) == state(second)


def test_diagnostics_are_read_only_group_moments_match_applied_and_reference_coefficients():
    policy = BalancedPolicy(NAMES, seed=499)
    episodes = [[record(2., 1.), record(3., 4., baseline=3., action=2)], [record(4., 2., action=2)]]
    before_policy, before_batch = state(policy), deepcopy(episodes)
    result = anchored.credit_diagnostics(policy, episodes)
    prepared = anchored.prepare_batch(policy, episodes)
    assert state(policy) == before_policy and _same_public(episodes, before_batch)
    assert result["first_coefficients_identical"] is True and set(result["groups"]) == set(anchored.GROUPS)
    assert result["groups"]["all"]["normalized_advantage"]["mean"] == prepared["advantages"].mean()
    assert result["groups"]["all"]["reference_normalized_advantage"]["variance"] == prepared["reference_advantages"].var()
    assert result["groups"]["later_stop"]["raw_advantage"]["mean"] == 0.
    assert result["groups"]["later_stop"]["normalized_advantage"]["mean"] != 0.
    assert result["groups"]["later_deploy"]["raw_advantage"] == {
        "count": 0, "mean": None, "variance": None, "negative": 0, "positive": 0, "zero": 0}
    assert result["groups"]["later_stop"]["baseline"]["mean"] == 4.
    assert result["groups"]["later_stop"]["credit_baseline"]["mean"] == 3.
    prepared["transitions"][0]["features"][:] = 99
    prepared["transitions"][0]["mask"][:] = False
    assert _same_public(episodes, before_batch)


def test_applied_gradient_matches_finite_difference_with_frozen_coefficients():
    policy = BalancedPolicy(NAMES, seed=499, hidden_size=3)
    episodes = [[record(2., 1.), record(-1., 2., baseline=-4., action=2)]]
    prepared = anchored.prepare_batch(policy, episodes)
    args = (prepared["transitions"], prepared["returns"], prepared["advantages"], .015, 0.)
    _, gradients, _ = policy._loss_and_gradients(*args)
    epsilon = 1e-6
    for key, parameter in policy.parameters.items():
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + epsilon
            positive = policy._loss_and_gradients(*args)[0]
            parameter[index] = original - epsilon
            negative = policy._loss_and_gradients(*args)[0]
            parameter[index] = original
            assert gradients[key][index] == pytest.approx((positive-negative)/(2*epsilon), abs=2e-8)


@pytest.mark.parametrize("change", ["masked", "boolean_action", "out_of_range", "nan_feature", "invalid_stop",
                                    "nan_reward", "nan_value", "infinite_credit", "boolean_credit", "changed_first"])
def test_invalid_batch_rejected_before_any_optimizer_or_rng_mutation(change):
    policy = BalancedPolicy(NAMES, seed=499)
    episodes = [[record(), record()]]
    row = episodes[0][1]
    if change == "masked": row["action"] = 1
    elif change == "boolean_action": row["action"] = True
    elif change == "out_of_range": row["action"] = 9
    elif change == "nan_feature": row["features"][0, 0] = np.nan
    elif change == "invalid_stop": row["features"][2, 1] = 0.
    elif change == "nan_reward": row["reward"] = np.nan
    elif change == "nan_value": row["value"] = np.nan
    elif change == "infinite_credit": row[anchored.CREDIT_FIELD] = np.inf
    elif change == "boolean_credit": row[anchored.CREDIT_FIELD] = False
    else: episodes[0][0][anchored.CREDIT_FIELD] = 100.
    before = state(policy)
    with pytest.raises(ValueError):
        anchored.update(policy, episodes)
    assert state(policy) == before


@pytest.mark.parametrize("setting,value", [("gamma", .99), ("gamma", True), ("gamma", np.nan),
                                          ("value_coef", .5), ("value_coef", False),
                                          ("learning_rate", 0.), ("learning_rate", np.inf),
                                          ("entropy_coef", -1.), ("max_grad_norm", 0.)])
def test_invalid_hyperparameters_fail_without_mutation(setting, value):
    policy = BalancedPolicy(NAMES, seed=499)
    before = state(policy)
    with pytest.raises(ValueError):
        anchored.update(policy, batch(), **{setting: value})
    assert state(policy) == before


@pytest.mark.parametrize("episodes", [[], [[]], [[record()], []]])
def test_empty_batch_or_episode_rejected(episodes):
    policy = BalancedPolicy(NAMES, seed=499)
    before = state(policy)
    with pytest.raises(ValueError):
        anchored.update(policy, episodes)
    assert state(policy) == before


@pytest.mark.parametrize("profile", ["normal", "stress", "capability", "mixed"])
def test_real_fixture_trajectory_live_and_actor_rng_match_ordinary_rollout(profile):
    ordinary_env = RobustPlacementEnv(seed=SEED, profile=profile)
    env = RobustPlacementEnv(seed=SEED, profile=profile)
    first = BalancedPolicy(ordinary_env.observe()["feature_names"], seed=499)
    second = deepcopy(first)
    before = state(second)
    expected, expected_info = ordinary_rollout(ordinary_env, first, SEED, training=True)
    actual, actual_info = anchored.rollout(env, second, SEED)
    assert len(expected) == len(actual)
    for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
        assert _same_public(left, {k:v for k,v in right.items() if k != anchored.CREDIT_FIELD})
        assert (anchored.CREDIT_FIELD in right) is (index > 0)
    assert actual_info["anchored_credit"]["branch_rollouts"] == len(actual)-1
    assert {k:v for k,v in actual_info.items() if k != "anchored_credit"} == expected_info
    assert state(first) == state(second)
    assert before["parameters"] == state(second)["parameters"]
    assert env._sequence_rng.bit_generator.state == ordinary_env._sequence_rng.bit_generator.state
    assert env._env._rng.bit_generator.state == ordinary_env._env._rng.bit_generator.state
    assert _same_public(env.observe(), ordinary_env.observe())


class CachedSampler:
    def __init__(self, first_stop=False):
        self.first_stop, self.observations, self.retained = first_stop, [], []

    def sample(self, observation):
        assert all(anchored.CREDIT_FIELD not in row and row["value"] == 777. and row["reward"] == 0. for row in self.retained)
        self.observations.append(deepcopy(observation))
        if self.first_stop or self.retained:
            action = next(i for i,o in enumerate(observation["options"]) if o["stop"])
        else:
            legal = [i for i,o in enumerate(observation["options"]) if not o["stop"] and observation["action_mask"][i]]
            column = observation["feature_names"].index("cost")
            action = min(legal, key=lambda i: observation["option_features"][i, column])
        row = {"features": observation["option_features"].copy(), "mask": observation["action_mask"].copy(),
               "action": action, "value": 777., "reward": 0.}
        self.retained.append(row)
        return action, row


def test_later_branch_only_after_sample_preserves_caches_and_public_inputs(monkeypatch):
    env = RobustPlacementEnv(seed=SEED, profile="normal")
    sampler = CachedSampler()
    real_branch = anchored.stop_return_to_go
    calls = []
    def audited_branch(**kwargs):
        assert len(sampler.retained) == 2  # Both sample calls occurred; no first branch.
        assert len(kwargs["prior_actions"]) == 1
        assert _same_public(kwargs["expected_observation"], sampler.observations[1])
        calls.append(kwargs["prior_actions"].copy())
        return real_branch(**kwargs)
    monkeypatch.setattr(anchored, "stop_return_to_go", audited_branch)
    global_before = np.random.get_state()
    records, info = anchored.rollout(env, sampler, SEED)
    global_after = np.random.get_state()
    assert len(records) == 2 and len(calls) == info["anchored_credit"]["branch_rollouts"] == 1
    assert anchored.CREDIT_FIELD not in records[0]
    assert records[1][anchored.CREDIT_FIELD] == records[1]["reward"]
    assert records[0]["value"] == records[1]["value"] == 777.
    assert global_before[0] == global_after[0] and global_before[2:] == global_after[2:]
    np.testing.assert_array_equal(global_before[1], global_after[1])
    for detached, retained in zip(records, sampler.retained, strict=True):
        assert detached is not retained and anchored.CREDIT_FIELD not in retained
        assert retained["reward"] == 0. and retained["value"] == 777.
        for key in ("features", "mask"):
            assert not np.shares_memory(detached[key], retained[key])
    ordinary = RobustPlacementEnv(seed=SEED, profile="normal")
    assert _same_public(sampler.observations[0], ordinary.observe())
    second, _, _, _ = ordinary.step(records[0]["action"])
    assert _same_public(sampler.observations[1], second)
    assert not any(key in second for key in (anchored.CREDIT_FIELD, "value", "scenario", "targets", "case_metadata"))


def test_first_stop_needs_no_branch_and_keeps_original_critic(monkeypatch):
    monkeypatch.setattr(anchored, "stop_return_to_go", lambda **kw: pytest.fail("First decision must not branch"))
    env = RobustPlacementEnv(seed=SEED, profile="normal")
    records, info = anchored.rollout(env, CachedSampler(first_stop=True), SEED)
    assert len(records) == 1 and anchored.CREDIT_FIELD not in records[0]
    assert records[0]["value"] == 777. and records[0]["reward"] == -10.5
    assert info["anchored_credit"]["branch_rollouts"] == 0


def test_forced_stop_only_mask_is_valid_and_has_no_actor_gradient():
    policy = BalancedPolicy(NAMES, seed=499)
    row = record(2., 1., action=2)
    row["mask"][:] = [False, False, True]
    before = {key:value.copy() for key,value in policy.parameters.items()}
    metrics = anchored.update(policy, [[row]])
    assert metrics["gradient_norm"] == 0. and metrics["stop_probability"] == 1.
    for key in before:
        np.testing.assert_array_equal(policy.parameters[key], before[key])


@pytest.mark.parametrize("seed", [True, -1, .5, 10**15, 2*10**15])
def test_nontraining_seed_rejected_before_reset_or_sample(seed):
    class NeverEnv:
        def reset(self, **kw): pytest.fail("Invalid seed must not reset an environment")
    with pytest.raises(ValueError):
        anchored.rollout(NeverEnv(), CachedSampler(), seed)


def test_sampler_cannot_supply_private_credit_baseline():
    class BadSampler(CachedSampler):
        def sample(self, observation):
            action, row = super().sample(observation)
            row[anchored.CREDIT_FIELD] = 0.
            return action, row
    env = RobustPlacementEnv(seed=SEED, profile="normal")
    with pytest.raises(ValueError, match="Sampler must not supply"):
        anchored.rollout(env, BadSampler(first_stop=True), SEED)


@pytest.mark.parametrize("group,key,value", [("parameters", "wv", np.nan), ("adam_m", "wa", np.inf),
                                           ("adam_v", "wa", -1.)])
def test_invalid_parameter_or_optimizer_state_fails_before_mutation(group, key, value):
    policy = BalancedPolicy(NAMES, seed=499)
    getattr(policy, group)[key][0] = value
    before = state(policy)
    with pytest.raises(ValueError, match="optimizer state"):
        anchored.update(policy, batch())
    assert state(policy) == before


def test_anchored_actor_changes_without_critic_head_gradient_and_keeps_existing_checkpoint_schema(tmp_path):
    policy = BalancedPolicy(NAMES, seed=499)
    episodes = [[record(2., 1.), record(-1., 2., baseline=-4., action=2)]]
    initial = deepcopy(policy)
    anchored.update(policy, episodes)
    assert not np.array_equal(policy.parameters["w1"], initial.parameters["w1"])
    for key in ("wv", "bv"):
        np.testing.assert_array_equal(policy.parameters[key], initial.parameters[key])
        assert not policy.adam_m[key].any() and not policy.adam_v[key].any()
    policy.save(tmp_path / "unchanged-balanced-schema")
    restored = BalancedPolicy.load(tmp_path / "unchanged-balanced-schema")
    assert anchored.update(policy, episodes) == anchored.update(restored, episodes)
    assert state(policy) == state(restored)


def test_nonfinite_later_branch_reward_is_never_attached_to_sampler_cache(monkeypatch):
    sampler = CachedSampler()
    monkeypatch.setattr(anchored, "stop_return_to_go", lambda **kw: np.inf)
    env = RobustPlacementEnv(seed=SEED, profile="normal")
    with pytest.raises(ValueError, match="Paired STOP baseline"):
        anchored.rollout(env, sampler, SEED)
    assert all(anchored.CREDIT_FIELD not in row for row in sampler.retained)
