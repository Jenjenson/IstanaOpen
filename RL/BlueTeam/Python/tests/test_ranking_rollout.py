"""Fixture-only paired ranking branches: public proposals, private labels."""
from copy import deepcopy

import numpy as np
import pytest

from triad_rl import ranking_rollout as ranking
from triad_rl.adaptive_evaluation import GreedyPublicCoverage, canonical_hash
from triad_rl.adaptive_inputs import FEATURE_NAMES, FEATURE_SCHEMA
from triad_rl.balanced_policy import BalancedPolicy
from triad_rl.counterfactual_rollout import _same_public
from triad_rl.robust_scenarios import RobustPlacementEnv


SEED = 499_000_000  # Test fixtures only: never pilot training/validation slots.


def synthetic():
    pairs = [("rf", 0), ("rf", 1), ("rf", 2), ("radar", 0),
             ("radar", 1), ("radar", 2), ("thermal", 2)]
    options = [{"sensor_id": sensor, "sensor_index": ("rf", "radar", "thermal").index(sensor),
                "site_index": site, "position": [float(site), 1.], "stop": False}
               for sensor, site in pairs]
    options.append({"sensor_id": None, "sensor_index": -1, "site_index": -1,
                    "position": [0., 0.], "stop": True})
    x = np.zeros((len(options), len(FEATURE_NAMES)), dtype=np.float64)
    x[:-1, FEATURE_NAMES.index("marginal_coverage")] = np.array([1., .9, .8, .95, .7, .6, .5]) / 3.
    x[-1, FEATURE_NAMES.index("stop")] = 1.
    return {"schema": FEATURE_SCHEMA, "feature_schema": FEATURE_SCHEMA,
            "feature_names": FEATURE_NAMES, "option_features": x,
            "action_mask": np.ones(len(options), dtype=bool), "options": options,
            "state": {"done": False}, "catalogue": []}


class Actor:
    def __init__(self, action=2):
        self.action = action

    def act(self, observation, deterministic=True):
        assert deterministic is True
        return self.action


class V3:
    def probabilities(self, observation):
        result = np.zeros(len(observation["options"]))
        # V3 prefers STOP overall, but its best deployment is still proposed.
        result[-1], result[3] = .9, .1
        return result


def test_exact_recipe_includes_stop_ranker_v3_deployment_distinct_site_and_balanced_random():
    obs = synthetic()
    rng, expected_rng = np.random.default_rng(499), np.random.default_rng(499)
    sensor = ["radar", "thermal"][int(expected_rng.integers(2))]
    rows = [4, 5] if sensor == "radar" else [6]
    random_row = rows[int(expected_rng.integers(len(rows)))]
    slate = ranking.propose_slate(obs, Actor(), V3(), rng=rng)
    assert slate == [7, 0, 2, 3, 1, random_row]
    assert rng.bit_generator.state == expected_rng.bit_generator.state
    assert obs["options"][slate[4]]["site_index"] != obs["options"][slate[1]]["site_index"]


def test_deduplication_then_stable_greedy_fill_and_small_priority_slates():
    obs = synthetic()
    class DuplicateV3:
        def probabilities(self, observation):
            return np.array([1., 0., 0., 0., 0., 0., 0., 0.])
    result = ranking.propose_slate(obs, Actor(0), DuplicateV3(), rng=np.random.default_rng(499))
    assert result[:3] == [7, 0, 1]
    assert len(result) == len(set(result)) == 6
    remaining = [i for i in ranking._greedy_order(obs["option_features"], obs["action_mask"], 7)
                 if i not in result[:4]]
    assert result[4:] == remaining[:2]
    assert ranking.propose_slate(obs, Actor(), V3(), rng=np.random.default_rng(499), max_actions=3) == [7, 0, 2]


def test_greedy_stop_still_offers_deployment_and_distinct_site():
    obs = synthetic()
    obs["option_features"][:, FEATURE_NAMES.index("marginal_coverage")] *= -1
    result = ranking.propose_slate(obs, Actor(), V3(), rng=np.random.default_rng(499))
    assert result[:4] == [7, 2, 3, 4]
    assert len(result) == 6
    assert GreedyPublicCoverage().act(obs) == 7


def test_stop_only_and_one_slot_skip_policy_calls_and_slate_rng():
    class Forbidden:
        def __deepcopy__(self, memo):
            pytest.fail("STOP-only slate must not touch policies")
    obs = synthetic()
    rng = np.random.default_rng(499)
    before = deepcopy(rng.bit_generator.state)
    assert ranking.propose_slate(obs, Forbidden(), Forbidden(), rng=rng, max_actions=1) == [7]
    obs["action_mask"][:-1] = False
    assert ranking.propose_slate(obs, Forbidden(), Forbidden(), rng=rng) == [7]
    assert rng.bit_generator.state == before


def test_mask_and_short_legal_slate():
    obs = synthetic()
    obs["action_mask"][[0, 1, 2, 4, 5, 6]] = False
    assert ranking.propose_slate(obs, Actor(3), V3(), rng=np.random.default_rng(499)) == [7, 3]


@pytest.mark.parametrize("limit", [0, 7, True, 2.5, "6"])
def test_bad_slate_limit(limit):
    with pytest.raises(ValueError, match="max_actions"):
        ranking.propose_slate(synthetic(), Actor(), V3(), rng=np.random.default_rng(499), max_actions=limit)


def test_public_only_detached_policy_inputs_and_actor_rng_cache_isolation():
    calls = []
    class StatefulActor:
        def __init__(self):
            self.rng = np.random.default_rng(499)
            self.cache = []
        def act(self, observation, deterministic=True):
            calls.append(deepcopy(observation))
            self.cache.append(observation)
            self.rng.random()
            observation["state"]["done"] = True
            return 2
    class StatefulV3(StatefulActor):
        def probabilities(self, observation):
            self.act(observation)
            return np.array([0., 0., 0., .1, 0., 0., 0., .9])
    obs, actor, v3 = synthetic(), StatefulActor(), StatefulV3()
    original = deepcopy(obs)
    before = deepcopy(actor.rng.bit_generator.state)
    global_before = np.random.get_state()
    ranking.propose_slate(obs, actor, v3, rng=np.random.default_rng(499))
    assert _same_public(obs, original)
    assert actor.cache == v3.cache == []
    assert actor.rng.bit_generator.state == v3.rng.bit_generator.state == before
    assert len(calls) == 2 and all(_same_public(row, original) for row in calls)
    assert all(set(row) == ranking.PUBLIC_KEYS for row in calls)
    assert np.random.get_state()[0] == global_before[0]
    np.testing.assert_array_equal(np.random.get_state()[1], global_before[1])
    assert np.random.get_state()[2:] == global_before[2:]


@pytest.mark.parametrize("wrap", [False, True])
def test_slate_rng_cannot_alias_actor_rng_or_its_bit_generator(wrap):
    policy = BalancedPolicy(FEATURE_NAMES, seed=499)
    before = deepcopy(policy.rng.bit_generator.state)
    rng = np.random.Generator(policy.rng.bit_generator) if wrap else policy.rng
    with pytest.raises(ValueError, match="separate"):
        ranking.propose_slate(synthetic(), policy, V3(), rng=rng)
    assert policy.rng.bit_generator.state == before


@pytest.mark.parametrize("mutation", ["extra_private", "nonfinite_feature", "integer_mask", "no_stop",
                                       "two_stop", "wrong_stop_feature", "done", "nonfinite_state"])
def test_bad_public_observations_rejected_before_actor(mutation):
    obs = synthetic()
    if mutation == "extra_private":
        obs["scenario"] = {"targets": []}
    elif mutation == "nonfinite_feature":
        obs["option_features"][0, 0] = np.nan
    elif mutation == "integer_mask":
        obs["action_mask"] = obs["action_mask"].astype(int)
    elif mutation == "no_stop":
        obs["action_mask"][-1] = False
    elif mutation == "two_stop":
        obs["options"][0]["stop"] = True
    elif mutation == "wrong_stop_feature":
        obs["option_features"][0, FEATURE_NAMES.index("stop")] = 1
    elif mutation == "done":
        obs["state"]["done"] = True
    else:
        obs["state"]["x"] = float("inf")
    with pytest.raises(ValueError):
        ranking.propose_slate(obs, None, None, rng=np.random.default_rng(499))


@pytest.mark.parametrize("action", [-1, 99, True, 1.5])
def test_invalid_ranker_action(action):
    with pytest.raises(ValueError, match="legal integer"):
        ranking.propose_slate(synthetic(), Actor(action), V3(), rng=np.random.default_rng(499))


@pytest.mark.parametrize("bad", [[1.], [np.nan] * 8, [-1.] + [0.] * 7, [0.] * 8])
def test_invalid_v3_probabilities(bad):
    class Bad:
        def probabilities(self, observation):
            return bad
    with pytest.raises(ValueError, match="probabilities"):
        ranking.propose_slate(synthetic(), Actor(), Bad(), rng=np.random.default_rng(499))


@pytest.mark.parametrize("profile", ["normal", "stress", "capability", "mixed"])
def test_exact_live_branch_parity_full_return_prefix_and_rng_isolation(profile):
    live = RobustPlacementEnv(seed=SEED, profile=profile)
    first = live.observe()
    # A cheap legal deployment ensures a nonterminal committed prefix.
    legal = [i for i in np.flatnonzero(first["action_mask"][:-1])]
    action = min(legal, key=lambda i: first["catalogue"][first["options"][i]["sensor_index"]]["cost"])
    observation, paid, done, _ = live.step(int(action))
    assert not done
    policy = BalancedPolicy(FEATURE_NAMES, seed=499)
    policy_before = deepcopy(policy.__dict__)
    slate = ranking.propose_slate(observation, policy, policy, rng=np.random.default_rng(499))
    assert policy.rng.bit_generator.state == policy_before["rng"].bit_generator.state
    for key in policy.parameters:
        np.testing.assert_array_equal(policy.parameters[key], policy_before["parameters"][key])
    before = canonical_hash(live.observe())
    total = live.total_reward
    rng_states = [deepcopy(live._sequence_rng.bit_generator.state), deepcopy(live._env._rng.bit_generator.state)]
    global_before = np.random.get_state()
    outcomes = ranking.evaluate_slate(seed=SEED, profile=profile, prior_actions=[int(action)],
                                      actions=slate, expected_observation=observation)
    assert canonical_hash(live.observe()) == before and live.total_reward == total == paid
    assert [live._sequence_rng.bit_generator.state, live._env._rng.bit_generator.state] == rng_states
    np.testing.assert_array_equal(np.random.get_state()[1], global_before[1])
    for choice, row in zip(slate, outcomes, strict=True):
        direct = RobustPlacementEnv(seed=SEED, profile=profile)
        direct.step(int(action))
        direct_obs, reward, done, info = direct.step(choice)
        accumulated, steps = paid + reward, 2
        while not done:
            direct_obs, reward, done, info = direct.step(GreedyPublicCoverage().act(direct_obs))
            accumulated += reward
            steps += 1
        assert row == {"action": choice, "timely_fraction": 1. - info["breached_fraction"],
                       "detection_fraction": info["detected_fraction"], "episode_return": accumulated,
                       "cost": info["cost"], "placements": info["placements"], "steps": steps,
                       "continuation_steps": steps - 2}
        assert row["episode_return"] == info["return"]
        assert row["episode_return"] == pytest.approx(sum(info["reward_components"].values()))
    assert set(outcomes[0]) == {"action", "timely_fraction", "detection_fraction", "episode_return",
                                "cost", "placements", "steps", "continuation_steps"}
    assert canonical_hash(observation) == before


def test_stop_label_full_return_not_postprefix_shaped_stop_reward():
    live = RobustPlacementEnv(seed=SEED, profile="normal")
    first = live.observe()
    legal = list(map(int, np.flatnonzero(first["action_mask"][:-1])))
    action = min(legal, key=lambda i: first["catalogue"][first["options"][i]["sensor_index"]]["cost"])
    obs, paid, done, _ = live.step(action)
    assert not done and abs(paid) > .01
    stop = len(obs["options"]) - 1
    row = ranking.evaluate_slate(seed=SEED, profile="normal", prior_actions=[action],
                                 actions=[stop], expected_observation=obs)[0]
    _, immediate, done, info = live.step(stop)
    assert done and row["episode_return"] == info["return"] == paid + immediate
    assert row["episode_return"] != pytest.approx(immediate)
    assert row["continuation_steps"] == 0


def test_each_action_uses_a_fresh_case_and_identical_prefix(monkeypatch):
    made = []
    class AuditedEnv(RobustPlacementEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.actions = []
            made.append(self)
        def step(self, action):
            self.actions.append(action)
            return super().step(action)
    live = RobustPlacementEnv(seed=SEED, profile="normal")
    initial = live.observe()
    legal = list(map(int, np.flatnonzero(initial["action_mask"][:-1])))
    first = min(legal, key=lambda i: initial["catalogue"][initial["options"][i]["sensor_index"]]["cost"])
    obs, _, done, _ = live.step(first)
    assert not done
    choices = [len(obs["options"]) - 1, int(np.flatnonzero(obs["action_mask"][:-1])[0])]
    monkeypatch.setattr(ranking, "RobustPlacementEnv", AuditedEnv)
    outcomes = ranking.evaluate_slate(seed=SEED, profile="normal", prior_actions=[first],
                                      actions=choices, expected_observation=obs)
    assert len(made) == len(outcomes) == len(choices)
    for branch, choice in zip(made, choices, strict=True):
        assert branch.actions[:2] == [first, choice]
        assert branch.done
        assert branch.scenario["seed"] == SEED


def test_forced_stop_only_label_and_no_actor_cache_receives_results(monkeypatch):
    class ForcedStopEnv(RobustPlacementEnv):
        def reset(self, seed=None, scenario=None, catalogue=None):
            super().reset(seed=seed, scenario=scenario, catalogue=catalogue)
            case = deepcopy(self.scenario)
            case["public"]["available_sensor_ids"] = []
            return super().reset(seed=seed, scenario=case, catalogue=self.catalogue)
    monkeypatch.setattr(ranking, "RobustPlacementEnv", ForcedStopEnv)
    obs = ForcedStopEnv(seed=SEED, profile="normal").observe()
    before = deepcopy(obs)
    slate = ranking.propose_slate(obs, None, None, rng=np.random.default_rng(499))
    result = ranking.evaluate_slate(seed=SEED, profile="normal", prior_actions=[],
                                    actions=slate, expected_observation=obs)
    assert result == [{"action": len(obs["options"]) - 1, "timely_fraction": 0.,
                       "detection_fraction": 0., "episode_return": -10.5, "cost": 0.,
                       "placements": [], "steps": 1, "continuation_steps": 0}]
    assert ranking.comparisons(result) == []
    assert _same_public(obs, before)


def test_real_ranker_api_greedy_initialization_and_nonmutating_proposal():
    from triad_rl.ranking_policy import RankPolicy
    obs = RobustPlacementEnv(seed=SEED, profile="capability").observe()
    policy, v3 = RankPolicy(FEATURE_NAMES, seed=499), BalancedPolicy(FEATURE_NAMES, seed=499)
    before = policy.weights_fingerprint(), deepcopy(policy.rng.bit_generator.state)
    choice = policy.act(obs, deterministic=True)
    assert choice == GreedyPublicCoverage().act(obs)
    slate = ranking.propose_slate(obs, policy, v3, rng=np.random.default_rng(499))
    assert choice in slate
    assert (policy.weights_fingerprint(), policy.rng.bit_generator.state) == before


@pytest.mark.parametrize("seed", [True, -1, .5, "499", 10**15, 2 * 10**15])
def test_invalid_or_heldout_seed_rejected_before_environment(seed, monkeypatch):
    monkeypatch.setattr(ranking, "RobustPlacementEnv", lambda **kw: pytest.fail("No scenario generation"))
    with pytest.raises(ValueError, match="training seed"):
        ranking.evaluate_slate(seed=seed, profile="normal", prior_actions=[], actions=[7],
                               expected_observation=synthetic())


@pytest.mark.parametrize("actions", [[], [0, 0], [True], [-1], [999999], list(range(7)), "1"])
def test_illegal_or_duplicate_slate_fails_before_branches(actions, monkeypatch):
    monkeypatch.setattr(ranking, "RobustPlacementEnv", lambda **kw: pytest.fail("No scenario generation"))
    with pytest.raises(ValueError):
        ranking.evaluate_slate(seed=SEED, profile="normal", prior_actions=[], actions=actions,
                               expected_observation=synthetic())


@pytest.mark.parametrize("prefix", [[0, 0], [-1], [True], "0", list(range(32))])
def test_invalid_prefix_structure_before_branch(prefix, monkeypatch):
    monkeypatch.setattr(ranking, "RobustPlacementEnv", lambda **kw: pytest.fail("No scenario generation"))
    with pytest.raises(ValueError):
        ranking.evaluate_slate(seed=SEED, profile="normal", prior_actions=prefix, actions=[7],
                               expected_observation=synthetic())


def test_prefix_stop_masked_site_and_terminal_exhaustion():
    env = RobustPlacementEnv(seed=SEED, profile="normal")
    initial = env.observe()
    stop = len(initial["options"]) - 1
    with pytest.raises(ValueError, match="terminal STOP"):
        ranking.evaluate_slate(seed=SEED, profile="normal", prior_actions=[stop], actions=[stop],
                               expected_observation=initial)
    prefix = []
    obs = initial
    while True:
        action = int(np.flatnonzero(obs["action_mask"][:-1])[0])
        prefix.append(action)
        obs, _, done, _ = env.step(action)
        if done:
            break
    with pytest.raises(ValueError, match="already terminated"):
        ranking.evaluate_slate(seed=SEED, profile="normal", prior_actions=prefix, actions=[stop],
                               expected_observation=initial)
    # A different sensor at the already-occupied site is also illegal.
    first = prefix[0]
    same_site = next(i for i, opt in enumerate(initial["options"])
                     if i != first and not opt["stop"] and initial["action_mask"][i]
                     and opt["site_index"] == initial["options"][first]["site_index"])
    with pytest.raises(ValueError, match="legal integer"):
        ranking.evaluate_slate(seed=SEED, profile="normal", prior_actions=[first, same_site],
                               actions=[stop], expected_observation=initial)


@pytest.mark.parametrize("drift", ["seed", "profile", "value", "dtype", "extra"])
def test_exact_expected_observation_guard(drift):
    obs = RobustPlacementEnv(seed=SEED, profile="normal").observe()
    seed, profile = SEED, "normal"
    if drift == "seed":
        seed += 1
    elif drift == "profile":
        profile = "stress"
    elif drift == "value":
        obs["state"]["budget_remaining"] = np.nextafter(obs["state"]["budget_remaining"], np.inf).item()
    elif drift == "dtype":
        obs["option_features"] = obs["option_features"].astype(np.float64)
    else:
        obs["training_label"] = 1
    with pytest.raises(ValueError, match="match|public observation"):
        ranking.evaluate_slate(seed=seed, profile=profile, prior_actions=[], actions=[len(obs["options"]) - 1],
                               expected_observation=obs)


def test_nonterminating_branch_bound(monkeypatch):
    obs = synthetic()
    class NeverDone:
        steps = 0
        def __init__(self, **kwargs):
            pass
        def observe(self):
            return deepcopy(obs)
        def step(self, action):
            type(self).steps += 1
            return deepcopy(obs), 0., False, {}
    monkeypatch.setattr(ranking, "RobustPlacementEnv", NeverDone)
    with pytest.raises(RuntimeError, match="bounded"):
        ranking.evaluate_slate(seed=SEED, profile="normal", prior_actions=[], actions=[0], expected_observation=obs)
    assert NeverDone.steps == ranking.MAX_STEPS


def outcome(action, timely=.5, detection=.8, value=2., cost=1.):
    return {"action": action, "timely_fraction": timely, "detection_fraction": detection,
            "episode_return": value, "cost": cost}


def test_lexicographic_priority_not_weighted_reward_or_cost_or_global_action_indices():
    rows = [outcome(900, timely=.5, detection=0., value=-1000., cost=4.),
            outcome(100, timely=.4, detection=1., value=1000., cost=.1)]
    assert ranking.comparisons(rows) == [{"preferred": 0, "rejected": 1, "weight": pytest.approx(.1)}]
    rows[1]["timely_fraction"] = .5
    assert ranking.comparisons(rows) == [{"preferred": 1, "rejected": 0, "weight": .1}]
    rows[0]["detection_fraction"] = 1.
    assert ranking.comparisons(rows) == [{"preferred": 1, "rejected": 0, "weight": .01}]


@pytest.mark.parametrize("delta,weight", [(10., .005), (20., .01), (100., .01)])
def test_return_tier_scale_and_cap(delta, weight):
    assert ranking.comparisons([outcome(1, value=0.), outcome(2, value=delta)]) == [
        {"preferred": 1, "rejected": 0, "weight": weight}]


def test_exact_ties_omit_and_tiny_timely_difference_never_uses_detection_tiebreak():
    assert ranking.comparisons([outcome(10), outcome(20, cost=100.)]) == []
    a, b = outcome(10, timely=.5, detection=0.), outcome(20, timely=np.nextafter(.5, 0.), detection=1.)
    assert ranking.comparisons([a, b]) == [{"preferred": 0, "rejected": 1,
                                           "weight": .5 - np.nextafter(.5, 0.)}]


def test_all_pairs_permutation_consistency_and_input_immutability():
    rows = [outcome(100 + i, timely=i / 6.) for i in range(6)]
    before = deepcopy(rows)
    pairs = ranking.comparisons(rows)
    assert len(pairs) == 15 and rows == before
    reverse = list(reversed(rows))
    mapped = lambda data, result: {(data[p["preferred"]]["action"], data[p["rejected"]]["action"], p["weight"])
                                    for p in result}
    assert mapped(rows, pairs) == mapped(reverse, ranking.comparisons(reverse))


@pytest.mark.parametrize("key,value", [("action", True), ("action", -1), ("action", ranking.MAX_OPTIONS),
                                      ("timely_fraction", np.nan), ("timely_fraction", 1.1),
                                      ("detection_fraction", -.1), ("detection_fraction", True),
                                      ("episode_return", np.inf), ("episode_return", "1"),
                                      ("cost", -1.)])
def test_invalid_outcomes(key, value):
    row = outcome(1)
    row[key] = value
    with pytest.raises(ValueError):
        ranking.comparisons([row, outcome(2)])


def test_bad_outcome_count_duplicates_and_overflow_difference():
    for rows in ([], [outcome(1), outcome(1)], [outcome(i) for i in range(7)],
                 [outcome(1, value=1e308), outcome(2, value=-1e308)]):
        with pytest.raises(ValueError):
            ranking.comparisons(rows)
