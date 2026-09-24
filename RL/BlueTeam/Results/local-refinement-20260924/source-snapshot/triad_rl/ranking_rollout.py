"""Training-only paired deployment-ranking labels, never an inference oracle.

Proposals use public observations only. Each label independently recreates the
same training case and committed prefix, tries one proposed action, and follows
the unchanged GreedyPublicCoverage policy to termination. These simulator
outcomes are detached training targets; they must never choose the live action
or enter an actor observation/cache. This is bounded, greedy-continuation
approximate policy improvement, not an optimal-layout or real-feed oracle.

All comparisons are lexicographic: timely fraction, then detection fraction,
then the original full episode return (including cost/redundancy penalties).
Only exact float equality advances to the next tier. Prefix rewards are included
in episode_return; no shaped return-to-go is mistaken for a full return.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np

from .adaptive_evaluation import GreedyPublicCoverage, canonical_hash
from .adaptive_inputs import FEATURE_NAMES, FEATURE_SCHEMA
from .counterfactual_rollout import TRAINING_SEED_LIMIT, _legal_action, _same_public
from .robust_scenarios import PROFILES, RobustPlacementEnv


RANKING_ROLLOUT_SCHEMA = "triad.ranking_rollout.v1"
MAX_ACTIONS = 6
MAX_STEPS = 32
MAX_OPTIONS = 32 * 512 + 1
PUBLIC_KEYS = frozenset(("schema", "feature_schema", "feature_names", "option_features",
                         "action_mask", "options", "state", "catalogue"))
SLATE_CONTRACT = (
    "STOP; greedy overall best; current ranker deterministic best; v3 best legal "
    "deployment; greedy best deployment at a different site_index from greedy's "
    "best deployment; one remaining deployment sampled uniformly by remaining "
    "sensor ID then uniformly by row; deduplicate and fill by stable greedy utility"
)
COMPARISON_CONTRACT = {
    "priority": ["timely_fraction", "detection_fraction", "episode_return"],
    "ties": "exact float equality; omit all-three ties",
    "weights": ["abs(delta_timely)", "0.1*abs(delta_detection)",
                "0.01*min(abs(delta_return)/20, 1)"],
    "indices": "local indices into the ordered outcomes/slate, not global action rows",
}


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(value)


def _sequence(value: Any, name: str, low: int, high: int) -> Sequence:
    if (isinstance(value, (str, bytes)) or not isinstance(value, Sequence)
            or not low <= len(value) <= high):
        raise ValueError(f"{name} must be a sequence of length {low}..{high}")
    return value


def _observation(observation: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, int]:
    if not isinstance(observation, Mapping) or set(observation) != PUBLIC_KEYS:
        raise ValueError("Ranking requires the complete public observation, without extra fields")
    if (observation["schema"] != FEATURE_SCHEMA or observation["feature_schema"] != FEATURE_SCHEMA
            or tuple(observation["feature_names"]) != FEATURE_NAMES):
        raise ValueError("Ranking public feature schema/name/order differs")
    x, mask = np.asarray(observation["option_features"]), np.asarray(observation["action_mask"])
    if (x.dtype.kind != "f" or x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES)
            or not 1 <= x.shape[0] <= MAX_OPTIONS or not np.isfinite(x).all()
            or mask.dtype != np.bool_ or mask.shape != (len(x),) or not mask.any()):
        raise ValueError("Ranking requires finite feature rows and a legal boolean mask")
    options = _sequence(observation["options"], "options", len(x), len(x))
    for option in options:
        if (not isinstance(option, Mapping)
                or set(option) != {"sensor_id", "sensor_index", "site_index", "position", "stop"}
                or type(option["stop"]) is not bool):
            raise ValueError("Ranking option must use the public option contract")
        position = np.asarray(option["position"])
        if position.shape != (2,) or position.dtype.kind not in "fi" or not np.isfinite(position).all():
            raise ValueError("Ranking option position must be finite")
        if not option["stop"]:
            if not isinstance(option["sensor_id"], str) or not option["sensor_id"]:
                raise ValueError("Deployment requires a sensor ID")
            _integer(option["sensor_index"], "sensor_index", 0, 31)
            _integer(option["site_index"], "site_index", 0, 511)
    stops = [i for i, option in enumerate(options) if option["stop"]]
    if stops != [len(x) - 1] or not mask[-1]:
        raise ValueError("Ranking requires one legal final STOP row, matching frozen greedy")
    stop = stops[0]
    expected_stop = np.zeros(len(x))
    expected_stop[stop] = 1.
    if not np.array_equal(x[:, FEATURE_NAMES.index("stop")], expected_stop):
        raise ValueError("Named STOP feature disagrees with options")
    if not isinstance(observation["state"], Mapping) or observation["state"].get("done") is not False:
        raise ValueError("Ranking requires a nonterminal public state")
    try:
        canonical_hash(observation)  # Reject nonfinite public JSON outside the feature matrix too.
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Ranking public observation must be finite JSON-compatible data") from error
    return x, mask, stop


def _greedy_order(features: np.ndarray, mask: np.ndarray, stop: int) -> list[int]:
    # Exact coefficients and accumulation order of frozen GreedyPublicCoverage.
    utility = sum(weight * features[:, FEATURE_NAMES.index(name)].astype(float)
                  for name, weight in (("marginal_coverage", 3.), ("marginal_early_coverage", 1.),
                                       ("cost", -.25), ("overlap", -.15)))
    utility[stop] = 0.
    if not np.isfinite(utility).all():
        raise ValueError("Nonfinite greedy utility")
    return sorted(map(int, np.flatnonzero(mask)), key=lambda i: (-utility[i], i))


def propose_slate(observation: Mapping[str, Any], policy: Any, v3: Any, *,
                  rng: np.random.Generator, max_actions: int = MAX_ACTIONS) -> list[int]:
    """Offer up to six unique legal alternatives, always including learned STOP.

    The explicit slate RNG alone advances. Policy calls run on detached copies,
    so even a stateful deterministic implementation cannot change the caller's
    actor RNG/cache or observation. No environment or private outcome is used.
    Smaller slates truncate recipe priority; production uses all six slots.
    """
    limit = _integer(max_actions, "max_actions", 1, MAX_ACTIONS)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("Slates require an explicit numpy Generator")
    if any(isinstance(getattr(actor, "rng", None), np.random.Generator)
           and actor.rng.bit_generator is rng.bit_generator for actor in (policy, v3)):
        raise ValueError("Slate RNG must be separate from each actor RNG")
    x, mask, stop = _observation(observation)
    ordered = _greedy_order(x, mask, stop)
    deployments = [i for i in ordered if i != stop]
    if not deployments or limit == 1:
        return [stop]
    # Defensive copies preserve policy state as well as input data. No labels
    # are ever attached to these observations, including retained copies.
    rank_action = _legal_action(observation, deepcopy(policy).act(deepcopy(observation), deterministic=True))
    probability = np.asarray(deepcopy(v3).probabilities(deepcopy(observation)), dtype=float)
    if (probability.shape != mask.shape or not np.isfinite(probability).all()
            or (probability < 0).any() or (probability[~mask] != 0).any()
            or not np.isclose(probability.sum(), 1., rtol=0., atol=1e-12)):
        raise ValueError("v3 must return finite legal joint probabilities")
    v3_deployment = max(sorted(deployments), key=lambda i: probability[i])
    greedy_site = observation["options"][deployments[0]]["site_index"]
    different_site = next((i for i in deployments
                           if observation["options"][i]["site_index"] != greedy_site), None)
    result: list[int] = []

    def add(action: int | None) -> None:
        if action is not None and action not in result and len(result) < limit:
            result.append(action)

    for action in (stop, ordered[0], rank_action, v3_deployment, different_site):
        add(action)
    remaining = [i for i in deployments if i not in result]
    if remaining and len(result) < limit:
        sensors = sorted({observation["options"][i]["sensor_id"] for i in remaining})
        sensor = sensors[int(rng.integers(len(sensors)))]
        rows = sorted(i for i in remaining if observation["options"][i]["sensor_id"] == sensor)
        add(rows[int(rng.integers(len(rows)))])
    for action in ordered:
        add(action)
    return result


def evaluate_slate(*, seed: int, profile: str, prior_actions: Sequence[int],
                   actions: Sequence[int], expected_observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Produce private training labels from independent seeded greedy branches.

    No live environment or policy is accepted. Every branch replays the full
    prefix and verifies the entire public observation exactly before trying its
    slate action. Validation/final-test namespaces and terminal prefixes fail.
    """
    seed = _integer(seed, "training seed", 0, TRAINING_SEED_LIMIT - 1)
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError("Invalid robust training profile")
    _observation(expected_observation)
    prefix = [_integer(i, "prefix action", 0, MAX_OPTIONS - 1)
              for i in _sequence(prior_actions, "prior_actions", 0, MAX_STEPS - 1)]
    choices = [_legal_action(expected_observation, i)
               for i in _sequence(actions, "actions", 1, MAX_ACTIONS)]
    if len(set(prefix)) != len(prefix) or len(set(choices)) != len(choices):
        raise ValueError("Prefix and slate actions must each be unique")
    results = []
    for choice in choices:
        branch = RobustPlacementEnv(seed=seed, profile=profile)
        observation = branch.observe()
        total_reward = 0.
        for action in prefix:
            action = _legal_action(observation, action)
            if observation["options"][action]["stop"]:
                raise ValueError("Ranking prefix cannot contain terminal STOP")
            observation, reward, done, _ = branch.step(action)
            if not np.isfinite(reward):
                raise ValueError("Ranking prefix returned nonfinite reward")
            total_reward += float(reward)
            if done:
                raise ValueError("Ranking prefix already terminated the episode")
        if not _same_public(observation, expected_observation):
            raise ValueError("Ranking seed/profile/prefix does not exactly match expected public observation")
        continuation = GreedyPublicCoverage()
        action, steps = choice, len(prefix)
        while True:
            if steps >= MAX_STEPS:
                raise RuntimeError("Ranking branch exceeded the bounded placement horizon")
            action = _legal_action(observation, action)
            observation, reward, done, info = branch.step(action)
            if not np.isfinite(reward):
                raise ValueError("Ranking branch returned nonfinite reward")
            total_reward += float(reward)
            steps += 1
            if done:
                break
            action = continuation.act(observation, deterministic=True)
        if not np.isfinite(total_reward) or total_reward != info["return"] or info["invalid_actions"] != 0:
            raise ValueError("Ranking branch must preserve the complete finite, legal episode return")
        row = {"action": choice, "timely_fraction": float(1. - info["breached_fraction"]),
               "detection_fraction": float(info["detected_fraction"]),
               "episode_return": float(info["return"]), "cost": float(info["cost"]),
               "placements": deepcopy(info["placements"]), "steps": steps,
               "continuation_steps": steps - len(prefix) - 1}
        _values(row)
        results.append(row)
    return results


def _values(row: Mapping[str, Any]) -> tuple[float, float, float]:
    if not isinstance(row, Mapping):
        raise ValueError("Outcome must be a mapping")
    _integer(row.get("action"), "outcome action", 0, MAX_OPTIONS - 1)
    result = []
    for key in ("timely_fraction", "detection_fraction", "episode_return"):
        value = row.get(key)
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or (key != "episode_return" and not 0 <= value <= 1)):
            raise ValueError(f"Outcome {key} must be finite and in its valid range")
        result.append(float(value))
    if "cost" in row and (isinstance(row["cost"], bool) or not isinstance(row["cost"], (int, float))
                          or not np.isfinite(row["cost"]) or row["cost"] < 0):
        raise ValueError("Outcome cost must be finite and nonnegative")
    return tuple(result)


def comparisons(outcomes: Sequence[Mapping[str, Any]]) -> list[dict[str, int | float]]:
    """All informative unordered pairs, with preferred/rejected *local* indices."""
    rows = _sequence(outcomes, "outcomes", 1, MAX_ACTIONS)
    values = [_values(row) for row in rows]
    if len({int(row["action"]) for row in rows}) != len(rows):
        raise ValueError("Outcome actions must be unique")
    result = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            for tier, (a, b) in enumerate(zip(values[i], values[j], strict=True)):
                if a == b:
                    continue
                delta = abs(a - b)
                if not np.isfinite(delta):
                    raise ValueError("Nonfinite preference difference")
                weight = delta if tier == 0 else .1 * delta if tier == 1 else .01 * min(delta / 20., 1.)
                if not np.isfinite(weight) or not 0 < weight <= 1:
                    raise ValueError("Preference weight must be finite and positive")
                result.append({"preferred": i if a > b else j,
                               "rejected": j if a > b else i, "weight": float(weight)})
                break
    return result
