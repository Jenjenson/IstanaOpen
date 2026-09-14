"""Training-only paired STOP reward baseline; no new policy or training run.

At each sampled decision, independently recreate the same seeded robust case,
replay the already committed legal placements, and STOP. The immediate STOP
reward is the return-to-go of that committed layout, including removal of the
current shaping potential, not its full episode return. It is independent of
the *next* sampled action and is therefore an action-independent control variate
for undiscounted REINFORCE. It is not an oracle label saying STOP is optimal.

This baseline uses simulator reward/hidden outcomes in training only. The actor
still receives exactly its ordinary public observation; the detached record's
``value`` is replaced only after sampling. It cannot run against real sensor
feeds, supplied scenarios, validation seeds or final-test seeds. It neither
updates weights nor implements a training experiment. A subsequent experimental
trainer must explicitly use gamma=1 and should ablate value_coef=0 to avoid
shared-critic gradients; a matched control and protocol are still required.
Its advantage-normalization rule must also be explicit: a sampled STOP has
zero raw (return-minus-STOP-baseline) advantage, but the existing update's batch
mean-centering can make that normalized advantage nonzero. Zero raw STOP
advantage therefore does not imply zero policy gradient under that update.

Initial STOP is always -10.5 in this environment, so this control variate cannot
remove first-placement cross-scenario difficulty. Its state-dependent benefit
starts after a placement; earlier paid rewards/penalties must not be counted a
second time in the baseline.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np

from .robust_scenarios import RobustPlacementEnv


BASELINE_SCHEMA = "triad.counterfactual_stop_baseline.v1"
TRAINING_SEED_LIMIT = 10 ** 15


def _settings(seed: int, gamma: float) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < TRAINING_SEED_LIMIT:
        raise ValueError("Counterfactual rollout accepts training seeds in [0, 10**15) only")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or gamma != 1.0:
        raise ValueError("Counterfactual STOP baseline requires gamma=1")


def _same_public(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and left.dtype == right.dtype and np.array_equal(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_same_public(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _same_public(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def _legal_action(observation: Mapping[str, Any], action: Any) -> int:
    mask = observation["action_mask"]
    if (isinstance(action, bool) or not isinstance(action, (int, np.integer))
            or not 0 <= action < len(mask) or not mask[action]):
        raise ValueError("Counterfactual prefix/action must contain only legal integer actions")
    return int(action)


def stop_return_to_go(*, seed: int, profile: str, prior_actions: Sequence[int] = (),
                      expected_observation: Mapping[str, Any] | None = None,
                      gamma: float = 1.0) -> float:
    """Recreate one case/prefix and return only its next, terminal STOP reward.

    No live environment or actor is accepted, so this cannot consume their RNG
    or mutate their trajectory. A prefix ending an episode is rejected: there
    is no next decision to baseline after automatic resource exhaustion/STOP.
    ``expected_observation`` guards against accidentally replaying the wrong
    seed/profile/prefix; comparison contains public data only.
    """
    _settings(seed, gamma)
    if isinstance(prior_actions, (str, bytes)) or not isinstance(prior_actions, Sequence):
        raise ValueError("prior_actions must be a sequence of legal deployment actions")
    if len(prior_actions) > 32:
        raise ValueError("Counterfactual prefix exceeds the bounded placement horizon")
    branch = RobustPlacementEnv(seed=seed, profile=profile)
    observation = branch.reset(seed=seed)
    for action in prior_actions:
        action = _legal_action(observation, action)
        if observation["options"][action]["stop"]:
            raise ValueError("Counterfactual prefix cannot include terminal STOP")
        observation, _, done, _ = branch.step(action)
        if done:
            raise ValueError("Counterfactual prefix already terminated the episode")
    if expected_observation is not None and not _same_public(observation, expected_observation):
        raise ValueError("Counterfactual seed/profile/prefix does not match the live public observation")
    stops = [i for i, option in enumerate(observation["options"]) if option["stop"]]
    if len(stops) != 1:
        raise ValueError("Counterfactual environment requires one STOP action")
    stop = _legal_action(observation, stops[0])
    _, reward, done, _ = branch.step(stop)
    if not done or not np.isfinite(reward):
        raise ValueError("Counterfactual STOP must return one finite terminal reward")
    return float(reward)


def counterfactual_rollout(env: RobustPlacementEnv, policy: Any, seed: int,
                           *, gamma: float = 1.0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect one normal sampled trajectory with detached paired STOP values.

    API matches the existing training rollout's return pair. The normal actor
    sampling and live environment steps remain unchanged; only record ``value``
    differs. No optimizer update is performed here. Extra simulation calls and
    their compute cost are explicitly recorded in the returned training audit,
    never inserted into actor observations or feature matrices.
    """
    _settings(seed, gamma)
    observation = env.reset(seed=seed)
    transitions: list[dict[str, Any]] = []
    prefix: list[int] = []
    total_reward = 0.0
    while True:
        baseline = stop_return_to_go(seed=seed, profile=env.profile, prior_actions=prefix,
                                      expected_observation=observation, gamma=gamma)
        action, sampled_record = policy.sample(observation)
        action = _legal_action(observation, action)
        # A generic sampler may retain the object it returned. Isolate training
        # annotations so private reward baselines cannot enter retained actor
        # state and be observed on its next sample() call.
        record = deepcopy(sampled_record)
        # sample() has completed before the training-only reward baseline is
        # attached. Neither it nor the normal observation exposes this value.
        record["value"] = baseline
        observation, reward, done, info = env.step(action)
        if not np.isfinite(reward):
            raise ValueError("Environment returned nonfinite reward")
        record["reward"] = float(reward)
        transitions.append(record)
        total_reward += float(reward)
        if done:
            break
        prefix.append(action)
        if len(transitions) >= 32:
            raise RuntimeError("Counterfactual rollout exceeded its bounded placement horizon")
    return transitions, {**info, "episode_return": total_reward, "steps": len(transitions),
                         "seed": seed, "case_metadata": deepcopy(env.case_metadata),
                         "counterfactual_baseline": {
                             "schema": BASELINE_SCHEMA, "gamma": 1.0,
                             "branch_rollouts": len(transitions),
                             "training_only": True, "private_reward_in_actor_observation": False,
                             "baseline": "committed-layout immediate STOP reward-to-go",
                             "optimal_stop_label": False}}
