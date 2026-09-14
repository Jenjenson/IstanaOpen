"""Training-only later-placement credit with a learned-critic normalization anchor.

The ordinary sampled ``value`` is never replaced. Later decisions alone may
carry a detached ``credit_baseline`` from a same-case committed-layout STOP
branch. First decisions retain the ordinary critic baseline and need no branch.
The complete batch's ordinary return-minus-value advantages determine the
frozen normalization rule once. Applied advantages use those same moments;
they are not centered or standardized a second time.

This preserves every first-decision coefficient on the same sampled batch, not
future stopping behavior: shared parameters and later trajectories still change.
The batch-normalized estimator is not claimed to be unbiased or to have lower
gradient variance. STOP's zero raw alternative advantage need not normalize to
zero. The baseline requires simulator rewards during training, is unavailable
on a real feed, and never enters an actor observation or retained sampler state.

Inference, checkpoint schemas, the balanced loss, clipping and Adam semantics
are unchanged. The additive update requires gamma=1 and value_coef=0. Without
baseline substitutions it is exactly the frozen BalancedPolicy.update control.
It does not reset historical Adam moments or create a training run/protocol.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np

from .balanced_policy import BalancedPolicy
from .counterfactual_rollout import _legal_action, _settings, stop_return_to_go
from .robust_scenarios import RobustPlacementEnv


SCHEMA = "triad.anchored_later_stop_credit.v1"
CREDIT_FIELD = "credit_baseline"
NORMALIZATION_RULE = "complete-batch-learned-critic-reference-mean-std;no-second-recentering"
GROUPS = ("all", "first", "later", "stop", "deploy", "first_stop", "later_stop", "first_deploy", "later_deploy")


def _finite(value, label):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)):
        raise ValueError(f"{label} must be a finite numeric scalar")
    return float(value)


def _gamma(gamma):
    if _finite(gamma, "gamma") != 1.:
        raise ValueError("Anchored later STOP credit requires gamma=1")


def prepare_batch(policy: BalancedPolicy, episodes: Sequence[Sequence[Mapping[str, Any]]],
                  *, gamma: float = 1.) -> dict[str, Any]:
    """Read-only detached coefficients and validated records for one full batch.

    ``reference_advantages`` are exactly the frozen update's coefficients.
    ``advantages`` are the coefficients the anchored update actually uses.
    Returned feature/mask arrays are copies, so diagnostics cannot mutate the
    caller's transitions through these records. Missing alternative baselines
    mean ordinary learned-critic credit (the exact control).
    """
    _gamma(gamma)
    if not isinstance(policy, BalancedPolicy):
        raise ValueError("Anchored update requires the unchanged BalancedPolicy")
    transitions, targets, first = [], [], []
    for episode in episodes:
        if not episode:
            raise ValueError("Training episodes must not be empty")
        episode_targets, cumulative = [], 0.
        for record in reversed(episode):
            cumulative = _finite(record["reward"], "Training reward") + gamma * cumulative
            if not np.isfinite(cumulative):
                raise ValueError("Training return-to-go must be finite")
            episode_targets.append(cumulative)
        for index, (record, target) in enumerate(zip(episode, reversed(episode_targets), strict=True)):
            features, mask = policy._input({"feature_names": policy.feature_names,
                                            "option_features": record["features"], "action_mask": record["mask"]})
            action = record["action"]
            if (isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer))
                    or not 0 <= action < len(mask) or not mask[action]):
                raise ValueError("Training action is invalid or masked")
            value = _finite(record["value"], "Training critic value")
            baseline = _finite(record.get(CREDIT_FIELD, value), "Training credit baseline")
            if index == 0 and baseline != value:
                raise ValueError("First-decision credit baseline must remain the ordinary critic value")
            detached = {**record, "features": features.copy(), "mask": mask.copy(), "value": value}
            if CREDIT_FIELD in detached:
                detached[CREDIT_FIELD] = value if index == 0 else baseline
            transitions.append(detached)
            targets.append(target)
            first.append(index == 0)
    if not transitions:
        raise ValueError("Need at least one training episode")
    returns = np.asarray(targets)
    reference_raw = returns - np.asarray([record["value"] for record in transitions])
    credit_raw = returns - np.asarray([record.get(CREDIT_FIELD, record["value"]) for record in transitions])
    if not np.isfinite(reference_raw).all() or not np.isfinite(credit_raw).all():
        raise ValueError("Training advantages must be finite")
    reference_mean, reference_std = float(reference_raw.mean()), float(reference_raw.std())
    if not np.isfinite(reference_mean) or not np.isfinite(reference_std):
        raise ValueError("Reference normalization moments must be finite")
    active = len(reference_raw) > 1 and reference_std > 1e-8
    center, denominator = (reference_mean, reference_std + 1e-8) if active else (0., 1.)
    reference = (reference_raw - center) / denominator if active else reference_raw.copy()
    applied = (credit_raw - center) / denominator if active else credit_raw.copy()
    first_mask = np.asarray(first, dtype=bool)
    if not np.isfinite(applied).all() or not np.isfinite(reference).all():
        raise ValueError("Normalized advantages must be finite")
    if applied[first_mask].tobytes() != reference[first_mask].tobytes():
        raise ValueError("First-decision normalized credit changed unexpectedly")
    return {"transitions": transitions, "returns": returns,
            "reference_raw_advantages": reference_raw, "reference_advantages": reference,
            "credit_raw_advantages": credit_raw, "advantages": applied, "first_mask": first_mask,
            "normalizer": {"rule": NORMALIZATION_RULE, "applied": active, "reference_mean": reference_mean,
                           "reference_std": reference_std, "center": center, "denominator": denominator,
                           "epsilon": 1e-8}}


def _stats(values):
    if not len(values):
        return {"count": 0, "mean": None, "variance": None, "negative": 0, "zero": 0, "positive": 0}
    mean, variance = float(values.mean()), float(values.var())
    if not np.isfinite(mean) or not np.isfinite(variance):
        raise ValueError("Credit diagnostic moments must be finite")
    return {"count": len(values), "mean": mean, "variance": variance,
            "negative": int(np.sum(values < 0)), "zero": int(np.sum(values == 0)),
            "positive": int(np.sum(values > 0))}


def credit_diagnostics(policy: BalancedPolicy, episodes, *, gamma: float = 1.) -> dict[str, Any]:
    """Read-only scalar credit moments; never calls a simulator or samples RNG."""
    batch = prepare_batch(policy, episodes, gamma=gamma)
    transitions, first = batch["transitions"], batch["first_mask"]
    stop = np.asarray([bool(row["features"][row["action"], policy.stop_feature_index]) for row in transitions])
    groups = {"all": np.ones(len(first), dtype=bool), "first": first, "later": ~first,
              "stop": stop, "deploy": ~stop, "first_stop": first & stop, "later_stop": ~first & stop,
              "first_deploy": first & ~stop, "later_deploy": ~first & ~stop}
    values = {"return_to_go": batch["returns"], "baseline": np.asarray([r["value"] for r in transitions]),
              "credit_baseline": np.asarray([r.get(CREDIT_FIELD, r["value"]) for r in transitions]),
              "reference_raw_advantage": batch["reference_raw_advantages"],
              "reference_normalized_advantage": batch["reference_advantages"],
              "raw_advantage": batch["credit_raw_advantages"], "normalized_advantage": batch["advantages"]}
    return {"schema": SCHEMA, "normalizer": batch["normalizer"], "first_coefficients_identical": True,
            "groups": {group: {name: _stats(vector[groups[group]]) for name, vector in values.items()} for group in GROUPS}}


def update(policy: BalancedPolicy, episodes, *, learning_rate: float = .002,
           entropy_coef: float = .015, gamma: float = 1., value_coef: float = 0.,
           max_grad_norm: float = 1.) -> dict[str, float]:
    """Apply the balanced loss with anchored coefficients and frozen Adam math.

    No-substitution batches exactly reproduce BalancedPolicy.update(gamma=1,
    value_coef=0). In particular, historical Adam moments are not reset here.
    The returned numeric metrics have precisely the frozen update's key set.
    """
    _gamma(gamma)
    learning_rate = _finite(learning_rate, "learning_rate")
    entropy_coef = _finite(entropy_coef, "entropy_coef")
    value_coef = _finite(value_coef, "value_coef")
    max_grad_norm = _finite(max_grad_norm, "max_grad_norm")
    if learning_rate <= 0 or entropy_coef < 0 or max_grad_norm <= 0 or value_coef != 0.:
        raise ValueError("Anchored actor update requires positive rate/clip, nonnegative entropy and value_coef=0")
    batch = prepare_batch(policy, episodes, gamma=gamma)
    if (type(policy.update_count) is not int or policy.update_count < 0
            or any(not np.isfinite(array).all() for group in (policy.parameters, policy.adam_m, policy.adam_v) for array in group.values())
            or any((array < 0).any() for array in policy.adam_v.values())):
        raise ValueError("Policy parameters and optimizer state must be finite and valid")
    loss, gradients, metrics = policy._loss_and_gradients(
        batch["transitions"], batch["returns"], batch["advantages"], entropy_coef, value_coef)
    grad_norm = float(np.sqrt(sum(np.sum(gradient * gradient) for gradient in gradients.values())))
    if not np.isfinite(loss) or not np.isfinite(grad_norm):
        raise ValueError("Nonfinite loss/gradient; update not applied")
    scale = min(1.0, max_grad_norm / max(grad_norm, 1e-12))
    policy.update_count += 1
    for key, parameter in policy.parameters.items():
        gradient = gradients[key] * scale
        policy.adam_m[key] = 0.9 * policy.adam_m[key] + 0.1 * gradient
        policy.adam_v[key] = 0.999 * policy.adam_v[key] + 0.001 * gradient * gradient
        corrected_m = policy.adam_m[key] / (1 - 0.9 ** policy.update_count)
        corrected_v = policy.adam_v[key] / (1 - 0.999 ** policy.update_count)
        parameter -= learning_rate * corrected_m / (np.sqrt(corrected_v) + 1e-8)
    return {**metrics, "loss": float(loss), "gradient_norm": grad_norm,
            "gradient_scale": scale, "mean_return_to_go": float(batch["returns"].mean()),
            "transitions": len(batch["transitions"]), "updates": policy.update_count}


def rollout(env: RobustPlacementEnv, policy: Any, seed: int,
            *, gamma: float = 1.) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect a normal trajectory; sample before attaching later-only baselines.

    Branches recreate the same seed/profile and committed prefix. They neither
    advance the live environment nor consume actor/global RNG. Deep copying
    each sampler record prevents detached baseline/reward leakage into caches.
    """
    _settings(seed, gamma)
    observation = env.reset(seed=seed)
    transitions, prefix, total_reward = [], [], 0.
    while True:
        action, sampled_record = policy.sample(observation)
        action = _legal_action(observation, action)
        record = deepcopy(sampled_record)
        if CREDIT_FIELD in record:
            raise ValueError("Sampler must not supply a training-only credit baseline")
        if (isinstance(record.get("action"), (bool, np.bool_))
                or not isinstance(record.get("action"), (int, np.integer)) or record["action"] != action):
            raise ValueError("Sampler record/action mismatch")
        _finite(record["value"], "Sampled critic value")
        if prefix:
            record[CREDIT_FIELD] = _finite(stop_return_to_go(seed=seed, profile=env.profile, prior_actions=prefix,
                                                            expected_observation=observation, gamma=gamma),
                                           "Paired STOP baseline")
        observation, reward, done, info = env.step(action)
        reward = _finite(reward, "Environment reward")
        record["reward"] = reward
        transitions.append(record)
        total_reward += reward
        if done:
            break
        prefix.append(action)
        if len(transitions) >= 32:
            raise RuntimeError("Anchored rollout exceeded its bounded placement horizon")
    return transitions, {**info, "episode_return": total_reward, "steps": len(transitions), "seed": seed,
                         "case_metadata": deepcopy(env.case_metadata), "anchored_credit": {
                             "schema": SCHEMA, "gamma": 1., "branch_rollouts": len(transitions) - 1,
                             "training_only": True, "private_reward_in_actor_observation": False,
                             "first_baseline": "ordinary learned critic; no branch",
                             "later_baseline": "committed-layout immediate STOP reward-to-go",
                             "normalization": NORMALIZATION_RULE, "optimal_stop_label": False}}
