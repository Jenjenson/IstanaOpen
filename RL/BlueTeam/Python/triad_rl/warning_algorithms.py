"""Alternative masked optimizers for the native warning-placement policy.

All algorithms intentionally share the same map-specific actor and public-only
action contract.  They differ only in how complete native episode returns
update the actor and, for actor-critic methods, a small placement-step critic.
This keeps algorithm comparisons from silently changing the environment,
sensor options, legal mask, reward, or deployment limits.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .directional_inputs import apply_placement, build_observation
from .istana_live import public_planning_inputs
from .warning_policy import WarningPolicy


class _MaskedActorCriticPolicy(WarningPolicy):
    algorithm = None
    schema = None

    def __init__(self, context, seed=917, *, sensor_count=None):
        super().__init__(context, seed=seed, sensor_count=sensor_count)
        state, _, _ = public_planning_inputs(context)
        if self.sensor_count is not None:
            state["max_sites"] = self.sensor_count
        self.values = np.zeros(state["max_sites"] + 1, dtype=float)
        self.value_updates = np.zeros_like(self.values)
        self.optimizer_steps = 0

    def _check_context(self, context):
        return self._planning_inputs(context)

    @staticmethod
    def _probabilities(logits, mask):
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.ndim != 1 or not mask.any():
            raise ValueError("Action mask must be a nonempty boolean vector")
        probabilities = np.zeros(len(mask), dtype=float)
        probabilities[mask] = np.exp(logits[mask] - logits[mask].max())
        probabilities /= probabilities.sum()
        return probabilities

    def plan(self, context, *, rng=None, deterministic=False):
        state, catalogue = self._check_context(context)
        records, placements = [], []
        generator = self.rng if rng is None else rng
        for step in range(state["max_sites"] + 1):
            observation = build_observation(state, catalogue)
            mask = self._action_mask(state, catalogue, observation)
            probabilities = self._probabilities(self.logits(), mask)
            action = (int(np.argmax(np.where(mask, self.logits(), -np.inf)))
                      if deterministic else int(generator.choice(len(probabilities), p=probabilities)))
            records.append({"action": action, "old_probability": float(probabilities[action]),
                            "probabilities": probabilities.copy(), "mask": mask.copy(),
                            "step": step, "value": float(self.values[step])})
            option = observation["options"][action]
            state = apply_placement(state, action, catalogue)
            if option["stop"]:
                return placements, records
            placements.append({"profileId": option["sensor_id"],
                               "siteId": option["site_index"],
                               "yawDeg": option.get("yaw_deg", 0.),
                               "pitchDeg": option.get("pitch_deg", 0.)})
        raise RuntimeError("Policy failed to STOP within the layout limit")

    def _samples(self, episodes):
        rewards = np.asarray([reward for _, reward in episodes], dtype=float)
        if not len(rewards) or not np.isfinite(rewards).all():
            raise ValueError("Need finite complete-episode rewards")
        samples = []
        for (records, reward) in episodes:
            if not isinstance(records, list) or not records:
                raise ValueError("Actor-critic updates require complete action records")
            for record in records:
                if not isinstance(record, dict) or set(record) != {
                        "action", "old_probability", "probabilities", "mask", "step", "value"}:
                    raise ValueError("Invalid actor-critic action record")
                action, step = record["action"], record["step"]
                mask = np.asarray(record["mask"])
                old = np.asarray(record["probabilities"], dtype=float)
                if (type(action) is not int or type(step) is not int
                        or not 0 <= step < len(self.values)
                        or mask.dtype != np.bool_ or mask.shape != self.option_logits.shape
                        or old.shape != self.option_logits.shape or not np.isfinite(old).all()
                        or not 0 <= action < len(mask) or not mask[action]
                        or not np.isfinite(record["old_probability"])
                        or record["old_probability"] <= 0
                        or not np.isfinite(record["value"])):
                    raise ValueError("Invalid actor-critic action record")
                samples.append((record, float(reward), float(reward - record["value"])))
        return rewards, samples

    def _critic_update(self, samples, learning_rate=.08):
        errors = [[] for _ in self.values]
        for record, reward, _ in samples:
            errors[record["step"]].append(reward - self.values[record["step"]])
        loss_terms = []
        for step, values in enumerate(errors):
            if not values:
                continue
            error = float(np.mean(values))
            loss_terms.extend(value * value for value in values)
            self.value_updates[step] += 1
            self.values[step] += learning_rate * error
        return float(np.mean(loss_terms)) if loss_terms else 0.

    def _actor_step(self, gradient, learning_rate):
        norm = float(np.linalg.norm(gradient))
        gradient /= max(1., norm)
        self.optimizer_steps += 1
        self.m = .9 * self.m + .1 * gradient
        self.v = .999 * self.v + .001 * gradient ** 2
        adjusted_m = self.m / (1 - .9 ** self.optimizer_steps)
        adjusted_v = self.v / (1 - .999 ** self.optimizer_steps)
        self.option_logits += learning_rate * adjusted_m / (np.sqrt(adjusted_v) + 1e-8)
        return norm

    def save(self, path):
        data = {"schema": self.schema, "algorithm": self.algorithm,
                "contract": self.contract, "option_logits": self.option_logits.tolist(),
                "sensor_count": self.sensor_count,
                "updates": self.updates, "baseline": self.baseline,
                "adam_m": self.m.tolist(), "adam_v": self.v.tolist(),
                "optimizer_steps": self.optimizer_steps, "values": self.values.tolist(),
                "value_updates": self.value_updates.tolist(),
                "rng_state": deepcopy(self.rng.bit_generator.state)}
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @classmethod
    def load(cls, path, context):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        policy = cls(context, sensor_count=data.get("sensor_count"))
        if (data.get("schema") != cls.schema or data.get("algorithm") != cls.algorithm
                or data.get("contract") != policy.contract):
            raise ValueError(f"Wrong {cls.algorithm} warning policy contract")
        arrays = {"option_logits": "option_logits", "m": "adam_m", "v": "adam_v",
                  "values": "values", "value_updates": "value_updates"}
        for name, key in arrays.items():
            value = np.asarray(data[key], dtype=float)
            if value.shape != getattr(policy, name).shape or not np.isfinite(value).all():
                raise ValueError("Invalid policy arrays")
            setattr(policy, name, value)
        policy.updates, policy.baseline = data["updates"], data["baseline"]
        policy.optimizer_steps = data["optimizer_steps"]
        if (type(policy.updates) is not int or policy.updates < 0
                or type(policy.optimizer_steps) is not int or policy.optimizer_steps < 0):
            raise ValueError("Invalid optimizer counters")
        policy.rng.bit_generator.state = data["rng_state"]
        return policy


class MaskedA2CPolicy(_MaskedActorCriticPolicy):
    """Synchronous advantage actor-critic using terminal native warning returns."""
    algorithm = "a2c"
    schema = "istana.warning_directional_masked_a2c.v1"

    def update(self, episodes, learning_rate=.08, value_learning_rate=.08):
        rewards, samples = self._samples(episodes)
        gradient = np.zeros_like(self.option_logits)
        for record, _, advantage in samples:
            score = -np.asarray(record["probabilities"], dtype=float).copy()
            score[record["action"]] += 1
            gradient += advantage * score
        gradient /= max(1, len(episodes))
        gradient_norm = self._actor_step(gradient, learning_rate)
        value_loss = self._critic_update(samples, value_learning_rate)
        self.updates += 1
        self.baseline = float(rewards.mean())
        return {"algorithm": self.algorithm, "update": self.updates,
                "mean_training_warning_s": self.baseline,
                "gradient_norm": gradient_norm, "value_loss": value_loss}


class MaskedPPOPolicy(_MaskedActorCriticPolicy):
    """Clipped PPO over the legal categorical placement actions."""
    algorithm = "ppo"
    schema = "istana.warning_directional_masked_ppo.v1"

    def update(self, episodes, learning_rate=.04, value_learning_rate=.08,
               clip_ratio=.2, epochs=4):
        if not 0 < clip_ratio < 1 or type(epochs) is not int or not 1 <= epochs <= 16:
            raise ValueError("PPO requires clip_ratio in (0, 1) and 1..16 epochs")
        rewards, samples = self._samples(episodes)
        advantages = np.asarray([sample[2] for sample in samples], dtype=float)
        if len(advantages) > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / advantages.std()
        gradient_norms, clipped = [], 0
        for _ in range(epochs):
            gradient = np.zeros_like(self.option_logits)
            for (record, _, _), advantage in zip(samples, advantages):
                probabilities = self._probabilities(self.option_logits, record["mask"])
                ratio = probabilities[record["action"]] / record["old_probability"]
                outside = ((advantage >= 0 and ratio > 1 + clip_ratio)
                           or (advantage < 0 and ratio < 1 - clip_ratio))
                if outside:
                    clipped += 1
                    continue
                score = -probabilities
                score[record["action"]] += 1
                gradient += advantage * ratio * score
            gradient /= max(1, len(samples))
            gradient_norms.append(self._actor_step(gradient, learning_rate))
        value_loss = self._critic_update(samples, value_learning_rate)
        self.updates += 1
        self.baseline = float(rewards.mean())
        return {"algorithm": self.algorithm, "update": self.updates,
                "mean_training_warning_s": self.baseline,
                "gradient_norm": float(np.mean(gradient_norms)),
                "value_loss": value_loss,
                "clip_fraction": clipped / max(1, epochs * len(samples)), "epochs": epochs}


TRAINING_ALGORITHMS = {
    "reinforce": {"label": "REINFORCE", "description":
        "Current masked policy-gradient baseline with a batch reward baseline.",
        "factory": WarningPolicy},
    "ppo": {"label": "Masked PPO", "description":
        "Clipped actor-critic updates over the same legal categorical placement actions.",
        "factory": MaskedPPOPolicy},
    "a2c": {"label": "Masked A2C", "description":
        "Synchronous advantage actor-critic over the same legal placement actions.",
        "factory": MaskedA2CPolicy},
}
