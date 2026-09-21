"""Native directional training adapters: existing REINFORCE and NumPy PPO.

Both plan with the application's directional input/action contract. Simulation
rewards are supplied by the manager after a complete native episode; no private
Red data is a policy input. PPO uses the clipped surrogate (Schulman et al.,
https://arxiv.org/abs/1707.06347) and terminal-episode GAE
(https://arxiv.org/abs/1506.02438). The actor/critic network reuses AdaptivePolicy's
NumPy forward pass; its frozen legacy observation/checkpoint schema is not reused.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from .adaptive_policy import AdaptivePolicy
from .directional_inputs import FEATURE_NAMES, FEATURE_SCHEMA, apply_placement, build_observation
from .istana_live import public_planning_inputs
from .warning_policy import WarningPolicy


CHECKPOINT_SCHEMA = "istana.native_training_checkpoint.v1"
ALGORITHMS = {
    "reinforce": {
        "id": "reinforce", "label": "REINFORCE (existing directional policy)",
        "description": "Existing masked profile/site/yaw/pitch policy with a batch return baseline and Adam.",
        "supported_metrics": ["policy_loss", "entropy"],
        "default_config": {"learning_rate": .12},
    },
    "ppo": {
        "id": "ppo", "label": "PPO",
        "description": "Clipped on-policy actor-critic with GAE and multiple shuffled minibatch epochs.",
        "supported_metrics": ["policy_loss", "value_loss", "entropy"],
        "default_config": {"learning_rate": .003, "hidden_size": 32, "gamma": 1.,
                           "gae_lambda": .95, "clip_ratio": .2, "epochs": 4,
                           "minibatch_size": 32, "entropy_coef": .01,
                           "value_coef": .5, "max_grad_norm": 1.,
                           "normalize_advantages": True},
    },
}


def algorithm_catalogue():
    """UI metadata; adding an adapter here never substitutes a fake algorithm."""
    return deepcopy(list(ALGORITHMS.values()))


def _number(value, name, low, high, *, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or not low <= value <= high
            or (integer and int(value) != value)):
        raise ValueError(f"{name} must be {'an integer' if integer else 'finite'} in [{low}, {high}]")
    return int(value) if integer else float(value)


def validate_algorithm_config(algorithm, config=None):
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unsupported native training algorithm: {algorithm}")
    defaults = ALGORITHMS[algorithm]["default_config"]
    if config is not None and not isinstance(config, dict):
        raise ValueError("algorithm configuration must be an object")
    if set(config or {}) - set(defaults):
        raise ValueError("Unknown algorithm configuration field")
    result = {**defaults, **(config or {})}
    result["learning_rate"] = _number(result["learning_rate"], "learning_rate", 1e-8, 1.)
    if algorithm == "ppo":
        for key, low, high in (("hidden_size", 1, 128), ("epochs", 1, 20), ("minibatch_size", 1, 1024)):
            result[key] = _number(result[key], key, low, high, integer=True)
        for key, low, high in (("gamma", 0., 1.), ("gae_lambda", 0., 1.), ("clip_ratio", .001, .5),
                               ("entropy_coef", 0., 1.), ("value_coef", 0., 10.), ("max_grad_norm", .001, 100.)):
            result[key] = _number(result[key], key, low, high)
        if type(result["normalize_advantages"]) is not bool:
            raise ValueError("normalize_advantages must be boolean")
    return result


def context_contract(context):
    """Static model compatibility, excluding changing seeds, tracks and Red truth."""
    state, catalogue, temporal = public_planning_inputs(context)
    observation = build_observation(state, catalogue)
    return {
        "coordinate_system": context["coordinateSystem"],
        "feature_schema": FEATURE_SCHEMA, "feature_names": list(FEATURE_NAMES),
        "action_schema": "masked_profile_site_yaw_pitch_stop.v1",
        "sites": state["sites"], "catalogue": catalogue,
        "options": [{key: row[key] for key in ("sensor_id", "site_index", "yaw_deg", "pitch_deg", "stop")}
                    for row in observation["options"]],
        "placement_constraints": {key: state[key] for key in (
            "budget_total", "max_sites", "min_separation", "deployment_min_radius",
            "deployment_max_radius", "available_sensor_ids", "blocked_sites")},
        "temporal_config": asdict(temporal), "sensor_model": context.get("sensorModel"),
        "training_configuration_version": context.get("trainingConfigurationVersion"),
        # Native mounted heights affect 3D FOV and LOS even if relative XY sites
        # have not changed. Keep that static public geometry in compatibility.
        "native_geometry": {key: deepcopy(context.get(key)) for key in (
            "worldOriginCm", "siteSurfacesWorldCm", "placementRule", "fixedStepSeconds",
            "timeLimitSeconds", "warningDefinition")},
    }


def _check_contract(expected, actual):
    if not isinstance(expected, dict):
        raise ValueError("Invalid native model compatibility contract")
    if expected != actual:
        changed = [key for key in actual if expected.get(key) != actual[key]]
        raise ValueError("Model is incompatible with this native simulation: " + ", ".join(changed))


def _json(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


class TrainingAlgorithm:
    algorithm_id = ""

    def __init__(self, context, seed=0, config=None):
        self.seed = _number(seed, "seed", 0, 2**32 - 1, integer=True)
        self.config = validate_algorithm_config(self.algorithm_id, config)
        self.contract = context_contract(context)
        self.generation = 0

    @property
    def metadata(self):
        return {"algorithm": self.algorithm_id, "algorithm_config": deepcopy(self.config),
                "seed": self.seed, "architecture": self.architecture,
                "supported_metrics": ALGORITHMS[self.algorithm_id]["supported_metrics"],
                "observation_space": {"schema": FEATURE_SCHEMA, "feature_names": list(FEATURE_NAMES)},
                "action_space": {"schema": self.contract["action_schema"],
                                 "options": len(self.contract["options"])},
                "contract": deepcopy(self.contract)}

    def save(self, path):
        """One atomic, non-pickle JSON checkpoint including optimizer and RNG."""
        path = Path(path)
        model = self._state()
        payload = {"schema": CHECKPOINT_SCHEMA, **self.metadata, "generation": self.generation,
                   "model": model, "model_sha256": hashlib.sha256(_json(model).encode()).hexdigest()}
        encoded = (_json(payload) + "\n").encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
            temporary.replace(path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return hashlib.sha256(encoded).hexdigest()


class ReinforceAlgorithm(TrainingAlgorithm):
    algorithm_id = "reinforce"

    def __init__(self, context, seed=0, config=None):
        super().__init__(context, seed, config)
        self.policy = WarningPolicy(context, seed=self.seed)

    @property
    def architecture(self):
        return {"kind": "existing_masked_option_logits", "options": len(self.policy.option_logits),
                "critic": None, "optimizer": "adam", "policy_schema": "istana.warning_directional_reinforce.v2"}

    def plan(self, context, *, deterministic=False, rng=None):
        _check_contract(self.contract, context_contract(context))
        placements, records = self.policy.plan(context, deterministic=deterministic, rng=rng)
        return placements, [{"action": action, "probabilities": p, "generation": self.generation}
                            for action, p in records]

    def update(self, episodes):
        if not episodes:
            raise ValueError("Need complete episodes")
        rewards = np.asarray([reward for _, reward in episodes], dtype=float)
        if not np.isfinite(rewards).all():
            raise ValueError("Need finite complete-episode rewards")
        advantages = (rewards - (rewards.sum() - rewards) / (len(rewards) - 1)
                      if len(rewards) > 1 else rewards - (self.policy.baseline or 0.))
        converted, policy_loss, entropy = [], 0., []
        for (records, reward), advantage in zip(episodes, advantages, strict=True):
            if not records:
                raise ValueError("Need nonempty complete layout records")
            rows = []
            for record in records:
                if record.get("generation") != self.generation:
                    raise ValueError("REINFORCE requires records from the current policy")
                action, p = record["action"], np.asarray(record["probabilities"], dtype=float)
                if (p.shape != self.policy.option_logits.shape or not np.isfinite(p).all()
                        or (p < 0).any() or not np.isclose(p.sum(), 1.)
                        or isinstance(action, bool) or not isinstance(action, (int, np.integer))
                        or not 0 <= action < len(p) or p[action] <= 0):
                    raise ValueError("Invalid sampled REINFORCE action/probabilities")
                rows.append((action, p))
                policy_loss -= float(advantage * np.log(p[action]))
                positive = p > 0
                entropy.append(-float(p[positive] @ np.log(p[positive])))
            converted.append((rows, float(reward)))
        result = self.policy.update(converted, learning_rate=self.config["learning_rate"])
        self.generation += 1
        return {**result, "policy_loss": policy_loss / len(episodes), "value_loss": None,
                "entropy": float(np.mean(entropy)), "transitions": len(entropy)}

    def _state(self):
        p = self.policy
        return {"option_logits": p.option_logits.tolist(), "updates": p.updates,
                "baseline": p.baseline, "adam_m": p.m.tolist(), "adam_v": p.v.tolist(),
                "rng_state": deepcopy(p.rng.bit_generator.state)}

    def _restore(self, state):
        for key, name in (("option_logits", "option_logits"), ("adam_m", "m"), ("adam_v", "v")):
            setattr(self.policy, name, _array(state[key], getattr(self.policy, name).shape, key,
                                             nonnegative=key == "adam_v"))
        self.policy.updates = _number(state["updates"], "updates", 0, 2**53, integer=True)
        self.policy.baseline = (None if state["baseline"] is None else
                               _number(state["baseline"], "baseline", -1e12, 1e12))
        self.policy.rng.bit_generator.state = state["rng_state"]


def generalized_advantages(rewards, values, gamma=1., gae_lambda=.95):
    """GAE for one complete episode; the terminal state's value is exactly zero."""
    rewards, values = np.asarray(rewards, dtype=float), np.asarray(values, dtype=float)
    if (rewards.ndim != 1 or not len(rewards) or rewards.shape != values.shape
            or not np.isfinite(rewards).all() or not np.isfinite(values).all()):
        raise ValueError("GAE needs equally sized finite reward/value vectors")
    gamma = _number(gamma, "gamma", 0., 1.)
    gae_lambda = _number(gae_lambda, "gae_lambda", 0., 1.)
    advantages, tail, next_value = np.zeros_like(rewards), 0., 0.
    for index in range(len(rewards) - 1, -1, -1):
        delta = rewards[index] + gamma * next_value - values[index]
        tail = delta + gamma * gae_lambda * tail
        advantages[index] = tail
        next_value = values[index]
    return advantages, advantages + values


class PPOAlgorithm(TrainingAlgorithm):
    algorithm_id = "ppo"

    def __init__(self, context, seed=0, config=None):
        super().__init__(context, seed, config)
        # Reuse the repository's shared tanh actor/pooled critic and initialization.
        # Only _forward is used: frozen legacy schema validation is never bypassed
        # to load a legacy checkpoint or pass one off as a directional model.
        self.network = AdaptivePolicy(FEATURE_NAMES, seed=self.seed, hidden_size=self.config["hidden_size"])

    @property
    def architecture(self):
        return {"kind": "shared_option_tanh_actor_pooled_critic", "input_size": len(FEATURE_NAMES),
                "hidden_size": self.config["hidden_size"], "actor": "masked_categorical",
                "critic": "mean_of_legal_option_embeddings", "optimizer": "adam"}

    def plan(self, context, *, deterministic=False, rng=None):
        _check_contract(self.contract, context_contract(context))
        state, catalogue, _ = public_planning_inputs(context)
        placements, records = [], []
        generator = self.network.rng if rng is None else rng
        for _ in range(state["max_sites"] + 1):
            observation = build_observation(state, catalogue)
            x = np.asarray(observation["option_features"], dtype=float)
            mask = observation["action_mask"]
            _, probabilities, value = self.network._forward(x, mask)
            action = int(np.argmax(probabilities)) if deterministic else int(generator.choice(len(mask), p=probabilities))
            records.append({"features": x.copy(), "mask": mask.copy(), "action": action,
                            "old_log_probability": float(np.log(probabilities[action])), "value": value,
                            "generation": self.generation})
            option = observation["options"][action]
            state = apply_placement(state, action, catalogue)
            if option["stop"]:
                return placements, records
            placements.append({"profileId": option["sensor_id"], "siteId": option["site_index"],
                               "yawDeg": option["yaw_deg"], "pitchDeg": option["pitch_deg"]})
        raise RuntimeError("PPO failed to STOP within the native layout limit")

    def _loss_and_gradients(self, records, returns, advantages):
        """Exact clipped-surrogate/critic/entropy gradients; old logp is frozen."""
        p, cfg = self.network.parameters, self.config
        gradients = {key: np.zeros_like(value) for key, value in p.items()}
        sums = dict(policy_loss=0., value_loss=0., entropy=0., clip_fraction=0., approximate_kl=0.)
        for row, target, advantage in zip(records, returns, advantages, strict=True):
            x, mask, action = row["features"], row["mask"], row["action"]
            hidden, probabilities, value = self.network._forward(x, mask)
            logs = np.zeros_like(probabilities)
            logs[mask] = np.log(np.maximum(probabilities[mask], np.finfo(float).tiny))
            log_ratio = logs[action] - row["old_log_probability"]
            ratio = float(np.exp(log_ratio))
            clipped = float(np.clip(ratio, 1 - cfg["clip_ratio"], 1 + cfg["clip_ratio"]))
            use_clipped = (advantage > 0 and ratio > 1 + cfg["clip_ratio"]
                           or advantage < 0 and ratio < 1 - cfg["clip_ratio"])
            entropy = -float(probabilities @ logs)
            error = value - target
            sums["policy_loss"] -= float(min(ratio * advantage, clipped * advantage))
            sums["value_loss"] += .5 * float(error * error)
            sums["entropy"] += entropy
            sums["clip_fraction"] += float(abs(ratio - 1) > cfg["clip_ratio"])
            sums["approximate_kl"] += float(ratio - 1 - log_ratio)
            coefficient = 0. if use_clipped else float(advantage * ratio)
            d_logits = coefficient * probabilities
            d_logits[action] -= coefficient
            d_logits += cfg["entropy_coef"] * probabilities * (logs + entropy)
            d_value = cfg["value_coef"] * error
            gradients["wa"] += hidden.T @ d_logits
            gradients["wv"] += hidden[mask].mean(axis=0) * d_value
            gradients["bv"][0] += d_value
            d_hidden = d_logits[:, None] * p["wa"][None, :]
            d_hidden[mask] += p["wv"][None, :] * (d_value / int(mask.sum()))
            d_z = d_hidden * (1 - hidden * hidden)
            gradients["w1"] += x.T @ d_z
            gradients["b1"] += d_z.sum(axis=0)
        for value in gradients.values():
            value /= len(records)
        metrics = {key: value / len(records) for key, value in sums.items()}
        loss = metrics["policy_loss"] + cfg["value_coef"] * metrics["value_loss"] - cfg["entropy_coef"] * metrics["entropy"]
        if not np.isfinite(loss) or not all(np.isfinite(value).all() for value in gradients.values()):
            raise ValueError("PPO produced nonfinite loss/gradients")
        return float(loss), gradients, metrics

    def update(self, episodes):
        records, advantages, returns = [], [], []
        for episode, reward in episodes:
            if not episode or not np.isfinite(reward):
                raise ValueError("PPO needs nonempty complete episodes and finite terminal rewards")
            rows = []
            for row in episode:
                x, mask, action = np.asarray(row["features"], dtype=float), np.asarray(row["mask"]), row["action"]
                if (x.ndim != 2 or x.shape != (len(self.contract["options"]), len(FEATURE_NAMES))
                        or not np.isfinite(x).all() or mask.dtype != np.bool_ or mask.shape != (len(x),)
                        or not mask.any() or isinstance(action, bool) or not isinstance(action, (int, np.integer))
                        or not 0 <= action < len(mask) or not mask[action]
                        or not np.isfinite(row["value"]) or not np.isfinite(row["old_log_probability"])
                        or row["old_log_probability"] > 1e-9 or row.get("generation") != self.generation):
                    raise ValueError("Invalid or stale PPO rollout record")
                rows.append({**row, "features": x, "mask": mask})
            rewards = np.zeros(len(rows)); rewards[-1] = float(reward)
            a, r = generalized_advantages(rewards, [row["value"] for row in rows],
                                          self.config["gamma"], self.config["gae_lambda"])
            records.extend(rows); advantages.extend(a); returns.extend(r)
        if not records:
            raise ValueError("PPO needs complete episodes")
        advantages, returns = np.asarray(advantages), np.asarray(returns)
        if self.config["normalize_advantages"] and len(advantages) > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        previous = deepcopy(self.network.__dict__)
        sums, count = {}, 0
        try:
            for _ in range(self.config["epochs"]):
                order = self.network.rng.permutation(len(records))
                for start in range(0, len(records), self.config["minibatch_size"]):
                    indices = order[start:start + self.config["minibatch_size"]]
                    loss, gradients, metrics = self._loss_and_gradients(
                        [records[i] for i in indices], returns[indices], advantages[indices])
                    norm = float(np.sqrt(sum(np.sum(g * g) for g in gradients.values())))
                    if not np.isfinite(norm):
                        raise ValueError("PPO gradient norm is nonfinite")
                    scale = min(1., self.config["max_grad_norm"] / max(norm, 1e-12))
                    self.network.update_count += 1
                    for key, parameter in self.network.parameters.items():
                        gradient = gradients[key] * scale
                        self.network.adam_m[key] = .9 * self.network.adam_m[key] + .1 * gradient
                        self.network.adam_v[key] = .999 * self.network.adam_v[key] + .001 * gradient ** 2
                        m = self.network.adam_m[key] / (1 - .9 ** self.network.update_count)
                        v = self.network.adam_v[key] / (1 - .999 ** self.network.update_count)
                        parameter -= self.config["learning_rate"] * m / (np.sqrt(v) + 1e-8)
                    for key, value in {**metrics, "loss": loss, "gradient_norm": norm}.items():
                        sums[key] = sums.get(key, 0.) + value * len(indices)
                    count += len(indices)
            if not all(np.isfinite(v).all() for v in self.network.parameters.values()):
                raise ValueError("PPO update produced nonfinite parameters")
        except Exception:
            self.network.__dict__ = previous
            raise
        self.generation += 1
        return {**{key: value / count for key, value in sums.items()}, "updates": self.network.update_count,
                "transitions": len(records), "mean_return_to_go": float(returns.mean())}

    def _state(self):
        return {"parameters": {k: v.tolist() for k, v in self.network.parameters.items()},
                "adam_m": {k: v.tolist() for k, v in self.network.adam_m.items()},
                "adam_v": {k: v.tolist() for k, v in self.network.adam_v.items()},
                "updates": self.network.update_count, "rng_state": deepcopy(self.network.rng.bit_generator.state)}

    def _restore(self, state):
        for field in ("parameters", "adam_m", "adam_v"):
            target = getattr(self.network, field)
            if set(state[field]) != set(target):
                raise ValueError("Invalid PPO checkpoint parameter names")
            for key, value in target.items():
                target[key] = _array(state[field][key], value.shape, key, nonnegative=field == "adam_v")
        self.network.update_count = _number(state["updates"], "updates", 0, 2**53, integer=True)
        self.network.rng.bit_generator.state = state["rng_state"]


def _array(values, shape, name, *, nonnegative=False):
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.isfinite(array).all() or (nonnegative and (array < 0).any()):
        raise ValueError(f"Invalid checkpoint array: {name}")
    return array.copy()


ADAPTERS = {"reinforce": ReinforceAlgorithm, "ppo": PPOAlgorithm}


def create_algorithm(algorithm, context, seed=0, config=None):
    validate_algorithm_config(algorithm, config)
    return ADAPTERS[algorithm](context, seed=seed, config=config)


def load_algorithm(path, context):
    path = Path(path)
    if path.stat().st_size > 32_000_000:
        raise ValueError("Training checkpoint exceeds 32 MB")
    try:
        data = json.loads(path.read_text(encoding="utf-8"),
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if data.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("Unsupported native training checkpoint schema")
        adapter = create_algorithm(data["algorithm"], context, data["seed"], data["algorithm_config"])
        _check_contract(data["contract"], adapter.contract)
        for key in ("architecture", "observation_space", "action_space"):
            if data[key] != adapter.metadata[key]:
                raise ValueError(f"Incompatible checkpoint {key}")
        if hashlib.sha256(_json(data["model"]).encode()).hexdigest() != data["model_sha256"]:
            raise ValueError("Checkpoint model checksum mismatch")
        adapter.generation = _number(data["generation"], "generation", 0, 2**53, integer=True)
        adapter._restore(data["model"])
        return adapter
    except (KeyError, TypeError, OverflowError, json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"Invalid native training checkpoint: {error}") from error
