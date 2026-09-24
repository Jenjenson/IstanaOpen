"""Map-specific, masked REINFORCE sensor-layout policy for native warning reward.

Separate from frozen temporal experiments. No private Red input: the actor sees
only the validated public layout and legal action mask. Each logit represents a
profile/site/yaw/pitch choice; STOP is separately learnable.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .directional_inputs import build_observation, apply_placement
from .istana_live import public_planning_inputs


def _entropy_and_gradient(probabilities):
    """Categorical entropy and its exact logit gradient, excluding masked actions."""
    p = np.asarray(probabilities, dtype=float)
    if (p.ndim != 1 or not np.isfinite(p).all() or np.any(p < 0)
            or not np.isclose(p.sum(), 1., atol=1e-8, rtol=0)):
        raise ValueError("Probabilities must be a finite categorical distribution")
    legal = p > 0
    logs = np.zeros_like(p)
    logs[legal] = np.log(p[legal])
    entropy = -float(np.dot(p, logs))
    gradient = np.zeros_like(p)
    gradient[legal] = -p[legal] * (logs[legal] + entropy)
    return entropy, gradient


def categorical_entropy(probabilities):
    """Entropy in nats; forced actions have zero entropy."""
    return _entropy_and_gradient(probabilities)[0]


class WarningPolicy:
    def __init__(self, context, seed=917, *, sensor_count=None, entropy_coefficient=.01,
                 allowed_sensor_ids=None):
        if (isinstance(entropy_coefficient, bool)
                or not isinstance(entropy_coefficient, (int, float))
                or not np.isfinite(entropy_coefficient) or not 0 <= entropy_coefficient <= 1):
            raise ValueError("Entropy coefficient must be finite in [0, 1]")
        self.entropy_coefficient = float(entropy_coefficient)
        state, catalogue, _ = public_planning_inputs(context)
        self.allowed_sensor_ids = None
        if allowed_sensor_ids is not None:
            if (not isinstance(allowed_sensor_ids, (list, tuple)) or not allowed_sensor_ids
                    or any(not isinstance(value, str) for value in allowed_sensor_ids)
                    or len(set(allowed_sensor_ids)) != len(allowed_sensor_ids)):
                raise ValueError("Allowed sensor IDs must be a nonempty list of unique profile IDs")
            unknown = set(allowed_sensor_ids) - {row["id"] for row in catalogue}
            if unknown:
                raise ValueError("Unknown allowed sensor profile: " + ", ".join(sorted(unknown)))
            self.allowed_sensor_ids = tuple(sorted(allowed_sensor_ids))
            self._restrict_sensors(state)
        if sensor_count is not None:
            if (type(sensor_count) is not int or not 1 <= sensor_count <= state["max_sites"]):
                raise ValueError(
                    f"Sensor count must be an integer from 1 to the native limit of {state['max_sites']}")
            minimum_cost = min(row["cost"] for row in catalogue
                               if row["id"] in state["available_sensor_ids"])
            if sensor_count * minimum_cost > state["budget_remaining"] + 1e-9:
                raise ValueError("Native budget cannot fund the requested sensor count")
        self.sensor_count = sensor_count
        self._initial_state = deepcopy(state)
        if sensor_count is not None:
            self._initial_state["max_sites"] = sensor_count
        self._warm_start_report = None
        observation = build_observation(state, catalogue)
        option_contract = [{"sensor_id": row["sensor_id"], "site_index": row["site_index"],
                            "yaw_deg": row.get("yaw_deg", 0.), "pitch_deg": row.get("pitch_deg", 0.)}
                           for row in observation["options"][:-1]]
        self.contract = {"sites": state["sites"], "catalogue": catalogue, "options": option_contract}
        self.n_sites, self.n_types = len(state["sites"]), len(catalogue)
        self.option_logits = np.zeros(len(option_contract) + 1)
        self.rng = np.random.default_rng(seed)
        self.updates = 0
        self.baseline = None
        self.m = np.zeros_like(self.option_logits)
        self.v = self.m.copy()

    def _restrict_sensors(self, state):
        """Keep the saved training catalogue restriction in every later episode."""
        if self.allowed_sensor_ids is not None:
            unavailable = set(self.allowed_sensor_ids) - set(state["available_sensor_ids"])
            if unavailable:
                raise ValueError("Training-selected sensor profile is unavailable in the native scene: "
                                 + ", ".join(sorted(unavailable)))
            state["available_sensor_ids"] = list(self.allowed_sensor_ids)

    def _planning_inputs(self, context):
        state, catalogue, _ = public_planning_inputs(context)
        self._restrict_sensors(state)
        initial = build_observation(state, catalogue)
        current_contract = {"sites": state["sites"], "catalogue": catalogue,
                            "options": [{"sensor_id": row["sensor_id"],
                                         "site_index": row["site_index"],
                                         "yaw_deg": row.get("yaw_deg", 0.),
                                         "pitch_deg": row.get("pitch_deg", 0.)}
                                        for row in initial["options"][:-1]]}
        if current_contract != self.contract:
            raise ValueError("Map-specific warning policy requires its original sites and catalogue")
        if self.sensor_count is not None:
            if state["max_sites"] < self.sensor_count:
                raise ValueError(
                    f"Native scene allows only {state['max_sites']} sensors; "
                    f"this policy requires {self.sensor_count}")
            state["max_sites"] = self.sensor_count
        return state, catalogue

    def _action_mask(self, state, catalogue, observation):
        """Apply the optional exact-count contract without bypassing legality.

        STOP remains part of the actor contract, but an exact-count training run
        masks it until the requested number has been placed. Costlier choices
        that would leave too little budget for the remaining slots are masked as
        well, so a sampled action cannot make the requested count impossible on
        the next step.
        """
        mask = np.asarray(observation["action_mask"], dtype=bool).copy()
        if self.sensor_count is None:
            return mask
        placed = len(state["placements"])
        remaining = self.sensor_count - placed
        if remaining <= 0:
            mask[:-1] = False
        else:
            mask[-1] = False
            if remaining > 1:
                minimum_cost = min(row["cost"] for row in catalogue
                                   if row["id"] in state["available_sensor_ids"])
                for index, option in enumerate(observation["options"][:-1]):
                    if mask[index]:
                        cost = catalogue[option["sensor_index"]]["cost"]
                        if cost + (remaining - 1) * minimum_cost > state["budget_remaining"] + 1e-9:
                            mask[index] = False
        if not mask.any():
            raise RuntimeError(
                f"No legal placement can complete the requested {self.sensor_count}-sensor layout")
        return mask

    def logits(self):
        return self.option_logits.copy()

    def initialize_from_placements(self, placements, *, baseline_action_probability=.8,
                                   strength=None):
        """Bias a fresh policy toward a validated layout without freezing it.

        The warm start changes only the actor's initial logits. Every later
        action is still sampled through the native legal mask and all logits
        remain trainable. By default the first legal decision puts the requested
        probability mass on the starting layout, rather than assigning it an
        arbitrary near-deterministic logit gap. Later masks change that mass as
        sites are occupied. An explicit strength supports historical callers.
        """
        if (self.updates or self.baseline is not None or np.any(self.option_logits)
                or self._warm_start_report is not None):
            raise ValueError("A placement warm start requires a fresh warning policy")
        if not isinstance(placements, list) or not placements:
            raise ValueError("Warm-start placements must be a nonempty list")
        if self.sensor_count is not None and len(placements) != self.sensor_count:
            raise ValueError(
                f"Warm-start placement must contain exactly {self.sensor_count} sensors")
        if (isinstance(baseline_action_probability, bool)
                or not isinstance(baseline_action_probability, (int, float))
                or not np.isfinite(baseline_action_probability)
                or not 0 < baseline_action_probability < 1):
            raise ValueError("Starting-layout action probability must be finite in (0, 1)")
        if strength is not None and (
                isinstance(strength, bool) or not isinstance(strength, (int, float))
                or not np.isfinite(strength) or not 0 < strength <= 20):
            raise ValueError("Warm-start strength must be finite in (0, 20]")
        by_option = {
            (row["sensor_id"], row["site_index"], row["yaw_deg"], row["pitch_deg"]): index
            for index, row in enumerate(self.contract["options"])
        }
        selected = []
        for number, row in enumerate(placements, 1):
            if not isinstance(row, dict):
                raise ValueError(f"Warm-start placement {number} must be an object")
            key = (row.get("profileId"), row.get("siteId"),
                   float(row.get("yawDeg", 0.)), float(row.get("pitchDeg", 0.)))
            if key not in by_option:
                raise ValueError(f"Warm-start placement {number} is outside the policy contract")
            index = by_option[key]
            if index in selected:
                raise ValueError("Warm-start placements must be unique")
            selected.append(index)
        # Check the complete seed layout against the same budget, selected
        # profiles, site separation and exact-count mask used during sampling.
        catalogue = self.contract["catalogue"]
        current = deepcopy(self._initial_state)
        first_mask = None
        for number, action in enumerate(selected, 1):
            observation = build_observation(current, catalogue)
            mask = self._action_mask(current, catalogue, observation)
            if first_mask is None:
                first_mask = mask
            if not mask[action]:
                raise ValueError(f"Warm-start placement {number} is not legal in the selected sensor configuration")
            current = apply_placement(current, action, catalogue)
        legal_seed = int(first_mask[selected].sum())
        legal_other = int(first_mask.sum()) - legal_seed
        calibrated = strength is None
        if strength is None:
            if legal_other:
                odds = baseline_action_probability / (1 - baseline_action_probability)
                strength = float(np.log(odds * legal_other / legal_seed))
            else:
                # No exploration is possible if the native legal mask offers
                # no alternative; report that rather than inventing an action.
                strength = 0.
        self.option_logits[selected] = float(strength)
        actual_probability = (legal_seed * np.exp(strength)
                              / (legal_seed * np.exp(strength) + legal_other))
        self._warm_start_report = {
            "placements": len(selected), "strength": float(strength),
            "option_indices": selected, "trainable": True,
            "target_initial_baseline_action_probability": (
                float(baseline_action_probability) if calibrated else None),
            "initial_baseline_action_probability": float(actual_probability),
            "initial_exploration_probability": float(1 - actual_probability),
            "legal_baseline_actions": legal_seed, "legal_alternative_actions": legal_other,
            "calibration": ("first legal placement decision; later masks may change probability mass"
                            if calibrated else "explicit logit strength"),
        }
        return deepcopy(self._warm_start_report)

    def plan(self, context, *, rng=None, deterministic=False):
        state, catalogue = self._planning_inputs(context)
        records, placements = [], []
        generator = self.rng if rng is None else rng
        for _ in range(state["max_sites"] + 1):
            obs = build_observation(state, catalogue)
            mask = self._action_mask(state, catalogue, obs)
            logits = self.logits()
            p = np.zeros(len(mask))
            p[mask] = np.exp(logits[mask] - logits[mask].max())
            p /= p.sum()
            action = int(np.argmax(np.where(mask, logits, -np.inf))) if deterministic else int(generator.choice(len(p), p=p))
            records.append((action, p.copy()))
            row = obs["options"][action]
            state = apply_placement(state, action, catalogue)
            if row["stop"]:
                return placements, records
            placements.append({"profileId": row["sensor_id"], "siteId": row["site_index"],
                               "yawDeg": row.get("yaw_deg", 0.), "pitchDeg": row.get("pitch_deg", 0.)})
        raise RuntimeError("Policy failed to STOP within the layout limit")

    def update(self, episodes, learning_rate=.12):
        rewards = np.asarray([reward for _, reward in episodes], dtype=float)
        if not len(rewards) or not np.isfinite(rewards).all():
            raise ValueError("Need finite complete-episode rewards")
        # Leave-one-out batch baseline is independent of each episode's action.
        # Scale is fixed in seconds (no seed-dependent reward normalization).
        advantages = (rewards - (rewards.sum() - rewards) / (len(rewards) - 1)) if len(rewards) > 1 else rewards - (self.baseline or 0.)
        gradient = np.zeros_like(self.option_logits)
        entropy_gradient = np.zeros_like(gradient)
        entropies, policy_loss = [], 0.
        for (records, _), advantage in zip(episodes, advantages):
            for action, p in records:
                p = np.asarray(p, dtype=float)
                entropy, regularizer = _entropy_and_gradient(p)
                if (p.shape != self.option_logits.shape or type(action) is not int
                        or not 0 <= action < len(p) or p[action] <= 0):
                    raise ValueError("Invalid policy action record")
                score = -p.copy(); score[action] += 1
                gradient += advantage * score
                policy_loss -= advantage * float(np.log(p[action]))
                entropy_gradient += regularizer
                entropies.append(entropy)
        gradient /= len(episodes)
        policy_loss /= len(episodes)
        entropy = float(np.mean(entropies)) if entropies else 0.
        gradient += self.entropy_coefficient * entropy_gradient / max(1, len(entropies))
        gradient_norm = float(np.linalg.norm(gradient))
        gradient /= max(1., gradient_norm)
        self.updates += 1
        self.m = .9 * self.m + .1 * gradient
        self.v = .999 * self.v + .001 * gradient ** 2
        delta = learning_rate * (self.m / (1 - .9 ** self.updates)) / (np.sqrt(self.v / (1 - .999 ** self.updates)) + 1e-8)
        self.option_logits += delta
        self.baseline = float(rewards.mean())
        return {"update": self.updates, "mean_training_warning_s": self.baseline,
                "gradient_norm": gradient_norm, "parameter_delta_norm": float(np.linalg.norm(delta)),
                "policy_loss": float(policy_loss), "entropy": entropy,
                "entropy_coefficient": self.entropy_coefficient,
                "objective_loss": float(policy_loss - self.entropy_coefficient * entropy)}

    def save(self, path):
        data = {"schema": "istana.warning_directional_reinforce.v2", "contract": self.contract,
                "sensor_count": self.sensor_count,
                "entropy_coefficient": self.entropy_coefficient,
                "allowed_sensor_ids": (list(self.allowed_sensor_ids)
                                       if self.allowed_sensor_ids is not None else None),
                "option_logits": self.option_logits.tolist(), "updates": self.updates,
                "baseline": self.baseline, "adam_m": self.m.tolist(), "adam_v": self.v.tolist(),
                "rng_state": deepcopy(self.rng.bit_generator.state)}
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @classmethod
    def load(cls, path, context):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        sensor_count = data.get("sensor_count")
        policy = cls(context, sensor_count=sensor_count,
                     entropy_coefficient=data.get("entropy_coefficient", 0.),
                     allowed_sensor_ids=data.get("allowed_sensor_ids"))
        if data.get("schema") != "istana.warning_directional_reinforce.v2" or data["contract"] != policy.contract:
            raise ValueError("Wrong warning policy contract")
        for name in ("option_logits", "m", "v"):
            value = np.asarray(data[{"m": "adam_m", "v": "adam_v"}.get(name, name)], dtype=float)
            if value.shape != getattr(policy, name).shape or not np.isfinite(value).all():
                raise ValueError("Invalid policy arrays")
            setattr(policy, name, value)
        policy.updates, policy.baseline = data["updates"], data["baseline"]
        policy.rng.bit_generator.state = data["rng_state"]
        return policy


def warning_metrics(observation):
    """Recompute native terminal evidence, counting undetected arrivals as zero.

    Fail closed on unresolved episodes rather than training on optimistic subsets.
    Team warning = first zone arrival - first detection of ANY Red drone.
    """
    if not observation.get("metricsAvailable"):
        raise ValueError("Terminal warning evidence required; rebuild native editor if missing")
    rows = observation.get("warningEvidenceForEvaluationOnly")
    if not rows or any(row["zoneEntrySeconds"] is None for row in rows):
        raise ValueError("Missing or unresolved warning evidence")
    warnings, detections, arrivals = [], [], []
    ids = set()
    for row in rows:
        if row["droneId"] in ids:
            raise ValueError("Duplicate target")
        ids.add(row["droneId"])
        first, arrival = row["firstDetectionSeconds"], row["zoneEntrySeconds"]
        if not np.isfinite(arrival) or arrival < 0 or (first is not None and (not np.isfinite(first) or first < 0)):
            raise ValueError("Invalid event time")
        value = max(0., arrival - first) if first is not None else 0.
        if not np.isclose(value, row["warningSeconds"], atol=1e-7, rtol=0):
            raise ValueError("Native/Python warning mismatch")
        warnings.append(value); arrivals.append(arrival)
        if first is not None:
            detections.append(first)
    team = max(0., min(arrivals) - min(detections)) if detections else 0.
    mean = float(np.mean(warnings))
    for key, value in (("mean_drone_warning_seconds_lower_bound", mean), ("team_warning_seconds_lower_bound", team)):
        if not np.isclose(observation["metrics"][key], value, atol=1e-7, rtol=0):
            raise ValueError("Native aggregate warning mismatch")
    return {"mean_drone_warning_s": mean, "team_warning_s": team,
            "first_detection_s": min(detections) if detections else None,
            "first_arrival_s": min(arrivals), "detected_fraction": len(detections) / len(rows),
            "targets": len(rows), "unresolved": 0, "cost": observation["metrics"]["cost"]}
