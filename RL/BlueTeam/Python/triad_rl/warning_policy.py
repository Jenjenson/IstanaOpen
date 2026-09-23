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


class WarningPolicy:
    def __init__(self, context, seed=917):
        state, catalogue, _ = public_planning_inputs(context)
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

    def logits(self):
        return self.option_logits.copy()

    def initialize_from_placements(self, placements, *, strength=10.):
        """Bias a fresh policy toward a validated layout without freezing it.

        The warm start changes only the actor's initial logits. Every later
        action is still sampled through the native legal mask and all logits
        remain trainable, so this is an initialization rather than a scripted
        policy or demonstration replay.
        """
        if self.updates or self.baseline is not None or np.any(self.option_logits):
            raise ValueError("A placement warm start requires a fresh warning policy")
        if not isinstance(placements, list) or not placements:
            raise ValueError("Warm-start placements must be a nonempty list")
        if isinstance(strength, bool) or not np.isfinite(strength) or not 0 < strength <= 20:
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
        self.option_logits[selected] = float(strength)
        return {"placements": len(selected), "strength": float(strength),
                "option_indices": selected, "trainable": True}

    def plan(self, context, *, rng=None, deterministic=False):
        state, catalogue, _ = public_planning_inputs(context)
        initial = build_observation(state, catalogue)
        current_contract = {"sites": state["sites"], "catalogue": catalogue,
                            "options": [{"sensor_id": row["sensor_id"], "site_index": row["site_index"],
                                         "yaw_deg": row.get("yaw_deg", 0.), "pitch_deg": row.get("pitch_deg", 0.)}
                                        for row in initial["options"][:-1]]}
        if current_contract != self.contract:
            raise ValueError("Map-specific warning policy requires its original sites and catalogue")
        records, placements = [], []
        generator = self.rng if rng is None else rng
        for _ in range(state["max_sites"] + 1):
            obs = build_observation(state, catalogue)
            mask = obs["action_mask"]
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
        for (records, _), advantage in zip(episodes, advantages):
            for action, p in records:
                score = -p.copy(); score[action] += 1
                gradient += advantage * score
        gradient /= len(episodes)
        gradient /= max(1., float(np.linalg.norm(gradient)))
        self.updates += 1
        self.m = .9 * self.m + .1 * gradient
        self.v = .999 * self.v + .001 * gradient ** 2
        delta = learning_rate * (self.m / (1 - .9 ** self.updates)) / (np.sqrt(self.v / (1 - .999 ** self.updates)) + 1e-8)
        self.option_logits += delta
        self.baseline = float(rewards.mean())
        return {"update": self.updates, "mean_training_warning_s": self.baseline,
                "gradient_norm": float(np.linalg.norm(gradient))}

    def save(self, path):
        data = {"schema": "istana.warning_directional_reinforce.v2", "contract": self.contract,
                "option_logits": self.option_logits.tolist(), "updates": self.updates,
                "baseline": self.baseline, "adam_m": self.m.tolist(), "adam_v": self.v.tolist(),
                "rng_state": deepcopy(self.rng.bit_generator.state)}
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @classmethod
    def load(cls, path, context):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        policy = cls(context)
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
