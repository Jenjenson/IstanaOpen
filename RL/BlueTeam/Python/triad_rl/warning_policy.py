"""Map-specific, masked REINFORCE sensor-layout policy for native warning reward.

Separate from frozen temporal experiments. No private Red input: the actor sees
only the validated public layout and legal action mask. Type preferences are
shared across sites; site preferences are learned per type. STOP is learnable.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .adaptive_inputs import build_observation, apply_placement
from .istana_live import public_planning_inputs


class WarningPolicy:
    def __init__(self, context, seed=917):
        state, catalogue, _ = public_planning_inputs(context)
        self.contract = {"sites": state["sites"], "catalogue": catalogue}
        self.n_sites, self.n_types = len(state["sites"]), len(catalogue)
        self.types = np.zeros(self.n_types + 1)
        self.sites = np.zeros((self.n_types, self.n_sites))
        self.rng = np.random.default_rng(seed)
        self.updates = 0
        self.baseline = None
        self.m = np.zeros(self.n_types + 1 + self.n_types * self.n_sites)
        self.v = self.m.copy()

    def logits(self):
        return np.r_[(self.types[:-1, None] + self.sites).ravel(), self.types[-1]]

    def plan(self, context, *, rng=None, deterministic=False):
        state, catalogue, _ = public_planning_inputs(context)
        if {"sites": state["sites"], "catalogue": catalogue} != self.contract:
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
            placements.append({"profileId": row["sensor_id"], "siteId": row["site_index"]})
        raise RuntimeError("Policy failed to STOP within the layout limit")

    def update(self, episodes, learning_rate=.12):
        rewards = np.asarray([reward for _, reward in episodes], dtype=float)
        if not len(rewards) or not np.isfinite(rewards).all():
            raise ValueError("Need finite complete-episode rewards")
        # Leave-one-out batch baseline is independent of each episode's action.
        # Scale is fixed in seconds (no seed-dependent reward normalization).
        advantages = (rewards - (rewards.sum() - rewards) / (len(rewards) - 1)) if len(rewards) > 1 else rewards - (self.baseline or 0.)
        gt, gs = np.zeros_like(self.types), np.zeros_like(self.sites)
        for (records, _), advantage in zip(episodes, advantages):
            for action, p in records:
                score = -p.copy(); score[action] += 1
                gs += advantage * score[:-1].reshape(self.sites.shape)
                gt[:-1] += advantage * score[:-1].reshape(self.sites.shape).sum(axis=1)
                gt[-1] += advantage * score[-1]
        gradient = np.r_[gt, gs.ravel()] / len(episodes)
        gradient /= max(1., float(np.linalg.norm(gradient)))
        self.updates += 1
        self.m = .9 * self.m + .1 * gradient
        self.v = .999 * self.v + .001 * gradient ** 2
        delta = learning_rate * (self.m / (1 - .9 ** self.updates)) / (np.sqrt(self.v / (1 - .999 ** self.updates)) + 1e-8)
        self.types += delta[:len(self.types)]
        self.sites += delta[len(self.types):].reshape(self.sites.shape)
        self.baseline = float(rewards.mean())
        return {"update": self.updates, "mean_training_warning_s": self.baseline,
                "gradient_norm": float(np.linalg.norm(gradient))}

    def save(self, path):
        data = {"schema": "istana.warning_reinforce.v1", "contract": self.contract,
                "types": self.types.tolist(), "sites": self.sites.tolist(), "updates": self.updates,
                "baseline": self.baseline, "adam_m": self.m.tolist(), "adam_v": self.v.tolist(),
                "rng_state": deepcopy(self.rng.bit_generator.state)}
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @classmethod
    def load(cls, path, context):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        policy = cls(context)
        if data.get("schema") != "istana.warning_reinforce.v1" or data["contract"] != policy.contract:
            raise ValueError("Wrong warning policy contract")
        for name in ("types", "sites", "m", "v"):
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
