"""Finite-action RL over public single-sensor edits with matched native returns.

This is an action-value bandit, not PPO. Each arm is a complete legal layout;
feedback is its warning advantage over the unchanged contractor on that route.
The existing local planner supplies geometry and legality only. Its optimizer,
logits and gradients are never used here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np

from .local_refinement_policy import LocalRefinementPPOPolicy, _placement_key


class PairedLayoutBanditPolicy:
    algorithm = "paired_bandit"
    schema = "istana.warning_directional_paired_layout_bandit.v1"
    reward_mode = "paired_contractor_delta"

    def __init__(self, context, seed=917, *, sensor_count=None, entropy_coefficient=.01,
                 allowed_sensor_ids=None, warmup_samples_per_arm=8, exploration_scale_seconds=2.):
        if type(warmup_samples_per_arm) is not int or not 1 <= warmup_samples_per_arm <= 1000:
            raise ValueError("Bandit warmup samples must be an integer in 1..1000")
        if (isinstance(exploration_scale_seconds, bool)
                or not isinstance(exploration_scale_seconds, Real)
                or not math.isfinite(exploration_scale_seconds)
                or not 0 < exploration_scale_seconds <= 1000):
            raise ValueError("Bandit exploration scale must be finite in (0,1000] seconds")
        # Reuse the exact public neighborhood contract without invoking its PPO.
        self._local = LocalRefinementPPOPolicy(context, seed, sensor_count=sensor_count,
            entropy_coefficient=entropy_coefficient, allowed_sensor_ids=allowed_sensor_ids)
        self.allowed_sensor_ids = self._local.allowed_sensor_ids
        self.sensor_count = sensor_count
        self.entropy_coefficient = self._local.entropy_coefficient  # Ignored legacy UI setting.
        self.warmup_samples_per_arm = warmup_samples_per_arm
        self.exploration_scale_seconds = float(exploration_scale_seconds)
        self.rng = np.random.default_rng(seed)
        self.base_placements = None
        self.arms = []
        self.action_space = None
        self._warm_start_report = None
        self.counts = np.zeros(0, dtype=np.int64)
        self.means = np.zeros(0, dtype=float)
        self.m2 = np.zeros(0, dtype=float)
        self.updates = 0
        self._next_reservation_id = 1
        self._pending = {}

    def _require_initialized(self):
        if self.base_placements is None:
            raise ValueError("Initialize the bandit's contractor layout before planning, updating or saving")

    def initialize_from_placements(self, placements, *, baseline_action_probability=None, strength=None):
        if self.base_placements is not None:
            raise ValueError("Bandit initialization requires a fresh policy")
        if strength is not None:
            raise ValueError("Action-value training does not use initial logit strength")
        self._local.initialize_from_placements(placements,
            baseline_action_probability=baseline_action_probability)
        self.base_placements = deepcopy(self._local.base_placements)
        self.sensor_count = self._local.sensor_count
        self.arms = [{"id": 0, "kind": "keep", "slot": None,
                      "placements": deepcopy(self.base_placements)}]
        seen = {tuple(sorted(_placement_key(row) for row in self.base_placements))}
        for slot, candidates in enumerate(self._local.slot_candidates):
            for candidate in candidates[1:]:
                layout = deepcopy(self.base_placements)
                layout[slot] = deepcopy(candidate["placement"])
                key = tuple(sorted(_placement_key(row) for row in layout))
                if key not in seen:
                    seen.add(key)
                    self.arms.append({"id": len(self.arms), "kind": candidate["kind"],
                                      "slot": slot, "placements": layout})
        self.counts = np.zeros(len(self.arms), dtype=np.int64)
        self.means = np.zeros(len(self.arms), dtype=float)
        self.m2 = np.zeros(len(self.arms), dtype=float)
        self.action_space = {
            "kind": "paired_layout_bandit", "arms": len(self.arms), "actions": len(self.arms),
            "slots": self.sensor_count, "contractorArm": 0, "fixedInventory": True,
            "initialSamplesPerAlternative": self.warmup_samples_per_arm,
            "explorationScaleSeconds": self.exploration_scale_seconds,
            "ucbFormula": "mean_delta_s + scale_s * sqrt(2 * log(max(2, total_alternative_allocations)) / arm_allocations)",
            "allocationDefinition": "observed samples plus pending reservations; pending rewards are not observations",
            "uncertaintyNote": "The exploration scale is a fixed heuristic, not a confidence-interval guarantee. Standard errors describe observed samples only.",
            "candidateGeneration": deepcopy(self._local.action_space),
            "limitation": "Contractor plus single-sensor edits only; no automatic combinations or summed edit gains."}
        self._warm_start_report = {
            "placements": self.sensor_count, "trainable": True, "initialization": "balanced_action_value",
            "calibration": "KEEP has known zero advantage; alternative values are learned from paired native returns.",
            "legacy_baseline_action_probability_ignored": baseline_action_probability,
            "legacy_entropy_coefficient_ignored": self.entropy_coefficient,
            "actionSpace": deepcopy(self.action_space)}
        return deepcopy(self._warm_start_report)

    def _legal_mask(self, context):
        self._require_initialized()
        state, catalogue = self._local._planning_inputs(context)
        mask = np.array([self._local._layout_legal(arm["placements"], state, catalogue)
                         for arm in self.arms], dtype=bool)
        if not mask[0]:
            raise ValueError("Bandit contractor layout is no longer legal in the native scene")
        return mask

    def _reservations(self):
        result = np.zeros(len(self.arms), dtype=np.int64)
        for record in self._pending.values():
            result[record["action"]] += 1
        return result

    def _standard_error(self, arm):
        if arm == 0:
            return 0.  # Identical-layout advantage is known exactly.
        count = int(self.counts[arm])
        return math.sqrt(self.m2[arm] / (count - 1) / count) if count >= 2 else None

    def plan(self, context, *, rng=None, deterministic=False):
        mask = self._legal_mask(context)
        if deterministic:
            # An unobserved alternative cannot displace the known zero control.
            scores = np.where(mask & (self.counts > 0), self.means, -np.inf)
            scores[0] = 0.
            arm = int(np.argmax(scores))
            return deepcopy(self.arms[arm]["placements"]), []
        allocations = self.counts + self._reservations()
        alternatives = np.flatnonzero(mask & (np.arange(len(mask)) != 0))
        ucb_score = None
        if not len(alternatives):
            eligible, selection = np.array([0]), "forced_keep"
        else:
            incomplete = alternatives[allocations[alternatives] < self.warmup_samples_per_arm]
            if len(incomplete):
                eligible = incomplete[allocations[incomplete] == allocations[incomplete].min()]
                selection = "balanced"
            else:
                total = max(2, int(allocations[1:].sum()))
                scores = self.means[alternatives] + self.exploration_scale_seconds * np.sqrt(
                    2 * math.log(total) / allocations[alternatives])
                maximum = float(scores.max())
                eligible = alternatives[np.isclose(scores, maximum, atol=1e-12, rtol=0)]
                selection, ucb_score = "ucb", maximum
        probabilities = np.zeros(len(self.arms), dtype=float)
        probabilities[eligible] = 1. / len(eligible)
        generator = self.rng if rng is None else rng
        arm = int(generator.choice(len(probabilities), p=probabilities))
        record = {"action": arm, "probabilities": probabilities, "mask": mask,
            "reservationId": self._next_reservation_id, "selection": selection,
            "observed_count": int(self.counts[arm]),
            "estimated_delta_s": float(self.means[arm]) if self.counts[arm] or arm == 0 else None,
            "standard_error_s": self._standard_error(arm), "ucb_score_s": ucb_score,
            "update": self.updates}
        self._pending[self._next_reservation_id] = self._record_data(record)
        self._next_reservation_id += 1
        return deepcopy(self.arms[arm]["placements"]), [record]

    def _record_data(self, record):
        keys = {"action", "probabilities", "mask", "reservationId", "selection", "observed_count",
                "estimated_delta_s", "standard_error_s", "ucb_score_s", "update"}
        if not isinstance(record, dict) or set(record) != keys:
            raise ValueError("Invalid bandit action record")
        action, reservation = record["action"], record["reservationId"]
        p, mask = np.asarray(record["probabilities"], dtype=float), np.asarray(record["mask"])
        if (type(action) is not int or not 0 <= action < len(self.arms)
                or type(reservation) is not int or reservation < 1
                or mask.dtype != np.bool_ or mask.shape != self.means.shape
                or p.shape != self.means.shape or not np.isfinite(p).all() or np.any(p < 0)
                or not np.isclose(p.sum(), 1., atol=1e-12, rtol=0)
                or not mask[action] or p[action] <= 0 or np.any(p[~mask])
                or record["selection"] not in ("balanced", "ucb", "forced_keep")
                or type(record["observed_count"]) is not int or record["observed_count"] < 0
                or type(record["update"]) is not int or record["update"] < 0):
            raise ValueError("Invalid bandit sampling record")
        for key in ("estimated_delta_s", "standard_error_s", "ucb_score_s"):
            value = record[key]
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real)
                                      or not math.isfinite(value)):
                raise ValueError("Invalid bandit sampling estimate")
        result = deepcopy(record)
        result["probabilities"], result["mask"] = p.tolist(), mask.tolist()
        return result

    def episode_diagnostics(self, records):
        self._require_initialized()
        if not isinstance(records, list) or len(records) != 1:
            raise ValueError("Bandit episodes require exactly one arm decision")
        record = self._record_data(records[0])
        return {"armId": record["action"], "reservationId": record["reservationId"],
            "selectionMode": record["selection"], "observedCount": record["observed_count"],
            "estimatedAdvantageSeconds": record["estimated_delta_s"],
            "standardErrorSeconds": record["standard_error_s"], "ucbScoreSeconds": record["ucb_score_s"],
            "samplingProbability": record["probabilities"][record["action"]],
            "legalArms": sum(record["mask"]), "update": record["update"]}

    def update(self, episodes):
        self._require_initialized()
        if not isinstance(episodes, (list, tuple)) or not episodes:
            raise ValueError("Bandit updates require complete paired episodes")
        validated, seen = [], set()
        for records, reward in episodes:
            if (not isinstance(records, list) or len(records) != 1
                    or isinstance(reward, bool) or not isinstance(reward, Real) or not math.isfinite(reward)):
                raise ValueError("Bandit updates need one reserved decision and a finite paired return")
            record = self._record_data(records[0])
            reservation = record["reservationId"]
            if (reservation in seen or self._pending.get(reservation) != record):
                raise ValueError("Bandit decision is duplicated, modified or not pending")
            if record["action"] == 0 and abs(reward) > 1e-9:
                raise ValueError("The unchanged contractor must have zero paired advantage")
            seen.add(reservation)
            validated.append((record, float(reward)))
        before = self.means.copy()
        for record, reward in validated:
            arm = record["action"]
            if arm == 0:
                reward = 0.
            self.counts[arm] += 1
            delta = reward - self.means[arm]
            self.means[arm] += delta / int(self.counts[arm])
            self.m2[arm] += delta * (reward - self.means[arm])
            del self._pending[record["reservationId"]]
        self.updates += 1
        return {"algorithm": self.algorithm, "update": self.updates, "reward_mode": self.reward_mode,
            "mean_training_advantage_s": float(np.mean([reward for _, reward in validated])),
            "observation_count": int(self.counts.sum()), "pending_reservations": len(self._pending),
            "arm_observation_counts": self.counts.tolist(),
            "arm_mean_advantage_s": [float(value) if self.counts[arm] or arm == 0 else None
                                     for arm, value in enumerate(self.means)],
            "arm_standard_error_s": [self._standard_error(arm) for arm in range(len(self.arms))],
            "selected_arms": [record["action"] for record, _ in validated],
            "selection_mode": sorted({record["selection"] for record, _ in validated}),
            "parameter_delta_norm": float(np.linalg.norm(self.means - before)),
            "parameter_delta_definition": "L2 change in observed arm mean advantage estimates; no gradient optimizer",
            "action_space": deepcopy(self.action_space)}

    def save(self, path):
        self._require_initialized()
        data = {"schema": self.schema, "algorithm": self.algorithm, "reward_mode": self.reward_mode,
            "contract": self._local.contract, "sensor_count": self.sensor_count,
            "allowed_sensor_ids": list(self.allowed_sensor_ids), "entropy_coefficient": self.entropy_coefficient,
            "warmup_samples_per_arm": self.warmup_samples_per_arm,
            "exploration_scale_seconds": self.exploration_scale_seconds,
            "base_placements": self.base_placements, "arms": self.arms,
            "action_space": self.action_space, "warm_start_report": self._warm_start_report,
            "counts": self.counts.tolist(), "means": self.means.tolist(), "m2": self.m2.tolist(),
            "updates": self.updates, "next_reservation_id": self._next_reservation_id,
            "pending": [deepcopy(value) for _, value in sorted(self._pending.items())],
            "rng_state": deepcopy(self.rng.bit_generator.state)}
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @classmethod
    def load(cls, path, context):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if (data.get("schema") != cls.schema or data.get("algorithm") != cls.algorithm
                or data.get("reward_mode") != cls.reward_mode):
            raise ValueError("Wrong paired bandit policy schema or reward mode")
        policy = cls(context, sensor_count=data["sensor_count"], entropy_coefficient=data["entropy_coefficient"],
            allowed_sensor_ids=data["allowed_sensor_ids"], warmup_samples_per_arm=data["warmup_samples_per_arm"],
            exploration_scale_seconds=data["exploration_scale_seconds"])
        if data.get("contract") != policy._local.contract:
            raise ValueError("Wrong paired bandit native geometry contract")
        report = data["warm_start_report"]
        policy.initialize_from_placements(data["base_placements"],
            baseline_action_probability=report.get("legacy_baseline_action_probability_ignored"))
        if (data.get("arms") != policy.arms or data.get("action_space") != policy.action_space
                or report != policy._warm_start_report):
            raise ValueError("Saved bandit arms do not match the recomputed public contract")
        counts = data["counts"]
        if (not isinstance(counts, list) or len(counts) != len(policy.arms)
                or any(type(count) is not int or not 0 <= count < 2**63 for count in counts)):
            raise ValueError("Invalid bandit observation counts")
        policy.counts = np.asarray(counts, dtype=np.int64)
        for key in ("means", "m2"):
            values = np.asarray(data[key], dtype=float)
            if values.shape != policy.means.shape or not np.isfinite(values).all():
                raise ValueError("Invalid bandit value statistics")
            setattr(policy, key, values)
        if (np.any(policy.m2 < 0) or np.any(policy.m2[policy.counts < 2])
                or np.any(policy.means[policy.counts == 0]) or policy.means[0] != 0 or policy.m2[0] != 0):
            raise ValueError("Bandit moments conflict with observation counts or known KEEP value")
        updates, next_id = data["updates"], data["next_reservation_id"]
        if (type(updates) is not int or not 0 <= updates <= sum(counts)
                or type(next_id) is not int or next_id < 1 or not isinstance(data["pending"], list)):
            raise ValueError("Invalid bandit update or reservation counters")
        for source in data["pending"]:
            record = policy._record_data(source)
            reservation = record["reservationId"]
            if (reservation >= next_id or reservation in policy._pending
                    or record["observed_count"] > counts[record["action"]] or record["update"] > updates):
                raise ValueError("Invalid pending bandit reservation")
            policy._pending[reservation] = record
        if sum(counts) + len(policy._pending) != next_id - 1:
            raise ValueError("Bandit observed and pending counts do not match issued reservations")
        policy.updates, policy._next_reservation_id = updates, next_id
        policy.rng.bit_generator.state = data["rng_state"]
        return policy
