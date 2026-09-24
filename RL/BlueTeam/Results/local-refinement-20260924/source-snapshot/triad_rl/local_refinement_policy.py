"""Versioned PPO over small, legal per-sensor edits to a public starting layout.

The full layout is present before the first decision. Each sensor has its own
KEEP, nearby-site, and neighboring-orientation choices. Earlier edits change
later legality masks; they never remove a sensor or relax the native contract.
The actor receives no Red scenario or reward evidence during planning.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .istana_live import public_planning_inputs
from .warning_algorithms import MaskedPPOPolicy


def _placement_key(row):
    return row["profileId"], row["siteId"], row["yawDeg"], row["pitchDeg"]


class LocalRefinementPPOPolicy(MaskedPPOPolicy):
    """Factorized local edit actor using the existing exact categorical PPO update."""

    algorithm = "local_ppo"
    schema = "istana.warning_directional_local_ppo.v1"
    reward_mode = "paired_contractor_delta"
    nearby_site_count = 2

    def __init__(self, context, seed=917, *, sensor_count=None, entropy_coefficient=.01,
                 allowed_sensor_ids=None):
        state, catalogue, _ = public_planning_inputs(context)
        if allowed_sensor_ids is None:
            allowed_sensor_ids = [row["id"] for row in catalogue if row.get("directional")
                                  and row["id"] in state["available_sensor_ids"]]
        super().__init__(context, seed=seed, sensor_count=sensor_count,
                         entropy_coefficient=entropy_coefficient,
                         allowed_sensor_ids=allowed_sensor_ids)
        directional = {row["id"] for row in catalogue if row.get("directional")}
        if not set(self.allowed_sensor_ids) <= directional:
            raise ValueError("Local refinement requires selected limited-FOV sensor profiles")
        self._geometry_options = {
            (row["sensor_id"], row["site_index"], row["yaw_deg"], row["pitch_deg"])
            for row in self.contract["options"]}
        self.base_placements = None
        self.slot_candidates = []
        self._slot_offsets = []
        self.action_space = None

    def _require_initialized(self):
        if self.base_placements is None:
            raise ValueError("Initialize the local refinement starting layout before planning, updating or saving")

    def _planning_inputs(self, context):
        state, catalogue, _ = public_planning_inputs(context)
        self._restrict_sensors(state)
        # Site and catalogue equality imply the same deterministic geometric
        # option list. Do not compute unused global coverage features per edit.
        if state["sites"] != self.contract["sites"] or catalogue != self.contract["catalogue"]:
            raise ValueError("Map-specific local policy requires its original sites and catalogue")
        if self.sensor_count is not None:
            if state["max_sites"] < self.sensor_count:
                raise ValueError("Native scene has too few sensor slots for this local policy")
            state["max_sites"] = self.sensor_count
        return state, catalogue

    def _canonical_placements(self, placements):
        if not isinstance(placements, list) or not placements:
            raise ValueError("Local refinement requires a nonempty starting layout")
        result = []
        for row in placements:
            if (not isinstance(row, dict) or not isinstance(row.get("profileId"), str)
                    or type(row.get("siteId")) is not int):
                raise ValueError("Starting layout requires profileId and an integer siteId")
            pose = {"profileId": row["profileId"], "siteId": row["siteId"]}
            for key in ("yawDeg", "pitchDeg"):
                value = row.get(key, 0.)
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value)):
                    raise ValueError("Starting sensor angles must be finite numbers")
                pose[key] = float(value)
            if _placement_key(pose) not in self._geometry_options:
                raise ValueError("Starting placement is outside the native geometry contract")
            result.append(pose)
        return result

    def _layout_legal(self, placements, state, catalogue):
        """Whole-layout equivalent of directional_inputs' public placement mask.

        Costs are positive, so total affordability implies affordability of
        every prefix. Pairwise separation and individually allowed geometry
        likewise imply legal sequential deployment. Native deploy still checks
        the authoritative terrain/support contract represented by blocked sites.
        """
        if (state["done"] or state["placements"] or not placements
                or len(placements) > state["max_sites"]):
            return False
        sensors = {row["id"]: row for row in catalogue}
        allowed, blocked = set(state["available_sensor_ids"]), set(state["blocked_sites"])
        positions, total_cost = [], 0.
        for row in placements:
            if (_placement_key(row) not in self._geometry_options
                    or row["profileId"] not in allowed or row["siteId"] in blocked):
                return False
            position = state["sites"][row["siteId"]]
            radius = math.hypot(*position)
            if not (state["deployment_min_radius"] - 1e-9 <= radius
                    <= state["deployment_max_radius"] + 1e-9):
                return False
            for previous in positions:
                distance = math.dist(position, previous)
                if distance < state["min_separation"] - 1e-9 or distance <= 1e-7:
                    return False
            positions.append(position)
            total_cost += sensors[row["profileId"]]["cost"]
        return total_cost <= state["budget_remaining"] + 1e-9

    def _candidates(self, base, state, catalogue):
        sensors = {row["id"]: row for row in catalogue}
        slots = []
        for slot, original in enumerate(base):
            candidates = [{"kind": "keep", "placement": deepcopy(original)}]
            seen = {_placement_key(original)}

            def add(kind, replacement):
                key = _placement_key(replacement)
                rows = deepcopy(base)
                rows[slot] = replacement
                if key not in seen and self._layout_legal(rows, state, catalogue):
                    seen.add(key)
                    candidates.append({"kind": kind, "placement": deepcopy(replacement)})
                    return True
                return False

            profile = sensors[original["profileId"]]
            # Yaw wraps; pitch does not. One adjacent choice in each direction.
            for axis, field, wraps in (("yaw", "yawDeg", True), ("pitch", "pitchDeg", False)):
                angles = sorted(profile[f"{axis}_bins_deg"])
                current = angles.index(original[field])
                for offset in (-1, 1):
                    index = current + offset
                    if wraps:
                        index %= len(angles)
                    if 0 <= index < len(angles):
                        add(axis, {**original, field: angles[index]})
            origin = state["sites"][original["siteId"]]
            sites = sorted(range(len(state["sites"])),
                           key=lambda index: (math.dist(state["sites"][index], origin), index))
            added_sites = 0
            for site in sites:
                if site != original["siteId"] and add("site", {**original, "siteId": site}):
                    added_sites += 1
                    if added_sites == self.nearby_site_count:
                        break
            slots.append(candidates)
        return slots

    def initialize_from_placements(self, placements, *, baseline_action_probability=None, strength=None):
        if self.base_placements is not None or self.updates or self.baseline is not None or np.any(self.option_logits):
            raise ValueError("Local initialization requires a fresh policy")
        if strength is not None:
            raise ValueError("Local refinement uses uniform edit logits, not a warm-start logit strength")
        if baseline_action_probability is not None and (
                isinstance(baseline_action_probability, bool)
                or not isinstance(baseline_action_probability, (int, float))
                or not math.isfinite(baseline_action_probability)
                or not 0 < baseline_action_probability < 1):
            raise ValueError("Legacy baseline action probability must be finite in (0,1)")
        base = self._canonical_placements(placements)
        count = self.sensor_count if self.sensor_count is not None else len(base)
        state = deepcopy(self._initial_state)
        if len(base) != count or count > state["max_sites"]:
            raise ValueError("Starting layout must match the exact selected sensor count")
        state["max_sites"] = count
        if not self._layout_legal(base, state, self.contract["catalogue"]):
            raise ValueError("Starting layout is not legal under the selected native sensor contract")
        slots = self._candidates(base, state, self.contract["catalogue"])
        self.sensor_count = count
        self._initial_state = state
        self.base_placements = base
        self.slot_candidates = slots
        self._slot_offsets = np.cumsum([0] + [len(rows) for rows in slots]).tolist()
        self.option_logits = np.zeros(self._slot_offsets[-1], dtype=float)
        self.m = np.zeros_like(self.option_logits)
        self.v = np.zeros_like(self.option_logits)
        self.values = np.zeros(count, dtype=float)
        self.value_updates = np.zeros(count, dtype=float)
        self.action_space = {"kind": "factorized_local_edits", "slots": count,
            "actions": len(self.option_logits), "actionsPerSlot": [len(rows) for rows in slots],
            "globalPlacementOptions": len(self.contract["options"]),
            "nearbySitesPerSlot": self.nearby_site_count, "fixedProfilePerSlot": True,
            "deploymentRadiusMeters": [state["deployment_min_radius"], state["deployment_max_radius"]],
            "maxSiteMoveMetersPerSlot": [max(math.dist(state["sites"][original["siteId"]],
                state["sites"][candidate["placement"]["siteId"]]) for candidate in candidates)
                for original, candidates in zip(base, slots)],
            "limitation": "One edit per sensor from its fixed starting pose; adjacent angle bins and at most two nearest individually legal sites. Not unrestricted placement search."}
        keep_probabilities = [1. / len(rows) for rows in slots]
        self._warm_start_report = {"placements": count, "trainable": True, "strength": 0.,
            "initialization": "uniform_local_edits", "calibration": "Uniform within each legal sensor slot; KEEP wins initial deterministic ties",
            "legacy_baseline_action_probability_ignored": baseline_action_probability,
            "initial_baseline_action_probability": keep_probabilities[0],
            "initial_exploration_probability": 1. - keep_probabilities[0],
            "initial_keep_probabilities": keep_probabilities,
            "initial_full_layout_probability": float(np.prod(keep_probabilities)),
            "actionSpace": deepcopy(self.action_space)}
        return deepcopy(self._warm_start_report)

    def plan(self, context, *, rng=None, deterministic=False):
        self._require_initialized()
        state, catalogue = self._planning_inputs(context)
        placements = deepcopy(self.base_placements)
        if not self._layout_legal(placements, state, catalogue):
            raise ValueError("Starting local layout is no longer legal in the native scene")
        generator = self.rng if rng is None else rng
        records = []
        for slot, candidates in enumerate(self.slot_candidates):
            mask = np.zeros(len(self.option_logits), dtype=bool)
            for local, candidate in enumerate(candidates):
                edited = list(placements)
                edited[slot] = candidate["placement"]
                mask[self._slot_offsets[slot] + local] = self._layout_legal(edited, state, catalogue)
            probabilities = self._probabilities(self.logits(), mask)
            action = (int(np.argmax(np.where(mask, self.logits(), -np.inf))) if deterministic
                      else int(generator.choice(len(probabilities), p=probabilities)))
            records.append({"action": action, "old_probability": float(probabilities[action]),
                "probabilities": probabilities.copy(), "mask": mask, "step": slot,
                "value": float(self.values[slot])})
            placements[slot] = deepcopy(candidates[action - self._slot_offsets[slot]]["placement"])
        return placements, records

    def _samples(self, episodes):
        self._require_initialized()
        rewards, samples = super()._samples(episodes)
        for records, _ in episodes:
            if len(records) != self.sensor_count:
                raise ValueError("Local PPO requires one recorded decision per sensor slot")
            for slot, record in enumerate(records):
                start, end = self._slot_offsets[slot:slot + 2]
                mask = record["mask"]
                if record["step"] != slot or np.any(mask[:start]) or np.any(mask[end:]):
                    raise ValueError("Local PPO record mask must belong only to its sensor slot")
        return rewards, samples

    def update(self, episodes, **kwargs):
        self._require_initialized()
        report = super().update(episodes, **kwargs)
        report["mean_training_advantage_s"] = report.pop("mean_training_warning_s")
        report["reward_mode"] = self.reward_mode
        report["action_space"] = deepcopy(self.action_space)
        return report

    def save(self, path):
        self._require_initialized()
        data = {"schema": self.schema, "algorithm": self.algorithm, "reward_mode": self.reward_mode,
            "contract": self.contract, "sensor_count": self.sensor_count,
            "allowed_sensor_ids": list(self.allowed_sensor_ids), "entropy_coefficient": self.entropy_coefficient,
            "base_placements": self.base_placements, "slot_candidates": self.slot_candidates,
            "action_space": self.action_space, "warm_start_report": self._warm_start_report,
            "option_logits": self.option_logits.tolist(), "adam_m": self.m.tolist(), "adam_v": self.v.tolist(),
            "values": self.values.tolist(), "value_updates": self.value_updates.tolist(),
            "updates": self.updates, "optimizer_steps": self.optimizer_steps, "baseline": self.baseline,
            "rng_state": deepcopy(self.rng.bit_generator.state)}
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @classmethod
    def load(cls, path, context):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if (data.get("schema") != cls.schema or data.get("algorithm") != cls.algorithm
                or data.get("reward_mode") != cls.reward_mode):
            raise ValueError("Wrong local refinement policy schema or reward mode")
        policy = cls(context, sensor_count=data["sensor_count"],
                     entropy_coefficient=data["entropy_coefficient"], allowed_sensor_ids=data["allowed_sensor_ids"])
        if data.get("contract") != policy.contract:
            raise ValueError("Wrong local refinement native geometry contract")
        report = data.get("warm_start_report", {})
        policy.initialize_from_placements(data["base_placements"],
            baseline_action_probability=report.get("legacy_baseline_action_probability_ignored"))
        if (data.get("slot_candidates") != policy.slot_candidates
                or data.get("action_space") != policy.action_space
                or report != policy._warm_start_report):
            raise ValueError("Saved local edit candidates do not match the recomputed public contract")
        for name, key in {"option_logits": "option_logits", "m": "adam_m", "v": "adam_v",
                          "values": "values", "value_updates": "value_updates"}.items():
            value = np.asarray(data[key], dtype=float)
            if value.shape != getattr(policy, name).shape or not np.isfinite(value).all():
                raise ValueError("Invalid local policy arrays")
            if name in ("v", "value_updates") and np.any(value < 0):
                raise ValueError("Invalid local optimizer state")
            setattr(policy, name, value)
        for name in ("updates", "optimizer_steps"):
            value = data[name]
            if type(value) is not int or value < 0:
                raise ValueError("Invalid local optimizer counters")
            setattr(policy, name, value)
        baseline = data["baseline"]
        if baseline is not None and (isinstance(baseline, bool) or not isinstance(baseline, (int, float))
                                     or not math.isfinite(baseline)):
            raise ValueError("Invalid local value baseline")
        policy.baseline = baseline
        policy.rng.bit_generator.state = data["rng_state"]
        return policy
