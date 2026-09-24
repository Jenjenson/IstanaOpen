"""Measured layout diversity and paired native evaluation, without policy truth input."""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np

from .directional_inputs import apply_placement, build_observation
from .istana_live import public_planning_inputs


def layout_key(placements):
    """Order does not make an otherwise identical sensor layout novel."""
    return tuple(sorted((row["profileId"], row["siteId"],
                         float(row.get("yawDeg", 0.)), float(row.get("pitchDeg", 0.)))
                        for row in placements))


def layout_changes(placements, initial):
    current, original = set(layout_key(placements)), set(layout_key(initial))
    return max(len(current - original), len(original - current))


def rollout_entropy(records):
    values, normalized = [], []
    for record in records:
        probabilities = np.asarray(record["probabilities"] if isinstance(record, dict)
                                   else record[1], dtype=float)
        positive = probabilities[probabilities > 0]
        if len(positive) <= 1:  # Forced STOP is not an exploration decision.
            continue
        entropy = float(-np.sum(positive * np.log(positive)))
        values.append(entropy)
        normalized.append(entropy / math.log(len(positive)))
    return {"entropy": float(np.mean(values)) if values else None,
            "normalizedEntropy": float(np.mean(normalized)) if normalized else None}


def evaluation_summary(runs, baseline=None):
    """Average matched cases; never substitute a lucky training episode."""
    def constraint_hash(run):
        context = run["context"]
        public = context["publicSnapshot"]
        value = {key: context[key] for key in ("catalogue", "temporalConfig", "worldOriginCm")}
        value["deployment"] = {key: public.get(key) for key in (
            "budget_total", "max_sites", "sites", "blocked_sites", "min_separation",
            "weather", "forecast", "available_sensor_ids")}
        return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
    cases = [{"seed": run["seed"], **deepcopy(run["metrics"]),
              "trajectorySha256": run["trajectory_sha256"],
              "constraintsSha256": constraint_hash(run)} for run in runs]
    warnings = np.asarray([row["mean_drone_warning_s"] for row in cases])
    result = {"cases": cases, "seeds": [row["seed"] for row in cases],
              "caseCount": len(cases), "meanWarningSeconds": float(warnings.mean()),
              "mean_drone_warning_s": float(warnings.mean()),
              "team_warning_s": float(np.mean([row["team_warning_s"] for row in cases])),
              "detected_fraction": float(np.mean([row["detected_fraction"] for row in cases])),
              "warningStdSeconds": float(warnings.std(ddof=1)) if len(cases) > 1 else 0.}
    if baseline is not None:
        if result["seeds"] != baseline["seeds"]:
            raise ValueError("Evaluation panels must use identical ordered scenario seeds")
        for actual, reference in zip(cases, baseline["cases"]):
            if actual["trajectorySha256"] != reference["trajectorySha256"]:
                raise ValueError("Paired evaluation trajectories differ")
            if actual["constraintsSha256"] != reference["constraintsSha256"]:
                raise ValueError("Paired evaluation sensor, budget, weather or scenario constraints differ")
        deltas = warnings - [row["mean_drone_warning_s"] for row in baseline["cases"]]
        result.update(baselineMeanWarningSeconds=baseline["meanWarningSeconds"],
                      deltaSeconds=float(deltas.mean()), pairedDeltasSeconds=deltas.tolist(),
                      improvedCases=int(np.sum(deltas > 1e-9)),
                      tiedCases=int(np.sum(np.abs(deltas) <= 1e-9)),
                      worseCases=int(np.sum(deltas < -1e-9)))
    return result


def paired_training_reward(run, contractor_run):
    """Subtract an action-independent native control on the identical scenario.

    The control is evaluation evidence only; it is never passed to plan().
    Reuse the formal comparison checks before allowing a delta into PPO.
    """
    reference = evaluation_summary([contractor_run])
    result = evaluation_summary([run], reference)
    if (run["metrics"]["targets"] != contractor_run["metrics"]["targets"]
            or run["metrics"]["cost"] != contractor_run["metrics"]["cost"]
            or sorted(row["profileId"] for row in run["placements"]) !=
               sorted(row["profileId"] for row in contractor_run["placements"])):
        raise ValueError("Paired training requires the same drones and sensor inventory")
    return {"mode": "paired_contractor_delta", "rewardSeconds": result["deltaSeconds"],
            "policyWarningSeconds": result["meanWarningSeconds"],
            "contractorWarningSeconds": reference["meanWarningSeconds"],
            "seed": run["seed"], "trajectorySha256": result["cases"][0]["trajectorySha256"],
            "constraintsSha256": result["cases"][0]["constraintsSha256"]}


def legal_layout_probes(context, placements, limit=4):
    """Predeclared small legal alternatives to test headroom, never train the actor.

    Candidates depend only on public layout/site/catalogue inputs, not Red truth
    or episode outcomes. Alternate orientation and site changes while keeping
    sensor types fixed, without weakening the contractor baseline.
    """
    if limit <= 0:
        return []
    state, catalogue, _ = public_planning_inputs(context)
    state["max_sites"] = len(placements)
    options = build_observation(state, catalogue)["options"][:-1]
    lookup = {(row["sensor_id"], row["site_index"], row["yaw_deg"], row["pitch_deg"]): index
              for index, row in enumerate(options)}
    def legal(rows):
        current = deepcopy(state)
        for row in rows:
            key = (row["profileId"], row["siteId"], row["yawDeg"], row["pitchDeg"])
            index = lookup.get(key)
            if index is None or not build_observation(current, catalogue)["action_mask"][index]:
                return False
            current = apply_placement(current, index, catalogue)
        return True
    result, seen = [], {layout_key(placements)}
    ranked = []
    for sensor_number, original in enumerate(placements):
        site = state["sites"][original["siteId"]]
        candidates = [row for row in options if row["sensor_id"] == original["profileId"]]
        candidates.sort(key=lambda row: (
            (row["site_index"] != original["siteId"]) if sensor_number % 2 == 0
                else (row["site_index"] == original["siteId"]),
            abs((row["yaw_deg"] - original["yawDeg"] + 180) % 360 - 180)
                + abs(row["pitch_deg"] - original["pitchDeg"]),
            math.dist(row["position"], site), row["site_index"]))
        ranked.append(candidates)
    for rank in range(max((len(rows) for rows in ranked), default=0)):
        for index, candidates in enumerate(ranked):
            if rank >= len(candidates):
                continue
            option = candidates[rank]
            rows = deepcopy(placements)
            rows[index] = {"profileId": option["sensor_id"], "siteId": option["site_index"],
                           "yawDeg": option["yaw_deg"], "pitchDeg": option["pitch_deg"]}
            key = layout_key(rows)
            if key not in seen and legal(rows):
                seen.add(key)
                result.append(rows)
                if len(result) >= limit:
                    return result
    return result
