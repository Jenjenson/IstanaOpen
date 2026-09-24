"""Explainable public-only sensor placement and validated human layouts.

The common-sense baseline is a fixed rule, not a trained policy or a measured
human participant: cover likely approach corridors, prefer useful coverage per
unit cost, and reduce redundant coverage. It uses the same static public
capability/weather features and legal options as the learner, but never the
temporal reward forecast, simulator outcomes, or hidden adversary trajectories.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

import numpy as np

from .adaptive_inputs import (
    LiveObservationAdapter, apply_placement, build_observation,
    validate_catalogue, validate_public_state,
)


RULE = "marginal_coverage / actual_cost * (1 - 0.5 * overlap / candidate_coverage)"
EXPLANATION = (
    "Cover likely approach corridors with weather-suited sensors; prefer new "
    "coverage per unit cost and discount overlap. Break equal scores by greater "
    "separation, then catalogue/site order. Stop when no useful legal coverage remains."
)


def _inputs(public_state, catalogue):
    catalogue = validate_catalogue(catalogue)
    state = validate_public_state(public_state, catalogue)
    if state["done"]:
        raise ValueError("Planning snapshot is done")
    return state, catalogue


def _report(kind, decisions, final):
    manual = kind == "manual"
    return {
        "schema": "istana.sensor_baseline_plan.v1", "public_only": True,
        "physical_commands_sent": False,
        "decisions": decisions,
        "new_placements": [deepcopy(row) for row in decisions if not row["stop"]],
        "final_public_state": final,
        "selection": {
            "kind": kind,
            "label": "Your sensor layout" if manual else "Common-sense placement (non-RL)",
            "explanation": ("User-selected sensors and approved sites, checked against the same placement rules."
                            if manual else EXPLANATION),
            "claim": ("User-authored layout; no claim of optimal placement."
                      if manual else "Fixed explainable heuristic; a proxy for sensible placement, not a human study or an optimal layout."),
            **({} if manual else {"rule": RULE}),
        },
    }


def plan_common_sense(public_state: Mapping, catalogue: Sequence | None = None) -> dict:
    """Return a deterministic bounded layout using only validated public inputs.

    Static marginal coverage already includes public ingress probabilities,
    declared sensor strengths/ranges, visibility, rain, RF noise and emitter
    likelihood. Scores are planning estimates, never measured detection rates.
    Existing placements are retained and consume the shared budget/site limits.
    """
    # Native directional profiles must be scored as frustums, not as the old
    # radial range proxy. Reuse the current public feature/legality contract;
    # no simulator outcome or future drone path enters this fixed rule.
    directional = any(sensor.get("directional") for sensor in (catalogue or []))
    observe, place = build_observation, apply_placement
    if directional:
        from . import directional_inputs
        catalogue = directional_inputs.validate_catalogue(catalogue)
        state = directional_inputs.validate_public_state(public_state, catalogue)
        if state["done"]:
            raise ValueError("Planning snapshot is done")
        observe, place = directional_inputs.build_observation, directional_inputs.apply_placement
    else:
        state, catalogue = _inputs(public_state, catalogue)
    decisions = []
    for _ in range(state["max_sites"] - len(state["placements"]) + 1):
        observation = observe(state, catalogue)
        names, features = observation["feature_names"], observation["option_features"]
        column = lambda name: features[:, names.index(name)].astype(float)
        gain, coverage, overlap = (column(name) for name in
                                   ("marginal_coverage", "candidate_coverage", "overlap"))
        distance = column("nearest_placement_distance")
        options = observation["options"]
        legal = [int(i) for i in np.flatnonzero(observation["action_mask"])
                 if not options[i]["stop"] and gain[i] > 1e-6]
        if legal:
            def score(index):
                cost = catalogue[options[index]["sensor_index"]]["cost"]
                redundant_fraction = min(1., max(0., overlap[index] / max(coverage[index], 1e-12)))
                return gain[index] / cost * (1. - .5 * redundant_fraction)
            action = max(legal, key=lambda index: (score(index), distance[index], -index))
            row = LiveObservationAdapter.recommendation(observation, action)
            row["rationale"] = {
                "rule": RULE, "score": float(score(action)),
                "new_coverage_estimate": float(gain[action]),
                "overlap_estimate": float(overlap[action]),
                "cost": catalogue[row["sensor_index"]]["cost"],
                "explanation": "Adds likely-approach coverage for the available budget, with an overlap discount.",
            }
        else:
            action = len(options) - 1
            row = LiveObservationAdapter.recommendation(observation, action)
            row["rationale"] = {"explanation": "No legal sensor adds useful public estimated coverage."}
        row.update(action_index=int(action), reason=row["rationale"]["explanation"])
        decisions.append(row)
        state = place(state, action, catalogue)
        if row["stop"]:
            report = _report("common_sense", decisions, state)
            if directional:
                report["selection"]["sensor_model"] = "directional public coverage with joint type/site/yaw/pitch choices"
            return report
    raise RuntimeError("Common-sense placement did not stop within the site limit")


def validate_manual_layout(public_state: Mapping, catalogue: Sequence | None,
                           placements: list[dict]) -> dict:
    """Validate ordered ``{sensor_id, site_index}`` choices without changing input.

    Coordinates, costs and sensor indices are resolved from the public catalogue
    and approved sites; callers cannot override them. Every choice is checked
    against the updated shared legal mask, so the whole layout either validates
    or raises. An empty layout is a valid explicit choice to deploy no sensors.
    """
    state, catalogue = _inputs(public_state, catalogue)
    remaining = state["max_sites"] - len(state["placements"])
    if not isinstance(placements, list) or len(placements) > remaining:
        raise ValueError(f"Manual layout must be a list with at most {remaining} sensors")
    sensor_indices = {sensor["id"]: index for index, sensor in enumerate(catalogue)}
    site_count = len(state["sites"])
    decisions = []
    for number, choice in enumerate(placements, 1):
        if not isinstance(choice, dict) or set(choice) != {"sensor_id", "site_index"}:
            raise ValueError("Each manual placement requires only sensor_id and site_index")
        sensor_id, site = choice["sensor_id"], choice["site_index"]
        if not isinstance(sensor_id, str) or sensor_id not in sensor_indices:
            raise ValueError(f"Manual sensor {number} is absent from the catalogue")
        if type(site) is not int or not 0 <= site < site_count:
            raise ValueError(f"Manual site {number} must index an approved site")
        observation = build_observation(state, catalogue)
        action = sensor_indices[sensor_id] * site_count + site
        if not observation["action_mask"][action]:
            raise ValueError(f"Manual placement {number} violates budget, availability, site, separation or deployment limits")
        row = LiveObservationAdapter.recommendation(observation, action)
        row["rationale"] = {"explanation": "Selected by the user; accepted by the shared placement rules."}
        row.update(action_index=action, reason=row["rationale"]["explanation"])
        decisions.append(row)
        state = apply_placement(state, action, catalogue)
    observation = build_observation(state, catalogue)
    action = len(observation["options"]) - 1
    stop = LiveObservationAdapter.recommendation(observation, action)
    stop["rationale"] = {"explanation": "User layout complete."}
    stop.update(action_index=action, reason=stop["rationale"]["explanation"])
    decisions.append(stop)
    return _report("manual", decisions, apply_placement(state, action, catalogue))
