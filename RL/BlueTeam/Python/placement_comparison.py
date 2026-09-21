"""Paired offline sensing demonstrations on unmodified archived scenarios.

The learner's saved actions and a public-only baseline are independently scored
by the same adaptive environment. Neither training nor fresh scenario generation
is performed. These are new synthetic comparisons, separate from the published
replay evidence. A common-sense heuristic is a human-readable proxy, not a study
of human performance; a manually authored layout can also be evaluated.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from triad_rl import adaptive_env
from triad_rl.adaptive_inputs import build_observation, validate_catalogue
from triad_rl.common_sense import plan_common_sense, validate_manual_layout


SCHEMA = "istana.sensor_placement_comparison.v1"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _label(replay):
    return f"Temporal {replay['temporal_seed']} · {replay['profile']} · case {replay['case_index']}"


def comparison_scenario(replay):
    """Return only public planning context; never score or expose future truth."""
    observation = build_observation(replay["scenario"]["public"], replay["catalogue"])
    public, catalogue = observation["state"], observation["catalogue"]
    plan = plan_common_sense(deepcopy(public), deepcopy(catalogue))
    return {
        "schema": SCHEMA, "label": _label(replay),
        "coordinateLabel": "Synthetic evaluation arena · objective-relative metres",
        "sites": public["sites"], "blockedSites": public["blocked_sites"],
        "eligibleSites": sorted({option["site_index"] for option, legal in
            zip(observation["options"], observation["action_mask"])
            if legal and not option["stop"]}),
        "catalogue": catalogue, "availableSensorIds": public["available_sensor_ids"],
        "budget": public["budget_total"], "maxSensors": public["max_sites"],
        "minSeparation": public["min_separation"],
        "deploymentMinRadius": public["deployment_min_radius"],
        "deploymentMaxRadius": public["deployment_max_radius"],
        "objectiveRadius": replay["scenario"]["objective_radius"],
        "weather": public["weather"], "forecast": public["forecast"], "tracks": public["tracks"],
        "suggestedPlacements": [{"sensor_id": row["sensor_id"], "site_index": row["site_index"],
            "position": deepcopy(row["position"]), "cost": catalogue[row["sensor_index"]]["cost"]}
            for row in plan["new_placements"]],
        "suggestionDecisions": deepcopy(plan["decisions"]),
        "selection": deepcopy(plan["selection"]),
        "audit": {"public_only": True, "scoring_performed": False,
                  "policy_truth_access": False, "live_unreal": False},
    }


def _core(replay):
    # Match the controlled initialization in TemporalPlacementEnv: the normal
    # constructor would generate an unrelated scenario before reset. Do not
    # invoke it, sample a new case, load a checkpoint or update policy state.
    env = adaptive_env.AdaptivePlacementEnv.__new__(adaptive_env.AdaptivePlacementEnv)
    env.catalogue = validate_catalogue(replay["catalogue"])
    env.split = replay["scenario"].get("split", "supplied")
    env._custom_catalogue = True
    env._rng = np.random.default_rng(replay["scenario"]["seed"])
    env.reset(scenario=deepcopy(replay["scenario"]))
    return env


def _score(replay, source_decisions, *, label, policy):
    env, decisions = _core(replay), []
    for source in source_decisions:
        option = source.get("action", source)
        if env.done:
            # The public planner emits STOP even when the scorer has already
            # terminated automatically after the last legal deployment.
            if option.get("stop"):
                continue
            raise ValueError("Layout contains a deployment after placement ended")
        observation = env.observe()
        action = source.get("action_index")
        if action is None:
            action = (len(observation["options"]) - 1 if option.get("stop") else
                      option["sensor_index"] * len(env.public_state["sites"]) + option["site_index"])
        if (type(action) is not int or not 0 <= action < len(observation["options"])
                or not observation["action_mask"][action]):
            raise ValueError("Layout contains an illegal sensor placement")
        resolved = observation["options"][action]
        if any(option.get(key) != resolved[key] for key in ("stop", "sensor_id", "site_index")):
            raise ValueError("Layout action differs from its declared sensor and site")
        _, reward, _, _ = env.step(action)
        decisions.append({"action_index": action, "action": deepcopy(resolved), "reward": reward,
                          **{key: deepcopy(source[key]) for key in ("reason", "rationale") if key in source}})
    if not env.done:
        action = len(env.observe()["options"]) - 1
        option = env.observe()["options"][action]
        _, reward, _, _ = env.step(action)
        decisions.append({"action_index": action, "action": option, "reward": reward,
                          "reason": "Layout complete."})
    info = deepcopy(env.info)
    frames = info.pop("frames")
    return {"mode": "comparison", "label": label,
            "coordinateLabel": "Paired synthetic comparison · objective-relative metres",
            "catalogue": deepcopy(env.catalogue), "placements": deepcopy(env.placements),
            "sites": deepcopy(env.scenario["public"]["sites"]),
            "blockedSites": deepcopy(env.scenario["public"].get("blocked_sites", [])),
            "budget": env.scenario["public"]["budget_total"],
            "objectiveRadius": env.scenario["objective_radius"],
            "frames": frames, "metrics": info, "decisions": decisions,
            "seed": env.scenario["seed"], "weather": deepcopy(env.scenario["public"]["weather"]),
            "policy": policy, "outcome": info["outcome"], "ended": True,
            "audit": {"recorded": False, "new_synthetic_comparison": True,
                      "live_unreal": False, "policy_truth_access": False}}


def _summary(view):
    info = view["metrics"]
    targets = info["target_results"]
    return {"target_count": len(targets),
            "detected_count": sum(row["first_detection"] is not None for row in targets),
            "confirmed_count": sum(row["first_confirmation"] is not None for row in targets),
            "timely_confirmed_count": sum(bool(row["timely_confirmed"]) for row in targets),
            "breached_count": sum(not row["timely_confirmed"] for row in targets),
            "sensors_placed": len(view["placements"]),
            **{key: info[key] for key in ("detection_rate", "confirmed_fraction", "breach_rate",
                                          "coverage", "cost", "return", "outcome")}}


def _same(left, right):
    """Allow only numeric roundoff when checking archived result reproduction."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    if type(left) in (int, float) and type(right) in (int, float):
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
    return left == right


def compare_placements(replay, *, baseline="common_sense", placements=None):
    """Evaluate saved RL choices and an independently planned layout fairly.

    Draws in the unchanged scorer are keyed by scenario seed/time/target/modality
    and do not depend on sensor count or ordering. Both methods therefore get
    identical paths, weather, catalogue, budget, legal sites and sensing draws.
    Signed deltas are always RL minus baseline; outcomes may favor either side.
    """
    if baseline not in ("common_sense", "manual"):
        raise ValueError("Choose common_sense or manual comparison")
    if baseline == "common_sense" and placements is not None:
        raise ValueError("Manual placements require the manual comparison method")
    public, catalogue = deepcopy(replay["scenario"]["public"]), deepcopy(replay["catalogue"])
    plan = (plan_common_sense(public, catalogue) if baseline == "common_sense" else
            validate_manual_layout(public, catalogue, placements))
    rl_label = f"RL checkpoint {replay['temporal_seed']}"
    baseline_label = plan["selection"]["label"]
    rl = _score(replay, replay["decisions"], label=rl_label,
                policy=f"Archived temporal {replay['temporal_seed']} choices, re-scored")
    other = _score(replay, plan["decisions"], label=baseline_label, policy=baseline_label)
    summaries = {"rl": _summary(rl), "baseline": _summary(other)}
    metrics_keys = ("detected_count", "detection_rate", "confirmed_count", "timely_confirmed_count",
                    "breached_count", "breach_rate", "coverage", "cost", "return", "sensors_placed")
    result = {
        "schema": SCHEMA, "label": _label(replay), "method": baseline,
        "rl": rl, "baseline": other, "metrics": summaries,
        "methods": [{"id": "rl", "label": rl_label, "view": rl, "metrics": summaries["rl"]},
                    {"id": baseline, "label": baseline_label, "view": other, "metrics": summaries["baseline"]}],
        "deltas": {key: summaries["rl"][key] - summaries["baseline"][key] for key in metrics_keys},
        "deltaDirection": "RL minus baseline", "selection": deepcopy(plan["selection"]),
        "audit": {"new_synthetic_comparison": True, "published_evidence_modified": False,
            "training_performed": False, "fresh_rl_inference": False, "scenario_generation_performed": False,
            "policy_truth_access": False, "physical_commands_sent": False, "live_unreal": False,
            "same_scenario": True, "same_budget": True, "same_catalogue": True, "same_sensing_draws": True,
            "sensing_draw_rule": "PCG64 SeedSequence([scenario seed, 0xB10E]); time/target/modality draws independent of layout",
            "scenario_sha256": _hash(replay["scenario"]), "catalogue_sha256": _hash(replay["catalogue"]),
            "scorer_sha256": hashlib.sha256(Path(adaptive_env.__file__).read_bytes()).hexdigest(),
            "archived_rl_result_reproduced": (_same(rl["frames"], replay["frames"])
                and _same(rl["metrics"], replay["metrics"]) and _same(rl["placements"], replay["placements"])),
            "interpretation": "Single synthetic scenario, not a human study or real-world validation. "
                "The RL result uses saved choices with fresh scoring. Either method can win; "
                "timely confirmation is detection before the deadline, not physical interception."},
    }
    # Fail here rather than emitting nonfinite values through the HTTP API.
    json.dumps(result, allow_nan=False)
    return result
