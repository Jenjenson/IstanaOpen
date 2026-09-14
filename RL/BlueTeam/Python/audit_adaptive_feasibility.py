"""Post-hoc optimistic observability audit of recorded evaluation scenarios.

Private scenario truth is used ONLY for diagnostic evidence after evaluation.
This is neither a policy, a new heldout evaluation, nor an achievable success
probability. If even unlimited initially legal sensor/site options cannot
support timely confirmation, a budget-constrained policy cannot succeed.
Positive support is only a necessary condition: sensing can still fail and the
options needed for different ticks/targets may violate joint resource limits.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from triad_rl.adaptive_env import AdaptivePlacementEnv
from triad_rl.adaptive_evaluation import canonical_hash, implementation_fingerprints, json_safe
from triad_rl.adaptive_inputs import MODALITIES, sensing_probabilities, validate_catalogue


AUDIT_SCHEMA = "triad.adaptive_feasibility_audit.v1"


def confirmation_support(positive_ticks: Any, times: Any, deadline: float,
                         *, window: int, required: int) -> dict[str, Any]:
    """Check existence of enough possible sightings in an allowed tick window.

    Mirrors the simulator's trailing tick-window rule. Several modalities or
    sensors at the same tick still count as just one target sighting.
    """
    positive = np.asarray(positive_ticks, dtype=bool)
    clock = np.asarray(times, dtype=float)
    if (positive.ndim != 1 or clock.shape != positive.shape or clock.size == 0
            or not np.isfinite(clock).all() or np.any(np.diff(clock) <= 0)
            or not np.isfinite(deadline) or not 1 <= required <= window):
        raise ValueError("Invalid confirmation support inputs")
    cumulative = np.r_[0, np.cumsum(positive)]
    rolling = cumulative[1:] - cumulative[np.maximum(0, np.arange(len(positive)) + 1 - window)]
    before = clock <= deadline
    feasible = np.flatnonzero(before & (rolling >= required))
    return {"possible": bool(len(feasible)),
            "earliest_possible_confirmation": float(clock[feasible[0]]) if len(feasible) else None,
            "positive_ticks_before_deadline": int(np.sum(positive & before)),
            "max_positive_ticks_in_window_before_deadline": int(rolling[before].max()) if before.any() else 0}


def audit_scenario(scenario: Mapping[str, Any], catalogue: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Enumerate an optimistic superset of every valid final deployment.

    Individual options honor availability, approval, geometry and individual
    affordability at reset. Simultaneous budget, slot, pairwise separation and
    same-site restrictions are deliberately relaxed. No probabilities are
    changed, no emitter settings are altered, and random hits are not sampled
    for the optimistic bound.
    """
    env = AdaptivePlacementEnv(seed=int(scenario["seed"]), catalogue=catalogue)
    obs = env.reset(scenario=deepcopy(scenario))
    options = [obs["options"][int(index)] for index in np.flatnonzero(obs["action_mask"][:-1])]
    # STOP reproduces the authoritative trajectory and emitter bits without
    # deploying sensors. Its zero-sensor outcome is not an evaluated method.
    _, _, _, terminal = env.step(len(obs["options"]) - 1)
    frames = terminal["frames"]
    times = np.asarray([frame["time"] for frame in frames], dtype=float)
    points = np.asarray([[target["position"] for target in frame["threats"]] for frame in frames], dtype=float)
    emissions = np.asarray([[target["emitting"] for target in frame["threats"]] for frame in frames], dtype=float)
    target_count = len(scenario["targets"])
    sensors = [env.catalogue[option["sensor_index"]] for option in options]
    positions = np.asarray([option["position"] for option in options], dtype=float)
    probabilities = sensing_probabilities(positions, sensors, points.reshape(-1, 3),
                                           scenario["public"]["weather"], emissions.ravel())
    probabilities = probabilities.reshape(len(options), len(times), target_count, len(MODALITIES))
    # Union support, not sum/probability approximation: positivity is exact
    # under the implemented sensing model and arbitrary optimistic hit draws.
    modality_support = (probabilities > 0).any(axis=0)
    support = modality_support.any(axis=-1)
    targets = []
    for index, target in enumerate(scenario["targets"]):
        duration = float(terminal["target_results"][index]["time_to_zone"])
        deadline = duration - float(scenario["defence_lead_time"])
        valid_ticks = times <= deadline
        result = confirmation_support(support[:, index], times, deadline,
                                      window=int(scenario["confirmation_window"]),
                                      required=int(scenario["required_confirmations"]))
        sensor_ids = sorted({option["sensor_id"] for j, option in enumerate(options)
                             if (probabilities[j, valid_ticks, index] > 0).any()})
        if result["possible"]:
            reason = "positive_support_only_not_a_feasible_deployment_proof"
        elif not options:
            reason = "no_initially_legal_sensor_site_options"
        elif not result["positive_ticks_before_deadline"]:
            reason = "zero_sensing_support_before_deadline"
        else:
            reason = "insufficient_positive_ticks_in_confirmation_window"
        maximum_probability = (float(probabilities[:, valid_ticks, index].max())
                               if options and valid_ticks.any() else 0.)
        targets.append({"id": target["id"], "altitude": target["altitude"], "speed": target["speed"],
                        "emitter_duty": target["emitter_duty"], "path": target["path"],
                        "time_to_zone": duration, "confirmation_deadline": deadline,
                        "possible_timely_confirmation": result.pop("possible"), **result,
                        "possible_any_sighting_before_arrival": bool((support[:, index] & (times <= duration)).any()),
                        "timely_support_sensor_ids": sensor_ids,
                        "timely_support_modalities": [m for j, m in enumerate(MODALITIES)
                                                       if modality_support[valid_ticks, index, j].any()],
                        "maximum_single_modality_single_look_probability_before_deadline": maximum_probability,
                        "classification": reason})
    return {"seed": scenario["seed"], "scenario_sha256": canonical_hash(scenario),
            "initially_legal_option_count": len(options),
            "all_targets_possibly_timely_confirmable": all(t["possible_timely_confirmation"] for t in targets),
            "any_target_provably_unconfirmable": any(not t["possible_timely_confirmation"] for t in targets),
            "possible_target_fraction": float(np.mean([t["possible_timely_confirmation"] for t in targets])),
            "targets": targets}


def audit_report(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("schema") != "triad.adaptive_evaluation.v1":
        raise ValueError("Expected a recorded adaptive evaluation report")
    current = implementation_fingerprints()
    if report.get("implementation_sha256") != current:
        raise ValueError("Recorded evaluation implementation differs from current code; refusing mixed-version audit")
    methods = report.get("methods", {})
    if not methods or "adaptive" not in methods:
        raise ValueError("Report must include adaptive episode records")
    reference = methods["adaptive"]["episodes"]
    if not reference:
        raise ValueError("Report contains no scenarios")
    catalogues = [episode["replay"]["catalogue"] for method in methods.values()
                  for episode in method["episodes"]
                  if episode.get("replay", {}).get("catalogue")]
    if not catalogues:
        raise ValueError("Report lacks a recorded replay catalogue; do not assume default hardware")
    catalogue = validate_catalogue(catalogues[0])
    if any(canonical_hash(validate_catalogue(candidate)) != canonical_hash(catalogue) for candidate in catalogues):
        raise ValueError("Report uses inconsistent catalogues")
    by_method = {name: {r["seed"]: r for r in method["episodes"]} for name, method in methods.items()}
    records = []
    for episode in reference:
        scenario = episode["scenario"]
        if canonical_hash(scenario) != episode["scenario_sha256"]:
            raise ValueError("Recorded scenario fingerprint mismatch")
        result = audit_scenario(scenario, catalogue)
        observed = {}
        for name, rows in by_method.items():
            row = rows.get(episode["seed"])
            if row is None or row["scenario_sha256"] != episode["scenario_sha256"]:
                raise ValueError("Method scenarios are not paired")
            success = bool(row["metrics"]["success"])
            if success and not result["all_targets_possibly_timely_confirmable"]:
                raise RuntimeError("Audit upper bound contradicts an observed successful layout")
            target_support = {t["id"]: t["possible_timely_confirmation"] for t in result["targets"]}
            if any(t["timely_confirmed"] and not target_support[t["id"]] for t in row.get("target_results", [])):
                raise RuntimeError("Audit upper bound contradicts observed timely confirmation")
            observed[name] = {"success": success, "cost": row["metrics"].get("cost"),
                              "sensors_placed": row["metrics"].get("sensors_placed")}
        result["observed_methods"] = observed
        records.append(result)
    all_possible = [r for r in records if r["all_targets_possibly_timely_confirmable"]]
    impossible = [r for r in records if r["any_target_provably_unconfirmable"]]
    all_targets = [t for r in records for t in r["targets"]]
    method_summaries = {}
    for name in methods:
        conditional_costs = [r["observed_methods"][name]["cost"] for r in impossible
                             if r["observed_methods"][name]["cost"] is not None]
        method_summaries[name] = {
            "observed_success_rate": float(np.mean([r["observed_methods"][name]["success"] for r in records])),
            "success_rate_on_positive_support_scenarios": float(np.mean([r["observed_methods"][name]["success"] for r in all_possible])) if all_possible else None,
            "mean_cost_on_provably_impossible_full_defence_scenarios": float(np.mean(conditional_costs)) if conditional_costs else None,
            "warning": "A failed positive-support scenario may reflect joint constraints or stochastic misses, not necessarily a policy error. Spending can still benefit partial detection in impossible full-defence scenarios.",
        }
    if implementation_fingerprints() != current:
        raise RuntimeError("Core implementation changed during feasibility audit")
    return json_safe({
        "schema": AUDIT_SCHEMA, "split": report["split"], "private_truth_used": True,
        "training_performed": False, "new_independent_test": False,
        "interpretation": "Post-hoc necessary-condition bound using recorded scenarios, not achievable defence rate, policy evaluation or deployable guidance.",
        "scenario_reuse": "These scenarios are now diagnostic evidence; do not relabel later tuning on them as independent final testing.",
        "bound": {"individual_constraints": "initial action mask: availability, blocked sites, radius, existing-site separation and individual affordability",
                  "relaxed_joint_constraints": ["total deployment budget", "maximum sensor count", "pairwise separation", "one sensor per site"],
                  "sensing": "unchanged actual weather, slant-range model, path and deterministic emitter schedule; p>0 support only",
                  "confirmation": "required distinct sensing ticks in trailing confirmation_window before time_to_zone-defence_lead_time",
                  "probability_claim": "none; arbitrarily small nonzero probability counts as optimistic support"},
        "implementation_sha256": current, "catalogue_sha256": canonical_hash(catalogue), "catalogue": catalogue,
        "source_scenario_sequence_sha256": report.get("scenario_sequence_sha256"),
        "seed_provenance": report.get("seed_provenance", {}),
        "summary": {"episodes": len(records), "targets": len(all_targets),
                    "optimistic_full_defence_support_upper_bound": len(all_possible) / len(records),
                    "episodes_with_provably_unconfirmable_target": len(impossible),
                    "provably_unconfirmable_target_count": sum(not t["possible_timely_confirmation"] for t in all_targets),
                    "target_classifications": dict(Counter(t["classification"] for t in all_targets))},
        "observed_method_summaries": method_summaries, "episodes": records,
    })


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    content = args.report.read_bytes()
    result = audit_report(json.loads(content))
    result["source_report"] = {"filename": args.report.name, "sha256": hashlib.sha256(content).hexdigest()}
    result["audit_implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))
    print(f"Saved post-hoc optimistic feasibility audit to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
