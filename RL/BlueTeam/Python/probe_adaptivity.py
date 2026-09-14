"""Measure frozen-policy sensitivity using validation-only public snapshots.

These controlled recommendation probes are not defence outcomes, a learning
step, or evidence of optimality. They change only public direction reports,
emitter priors, or weather inputs, and then ask the same frozen policy again.
No scenario truth, future trajectory, simulator step, or actuator is accessed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from triad_rl.adaptive_env import AdaptivePlacementEnv
from triad_rl.adaptive_inputs import FEATURE_SCHEMA, INPUT_SCHEMA, LiveObservationAdapter
from triad_rl.adaptive_policy import AdaptivePolicy
from triad_rl.train_adaptive import FINAL_TEST_SEED_BASE, VALIDATION_SEED_BASE, source_provenance


DEFAULT_PROBE_SEED = 1_000_000_998_000_000
PROBE_SCHEMA = "triad.adaptive_public_input_sensitivity.v1"


def public_variants(state: Mapping[str, Any]) -> dict[str, tuple[dict, dict]]:
    """Return independent intervention pairs; never mutate the source snapshot."""
    variants = {}
    left, right = deepcopy(dict(state)), deepcopy(dict(state))
    # Offered sites stay fixed while approach probabilities and current tracks
    # are rotated 180 degrees. Reports, not hidden target truth, are changed.
    right["forecast"]["approach_weights"] = np.roll(right["forecast"]["approach_weights"], 4).tolist()
    for track in right["tracks"]:
        track["position"][:2] = [-value for value in track["position"][:2]]
        track["velocity"][:2] = [-value for value in track["velocity"][:2]]
    variants["direction"] = (left, right)
    left, right = deepcopy(dict(state)), deepcopy(dict(state))
    left["forecast"]["emitter_probability"], right["forecast"]["emitter_probability"] = 0., 1.
    for track in left["tracks"]:
        track["emitter_probability"] = 0.
    for track in right["tracks"]:
        track["emitter_probability"] = 1.
    variants["emitter"] = (left, right)
    left, right = deepcopy(dict(state)), deepcopy(dict(state))
    left["weather"].update(visibility=1., illumination=1., rain=0., humidity=.2)
    right["weather"].update(visibility=.1, illumination=.08, rain=.9, humidity=.9)
    variants["weather"] = (left, right)
    return variants


def first_recommendation(policy: AdaptivePolicy, public: Mapping[str, Any], catalogue: list
                         ) -> tuple[dict, np.ndarray]:
    """Use the same validated live/simulation adapter; no truth argument exists."""
    adapter = LiveObservationAdapter(catalogue)
    observation = adapter.observe(public)
    probabilities = policy.probabilities(observation)
    action = policy.act(observation, deterministic=True)
    option = adapter.recommendation(observation, action)
    mass: Counter = Counter()
    for candidate, probability in zip(observation["options"], probabilities, strict=True):
        mass[candidate["sensor_id"] or "stop"] += float(probability)
    return {"action": option, "sensor_probability": dict(mass)}, probabilities


def run_probes(checkpoint: str | Path, *, episodes: int = 40,
               seed: int = DEFAULT_PROBE_SEED) -> dict:
    if (isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1
            or isinstance(seed, bool) or not isinstance(seed, int)
            or seed < VALIDATION_SEED_BASE or seed + episodes > FINAL_TEST_SEED_BASE):
        raise ValueError("Probes require positive episodes entirely within the validation-only seed band")
    policy = AdaptivePolicy.load(checkpoint)
    before_hash = policy.weights_fingerprint()
    before_rng = json.dumps(policy.rng.bit_generator.state, sort_keys=True)
    source_hashes = source_provenance()
    report = {"schema": PROBE_SCHEMA, "weights_sha256": before_hash,
              "training_seed": policy.training_state.get("config", {}).get("seed"),
              "seed_provenance": {"validation": {"start": seed, "count": episodes}},
              "input_schema": INPUT_SCHEMA, "feature_schema": FEATURE_SCHEMA,
              "source_sha256": source_hashes,
              "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "training_performed": False, "final_test_accessed": False,
              "interpretation": "Controlled recommendation-only interventions on public snapshots. These are sensitivity probes, not defence outcomes or proof of optimality; no policy updates occur.",
              "controls": {"direction": "Rotate reported approach sectors and current tracks 180 degrees; keep sites fixed.",
                           "emitter": "Set forecast/current-track emitter probabilities to 0 versus 1.",
                           "weather": "Clear daylight/dry versus poor visibility/dark/rain/humidity; keep RF noise fixed."},
              "variants": {}, "examples": [], "episodes": []}
    grouped: dict[str, list[dict]] = {name: [] for name in ("direction", "emitter", "weather")}
    env = AdaptivePlacementEnv(split="heldout")
    for index in range(episodes):
        # reset returns only the public observation. This function deliberately
        # does not read env.scenario and never advances the physical simulation.
        observation = env.reset(seed=seed + index)
        for name, (left, right) in public_variants(observation["state"]).items():
            a, pa = first_recommendation(policy, left, observation["catalogue"])
            b, pb = first_recommendation(policy, right, observation["catalogue"])
            displacement = float(np.linalg.norm(np.asarray(a["action"]["position"]) - b["action"]["position"]))
            row = {"scenario_seed": seed + index,
                   "sensor_changed": a["action"]["sensor_id"] != b["action"]["sensor_id"],
                   "position_distance_m": displacement,
                   "action_distribution_l1": float(np.abs(pa - pb).sum()), "before": a, "after": b}
            grouped[name].append(row)
            report["episodes"].append({"variant": name, **row})
            if index < 3:
                report["examples"].append({"variant": name, **row})
    for name, rows in grouped.items():
        report["variants"][name] = {
            "count": len(rows),
            "sensor_change_rate": float(np.mean([row["sensor_changed"] for row in rows])),
            "position_change_rate": float(np.mean([row["position_distance_m"] > 1e-6 for row in rows])),
            "mean_position_distance_m": float(np.mean([row["position_distance_m"] for row in rows])),
            "mean_action_distribution_l1": float(np.mean([row["action_distribution_l1"] for row in rows])),
            "before_sensor_counts": dict(Counter(row["before"]["action"]["sensor_id"] or "stop" for row in rows)),
            "after_sensor_counts": dict(Counter(row["after"]["action"]["sensor_id"] or "stop" for row in rows))}
    if (policy.weights_fingerprint() != before_hash
            or json.dumps(policy.rng.bit_generator.state, sort_keys=True) != before_rng):
        raise RuntimeError("Probe unexpectedly mutated policy weights or RNG")
    if source_provenance() != source_hashes:
        raise RuntimeError("Implementation changed during probe; refusing mixed-version evidence")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=DEFAULT_PROBE_SEED)
    args = parser.parse_args(argv)
    report = run_probes(args.checkpoint, episodes=args.episodes, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["variants"], indent=2))
    print(f"Validation-only recommendation probe saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
