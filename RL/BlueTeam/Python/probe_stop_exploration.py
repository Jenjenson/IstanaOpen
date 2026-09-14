"""Public-only first-action STOP exploration diagnostic for a frozen policy.

This probe consumes only reset/observe snapshots, never simulator step,
private scenario truth or sensing rollouts. Marginal coverage is a public
forecast feature, not a feasibility label. Low or zero forecast coverage
does not establish that stopping is optimal or that sensing is impossible.

Use matching validation seeds/profile to compare checkpoints. This is reused
development evidence, not an independent final evaluation. It reports only
the first recommendation and its cost, not a completed layout or outcome.
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

from triad_rl.adaptive_inputs import FEATURE_NAMES, FEATURE_SCHEMA, INPUT_SCHEMA
from triad_rl.adaptive_policy import AdaptivePolicy
from triad_rl.robust_scenarios import PROFILES, RobustPlacementEnv


PROBE_SCHEMA = "triad.stop_exploration_probe.v1"
DEFAULT_PROBE_SEED = 1_000_990_101_000_000
VALIDATION_BASE = 10 ** 15
FINAL_TEST_BASE = 2 * 10 ** 15


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def checkpoint_fingerprints(path: Path) -> dict[str, str]:
    return {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
            for name in ("checkpoint.json", "arrays.npz")}


def source_fingerprints() -> dict[str, str]:
    """Bind the probe, public feature builder, actor and scenario distribution."""
    directory = Path(__file__).resolve().parent
    paths = [Path(__file__).resolve(), *[directory / "triad_rl" / name for name in (
        "robust_scenarios.py", "adaptive_env.py", "adaptive_inputs.py", "adaptive_policy.py")]]
    return {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def first_action_record(policy: AdaptivePolicy, observation: Mapping[str, Any], *, seed: int) -> dict:
    """Analyze one public snapshot without sampling or modifying policy RNG."""
    options = observation["options"]
    mask = np.asarray(observation["action_mask"])
    stop_indices = [index for index, option in enumerate(options) if option["stop"]]
    if stop_indices != [len(options) - 1] or not bool(mask[-1]):
        raise ValueError("Public observation requires one legal final STOP option")
    probability = policy.probabilities(observation)
    action = int(np.argmax(probability))
    option = options[action]
    legal = mask[:-1]
    coverage = np.asarray(observation["option_features"])[
        :-1, FEATURE_NAMES.index("marginal_coverage")]
    stop_probability = float(probability[-1])
    best_deployment_probability = float(probability[:-1][legal].max()) if legal.any() else None
    positive = probability[probability > 0]
    sensor = None if option["stop"] else observation["catalogue"][option["sensor_index"]]
    public_input = {"state": observation["state"], "catalogue": observation["catalogue"]}
    return {
        "scenario_seed": seed,
        "public_input_sha256": hashlib.sha256(_json(public_input).encode()).hexdigest(),
        "legal_deployment_count": int(legal.sum()),
        "forced_stop": not bool(legal.any()),
        "max_legal_marginal_coverage": float(coverage[legal].max()) if legal.any() else 0.,
        "stop_probability": stop_probability,
        "uniform_stop_probability": float(1 / mask.sum()),
        "categorical_entropy_nats": float(-(positive * np.log(positive)).sum()),
        "best_deployment_probability": best_deployment_probability,
        "stop_versus_best_deployment_probability": (
            stop_probability / (stop_probability + best_deployment_probability)
            if best_deployment_probability is not None else None),
        "log_probability_ratio_stop_vs_best_deployment": (
            float(np.log(stop_probability) - np.log(best_deployment_probability))
            if best_deployment_probability is not None and stop_probability > 0 else None),
        "deterministic_stop": bool(option["stop"]),
        "first_action": {"index": action, "stop": bool(option["stop"]),
                         "sensor_id": option["sensor_id"], "site_index": option["site_index"],
                         "position": deepcopy(option["position"]),
                         "cost": float(sensor["cost"]) if sensor is not None else 0.},
    }


def summarize_records(records: list[dict]) -> dict:
    result = {
        "count": len(records),
        "deterministic_stop_count": sum(row["deterministic_stop"] for row in records),
        "forced_stop_count": sum(row["forced_stop"] for row in records),
        "deterministic_stop_with_deployment_available_count": sum(
            row["deterministic_stop"] and not row["forced_stop"] for row in records),
        "first_sensor_counts": dict(Counter(row["first_action"]["sensor_id"] or "stop" for row in records)),
    }
    numeric = ("legal_deployment_count", "max_legal_marginal_coverage", "stop_probability",
               "uniform_stop_probability", "categorical_entropy_nats", "best_deployment_probability",
               "stop_versus_best_deployment_probability", "log_probability_ratio_stop_vs_best_deployment",
               "first_action_cost")
    for key in numeric:
        values = np.asarray([row["first_action"]["cost"] if key == "first_action_cost" else row[key]
                             for row in records
                             if key == "first_action_cost" or row[key] is not None], dtype=float)
        result[key] = ({"count": int(values.size), "min": float(values.min()),
                        "median": float(np.median(values)), "mean": float(values.mean()), "max": float(values.max())}
                       if values.size else None)
    return result


def run_probe(checkpoint: str | Path, *, seed: int = DEFAULT_PROBE_SEED,
              episodes: int = 60, profile: str = "stress") -> dict:
    """Return an integrity-bound first-action report using validation seeds only.

    Validation overlap is permitted deliberately: this is a development
    diagnostic, including on a trainer's declared validation set, not a claim
    that these inputs were unseen by the checkpoint or its model selection.
    """
    if (isinstance(seed, bool) or not isinstance(seed, int)
            or isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1
            or seed < VALIDATION_BASE or seed + episodes > FINAL_TEST_BASE):
        raise ValueError("Probe seeds must stay entirely within the validation-only band [10**15, 2*10**15)")
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    checkpoint = Path(checkpoint)
    source_before = source_fingerprints()
    files_before = checkpoint_fingerprints(checkpoint)
    policy = AdaptivePolicy.load(checkpoint, feature_names=FEATURE_NAMES)
    weights_before = policy.weights_fingerprint()
    rng_before = _json(policy.rng.bit_generator.state)
    # Even the constructor uses the first declared validation seed, never an
    # incidental default training/test case. No environment attributes except
    # its reset method are read by the probe.
    env = RobustPlacementEnv(seed=seed, profile=profile)
    records = [first_action_record(policy, env.reset(seed=seed + index), seed=seed + index)
               for index in range(episodes)]
    groups = {"all": summarize_records(records)}
    for label, threshold in (("coverage_le_0.01", .01), ("coverage_le_0.001", .001), ("coverage_zero", 0.)):
        groups[label] = summarize_records([row for row in records if row["max_legal_marginal_coverage"] <= threshold])
    weights_after = policy.weights_fingerprint()
    rng_after = _json(policy.rng.bit_generator.state)
    if weights_after != weights_before or rng_after != rng_before:
        raise RuntimeError("Probe unexpectedly mutated policy weights or RNG")
    files_after, source_after = checkpoint_fingerprints(checkpoint), source_fingerprints()
    if files_after != files_before:
        raise RuntimeError("Checkpoint files changed during probe")
    if source_after != source_before:
        raise RuntimeError("Implementation changed during probe")
    report = {
        "schema": PROBE_SCHEMA,
        "profile": profile,
        "checkpoint_name": checkpoint.name,
        "weights_sha256": weights_before,
        "weights_sha256_after": weights_after,
        "checkpoint_files_sha256_before": files_before,
        "checkpoint_files_sha256_after": files_after,
        "source_sha256_before": source_before,
        "source_sha256_after": source_after,
        "policy_rng_sha256_before": hashlib.sha256(rng_before.encode()).hexdigest(),
        "policy_rng_sha256_after": hashlib.sha256(rng_after.encode()).hexdigest(),
        "checkpoint_training_schema": policy.training_state.get("schema"),
        "checkpoint_completed_episodes": policy.training_state.get("completed_episodes"),
        "input_schema": INPUT_SCHEMA, "feature_schema": FEATURE_SCHEMA,
        "seed_provenance": {"validation_diagnostic": {"start": seed, "count": episodes}},
        "training_performed": False, "final_test_accessed": False,
        "independent_unseen_evidence": False,
        "simulation_rollouts_performed": False, "private_truth_accessed": False,
        "scope": "First-action recommendation on public reset snapshots only; no full layout or episode outcome.",
        "interpretation": [
            "Low/zero marginal coverage is a public forecast feature, not proof of infeasibility or optimal stopping.",
            "Categorical entropy distributes mass over individual sensor-site options and one STOP option.",
            "The stop-versus-best probability renormalizes two existing probabilities for interpretation; it is not another evaluated policy.",
            "Forced STOP means no legal deployment is available and is counted separately from a learned STOP preference.",
            "Validation inputs may have been used during training/model selection; comparisons must match profile and public input hashes.",
        ],
        "groups": groups,
        "records": records,
    }
    _json(report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_PROBE_SEED)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--profile", choices=PROFILES, default="stress")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Probe output already exists; use a new report path")
    report = run_probe(args.checkpoint, seed=args.seed, episodes=args.episodes, profile=args.profile)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also avoids clobbering a report appearing mid-probe.
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    print(json.dumps(report["groups"], indent=2, allow_nan=False))
    print(f"Public-only validation STOP probe saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
