"""Public-only STOP probe using each version's actual deterministic decision rule."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from evaluate_robust import validate_seed_range
from probe_stop_exploration import (
    checkpoint_fingerprints, first_action_record as joint_record,
    source_fingerprints as base_sources, summarize_records,
)
from triad_rl.adaptive_evaluation import canonical_hash
from triad_rl.adaptive_inputs import FEATURE_NAMES
from triad_rl.adaptive_policy import AdaptivePolicy
from triad_rl.balanced_policy import BalancedPolicy
from triad_rl.robust_scenarios import RobustPlacementEnv


DEFAULT_SEED = 1_000_990_101_000_000  # Intentional reuse of published v2 diagnostic.


def source_fingerprints():
    root = Path(__file__).resolve().parent
    return {**base_sources(), **{
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("probe_balanced.py", "evaluate_robust.py", "triad_rl/balanced_policy.py",
                     "triad_rl/adaptive_evaluation.py")}}


def first_action_record(policy, observation, *, seed):
    record = joint_record(policy, observation, seed=seed)
    record["joint_argmax_stop"] = record["deterministic_stop"]
    action = policy.act(observation, deterministic=True)
    option = observation["options"][action]
    sensor = None if option["stop"] else observation["catalogue"][option["sensor_index"]]
    record["deterministic_stop"] = bool(option["stop"])
    record["first_action"] = {
        "index": action, "stop": bool(option["stop"]), "sensor_id": option["sensor_id"],
        "site_index": option["site_index"], "position": deepcopy(option["position"]),
        "cost": float(sensor["cost"]) if sensor else 0.}
    return record


def run_probe(checkpoint, *, policy_kind="balanced", seed=DEFAULT_SEED, episodes=60, profile="stress"):
    # Reuse is intentional; no claim of held-out or independent evidence.
    validate_seed_range(seed, episodes, "validation")
    if policy_kind not in ("balanced", "adaptive"):
        raise ValueError("policy_kind must be balanced or adaptive")
    if profile not in ("normal", "stress", "capability", "mixed"):
        raise ValueError("Unknown profile")
    path = Path(checkpoint)
    sources, files = source_fingerprints(), checkpoint_fingerprints(path)
    policy = (BalancedPolicy if policy_kind == "balanced" else AdaptivePolicy).load(path, feature_names=FEATURE_NAMES)
    weights, rng = policy.weights_fingerprint(), canonical_hash(policy.rng.bit_generator.state)
    env = RobustPlacementEnv(seed=seed, profile=profile)
    records = [first_action_record(policy, env.reset(seed=seed + index), seed=seed + index)
               for index in range(episodes)]
    groups = {"all": summarize_records(records)}
    for name, threshold in (("coverage_le_0.01", .01), ("coverage_le_0.001", .001), ("coverage_zero", 0.)):
        groups[name] = summarize_records([row for row in records if row["max_legal_marginal_coverage"] <= threshold])
    if (weights != policy.weights_fingerprint() or rng != canonical_hash(policy.rng.bit_generator.state)
            or sources != source_fingerprints() or files != checkpoint_fingerprints(path)):
        raise RuntimeError("Probe source/checkpoint/RNG drift")
    return {
        "schema": "triad.balanced_stop_probe.v1", "policy_kind": policy_kind,
        "policy_schema": policy.metadata["policy_schema"], "profile": profile,
        "weights_sha256": weights, "weights_sha256_after": weights,
        "checkpoint_files_sha256_before": files, "checkpoint_files_sha256_after": files,
        "policy_rng_sha256_before": rng, "policy_rng_sha256_after": rng,
        "source_sha256_before": sources, "source_sha256_after": sources,
        "stage": "validation", "training_performed": False, "final_test_accessed": False,
        "independent_unseen_evidence": False, "private_truth_accessed": False,
        "simulation_rollouts_performed": False,
        "seed_provenance": {"validation_diagnostic": {"start": seed, "count": episodes}},
        "scope": "Public initial snapshot and first recommendation only; no layout or outcome.",
        "interpretation": "Low forecast coverage is not a feasibility label. Deterministic STOP uses the actual versioned actor; joint_argmax_stop is diagnostic only. uniform_stop_probability describes the old flat categorical reference, not the balanced prior.",
        "groups": groups, "records": records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-kind", choices=("balanced", "adaptive"), default="balanced")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--profile", choices=("normal", "stress", "capability", "mixed"), default="stress")
    args = vars(parser.parse_args(argv))
    output = args.pop("output")
    if output.exists():
        raise FileExistsError("Use a new probe output path")
    report = run_probe(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["groups"], indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
