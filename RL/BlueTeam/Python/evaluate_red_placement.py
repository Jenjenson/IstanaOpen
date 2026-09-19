"""Compare frozen Red placement policies on identical live Unreal seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from triad_rl.istana_live import BridgeRejected, IstanaLiveClient, run_episode
from triad_rl.red_policy import (LearnedRedPlacementPolicy, RandomLegalRedPolicy,
    ScriptedRadialRedPolicy, public_approach_exposure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--red-checkpoint", type=Path, required=True)
    parser.add_argument("--blue-checkpoint", type=Path,
                        help="Optional frozen temporal checkpoint; default is temporal greedy")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1500000000)
    parser.add_argument("--random-policy-seed", type=int, default=8128)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=120.)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--step-batch", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.episodes <= 10000:
        parser.error("--episodes must be in [1, 10000]")
    if not (args.red_checkpoint / "checkpoint.json").is_file():
        parser.error("--red-checkpoint must contain checkpoint.json")
    if args.blue_checkpoint is not None and not (args.blue_checkpoint / "checkpoint.json").is_file():
        parser.error("--blue-checkpoint must contain checkpoint.json")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    policies = [ScriptedRadialRedPolicy(), RandomLegalRedPolicy(args.random_policy_seed),
                LearnedRedPlacementPolicy.load(args.red_checkpoint)]
    rows = []
    with IstanaLiveClient(args.port, args.timeout) as client:
        for policy in policies:
            for index in range(args.episodes):
                seed = args.seed + index
                try:
                    report = run_episode(client, seed=seed, checkpoint=args.blue_checkpoint,
                        temporal_public_control=args.blue_checkpoint is None,
                        red_policy=policy, red_deterministic=True,
                        max_steps=args.max_steps, step_batch=args.step_batch)
                    if report["measured_red_reward"] is None or report["measured_synthetic_metrics"] is None:
                        raise RuntimeError("Unreal episode ended without terminal reward/metrics")
                    rows.append({"policy": policy.name, "seed": seed, "valid": True,
                                 "red_reward": report["measured_red_reward"],
                                 "public_approach_exposure": public_approach_exposure(
                                     report["red_context"], report["public_blue_context"],
                                     report["plan"]["placements"], report["red_decision"]["centers"]),
                                 **report["measured_synthetic_metrics"]})
                except BridgeRejected as error:
                    rows.append({"policy": policy.name, "seed": seed, "valid": False,
                                 "error": str(error), "red_reward": None})
                print(f"policy={policy.name} episode={index + 1}/{args.episodes}", flush=True)
    aggregates = {}
    for policy in policies:
        selected = [row for row in rows if row["policy"] == policy.name]
        valid = [row for row in selected if row["valid"]]
        def mean(name):
            values = [row[name] for row in valid if row.get(name) is not None]
            return float(np.mean(values)) if values else None
        aggregates[policy.name] = {"episodes": len(selected), "valid_episodes": len(valid),
            "invalid_placements": len(selected) - len(valid), "mean_red_reward": mean("red_reward"),
            "mean_blue_reward": -mean("red_reward") if mean("red_reward") is not None else None,
            "mean_detected_fraction": mean("detected_fraction"),
            "mean_confirmed_fraction": mean("confirmed_fraction"),
            "mean_timely_fraction": mean("timely_fraction"),
            "mean_breached_fraction": mean("breached_fraction"), "mean_cost": mean("cost")}
        aggregates[policy.name]["mean_public_approach_exposure"] = mean("public_approach_exposure")
    result = {"schema": "istana.red_evaluation.v1", "first_seed": args.seed,
              "episodes_per_policy": args.episodes,
              "blue_opponent": str(args.blue_checkpoint) if args.blue_checkpoint else "temporal_public_control",
              "aggregates": aggregates, "episodes": rows,
              "claim": "Frozen-policy synthetic live comparison; not simultaneous self-play or real sensor validation"}
    (args.output_dir / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                                      encoding="utf-8")
    print(json.dumps(aggregates, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
