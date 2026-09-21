"""Run a temporal Blue sensor plan against the loopback Istana Unreal bridge.

Start Unreal with -IstanaBlueLive. Select --checkpoint explicitly (the checked
in temporal-v6 seed-406/last is a demo candidate, not a selected winner), or
--temporal-public-control for the greedy control, or --common-sense for a
simple public-coverage baseline. Use --paced for visible PIE.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from triad_rl.istana_live import IstanaLiveClient, run_episode
from triad_rl.red_policy import (DispersedRandomRedPolicy, LearnedRedPlacementPolicy,
                                 RandomLegalRedPolicy)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--checkpoint", type=Path, help="Explicit temporal checkpoint directory, never auto-selected")
    selection.add_argument("--temporal-public-control", "--greedy", action="store_true", help="Greedy non-RL placement using public predicted marginal return")
    selection.add_argument("--common-sense", action="store_true", help="Common-sense non-RL placement using ingress coverage, weather, spread and budget")
    red = parser.add_mutually_exclusive_group()
    red.add_argument("--red-checkpoint", type=Path, help="Trained Red placement checkpoint directory")
    red.add_argument("--red-random", action="store_true", help="Use a seeded random legal Red layout")
    red.add_argument("--red-dispersed", action="store_true",
                     help="Place Red groups at independent random directions for a visual demo")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=120.)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--step-batch", type=int, default=10)
    parser.add_argument("--paced", action="store_true", help="Pace fixed steps to simulation time for a visible demo")
    parser.add_argument("--output-dir", type=Path, required=True, help="A NEW directory for report.json and replay.jsonl")
    args = parser.parse_args(argv)
    if args.checkpoint is not None and not (args.checkpoint / "checkpoint.json").is_file():
        parser.error("--checkpoint must contain checkpoint.json")
    if args.red_checkpoint is not None and not (args.red_checkpoint / "checkpoint.json").is_file():
        parser.error("--red-checkpoint must contain checkpoint.json")
    # Reserve a new output directory before any reset/deployment occurs.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path, replay_path = args.output_dir / "report.json", args.output_dir / "replay.jsonl"
    with replay_path.open("x", encoding="utf-8") as replay:
        def record(value):
            replay.write(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")
            replay.flush()
        try:
            red_policy = (LearnedRedPlacementPolicy.load(args.red_checkpoint) if args.red_checkpoint else
                          DispersedRandomRedPolicy(args.seed) if args.red_dispersed else
                          RandomLegalRedPolicy(args.seed) if args.red_random else None)
            with IstanaLiveClient(args.port, args.timeout,
                                  record=lambda item: record({"type": "wire_audit", **item})) as client:
                report = run_episode(client, seed=args.seed, checkpoint=args.checkpoint,
                    temporal_public_control=args.temporal_public_control, common_sense=args.common_sense,
                    max_steps=args.max_steps,
                    red_policy=red_policy, step_batch=args.step_batch, paced=args.paced, frame=record)
        except Exception as error:
            report = {"schema": "istana.blue_live_run_failure.v1", "status": "failed",
                      "error_type": type(error).__name__, "error": str(error),
                      "recovery": "Inspect replay; reconnect and reset before another run. Mutations are never retried automatically."}
            with report_path.open("x", encoding="utf-8") as output:
                output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
            raise
    with report_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Unreal accepted {len(report['plan']['placements'])} Blue sensors; stepped {report['completed_steps']} ticks.")
    print(f"Measured synthetic metrics: {report['measured_synthetic_metrics']}")
    print(f"Saved report and simulation replay in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
