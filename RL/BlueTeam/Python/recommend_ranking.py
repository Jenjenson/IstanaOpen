"""Offline public-input sensor/site plans from an explicitly versioned ranker.

No rollout-label or environment module is imported. This emits recommendations,
not physical device commands. A custom capability catalogue is optional.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recommend_adaptive import recommend_layout
from triad_rl.ranking_policy import RankPolicy, POLICY_SCHEMA


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--catalogue", type=Path)
    parser.add_argument("--now", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a new output file; existing plans are never overwritten")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    catalogue = json.loads(args.catalogue.read_text(encoding="utf-8")) if args.catalogue else None
    policy = RankPolicy.load(args.checkpoint)
    report = recommend_layout(policy, payload, catalogue=catalogue, now=args.now)
    report.update(policy_schema=POLICY_SCHEMA,
                  decision_rule="joint legal argmax of frozen greedy prior plus learned residual, including STOP",
                  status="experimental_offline_recommendation_not_a_device_command")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        target.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(report['new_placements'])} recommendations; no device commands sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
