"""Offline external-input planner for the explicitly versioned balanced actor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recommend_adaptive import recommend_layout
from triad_rl.balanced_policy import BalancedPolicy, POLICY_SCHEMA


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--catalogue", type=Path)
    parser.add_argument("--now", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    catalogue = json.loads(args.catalogue.read_text(encoding="utf-8")) if args.catalogue else None
    policy = BalancedPolicy.load(args.checkpoint)
    report = recommend_layout(policy, payload, catalogue=catalogue, now=args.now)
    report["policy_schema"] = POLICY_SCHEMA
    report["decision_rule"] = "deploy-or-stop gate first; best conditional sensor/site on deploy"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(report['new_placements'])} recommendations; no device commands sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
