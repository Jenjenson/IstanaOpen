"""Recommend a coordinated layout from a simulated or external public snapshot.

This command is offline and read-only with respect to devices. It emits a plan,
not a deployment command. The same observation builder is used in training.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from triad_rl.adaptive_inputs import LiveObservationAdapter, apply_placement
from triad_rl.adaptive_policy import AdaptivePolicy


def recommend_layout(policy, payload: dict, *, catalogue=None, now=None) -> dict:
    adapter = LiveObservationAdapter(catalogue)
    state = copy.deepcopy(payload)
    initial = adapter.observe(state, now=now)
    decisions = []
    for _ in range(initial["state"]["max_sites"] + 1):
        observation = adapter.observe(state, now=now)
        action = policy.act(observation, deterministic=True)
        recommendation = adapter.recommendation(observation, action)
        decisions.append(recommendation)
        state = apply_placement(state, action, adapter.catalogue, now=now)
        if recommendation["stop"]:
            break
    else:
        raise RuntimeError("Policy did not commit a bounded deployment plan")
    return {
        "schema": "triad.deployment_recommendation.v1",
        "checkpoint_weights_sha256": policy.weights_fingerprint(),
        "snapshot_timestamp": initial["state"]["timestamp"],
        "evaluation_time": now if now is not None else initial["state"]["timestamp"],
        "source": initial["state"]["source"],
        "physical_commands_sent": False,
        "decisions": decisions,
        "new_placements": [decision for decision in decisions if not decision["stop"]],
        "final_public_state": state,
        "limitations": "Synthetic-trained recommendation only; sensor calibration and physical deployment require separate validation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="triad.sensor_input.v1 public snapshot")
    parser.add_argument("--catalogue", type=Path, help="Capability catalogue; defaults to the five synthetic profiles")
    parser.add_argument("--now", type=float, help="Current time in the provider clock; defaults to the snapshot time for offline replay")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    catalogue = json.loads(args.catalogue.read_text(encoding="utf-8")) if args.catalogue else None
    policy = AdaptivePolicy.load(args.checkpoint)
    report = recommend_layout(policy, payload, catalogue=catalogue, now=args.now)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(report['new_placements'])} placement recommendations; no device commands sent.")


if __name__ == "__main__":
    main()
