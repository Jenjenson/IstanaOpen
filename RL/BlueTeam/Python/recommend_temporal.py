"""Offline temporal sensor/site recommendations, never device commands.

The caller supplies public mission rules explicitly. Neither a private
scenario nor a simulator is imported by this inference entry point.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from triad_rl.adaptive_inputs import LiveObservationAdapter, apply_placement
from triad_rl.temporal_inputs import FEATURE_NAMES, TemporalConfig, TemporalObservationBuilder
from triad_rl.temporal_policy import POLICY_SCHEMA, TemporalPolicy


def recommend_layout(policy, payload, *, config: TemporalConfig, catalogue=None, now=None):
    if type(config) is not TemporalConfig or asdict(config) != asdict(policy.config):
        raise ValueError("Explicit public mission configuration must match the checkpoint")
    builder = TemporalObservationBuilder(config)
    state = json.loads(payload) if isinstance(payload, str) else deepcopy(payload)
    initial = builder.observe(state, catalogue, now=now)
    decisions, explanations = [], []
    if not initial["state"]["done"]:
        for _ in range(initial["state"]["max_sites"] + 1):
            observation = builder.observe(state, catalogue, now=now)
            action = policy.act(observation, deterministic=True)
            option = LiveObservationAdapter.recommendation(observation, action)
            decisions.append(option)
            row = observation["option_features"][action]
            explanations.append({"action": action, "score": float(policy.scores(observation)[action]),
                "forecast": {name: float(row[FEATURE_NAMES.index(name)]) for name in
                    ("temporal_detection", "temporal_confirmation", "temporal_timely",
                     "temporal_marginal_timely", "temporal_marginal_return")}})
            state = apply_placement(state, action, observation["catalogue"], now=now)
            if option["stop"]:
                break
        else:
            raise RuntimeError("Policy did not commit a bounded layout")
    return {"schema": "triad.temporal_deployment_recommendation.v1", "policy_schema": POLICY_SCHEMA,
        "checkpoint_weights_sha256": policy.weights_fingerprint(), "temporal_config": asdict(config),
        "snapshot_timestamp": initial["state"]["timestamp"],
        "evaluation_time": initial["state"]["timestamp"] if now is None else now,
        "source": initial["state"]["source"], "physical_commands_sent": False,
        "status": "experimental_offline_recommendation_not_a_device_command",
        "decisions": decisions, "new_placements": [row for row in decisions if not row["stop"]],
        "forecast_explanations": explanations, "final_public_state": state,
        "limitations": "Forecast values are approximate public-model estimates, not observed outcomes or calibrated real-world probabilities. Changed capabilities need evaluation; physical integration is not validated."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Explicit public TemporalConfig JSON")
    parser.add_argument("--catalogue", type=Path)
    parser.add_argument("--now", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Use a new output file; existing plans are never overwritten")
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    config = TemporalConfig(**read(args.config))
    policy = TemporalPolicy.load(args.checkpoint, config=config)
    report = recommend_layout(policy, read(args.input), config=config,
        catalogue=read(args.catalogue) if args.catalogue else None, now=args.now)
    raw = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(raw)
    print(f"Saved {len(report['new_placements'])} recommendations; no device commands sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
