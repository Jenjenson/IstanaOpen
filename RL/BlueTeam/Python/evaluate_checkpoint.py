"""Evaluate a NumPy Blue-placement checkpoint on held-out episode seeds.

Red follows the same reproducible script format as ``train_blue_placement.py``.
This command never calls the policy optimizer. Use ``--initial --stochastic``
as a dependency-free random-policy baseline, then rerun with ``--checkpoint``
and the same seed range for an apples-to-apples comparison. Live evaluation
requires Unreal to have been launched with ``-TRIADRLEvaluation``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Sequence

from train_blue_placement import DryRunTRIADEnv, _vector_argument
from triad_rl.environment import TRIADRedBlueEnv
from triad_rl.placement_policy import DynamicPlacementPolicy, PlacementFeatureAdapter
from triad_rl.remote_control import TRIADRemoteControlClient
from triad_rl.rollout import RedActionScript, aggregate_metrics, collect_blue_episode


EVALUATION_SCHEMA = "triad.dynamic_placement_evaluation.v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_config = Path(__file__).resolve().parents[1] / "DefaultTrainingConfig.json"
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Dynamic placement checkpoint directory")
    source.add_argument(
        "--initial", action="store_true",
        help="Evaluate a fresh policy (combine with --stochastic for the random baseline)",
    )
    parser.add_argument("--object-path", help="PIE object path of TRIAD_AdversarialTraining_Manager")
    parser.add_argument("--endpoint", default="http://127.0.0.1:30010")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output", type=Path, required=True, help="Fresh result directory")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2_000_000_000)
    parser.add_argument("--policy-seed", type=int, default=1337)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Sample reproducibly; default is catalogue argmax and position mean",
    )
    parser.add_argument("--red-script", type=Path, help="JSON RedActionScript configuration")
    parser.add_argument("--red-mode", choices=("fixed", "uniform"), default="fixed")
    parser.add_argument("--red-deployment", default="0,0,0,0,0")
    parser.add_argument("--red-velocity", default="0,-1,0")
    parser.add_argument(
        "--pace-seconds", "--step-seconds", dest="pace_seconds", type=float, default=0.0,
        help="Optional wall-clock pause after each action",
    )
    parser.add_argument("--hold-seconds", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true", help="Evaluate against the local CI toy")
    parser.add_argument("--dry-run-options", type=int, default=8)
    args = parser.parse_args(argv)
    if not args.dry_run and not args.object_path:
        parser.error("--object-path is required unless --dry-run is used")
    if not 1 <= args.episodes <= 10_000:
        parser.error("--episodes must be in [1, 10000]")
    if not 1_000_000_000 <= args.seed or args.seed + args.episodes > 2**31:
        parser.error("evaluation seeds must be a held-out range within [1e9, 2^31)")
    if not 0 <= args.policy_seed < 2**31:
        parser.error("--policy-seed must be in [0, 2^31)")
    if not 1 <= args.hidden_size <= 1024:
        parser.error("--hidden-size must be in [1, 1024]")
    if not 0.0 <= args.pace_seconds <= 10.0 or not 0.0 <= args.hold_seconds <= 30.0:
        parser.error("invalid pacing limits")
    if not 2 <= args.dry_run_options <= 256:
        parser.error("--dry-run-options must be in [2, 256]")
    try:
        args.red_deployment = _vector_argument(args.red_deployment, 5, "--red-deployment")
        args.red_velocity = _vector_argument(args.red_velocity, 3, "--red-velocity")
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    return args


def _red_script(args: argparse.Namespace) -> RedActionScript:
    if args.red_script:
        return RedActionScript.from_json(args.red_script)
    return RedActionScript(
        mode=args.red_mode,
        deployment=tuple(args.red_deployment),
        movement=tuple(args.red_velocity),
    )


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run:
        env: Any = DryRunTRIADEnv(args.dry_run_options)
    else:
        client = TRIADRemoteControlClient(args.object_path, endpoint=args.endpoint)
        env = TRIADRedBlueEnv(client, args.config)

    # Establish the exact named feature contract before accepting checkpoint
    # arrays. The first evaluated episode performs another clean reset.
    env.reset(seed=args.seed)
    adapter = PlacementFeatureAdapter(env)
    adapter.extract(env.observe("blue_placement"))
    if args.checkpoint:
        policy, metadata = DynamicPlacementPolicy.load_checkpoint(
            args.checkpoint, expected_feature_contract=adapter.feature_contract
        )
        source = {
            "kind": "checkpoint",
            "path": str(args.checkpoint.resolve()),
            "metadata": metadata,
        }
    else:
        policy = DynamicPlacementPolicy.from_adapter(
            adapter, hidden_size=args.hidden_size, seed=args.policy_seed
        )
        source = {
            "kind": "initial",
            "policy_seed": args.policy_seed,
            "specification": policy.specification,
        }

    red_script = _red_script(args)
    before = policy.parameter_hash()
    if not args.dry_run:
        blue_label = (
            f"placement-{source['kind']}:{before[:12]}:"
            f"{'stochastic' if args.stochastic else 'deterministic'}"
        )
        env.client.evaluation_labels(blue_label[:120], f"scripted-red:{red_script.mode}"[:120])

    args.output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        rollout = collect_blue_episode(
            env,
            policy,
            adapter,
            red_script,
            episode_seed,
            deterministic=not args.stochastic,
            pace_seconds=args.pace_seconds,
        )
        row = {"episode": episode + 1, **dict(rollout.metrics)}
        rows.append(row)
        (args.output / f"episode_{episode + 1:04d}.json").write_text(
            json.dumps(row, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(json.dumps({"kind": "episode", **row}, allow_nan=False), flush=True)
        if args.hold_seconds:
            time.sleep(args.hold_seconds)

    after = policy.parameter_hash()
    if before != after:
        raise RuntimeError("Evaluation changed policy parameters")
    report: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "training_performed": False,
        "source": source,
        "environment": {
            "observation_schema": str(getattr(env, "observation_schema", "unknown")),
            "config_fingerprint": getattr(
                env, "config_fingerprint", f"dry-run-options:{getattr(env, 'option_count', 0)}"
            ),
            "config_path": None if args.dry_run else str(args.config.resolve()),
        },
        "deterministic": not args.stochastic,
        "held_out_seed_start": args.seed,
        "held_out_seed_count": args.episodes,
        "parameter_sha256_before": before,
        "parameter_sha256_after": after,
        "red_script": {
            "mode": red_script.mode,
            "deployment": list(red_script.deployment),
            "movement": list(red_script.movement),
            "movement_steps": [list(row) for row in red_script.movement_steps],
        },
        "summary": aggregate_metrics(rows),
        "episodes": rows,
    }
    (args.output / "evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"kind": "complete", "output": str(args.output.resolve()),
                      "summary": report["summary"]}, allow_nan=False), flush=True)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    run_evaluation(parse_args(argv))


if __name__ == "__main__":
    main()
