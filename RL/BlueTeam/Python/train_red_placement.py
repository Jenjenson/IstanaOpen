"""Train the one-decision Red placement policy against a frozen Blue planner.

This is not simultaneous self-play: only Red parameters are updated. Unreal
owns placement validation, transitions, sensing, terminal reward and timing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from triad_rl.istana_live import BridgeRejected, IstanaLiveClient, run_episode
from triad_rl.red_policy import LearnedRedPlacementPolicy, public_approach_exposure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    blue = parser.add_mutually_exclusive_group()
    blue.add_argument("--blue-checkpoint", type=Path,
                      help="Optional frozen experimental temporal checkpoint; default is temporal greedy")
    blue.add_argument("--temporal-public-control", action="store_true",
                      help="Explicitly select the default frozen temporal greedy opponent")
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--seed", type=int, default=91000, help="First Unreal episode seed")
    parser.add_argument("--policy-seed", type=int, default=7301)
    parser.add_argument("--resume", type=Path,
                        help="Continue from a Red checkpoint; --episodes is the total target episode count")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Save a recovery checkpoint every N completed episodes (0 disables)")
    parser.add_argument("--learning-rate", type=float, default=.03)
    parser.add_argument("--baseline-rate", type=float, default=.1)
    parser.add_argument("--entropy-coefficient", type=float, default=.01)
    parser.add_argument("--invalid-reward", type=float, default=-10.)
    parser.add_argument("--exposure-weight", type=float, default=20.,
                        help="Train-only public path-exposure penalty; native reward remains separately logged")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=120.)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--step-batch", type=int, default=100)
    parser.add_argument("--paced", action="store_true",
                        help="Pace Unreal steps to simulation time for screen recording")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.episodes <= 100000:
        parser.error("--episodes must be in [1, 100000]")
    if not 0 <= args.checkpoint_every <= 100000:
        parser.error("--checkpoint-every must be in [0, 100000]")
    if args.blue_checkpoint is not None and not (args.blue_checkpoint / "checkpoint.json").is_file():
        parser.error("--blue-checkpoint must contain checkpoint.json")
    if args.resume is not None and not (args.resume / "checkpoint.json").is_file():
        parser.error("--resume must contain checkpoint.json")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = args.output_dir / "training.jsonl"
    policy = (LearnedRedPlacementPolicy.load(args.resume) if args.resume is not None
              else LearnedRedPlacementPolicy(seed=args.policy_seed))
    initial_episode = policy.episodes
    if initial_episode >= args.episodes:
        parser.error("--episodes must exceed the resumed checkpoint episode count")
    policy.save(args.output_dir / "initialized")
    rewards, native_rewards, exposures, invalid = [], [], [], 0
    completed_episode = initial_episode
    try:
        with metrics_path.open("x", encoding="utf-8") as metrics:
            with IstanaLiveClient(args.port, args.timeout) as client:
                for episode in range(initial_episode, args.episodes):
                    episode_seed = args.seed + episode
                    try:
                        report = run_episode(client, seed=episode_seed,
                            checkpoint=args.blue_checkpoint,
                        temporal_public_control=args.blue_checkpoint is None,
                        red_policy=policy, red_deterministic=False,
                        max_steps=args.max_steps, step_batch=args.step_batch,
                        paced=args.paced)
                        native_reward = report["measured_red_reward"]
                        if native_reward is None:
                            raise RuntimeError("Unreal episode ended without a terminal Red reward")
                        record = report["red_decision"]["training_record"]
                        exposure = public_approach_exposure(report["red_context"],
                            report["public_blue_context"], report["plan"]["placements"],
                            report["red_decision"]["centers"])
                        reward = native_reward - args.exposure_weight * exposure
                        outcome = {"metrics": report["measured_synthetic_metrics"],
                                   "completed_steps": report["completed_steps"],
                                   "native_red_reward": native_reward, "public_approach_exposure": exposure,
                                   "decision": {key: value for key, value in report["red_decision"].items()
                                                if key not in ("centers", "training_record")}}
                    except BridgeRejected as error:
                        # A rejected placement is an unambiguous non-transition. The
                        # bridge remains usable and the next reset starts a fresh run.
                        if not hasattr(policy, "last_record"):
                            raise
                        reward, native_reward, exposure, record, outcome = (args.invalid_reward, None, None,
                            policy.last_record, {
                            "invalid_placement": True, "error": str(error)}
                        )
                        invalid += 1
                    update = policy.update(record, reward, learning_rate=args.learning_rate,
                        baseline_rate=args.baseline_rate,
                        entropy_coefficient=args.entropy_coefficient)
                    rewards.append(reward)
                    if native_reward is not None: native_rewards.append(native_reward)
                    if exposure is not None: exposures.append(exposure)
                    completed_episode = episode + 1
                    row = {"schema": "istana.red_training_episode.v1", "episode": completed_episode,
                           "seed": episode_seed, "reward": reward, "update": update, **outcome}
                    metrics.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                    metrics.flush()
                    if args.checkpoint_every and completed_episode % args.checkpoint_every == 0:
                        policy.save(args.output_dir / f"checkpoint-{completed_episode:06d}")
                    print(f"episode={completed_episode} reward={reward:.4f} baseline={policy.baseline:.4f} "
                          f"action={record['action']} invalid={invalid}", flush=True)
    except BaseException:
        # A completed update is always recoverable even if Unreal closes or the
        # operator stops a recording between episodes.
        recovery = args.output_dir / f"interrupted-{completed_episode:06d}"
        if not recovery.exists():
            policy.save(recovery)
        raise
    policy.save(args.output_dir / "final")
    summary = {"schema": "istana.red_training_summary.v1", "episodes": args.episodes,
               "initial_episode": initial_episode, "episodes_this_run": len(rewards),
               "first_episode_seed": args.seed, "policy_seed": args.policy_seed,
               "resumed_from": str(args.resume) if args.resume is not None else None,
               "blue_opponent": (str(args.blue_checkpoint) if args.blue_checkpoint else "temporal_public_control"),
               "mean_training_reward": sum(rewards) / len(rewards),
               "mean_native_red_reward": sum(native_rewards) / len(native_rewards) if native_rewards else None,
               "mean_public_approach_exposure": sum(exposures) / len(exposures) if exposures else None,
               "exposure_weight": args.exposure_weight, "invalid_placements": invalid,
               "final_baseline": policy.baseline, "checkpoint": "final"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                                   encoding="utf-8")
    print(f"Saved Red checkpoint and {args.episodes} episodes in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
