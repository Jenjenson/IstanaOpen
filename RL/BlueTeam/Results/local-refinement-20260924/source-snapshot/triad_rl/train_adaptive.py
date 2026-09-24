"""Train joint Blue sensor/site decisions using randomized simulation episodes.

Run ``python -m triad_rl.train_adaptive --help``. Validation may select the
checkpoint; final test scenarios are never sampled here. Resume restores
weights, optimizer, RNG, completed episode index, and selection state exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

from .adaptive_env import AdaptivePlacementEnv
from .adaptive_policy import AdaptivePolicy


TRAINING_SCHEMA = "triad.adaptive_training.v1"
VALIDATION_SEED_BASE = 10 ** 15
FINAL_TEST_SEED_BASE = 2 * 10 ** 15
SEED_RUN_WIDTH = 1_000_000


def source_provenance() -> dict[str, str]:
    """Bind resumed training to the actual feature, physics and learning code."""
    package = Path(__file__).parent
    return {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
            for name in ("adaptive_env.py", "adaptive_inputs.py", "adaptive_policy.py", "train_adaptive.py")}


def scenario_seed(split: str, run_seed: int, episode: int) -> int:
    """Nonoverlapping train/validation/test bands, each with bounded run slots."""
    if (isinstance(run_seed, bool) or not isinstance(run_seed, int) or not 0 <= run_seed < 1_000_000
            or isinstance(episode, bool) or not isinstance(episode, int) or not 0 <= episode < SEED_RUN_WIDTH):
        raise ValueError("run_seed and episode must be integers in [0, 1000000)")
    bases = {"train": 0, "validation": VALIDATION_SEED_BASE, "test": FINAL_TEST_SEED_BASE}
    return bases[split] + int(run_seed) * SEED_RUN_WIDTH + int(episode)


def _summary(episodes: list[dict[str, Any]]) -> dict[str, float]:
    if not episodes:
        return {}
    mappings = {"mean_return": "episode_return", "success_rate": "success",
                "detection_rate": "detection_rate", "coverage": "coverage",
                "breach_rate": "breach_rate", "mean_cost": "total_cost",
                "mean_invalid_actions": "invalid_actions", "mean_steps": "steps"}
    return {target: float(np.mean([episode.get(source, 0.0) for episode in episodes]))
            for target, source in mappings.items()}


def rollout(env: AdaptivePlacementEnv, policy: AdaptivePolicy, seed: int,
            training: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observation = env.reset(seed=seed)
    transitions: list[dict[str, Any]] = []
    total_reward = 0.0
    done = False
    info: dict[str, Any] = {}
    steps = 0
    while not done:
        if training:
            action, transition = policy.sample(observation)
        else:
            action = policy.act(observation, deterministic=True)
            transition = {}
        observation, reward, done, info = env.step(action)
        if not np.isfinite(reward):
            raise ValueError("Environment returned nonfinite reward")
        total_reward += float(reward)
        steps += 1
        if training:
            transition["reward"] = float(reward)
            transitions.append(transition)
        if steps > 1000:
            raise RuntimeError("Episode did not terminate within 1000 decisions")
    return transitions, {**info, "episode_return": total_reward, "steps": steps, "seed": seed}


def validate(policy: AdaptivePolicy, count: int, run_seed: int) -> dict[str, float]:
    env = AdaptivePlacementEnv(split="heldout")
    records = [rollout(env, policy, scenario_seed("validation", run_seed, index))[1]
               for index in range(count)]
    return _summary(records)


def run_training(*, output: str | Path, episodes: int = 4000, batch_size: int = 16,
                 seed: int = 42, hidden_size: int = 32, learning_rate: float = 0.004,
                 entropy_coef: float = 0.015, gamma: float = 0.99,
                 value_coef: float = 0.5, max_grad_norm: float = 1.0,
                 validation_every: int = 400, validation_episodes: int = 100,
                 resume: str | Path | None = None) -> dict[str, Any]:
    if (not 1 <= episodes < SEED_RUN_WIDTH or batch_size < 1 or validation_every < 1
            or not 1 <= validation_episodes < SEED_RUN_WIDTH):
        raise ValueError("Positive episodes/batch/validation sizes required; seed slots cannot overlap")
    scenario_seed("train", seed, episodes - 1)
    output = Path(output)
    config = {"schema": TRAINING_SCHEMA, "seed": seed, "batch_size": batch_size,
              "hidden_size": hidden_size, "learning_rate": learning_rate,
              "entropy_coef": entropy_coef, "gamma": gamma, "value_coef": value_coef,
              "max_grad_norm": max_grad_norm, "validation_every": validation_every,
              "validation_episodes": validation_episodes, "source_sha256": source_provenance()}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()
    train_env = AdaptivePlacementEnv(split="train")
    first_observation = train_env.reset(seed=scenario_seed("train", seed, 0))
    if resume is not None:
        policy = AdaptivePolicy.load(resume, feature_names=first_observation["feature_names"])
        state = policy.training_state
        if state.get("config_sha256") != config_hash:
            raise ValueError("Resume configuration differs; preserve optimizer/training hyperparameters")
        completed = int(state["completed_episodes"])
        best_score = tuple(state["best_validation_score"])
        best_episode = int(state["best_episode"])
        if completed >= episodes:
            raise ValueError("Requested total episodes must exceed resumed completed_episodes")
        if completed % batch_size:
            raise ValueError("Exact resume requires a full-batch checkpoint; choose a batch-aligned total")
    else:
        if (output / "training.jsonl").exists() or (output / "last").exists():
            raise ValueError("Output already has a run; use --resume or a new output directory")
        policy = AdaptivePolicy(first_observation["feature_names"], seed=seed, hidden_size=hidden_size)
        completed, best_episode = 0, 0
        best_score = (-float("inf"), -float("inf"))
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    config_path.write_text(json.dumps({**config, "target_episodes": episodes,
                                      "config_sha256": config_hash,
                                      "python": platform.python_version(), "numpy": np.__version__},
                                     indent=2, allow_nan=False) + "\n", encoding="utf-8")
    log_path = output / "training.jsonl"
    started = time.monotonic()

    def write_event(event: dict[str, Any]) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")

    def checkpoint_state() -> dict[str, Any]:
        return {"schema": TRAINING_SCHEMA, "config": config, "config_sha256": config_hash,
                "completed_episodes": completed, "best_validation_score": list(best_score),
                "best_episode": best_episode,
                "seed_provenance": {
                    "training": {"start": scenario_seed("train", seed, 0), "count": completed},
                    "validation": {"start": scenario_seed("validation", seed, 0),
                                   "count": validation_episodes},
                    "selection": "deterministic validation success_rate then mean_return",
                    "final_test": "never accessed by training"}}

    if resume is None:
        initial = validate(policy, validation_episodes, seed)
        best_score = (initial["success_rate"], initial["mean_return"])
        state = checkpoint_state()
        policy.save(output / "initialized", state)
        policy.save(output / "best", state)
        write_event({"event": "validation", "episode": 0, "selected": True, **initial,
                     "weights_sha256": policy.weights_fingerprint()})
        print(json.dumps({"episode": 0, "validation": initial}), flush=True)
    else:
        write_event({"event": "resume", "episode": completed,
                     "weights_sha256": policy.weights_fingerprint()})

    next_validation = (completed // validation_every + 1) * validation_every
    while completed < episodes:
        records, batch = [], []
        batch_start = completed
        for index in range(completed, min(episodes, completed + batch_size)):
            transitions, record = rollout(train_env, policy, scenario_seed("train", seed, index), training=True)
            batch.append(transitions)
            records.append(record)
        update = policy.update(batch, learning_rate=learning_rate, entropy_coef=entropy_coef,
                               gamma=gamma, value_coef=value_coef, max_grad_norm=max_grad_norm)
        completed += len(batch)
        write_event({"event": "training_batch", "episode_start": batch_start,
                     "episode": completed, "elapsed_seconds": time.monotonic() - started,
                     **_summary(records), **update})
        if completed >= next_validation or completed == episodes:
            validation = validate(policy, validation_episodes, seed)
            score = (validation["success_rate"], validation["mean_return"])
            selected = score > best_score
            if selected:
                best_score, best_episode = score, completed
                policy.save(output / "best", checkpoint_state())
            policy.save(output / "last", checkpoint_state())
            event = {"event": "validation", "episode": completed, "selected": selected,
                     **validation, "weights_sha256": policy.weights_fingerprint(),
                     "elapsed_seconds": time.monotonic() - started}
            write_event(event)
            print(json.dumps(event), flush=True)
            next_validation = (completed // validation_every + 1) * validation_every
    summary = {"completed_episodes": completed, "best_episode": best_episode,
               "best_validation_success_rate": best_score[0],
               "best_validation_mean_return": best_score[1],
               "elapsed_seconds": time.monotonic() - started,
               "last_weights_sha256": policy.weights_fingerprint(),
               "config_sha256": config_hash, "seed_provenance": checkpoint_state()["seed_provenance"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=4000, help="Total episodes including resumed episodes")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.004)
    parser.add_argument("--entropy-coef", type=float, default=0.015)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validation-every", type=int, default=400)
    parser.add_argument("--validation-episodes", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    print(json.dumps(run_training(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
