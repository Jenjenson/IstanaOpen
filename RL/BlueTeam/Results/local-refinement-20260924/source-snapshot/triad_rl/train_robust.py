"""On-policy mixed-scenario training, without changing the frozen v1 benchmark.

Run ``python -m triad_rl.train_robust --help``. Checkpoint selection uses the
equal-weight mean return on three fixed validation profiles, never final tests.
Transfer initializes weights only; its new Adam state and sampling RNG are
explicitly distinguished from exact resume, which restores both unchanged.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

from .adaptive_policy import AdaptivePolicy
from .robust_scenarios import RobustPlacementEnv, curriculum_manifest
from .train_adaptive import SEED_RUN_WIDTH, _summary, scenario_seed


TRAINING_SCHEMA = "triad.robust_training.v1"
VALIDATION_PROFILES = ("normal", "stress", "capability")
SOURCE_FILES = ("adaptive_env.py", "adaptive_inputs.py", "adaptive_policy.py",
                "train_adaptive.py", "robust_scenarios.py", "train_robust.py")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"),
                      parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def source_provenance() -> dict[str, str]:
    return {name: _file_hash(Path(__file__).parent / name) for name in SOURCE_FILES}


def validation_seed_ranges(run_seed: int, count: int) -> dict[str, dict[str, int]]:
    """Each profile owns a separate bounded slot in the v1 validation band."""
    _integer(count, "validation_episodes", 1, SEED_RUN_WIDTH - 1)
    return {profile: {"start": scenario_seed("validation", run_seed + index, 0),
                      "count": count}
            for index, profile in enumerate(VALIDATION_PROFILES)}


def rollout(env: RobustPlacementEnv, policy: AdaptivePolicy, seed: int,
            training: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observation = env.reset(seed=seed)
    transitions: list[dict[str, Any]] = []
    total_reward, steps, done = 0.0, 0, False
    while not done:
        if training:
            action, transition = policy.sample(observation)
        else:
            action, transition = policy.act(observation, deterministic=True), {}
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
    return transitions, {**info, "episode_return": total_reward, "steps": steps,
                         "seed": seed, "case_metadata": copy.deepcopy(env.case_metadata)}


def validate(policy: AdaptivePolicy, count: int, run_seed: int) -> dict[str, Any]:
    """Deterministic equal-profile validation; neither weights nor RNG may move."""
    ranges = validation_seed_ranges(run_seed, count)
    before_weights, before_rng = policy.weights_fingerprint(), _json(policy.rng.bit_generator.state)
    profiles = {}
    for profile, seed_range in ranges.items():
        env = RobustPlacementEnv(profile=profile)
        records = [rollout(env, policy, seed_range["start"] + index)[1]
                   for index in range(count)]
        profiles[profile] = {"episodes": count, **_summary(records)}
    if policy.weights_fingerprint() != before_weights or _json(policy.rng.bit_generator.state) != before_rng:
        raise RuntimeError("Validation changed the policy or its training RNG")
    return {"profiles": profiles,
            "balanced_mean_return": float(np.mean([profiles[p]["mean_return"] for p in VALIDATION_PROFILES])),
            "balanced_success_rate": float(np.mean([profiles[p]["success_rate"] for p in VALIDATION_PROFILES])),
            "seed_ranges": ranges}


def _checkpoint_files(path: Path) -> dict[str, str]:
    return {name: _file_hash(path / name) for name in ("checkpoint.json", "arrays.npz")}


def _initial_policy(observation: dict[str, Any], seed: int, hidden_size: int | None,
                    checkpoint: str | Path | None,
                    selection_report: str | Path | None) -> tuple[AdaptivePolicy, dict[str, Any]]:
    if checkpoint is None:
        if selection_report is not None:
            raise ValueError("initial_selection_report requires an initial_checkpoint")
        policy = AdaptivePolicy(observation["feature_names"], seed=seed,
                                hidden_size=32 if hidden_size is None else hidden_size)
        return policy, {"kind": "fresh", "weights_sha256": policy.weights_fingerprint(),
                        "optimizer": "new_zero_moments", "sampling_rng_seed": seed}
    source = Path(checkpoint)
    original = AdaptivePolicy.load(source, feature_names=observation["feature_names"])
    if hidden_size is not None and hidden_size != original.hidden_size:
        raise ValueError("hidden_size differs from initial checkpoint")
    policy = AdaptivePolicy(observation["feature_names"], seed=seed, hidden_size=original.hidden_size)
    policy.parameters = {name: value.copy() for name, value in original.parameters.items()}
    provenance = {"kind": "transferred_weights", "weights_sha256": policy.weights_fingerprint(),
                    "source_files_sha256": _checkpoint_files(source),
                    "source_update_count": original.update_count,
                    "source_training_schema": original.training_state.get("schema"),
                    "source_seed_provenance": original.training_state.get("seed_provenance", {}),
                    "optimizer": "reset_zero_moments", "sampling_rng_seed": seed}
    if selection_report is not None:
        selection = _read_json(Path(selection_report))
        if (selection.get("schema") != "triad.adaptive_model_selection.v1"
                or selection.get("selection_split") != "validation"
                or selection.get("selected_weights_sha256") != original.weights_fingerprint()
                or not isinstance(selection.get("seed_provenance"), dict)
                or not selection["seed_provenance"]):
            raise ValueError("Initial selection report schema, split, weights or seed provenance differs")
        provenance["source_selection"] = {"schema": selection["schema"],
                                           "report_sha256": _file_hash(Path(selection_report)),
                                           "seed_provenance": selection["seed_provenance"]}
    return policy, provenance


def _integer(value: Any, name: str, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


def _ranges(value: Any) -> list[tuple[int, int]]:
    """Collect inherited exposure ranges without relying on an ancestor layout."""
    result: list[tuple[int, int]] = []
    if isinstance(value, dict):
        if "start" in value and "count" in value:
            start, count = value["start"], value["count"]
            if (isinstance(start, bool) or not isinstance(start, int) or start < 0
                    or isinstance(count, bool) or not isinstance(count, int) or count < 0):
                raise ValueError("Inherited seed range must use nonnegative integer start/count")
            if count:
                result.append((start, start + count))
        for item in value.values():
            result.extend(_ranges(item))
    elif isinstance(value, list):
        for item in value:
            result.extend(_ranges(item))
    return result


def run_training(*, output: str | Path, episodes: int = 8000, batch_size: int = 32,
                 seed: int = 101, hidden_size: int | None = None,
                 learning_rate: float = 0.002, entropy_coef: float = 0.02,
                 gamma: float = 1.0, value_coef: float = 0.5, max_grad_norm: float = 1.0,
                 validation_every: int = 1000, validation_episodes: int = 60,
                 validation_run_seed: int = 990100,
                 initial_checkpoint: str | Path | None = None,
                 initial_selection_report: str | Path | None = None,
                 resume: str | Path | None = None) -> dict[str, Any]:
    """Train a new run or exactly resume its most recent full-batch checkpoint.

    The requested total must be batch-aligned. Validation happens at fixed
    cadence crossings, not additionally at the requested endpoint: a temporary
    stop at any batch boundary cannot silently change checkpoint selection.
    Existing run files are never reused without an explicit matching resume.
    """
    for value, name in ((episodes, "episodes"), (batch_size, "batch_size"),
                        (validation_every, "validation_every"),
                        (validation_episodes, "validation_episodes")):
        _integer(value, name, 1, SEED_RUN_WIDTH - 1)
    if episodes % batch_size:
        raise ValueError("episodes must be a multiple of batch_size for exact full-batch resume")
    scenario_seed("train", seed, episodes - 1)
    _integer(validation_run_seed, "validation_run_seed", 0, 999997)
    validation_ranges = validation_seed_ranges(validation_run_seed, validation_episodes)
    numeric = (learning_rate, entropy_coef, gamma, value_coef, max_grad_norm)
    if (not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                and np.isfinite(value) for value in numeric)
            or learning_rate <= 0 or entropy_coef < 0 or not 0 <= gamma <= 1
            or value_coef < 0 or max_grad_norm <= 0):
        raise ValueError("Invalid finite training hyperparameters")
    if hidden_size is not None:
        _integer(hidden_size, "hidden_size", 1, 2048)
    if resume is not None and (initial_checkpoint is not None or initial_selection_report is not None):
        raise ValueError("initial_checkpoint/selection are a new transfer; they cannot be combined with resume")
    output = Path(output).resolve()
    config_path, log_path = output / "config.json", output / "training.jsonl"
    if resume is None and output.exists() and any(output.iterdir()):
        raise ValueError("Output already contains files; use explicit --resume or an empty output directory")
    train_env = RobustPlacementEnv(profile="mixed")
    first_observation = train_env.reset(seed=scenario_seed("train", seed, 0))
    saved = None
    if resume is not None:
        if Path(resume).resolve() != output / "last":
            raise ValueError("Resume must use this run directory's latest last checkpoint")
        saved = _read_json(config_path)
        policy = AdaptivePolicy.load(resume, feature_names=first_observation["feature_names"])
        initialization = saved["initialization"]
        if hidden_size is not None and hidden_size != policy.hidden_size:
            raise ValueError("hidden_size differs from resumed checkpoint")
    else:
        policy, initialization = _initial_policy(first_observation, seed, hidden_size,
                                                 initial_checkpoint, initial_selection_report)
    new_ranges = _ranges({"train": {"start": scenario_seed("train", seed, 0), "count": episodes},
                          "validation": validation_ranges})
    if any(max(start, old_start) < min(end, old_end)
           for start, end in new_ranges for old_start, old_end in _ranges(initialization)):
        raise ValueError("New training/validation seeds overlap inherited pretraining or selection exposure")
    config = {"schema": TRAINING_SCHEMA, "seed": seed, "batch_size": batch_size,
              "hidden_size": policy.hidden_size, "learning_rate": learning_rate,
              "entropy_coef": entropy_coef, "gamma": gamma, "value_coef": value_coef,
              "max_grad_norm": max_grad_norm, "validation_every": validation_every,
              "validation_episodes": validation_episodes, "validation_run_seed": validation_run_seed,
              "training_profile": "mixed", "validation_profiles": list(VALIDATION_PROFILES),
              "curriculum": curriculum_manifest(),
              "runtime": {"python": platform.python_version(), "numpy": np.__version__},
              "selection": "strictly greater equal-profile balanced_mean_return; ties keep earlier checkpoint",
              "initialization": initialization, "source_sha256": source_provenance()}
    config_hash = _hash(config)
    initialized_files: dict[str, str] = {}
    best_files: dict[str, str] = {}
    if saved is not None:
        state = policy.training_state
        if (saved.get("config_sha256") != config_hash
                or state.get("schema") != TRAINING_SCHEMA
                or state.get("config") != config
                or state.get("config_sha256") != config_hash
                or any(saved.get(key) != value for key, value in config.items())):
            raise ValueError("Resume source/configuration hash differs; do not change a recorded experiment")
        completed, best_episode = state["completed_episodes"], state["best_episode"]
        _integer(completed, "completed_episodes", 0, SEED_RUN_WIDTH - 1)
        _integer(best_episode, "best_episode", 0, completed)
        if completed % batch_size or policy.update_count != completed // batch_size:
            raise ValueError("Resume checkpoint is not a matching full-batch optimizer state")
        if completed >= episodes:
            raise ValueError("Requested total episodes must exceed resumed completed_episodes")
        best_validation = state["best_validation"]
        best_score = float(best_validation["balanced_mean_return"])
        if not np.isfinite(best_score):
            raise ValueError("Invalid resume selection score")
        initialized_files = _checkpoint_files(output / "initialized")
        best_files = _checkpoint_files(output / "best")
        if initialized_files != state.get("initialized_files_sha256"):
            raise ValueError("Immutable initialized checkpoint changed")
        if best_files != state.get("best_files_sha256"):
            raise ValueError("Selected best checkpoint changed")
        if _file_hash(log_path) != state.get("log_sha256"):
            raise ValueError("Training log differs from the latest checkpoint; cannot resume exactly")
        initial = AdaptivePolicy.load(output / "initialized", feature_names=policy.feature_names)
        best = AdaptivePolicy.load(output / "best", feature_names=policy.feature_names)
        if (initial.weights_fingerprint() != initialization["weights_sha256"]
                or initial.update_count != 0 or initial.training_state.get("completed_episodes") != 0
                or initial.training_state.get("config_sha256") != config_hash
                or best.training_state.get("completed_episodes") != best_episode
                or best.training_state.get("best_validation") != best_validation
                or best.training_state.get("config_sha256") != config_hash):
            raise ValueError("Initialization/selection state is inconsistent with the recorded run")
    else:
        completed, best_episode = 0, 0
        best_validation = validate(policy, validation_episodes, validation_run_seed)
        best_score = best_validation["balanced_mean_return"]
    # All preflight checks, including checkpoint and source integrity, finish
    # before changing an existing output directory.
    output.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({**config, "target_episodes": episodes,
                                      "config_sha256": config_hash,
                                      "python": platform.python_version(), "numpy": np.__version__},
                                     indent=2, allow_nan=False) + "\n", encoding="utf-8")
    started = time.monotonic()

    def write_event(event: dict[str, Any]) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(_json(event) + "\n")

    def checkpoint_state() -> dict[str, Any]:
        return {"schema": TRAINING_SCHEMA, "config": config, "config_sha256": config_hash,
                "completed_episodes": completed, "best_episode": best_episode,
                "best_validation": best_validation,
                "initialization": initialization,
                "initialized_files_sha256": initialized_files,
                "best_files_sha256": best_files,
                "log_sha256": _file_hash(log_path),
                "seed_provenance": {
                    "training": {"start": scenario_seed("train", seed, 0), "count": completed},
                    "validation": validation_ranges,
                    "inherited_training": initialization.get("source_seed_provenance", {}),
                    "inherited_selection": initialization.get("source_selection", {}).get("seed_provenance", {}),
                    "selection": config["selection"], "final_test": "never accessed by training"}}

    if saved is None:
        if source_provenance() != config["source_sha256"]:
            raise RuntimeError("Source changed during training; no new checkpoint will be published")
        write_event({"event": "validation", "episode": 0, "selected": True,
                     **best_validation, "weights_sha256": policy.weights_fingerprint(),
                     "initialization_kind": initialization["kind"]})
        policy.save(output / "initialized", checkpoint_state())
        initialized_files = _checkpoint_files(output / "initialized")
        policy.save(output / "best", checkpoint_state())
        best_files = _checkpoint_files(output / "best")
        policy.save(output / "last", checkpoint_state())
        print(_json({"episode": 0, "validation": best_validation,
                     "initialization_kind": initialization["kind"]}), flush=True)
    else:
        write_event({"event": "resume", "episode": completed,
                     "weights_sha256": policy.weights_fingerprint()})
    next_validation = (completed // validation_every + 1) * validation_every
    while completed < episodes:
        batch, records = [], []
        batch_start = completed
        for index in range(completed, completed + batch_size):
            transitions, record = rollout(train_env, policy, scenario_seed("train", seed, index), training=True)
            batch.append(transitions)
            records.append(record)
        update = policy.update(batch, learning_rate=learning_rate, entropy_coef=entropy_coef,
                               gamma=gamma, value_coef=value_coef, max_grad_norm=max_grad_norm)
        if source_provenance() != config["source_sha256"]:
            raise RuntimeError("Source changed during training; no new checkpoint will be published")
        completed += batch_size
        counts: dict[str, int] = {}
        for record in records:
            name = str(record["case_metadata"].get("profile", "unreported"))
            counts[name] = counts.get(name, 0) + 1
        write_event({"event": "training_batch", "episode_start": batch_start, "episode": completed,
                     "elapsed_seconds": time.monotonic() - started, "case_profile_counts": counts,
                     **_summary(records), **update})
        if completed >= next_validation:
            validation = validate(policy, validation_episodes, validation_run_seed)
            if source_provenance() != config["source_sha256"]:
                raise RuntimeError("Source changed during training; no new checkpoint will be published")
            selected = validation["balanced_mean_return"] > best_score
            if selected:
                best_score, best_episode, best_validation = validation["balanced_mean_return"], completed, validation
            event = {"event": "validation", "episode": completed, "selected": selected,
                     **validation, "weights_sha256": policy.weights_fingerprint(),
                     "elapsed_seconds": time.monotonic() - started}
            write_event(event)
            if selected:
                policy.save(output / "best", checkpoint_state())
                best_files = _checkpoint_files(output / "best")
            print(_json(event), flush=True)
            next_validation = (completed // validation_every + 1) * validation_every
        policy.save(output / "last", checkpoint_state())
    summary = {"schema": TRAINING_SCHEMA, "completed_episodes": completed,
               "optimizer_updates": policy.update_count, "best_episode": best_episode,
               "best_validation": best_validation,
               "best_validation_balanced_mean_return": best_score,
               "initialization": initialization, "initialized_files_sha256": initialized_files,
               "elapsed_seconds": time.monotonic() - started,
               "last_weights_sha256": policy.weights_fingerprint(),
               "best_weights_sha256": AdaptivePolicy.load(output / "best").weights_fingerprint(),
               "config_sha256": config_hash,
               "seed_provenance": checkpoint_state()["seed_provenance"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=8000, help="Batch-aligned total including resumed episodes")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--hidden-size", type=int, help="Fresh default 32; transfer inherits checkpoint size")
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validation-every", type=int, default=1000)
    parser.add_argument("--validation-episodes", type=int, default=60, help="Fixed cases per validation profile")
    parser.add_argument("--validation-run-seed", type=int, default=990100,
                        help="First of three adjacent reserved validation seed slots")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--initial-checkpoint", type=Path, help="Transfer weights only; resets Adam and sampling RNG")
    group.add_argument("--resume", type=Path, help="Exactly resume the same output run's latest last checkpoint")
    parser.add_argument("--initial-selection-report", type=Path,
                        help="v1 raw validation selection report; records and reserves inherited exposures")
    print(json.dumps(run_training(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
