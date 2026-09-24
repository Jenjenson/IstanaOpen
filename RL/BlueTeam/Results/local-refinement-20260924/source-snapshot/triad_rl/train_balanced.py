"""Train a count-balanced, gate-first sensor/site policy on robust scenarios.

The environment and reward remain unchanged. Transfer copies the declared
adaptive-v2 weights into a versioned BalancedPolicy and resets Adam/RNG; exact
resume preserves the complete optimizer, sampling RNG and validation cadence.
Model selection uses equal-profile validation return, never reserved final tests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

from .adaptive_policy import AdaptivePolicy
from .balanced_policy import BalancedPolicy, POLICY_SCHEMA
from .robust_scenarios import RobustPlacementEnv, curriculum_manifest
from .train_adaptive import SEED_RUN_WIDTH, _summary, scenario_seed
from .train_robust import (_json, _hash, _file_hash, _read_json, _integer, _ranges,
                           _checkpoint_files, rollout, validate, validation_seed_ranges)


TRAINING_SCHEMA = "triad.balanced_training.v1"
VALIDATION_PROFILES = ("normal", "stress", "capability")
SOURCE_FILES = ("adaptive_env.py", "adaptive_inputs.py", "adaptive_policy.py",
                "train_adaptive.py", "robust_scenarios.py", "train_robust.py",
                "balanced_policy.py", "train_balanced.py")


def source_provenance() -> dict[str, str]:
    """Bind this implementation and every reused training/scoring dependency."""
    return {name: _file_hash(Path(__file__).parent / name) for name in SOURCE_FILES}


def _selection_evidence(source: Path, original: AdaptivePolicy, report: Path) -> dict[str, Any]:
    """Bind v2's selected checkpoint and all candidates' complete seed exposure."""
    selection = _read_json(report)
    chosen = selection.get("selected", {})
    candidates = selection.get("candidate_metadata", {})
    exposure = selection.get("seed_provenance", {})
    lineage = exposure.get("lineage", {})
    candidate = candidates.get(str(chosen.get("seed")), {})
    if (selection.get("schema") != "triad.robust_model_selection.v1"
            or selection.get("stage") != "validation"
            or selection.get("final_test_accessed") is not False
            or chosen.get("weights_sha256") != original.weights_fingerprint()
            or not candidates or not isinstance(candidates, dict)
            or candidate.get("weights_sha256") != original.weights_fingerprint()
            or candidate.get("metadata") != original.metadata
            or candidate.get("checkpoint_files_sha256") != _checkpoint_files(source)
            or original.training_state.get("schema") != "triad.robust_training.v1"
            or not isinstance(exposure, dict) or not exposure):
        raise ValueError("Initial selection report does not bind the selected v2 checkpoint")
    metadata = {key: value.get("metadata") for key, value in candidates.items()}
    full_runs = {key: value.get("complete_run", {}).get("last_checkpoint_metadata")
                 for key, value in candidates.items()}
    if (lineage.get("candidate_checkpoints") != metadata
            or lineage.get("complete_training_runs") != full_runs):
        raise ValueError("Initial selection report omits complete candidate lineage")
    for key, value in candidates.items():
        last = full_runs[key]
        if not isinstance(last, dict):
            raise ValueError("Initial selection report is missing complete run metadata")
        state = last.get("training_state", {})
        config = state.get("config", {})
        summary = value.get("complete_run", {}).get("run_summary", {})
        completed = state.get("completed_episodes")
        _integer(completed, "inherited completed_episodes", 1, SEED_RUN_WIDTH - 1)
        batch = config.get("batch_size")
        _integer(batch, "inherited batch_size", 1, SEED_RUN_WIDTH - 1)
        seed = config.get("seed")
        expected_training = {"start": scenario_seed("train", seed, 0), "count": completed}
        if (state.get("schema") != "triad.robust_training.v1"
                or state.get("config_sha256") != _hash(config)
                or state.get("seed_provenance", {}).get("training") != expected_training
                or last.get("seed_provenance") != state.get("seed_provenance")
                or summary.get("seed_provenance") != state.get("seed_provenance")
                or summary.get("completed_episodes") != completed
                or completed % batch or last.get("update_count") != completed // batch
                or summary.get("last_weights_sha256") != last.get("weights_sha256")
                or summary.get("best_weights_sha256") != value.get("weights_sha256")):
            raise ValueError("Initial selection report has inconsistent complete run exposure")
    ranges = exposure.get("selection_validation")
    if (not isinstance(ranges, list) or len(ranges) != 3
            or {r.get("profile") for r in ranges} != set(VALIDATION_PROFILES)):
        raise ValueError("Initial selection report omits common-validation exposure")
    for interval in ranges:
        start, count = interval.get("start"), interval.get("count")
        _integer(start, "inherited validation start", 10**15, 2 * 10**15 - 1)
        _integer(count, "inherited validation count", 1, SEED_RUN_WIDTH - 1)
        if start + count > 2 * 10**15:
            raise ValueError("Initial selection report crosses the validation seed band")
    _ranges(exposure)
    return {"schema": selection["schema"], "report_sha256": _file_hash(report),
            "seed_provenance": exposure}


def _initial_policy(observation: dict[str, Any], seed: int, hidden_size: int | None,
                    checkpoint: str | Path | None,
                    selection_report: str | Path | None) -> tuple[BalancedPolicy, dict[str, Any]]:
    if checkpoint is None or selection_report is None:
        raise ValueError("New balanced training requires initial_checkpoint and initial_selection_report")
    source = Path(checkpoint)
    before_files, before_report = _checkpoint_files(source), _file_hash(Path(selection_report))
    original = AdaptivePolicy.load(source, feature_names=observation["feature_names"])
    if hidden_size is not None and hidden_size != original.hidden_size:
        raise ValueError("hidden_size differs from initial checkpoint")
    selection = _selection_evidence(source, original, Path(selection_report))
    policy = BalancedPolicy.transfer_from_adaptive(original, seed=seed)
    if (_checkpoint_files(source) != before_files
            or _file_hash(Path(selection_report)) != before_report):
        raise ValueError("Initial checkpoint or selection report changed while loading")
    return policy, {
        "kind": "transferred_adaptive_weights", "weights_sha256": policy.weights_fingerprint(),
        "source_weights_sha256": original.weights_fingerprint(),
        "source_policy_schema": original.metadata["policy_schema"],
        "source_files_sha256": before_files, "source_update_count": original.update_count,
        "source_training_schema": original.training_state.get("schema"),
        "source_seed_provenance": original.training_state.get("seed_provenance", {}),
        "source_selection": selection, "optimizer": "reset_zero_moments", "sampling_rng_seed": seed,
        "decision_semantics_changed": True}


def run_training(*, output: str | Path, episodes: int = 8000, batch_size: int = 16,
                 seed: int = 201, hidden_size: int | None = None,
                 learning_rate: float = 0.002, entropy_coef: float = 0.015,
                 gamma: float = 1.0, value_coef: float = 0.5, max_grad_norm: float = 1.0,
                 validation_every: int = 1000, validation_episodes: int = 60,
                 validation_run_seed: int = 991300,
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
        policy = BalancedPolicy.load(resume, feature_names=first_observation["feature_names"])
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
              "policy_schema": POLICY_SCHEMA,
              "deterministic_decision": "gate first: stop on >= 0.5; otherwise highest conditional deployment score",
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
        initial = BalancedPolicy.load(output / "initialized", feature_names=policy.feature_names)
        best = BalancedPolicy.load(output / "best", feature_names=policy.feature_names)
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
    if source_provenance() != config["source_sha256"]:
        raise RuntimeError("Source changed during preflight; output remains untouched")
    if saved is None and (
            _checkpoint_files(Path(initial_checkpoint)) != initialization["source_files_sha256"]
            or _file_hash(Path(initial_selection_report)) != initialization["source_selection"]["report_sha256"]):
        raise RuntimeError("Initial transfer artifacts changed during preflight; output remains untouched")
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
               "best_weights_sha256": BalancedPolicy.load(output / "best").weights_fingerprint(),
               "config_sha256": config_hash,
               "seed_provenance": checkpoint_state()["seed_provenance"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=8000, help="Batch-aligned total including resumed episodes")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--hidden-size", type=int, help="Must match the source checkpoint; normally omit")
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--entropy-coef", type=float, default=0.015)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validation-every", type=int, default=1000)
    parser.add_argument("--validation-episodes", type=int, default=60, help="Fixed cases per validation profile")
    parser.add_argument("--validation-run-seed", type=int, default=991300,
                        help="First of three adjacent reserved validation seed slots")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--initial-checkpoint", type=Path, help="Transfer weights only; resets Adam and sampling RNG")
    group.add_argument("--resume", type=Path, help="Exactly resume the same output run's latest last checkpoint")
    parser.add_argument("--initial-selection-report", type=Path,
                        help="Required for a new transfer: v2 selection report binding every inherited exposure")
    print(json.dumps(run_training(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
