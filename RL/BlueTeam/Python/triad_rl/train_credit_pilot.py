"""Bounded, fixed-endpoint credit-assignment ablation from the frozen v3 actor.

All arms use the unchanged BalancedPolicy.update, including its global batch
advantage normalization. Only the value-gradient coefficient and the detached
training baseline differ. Paired STOP is a training-only control variate, not
an optimal-action label. Equal initial RNG state does not promise matched later
action draws after different trajectory lengths. No validation or selection is
performed here; endpoints and implementation hashes must be predeclared.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from .adaptive_inputs import FEATURE_NAMES
from .balanced_policy import BalancedPolicy, POLICY_SCHEMA
from .counterfactual_rollout import counterfactual_rollout
from .credit_lineage import assert_disjoint, load_published_lineage
from .robust_scenarios import RobustPlacementEnv, curriculum_manifest
from .train_adaptive import _summary
from .train_robust import rollout


TRAINING_SCHEMA = "triad.credit_pilot_training.v1"
PROTOCOL_SCHEMA = "triad.credit_assignment_pilot_protocol.v1"
BLUE_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = BLUE_ROOT / "Python"
ARMS = {"shared_critic": {"value_coef": .5, "baseline": "learned_critic"},
        "no_critic_gradient": {"value_coef": 0., "baseline": "learned_critic"},
        "paired_stop": {"value_coef": 0., "baseline": "paired_stop"}}
SOURCE_FILES = tuple("triad_rl/" + name for name in (
    "adaptive_env.py", "adaptive_inputs.py", "adaptive_policy.py", "balanced_policy.py",
    "robust_scenarios.py", "train_adaptive.py", "train_robust.py", "counterfactual_rollout.py",
    "credit_lineage.py", "train_credit_pilot.py"))
STAT_FIELDS = ("return_to_go", "baseline", "raw_advantage", "normalized_advantage")
GROUPS = ("all", "first", "later", "stop", "deploy", "first_stop", "later_stop", "first_deploy", "later_deploy")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def bad(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)
    _json(value)  # Also rejects numeric overflow such as 1e999.
    if not isinstance(value, dict):
        raise ValueError("Protocol must be a JSON object")
    return value


def source_provenance():
    """Every directly reused policy/environment/rollout/lineage implementation."""
    return {name: _file_hash(PYTHON_ROOT / name) for name in SOURCE_FILES}


def _integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")


def _relative(path):
    if not isinstance(path, str) or "\\" in path:
        raise ValueError("Reference must be a BlueTeam-relative path")
    target = (BLUE_ROOT / path).resolve()
    if Path(path).is_absolute() or ".." in Path(path).parts or not target.is_relative_to(BLUE_ROOT.resolve()):
        raise ValueError("Reference escapes BlueTeam")
    return target


def _checkpoint_files(path):
    return {name: _file_hash(path / name) for name in ("checkpoint.json", "arrays.npz")}


def _empty_output(output):
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Pilot output must be new or empty; no resume or overwrite is supported")


def _protocol(path, arm):
    with path.open("rb") as handle:
        raw = handle.read(4_000_001)
    if len(raw) > 4_000_000:
        raise ValueError("Protocol exceeds size bound")
    protocol = _strict_json(raw)
    if protocol.get("schema") != PROTOCOL_SCHEMA or arm not in ARMS:
        raise ValueError("Unknown pilot protocol schema or arm")
    arms = protocol.get("arms")
    if (not isinstance(arms, dict) or set(arms) != set(ARMS)
            or any(not isinstance(arms[name], dict) or set(arms[name]) != set(spec)
                   or arms[name].get("baseline") != spec["baseline"]
                   or type(arms[name].get("value_coef")) not in (int, float)
                   or arms[name]["value_coef"] != spec["value_coef"] for name, spec in ARMS.items())):
        raise ValueError("Protocol must declare exactly the three fixed pilot arms")
    train = protocol.get("training", {})
    expected = {"policy_seed", "scenario_seed_start", "episodes", "batch_size", "learning_rate",
                "entropy_coef", "gamma", "max_grad_norm", "profile"}
    descriptive = {"initialization", "endpoint", "advantage_normalization", "experience_pairing"}
    if (not isinstance(train, dict) or not expected <= set(train) <= expected | descriptive
            or any(not isinstance(train[key], str) or not train[key] for key in set(train) & descriptive)):
        raise ValueError("Protocol training fields differ from the bounded pilot contract")
    _integer(train["policy_seed"], "policy_seed", 0, 999_999)
    _integer(train["episodes"], "episodes", 1, 1024)
    _integer(train["batch_size"], "batch_size", 1, 16)
    _integer(train["scenario_seed_start"], "scenario_seed_start", 0, 10**15 - 1024)
    if (train["episodes"] % train["batch_size"]
            or train["scenario_seed_start"] != train["policy_seed"] * 1_000_000):
        raise ValueError("Pilot requires full batches and its declared policy-seed scenario slot")
    for name, expected_value in (("learning_rate", .002), ("entropy_coef", .015),
                                 ("gamma", 1.), ("max_grad_norm", 1.)):
        value = train[name]
        if type(value) not in (int, float) or value != expected_value:
            raise ValueError(f"Pilot fixes {name}={expected_value}")
    if train["profile"] != "mixed":
        raise ValueError("Pilot training profile must be mixed")
    if protocol.get("training_implementation_sha256") != source_provenance():
        raise ValueError("Predeclared training source hashes differ")
    reference = protocol.get("frozen_reference", {})
    paths = {"checkpoint": "Checkpoints/balanced-v3-candidate",
             "selection_report": "Results/balanced-v3/selection.json",
             "publication_manifest": "Results/balanced-v3/artifact-manifest.json"}
    if any(reference.get(key) != value for key, value in paths.items()):
        raise ValueError("Pilot must start from the declared published v3 selection")
    return protocol, raw


def _initialize(reference, seed, lineage):
    checkpoint = _relative(reference["checkpoint"])
    base = BalancedPolicy.load(checkpoint, feature_names=FEATURE_NAMES)
    source = lineage["source_checkpoint"]
    files = _checkpoint_files(checkpoint)
    report_hash = _file_hash(_relative(reference["selection_report"]))
    manifest_hash = _file_hash(_relative(reference["publication_manifest"]))
    if (base.weights_fingerprint() != reference.get("weights_sha256")
            or base.weights_fingerprint() != source["weights_sha256"]
            or source["path"] != reference["checkpoint"]
            or source["checkpoint_files_sha256"] != files
            or source["selection_report"] != {"path": reference["selection_report"], "sha256": report_hash}
            or reference.get("selection_report_sha256") != report_hash
            or reference.get("publication_manifest_sha256") != manifest_hash):
        raise ValueError("Initial weights differ from the predeclared reference")
    policy = BalancedPolicy(base.feature_names, seed=seed, hidden_size=base.hidden_size)
    policy.parameters = {key: value.copy() for key, value in base.parameters.items()}
    # Constructor random draws are deliberately excluded from every arm's
    # starting sampling stream. Adam is already zero, and is checked below.
    policy.rng = np.random.default_rng(seed)
    if (policy.weights_fingerprint() != base.weights_fingerprint() or policy.update_count != 0
            or any(np.any(value) for values in (policy.adam_m, policy.adam_v) for value in values.values())):
        raise ValueError("Pilot initialization failed exact copied weights/zero optimizer contract")
    evidence = {"kind": "copied_balanced_v3_parameters", "weights_sha256": base.weights_fingerprint(),
                "checkpoint_files_sha256": files,
                "selection_report_sha256": report_hash,
                "publication_manifest_sha256": manifest_hash,
                "source_update_count": base.update_count, "sampling_rng_seed": seed,
                "sampling_rng_sha256": _hash(policy.rng.bit_generator.state),
                "optimizer": "zero_adam_moments_and_update_count", "decision_semantics_changed": False,
                "inherited_lineage_sha256": _hash(lineage)}
    return policy, evidence


def _credit_arrays(batch, gamma=1.):
    """Read-only mirror of the frozen update's exact RTG/normalization rule."""
    transitions, targets, first, stops = [], [], [], []
    for episode in batch:
        if not episode or len(episode) > 32:
            raise ValueError("Pilot requires complete bounded episodes")
        cumulative, episode_targets = 0., []
        for record in reversed(episode):
            cumulative = float(record["reward"]) + gamma * cumulative
            episode_targets.append(cumulative)
        for index, (record, target) in enumerate(zip(episode, reversed(episode_targets), strict=True)):
            transitions.append(record)
            targets.append(target)
            first.append(index == 0)
            stops.append(bool(record["features"][record["action"], FEATURE_NAMES.index("stop")]))
    rtg = np.asarray(targets)
    baseline = np.asarray([record["value"] for record in transitions])
    raw = rtg - baseline
    normalized = raw.copy()
    if len(normalized) > 1 and normalized.std() > 1e-8:
        normalized = (normalized - normalized.mean()) / (normalized.std() + 1e-8)
    matrix = np.column_stack((rtg, baseline, raw, normalized))
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite credit diagnostic")
    first, stops = np.asarray(first), np.asarray(stops)
    masks = {"all": np.ones(len(first), dtype=bool), "first": first, "later": ~first,
             "stop": stops, "deploy": ~stops, "first_stop": first & stops, "later_stop": ~first & stops,
             "first_deploy": first & ~stops, "later_deploy": ~first & ~stops}
    return transitions, rtg, normalized, {name: matrix[mask] for name, mask in masks.items()}


def _distribution(values):
    values = np.asarray(values)
    if not len(values):
        return {"count": 0, "mean": None, "variance": None, "negative": 0, "zero": 0, "positive": 0}
    return {"count": len(values), "mean": float(values.mean()), "variance": float(values.var()),
            "negative": int((values < 0).sum()), "zero": int((values == 0).sum()),
            "positive": int((values > 0).sum())}


def _credit_summary(groups):
    return {group: {field: _distribution(values[:, index]) for index, field in enumerate(STAT_FIELDS)}
            for group, values in groups.items()}


def _gradient_audit(policy, transitions, returns, advantages, entropy_coef, value_coef):
    _, actor, _ = policy._loss_and_gradients(transitions, returns, advantages, entropy_coef, 0.)
    if value_coef:
        _, total, _ = policy._loss_and_gradients(transitions, returns, advantages, entropy_coef, value_coef)
    else:
        total = actor
    critic = {key: total[key] - actor[key] for key in total}
    norm = lambda gradients: float(np.sqrt(sum(np.sum(value * value) for value in gradients.values())))
    actor_norm, critic_norm = norm(actor), norm(critic)
    dot = float(sum(np.sum(actor[key] * critic[key]) for key in actor))
    cosine = max(-1., min(1., dot / (actor_norm * critic_norm))) if actor_norm and critic_norm else None
    return {"actor_plus_entropy_gradient_norm": actor_norm, "weighted_critic_gradient_norm": critic_norm,
            "diagnostic_total_gradient_norm": norm(total), "actor_critic_gradient_dot": dot,
            "actor_critic_gradient_cosine": cosine}


def _trajectory_digest(batch, seed_start):
    """Exact sampled trajectory identity, deliberately excluding all baselines."""
    digest = hashlib.sha256(_json({"schema": "triad.credit_sampled_trajectory.v1",
                                   "feature_names": FEATURE_NAMES, "episode_count": len(batch)}).encode())
    for offset, episode in enumerate(batch):
        digest.update(_json({"scenario_seed": seed_start + offset, "decisions": len(episode)}).encode())
        for record in episode:
            matrix = np.asarray(record["features"], dtype="<f8")
            mask = np.asarray(record["mask"], dtype=np.bool_)
            digest.update(_json({"shape": matrix.shape, "action": int(record["action"]),
                                 "reward": float(record["reward"])}).encode())
            digest.update(matrix.tobytes(order="C"))
            digest.update(mask.tobytes(order="C"))
    return digest.hexdigest()


def run_training(protocol_path: str | Path, arm: str, output: str | Path):
    """Run one predeclared arm to its fixed endpoint; never select or resume."""
    started = time.perf_counter()
    output, protocol_path = Path(output).resolve(), Path(protocol_path).resolve()
    _empty_output(output)
    protocol, protocol_bytes = _protocol(protocol_path, arm)
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    lineage = load_published_lineage()
    train, spec = protocol["training"], ARMS[arm]
    seed_start, count = train["scenario_seed_start"], train["episodes"]
    assert_disjoint(seed_start, count, lineage, label="credit pilot training")
    policy, initialization = _initialize(protocol["frozen_reference"], train["policy_seed"], lineage)
    initial_critic = {group: {key: getattr(policy, group)[key].copy() for key in ("wv", "bv")}
                      for group in ("parameters", "adam_m", "adam_v")}
    sources = source_provenance()

    def stable():
        reference = protocol["frozen_reference"]
        if (source_provenance() != sources or protocol_path.read_bytes() != protocol_bytes
                or any(_file_hash(_relative(entry["path"])) != entry["sha256"]
                       for entry in lineage["publication_manifests"])
                or _checkpoint_files(_relative(reference["checkpoint"])) != initialization["checkpoint_files_sha256"]
                or _file_hash(_relative(reference["selection_report"])) != initialization["selection_report_sha256"]
                or _file_hash(_relative(reference["publication_manifest"])) != initialization["publication_manifest_sha256"]):
            raise ValueError("Pilot source/protocol/reference changed; no further output is permitted")

    config = {"schema": TRAINING_SCHEMA, "arm": arm, "training": train, "arm_specification": spec,
              "policy_schema": POLICY_SCHEMA, "protocol_sha256": protocol_hash,
              "source_sha256": sources, "initialization": initialization, "inherited_lineage": lineage,
              "curriculum": curriculum_manifest(), "checkpoint_rule": "fixed_endpoint_only",
              "advantage_normalization": "unchanged BalancedPolicy.update: global batch centering/std if n>1 and std>1e-8",
              "baseline_caveats": "No-critic-gradient retains learned value estimates; shared actor encoder can change them. Paired STOP is not an optimal-action label. Raw STOP advantage zero may normalize nonzero.",
              "rng_pairing": "same initial sampling RNG state; later streams need not stay episode-paired",
              "runtime": {"python": platform.python_version(), "numpy": np.__version__}}
    config_hash = _hash(config)
    stable()
    _empty_output(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_bytes(protocol_bytes)
    (output / "config.json").write_text(_json({**config, "config_sha256": config_hash}) + "\n", encoding="utf-8")
    log_path = output / "training.jsonl"
    completed, branch_count, live_steps = 0, 0, 0
    total_rollout_seconds = total_update_seconds = total_audit_seconds = 0.
    all_credit = {group: [] for group in GROUPS}
    all_records, initialized_files, first_trajectory = [], {}, None

    def state():
        return {"schema": TRAINING_SCHEMA, "config": config, "config_sha256": config_hash,
                "protocol_sha256": protocol_hash, "source_sha256": sources,
                "arm": arm, "completed_episodes": completed, "target_episodes": count,
                "initialized_files_sha256": initialized_files,
                "log_sha256": _file_hash(log_path),
                "seed_provenance": {"training": {"start": seed_start, "count": completed},
                                    "inherited": lineage, "validation": "never accessed by trainer",
                                    "final_test": "never accessed by trainer"}}

    def event(row):
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(_json(row) + "\n")

    event({"event": "initialization", "episode": 0, "arm": arm,
           "weights_sha256": policy.weights_fingerprint(), "policy_rng_sha256": initialization["sampling_rng_sha256"]})
    policy.save(output / "initialized", state())
    initialized_files = _checkpoint_files(output / "initialized")
    env = RobustPlacementEnv(seed=seed_start, profile="mixed")
    training_started = time.perf_counter()
    while completed < count:
        batch, records = [], []
        batch_started = time.perf_counter()
        for index in range(completed, completed + train["batch_size"]):
            if arm == "paired_stop":
                transitions, record = counterfactual_rollout(env, policy, seed_start + index, gamma=1.)
                if record["counterfactual_baseline"]["branch_rollouts"] != len(transitions):
                    raise ValueError("Paired STOP branch accounting differs from actual decisions")
            else:
                transitions, record = rollout(env, policy, seed_start + index, training=True)
            batch.append(transitions)
            records.append(record)
        rollout_seconds = time.perf_counter() - batch_started
        audit_started = time.perf_counter()
        transitions, returns, advantages, credit = _credit_arrays(batch)
        trajectory = _trajectory_digest(batch, seed_start + completed)
        if completed == 0:
            first_trajectory = trajectory
        gradients = _gradient_audit(policy, transitions, returns, advantages, train["entropy_coef"], spec["value_coef"])
        audit_seconds = time.perf_counter() - audit_started
        old_parameters = {key: value.copy() for key, value in policy.parameters.items()}
        update_started = time.perf_counter()
        metrics = policy.update(batch, learning_rate=train["learning_rate"], entropy_coef=train["entropy_coef"],
                                gamma=train["gamma"], value_coef=spec["value_coef"], max_grad_norm=train["max_grad_norm"])
        update_seconds = time.perf_counter() - update_started
        if metrics["gradient_norm"] != gradients["diagnostic_total_gradient_norm"]:
            raise ValueError("Read-only gradient audit differs from unchanged update")
        stable()
        begin, completed = completed, completed + train["batch_size"]
        extra = sum(record.get("counterfactual_baseline", {}).get("branch_rollouts", 0) for record in records)
        branch_count += extra
        live_steps += len(transitions)
        total_rollout_seconds += rollout_seconds
        total_update_seconds += update_seconds
        total_audit_seconds += audit_seconds
        for group, values in credit.items():
            all_credit[group].append(values)
        # Store only scalar episode summaries, never simulator truth/replay data.
        all_records.extend({key: record[key] for key in ("episode_return", "success", "detection_rate", "coverage",
                           "breach_rate", "total_cost", "invalid_actions", "steps")} for record in records)
        sensor_counts = Counter(placement["sensor_id"] for record in records for placement in record["placements"])
        parameter_delta = lambda keys: float(np.sqrt(sum(np.sum((policy.parameters[key] - old_parameters[key]) ** 2) for key in keys)))
        row = {"event": "training_batch", "arm": arm, "episode_start": begin, "episode": completed,
               "trajectory_sha256": trajectory,
               "scenario_seed_range": {"start": seed_start + begin, "count": train["batch_size"]},
               "case_profile_counts": dict(Counter(record["case_metadata"]["profile"] for record in records)),
               "sensor_deployment_counts": dict(sensor_counts), "credit": _credit_summary(credit),
               **_summary(records), **metrics, **gradients,
               "clipped_gradient_norm": metrics["gradient_norm"] * metrics["gradient_scale"],
               "parameter_delta_l2": parameter_delta(policy.parameters), "critic_head_delta_l2": parameter_delta(("wv", "bv")),
               "compute": {"live_episode_rollouts": len(records), "extra_branch_rollouts": extra,
                           "live_decisions": len(transitions), "cumulative_live_episode_rollouts": completed,
                           "cumulative_extra_branch_rollouts": branch_count, "rollout_wall_seconds": rollout_seconds,
                           "gradient_audit_wall_seconds": audit_seconds, "update_wall_seconds": update_seconds,
                           "training_elapsed_seconds": time.perf_counter() - training_started},
               "weights_sha256": policy.weights_fingerprint()}
        event(row)
        print(_json({"arm": arm, "episode": completed, "mean_return": row["mean_return"],
                     "extra_branch_rollouts": branch_count}), flush=True)
    if load_published_lineage() != lineage:
        raise ValueError("Inherited published evidence changed during pilot")
    if not spec["value_coef"] and any(not np.array_equal(getattr(policy, group)[key], value)
            for group, arrays in initial_critic.items() for key, value in arrays.items()):
        raise ValueError("Zero-value-gradient arm changed critic head parameters or Adam moments")
    if _checkpoint_files(output / "initialized") != initialized_files:
        raise ValueError("Immutable initialized checkpoint changed during pilot")
    stable()
    policy.save(output / "last", state())
    summary = {"schema": TRAINING_SCHEMA, "arm": arm, "completed_episodes": completed,
               "optimizer_updates": policy.update_count, "checkpoint_rule": "fixed_endpoint_only",
               "initial_weights_sha256": initialization["weights_sha256"], "last_weights_sha256": policy.weights_fingerprint(),
               "first_batch_trajectory_sha256": first_trajectory,
               "initialized_files_sha256": initialized_files, "last_files_sha256": _checkpoint_files(output / "last"),
               "config_sha256": config_hash, "protocol_sha256": protocol_hash, "source_sha256": sources,
               "log_sha256": _file_hash(log_path), "seed_provenance": state()["seed_provenance"],
               "credit": _credit_summary({group: np.concatenate(values, axis=0) for group, values in all_credit.items()}),
               **_summary(all_records),
               "compute": {"live_episode_rollouts": completed, "extra_branch_rollouts": branch_count,
                           "live_decisions": live_steps, "rollout_wall_seconds": total_rollout_seconds,
                           "gradient_audit_wall_seconds": total_audit_seconds, "update_wall_seconds": total_update_seconds,
                           "training_elapsed_seconds": time.perf_counter() - training_started,
                           "total_elapsed_seconds": time.perf_counter() - started,
                           "interpretation": "Matched live episodes are not matched simulation compute; paired STOP runs one extra terminal branch per live decision."}}
    (output / "summary.json").write_text(_json(summary) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", dest="protocol_path", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    summary = run_training(**vars(parser.parse_args(argv)))
    print(_json({key: summary[key] for key in ("arm", "completed_episodes", "optimizer_updates",
                                              "last_weights_sha256", "mean_return")}))


if __name__ == "__main__":
    main()
