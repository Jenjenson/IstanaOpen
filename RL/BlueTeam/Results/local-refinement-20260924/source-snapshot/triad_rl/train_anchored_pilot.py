"""Fixed-endpoint anchored-credit pilot; no selection, resume or physical control.

Controls retain the frozen BalancedPolicy updater. The experimental helper only
changes later-decision coefficients; each first coefficient must be byte-exact
to the ordinary critic-normalized reference on the same sampled batch. All
published predecessors remain unchanged and are retained through compact lineage.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import platform
import time

import numpy as np

from . import anchored_credit
from .anchored_lineage import assert_disjoint, load_published_lineage
from .balanced_policy import BalancedPolicy, POLICY_SCHEMA
from .robust_scenarios import RobustPlacementEnv, curriculum_manifest
from .train_adaptive import _summary
from .train_robust import rollout
from . import train_credit_pilot as shared


TRAINING_SCHEMA = "triad.anchored_pilot_training.v1"
PROTOCOL_SCHEMA = "triad.anchored_credit_pilot_protocol.v1"
ARMS = {"shared_critic": {"baseline": "learned_critic", "value_coef": .5},
        "no_critic_gradient": {"baseline": "learned_critic", "value_coef": 0.},
        "anchored_later_stop": {"baseline": "anchored_later_stop", "value_coef": 0.}}
RUN_FILES = ("config.json", "protocol.json", "training.jsonl", "summary.json",
             "initialized/checkpoint.json", "initialized/arrays.npz", "last/checkpoint.json", "last/arrays.npz")
SOURCE_FILES = (*shared.SOURCE_FILES, "triad_rl/anchored_credit.py", "triad_rl/anchored_lineage.py",
                "triad_rl/train_anchored_pilot.py")
CREDIT_FIELDS = ("return_to_go", "baseline", "credit_baseline", "reference_raw_advantage",
                 "reference_normalized_advantage", "raw_advantage", "normalized_advantage")
EPISODE_FIELDS = ("episode_return", "success", "detection_rate", "coverage", "breach_rate",
                  "total_cost", "invalid_actions", "steps")


def source_provenance():
    return {name: shared._file_hash(shared.PYTHON_ROOT / name) for name in SOURCE_FILES}


def load_protocol(path):
    """Validate only the training contract and its predeclared implementation."""
    with Path(path).open("rb") as handle:
        raw = handle.read(4_000_001)
    if len(raw) > 4_000_000:
        raise ValueError("Protocol exceeds bounded size")
    protocol = shared._strict_json(raw)
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("Unsupported anchored protocol schema")
    arms = protocol.get("arms")
    if (not isinstance(arms, dict) or set(arms) != set(ARMS)
            or any(not isinstance(arms[name], dict) or set(arms[name]) != set(spec)
                   or arms[name].get("baseline") != spec["baseline"]
                   or type(arms[name].get("value_coef")) not in (int, float)
                   or arms[name]["value_coef"] != spec["value_coef"] for name, spec in ARMS.items())):
        raise ValueError("Protocol must declare all three exact anchored arms")
    train = protocol.get("training", {})
    required = {"policy_seed", "scenario_seed_start", "episodes", "batch_size", "learning_rate",
                "entropy_coef", "gamma", "max_grad_norm", "profile"}
    descriptions = {"initialization", "endpoint", "advantage_normalization", "experience_pairing",
                    "normalization", "anchored_baseline", "rationale"}
    if (not isinstance(train, dict) or not required <= set(train) <= required | descriptions
            or any(not isinstance(train[key], str) or not train[key] for key in set(train) & descriptions)):
        raise ValueError("Anchored training fields differ")
    for key, low, high in (("policy_seed", 0, 999999), ("scenario_seed_start", 0, 10**15 - 4096),
                           ("episodes", 1, 4096), ("batch_size", 1, 16)):
        shared._integer(train[key], key, low, high)
    if train["episodes"] % train["batch_size"] or train["scenario_seed_start"] != train["policy_seed"] * 1_000_000:
        raise ValueError("Anchored pilot requires complete batches in its declared training seed slot")
    for key, value in (("learning_rate", .002), ("entropy_coef", .015), ("gamma", 1.), ("max_grad_norm", 1.)):
        if type(train[key]) not in (int, float) or train[key] != value:
            raise ValueError(f"Anchored pilot fixes {key}={value}")
    if train["profile"] != "mixed" or protocol.get("training_implementation_sha256") != source_provenance():
        raise ValueError("Training profile or predeclared source hashes differ")
    reference = protocol.get("frozen_reference", {})
    for key, value in (("checkpoint", "Checkpoints/balanced-v3-candidate"),
                       ("selection_report", "Results/balanced-v3/selection.json"),
                       ("publication_manifest", "Results/balanced-v3/artifact-manifest.json")):
        if reference.get(key) != value:
            raise ValueError("Anchored pilot must copy the frozen published v3 reference")
    return protocol, raw


def _config(protocol, raw, arm, initialization, lineage, runtime):
    declared_prior = protocol.get("prior_pilot_publication", {})
    inherited_prior = lineage["prior_pilot"]["publication_manifest"]
    if (declared_prior.get("manifest") != inherited_prior["path"]
            or declared_prior.get("sha256") != inherited_prior["sha256"]):
        raise ValueError("Declared prior pilot differs from pinned inherited publication")
    return {"schema": TRAINING_SCHEMA, "arm": arm, "training": protocol["training"],
            "arm_specification": ARMS[arm], "policy_schema": POLICY_SCHEMA,
            "protocol_sha256": hashlib.sha256(raw).hexdigest(), "source_sha256": source_provenance(),
            "initialization": initialization, "inherited_lineage": lineage, "curriculum": curriculum_manifest(),
            "checkpoint_rule": "fixed_endpoint_only", "runtime": runtime,
            "credit_contract": "Controls use unchanged critic-normalized coefficients. Anchored arm preserves exact reference first coefficients and modifies only later coefficients; no first-decision branch.",
            "rng_pairing": "Same initial actor RNG; later episode trajectories and draw counts can diverge."}


def _state(config, initialized_files, completed, log_hash):
    return {"schema": TRAINING_SCHEMA, "config": config, "config_sha256": shared._hash(config),
            "protocol_sha256": config["protocol_sha256"], "source_sha256": config["source_sha256"],
            "arm": config["arm"], "completed_episodes": completed,
            "target_episodes": config["training"]["episodes"], "initialized_files_sha256": initialized_files,
            "log_sha256": log_hash,
            "seed_provenance": {"training": {"start": config["training"]["scenario_seed_start"], "count": completed},
                                "inherited": config["inherited_lineage"], "validation": "never accessed by trainer",
                                "final_test": "never accessed by trainer"}}


def _stable(protocol_path, raw, config):
    reference = shared._strict_json(raw)["frozen_reference"]
    initial = config["initialization"]
    if (Path(protocol_path).read_bytes() != raw or source_provenance() != config["source_sha256"]
            or any(shared._file_hash(shared._relative(row["path"])) != row["sha256"]
                   for row in config["inherited_lineage"]["publication_manifests"])
            or shared._checkpoint_files(shared._relative(reference["checkpoint"])) != initial["checkpoint_files_sha256"]
            or shared._file_hash(shared._relative(reference["selection_report"])) != initial["selection_report_sha256"]
            or shared._file_hash(shared._relative(reference["publication_manifest"])) != initial["publication_manifest_sha256"]):
        raise ValueError("Anchored source/protocol/reference changed; no further output permitted")


def _prepare(policy, batch, arm):
    transitions, returns, reference, _ = shared._credit_arrays(batch)
    ordinary = np.asarray([row["value"] for row in transitions])
    first = np.asarray([index == 0 for episode in batch for index in range(len(episode))])
    reference_raw = returns - ordinary
    if arm == "anchored_later_stop":
        prepared = anchored_credit.prepare_batch(policy, batch, gamma=1.)
        for key, expected in (("returns", returns), ("reference_raw_advantages", reference_raw),
                              ("reference_advantages", reference), ("first_mask", first)):
            if not np.array_equal(prepared[key], expected):
                raise ValueError("Anchored helper differs from the frozen reference preparation")
        applied, credit_raw = prepared["advantages"], prepared["credit_raw_advantages"]
        credit_baseline = np.asarray([row.get("credit_baseline", row["value"]) for row in transitions])
    else:
        applied, credit_raw, credit_baseline = reference.copy(), reference_raw.copy(), ordinary.copy()
    if np.asarray(applied[first], dtype="<f8").tobytes() != np.asarray(reference[first], dtype="<f8").tobytes():
        raise ValueError("First-decision coefficients changed")
    stop = np.asarray([bool(row["features"][row["action"], shared.FEATURE_NAMES.index("stop")]) for row in transitions])
    matrix = np.column_stack((returns, ordinary, credit_baseline, reference_raw, reference, credit_raw, applied))
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite anchored credit diagnostics")
    masks = {"all": np.ones(len(first), dtype=bool), "first": first, "later": ~first,
             "stop": stop, "deploy": ~stop, "first_stop": first & stop, "later_stop": ~first & stop,
             "first_deploy": first & ~stop, "later_deploy": ~first & ~stop}
    digest = lambda array: hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()
    proof = {"first_coefficients_identical": True, "first_coefficients_count": int(first.sum()),
             "reference_first_coefficients_sha256": digest(reference[first]),
             "applied_first_coefficients_sha256": digest(applied[first]),
             "reference_coefficients_sha256": digest(reference)}
    return transitions, returns, applied, {name: matrix[mask] for name, mask in masks.items()}, proof


def _credit_summary(groups):
    return {group: {field: shared._distribution(values[:, index]) for index, field in enumerate(CREDIT_FIELDS)}
            for group, values in groups.items()}


def _head_check(policy, initial):
    if any(getattr(policy, group)[key].tobytes() != getattr(initial, group)[key].tobytes()
           for group in ("parameters", "adam_m", "adam_v") for key in ("wv", "bv")):
        raise ValueError("Zero-value-gradient arm changed critic head or Adam moments")


def run_training(protocol_path, arm, output):
    started = time.perf_counter()
    output, protocol_path = Path(output).resolve(), Path(protocol_path).resolve()
    shared._empty_output(output)
    if arm not in ARMS:
        raise ValueError("Unknown anchored arm")
    protocol, raw = load_protocol(protocol_path)
    train, spec = protocol["training"], ARMS[arm]
    lineage = load_published_lineage()
    assert_disjoint(train["scenario_seed_start"], train["episodes"], lineage, label="anchored pilot training")
    policy, initialization = shared._initialize(protocol["frozen_reference"], train["policy_seed"], lineage)
    config = _config(protocol, raw, arm, initialization, lineage,
                     {"python": platform.python_version(), "numpy": np.__version__})
    _stable(protocol_path, raw, config)
    shared._empty_output(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_bytes(raw)
    (output / "config.json").write_text(shared._json({**config, "config_sha256": shared._hash(config)}) + "\n", encoding="utf-8")
    log = output / "training.jsonl"
    def event(value):
        with log.open("a", encoding="utf-8") as handle:
            handle.write(shared._json(value) + "\n")
    event({"event": "initialization", "episode": 0, "arm": arm, "weights_sha256": policy.weights_fingerprint(),
           "policy_rng_sha256": initialization["sampling_rng_sha256"]})
    policy.save(output / "initialized", _state(config, {}, 0, shared._file_hash(log)))
    initial_files = shared._checkpoint_files(output / "initialized")
    initial_policy = BalancedPolicy.load(output / "initialized")
    env = RobustPlacementEnv(seed=train["scenario_seed_start"], profile="mixed")
    completed = branches = decisions = 0
    totals = {"rollout_wall_seconds": 0., "gradient_audit_wall_seconds": 0., "update_wall_seconds": 0.}
    accumulated = {name: [] for name in shared.GROUPS}
    records_all, first_trajectory, first_proof = [], None, None
    training_started = time.perf_counter()
    while completed < train["episodes"]:
        batch, records = [], []
        step_started = time.perf_counter()
        for index in range(completed, completed + train["batch_size"]):
            seed = train["scenario_seed_start"] + index
            if arm == "anchored_later_stop":
                episode, record = anchored_credit.rollout(env, policy, seed, gamma=1.)
                if record["anchored_credit"]["branch_rollouts"] != len(episode) - 1:
                    raise ValueError("Anchored branch accounting must exclude the first decision")
            else:
                episode, record = rollout(env, policy, seed, training=True)
            batch.append(episode)
            records.append(record)
        timings = {"rollout_wall_seconds": time.perf_counter() - step_started}
        step_started = time.perf_counter()
        transitions, returns, coefficients, credit, proof = _prepare(policy, batch, arm)
        trajectory = shared._trajectory_digest(batch, train["scenario_seed_start"] + completed)
        if completed == 0:
            first_trajectory, first_proof = trajectory, proof
        gradients = shared._gradient_audit(policy, transitions, returns, coefficients, train["entropy_coef"], spec["value_coef"])
        timings["gradient_audit_wall_seconds"] = time.perf_counter() - step_started
        before = {key: value.copy() for key, value in policy.parameters.items()}
        step_started = time.perf_counter()
        update = anchored_credit.update if arm == "anchored_later_stop" else lambda actor, episodes, **kw: actor.update(episodes, **kw)
        metrics = update(policy, batch, learning_rate=train["learning_rate"], entropy_coef=train["entropy_coef"],
                         gamma=1., value_coef=spec["value_coef"], max_grad_norm=train["max_grad_norm"])
        timings["update_wall_seconds"] = time.perf_counter() - step_started
        if metrics["gradient_norm"] != gradients["diagnostic_total_gradient_norm"]:
            raise ValueError("Gradient audit differs from the applied update")
        _stable(protocol_path, raw, config)
        begin, completed = completed, completed + train["batch_size"]
        extra = sum(record.get("anchored_credit", {}).get("branch_rollouts", 0) for record in records)
        branches += extra
        decisions += len(transitions)
        for key, value in timings.items():
            totals[key] += value
        for group, values in credit.items():
            accumulated[group].append(values)
        records_all.extend({key: record[key] for key in EPISODE_FIELDS} for record in records)
        delta = lambda keys: float(np.sqrt(sum(np.sum((policy.parameters[key] - before[key])**2) for key in keys)))
        row = {"event": "training_batch", "arm": arm, "episode_start": begin, "episode": completed,
               "scenario_seed_range": {"start": train["scenario_seed_start"] + begin, "count": train["batch_size"]},
               "trajectory_sha256": trajectory, "coefficient_identity": proof, "credit": _credit_summary(credit),
               **_summary(records), **metrics, **gradients, "weights_sha256": policy.weights_fingerprint(),
               "clipped_gradient_norm": metrics["gradient_norm"] * metrics["gradient_scale"],
               "parameter_delta_l2": delta(policy.parameters), "critic_head_delta_l2": delta(("wv", "bv")),
               "case_profile_counts": dict(Counter(record["case_metadata"]["profile"] for record in records)),
               "sensor_deployment_counts": dict(Counter(p["sensor_id"] for record in records for p in record["placements"])),
               "compute": {**timings, "live_episode_rollouts": len(records), "live_decisions": len(transitions),
                           "extra_branch_rollouts": extra, "cumulative_live_episode_rollouts": completed,
                           "cumulative_extra_branch_rollouts": branches,
                           "training_elapsed_seconds": time.perf_counter() - training_started}}
        event(row)
        print(shared._json({"arm": arm, "episode": completed, "mean_return": row["mean_return"], "extra_branch_rollouts": branches}), flush=True)
    if load_published_lineage() != lineage:
        raise ValueError("Inherited evidence changed during anchored training")
    if not spec["value_coef"]:
        _head_check(policy, initial_policy)
    if shared._checkpoint_files(output / "initialized") != initial_files:
        raise ValueError("Immutable initialized checkpoint changed")
    _stable(protocol_path, raw, config)
    state = _state(config, initial_files, completed, shared._file_hash(log))
    policy.save(output / "last", state)
    summary = {"schema": TRAINING_SCHEMA, "arm": arm, "completed_episodes": completed, "optimizer_updates": policy.update_count,
               "checkpoint_rule": "fixed_endpoint_only", "initial_weights_sha256": initialization["weights_sha256"],
               "last_weights_sha256": policy.weights_fingerprint(), "first_batch_trajectory_sha256": first_trajectory,
               "first_batch_coefficient_identity": first_proof, "all_first_coefficients_identical": True,
               "initialized_files_sha256": initial_files, "last_files_sha256": shared._checkpoint_files(output / "last"),
               "config_sha256": shared._hash(config), "protocol_sha256": config["protocol_sha256"],
               "source_sha256": config["source_sha256"], "log_sha256": state["log_sha256"], "seed_provenance": state["seed_provenance"],
               "credit": _credit_summary({key: np.concatenate(value, axis=0) for key, value in accumulated.items()}),
               **_summary(records_all), "compute": {**totals, "live_episode_rollouts": completed, "live_decisions": decisions,
                   "extra_branch_rollouts": branches, "training_elapsed_seconds": time.perf_counter() - training_started,
                   "total_elapsed_seconds": time.perf_counter() - started,
                   "interpretation": "Extra terminal branches occur only after the first decision; equal live episodes are not equal compute."}}
    (output / "summary.json").write_text(shared._json(summary) + "\n", encoding="utf-8")
    return summary


def _same(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return bool(np.isfinite(left) and np.isfinite(right) and np.isclose(left, right, rtol=1e-12, atol=1e-12))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _pool(rows):
    count = sum(row["count"] for row in rows)
    if not count:
        return shared._distribution([])
    mean = sum(row["count"] * row["mean"] for row in rows if row["count"]) / count
    variance = sum(row["count"] * (row["variance"] + (row["mean"] - mean)**2)
                   for row in rows if row["count"]) / count
    return {"count": count, "mean": mean, "variance": variance,
            **{key: sum(row[key] for row in rows) for key in ("negative", "zero", "positive")}}


def validate_run(run_dir, protocol_path, arm, lineage):
    """Read-only complete endpoint validation; no inference or scenario sampling."""
    protocol, raw = load_protocol(protocol_path)
    if arm not in ARMS:
        raise ValueError("Unknown anchored arm")
    run = Path(run_dir).resolve()
    targets = [(run / name).resolve() for name in RUN_FILES]
    if (len(set(targets)) != len(targets) or any(not path.is_relative_to(run) for path in targets)
            or {path.relative_to(run).as_posix() for path in run.rglob("*") if path.is_file()} != set(RUN_FILES)):
        raise ValueError("Run must contain exactly eight independent in-directory artifacts")
    blobs = {}
    for name, path in zip(RUN_FILES, targets):
        with path.open("rb") as handle:
            blobs[name] = handle.read(16_000_001)
        if len(blobs[name]) > 16_000_000:
            raise ValueError("Run artifact exceeds bounded size")
    if blobs["protocol.json"] != raw:
        raise ValueError("Run protocol bytes differ from the declaration")
    config_file, summary = (shared._strict_json(blobs[name]) for name in ("config.json", "summary.json"))
    for name, value in (("config.json", config_file), ("summary.json", summary)):
        canonical = shared._json(value).encode()
        if blobs[name] not in (canonical + b"\n", canonical + b"\r\n"):
            raise ValueError("Run config/summary serialization changed")
    train = protocol["training"]
    count, batch_size = train["episodes"], train["batch_size"]
    assert_disjoint(train["scenario_seed_start"], count, lineage, label="anchored run exposure")
    expected_initial, initialization = shared._initialize(protocol["frozen_reference"], train["policy_seed"], lineage)
    runtime = config_file.get("runtime", {})
    if set(runtime) != {"python", "numpy"} or any(not isinstance(value, str) or not value for value in runtime.values()):
        raise ValueError("Recorded runtime is malformed")
    config = _config(protocol, raw, arm, initialization, lineage, runtime)
    if shared._json(config_file) != shared._json({**config, "config_sha256": shared._hash(config)}):
        raise ValueError("Run configuration, sources or initialization differs")
    initial, endpoint = (BalancedPolicy.load(run / name, feature_names=shared.FEATURE_NAMES) for name in ("initialized", "last"))
    for group in ("parameters", "adam_m", "adam_v"):
        if any(getattr(initial, group)[key].tobytes() != value.tobytes() for key, value in getattr(expected_initial, group).items()):
            raise ValueError("Initialized arrays differ from exact frozen v3 and zero Adam")
    if (initial.update_count != 0 or shared._json(initial.rng.bit_generator.state) != shared._json(expected_initial.rng.bit_generator.state)
            or endpoint.update_count != count // batch_size):
        raise ValueError("Initializer RNG or endpoint optimizer count differs")
    if not ARMS[arm]["value_coef"]:
        _head_check(endpoint, initial)
    files = {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()}
    initial_files = {name: files[f"initialized/{name}"] for name in ("checkpoint.json", "arrays.npz")}
    last_files = {name: files[f"last/{name}"] for name in ("checkpoint.json", "arrays.npz")}
    lines = blobs["training.jsonl"].splitlines(keepends=True)
    events = [shared._strict_json(line) for line in lines]
    expected_event = {"event": "initialization", "episode": 0, "arm": arm,
                      "weights_sha256": initial.weights_fingerprint(), "policy_rng_sha256": initialization["sampling_rng_sha256"]}
    if len(events) != 1 + count // batch_size or shared._json(events[0]) != shared._json(expected_event):
        raise ValueError("Run initialization event or complete batch count differs")
    initial_state = _state(config, {}, 0, hashlib.sha256(lines[0]).hexdigest())
    last_state = _state(config, initial_files, count, files["training.jsonl"])
    for actor, expected in ((initial, initial_state), (endpoint, last_state)):
        if (shared._json(actor.training_state) != shared._json(expected)
                or shared._json(actor.metadata["seed_provenance"]) != shared._json(expected["seed_provenance"])):
            raise ValueError("Checkpoint training state, log or exposure differs")
    batches, branches, decisions = events[1:], 0, 0
    def integer(value, expected=None):
        if type(value) is not int or value < 0 or (expected is not None and value != expected):
            raise ValueError("Run counts or declared episode cadence differs")
    def digest(value):
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("Run digest is malformed")
    for index, event in enumerate(batches):
        start = index * batch_size
        if event.get("event") != "training_batch" or event.get("arm") != arm:
            raise ValueError("Unexpected training log event or arm")
        integer(event["episode_start"], start)
        integer(event["episode"], start + batch_size)
        integer(event["updates"], index + 1)
        if shared._json(event["scenario_seed_range"]) != shared._json({"start": train["scenario_seed_start"] + start, "count": batch_size}):
            raise ValueError("Run does not cover the exact training range")
        n = event["transitions"]
        integer(n)
        if not batch_size <= n <= 32 * batch_size:
            raise ValueError("Run decision horizon differs")
        digest(event["trajectory_sha256"])
        digest(event["weights_sha256"])
        proof = event["coefficient_identity"]
        if set(proof) != {"first_coefficients_identical", "first_coefficients_count", "reference_first_coefficients_sha256", "applied_first_coefficients_sha256", "reference_coefficients_sha256"}:
            raise ValueError("Incomplete coefficient identity evidence")
        integer(proof["first_coefficients_count"], batch_size)
        for key in ("reference_first_coefficients_sha256", "applied_first_coefficients_sha256", "reference_coefficients_sha256"):
            digest(proof[key])
        if proof["first_coefficients_identical"] is not True or proof["reference_first_coefficients_sha256"] != proof["applied_first_coefficients_sha256"]:
            raise ValueError("First coefficient identity is not preserved")
        credit = event["credit"]
        if set(credit) != set(shared.GROUPS):
            raise ValueError("Credit decision groups incomplete")
        group_counts = {}
        for group, fields in credit.items():
            if set(fields) != set(CREDIT_FIELDS):
                raise ValueError("Credit statistics fields incomplete")
            group_counts[group] = fields["return_to_go"]["count"]
            for stats in fields.values():
                if set(stats) != {"count", "mean", "variance", "negative", "zero", "positive"}:
                    raise ValueError("Credit moments schema differs")
                integer(stats["count"], group_counts[group])
                for key in ("negative", "zero", "positive"):
                    integer(stats[key])
                if sum(stats[key] for key in ("negative", "zero", "positive")) != stats["count"]:
                    raise ValueError("Credit sign counts differ")
                if stats["count"]:
                    if any(type(stats[key]) is not float or not np.isfinite(stats[key]) for key in ("mean", "variance")) or stats["variance"] < 0:
                        raise ValueError("Credit moments must be finite")
                elif stats["mean"] is not None or stats["variance"] is not None:
                    raise ValueError("Empty credit moments must be null")
            if group.startswith("first") or arm != "anchored_later_stop":
                for a, b in (("baseline", "credit_baseline"), ("reference_raw_advantage", "raw_advantage"), ("reference_normalized_advantage", "normalized_advantage")):
                    if shared._json(fields[a]) != shared._json(fields[b]):
                        raise ValueError("Reference/applied credit identity differs")
        if (group_counts["all"] != n or group_counts["first"] != batch_size or group_counts["later"] != n - batch_size
                or any(group_counts[a] + group_counts[b] != group_counts[target] for a, b, target in (
                    ("stop", "deploy", "all"), ("first_stop", "first_deploy", "first"),
                    ("later_stop", "later_deploy", "later"), ("first_stop", "later_stop", "stop"),
                    ("first_deploy", "later_deploy", "deploy")))):
            raise ValueError("Credit group counts do not partition live decisions")
        compute = event["compute"]
        extra = n - batch_size if arm == "anchored_later_stop" else 0
        branches += extra
        decisions += n
        for key, value in (("live_episode_rollouts", batch_size), ("live_decisions", n), ("extra_branch_rollouts", extra),
                           ("cumulative_live_episode_rollouts", start + batch_size), ("cumulative_extra_branch_rollouts", branches)):
            integer(compute[key], value)
        if any(type(compute[key]) is not float or not np.isfinite(compute[key]) or compute[key] < 0
               for key in ("rollout_wall_seconds", "gradient_audit_wall_seconds", "update_wall_seconds", "training_elapsed_seconds")):
            raise ValueError("Invalid measured compute time")
        if not event["case_profile_counts"] or set(event["case_profile_counts"]) - {"normal", "stress", "capability"}:
            raise ValueError("Invalid curriculum profile count")
        for value in (*event["case_profile_counts"].values(), *event["sensor_deployment_counts"].values()):
            integer(value)
        if sum(event["case_profile_counts"].values()) != batch_size or sum(event["sensor_deployment_counts"].values()) != group_counts["deploy"]:
            raise ValueError("Profile or sensor deployment totals differ")
        if (not _same(event["gradient_norm"], event["diagnostic_total_gradient_norm"])
                or not _same(event["gradient_scale"], min(1., train["max_grad_norm"] / max(event["gradient_norm"], 1e-12)))
                or not _same(event["clipped_gradient_norm"], event["gradient_norm"] * event["gradient_scale"])):
            raise ValueError("Applied gradient norm or clip scale differs")
        if not ARMS[arm]["value_coef"] and any(event[key] != 0 for key in ("weighted_critic_gradient_norm", "actor_critic_gradient_dot", "critic_head_delta_l2")):
            raise ValueError("Zero-critic arm logged a critic gradient or head update")
    expected_summary = {"schema": TRAINING_SCHEMA, "arm": arm, "completed_episodes": count,
                        "optimizer_updates": count // batch_size, "checkpoint_rule": "fixed_endpoint_only",
                        "initial_weights_sha256": initial.weights_fingerprint(), "last_weights_sha256": endpoint.weights_fingerprint(),
                        "first_batch_trajectory_sha256": batches[0]["trajectory_sha256"], "first_batch_coefficient_identity": batches[0]["coefficient_identity"],
                        "all_first_coefficients_identical": True, "initialized_files_sha256": initial_files,
                        "last_files_sha256": last_files, "config_sha256": shared._hash(config), "protocol_sha256": config["protocol_sha256"],
                        "source_sha256": config["source_sha256"], "log_sha256": files["training.jsonl"], "seed_provenance": last_state["seed_provenance"]}
    if (shared._json({key: summary.get(key) for key in expected_summary}) != shared._json(expected_summary)
            or batches[-1]["weights_sha256"] != endpoint.weights_fingerprint()):
        raise ValueError("Endpoint summary identities differ")
    pooled = {group: {field: _pool([row["credit"][group][field] for row in batches]) for field in CREDIT_FIELDS} for group in shared.GROUPS}
    if not _same(summary["credit"], pooled):
        raise ValueError("Summary credit moments differ from complete batch evidence")
    for key in _summary([{field: 0. for field in EPISODE_FIELDS}]):
        if not _same(summary[key], float(np.mean([row[key] for row in batches]))):
            raise ValueError("Summary sensing/cost metrics differ from complete batch evidence")
    for key, value in (("live_episode_rollouts", count), ("live_decisions", decisions), ("extra_branch_rollouts", branches)):
        integer(summary["compute"][key], value)
    for key in ("rollout_wall_seconds", "gradient_audit_wall_seconds", "update_wall_seconds"):
        if not _same(summary["compute"][key], sum(row["compute"][key] for row in batches)):
            raise ValueError("Summary compute totals differ")
    _stable(protocol_path, raw, config)
    if any((run / name).read_bytes() != blob for name, blob in blobs.items()):
        raise ValueError("Run evidence changed while validating")
    return endpoint, {"weights_sha256": endpoint.weights_fingerprint(), "run_files_sha256": files,
                      "first_batch_trajectory_sha256": summary["first_batch_trajectory_sha256"],
                      "first_batch_coefficient_identity": summary["first_batch_coefficient_identity"], "summary": summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", dest="protocol_path", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run_training(**vars(parser.parse_args(argv)))
    print(shared._json({key: result[key] for key in ("arm", "completed_episodes", "optimizer_updates", "last_weights_sha256")}))


if __name__ == "__main__":
    main()
