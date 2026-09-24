"""Fixed-endpoint, sensing-first development evaluation of the credit pilot.

Three endpoints and four frozen references receive paired validation cases.
The paired-STOP arm is primary; no best-arm/checkpoint selection occurs here.
Completed profile reports are persisted independently and may be explicitly
resumed only with identical protocol, implementations, checkpoints and lineage.
Interrupted writes fail closed and need manual recovery; existing evidence is
never silently replaced or discarded by this runner.
The scaling screen authorizes multi-seed replication, never policy promotion.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from evaluate_robust import (evaluate_robust_methods, validate_seed_range,
                            implementation_fingerprints as robust_sources, V1_WEIGHTS)
from .adaptive_evaluation import GreedyPublicCoverage, canonical_hash, summarize
from .adaptive_inputs import FEATURE_NAMES
from .adaptive_policy import AdaptivePolicy
from .balanced_policy import BalancedPolicy
from .credit_lineage import load_published_lineage, assert_disjoint
from . import train_credit_pilot as trainer
from .train_adaptive import scenario_seed


BLUE_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = BLUE_ROOT / "Python"
SCHEMA = "triad.credit_pilot_evaluation.v1"
INPUT_SCHEMA = "triad.credit_pilot_evaluation_inputs.v1"
AGGREGATE_SCHEMA = "triad.credit_pilot_aggregate.v1"
PROFILES = ("normal", "stress", "capability")
ARMS = ("shared_critic", "no_critic_gradient", "paired_stop")
METHODS = (*ARMS, "balanced_v3", "robust_v2", "adaptive_v1", "greedy_public")
METRICS = ("timely_fraction", "all_threat_success", "detection_rate", "return", "cost")
V2_WEIGHTS = "5c982c86a5da2e5b4960ba437021f751a36dc3af7187515903cf1b1ee2ab27ce"
RUN_FILES = ("protocol.json", "config.json", "summary.json", "training.jsonl",
             "initialized/checkpoint.json", "initialized/arrays.npz", "last/checkpoint.json", "last/arrays.npz")
MAX_BYTES = 128 * 1024 * 1024


def _bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _same(left, right):
    return _bytes(left) == _bytes(right)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _file(path):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Pilot evidence exceeds bounded file size")
    with path.open("rb") as handle:
        data = handle.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("Pilot evidence exceeds bounded file size")
    return data


def _read(path):
    return trainer._strict_json(_file(path))


def _checkpoint_files(path):
    return {name: _sha(_file(path / name)) for name in ("checkpoint.json", "arrays.npz")}


def source_provenance():
    """Bind every reused evaluation, training and lineage implementation."""
    return {**robust_sources(), **trainer.source_provenance(),
            "triad_rl/evaluate_credit_pilot.py": _sha(Path(__file__).read_bytes()),
            "triad_rl/placement_policy.py": _sha(_file(PYTHON_ROOT / "triad_rl/placement_policy.py"))}


def _protocol(path):
    protocol, raw = trainer._protocol(path, "shared_critic")
    validation = protocol.get("validation", {})
    if (validation.get("stage") != "validation" or validation.get("methods") != list(METHODS)
            or type(validation.get("episodes_per_profile")) is not int
            or not 1 <= validation["episodes_per_profile"] <= 200
            or type(validation.get("bootstrap_samples")) is not int or validation["bootstrap_samples"] != 2000
            or type(validation.get("bootstrap_seed")) is not int or validation["bootstrap_seed"] != 73041
            or set(validation.get("scenario_seed_ranges", {})) != set(PROFILES)
            or set(validation.get("run_seeds", {})) != set(PROFILES)):
        raise ValueError("Pilot validation contract differs from the predeclared bounded design")
    ranges = validation["scenario_seed_ranges"]
    for profile in PROFILES:
        row = ranges[profile]
        trainer._integer(validation["run_seeds"][profile], "validation run seed", 0, 999_999)
        if not _same(row, {"start": scenario_seed("validation", validation["run_seeds"][profile], 0),
                           "count": validation["episodes_per_profile"]}):
            raise ValueError("Validation seed slots/counts differ from their run seeds")
        validate_seed_range(row["start"], row["count"], "validation")
        validate_seed_range(row["start"], row["count"], "validation",
                            {p: r for p, r in ranges.items() if p != profile})
    required_gate = {"primary_arm": "paired_stop", "comparators": ["shared_critic", "balanced_v3"],
                     "primary_metric": "timely_confirmed_fraction = 1 - breached_fraction",
                     "minimum_equal_profile_gain": .01, "require_paired_ci_lower_above_zero": True,
                     "maximum_profile_point_decline": .01,
                     "require_equal_profile_all_threat_success_not_lower": True}
    gate = protocol.get("scale_gate", {})
    if not _same({key: gate.get(key) for key in required_gate}, required_gate):
        raise ValueError("Primary sensing-first scaling gate differs from its fixed contract")
    if not _same(protocol.get("evaluation_implementation_sha256"), source_provenance()):
        raise ValueError("Predeclared evaluation implementation differs")
    return protocol, raw


def _endpoint(run, arm, protocol, protocol_bytes, lineage):
    """Bind a complete fixed endpoint, its zero-Adam initialization and full log."""
    run = Path(run)
    blobs = {name: _file(run / name) for name in RUN_FILES}
    if blobs["protocol.json"] != protocol_bytes or (run / "best").exists():
        raise ValueError("Pilot run must use the exact protocol and no selected best checkpoint")
    config = trainer._strict_json(blobs["config.json"])
    summary = trainer._strict_json(blobs["summary.json"])
    if any(blobs[name] not in (_bytes(value), _bytes(value)[:-1] + b"\r\n")
           for name, value in (("config.json", config), ("summary.json", summary))):
        raise ValueError("Endpoint config/summary must retain exact canonical trainer serialization")
    last, initial = BalancedPolicy.load(run / "last", feature_names=FEATURE_NAMES), BalancedPolicy.load(run / "initialized", feature_names=FEATURE_NAMES)
    state = last.training_state
    core_config = {key: value for key, value in config.items() if key != "config_sha256"}
    config_hash, protocol_hash = canonical_hash(core_config), _sha(protocol_bytes)
    train = protocol["training"]
    count, batch = train["episodes"], train["batch_size"]
    training_range = {"start": train["scenario_seed_start"], "count": count}
    expected_provenance = {"training": training_range, "inherited": lineage,
                           "validation": "never accessed by trainer", "final_test": "never accessed by trainer"}
    if (config.get("schema") != trainer.TRAINING_SCHEMA or summary.get("schema") != trainer.TRAINING_SCHEMA
            or state.get("schema") != trainer.TRAINING_SCHEMA
            or any(value.get("arm") != arm for value in (config, summary, state))
            or not _same(config.get("training"), train)
            or not _same(config.get("arm_specification"), trainer.ARMS[arm])
            or config.get("checkpoint_rule") != "fixed_endpoint_only"
            or summary.get("checkpoint_rule") != "fixed_endpoint_only"
            or not _same(state.get("config"), core_config)
            or not _same(initial.training_state.get("config"), core_config)
            or not _same(config.get("inherited_lineage"), lineage)
            or not _same(state.get("seed_provenance"), expected_provenance)
            or not _same(summary.get("seed_provenance"), expected_provenance)):
        raise ValueError("Endpoint schema/configuration/arm/exposure differs from the complete pilot")
    for value in (config, summary, state, initial.training_state):
        if (value.get("config_sha256") != config_hash or value.get("protocol_sha256") != protocol_hash
                or not _same(value.get("source_sha256"), trainer.source_provenance())):
            raise ValueError("Endpoint source, configuration or protocol fingerprint differs")
    if (type(state.get("completed_episodes")) is not int or state["completed_episodes"] != count
            or type(summary.get("completed_episodes")) is not int or summary["completed_episodes"] != count
            or not _same(state.get("target_episodes"), count) or last.update_count != count // batch
            or not _same(summary.get("optimizer_updates"), count // batch)
            or initial.update_count != 0 or not _same(initial.training_state.get("completed_episodes"), 0)):
        raise ValueError("Endpoint is not the declared complete optimizer state")
    last_files, initial_files = _checkpoint_files(run / "last"), _checkpoint_files(run / "initialized")
    if (summary.get("last_weights_sha256") != last.weights_fingerprint()
            or summary.get("initial_weights_sha256") != initial.weights_fingerprint()
            or not _same(summary.get("last_files_sha256"), last_files)
            or not _same(summary.get("initialized_files_sha256"), initial_files)
            or not _same(state.get("initialized_files_sha256"), initial_files)
            or summary.get("log_sha256") != _sha(blobs["training.jsonl"])
            or state.get("log_sha256") != _sha(blobs["training.jsonl"])):
        raise ValueError("Endpoint checkpoint or training-log bytes differ")
    base_path = BLUE_ROOT / "Checkpoints/balanced-v3-candidate"
    base = BalancedPolicy.load(base_path, feature_names=FEATURE_NAMES)
    initialization = config.get("initialization", {})
    expected_rng = canonical_hash(np.random.default_rng(train["policy_seed"]).bit_generator.state)
    _, expected_initialization = trainer._initialize(protocol["frozen_reference"], train["policy_seed"], lineage)
    if (base.weights_fingerprint() != protocol["frozen_reference"]["weights_sha256"]
            or initial.weights_fingerprint() != base.weights_fingerprint()
            or any(not np.array_equal(initial.parameters[key], base.parameters[key]) for key in base.parameters)
            or any(np.any(value) for collection in (initial.adam_m, initial.adam_v) for value in collection.values())
            or canonical_hash(initial.rng.bit_generator.state) != expected_rng
            or initialization.get("sampling_rng_sha256") != expected_rng
            or initialization.get("inherited_lineage_sha256") != canonical_hash(lineage)
            or not _same(initialization.get("checkpoint_files_sha256"), _checkpoint_files(base_path))
            or initialization.get("weights_sha256") != base.weights_fingerprint()
            or not _same(initialization, expected_initialization)):
        raise ValueError("Pilot initialization differs from copied v3 parameters, zero Adam or matched RNG")
    if arm != "shared_critic" and any(not np.array_equal(last.parameters[key], initial.parameters[key])
                                       or np.any(last.adam_m[key]) or np.any(last.adam_v[key]) for key in ("wv", "bv")):
        raise ValueError("Actor-only endpoint changed its critic head/optimizer")
    events = [trainer._strict_json(line) for line in blobs["training.jsonl"].splitlines()]
    batches = events[1:]
    if (len(events) != 1 + count // batch or events[0].get("event") != "initialization"
            or not _same(events[0].get("episode"), 0) or events[0].get("arm") != arm
            or events[0].get("weights_sha256") != initial.weights_fingerprint()
            or events[0].get("policy_rng_sha256") != expected_rng):
        raise ValueError("Training log initialization or batch count differs")
    for index, event in enumerate(batches):
        start = index * batch
        if (event.get("event") != "training_batch" or event.get("arm") != arm
                or not _same(event.get("episode_start"), start) or not _same(event.get("episode"), start + batch)
                or not _same(event.get("scenario_seed_range"), {"start": train["scenario_seed_start"] + start, "count": batch})):
            raise ValueError("Training log does not cover every declared episode exactly once")
    first_trajectory = batches[0].get("trajectory_sha256")
    if (not isinstance(first_trajectory, str) or len(first_trajectory) != 64
            or any(char not in "0123456789abcdef" for char in first_trajectory)
            or summary.get("first_batch_trajectory_sha256") != first_trajectory
            or batches[-1].get("weights_sha256") != last.weights_fingerprint()):
        raise ValueError("First-batch trajectory or final training fingerprint differs")
    return last, {"weights_sha256": last.weights_fingerprint(), "metadata": deepcopy(last.metadata),
                  "run_files_sha256": {name: _sha(raw) for name, raw in blobs.items()},
                  "first_batch_trajectory_sha256": first_trajectory,
                  "summary": summary}


def _vectors(report, method):
    values = {name: [] for name in METRICS}
    for row in report["methods"][method]["episodes"]:
        metrics = row["metrics"]
        if any(type(metrics[key]) not in (int, float) for key in ("breached_fraction", "detected_fraction", "return", "cost")):
            raise ValueError("Sensing fractions, return and cost must be numeric, not boolean")
        numbers = {"timely_fraction": 1. - metrics["breached_fraction"],
                   "all_threat_success": metrics["success"], "detection_rate": metrics["detected_fraction"],
                   "return": metrics["return"], "cost": metrics["cost"]}
        if type(metrics["success"]) is not bool:
            raise ValueError("All-threat success must be a boolean outcome")
        for key, value in numbers.items():
            if ((key != "all_threat_success" and type(value) not in (int, float))
                    or not np.isfinite(value)):
                raise ValueError("Nonfinite/missing paired sensing metric")
            if key in ("timely_fraction", "all_threat_success", "detection_rate") and not 0 <= value <= 1:
                raise ValueError("Sensing fractions must be in [0,1]")
            values[key].append(float(value))
    return {key: np.asarray(rows) for key, rows in values.items()}


def stratified_interval(deltas, *, samples=2000, seed=73041):
    """Paired percentile95 interval with fixed equal weights across profiles."""
    if set(deltas) != set(PROFILES) or type(samples) is not int or not 1 <= samples <= 10000:
        raise ValueError("Need three nonempty profile differences and bounded bootstrap count")
    arrays = {profile: np.asarray(deltas[profile], dtype=float) for profile in PROFILES}
    if any(value.ndim != 1 or not len(value) or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Paired profile differences must be finite nonempty vectors")
    rng = np.random.default_rng(seed)
    bootstraps = np.zeros(samples)
    for profile in PROFILES:
        vector = arrays[profile]
        for start in range(0, samples, 128):
            count = min(128, samples - start)
            indices = rng.integers(0, len(vector), size=(count, len(vector)))
            bootstraps[start:start + count] += vector[indices].mean(axis=1) / 3.
    return {"difference": float(np.mean([array.mean() for array in arrays.values()])),
            "lower95": float(np.quantile(bootstraps, .025)), "upper95": float(np.quantile(bootstraps, .975)),
            "profile_differences": {profile: float(value.mean()) for profile, value in arrays.items()},
            "pairs_per_profile": {profile: len(value) for profile, value in arrays.items()},
            "bootstrap_samples": samples, "bootstrap_seed": seed,
            "scope": "paired scenario uncertainty, conditional on these trained endpoints; not training-seed uncertainty"}


def _layout_groups(reports):
    result = {}
    for reference in ("shared_critic", "no_critic_gradient", "balanced_v3"):
        profiles = {}
        for profile, report in reports.items():
            buckets = {}
            for primary, other in zip(report["methods"]["paired_stop"]["episodes"], report["methods"][reference]["episodes"], strict=True):
                a, b = primary.get("placements", []), other.get("placements", [])
                if _same(a, b):
                    group = "unchanged_layout"
                elif bool(a) != bool(b):
                    group = "initial_stop_flip"
                elif a and b and _same(a[0], b[0]):
                    group = "same_first_placement_changed_later"
                else:
                    group = "changed_first_placement"
                metrics = buckets.setdefault(group, [])
                x, y = primary["metrics"], other["metrics"]
                metrics.append({"timely_fraction": y["breached_fraction"] - x["breached_fraction"],
                                "all_threat_success": float(x["success"]) - float(y["success"]),
                                "cost": x["cost"] - y["cost"]})
            profiles[profile] = {group: {"pairs": len(rows), "mean_differences": {
                key: float(np.mean([row[key] for row in rows])) for key in rows[0]}} for group, rows in buckets.items()}
        result[f"paired_stop_minus_{reference}"] = profiles
    return {"scope": "exploratory post-treatment groups; descriptive, not causal estimates or scaling criteria", "comparisons": result}


def aggregate_reports(reports, protocol):
    """Recompute fixed-primary sensing gate; never choose whichever arm won."""
    if set(reports) != set(PROFILES) or any(set(report.get("methods", {})) != set(METHODS) for report in reports.values()):
        raise ValueError("Aggregation requires all profiles and all seven paired methods")
    if any(report.get("stage") != "validation" or report.get("profile") != profile for profile, report in reports.items()):
        raise ValueError("Only profile-matched validation reports may enter pilot aggregation")
    vectors = {profile: {method: _vectors(report, method) for method in METHODS} for profile, report in reports.items()}
    for profile, report in reports.items():
        reference = report["methods"][METHODS[0]]["episodes"]
        if not reference:
            raise ValueError("Aggregation requires nonempty paired episodes")
        identities = [(row["seed"], row["scenario_sha256"]) for row in reference]
        if any(not _same([(row["seed"], row["scenario_sha256"]) for row in report["methods"][method]["episodes"]], identities) for method in METHODS):
            raise ValueError("Aggregation requires identical ordered case pairing")
    methods = {method: {"profiles": {profile: {key: float(vector.mean()) for key, vector in vectors[profile][method].items()} for profile in PROFILES},
                        "equal_profile": {key: float(np.mean([vectors[p][method][key].mean() for p in PROFILES])) for key in METRICS}} for method in METHODS}
    pairs = [(arm, reference) for arm in ARMS for reference in ("balanced_v3", "robust_v2", "adaptive_v1", "greedy_public")]
    pairs += [("paired_stop", "shared_critic"), ("paired_stop", "no_critic_gradient")]
    comparisons = {}
    validation = protocol["validation"]
    for arm, reference in pairs:
        comparisons[f"{arm}_minus_{reference}"] = {key: stratified_interval(
            {profile: vectors[profile][arm][key] - vectors[profile][reference][key] for profile in PROFILES},
            samples=validation["bootstrap_samples"], seed=validation["bootstrap_seed"]) for key in METRICS}
    spec, conditions = protocol["scale_gate"], {}
    for reference in spec["comparators"]:
        row = comparisons[f"paired_stop_minus_{reference}"]
        timely, success = row["timely_fraction"], row["all_threat_success"]
        checks = {"minimum_timely_gain": timely["difference"] >= spec["minimum_equal_profile_gain"],
                  "timely_ci_lower_above_zero": timely["lower95"] > 0.,
                  "no_profile_timely_decline_over_margin": all(value >= -spec["maximum_profile_point_decline"] for value in timely["profile_differences"].values()),
                  "all_threat_success_not_lower": success["difference"] >= 0.}
        conditions[reference] = {"passed": all(checks.values()), "checks": checks}
    passed = all(row["passed"] for row in conditions.values())
    return {"schema": AGGREGATE_SCHEMA, "stage": "validation", "final_test_accessed": False,
            "methods": methods, "paired_stratified_differences": comparisons,
            "scale_gate": {"primary_arm": "paired_stop", "passed": passed, "comparators": conditions,
                           "decision": "eligible_for_multiseed_replication" if passed else "do_not_scale_from_this_pilot",
                           "not_policy_promotion": True, "point_guards_are_not_noninferiority_tests": True},
            "exploratory_layout_groups": _layout_groups(reports)}


def _validate_report(report, profile, inputs, weights):
    expected_range = inputs["validation_ranges"][profile]
    if (report.get("schema") != SCHEMA or report.get("profile") != profile or report.get("stage") != "validation"
            or report.get("training_performed") is not False
            or report.get("protocol", {}).get("independent_final_test_evidence") is not False
            or set(report.get("methods", {})) != set(METHODS)
            or not _same(report.get("seed_provenance", {}).get("evaluation"), expected_range)
            or not _same(report.get("seed_provenance", {}).get("declared_lineage"), inputs["lineage"])
            or not _same(report.get("implementation_sha256"), robust_sources())
            or not _same(report.get("implementation_sha256_after"), robust_sources())
            or report.get("credit_pilot_inputs_sha256") != canonical_hash(inputs)):
        raise ValueError("Persisted profile report scope/implementation/input binding differs")
    identities = None
    seeds = list(range(expected_range["start"], expected_range["start"] + expected_range["count"]))
    for name, method in report["methods"].items():
        records = method["episodes"]
        if not _same([row["seed"] for row in records], seeds) or not _same(summarize(records), method["summary"]):
            raise ValueError("Profile report episodes or recomputed summary differ")
        hashes = [canonical_hash(row["scenario"]) for row in records]
        if hashes != [row["scenario_sha256"] for row in records] or (identities is not None and hashes != identities):
            raise ValueError("Profile report scenarios are not exactly paired")
        identities = hashes
        expected = weights.get(name)
        if any(row.get("weights_sha256_before") != expected or row.get("weights_sha256_after") != expected for row in records):
            raise ValueError("Profile report method weights changed")
        _vectors(report, name)


def _write_new(path, data):
    with path.open("xb") as handle:
        handle.write(data)


def _cached_report(destination, receipt_path, profile, inputs, weights):
    data, receipt = _file(destination), _read(receipt_path)
    if not _same(receipt, {"profile": profile, "inputs_sha256": canonical_hash(inputs), "sha256": _sha(data), "bytes": len(data)}):
        raise ValueError("Persisted profile compressed hash/input receipt differs")
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
        expanded = handle.read(MAX_BYTES + 1)
    if len(expanded) > MAX_BYTES:
        raise ValueError("Persisted profile exceeds expanded size bound")
    report = trainer._strict_json(expanded)
    _validate_report(report, profile, inputs, weights)
    return report, data


def run_evaluation(runs, *, protocol_path, output, resume=False):
    """Evaluate all fixed endpoints; explicitly resume verified complete profiles."""
    if set(runs) != set(ARMS):
        raise ValueError("Exactly the three named pilot endpoint runs are required")
    protocol_path, output = Path(protocol_path).resolve(), Path(output).resolve()
    protocol, protocol_bytes = _protocol(protocol_path)
    lineage = load_published_lineage()
    reservations = protocol.get("reserved_final_tests_unopened", {})
    if (set(reservations) != set(PROFILES)
            or not _same(sorted(reservations.values(), key=lambda row: row["start"]), lineage["reserved_final_seed_ranges"])):
        raise ValueError("Protocol final-test reservations differ from the published unopened slots")
    train = protocol["training"]
    exposure = {"published": lineage, "pilot_training": {"start": train["scenario_seed_start"], "count": train["episodes"]}}
    for profile, row in protocol["validation"]["scenario_seed_ranges"].items():
        assert_disjoint(row["start"], row["count"], lineage, label="credit pilot validation")
        validate_seed_range(row["start"], row["count"], "validation", exposure)
    paths = {name: Path(path).resolve() for name, path in runs.items()}
    endpoints = {arm: _endpoint(paths[arm], arm, protocol, protocol_bytes, lineage) for arm in ARMS}
    if len({row[1]["first_batch_trajectory_sha256"] for row in endpoints.values()}) != 1:
        raise ValueError("Pilot arms did not share the same sampled first batch before updates")
    reference_paths = {"balanced_v3": BLUE_ROOT / "Checkpoints/balanced-v3-candidate",
                       "robust_v2": BLUE_ROOT / "Checkpoints/robust-v2-candidate",
                       "adaptive_v1": BLUE_ROOT / "Checkpoints/adaptive-v1"}
    expected_references = {"balanced_v3": protocol["frozen_reference"]["weights_sha256"], "robust_v2": V2_WEIGHTS, "adaptive_v1": V1_WEIGHTS}
    reference_files = {name: _checkpoint_files(path) for name, path in reference_paths.items()}
    for name, path in reference_paths.items():
        actor = (BalancedPolicy if name == "balanced_v3" else AdaptivePolicy).load(path, feature_names=FEATURE_NAMES)
        if actor.weights_fingerprint() != expected_references[name]:
            raise ValueError("Frozen reference weights differ")
    sources = source_provenance()
    if not _same(sources, protocol["evaluation_implementation_sha256"]):
        raise ValueError("Predeclared evaluation implementation changed during preflight")
    inputs = {"schema": INPUT_SCHEMA, "protocol_sha256": _sha(protocol_bytes), "implementation_sha256": sources,
              "endpoint_evidence": {arm: evidence for arm, (_, evidence) in endpoints.items()},
              "reference_checkpoint_files_sha256": reference_files, "reference_weights_sha256": expected_references,
              "lineage": exposure, "validation_ranges": protocol["validation"]["scenario_seed_ranges"]}
    weights = {arm: policy.weights_fingerprint() for arm, (policy, _) in endpoints.items()} | expected_references

    def stable():
        if (protocol_path.read_bytes() != protocol_bytes or not _same(source_provenance(), sources)
                or any({name: _sha(_file(paths[arm] / name)) for name in RUN_FILES} != evidence["run_files_sha256"] for arm, (_, evidence) in endpoints.items())
                or any(_checkpoint_files(path) != reference_files[name] for name, path in reference_paths.items())):
            raise ValueError("Evaluation source/protocol/endpoint/reference changed; no further persistence")

    stable()
    lock = output / "evaluation-inputs.json"
    names = {"evaluation-inputs.json", "aggregate.json", *[f"validation-{p}{suffix}" for p in PROFILES for suffix in (".json.gz", ".receipt.json")]}
    if output.exists() and (not output.is_dir() or any(path.name not in names or not path.is_file() for path in output.iterdir())):
        raise ValueError("Evaluation output contains unexpected files")
    if output.exists() and any(output.iterdir()):
        if not resume or not lock.exists() or lock.read_bytes() != _bytes(inputs):
            raise ValueError("Explicit resume requires the identical persisted evaluation inputs")
    else:
        if resume:
            raise ValueError("There is no persisted evaluation to resume")
        output.mkdir(parents=True, exist_ok=True)
        _write_new(lock, _bytes(inputs))
    # Validate every existing profile before sampling any missing profile.
    cached = {}
    for profile in PROFILES:
        destination, receipt_path = output / f"validation-{profile}.json.gz", output / f"validation-{profile}.receipt.json"
        if destination.exists() or receipt_path.exists():
            if not resume or not destination.exists() or not receipt_path.exists():
                raise ValueError("Incomplete/existing profile output will not be overwritten")
            cached[profile] = _cached_report(destination, receipt_path, profile, inputs, weights)
    stable()
    factories = {}
    for name in METHODS[:-1]:
        path = paths[name] / "last" if name in paths else reference_paths[name]
        kind = BalancedPolicy if name in ARMS or name == "balanced_v3" else AdaptivePolicy
        expected = weights[name]
        def factory(seed, path=path, kind=kind, expected=expected):
            policy = kind.load(path, feature_names=FEATURE_NAMES)
            if policy.weights_fingerprint() != expected:
                raise ValueError("Endpoint/reference changed between cases")
            return policy
        factories[name] = factory
    factories["greedy_public"] = GreedyPublicCoverage
    reports, report_files = {}, {}
    for profile in PROFILES:
        destination, receipt_path = output / f"validation-{profile}.json.gz", output / f"validation-{profile}.receipt.json"
        if profile in cached:
            report, data = cached[profile]
        else:
            row = inputs["validation_ranges"][profile]
            report = evaluate_robust_methods(factories, seed=row["start"], episodes=row["count"], profile=profile,
                                             stage="validation", replay_count=0,
                                             bootstrap_samples=protocol["validation"]["bootstrap_samples"], seed_provenance=exposure)
            report["schema"] = SCHEMA
            report["credit_pilot_inputs_sha256"] = canonical_hash(inputs)
            data = gzip.compress(_bytes(report), mtime=0)
        _validate_report(report, profile, inputs, weights)
        stable()
        if not destination.exists():
            _write_new(destination, data)
            _write_new(receipt_path, _bytes({"profile": profile, "inputs_sha256": canonical_hash(inputs), "sha256": _sha(data), "bytes": len(data)}))
        reports[profile] = report
        report_files[profile] = {"path": destination.name, "sha256": _sha(data), "bytes": len(data)}
        print(json.dumps({"profile": profile, "persisted": True, "mean_returns": {name: method["summary"]["mean_return"] for name, method in report["methods"].items()}}), flush=True)
    result = {**aggregate_reports(reports, protocol), "protocol_sha256": _sha(protocol_bytes),
              "inputs_sha256": canonical_hash(inputs), "implementation_sha256": sources, "reports": report_files,
              "seed_provenance": {**exposure, "pilot_validation": inputs["validation_ranges"]}}
    stable()
    if not _same(load_published_lineage(), lineage):
        raise ValueError("Published inherited lineage changed during evaluation")
    aggregate_path = output / "aggregate.json"
    if aggregate_path.exists():
        if not resume or aggregate_path.read_bytes() != _bytes(result):
            raise ValueError("Existing aggregate differs and will not be overwritten")
    else:
        _write_new(aggregate_path, _bytes(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="ARM=RUN_DIRECTORY")
    parser.add_argument("--protocol", dest="protocol_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args(argv))
    runs = {}
    for item in args.pop("run"):
        arm, separator, path = item.partition("=")
        if not separator or arm not in ARMS or arm in runs or not path:
            parser.error("Each --run must be a unique known ARM=RUN_DIRECTORY")
        runs[arm] = Path(path)
    result = run_evaluation(runs, **args)
    print(json.dumps(result["scale_gate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
