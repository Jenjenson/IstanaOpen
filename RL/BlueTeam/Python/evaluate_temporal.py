"""Paired fixed-endpoint temporal pilot validation, never final-test evaluation.

Each scenario/catalogue is generated once, then supplied to the unchanged
scoring core for every method. Temporal policies receive only public v2 inputs;
legacy policies receive their unchanged v1 inputs. The seed mean is formed
within each case before bootstrap: seed-cases are not independent observations.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import io
from pathlib import Path

import numpy as np

import train_temporal as trainer
from evaluate_robust import evaluate_robust_methods, implementation_fingerprints, validate_seed_range, V1_WEIGHTS
from triad_rl.adaptive_env import AdaptivePlacementEnv
from triad_rl.adaptive_evaluation import GreedyPublicCoverage, canonical_hash, summarize
from triad_rl.adaptive_inputs import FEATURE_NAMES as LEGACY_NAMES, validate_catalogue
from triad_rl.adaptive_policy import AdaptivePolicy, _json
from triad_rl.balanced_policy import BalancedPolicy
from triad_rl.evaluate_credit_pilot import _vectors, stratified_interval
from triad_rl.ranking_policy import _strict_json
from triad_rl.robust_scenarios import make_case
from triad_rl.temporal_env import _MISSION_RULES
from triad_rl.temporal_inputs import TemporalConfig, TemporalObservationBuilder, TemporalPublicGreedy
from triad_rl.temporal_lineage import load_published_lineage, assert_disjoint

BLUE_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = BLUE_ROOT / "Python"
REPORT_SCHEMA = "triad.temporal_evaluation.v1"
INPUT_SCHEMA = "triad.temporal_evaluation_inputs.v1"
AGGREGATE_SCHEMA = "triad.temporal_aggregate.v1"
PROFILES = ("normal", "stress", "capability")
METRICS = ("timely_fraction", "all_threat_success", "detection_rate", "return", "cost")
REFERENCES = {
    "balanced_v3": ("Checkpoints/balanced-v3-candidate", "a48d3c3a5b43e99dd4545cfb7321eb64077dceb09432a60658c2649dd19f21c0", BalancedPolicy),
    "robust_v2": ("Checkpoints/robust-v2-candidate", "5c982c86a5da2e5b4960ba437021f751a36dc3af7187515903cf1b1ee2ab27ce", AdaptivePolicy),
    "adaptive_v1": ("Checkpoints/adaptive-v1", V1_WEIGHTS, AdaptivePolicy),
    "anchored_later_stop": ("Results/anchored-v4-pilot/training/anchored_later_stop/last", "9473c31f8e993a23c41c5ae1ead68ac04b9fa311c22063c026f8657924cbb484", BalancedPolicy),
}
BASELINES = ("temporal_public", "greedy_public", *REFERENCES)
VALIDATION_SETTINGS = {"stage": "validation", "episodes_per_profile": 200, "bootstrap_samples": 2000,
                       "bootstrap_seed": 73044, "run_seeds": dict(zip(PROFILES, (995400, 995401, 995402)))}
GATE_SPEC = {"comparators": ["temporal_public", "greedy_public", "balanced_v3"], "minimum_equal_profile_gain": .01,
            "require_paired_ci_lower_above_zero": True, "maximum_profile_point_decline": .01,
            "maximum_individual_seed_point_decline": .01, "require_equal_profile_detection_not_lower": True,
            "require_equal_profile_all_threat_success_not_lower": True, "require_equal_profile_mean_return_not_lower": True}
EVALUATION_FILES = ("evaluation-inputs.json", "aggregate.json", *(
    f"validation-{profile}{suffix}" for profile in PROFILES for suffix in (".json.gz", ".receipt.json")))
MAX_BYTES = 128 * 1024 * 1024


def _bytes(value):
    return (_json(value) + "\n").encode()


def _same(left, right):
    return _json(left) == _json(right)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _file(path):
    if Path(path).is_symlink():
        raise ValueError("Evaluation evidence must not be a symlink")
    with Path(path).open("rb") as handle:
        data = handle.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("Evaluation evidence exceeds bounded size")
    return data


def _checkpoint_files(path):
    return {name: _sha(_file(path / name)) for name in ("checkpoint.json", "arrays.npz")}


def derived_same(left, right):
    """Only recomputed floating values tolerate roundoff; discrete fields are exact."""
    if type(left) is not type(right): return False
    if isinstance(left, float):
        return bool(np.isfinite(left) and np.isfinite(right) and np.isclose(left, right, rtol=1e-12, atol=1e-12))
    if isinstance(left, dict): return set(left) == set(right) and all(derived_same(left[key], right[key]) for key in left)
    if isinstance(left, list): return len(left) == len(right) and all(derived_same(a, b) for a, b in zip(left, right))
    return left == right


def temporal_names(protocol):
    return tuple(f"temporal_{seed}" for seed in sorted(protocol["training"]["policy_seeds"]))


def method_names(protocol):
    return (*temporal_names(protocol), *BASELINES)


def source_provenance():
    names = ("evaluate_temporal.py", "triad_rl/evaluate_credit_pilot.py", "triad_rl/placement_policy.py",
             "triad_rl/temporal_lineage.py", "triad_rl/credit_lineage.py", "triad_rl/balanced_policy.py")
    return {**implementation_fingerprints(), **trainer.source_provenance(),
            **{name: _sha((PYTHON_ROOT / name).read_bytes()) for name in names}}


def load_protocol(path):
    protocol, raw = trainer.load_protocol(path)
    validation = protocol.get("validation", {})
    if (len(protocol["training"]["policy_seeds"]) != 3
            or not _same({key: validation.get(key) for key in VALIDATION_SETTINGS}, VALIDATION_SETTINGS)
            or not _same(validation.get("methods"), list(method_names(protocol)))
            or set(validation.get("scenario_seed_ranges", {})) != set(PROFILES)
            or not _same({key: protocol.get("scale_gate", {}).get(key) for key in GATE_SPEC}, GATE_SPEC)):
        raise ValueError("Temporal validation/gate differs from the fixed design")
    for profile, interval in validation["scenario_seed_ranges"].items():
        expected = {"start": 10**15 + validation["run_seeds"][profile] * 1_000_000,
                    "count": validation["episodes_per_profile"]}
        if not _same(interval, expected): raise ValueError("Temporal validation slots differ")
        validate_seed_range(interval["start"], interval["count"], "validation", {
            name: row for name, row in validation["scenario_seed_ranges"].items() if name != profile})
    if not _same(protocol.get("evaluation_implementation_sha256"), source_provenance()):
        raise ValueError("Temporal evaluation source declaration differs")
    return protocol, raw


def aggregate_reports(reports, protocol):
    """Equal trained-seed mean within case; paired, equal-profile bootstrap."""
    actors, methods = temporal_names(protocol), method_names(protocol)
    if set(reports) != set(PROFILES): raise ValueError("All three profiles are required")
    vectors = {}
    for profile, report in reports.items():
        if (report.get("profile") != profile or report.get("stage") != "validation"
                or set(report.get("methods", {})) != set(methods)):
            raise ValueError("Aggregation requires profile-matched validation with every method")
        reference = report["methods"][actors[0]]["episodes"]
        identities = [(row["seed"], row["scenario_sha256"]) for row in reference]
        if not identities or any(not _same([(row["seed"], row["scenario_sha256"]) for row in report["methods"][name]["episodes"]], identities) for name in methods):
            raise ValueError("Methods must share exactly ordered paired cases")
        vectors[profile] = {name: _vectors(report, name) for name in methods}
        vectors[profile]["temporal_mean"] = {metric: np.stack([vectors[profile][name][metric] for name in actors]).mean(axis=0) for metric in METRICS}
    summaries = {name: {
        "profiles": {profile: {key: float(value.mean()) for key, value in vectors[profile][name].items()} for profile in PROFILES},
        "equal_profile": {key: float(np.mean([vectors[profile][name][key].mean() for profile in PROFILES])) for key in METRICS},
    } for name in (*methods, "temporal_mean")}
    comparisons, settings = {}, protocol["validation"]
    for name in (*actors, "temporal_mean"):
        for reference in BASELINES:
            comparisons[f"{name}_minus_{reference}"] = {metric: stratified_interval(
                {profile: vectors[profile][name][metric] - vectors[profile][reference][metric] for profile in PROFILES},
                samples=settings["bootstrap_samples"], seed=settings["bootstrap_seed"]) for metric in METRICS}
    spec, conditions = protocol["scale_gate"], {}
    for reference in spec["comparators"]:
        compared = comparisons[f"temporal_mean_minus_{reference}"]
        timely, detection = compared["timely_fraction"], compared["detection_rate"]
        individual = {name: comparisons[f"{name}_minus_{reference}"]["timely_fraction"]["difference"] >= -spec["maximum_individual_seed_point_decline"] for name in actors}
        checks = {"minimum_mean_timely_gain": timely["difference"] >= spec["minimum_equal_profile_gain"],
                  "mean_timely_ci_lower_above_zero": timely["lower95"] > 0.,
                  "profile_timely_point_safeguard": all(value >= -spec["maximum_profile_point_decline"] for value in timely["profile_differences"].values()),
                  "profile_detection_point_safeguard": all(value >= -spec["maximum_profile_point_decline"] for value in detection["profile_differences"].values()),
                  "individual_seed_timely_point_safeguard": all(individual.values()),
                  "mean_detection_not_lower": detection["difference"] >= 0.,
                  "mean_all_threat_success_not_lower": compared["all_threat_success"]["difference"] >= 0.,
                  "mean_return_not_lower": compared["return"]["difference"] >= 0.}
        conditions[reference] = {"passed": all(checks.values()), "checks": checks, "individual_seed_checks": individual}
    passed = all(row["passed"] for row in conditions.values())
    return {"schema": AGGREGATE_SCHEMA, "stage": "validation", "final_test_accessed": False,
            "training_seeds": sorted(protocol["training"]["policy_seeds"]), "methods": summaries,
            "paired_stratified_differences": comparisons,
            "mean_contract": "Equal mean of all three fixed temporal endpoints within each paired case, then equal profile means; no selection.",
            "interval_scope": "Scenario uncertainty conditional on these trained endpoints, not training-population uncertainty; seed-cases are not independent.",
            "scale_gate": {"primary_statistic": "temporal_mean", "passed": passed, "comparators": conditions,
                           "decision": "eligible_for_further_replication" if passed else "do_not_scale_from_this_experiment",
                           "not_policy_promotion": True, "point_guards_are_not_noninferiority_tests": True}}


class _PublicActor:
    """Dispatch exact raw public snapshots, never private case truth, to v1/v2."""
    def __init__(self, policy, config, temporal):
        self.policy, self.builder = policy, TemporalObservationBuilder(config) if temporal else None
        self.metadata = deepcopy(getattr(policy, "metadata", {}))
        self.metadata.update(temporal_public_features=temporal, physical_commands=False)

    def act(self, observation, deterministic=True):
        if deterministic is not True: raise ValueError("Evaluation requires deterministic actions")
        legacy = dict(observation)
        public = legacy.pop("temporal_public_snapshot")
        selected = legacy
        if self.builder is not None:
            selected = self.builder.observe(public, legacy["catalogue"])
            if (not _same(selected["options"], legacy["options"])
                    or not np.array_equal(selected["action_mask"], legacy["action_mask"])
                    or not np.array_equal(selected["option_features"][:, :len(LEGACY_NAMES)], legacy["option_features"])):
                raise ValueError("Temporal/legacy public action or feature prefix differs")
        rng = getattr(self.policy, "rng", None)
        before = None if rng is None else canonical_hash(rng.bit_generator.state)
        action = self.policy.act(deepcopy(selected), deterministic=True)
        if (isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer))
                or not 0 <= action < len(selected["action_mask"]) or not selected["action_mask"][action]):
            raise ValueError("Evaluation policy returned an invalid action")
        if before != (None if rng is None else canonical_hash(rng.bit_generator.state)):
            raise ValueError("Deterministic evaluation consumed policy RNG")
        return int(action)


class _CaseFactory:
    """Lazy construction: one make_case per seed, zero constructor ghost cases."""
    def __init__(self, interval, profile, config):
        self.interval, self.profile, self.config, self.cases = interval, profile, config, {}

    def __call__(self, *, seed, profile):
        if (type(seed) is not int or profile != self.profile
                or not self.interval["start"] <= seed < self.interval["start"] + self.interval["count"]):
            raise ValueError("Case request is outside the declared validation profile/range")
        return _CaseView(self, seed)


class _CaseView:
    def __init__(self, owner, seed):
        self.owner, self.seed = owner, seed  # No core constructor or sampling.

    def reset(self, *, seed):
        if type(seed) is not int or seed != self.seed: raise ValueError("Paired reset seed changed")
        if seed not in self.owner.cases:
            self.owner.cases[seed] = deepcopy(make_case(seed, self.owner.profile))
        scenario, catalogue, self.case_metadata = deepcopy(self.owner.cases[seed])
        for name, setting in _MISSION_RULES.items():
            if isinstance(scenario.get(name), bool) or scenario.get(name) != getattr(self.owner.config, setting):
                raise ValueError("Scoring mission rules differ from explicit temporal config")
        core = AdaptivePlacementEnv.__new__(AdaptivePlacementEnv)
        core.catalogue, core.split = validate_catalogue(catalogue), scenario.get("split", "supplied")
        core._custom_catalogue, core._rng = True, np.random.default_rng(seed)
        core.reset(scenario=scenario)
        self.core, self.scenario, self.catalogue = core, core.scenario, core.catalogue
        return self.observe()

    def observe(self):
        return {**self.core.observe(), "temporal_public_snapshot": deepcopy(self.core.public_state)}

    def step(self, action):
        _, reward, done, info = self.core.step(action)
        return self.observe(), reward, done, info


def prepare_evaluation(runs, protocol_path):
    protocol_path = Path(protocol_path).resolve()
    protocol, raw = load_protocol(protocol_path)
    seeds = sorted(protocol["training"]["policy_seeds"])
    if any(type(seed) is not int for seed in runs) or set(runs) != set(seeds):
        raise ValueError("Every declared fixed endpoint is required exactly once")
    lineage = load_published_lineage()
    reservations = protocol.get("reserved_final_tests_unopened", {})
    if (set(reservations) != set(PROFILES)
            or not _same(sorted(reservations.values(), key=lambda row: row["start"]), lineage["reserved_final_seed_ranges"])
            or not _same({key: protocol.get("prior_publication", {}).get(key) for key in ("path", "sha256")}, lineage["prior_pilot"]["publication_manifest"])):
        raise ValueError("Prior publication or unopened final reservations differ")
    exposure = {"inherited": lineage, "temporal_training": protocol["training"]["scenario_seed_ranges"]}
    for row in protocol["validation"]["scenario_seed_ranges"].values():
        assert_disjoint(row["start"], row["count"], lineage, label="temporal validation")
        validate_seed_range(row["start"], row["count"], "validation", exposure)
    paths = {seed: Path(path).resolve() for seed, path in runs.items()}
    endpoints = {seed: trainer.validate_run(path, protocol_path, lineage=lineage) for seed, path in paths.items()}
    if any(evidence["summary"]["policy_seed"] != seed for seed, (_, evidence) in endpoints.items()):
        raise ValueError("Endpoint training seed differs")
    actors = {f"temporal_{seed}": actor for seed, (actor, _) in endpoints.items()}
    weights = {name: actor.weights_fingerprint() for name, actor in actors.items()}
    reference_files = {}
    for name, (relative, expected, kind) in REFERENCES.items():
        path = BLUE_ROOT / relative
        reference_files[name] = _checkpoint_files(path)
        actor = kind.load(path, feature_names=LEGACY_NAMES)
        if actor.weights_fingerprint() != expected: raise ValueError("Frozen reference fingerprint differs")
        actors[name], weights[name] = actor, expected
    sources = source_provenance()
    if not _same(sources, protocol["evaluation_implementation_sha256"]): raise ValueError("Evaluation sources changed during preflight")
    inputs = {"schema": INPUT_SCHEMA, "protocol_sha256": _sha(raw), "implementation_sha256": sources,
              "endpoint_evidence": {str(seed): evidence for seed, (_, evidence) in endpoints.items()},
              "reference_files_sha256": reference_files, "weights_sha256": weights,
              "lineage": exposure, "validation_ranges": protocol["validation"]["scenario_seed_ranges"],
              "temporal_config": protocol["temporal_config"]}
    prepared = {"protocol": protocol, "protocol_bytes": raw, "protocol_path": protocol_path,
                "paths": paths, "inputs": inputs, "weights": weights, "actors": actors}
    _stable(prepared)
    return prepared


def _stable(prepared):
    inputs = prepared["inputs"]
    if (_file(prepared["protocol_path"]) != prepared["protocol_bytes"]
            or not _same(source_provenance(), inputs["implementation_sha256"])
            or any({name: _sha(_file(path / name)) for name in trainer.RUN_FILES} != inputs["endpoint_evidence"][str(seed)]["run_files_sha256"] for seed, path in prepared["paths"].items())
            or any(_checkpoint_files(BLUE_ROOT / relative) != inputs["reference_files_sha256"][name] for name, (relative, _, _) in REFERENCES.items())
            or any(actor.weights_fingerprint() != prepared["weights"][name] for name, actor in prepared["actors"].items())):
        raise ValueError("Temporal protocol/source/complete endpoint/reference changed")


def validate_report(report, profile, prepared):
    inputs, protocol = prepared["inputs"], prepared["protocol"]
    interval = inputs["validation_ranges"][profile]
    if (report.get("schema") != REPORT_SCHEMA or report.get("stage") != "validation" or report.get("profile") != profile
            or report.get("training_performed") is not False or report.get("protocol", {}).get("independent_final_test_evidence") is not False
            or report.get("temporal_inputs_sha256") != canonical_hash(inputs)
            or not _same(report.get("seed_provenance", {}).get("evaluation"), interval)
            or not _same(report.get("seed_provenance", {}).get("declared_lineage"), inputs["lineage"])
            or not _same(report.get("implementation_sha256"), implementation_fingerprints())
            or not _same(report.get("implementation_sha256_after"), implementation_fingerprints())
            or set(report.get("methods", {})) != set(method_names(protocol))):
        raise ValueError("Temporal report scope, inputs, sources or lineage differs")
    identities = None
    for name, method in report["methods"].items():
        rows = method["episodes"]
        hashes = [canonical_hash(row["scenario"]) for row in rows]
        if (not _same([row["seed"] for row in rows], list(range(interval["start"], interval["start"] + interval["count"])))
                or hashes != [row["scenario_sha256"] for row in rows] or identities is not None and hashes != identities
                or not derived_same(summarize(rows), method["summary"])
                or any(row["weights_sha256_before"] != prepared["weights"].get(name) or row["weights_sha256_after"] != prepared["weights"].get(name) for row in rows)):
            raise ValueError("Temporal report pairing, summaries or endpoint weights differ")
        identities = hashes
        _vectors(report, name)
    if report.get("scenario_sequence_sha256") != canonical_hash(identities):
        raise ValueError("Temporal report scenario sequence hash differs")


def _read_profiles(output, prepared, require_complete):
    if any(path.name not in EVALUATION_FILES or not path.is_file() for path in output.iterdir()):
        raise ValueError("Unexpected temporal evaluation artifacts")
    lock = output / "evaluation-inputs.json"
    if _file(lock) != _bytes(prepared["inputs"]): raise ValueError("Persisted temporal inputs differ")
    reports, compressed, snapshots = {}, {}, {lock: _file(lock)}
    for profile in PROFILES:
        path, receipt_path = output / f"validation-{profile}.json.gz", output / f"validation-{profile}.receipt.json"
        if require_complete or path.exists() or receipt_path.exists() or (output / "aggregate.json").exists():
            data, receipt_data = _file(path), _file(receipt_path)
            expected = {"profile": profile, "sha256": _sha(data), "bytes": len(data), "inputs_sha256": canonical_hash(prepared["inputs"])}
            if not _same(_strict_json(receipt_data), expected): raise ValueError("Temporal compressed profile receipt differs")
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle: expanded = handle.read(MAX_BYTES + 1)
            if len(expanded) > MAX_BYTES: raise ValueError("Expanded temporal report exceeds bound")
            report = _strict_json(expanded)
            validate_report(report, profile, prepared)
            reports[profile], compressed[profile] = report, data
            snapshots.update({path: data, receipt_path: receipt_data})
    return reports, compressed, snapshots


def _result(reports, compressed, prepared):
    return {**aggregate_reports(reports, prepared["protocol"]), "protocol_sha256": _sha(prepared["protocol_bytes"]),
            "inputs_sha256": canonical_hash(prepared["inputs"]), "implementation_sha256": prepared["inputs"]["implementation_sha256"],
            "reports": {profile: {"path": f"validation-{profile}.json.gz", "sha256": _sha(data), "bytes": len(data)} for profile, data in compressed.items()},
            "seed_provenance": {**prepared["inputs"]["lineage"], "temporal_validation": prepared["inputs"]["validation_ranges"]}}


def _finish(prepared, snapshots):
    _stable(prepared)
    if any(_file(path) != data for path, data in snapshots.items()): raise ValueError("Persisted temporal evidence changed")
    if not _same(load_published_lineage(), prepared["inputs"]["lineage"]["inherited"]): raise ValueError("Inherited temporal lineage changed")


def verify_completed_evaluation(runs, *, protocol_path, output):
    """Read-only bytes/metadata/statistics verification: no inference or replay."""
    prepared, output = prepare_evaluation(runs, protocol_path), Path(output).resolve()
    aggregate_bytes = _file(output / "aggregate.json")
    reports, compressed, snapshots = _read_profiles(output, prepared, True)
    stored = _strict_json(aggregate_bytes)
    if not derived_same(stored, _result(reports, compressed, prepared)): raise ValueError("Temporal aggregate/gate differs")
    snapshots[output / "aggregate.json"] = aggregate_bytes
    _finish(prepared, snapshots)
    return stored


def _new(path, data):
    with path.open("xb") as handle: handle.write(data)


def run_evaluation(runs, *, protocol_path, output, resume=False):
    prepared, output = prepare_evaluation(runs, protocol_path), Path(output).resolve()
    if type(resume) is not bool: raise ValueError("resume must be boolean")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        if not resume: raise ValueError("Existing temporal evaluation requires explicit resume")
    else:
        if resume: raise ValueError("No temporal evaluation exists to resume")
        output.mkdir(parents=True, exist_ok=True)
        _new(output / "evaluation-inputs.json", _bytes(prepared["inputs"]))
    reports, compressed, snapshots = _read_profiles(output, prepared, False)
    _stable(prepared)
    config = TemporalConfig(**prepared["protocol"]["temporal_config"])
    actors = {**prepared["actors"], "temporal_public": TemporalPublicGreedy(), "greedy_public": GreedyPublicCoverage()}
    factories = {name: (lambda unused_seed, name=name: _PublicActor(deepcopy(actors[name]), config, name.startswith("temporal_")))
                 for name in method_names(prepared["protocol"])}
    for profile in PROFILES:
        if profile not in reports:
            interval = prepared["inputs"]["validation_ranges"][profile]
            factory = _CaseFactory(interval, profile, config)
            report = evaluate_robust_methods(factories, seed=interval["start"], episodes=interval["count"], profile=profile,
                stage="validation", replay_count=0, bootstrap_samples=2000, seed_provenance=prepared["inputs"]["lineage"], env_factory=factory)
            report.update(schema=REPORT_SCHEMA, temporal_inputs_sha256=canonical_hash(prepared["inputs"]))
            validate_report(report, profile, prepared)
            if len(factory.cases) != interval["count"]: raise ValueError("Case generator did not cover exactly the declared slots")
            data = gzip.compress(_bytes(report), mtime=0)
            _stable(prepared)
            path, receipt = output / f"validation-{profile}.json.gz", output / f"validation-{profile}.receipt.json"
            _new(path, data)
            _new(receipt, _bytes({"profile": profile, "sha256": _sha(data), "bytes": len(data), "inputs_sha256": canonical_hash(prepared["inputs"])}))
            reports[profile], compressed[profile] = report, data
            snapshots.update({path: data, receipt: _file(receipt)})
        print(f"Temporal validation {profile}: complete", flush=True)
    result = _result(reports, compressed, prepared)
    _finish(prepared, snapshots)
    path = output / "aggregate.json"
    if path.exists():
        stored = _strict_json(_file(path))
        if not derived_same(stored, result): raise ValueError("Existing temporal aggregate differs; no overwrite")
        return stored
    _new(path, _bytes(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="SEED=RUN_DIRECTORY")
    parser.add_argument("--protocol", dest="protocol_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args, runs = vars(parser.parse_args(argv)), {}
    for item in args.pop("run"):
        seed, separator, path = item.partition("=")
        if not separator or not seed.isdecimal() or int(seed) in runs or not path:
            parser.error("Each run must be a unique integer SEED=RUN_DIRECTORY")
        runs[int(seed)] = Path(path)
    print(_bytes(run_evaluation(runs, **args)["scale_gate"]).decode())


if __name__ == "__main__":
    main()
