"""Fixed-endpoint multi-seed ranking evaluation on paired development scenarios.

All declared training seeds contribute equally; none is selected. The seed-mean
is formed within each scenario before paired, profile-stratified bootstrapping.
Its interval is conditional on these trained seeds, not training-population
uncertainty. Reserved final tests are never evaluated by this command.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import io
from pathlib import Path

import numpy as np

from evaluate_robust import evaluate_robust_methods, implementation_fingerprints, validate_seed_range, V1_WEIGHTS
from triad_rl.adaptive_evaluation import GreedyPublicCoverage, canonical_hash, summarize
from triad_rl.adaptive_inputs import FEATURE_NAMES
from triad_rl.adaptive_policy import AdaptivePolicy
from triad_rl.balanced_policy import BalancedPolicy
from triad_rl.ranking_policy import RankPolicy
from triad_rl.ranking_lineage import load_published_lineage, assert_disjoint
from triad_rl.evaluate_credit_pilot import _bytes, _same, _sha, _file, _read, _checkpoint_files, _vectors, stratified_interval
from triad_rl import train_ranking as trainer
from triad_rl.train_adaptive import scenario_seed


BLUE_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = BLUE_ROOT / "Python"
REPORT_SCHEMA = "triad.ranking_evaluation.v1"
INPUT_SCHEMA = "triad.ranking_evaluation_inputs.v1"
AGGREGATE_SCHEMA = "triad.ranking_aggregate.v1"
PROFILES = ("normal", "stress", "capability")
METRICS = ("timely_fraction", "all_threat_success", "detection_rate", "return", "cost")
REFERENCES = {
    "balanced_v3": ("Checkpoints/balanced-v3-candidate", "a48d3c3a5b43e99dd4545cfb7321eb64077dceb09432a60658c2649dd19f21c0", BalancedPolicy),
    "robust_v2": ("Checkpoints/robust-v2-candidate", "5c982c86a5da2e5b4960ba437021f751a36dc3af7187515903cf1b1ee2ab27ce", AdaptivePolicy),
    "adaptive_v1": ("Checkpoints/adaptive-v1", V1_WEIGHTS, AdaptivePolicy),
    "anchored_later_stop": ("Results/anchored-v4-pilot/training/anchored_later_stop/last", "9473c31f8e993a23c41c5ae1ead68ac04b9fa311c22063c026f8657924cbb484", BalancedPolicy),
}
BASELINES = ("greedy_public", *REFERENCES)
EVALUATION_FILES = ("evaluation-inputs.json", "aggregate.json", *(
    f"validation-{profile}{suffix}" for profile in PROFILES for suffix in (".json.gz", ".receipt.json")))
MAX_BYTES = 128 * 1024 * 1024


def ranker_names(protocol):
    return tuple(f"ranker_{seed}" for seed in sorted(protocol["training"]["policy_seeds"]))


def method_names(protocol):
    return (*ranker_names(protocol), *BASELINES)


def derived_same(left, right):
    """Tolerate floating arithmetic only; structure, discrete values and hashes stay exact."""
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return bool(np.isfinite(left) and np.isfinite(right) and np.isclose(left, right, rtol=1e-12, atol=1e-12))
    if isinstance(left, dict):
        return set(left) == set(right) and all(derived_same(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(derived_same(a, b) for a, b in zip(left, right))
    return left == right


def source_provenance():
    return {**implementation_fingerprints(), **trainer.source_provenance(), **{
        name: _sha((PYTHON_ROOT / name).read_bytes()) for name in (
            "evaluate_ranking.py", "triad_rl/evaluate_credit_pilot.py", "triad_rl/placement_policy.py")}}


def load_protocol(path):
    protocol, raw = trainer.load_protocol(path)
    validation = protocol.get("validation", {})
    if (not 1 <= len(protocol["training"]["policy_seeds"]) <= 3
            or validation.get("stage") != "validation" or validation.get("methods") != list(method_names(protocol))
            or type(validation.get("episodes_per_profile")) is not int or not 1 <= validation["episodes_per_profile"] <= 200
            or type(validation.get("bootstrap_samples")) is not int or validation["bootstrap_samples"] != 2000
            or type(validation.get("bootstrap_seed")) is not int or validation["bootstrap_seed"] != 73043
            or set(validation.get("run_seeds", {})) != set(PROFILES)
            or set(validation.get("scenario_seed_ranges", {})) != set(PROFILES)):
        raise ValueError("Ranking validation differs from the fixed bounded design")
    for profile, row in validation["scenario_seed_ranges"].items():
        if not _same(row, {"start": scenario_seed("validation", validation["run_seeds"][profile], 0),
                           "count": validation["episodes_per_profile"]}):
            raise ValueError("Ranking validation seed slots differ")
        validate_seed_range(row["start"], row["count"], "validation", {
            other: value for other, value in validation["scenario_seed_ranges"].items() if other != profile})
    gate = {"comparators": ["greedy_public", "balanced_v3"], "minimum_equal_profile_gain": .01,
            "require_paired_ci_lower_above_zero": True, "maximum_profile_point_decline": .01,
            "maximum_individual_seed_point_decline": .01, "require_equal_profile_detection_not_lower": True,
            "require_equal_profile_all_threat_success_not_lower": True, "require_equal_profile_mean_return_not_lower": True}
    if not _same({key: protocol.get("scale_gate", {}).get(key) for key in gate}, gate):
        raise ValueError("Ranking multi-seed sensing gate differs from its fixed contract")
    if not _same(protocol.get("evaluation_implementation_sha256"), source_provenance()):
        raise ValueError("Ranking evaluation sources differ from the declaration")
    return protocol, raw


def aggregate_reports(reports, protocol):
    """Average rankers within paired cases, never treat seed-cases as independent."""
    rankers, methods = ranker_names(protocol), method_names(protocol)
    if set(reports) != set(PROFILES) or any(set(report.get("methods", {})) != set(methods) for report in reports.values()):
        raise ValueError("Need every declared ranker, baseline and profile")
    vectors = {profile: {method: _vectors(report, method) for method in methods} for profile, report in reports.items()}
    for profile, report in reports.items():
        identities = [(row["seed"], row["scenario_sha256"]) for row in report["methods"][rankers[0]]["episodes"]]
        if not identities or any([(row["seed"], row["scenario_sha256"]) for row in report["methods"][name]["episodes"]] != identities for name in methods):
            raise ValueError("All rankers and baselines must share identical ordered cases")
        vectors[profile]["ranker_mean"] = {metric: np.stack([vectors[profile][name][metric] for name in rankers]).mean(axis=0)
                                             for metric in METRICS}
    summaries = {name: {
        "profiles": {profile: {metric: float(vector.mean()) for metric, vector in vectors[profile][name].items()} for profile in PROFILES},
        "equal_profile": {metric: float(np.mean([vectors[profile][name][metric].mean() for profile in PROFILES])) for metric in METRICS},
    } for name in (*methods, "ranker_mean")}
    validation, comparisons = protocol["validation"], {}
    for name in (*rankers, "ranker_mean"):
        for reference in BASELINES:
            comparisons[f"{name}_minus_{reference}"] = {metric: stratified_interval(
                {profile: vectors[profile][name][metric] - vectors[profile][reference][metric] for profile in PROFILES},
                samples=validation["bootstrap_samples"], seed=validation["bootstrap_seed"]) for metric in METRICS}
    spec, conditions = protocol["scale_gate"], {}
    for reference in spec["comparators"]:
        compared = comparisons[f"ranker_mean_minus_{reference}"]
        timely, detection = compared["timely_fraction"], compared["detection_rate"]
        individual = {name: comparisons[f"{name}_minus_{reference}"]["timely_fraction"]["difference"]
                      >= -spec["maximum_individual_seed_point_decline"] for name in rankers}
        checks = {
            "minimum_mean_timely_gain": timely["difference"] >= spec["minimum_equal_profile_gain"],
            "mean_timely_ci_lower_above_zero": timely["lower95"] > 0.,
            "profile_timely_point_safeguard": all(value >= -spec["maximum_profile_point_decline"] for value in timely["profile_differences"].values()),
            "profile_detection_point_safeguard": all(value >= -spec["maximum_profile_point_decline"] for value in detection["profile_differences"].values()),
            "individual_seed_timely_point_safeguard": all(individual.values()),
            "mean_detection_not_lower": detection["difference"] >= 0.,
            "mean_all_threat_success_not_lower": compared["all_threat_success"]["difference"] >= 0.,
            "mean_return_not_lower": compared["return"]["difference"] >= 0.,
        }
        conditions[reference] = {"passed": all(checks.values()), "checks": checks, "individual_seed_checks": individual}
    passed = all(row["passed"] for row in conditions.values())
    return {"schema": AGGREGATE_SCHEMA, "stage": "validation", "final_test_accessed": False,
            "training_seeds": sorted(protocol["training"]["policy_seeds"]), "methods": summaries,
            "paired_stratified_differences": comparisons,
            "mean_contract": "Equal mean across every declared ranker within each paired scenario, then equal profile means; no ranker selection.",
            "interval_scope": "Paired scenario uncertainty conditional on these trained seeds, not full training-population robustness; seed-cases are not independent scenarios.",
            "scale_gate": {"primary_statistic": "ranker_mean", "passed": passed, "comparators": conditions,
                           "decision": "eligible_for_further_replication" if passed else "do_not_scale_from_this_experiment",
                           "not_policy_promotion": True, "point_guards_are_not_noninferiority_tests": True}}


class GreedyInitializationControl(GreedyPublicCoverage):
    """Frozen greedy actions, checked against all zero-residual initial actors."""
    def __init__(self, initializers):
        self.initializers = initializers
        self.metadata = {**deepcopy(GreedyPublicCoverage.metadata),
                         "initialized_weights_sha256": {str(seed): actor.weights_fingerprint() for seed, actor in initializers.items()},
                         "initialization_control": "Every greedy decision is asserted equal to every declared initialized ranker; no extra scenarios."}

    def act(self, observation, deterministic=True):
        action = super().act(observation, deterministic=True)
        for seed, actor in self.initializers.items():
            if (actor.act(deepcopy(observation), deterministic=True) != action
                    or actor.weights_fingerprint() != self.metadata["initialized_weights_sha256"][str(seed)]):
                raise ValueError("Initialized ranker is not the exact frozen greedy control")
        return action


def prepare_evaluation(runs, protocol_path):
    """Read-only preflight; replay trainer evidence in memory, never sample cases."""
    protocol_path = Path(protocol_path).resolve()
    protocol, raw = load_protocol(protocol_path)
    seeds = sorted(protocol["training"]["policy_seeds"])
    if any(type(seed) is not int for seed in runs) or set(runs) != set(seeds):
        raise ValueError("Every declared fixed-endpoint training seed is required exactly once")
    lineage = load_published_lineage()
    reservations = protocol.get("reserved_final_tests_unopened", {})
    if (set(reservations) != set(PROFILES)
            or not _same(sorted(reservations.values(), key=lambda row: row["start"]), lineage["reserved_final_seed_ranges"])
            or not _same({key: protocol.get("prior_publication", {}).get(key) for key in ("path", "sha256")},
                         lineage["prior_pilot"]["publication_manifest"])):
        raise ValueError("Ranking prior publication or unopened final reservations differ")
    exposure = {"inherited": lineage, "ranking_training": protocol["training"]["scenario_seed_ranges"]}
    for row in protocol["validation"]["scenario_seed_ranges"].values():
        assert_disjoint(row["start"], row["count"], lineage, label="ranking validation")
        validate_seed_range(row["start"], row["count"], "validation", exposure)
    paths = {seed: Path(runs[seed]).resolve() for seed in seeds}
    endpoints = {seed: trainer.validate_run(path, protocol_path, lineage=lineage) for seed, path in paths.items()}
    for seed, (_, evidence) in endpoints.items():
        if evidence["summary"]["policy_seed"] != seed:
            raise ValueError("Endpoint does not belong to its declared training seed")
    weights = {f"ranker_{seed}": actor.weights_fingerprint() for seed, (actor, _) in endpoints.items()}
    reference_files = {}
    for name, (relative, expected, kind) in REFERENCES.items():
        path = BLUE_ROOT / relative
        reference_files[name] = _checkpoint_files(path)
        if kind.load(path, feature_names=FEATURE_NAMES).weights_fingerprint() != expected:
            raise ValueError("Frozen ranking reference fingerprint differs")
        weights[name] = expected
    sources = source_provenance()
    if not _same(sources, protocol["evaluation_implementation_sha256"]):
        raise ValueError("Ranking evaluation sources changed during preflight")
    inputs = {"schema": INPUT_SCHEMA, "protocol_sha256": _sha(raw), "implementation_sha256": sources,
              "endpoint_evidence": {str(seed): evidence for seed, (_, evidence) in endpoints.items()},
              "reference_files_sha256": reference_files, "weights_sha256": weights,
              "lineage": exposure, "validation_ranges": protocol["validation"]["scenario_seed_ranges"]}
    prepared = {"protocol": protocol, "protocol_bytes": raw, "protocol_path": protocol_path,
                "paths": paths, "inputs": inputs, "weights": weights}
    _stable(prepared)
    return prepared


def _stable(prepared):
    inputs = prepared["inputs"]
    if (prepared["protocol_path"].read_bytes() != prepared["protocol_bytes"]
            or not _same(source_provenance(), inputs["implementation_sha256"])
            or any({name: _sha(_file(path / name)) for name in trainer.RUN_FILES} != inputs["endpoint_evidence"][str(seed)]["run_files_sha256"]
                   for seed, path in prepared["paths"].items())
            or any(_checkpoint_files(BLUE_ROOT / relative) != inputs["reference_files_sha256"][name]
                   for name, (relative, _, _) in REFERENCES.items())):
        raise ValueError("Ranking protocol/source/complete-run/reference changed")


def validate_report(report, profile, prepared):
    inputs, weights, protocol = prepared["inputs"], prepared["weights"], prepared["protocol"]
    interval = inputs["validation_ranges"][profile]
    if (report.get("schema") != REPORT_SCHEMA or report.get("profile") != profile or report.get("stage") != "validation"
            or report.get("training_performed") is not False
            or report.get("protocol", {}).get("independent_final_test_evidence") is not False
            or report.get("ranking_inputs_sha256") != canonical_hash(inputs)
            or not _same(report.get("seed_provenance", {}).get("evaluation"), interval)
            or not _same(report.get("seed_provenance", {}).get("declared_lineage"), inputs["lineage"])
            or not _same(report.get("implementation_sha256"), implementation_fingerprints())
            or not _same(report.get("implementation_sha256_after"), implementation_fingerprints())
            or set(report.get("methods", {})) != set(method_names(protocol))):
        raise ValueError("Ranking report scope, source or lineage binding differs")
    seeds = list(range(interval["start"], interval["start"] + interval["count"]))
    identities = None
    for name, method in report["methods"].items():
        rows = method["episodes"]
        hashes = [canonical_hash(row["scenario"]) for row in rows]
        if (any(type(row["seed"]) is not int for row in rows) or [row["seed"] for row in rows] != seeds
                or hashes != [row["scenario_sha256"] for row in rows]
                or identities is not None and hashes != identities or not derived_same(summarize(rows), method["summary"])
                or any(row.get("weights_sha256_before") != weights.get(name) or row.get("weights_sha256_after") != weights.get(name) for row in rows)):
            raise ValueError("Ranking report pairing, summary or frozen weights differ")
        identities = hashes
        _vectors(report, name)
    expected_initials = {seed: evidence["initialized_weights_sha256"] for seed, evidence in inputs["endpoint_evidence"].items()}
    if not _same(report["methods"]["greedy_public"]["metadata"].get("initialized_weights_sha256"), expected_initials):
        raise ValueError("Ranking report omits its exact greedy initialization control")


def _read_profiles(output, prepared, *, require_complete):
    """Read all present profiles before any missing one may be evaluated."""
    reports, compressed, snapshots = {}, {}, {}
    if output.exists() and (not output.is_dir() or any(not path.is_file() or path.name not in EVALUATION_FILES for path in output.iterdir())):
        raise ValueError("Ranking output contains unexpected artifacts")
    lock = output / "evaluation-inputs.json"
    if not lock.is_file() or _file(lock) != _bytes(prepared["inputs"]):
        raise ValueError("Saved ranking evaluation inputs differ")
    snapshots[lock] = _file(lock)
    for profile in PROFILES:
        path, receipt_path = output / f"validation-{profile}.json.gz", output / f"validation-{profile}.receipt.json"
        if path.exists() or receipt_path.exists() or require_complete or (output / "aggregate.json").exists():
            if not path.is_file() or not receipt_path.is_file():
                raise ValueError("Incomplete ranking profile cannot be reused or overwritten")
            data, receipt_data = _file(path), _file(receipt_path)
            receipt = trainer.shared._strict_json(receipt_data)
            expected = {"profile": profile, "sha256": _sha(data), "bytes": len(data), "inputs_sha256": canonical_hash(prepared["inputs"])}
            if not _same(receipt, expected):
                raise ValueError("Ranking profile compressed-byte receipt differs")
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
                expanded = handle.read(MAX_BYTES + 1)
            if len(expanded) > MAX_BYTES:
                raise ValueError("Ranking expanded report exceeds size bound")
            report = trainer.shared._strict_json(expanded)
            validate_report(report, profile, prepared)
            reports[profile], compressed[profile] = report, data
            snapshots.update({path: data, receipt_path: receipt_data})
    return reports, compressed, snapshots


def _result(reports, compressed, prepared):
    inputs = prepared["inputs"]
    return {**aggregate_reports(reports, prepared["protocol"]), "protocol_sha256": _sha(prepared["protocol_bytes"]),
            "inputs_sha256": canonical_hash(inputs), "implementation_sha256": inputs["implementation_sha256"],
            "reports": {profile: {"path": f"validation-{profile}.json.gz", "sha256": _sha(data), "bytes": len(data)} for profile, data in compressed.items()},
            "seed_provenance": {**inputs["lineage"], "ranking_validation": inputs["validation_ranges"]}}


def _finish_checks(prepared, snapshots):
    _stable(prepared)
    if any(_file(path) != data for path, data in snapshots.items()):
        raise ValueError("Persisted ranking reports changed during verification")
    if not _same(load_published_lineage(), prepared["inputs"]["lineage"]["inherited"]):
        raise ValueError("Inherited ranking publication changed")


def verify_completed_evaluation(runs, *, protocol_path, output):
    """Read-only terminal verifier for archival; no runner, act or simulator calls."""
    prepared, output = prepare_evaluation(runs, protocol_path), Path(output).resolve()
    aggregate_bytes = _file(output / "aggregate.json")
    reports, compressed, snapshots = _read_profiles(output, prepared, require_complete=True)
    result = _result(reports, compressed, prepared)
    stored = trainer.shared._strict_json(aggregate_bytes)
    if not derived_same(stored, result):
        raise ValueError("Stored ranking aggregate, paired intervals or gate differs")
    snapshots[output / "aggregate.json"] = aggregate_bytes
    _finish_checks(prepared, snapshots)
    return stored


def _new(path, data):
    with path.open("xb") as handle:
        handle.write(data)


def run_evaluation(runs, *, protocol_path, output, resume=False):
    prepared, output = prepare_evaluation(runs, protocol_path), Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        if not resume:
            raise ValueError("Existing ranking evaluation requires explicit resume")
    else:
        if resume:
            raise ValueError("No ranking evaluation exists to resume")
        output.mkdir(parents=True, exist_ok=True)
        _new(output / "evaluation-inputs.json", _bytes(prepared["inputs"]))
    reports, compressed, snapshots = _read_profiles(output, prepared, require_complete=False)
    _stable(prepared)
    factories = {}
    for seed, path in prepared["paths"].items():
        def factory(unused_seed, path=path, expected=prepared["weights"][f"ranker_{seed}"]):
            actor = RankPolicy.load(path / "last", feature_names=FEATURE_NAMES)
            if actor.weights_fingerprint() != expected:
                raise ValueError("Ranker checkpoint changed between cases")
            return actor
        factories[f"ranker_{seed}"] = factory
    def greedy(unused_seed):
        initializers = {seed: RankPolicy.load(path / "initialized", feature_names=FEATURE_NAMES) for seed, path in prepared["paths"].items()}
        if any(actor.weights_fingerprint() != prepared["inputs"]["endpoint_evidence"][str(seed)]["initialized_weights_sha256"]
               for seed, actor in initializers.items()):
            raise ValueError("Initialized ranker checkpoint changed between cases")
        return GreedyInitializationControl(initializers)
    factories["greedy_public"] = greedy
    for name, (relative, expected, kind) in REFERENCES.items():
        def reference(unused_seed, relative=relative, expected=expected, kind=kind):
            actor = kind.load(BLUE_ROOT / relative, feature_names=FEATURE_NAMES)
            if actor.weights_fingerprint() != expected:
                raise ValueError("Frozen reference changed between cases")
            return actor
        factories[name] = reference
    for profile in PROFILES:
        if profile not in reports:
            row = prepared["inputs"]["validation_ranges"][profile]
            report = evaluate_robust_methods(factories, seed=row["start"], episodes=row["count"], profile=profile,
                                             stage="validation", replay_count=0, bootstrap_samples=2000,
                                             seed_provenance=prepared["inputs"]["lineage"])
            report.update(schema=REPORT_SCHEMA, ranking_inputs_sha256=canonical_hash(prepared["inputs"]))
            validate_report(report, profile, prepared)
            data = gzip.compress(_bytes(report), mtime=0)
            _stable(prepared)
            path, receipt = output / f"validation-{profile}.json.gz", output / f"validation-{profile}.receipt.json"
            _new(path, data)
            _new(receipt, _bytes({"profile": profile, "sha256": _sha(data), "bytes": len(data), "inputs_sha256": canonical_hash(prepared["inputs"])}))
            reports[profile], compressed[profile] = report, data
            snapshots.update({path: data, receipt: _file(receipt)})
        print(f"Ranking validation {profile}: complete", flush=True)
    result = _result(reports, compressed, prepared)
    _finish_checks(prepared, snapshots)
    path = output / "aggregate.json"
    if path.exists():
        stored = _read(path)
        if not resume or not derived_same(stored, result):
            raise ValueError("Existing ranking aggregate differs and will not be overwritten")
        return stored
    else:
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
        if not separator or not seed.isdigit() or int(seed) in runs or not path:
            parser.error("Each run must be a unique SEED=RUN_DIRECTORY")
        runs[int(seed)] = Path(path)
    print(_bytes(run_evaluation(runs, **args)["scale_gate"]).decode())


if __name__ == "__main__":
    main()
