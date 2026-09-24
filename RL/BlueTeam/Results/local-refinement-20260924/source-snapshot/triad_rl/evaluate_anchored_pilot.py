"""Paired fixed-endpoint evaluation of anchored later-placement training.

Development evidence only. The primary rule is fixed before training; a pass
authorizes replication, not promotion. Reuses the frozen simulator, reference
actors and paired bootstrap mathematics without changing their contracts.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path

import numpy as np

from evaluate_robust import evaluate_robust_methods, implementation_fingerprints, validate_seed_range, V1_WEIGHTS
from .adaptive_evaluation import GreedyPublicCoverage, canonical_hash, summarize
from .adaptive_inputs import FEATURE_NAMES
from .adaptive_policy import AdaptivePolicy
from .balanced_policy import BalancedPolicy
from .anchored_lineage import load_published_lineage, assert_disjoint
from .evaluate_credit_pilot import _bytes, _same, _sha, _file, _read, _checkpoint_files, _vectors, stratified_interval
from . import train_anchored_pilot as trainer
from .train_adaptive import scenario_seed

BLUE_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = BLUE_ROOT / "Python"
SCHEMA = "triad.anchored_pilot_evaluation.v1"
PROFILES = ("normal", "stress", "capability")
ARMS = ("shared_critic", "no_critic_gradient", "anchored_later_stop")
PRIMARY = "anchored_later_stop"
REFERENCES = {
    "balanced_v3": ("Checkpoints/balanced-v3-candidate", "a48d3c3a5b43e99dd4545cfb7321eb64077dceb09432a60658c2649dd19f21c0", BalancedPolicy),
    "robust_v2": ("Checkpoints/robust-v2-candidate", "5c982c86a5da2e5b4960ba437021f751a36dc3af7187515903cf1b1ee2ab27ce", AdaptivePolicy),
    "adaptive_v1": ("Checkpoints/adaptive-v1", V1_WEIGHTS, AdaptivePolicy),
    "all_step_stop_pilot": ("Results/credit-v4-pilot/training/paired_stop/last", "e6ed28a7ae25ed0041162a43205c266a1995115e31d4abe2889e74906551898a", BalancedPolicy),
}
METHODS = (*ARMS, *REFERENCES, "greedy_public")
METRICS = ("timely_fraction", "all_threat_success", "detection_rate", "return", "cost")
EVALUATION_FILES = ("evaluation-inputs.json", "aggregate.json", *(
    f"validation-{profile}{suffix}" for profile in PROFILES for suffix in (".json.gz", ".receipt.json")))
MAX_BYTES = 128 * 1024 * 1024


def source_provenance():
    return {**implementation_fingerprints(), **trainer.source_provenance(), **{
        name: _sha((PYTHON_ROOT / name).read_bytes()) for name in (
            "triad_rl/evaluate_credit_pilot.py", "triad_rl/evaluate_anchored_pilot.py", "triad_rl/placement_policy.py")}}


def load_protocol(path):
    protocol, raw = trainer.load_protocol(path)
    validation = protocol.get("validation", {})
    if (validation.get("stage") != "validation" or validation.get("methods") != list(METHODS)
            or type(validation.get("episodes_per_profile")) is not int or not 1 <= validation["episodes_per_profile"] <= 200
            or type(validation.get("bootstrap_samples")) is not int or validation["bootstrap_samples"] != 2000
            or type(validation.get("bootstrap_seed")) is not int or validation["bootstrap_seed"] != 73042
            or set(validation.get("scenario_seed_ranges", {})) != set(PROFILES)
            or set(validation.get("run_seeds", {})) != set(PROFILES)):
        raise ValueError("Anchored validation differs from the bounded predeclared design")
    for profile, row in validation["scenario_seed_ranges"].items():
        if not _same(row, {"start": scenario_seed("validation", validation["run_seeds"][profile], 0),
                           "count": validation["episodes_per_profile"]}):
            raise ValueError("Validation scenario range differs from its declared slot")
        validate_seed_range(row["start"], row["count"], "validation", {
            key: value for key, value in validation["scenario_seed_ranges"].items() if key != profile})
    gate = {"primary_arm": PRIMARY, "comparators": ["no_critic_gradient", "balanced_v3"],
            "primary_metric": "timely_confirmed_fraction = 1 - breached_fraction",
            "minimum_equal_profile_gain": .01, "require_paired_ci_lower_above_zero": True,
            "maximum_profile_point_decline": .01, "require_equal_profile_all_threat_success_not_lower": True,
            "require_equal_profile_mean_return_not_lower": True}
    if not _same({key: protocol.get("scale_gate", {}).get(key) for key in gate}, gate):
        raise ValueError("Anchored sensing/return gate differs from its fixed contract")
    if not _same(protocol.get("evaluation_implementation_sha256"), source_provenance()):
        raise ValueError("Predeclared evaluation source differs")
    return protocol, raw


def layout_groups(reports):
    result = {}
    for reference in ("no_critic_gradient", "balanced_v3", "all_step_stop_pilot"):
        profiles = {}
        for profile, report in reports.items():
            groups = {}
            for a, b in zip(report["methods"][PRIMARY]["episodes"], report["methods"][reference]["episodes"], strict=True):
                left, right = a["placements"], b["placements"]
                key = ("unchanged_layout" if _same(left, right) else "initial_stop_flip" if bool(left) != bool(right)
                       else "same_first_placement_changed_later" if left and right and _same(left[0], right[0])
                       else "changed_first_placement")
                groups.setdefault(key, []).append({
                    "timely_fraction": b["metrics"]["breached_fraction"] - a["metrics"]["breached_fraction"],
                    "cost": a["metrics"]["cost"] - b["metrics"]["cost"],
                    "all_threat_success": float(a["metrics"]["success"]) - float(b["metrics"]["success"])})
            profiles[profile] = {key: {"pairs": len(rows), "mean_differences": {
                metric: float(np.mean([row[metric] for row in rows])) for metric in rows[0]}} for key, rows in groups.items()}
        result[reference] = profiles
    return {"scope": "Descriptive post-treatment groups, not causal effects or additional pass criteria", "comparisons": result}


def aggregate_reports(reports, protocol):
    if set(reports) != set(PROFILES) or any(
            set(r.get("methods", {})) != set(METHODS) or r.get("stage") != "validation" or r.get("profile") != p
            for p, r in reports.items()):
        raise ValueError("Need all three profiles and all eight methods")
    vectors = {p: {m: _vectors(r, m) for m in METHODS} for p, r in reports.items()}
    for report in reports.values():
        identities = [(r["seed"], r["scenario_sha256"]) for r in report["methods"][PRIMARY]["episodes"]]
        if not identities or any(not _same([(r["seed"], r["scenario_sha256"]) for r in report["methods"][m]["episodes"]], identities) for m in METHODS):
            raise ValueError("Scenario pairing differs between methods")
    methods = {m: {
        "profiles": {p: {k: float(v.mean()) for k, v in vectors[p][m].items()} for p in PROFILES},
        "equal_profile": {k: float(np.mean([vectors[p][m][k].mean() for p in PROFILES])) for k in METRICS}
    } for m in METHODS}
    comparisons = {}
    pairs = [(arm, ref) for arm in ARMS for ref in (*REFERENCES, "greedy_public")]
    pairs += [(PRIMARY, "no_critic_gradient"), (PRIMARY, "shared_critic")]
    for arm, ref in pairs:
        comparisons[f"{arm}_minus_{ref}"] = {k: stratified_interval(
            {p: vectors[p][arm][k] - vectors[p][ref][k] for p in PROFILES},
            samples=protocol["validation"]["bootstrap_samples"], seed=protocol["validation"]["bootstrap_seed"])
            for k in METRICS}
    spec, conditions = protocol["scale_gate"], {}
    for ref in spec["comparators"]:
        comparison = comparisons[f"{PRIMARY}_minus_{ref}"]
        timely = comparison["timely_fraction"]
        checks = {
            "minimum_timely_gain": timely["difference"] >= spec["minimum_equal_profile_gain"],
            "timely_ci_lower_above_zero": timely["lower95"] > 0.,
            "profile_point_safeguard": all(x >= -spec["maximum_profile_point_decline"] for x in timely["profile_differences"].values()),
            "all_threat_success_point_safeguard": comparison["all_threat_success"]["difference"] >= 0.,
            "mean_return_point_safeguard": comparison["return"]["difference"] >= 0.,
        }
        conditions[ref] = {"passed": all(checks.values()), "checks": checks}
    passed = all(row["passed"] for row in conditions.values())
    return {"schema": "triad.anchored_pilot_aggregate.v1", "stage": "validation", "final_test_accessed": False,
            "methods": methods, "paired_stratified_differences": comparisons,
            "scale_gate": {"primary_arm": PRIMARY, "comparators": conditions, "passed": passed,
                           "decision": "eligible_for_multiseed_replication" if passed else "do_not_scale_from_this_pilot",
                           "not_policy_promotion": True, "point_guards_are_not_noninferiority_tests": True},
            "exploratory_layout_groups": layout_groups(reports)}


def validate_report(report, profile, inputs, weights):
    interval = inputs["validation_ranges"][profile]
    if (report.get("schema") != SCHEMA or report.get("profile") != profile or report.get("stage") != "validation"
            or report.get("training_performed") is not False
            or report.get("protocol", {}).get("independent_final_test_evidence") is not False
            or report.get("anchored_inputs_sha256") != canonical_hash(inputs)
            or not _same(report.get("seed_provenance", {}).get("evaluation"), interval)
            or not _same(report.get("seed_provenance", {}).get("declared_lineage"), inputs["lineage"])
            or not _same(report.get("implementation_sha256"), implementation_fingerprints())
            or not _same(report.get("implementation_sha256_after"), implementation_fingerprints())
            or set(report.get("methods", {})) != set(METHODS)):
        raise ValueError("Profile scope, implementation or input binding differs")
    identities = None
    seeds = list(range(interval["start"], interval["start"] + interval["count"]))
    for name, method in report["methods"].items():
        rows = method["episodes"]
        hashes = [canonical_hash(row["scenario"]) for row in rows]
        if (not _same([row["seed"] for row in rows], seeds) or hashes != [row["scenario_sha256"] for row in rows]
                or identities is not None and hashes != identities or not _same(summarize(rows), method["summary"])
                or any(row.get("weights_sha256_before") != weights.get(name) or row.get("weights_sha256_after") != weights.get(name) for row in rows)):
            raise ValueError("Profile episode pairing, summary or frozen weights differ")
        identities = hashes
        _vectors(report, name)


def _new(path, data):
    with path.open("xb") as handle:
        handle.write(data)


def run_evaluation(runs, *, protocol_path, output, resume=False):
    if set(runs) != set(ARMS):
        raise ValueError("Exactly the three anchored experiment arms are required")
    protocol_path, output = Path(protocol_path).resolve(), Path(output).resolve()
    protocol, raw = load_protocol(protocol_path)
    lineage = load_published_lineage()
    reservations = sorted(protocol.get("reserved_final_tests_unopened", {}).values(), key=lambda r: r["start"])
    if (set(protocol.get("reserved_final_tests_unopened", {})) != set(PROFILES)
            or not _same(reservations, lineage["reserved_final_seed_ranges"])
            or protocol["prior_pilot_publication"]["sha256"] != lineage["prior_pilot"]["publication_manifest"]["sha256"]
            or protocol["prior_pilot_publication"]["manifest"] != lineage["prior_pilot"]["publication_manifest"]["path"]):
        raise ValueError("Prior pilot publication or reserved final tests differ")
    train = protocol["training"]
    exposure = {"inherited": lineage, "pilot_training": {"start": train["scenario_seed_start"], "count": train["episodes"]}}
    for row in protocol["validation"]["scenario_seed_ranges"].values():
        assert_disjoint(row["start"], row["count"], lineage, label="anchored validation")
        validate_seed_range(row["start"], row["count"], "validation", exposure)
    paths = {arm: Path(path).resolve() for arm, path in runs.items()}
    endpoints = {arm: trainer.validate_run(path, protocol_path, arm, lineage) for arm, path in paths.items()}
    if len({e[1]["first_batch_trajectory_sha256"] for e in endpoints.values()}) != 1:
        raise ValueError("First batch trajectories are not identical")
    if len({canonical_hash(e[1]["first_batch_coefficient_identity"]) for e in endpoints.values()}) != 1:
        raise ValueError("First batch reference coefficients are not identical")
    reference_files, weights = {}, {arm: p.weights_fingerprint() for arm, (p, _) in endpoints.items()}
    for name, (path, expected, kind) in REFERENCES.items():
        reference_files[name] = _checkpoint_files(BLUE_ROOT / path)
        if kind.load(BLUE_ROOT / path, feature_names=FEATURE_NAMES).weights_fingerprint() != expected:
            raise ValueError("Frozen reference weights differ")
        weights[name] = expected
    sources = source_provenance()
    if not _same(sources, protocol["evaluation_implementation_sha256"]):
        raise ValueError("Evaluation sources changed during preflight")
    inputs = {"schema": "triad.anchored_pilot_inputs.v1", "protocol_sha256": _sha(raw), "implementation_sha256": sources,
              "endpoint_evidence": {arm: evidence for arm, (_, evidence) in endpoints.items()},
              "reference_files_sha256": reference_files, "weights_sha256": weights,
              "lineage": exposure, "validation_ranges": protocol["validation"]["scenario_seed_ranges"]}

    def stable():
        if (protocol_path.read_bytes() != raw or not _same(source_provenance(), sources)
                or any({name: _sha(_file(paths[arm] / name)) for name in trainer.RUN_FILES} != evidence["run_files_sha256"] for arm, (_, evidence) in endpoints.items())
                or any(_checkpoint_files(BLUE_ROOT / path) != reference_files[name] for name, (path, _, _) in REFERENCES.items())):
            raise ValueError("Evaluation input or implementation changed")

    stable()
    if output.exists() and (not output.is_dir() or any(not p.is_file() or p.name not in EVALUATION_FILES for p in output.iterdir())):
        raise ValueError("Unexpected evaluation destination contents")
    lock = output / "evaluation-inputs.json"
    if output.exists() and any(output.iterdir()):
        if not resume or not lock.exists() or lock.read_bytes() != _bytes(inputs):
            raise ValueError("Explicit resume requires identical saved inputs")
    else:
        if resume:
            raise ValueError("No evaluation exists to resume")
        output.mkdir(parents=True, exist_ok=True)
        _new(lock, _bytes(inputs))
    cached = {}
    for profile in PROFILES:
        path, receipt_path = output / f"validation-{profile}.json.gz", output / f"validation-{profile}.receipt.json"
        if path.exists() or receipt_path.exists():
            if not resume or not path.exists() or not receipt_path.exists():
                raise ValueError("Incomplete or existing report will not be overwritten")
            data = _file(path)
            if not _same(_read(receipt_path), {"profile": profile, "sha256": _sha(data), "bytes": len(data), "inputs_sha256": canonical_hash(inputs)}):
                raise ValueError("Profile byte receipt differs")
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
                expanded = handle.read(MAX_BYTES + 1)
            if len(expanded) > MAX_BYTES:
                raise ValueError("Profile expanded size exceeds bound")
            report = trainer.shared._strict_json(expanded)
            validate_report(report, profile, inputs, weights)
            cached[profile] = report, data
    stable()
    factories = {"greedy_public": GreedyPublicCoverage}
    for name in METHODS[:-1]:
        path = paths[name] / "last" if name in paths else BLUE_ROOT / REFERENCES[name][0]
        kind = BalancedPolicy if name in paths else REFERENCES[name][2]
        def factory(seed, path=path, kind=kind, expected=weights[name]):
            policy = kind.load(path, feature_names=FEATURE_NAMES)
            if policy.weights_fingerprint() != expected:
                raise ValueError("Frozen actor changed between evaluation cases")
            return policy
        factories[name] = factory
    reports, receipts = {}, {}
    for profile in PROFILES:
        if profile in cached:
            report, data = cached[profile]
        else:
            row = inputs["validation_ranges"][profile]
            report = evaluate_robust_methods(factories, seed=row["start"], episodes=row["count"], profile=profile,
                                             stage="validation", replay_count=0, bootstrap_samples=2000, seed_provenance=exposure)
            report.update(schema=SCHEMA, anchored_inputs_sha256=canonical_hash(inputs))
            validate_report(report, profile, inputs, weights)
            data = gzip.compress(_bytes(report), mtime=0)
        stable()
        receipt = {"profile": profile, "sha256": _sha(data), "bytes": len(data), "inputs_sha256": canonical_hash(inputs)}
        if profile not in cached:
            _new(output / f"validation-{profile}.json.gz", data)
            _new(output / f"validation-{profile}.receipt.json", _bytes(receipt))
        reports[profile], receipts[profile] = report, receipt
        print(json.dumps({"profile": profile, "persisted": True}), flush=True)
    result = {**aggregate_reports(reports, protocol), "protocol_sha256": _sha(raw), "inputs_sha256": canonical_hash(inputs),
              "implementation_sha256": sources, "reports": receipts,
              "seed_provenance": {**exposure, "pilot_validation": inputs["validation_ranges"]}}
    stable()
    if not _same(load_published_lineage(), lineage):
        raise ValueError("Inherited evidence changed during evaluation")
    destination = output / "aggregate.json"
    if destination.exists():
        if not resume or destination.read_bytes() != _bytes(result):
            raise ValueError("Existing aggregate differs and will not be overwritten")
    else:
        _new(destination, _bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="ARM=RUN_DIRECTORY")
    parser.add_argument("--protocol", dest="protocol_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    runs = {}
    for item in args.pop("run"):
        arm, separator, path = item.partition("=")
        if not separator or arm not in ARMS or arm in runs or not path:
            parser.error("Run must be a unique known ARM=RUN_DIRECTORY")
        runs[arm] = Path(path)
    print(json.dumps(run_evaluation(runs, **args)["scale_gate"], indent=2))


if __name__ == "__main__":
    main()
