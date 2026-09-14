"""Select count-balanced Blue policies using paired development validation only.

The original physics, robust curriculum and generic evaluator remain frozen.
Every candidate, its preserved pre-training control and the fixed references
receive exactly the same scenarios. This command cannot open final-test seeds.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from evaluate_robust import (
    BLUE_ROOT, V1_WEIGHTS, checkpoint_files, evaluate_robust_methods,
    frozen_factory, implementation_fingerprints as robust_fingerprints,
    nested_seed_ranges, read_selection_report, validate_seed_range,
)
from triad_rl.adaptive_evaluation import (
    GreedyPublicCoverage, LegacyToy210, RandomLegal, UniformFixedRFAndRadar,
    canonical_hash, nearest_same_sensor, paired_bootstrap, summarize,
)
from triad_rl.adaptive_inputs import FEATURE_NAMES
from triad_rl.adaptive_policy import AdaptivePolicy
from triad_rl.balanced_policy import BalancedPolicy
from triad_rl.train_adaptive import scenario_seed
from triad_rl.train_balanced import TRAINING_SCHEMA, source_provenance, validation_seed_ranges


SCHEMA = "triad.balanced_model_selection.v1"
REPORT_SCHEMA = "triad.balanced_evaluation.v1"
PROTOCOL_SCHEMA = "triad.balanced_experiment_protocol.v1"
PROGRESS_SCHEMA = "triad.balanced_selection_progress.v1"
PROFILES = ("normal", "stress", "capability")
V2_WEIGHTS = "5c982c86a5da2e5b4960ba437021f751a36dc3af7187515903cf1b1ee2ab27ce"
V2_SELECTION_SHA256 = "01768ca0764ad3c9c1ed6a626ccfc26637b9271e35c77b6cc4ab1693073ebc69"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    json.dumps(value, allow_nan=False)
    return value


def implementation_fingerprints():
    directory = Path(__file__).resolve().parent
    extra = ("evaluate_balanced.py", "triad_rl/balanced_policy.py",
             "triad_rl/train_balanced.py", "triad_rl/placement_policy.py")
    return {**robust_fingerprints(), **{name: _sha(directory / name) for name in extra}}


def balanced_factory(path, expected):
    def factory(seed):
        actor = BalancedPolicy.load(path, feature_names=FEATURE_NAMES)
        if actor.weights_fingerprint() != expected:
            raise RuntimeError("Balanced checkpoint changed between evaluation episodes")
        return actor
    return factory


class FixedSingleModality:
    """Two fixed opposite sites, projected to the same sensor's legal sites."""

    def __init__(self, sensor_id, seed=0):
        if sensor_id not in ("rf", "radar"):
            raise ValueError("Fixed baseline supports only RF or radar")
        self.sensor_id, self.index, self.projections = sensor_id, 0, []
        self.description = (f"Fixed {sensor_id} at (0,100m) then (0,-100m), then STOP; "
                            "nearest legal same-sensor sites, skipping unavailable plans.")

    def act(self, observation, deterministic=True):
        positions = ([0., 100.], [0., -100.])
        while self.index < len(positions):
            position = positions[self.index]
            self.index += 1
            action, distance = nearest_same_sensor(observation, self.sensor_id, position)
            self.projections.append({"sensor_id": self.sensor_id, "requested_position": position,
                                     "distance_m": distance, "stopped_unavailable": False,
                                     "skipped_unavailable": distance is None})
            if distance is not None:
                return action
        return len(observation["action_mask"]) - 1


def _run_evidence(path, policy, seed, protocol):
    """Bind the best checkpoint to all training, including post-best exposure."""
    path = Path(path)
    if path.name != "best":
        raise ValueError("Candidate must be the run's validation-selected best checkpoint")
    run = path.parent
    config, summary = _read_json(run / "config.json"), _read_json(run / "summary.json")
    last = BalancedPolicy.load(run / "last", feature_names=FEATURE_NAMES)
    initial = BalancedPolicy.load(run / "initialized", feature_names=FEATURE_NAMES)
    expected, state = protocol["candidate_training"], policy.training_state
    episodes, batch = expected["episodes_per_run"], expected["batch_size"]
    mappings = {"batch_size": "batch_size", "learning_rate": "learning_rate",
                "entropy_coef": "entropy_coef", "gamma": "gamma",
                "validation_every": "validation_every",
                "validation_episodes": "validation_episodes_per_profile",
                "training_profile": "training_profile"}
    if (state.get("schema") != TRAINING_SCHEMA or summary.get("schema") != TRAINING_SCHEMA
            or config.get("schema") != TRAINING_SCHEMA or config.get("seed") != seed
            or config.get("target_episodes") != episodes or episodes % batch
            or any(config.get(k) != expected[v] for k, v in mappings.items())
            or config.get("value_coef") != expected.get("value_coef", .5)
            or config.get("max_grad_norm") != expected.get("max_grad_norm", 1.)
            or config.get("source_sha256") != source_provenance()
            or state.get("config") != last.training_state.get("config")
            or any(config.get(k) != v for k, v in state.get("config", {}).items())):
        raise ValueError("Candidate training configuration/source differs from the declared protocol")
    config_hash = state.get("config_sha256")
    if (not config_hash or canonical_hash(state["config"]) != config_hash
            or config.get("config_sha256") != config_hash
            or last.training_state.get("config_sha256") != config_hash
            or summary.get("config_sha256") != config_hash
            or summary.get("completed_episodes") != episodes
            or last.training_state.get("completed_episodes") != episodes
            or summary.get("optimizer_updates") != episodes // batch
            or last.update_count != episodes // batch
            or summary.get("last_weights_sha256") != last.weights_fingerprint()
            or summary.get("best_weights_sha256") != policy.weights_fingerprint()
            or state.get("completed_episodes") != summary.get("best_episode")
            or state.get("best_episode") != summary.get("best_episode")
            or summary.get("best_validation") != state.get("best_validation")
            or last.training_state.get("best_episode") != summary.get("best_episode")
            or last.training_state.get("best_validation") != state.get("best_validation")
            or policy.update_count != state["completed_episodes"] // batch
            or last.training_state.get("best_files_sha256") != checkpoint_files(path)):
        raise ValueError("Candidate does not match a completed run and its selected checkpoint")
    initialization = config.get("initialization", {})
    v2 = AdaptivePolicy.load(BLUE_ROOT / "Checkpoints" / "robust-v2-candidate", feature_names=FEATURE_NAMES)
    selection_path = BLUE_ROOT / "Results" / "robust-v2" / "selection.json"
    selection_lineage, _, _ = read_selection_report(selection_path, V2_WEIGHTS)
    if (initialization.get("kind") != "transferred_adaptive_weights"
            or initialization.get("source_weights_sha256") != V2_WEIGHTS
            or v2.weights_fingerprint() != V2_WEIGHTS
            or initialization.get("source_update_count") != v2.update_count
            or initialization.get("source_policy_schema") != v2.metadata["policy_schema"]
            or initialization.get("source_seed_provenance") != v2.training_state.get("seed_provenance")
            or initialization.get("weights_sha256") != initial.weights_fingerprint()
            or initial.update_count != 0 or initial.training_state.get("completed_episodes") != 0
            or initial.training_state.get("config_sha256") != config_hash
            or last.training_state.get("initialized_files_sha256") != checkpoint_files(run / "initialized")
            or set(initial.parameters) != set(v2.parameters)
            or any(not np.array_equal(initial.parameters[k], v2.parameters[k]) for k in v2.parameters)
            or initialization.get("source_selection", {}).get("report_sha256") != V2_SELECTION_SHA256
            or _sha(selection_path) != V2_SELECTION_SHA256
            or initialization.get("source_selection", {}).get("seed_provenance") != selection_lineage
            or initialization.get("source_files_sha256") != checkpoint_files(BLUE_ROOT / "Checkpoints" / "robust-v2-candidate")
            or initial.training_state.get("initialization") != initialization
            or policy.training_state.get("initialization") != initialization
            or last.training_state.get("initialization") != initialization
            or summary.get("initialization") != initialization):
        raise ValueError("Candidate initialization is not the declared frozen-v2 transfer with full selection lineage")
    full_train = {"start": scenario_seed("train", seed, 0), "count": episodes}
    declared_train = [{"start": row["start"], "count": row["count"]}
                      for row in expected["training_seed_ranges"]]
    validation = validation_seed_ranges(config["validation_run_seed"], config["validation_episodes"])
    declared_validation = {row["profile"]: {"start": row["start"], "count": row["count"]}
                           for row in expected["validation_seed_ranges"]}
    exposure = last.training_state.get("seed_provenance", {})
    if (full_train not in declared_train or validation != declared_validation
            or exposure.get("training") != full_train or exposure.get("validation") != validation
            or exposure.get("inherited_selection") != selection_lineage
            or exposure.get("inherited_training") != v2.training_state.get("seed_provenance")
            or summary.get("seed_provenance") != exposure):
        raise ValueError("Run seed exposure differs from the declared protocol or transfer lineage")
    files = {name: _sha(run / name) for name in ("config.json", "summary.json", "training.jsonl")}
    if files["training.jsonl"] != last.training_state.get("log_sha256"):
        raise ValueError("Completed training log changed after its last checkpoint")
    events = [json.loads(line) for line in (run / "training.jsonl").read_text(encoding="utf-8").splitlines()]
    json.dumps(events, allow_nan=False)
    batches = [row for row in events if row.get("event") == "training_batch"]
    if ([(row.get("episode_start"), row.get("episode")) for row in batches]
            != [(start, start + batch) for start in range(0, episodes, batch)]):
        raise ValueError("Training log does not cover the whole declared run exactly once")
    if any(row.get("event") not in ("training_batch", "validation", "resume") for row in events):
        raise ValueError("Training log contains an unknown event")
    validations = [row for row in events if row.get("event") == "validation"]
    cadence = config["validation_every"]
    expected_events = [0] + [end for end in range(batch, episodes + 1, batch)
                             if end // cadence > (end - batch) // cadence]
    if [row.get("episode") for row in validations] != expected_events:
        raise ValueError("Training log validation cadence differs from the declared protocol")
    selected_event, best_score = None, -float("inf")
    for row in validations:
        profiles = row.get("profiles", {})
        if set(profiles) != set(PROFILES) or row.get("seed_ranges") != validation:
            raise ValueError("Validation log is missing a profile or declared seed range")
        score = sum(profiles[p]["mean_return"] for p in PROFILES) / len(PROFILES)
        if not np.isclose(score, row.get("balanced_mean_return"), atol=1e-12, rtol=1e-12):
            raise ValueError("Validation score differs from the equal-profile means")
        is_selected = row["balanced_mean_return"] > best_score
        if row.get("selected") is not is_selected:
            raise ValueError("Logged checkpoint selection is not strict-greater equal-profile return")
        if is_selected:
            best_score, selected_event = row["balanced_mean_return"], row
    if (selected_event is None or selected_event["episode"] != summary["best_episode"]
            or selected_event["weights_sha256"] != policy.weights_fingerprint()
            or validations[0].get("weights_sha256") != initial.weights_fingerprint()
            or summary["best_validation"] != {key: selected_event[key] for key in (
                "profiles", "balanced_mean_return", "balanced_success_rate", "seed_ranges")}):
        raise ValueError("Best checkpoint differs from the complete validation selection trace")
    for checkpoint in ("best", "last", "initialized"):
        files.update({f"{checkpoint}/{name}": digest
                      for name, digest in checkpoint_files(run / checkpoint).items()})
    return {"last_checkpoint_metadata": deepcopy(last.metadata), "run_summary": summary,
            "run_files_sha256": files, "initialized_weights_sha256": initial.weights_fingerprint()}


def _reference_evidence():
    """Verify the published v2 bundle and bind its complete selection exposure."""
    root = BLUE_ROOT.parent.parent
    manifest_path = BLUE_ROOT / "Results" / "robust-v2" / "artifact-manifest.json"
    manifest = _read_json(manifest_path)
    if (manifest.get("schema") != "triad.robust_publication.v1"
            or manifest.get("selected_weights_sha256") != V2_WEIGHTS
            or manifest.get("selection_sha256") != V2_SELECTION_SHA256):
        raise ValueError("Frozen v2 publication manifest changed")
    files = {"Results/robust-v2/artifact-manifest.json": _sha(manifest_path)}
    for entry in manifest["files"]:
        path = (BLUE_ROOT / entry["path"]).resolve()
        if BLUE_ROOT.resolve() not in path.parents:
            raise ValueError("Reference artifact escapes the Blue repository")
        if _sha(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError("Frozen v2 published artifact changed")
        files[entry["path"]] = entry["sha256"]
    for name, digest in manifest["source_files_sha256"].items():
        path = (root / name).resolve()
        if root.resolve() not in path.parents or _sha(path) != digest:
            raise ValueError("Frozen v2 source changed")
    selection_path = BLUE_ROOT / "Results" / "robust-v2" / "selection.json"
    lineage, selection, _ = read_selection_report(selection_path, V2_WEIGHTS)
    if _sha(selection_path) != V2_SELECTION_SHA256:
        raise ValueError("Frozen v2 selection changed")
    old_path = BLUE_ROOT / "Results" / "adaptive-v1" / "common-validation-selection.json"
    old = _read_json(old_path)
    if (old.get("schema") != "triad.adaptive_model_selection.v1"
            or old.get("selected", {}).get("weights_sha256") != V1_WEIGHTS):
        raise ValueError("Frozen v1 selection differs from the reference")
    files["Results/adaptive-v1/common-validation-selection.json"] = _sha(old_path)
    return {"published_v2_selection": lineage, "published_v1_selection": old["seed_provenance"]}, {
        "publication_files_sha256": files, "v2_source_files_sha256": manifest["source_files_sha256"],
        "v2_selection": selection}


def select_from_reports(reports, candidates):
    """Recompute paired summaries and rank by equal-profile mean return."""
    if set(reports) != set(PROFILES) or not candidates:
        raise ValueError("Selection requires candidates and all three profiles")
    sources, methods = None, None
    for profile, report in reports.items():
        if (report.get("stage") != "validation" or report.get("profile") != profile
                or report.get("schema") != REPORT_SCHEMA
                or report.get("protocol", {}).get("independent_final_test_evidence") is not False):
            raise ValueError("Selection accepts only balanced validation evidence")
        if sources is None:
            sources, methods = report["implementation_sha256"], set(report["methods"])
        if (report["implementation_sha256"] != sources or report.get("implementation_sha256_after") != sources
                or set(report["methods"]) != methods):
            raise ValueError("Candidate selection spans different implementations or method sets")
        interval = report["seed_provenance"]["evaluation"]
        validate_seed_range(interval["start"], interval["count"], "validation")
        expected_seeds = list(range(interval["start"], interval["start"] + interval["count"]))
        paired_hashes = None
        for method in report["methods"].values():
            records = method["episodes"]
            if [row["seed"] for row in records] != expected_seeds:
                raise ValueError("Selection requires complete ordered evaluation episodes")
            hashes = [row["scenario_sha256"] for row in records]
            if any(canonical_hash(row["scenario"]) != digest for row, digest in zip(records, hashes)):
                raise ValueError("Selection scenario fingerprint mismatch")
            if paired_hashes is None:
                paired_hashes = hashes
            if hashes != paired_hashes or summarize(records) != method["summary"]:
                raise ValueError("Selection requires paired scenarios and recomputed summaries")
    scores = []
    for seed, candidate in sorted(candidates.items()):
        summaries = {}
        for profile, report in reports.items():
            for name, fingerprint in ((f"candidate_{seed}", candidate["weights_sha256"]),
                                      (f"initialized_{seed}", candidate["initialized_weights_sha256"])):
                if name not in report["methods"]:
                    raise ValueError("Candidate or matched initialized control is missing")
                if any(row["weights_sha256_before"] != fingerprint or row["weights_sha256_after"] != fingerprint
                       for row in report["methods"][name]["episodes"]):
                    raise ValueError("Candidate or initialized-control selection weight mismatch")
            summaries[profile] = deepcopy(report["methods"][f"candidate_{seed}"]["summary"])
        score = sum(summaries[p]["mean_return"] for p in PROFILES) / len(PROFILES)
        json.dumps(score, allow_nan=False)
        scores.append({"seed": seed, "weights_sha256": candidate["weights_sha256"],
                       "balanced_mean_return": score, "profiles": summaries})
    return deepcopy(max(scores, key=lambda row: row["balanced_mean_return"])), scores


def _selected_comparisons(report, selected_name, samples):
    reference = report["methods"][selected_name]["episodes"]
    result = {}
    for name, method in report["methods"].items():
        if name == selected_name:
            continue
        metrics = {}
        for key in reference[0]["metrics"]:
            a = [row["metrics"].get(key) for row in reference]
            b = [row["metrics"].get(key) for row in method["episodes"]]
            if all(isinstance(value, (int, float, bool)) for value in a + b) and np.isfinite(a + b).all():
                metrics[key] = paired_bootstrap(a, b, seed=report["seed_provenance"]["evaluation"]["start"] % 2**32,
                                                samples=samples)
        result[f"{selected_name}_minus_{name}"] = metrics
    return result


def _prepare_context(candidates, protocol_path, bootstrap_samples):
    """Read-only preflight; the persisted binding contains no workstation paths."""
    protocol_path = Path(protocol_path).resolve()
    protocol_bytes = protocol_path.read_bytes()
    protocol = _read_json(protocol_path)
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("Unsupported balanced experiment protocol")
    frozen = protocol.get("frozen_reference", {})
    if (frozen.get("weights_sha256") != V2_WEIGHTS
            or frozen.get("checkpoint") != "Checkpoints/robust-v2-candidate"
            or frozen.get("selection_report") != "Results/robust-v2/selection.json"):
        raise ValueError("Protocol must declare the preserved robust-v2 transfer reference")
    if sorted(candidates) != sorted(protocol["candidate_training"]["run_seeds"]) or not candidates:
        raise ValueError("Candidates differ from predeclared run seeds")
    entries = protocol["candidate_selection"]["seed_ranges"]
    if len(entries) != 3 or {row["profile"] for row in entries} != set(PROFILES):
        raise ValueError("Protocol requires exactly one range per profile")
    ranges = {row["profile"]: row for row in entries}
    for row in ranges.values():
        validate_seed_range(row["start"], row["count"], "validation")
    paths = {seed: Path(candidates[seed]).resolve() for seed in sorted(candidates)}
    policies = {seed: BalancedPolicy.load(path, feature_names=FEATURE_NAMES) for seed, path in paths.items()}
    info = {}
    for seed, policy in policies.items():
        evidence = _run_evidence(paths[seed], policy, seed, protocol)
        info[seed] = {"weights_sha256": policy.weights_fingerprint(), "metadata": deepcopy(policy.metadata),
                      "checkpoint_files_sha256": checkpoint_files(paths[seed]),
                      "initialized_weights_sha256": evidence["initialized_weights_sha256"],
                      "complete_run": evidence}
    old_lineage, old_evidence = _reference_evidence()
    reference_paths = {"robust_v2": BLUE_ROOT / "Checkpoints" / "robust-v2-candidate",
                       "adaptive_v1": BLUE_ROOT / "Checkpoints" / "adaptive-v1"}
    references = {name: AdaptivePolicy.load(path, feature_names=FEATURE_NAMES) for name, path in reference_paths.items()}
    expected = {"robust_v2": V2_WEIGHTS, "adaptive_v1": V1_WEIGHTS}
    if any(policy.weights_fingerprint() != expected[name] for name, policy in references.items()):
        raise ValueError("Frozen reference weights changed")
    reference_files = {name: checkpoint_files(path) for name, path in reference_paths.items()}
    legacy_path = BLUE_ROOT / "Checkpoints" / "toy-210"
    reference_files["legacy_toy210_projected"] = checkpoint_files(legacy_path)
    lineage = {**old_lineage, "reference_checkpoints": {name: deepcopy(policy.metadata) for name, policy in references.items()},
               "candidate_checkpoints": {str(seed): row["metadata"] for seed, row in info.items()},
               "complete_training_runs": {str(seed): row["complete_run"]["last_checkpoint_metadata"] for seed, row in info.items()}}
    nested_seed_ranges(lineage)
    for profile, row in ranges.items():
        validate_seed_range(row["start"], row["count"], "validation", lineage)
        validate_seed_range(row["start"], row["count"], "validation", {p: r for p, r in ranges.items() if p != profile})
    sources = implementation_fingerprints()
    binding = {"schema": PROGRESS_SCHEMA, "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
               "implementation_sha256": sources, "bootstrap_samples": bootstrap_samples,
               "candidate_metadata": {str(seed): row for seed, row in info.items()},
               "reference_evidence": old_evidence, "reference_checkpoint_files_sha256": reference_files,
               "seed_provenance": {"lineage": lineage, "selection_validation": list(ranges.values())}}
    context = {"paths": paths, "info": info, "protocol_path": protocol_path,
               "protocol_bytes": protocol_bytes, "protocol": protocol, "ranges": ranges,
               "reference_paths": reference_paths, "reference_files": reference_files,
               "old_lineage": old_lineage, "old_evidence": old_evidence, "lineage": lineage,
               "sources": sources, "bootstrap_samples": bootstrap_samples,
               "binding": binding, "binding_sha256": canonical_hash(binding)}
    _assert_unchanged(context)
    return context


def _assert_unchanged(context):
    """Recheck full frozen inputs before sampling and each durable write."""
    c = context
    if (implementation_fingerprints() != c["sources"]
            or c["protocol_path"].read_bytes() != c["protocol_bytes"]
            or _reference_evidence() != (c["old_lineage"], c["old_evidence"])
            or any(_run_evidence(path, BalancedPolicy.load(path, feature_names=FEATURE_NAMES), seed, c["protocol"])
                   != c["info"][seed]["complete_run"] for seed, path in c["paths"].items())
            or any(checkpoint_files(path) != c["info"][seed]["checkpoint_files_sha256"]
                   for seed, path in c["paths"].items())
            or any(checkpoint_files(path) != c["reference_files"][name] for name, path in c["reference_paths"].items())
            or checkpoint_files(BLUE_ROOT / "Checkpoints" / "toy-210") != c["reference_files"]["legacy_toy210_projected"]):
        raise RuntimeError("Selection implementation, protocol, completed run or reference changed")


def _factories(context):
    """Created inside each worker; closures never cross process boundaries."""
    paths, info = context["paths"], context["info"]
    reference_paths = context["reference_paths"]
    expected = {"robust_v2": V2_WEIGHTS, "adaptive_v1": V1_WEIGHTS}
    legacy_path = BLUE_ROOT / "Checkpoints" / "toy-210"
    factories = {f"candidate_{seed}": balanced_factory(paths[seed], info[seed]["weights_sha256"]) for seed in sorted(paths)}
    factories.update({f"initialized_{seed}": balanced_factory(paths[seed].parent / "initialized", info[seed]["initialized_weights_sha256"])
                      for seed in sorted(paths)})
    factories.update({name: frozen_factory(reference_paths[name], fingerprint) for name, fingerprint in expected.items()})
    factories.update({"greedy_public": GreedyPublicCoverage, "random_legal": RandomLegal,
                      "uniform_fixed_rf_radar": UniformFixedRFAndRadar,
                      "fixed_rf": lambda seed: FixedSingleModality("rf", seed),
                      "fixed_radar": lambda seed: FixedSingleModality("radar", seed),
                      "legacy_toy210_projected": lambda seed: LegacyToy210(legacy_path, seed=seed)})
    return factories


def _profile_worker(profile, context):
    """Top-level picklable task: one complete, independently paired profile."""
    _assert_unchanged(context)
    row = context["ranges"][profile]
    report = evaluate_robust_methods(_factories(context), profile=profile, stage="validation", seed=row["start"],
                                     episodes=row["count"], replay_count=0,
                                     bootstrap_samples=context["bootstrap_samples"], seed_provenance=context["lineage"])
    report["schema"] = REPORT_SCHEMA
    report["implementation_sha256"] = context["sources"]
    report["implementation_sha256_after"] = implementation_fingerprints()
    report["resume_binding_sha256"] = context["binding_sha256"]
    report["protocol"]["initialized_controls"] = "Each run's preserved transferred-v2 parameters under balanced inference; not untrained"
    report["reference_evidence"] = context["old_evidence"]
    for seed in context["paths"]:
        report["methods"][f"initialized_{seed}"]["metadata"].update({
            "is_claimed_untrained": False, "is_preserved_before_balanced_training": True,
            "source_adaptive_weights_sha256": V2_WEIGHTS})
    _assert_unchanged(context)
    return report


def _final_artifacts(reports, context):
    """Deterministic final output, independent of worker count/completion order."""
    reports = {profile: deepcopy(reports[profile]) for profile in PROFILES}
    info, ranges, lineage = context["info"], context["ranges"], context["lineage"]
    bootstrap_samples = context["bootstrap_samples"]
    selected, scores = select_from_reports(reports, info)
    selected_name = f"candidate_{selected['seed']}"
    for report in reports.values():
        report["selected_candidate_paired_differences"] = _selected_comparisons(report, selected_name, bootstrap_samples)
        report["protocol"]["post_selection_intervals"] = "Descriptive paired scenario bootstrap, not selection-adjusted independent-test intervals"
    summary = {"schema": SCHEMA, "stage": "validation", "final_test_accessed": False,
               "criterion": "maximum equally weighted normal/stress/capability mean return; ascending-seed ties",
               "selected": selected, "candidates": scores,
               "candidate_metadata": {str(seed): row for seed, row in info.items()},
               "seed_provenance": {"lineage": lineage, "selection_validation": list(ranges.values())},
               "protocol_sha256": context["binding"]["protocol_sha256"],
               "implementation_sha256": context["sources"], "reference_evidence": context["old_evidence"],
               "reference_checkpoint_files_sha256": context["reference_files"],
               "reference_summaries": {profile: {name: method["summary"] for name, method in report["methods"].items()
                                                  if not name.startswith("candidate_")}
                                       for profile, report in reports.items()}, "reports": {}}
    encoded = {}
    for profile, report in reports.items():
        data = (json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        compressed = gzip.compress(data, mtime=0)
        filename = f"common-validation-{profile}.json.gz"
        encoded[filename] = compressed
        summary["reports"][profile] = {"path": filename, "sha256": hashlib.sha256(compressed).hexdigest(), "bytes": len(compressed)}
    encoded["selection.json"] = (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return summary, encoded


def _canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _strict_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key in selection recovery evidence")
            result[key] = value
        return result
    result = json.loads(data, object_pairs_hook=unique)
    json.dumps(result, allow_nan=False)
    return result


def _atomic_write(path, data):
    """Atomic replace with fsync; incomplete owned temporaries are never reused."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _profile_filename(profile):
    return f"completed-validation-{profile}.json.gz"


def _validate_completed_report(report, profile, context):
    """Never trust hashes alone: recheck pairing, weights, summaries and CIs."""
    row, c = context["ranges"][profile], context
    interval = {"start": row["start"], "count": row["count"]}
    if (report.get("schema") != REPORT_SCHEMA or report.get("stage") != "validation"
            or report.get("profile") != profile or report.get("training_performed") is not False
            or report.get("resume_binding_sha256") != c["binding_sha256"]
            or report.get("implementation_sha256") != c["sources"]
            or report.get("implementation_sha256_after") != c["sources"]
            or report.get("reference_evidence") != c["old_evidence"]
            or report.get("seed_provenance", {}).get("evaluation") != interval
            or report.get("seed_provenance", {}).get("declared_lineage") != c["lineage"]
            or report.get("protocol", {}).get("independent_final_test_evidence") is not False
            or "selected_candidate_paired_differences" in report
            or set(report.get("methods", {})) != set(_factories(c))):
        raise ValueError("Completed profile metadata/source/seed binding differs from this experiment")
    expected_weights = {f"candidate_{seed}": info["weights_sha256"] for seed, info in c["info"].items()}
    expected_weights.update({f"initialized_{seed}": info["initialized_weights_sha256"] for seed, info in c["info"].items()})
    from triad_rl.adaptive_evaluation import LEGACY_PARAMETER_SHA256
    expected_weights.update(robust_v2=V2_WEIGHTS, adaptive_v1=V1_WEIGHTS, legacy_toy210_projected=LEGACY_PARAMETER_SHA256)
    seeds = list(range(row["start"], row["start"] + row["count"]))
    expected_groups = {"weather", "rain", "illumination", "swarm_size", "emitter", "path", "altitude", "speed", "curriculum_profile"}
    expected_metrics = {"blind_spot_fraction", "breach_rate", "breached_fraction", "budget_fraction_spent",
                        "confirmed_fraction", "cost", "coverage", "detected_fraction", "detection_rate",
                        "early_detection", "invalid_actions", "mean_confirmation_time_censored", "return",
                        "reward", "sensors_placed", "success", "total_cost"}
    paired_hashes = None
    for name, method in report["methods"].items():
        records = method["episodes"]
        hashes = [record["scenario_sha256"] for record in records]
        if (not records or [record["seed"] for record in records] != seeds
                or any(canonical_hash(record["scenario"]) != digest for record, digest in zip(records, hashes))
                or any(set(record["metrics"]) != expected_metrics for record in records)
                or summarize(records) != method["summary"]):
            raise ValueError("Completed profile has incomplete/tampered episodes, scenarios or summary metrics")
        if paired_hashes is None:
            paired_hashes = hashes
        if hashes != paired_hashes:
            raise ValueError("Completed profile is not paired across methods")
        if name in expected_weights and any(record[key] != expected_weights[name] for record in records
                                            for key in ("weights_sha256_before", "weights_sha256_after")):
            raise ValueError("Completed profile weights differ from the frozen candidate/control/reference")
        if set(method.get("subgroups", {})) != expected_groups:
            raise ValueError("Completed profile is missing required subgroup metric categories")
        for category, groups in method["subgroups"].items():
            labels = [str(record["scenario"]["curriculum_metadata"].get("profile", profile))
                      if category == "curriculum_profile" else record["groups"][category] for record in records]
            expected_group_values = {label: summarize([record for record, actual in zip(records, labels) if actual == label])
                                     for label in sorted(set(labels))}
            if groups != expected_group_values:
                raise ValueError("Completed profile subgroup metrics differ from raw episodes")
    if report.get("scenario_sequence_sha256") != canonical_hash(paired_hashes):
        raise ValueError("Completed profile scenario sequence fingerprint differs")
    if report.get("paired_differences") != _selected_comparisons(report, f"candidate_{min(c['paths'])}", c["bootstrap_samples"]):
        raise ValueError("Completed profile paired comparison metrics differ from raw episodes")


def _profile_entry(profile, data, report):
    return {"path": _profile_filename(profile), "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data), "report_sha256": canonical_hash(report),
            "seed_range": report["seed_provenance"]["evaluation"]}


def _progress(context, completed):
    return {"schema": PROGRESS_SCHEMA, "stage": "validation", "final_test_accessed": False,
            "experiment": context["binding"], "experiment_sha256": context["binding_sha256"],
            "completed_profiles": {p: completed[p] for p in PROFILES if p in completed}}


def _load_progress(output, context):
    path = output / "progress.json"
    if not path.is_file():
        raise ValueError("Explicit resume requires this evaluation's progress.json")
    progress = _strict_json(path.read_bytes())
    if (progress.get("schema") != PROGRESS_SCHEMA or progress.get("stage") != "validation"
            or progress.get("final_test_accessed") is not False
            or progress.get("experiment") != context["binding"]
            or progress.get("experiment_sha256") != context["binding_sha256"]):
        raise ValueError("Resume source/protocol/checkpoint/statistical configuration binding changed")
    declared = progress.get("completed_profiles")
    if not isinstance(declared, dict) or not set(declared).issubset(PROFILES):
        raise ValueError("Invalid completed-profile progress metadata")
    # A crash can leave an owned atomic-write temporary. Keep it untouched;
    # never treat partial bytes as completed evidence or remove user data.
    names = {"progress.json", "selection.json", *(_profile_filename(p) for p in PROFILES),
             *(f"common-validation-{p}.json.gz" for p in PROFILES)}
    temporary = re.compile(r"^\.(?:" + "|".join(re.escape(name) for name in names) + r")\.[A-Za-z0-9_-]+\.tmp$")
    for item in output.iterdir():
        if not item.is_file() or (item.name not in names and not temporary.fullmatch(item.name)):
            raise ValueError("Resume directory contains an unrecognized artifact")
    reports, completed = {}, {}
    for profile in PROFILES:
        target = output / _profile_filename(profile)
        if profile in declared and not target.is_file():
            raise ValueError("A committed completed-profile snapshot is missing")
        if not target.exists():
            continue
        data = target.read_bytes()
        if profile in declared and (hashlib.sha256(data).hexdigest() != declared[profile].get("sha256")
                                     or len(data) != declared[profile].get("bytes")):
            raise ValueError("Completed profile compressed bytes differ from progress checksum")
        # Bound expanded recovery input; a corrupt gzip must not exhaust RAM.
        with gzip.open(target, "rb") as handle:
            decoded = handle.read(256 * 1024 * 1024 + 1)
        if len(decoded) > 256 * 1024 * 1024:
            raise ValueError("Completed profile exceeds recovery size limit")
        report = _strict_json(decoded)
        _validate_completed_report(report, profile, context)
        entry = _profile_entry(profile, data, report)
        if profile in declared and entry != declared[profile]:
            raise ValueError("Completed profile data/metadata differs from progress")
        # A complete atomic profile may precede its progress update at crash.
        # Its embedded experiment binding and recomputed statistics make this
        # orphan recoverable without re-running its scenarios.
        reports[profile], completed[profile] = report, entry
    if (output / "selection.json").exists() and len(reports) != len(PROFILES):
        raise ValueError("Final selection exists without all committed profile evidence")
    _assert_unchanged(context)
    if completed != declared:
        _atomic_write(path, _canonical_bytes(_progress(context, completed)))
    return reports, completed


def run_selection(candidates, *, protocol_path, output, bootstrap_samples=2000, workers=1, resume=False):
    """Run/resume profile-atomic selection; worker count never changes evidence."""
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 3:
        raise ValueError("workers must be an integer in [1, 3]")
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or not 1 <= bootstrap_samples <= 10000:
        raise ValueError("bootstrap_samples must be an integer in [1, 10000]")
    if not isinstance(resume, bool):
        raise ValueError("resume must be explicit boolean")
    output = Path(output).resolve()
    if output.exists() and not output.is_dir():
        raise ValueError("Selection output must be a directory")
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError("Selection output must be an empty directory; use explicit --resume")
    context = _prepare_context(candidates, protocol_path, bootstrap_samples)
    reports, completed = _load_progress(output, context) if resume else ({}, {})
    if resume:
        print(json.dumps({"resumed_profiles": [p for p in PROFILES if p in reports],
                          "pending_profiles": [p for p in PROFILES if p not in reports]}), flush=True)

    def persist(profile, report):
        _validate_completed_report(report, profile, context)
        _assert_unchanged(context)
        data = gzip.compress(_canonical_bytes(report), mtime=0)
        output.mkdir(parents=True, exist_ok=True)
        if not (output / "progress.json").exists():
            _atomic_write(output / "progress.json", _canonical_bytes(_progress(context, completed)))
        target = output / _profile_filename(profile)
        if target.exists() and target.read_bytes() != data:
            raise ValueError("Completed profile destination contains conflicting data")
        if not target.exists():
            _atomic_write(target, data)
        reports[profile], completed[profile] = report, _profile_entry(profile, data, report)
        _assert_unchanged(context)
        _atomic_write(output / "progress.json", _canonical_bytes(_progress(context, completed)))
        print(json.dumps({"profile": profile, "persisted": True,
                          "summaries": {name: method["summary"] for name, method in report["methods"].items()}}), flush=True)

    pending = [p for p in PROFILES if p not in reports]
    if workers == 1:
        for profile in pending:
            persist(profile, _profile_worker(profile, context))
    elif pending:
        # spawn has consistent semantics on Windows/Linux and never forks a
        # live NumPy/BLAS runtime. Only plain context and top-level functions
        # cross process boundaries; actors are constructed inside each worker.
        with ProcessPoolExecutor(max_workers=min(workers, len(pending)),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_profile_worker, p, context): p for p in pending}
            try:
                for future in as_completed(futures):
                    persist(futures[future], future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    _assert_unchanged(context)
    summary, artifacts = _final_artifacts(reports, context)
    # On interruption during finalization, exact matching files are reused.
    # Differing final artifacts are corruption, never silently overwritten.
    for name, data in artifacts.items():
        path = output / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError("Existing final selection/report differs from verified completed profiles")
    _assert_unchanged(context)
    for name, data in artifacts.items():
        if not (output / name).exists():
            _atomic_write(output / name, data)
    selected = summary["selected"]
    print(json.dumps({"selected_seed": selected["seed"], "weights_sha256": selected["weights_sha256"],
                      "balanced_mean_return": selected["balanced_mean_return"]}), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True, metavar="SEED=RUN/best")
    parser.add_argument("--protocol", type=Path, default=BLUE_ROOT / "Results" / "balanced-v3" / "protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=1, help="Independent profile processes (1-3); does not alter statistical design")
    parser.add_argument("--resume", action="store_true", help="Verify and reuse this output directory's completed profile snapshots")
    args = parser.parse_args(argv)
    candidates = {}
    for item in args.candidate:
        seed, separator, path = item.partition("=")
        if not separator or not seed.isdigit() or not path or int(seed) in candidates:
            parser.error("Each --candidate must be a unique SEED=RUN/best pair")
        candidates[int(seed)] = Path(path)
    run_selection(candidates, protocol_path=args.protocol, output=args.output, bootstrap_samples=args.bootstrap_samples,
                  workers=args.workers, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
