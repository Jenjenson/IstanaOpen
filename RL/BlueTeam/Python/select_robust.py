"""Select frozen robust candidates on common validation cases, never test cases.

All candidates and two reference methods face each profile's identical cases.
The selected weight hash and every exposed seed range travel with the result.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

from evaluate_robust import (
    BLUE_ROOT, V1_WEIGHTS, checkpoint_files, evaluate_robust_methods,
    frozen_factory, implementation_fingerprints, validate_seed_range,
)
from triad_rl.adaptive_evaluation import GreedyPublicCoverage, canonical_hash, summarize
from triad_rl.adaptive_inputs import FEATURE_NAMES
from triad_rl.adaptive_policy import AdaptivePolicy
from triad_rl.train_adaptive import scenario_seed
from triad_rl.train_robust import TRAINING_SCHEMA, source_provenance, validation_seed_ranges


SCHEMA = "triad.robust_model_selection.v1"
PROFILES = ("normal", "stress", "capability")


def _run_evidence(path, policy, seed, protocol):
    """Verify a selected checkpoint belongs to a complete, declared run."""
    if path.name != "best":
        raise ValueError("Candidate must be the run's validation-selected best checkpoint")
    run = path.parent
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    last = AdaptivePolicy.load(run / "last", feature_names=FEATURE_NAMES)
    initial = AdaptivePolicy.load(run / "initialized", feature_names=FEATURE_NAMES)
    expected = protocol["candidate_training"]
    episodes = expected["episodes_per_run"]
    state = policy.training_state
    mappings = {"batch_size": "batch_size", "learning_rate": "learning_rate",
                "entropy_coef": "entropy_coef", "gamma": "gamma",
                "validation_every": "validation_every",
                "validation_episodes": "validation_episodes_per_profile",
                "training_profile": "training_profile"}
    if (state.get("schema") != TRAINING_SCHEMA or summary.get("schema") != TRAINING_SCHEMA
            or config.get("seed") != seed or config.get("target_episodes") != episodes
            or any(config.get(k) != expected[v] for k, v in mappings.items())
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
            or summary.get("optimizer_updates") != episodes // expected["batch_size"]
            or last.update_count != episodes // expected["batch_size"]
            or summary.get("last_weights_sha256") != last.weights_fingerprint()
            or summary.get("best_weights_sha256") != policy.weights_fingerprint()
            or state.get("completed_episodes") != summary.get("best_episode")
            or state.get("best_episode") != summary.get("best_episode")
            or summary.get("best_validation") != state.get("best_validation")
            or last.training_state.get("best_episode") != summary.get("best_episode")
            or last.training_state.get("best_validation") != state.get("best_validation")
            or policy.update_count != state["completed_episodes"] // expected["batch_size"]
            or last.training_state.get("best_files_sha256") != checkpoint_files(path)):
        raise ValueError("Candidate does not match a completed run and its selected checkpoint")
    initialization = config.get("initialization", {})
    if (initialization.get("kind") != "transferred_weights"
            or initialization.get("weights_sha256") != V1_WEIGHTS
            or initial.weights_fingerprint() != V1_WEIGHTS or initial.update_count != 0
            or initial.training_state.get("completed_episodes") != 0
            or initial.training_state.get("config_sha256") != config_hash
            or last.training_state.get("initialized_files_sha256") != checkpoint_files(run / "initialized")):
        raise ValueError("Candidate initialization is not the declared frozen-v1 transfer")
    full_train = {"start": scenario_seed("train", seed, 0), "count": episodes}
    declared_train = [{"start": r["start"], "count": r["count"]} for r in expected["training_seed_ranges"]]
    actual_validation = validation_seed_ranges(config["validation_run_seed"], config["validation_episodes"])
    declared_validation = {r["profile"]: {"start": r["start"], "count": r["count"]}
                           for r in expected["validation_seed_ranges"]}
    if (full_train not in declared_train or actual_validation != declared_validation
            or last.training_state["seed_provenance"]["training"] != full_train
            or last.training_state["seed_provenance"]["validation"] != actual_validation):
        raise ValueError("Run seed exposure differs from the declared protocol")
    files = {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
             for name in ("config.json", "summary.json", "training.jsonl")}
    if files["training.jsonl"] != last.training_state.get("log_sha256"):
        raise ValueError("Completed training log changed after its last checkpoint")
    files.update({f"last/{name}": digest for name, digest in checkpoint_files(run / "last").items()})
    files.update({f"initialized/{name}": digest for name, digest in checkpoint_files(run / "initialized").items()})
    return {"last_checkpoint_metadata": deepcopy(last.metadata),
            "run_summary": summary, "run_files_sha256": files}


def select_from_reports(reports, candidates):
    """Rank by the declared equal-profile return; stable ascending-seed ties."""
    if set(reports) != set(PROFILES) or not candidates:
        raise ValueError("Selection requires candidates and all three profiles")
    expected_sources = None
    for profile, report in reports.items():
        if (report.get("stage") != "validation" or report.get("profile") != profile
                or report.get("schema") != "triad.robust_evaluation.v1"
                or report.get("protocol", {}).get("independent_final_test_evidence") is not False):
            raise ValueError("Selection accepts only explicitly labeled validation evidence")
        if expected_sources is None:
            expected_sources = report["implementation_sha256"]
        if report["implementation_sha256"] != expected_sources:
            raise ValueError("Candidate selection spans different implementations")
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
        name = f"candidate_{seed}"
        summaries = {}
        for profile, report in reports.items():
            method = report["methods"][name]
            for record in method["episodes"]:
                if (record["weights_sha256_before"] != candidate["weights_sha256"]
                        or record["weights_sha256_after"] != candidate["weights_sha256"]):
                    raise ValueError("Candidate selection weight mismatch")
            summaries[profile] = deepcopy(method["summary"])
        score = sum(summaries[p]["mean_return"] for p in PROFILES) / len(PROFILES)
        json.dumps(score, allow_nan=False)
        scores.append({"seed": seed, "weights_sha256": candidate["weights_sha256"],
                       "balanced_mean_return": score, "profiles": summaries})
    selected = max(scores, key=lambda item: item["balanced_mean_return"])
    return deepcopy(selected), scores


def run_selection(candidates, *, protocol_path, output, bootstrap_samples=2000):
    output, protocol_path = Path(output), Path(protocol_path)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Selection output must be an empty directory")
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol.get("schema") != "triad.robust_experiment_protocol.v1":
        raise ValueError("Unsupported robust experiment protocol")
    if sorted(candidates) != sorted(protocol["candidate_training"]["run_seeds"]):
        raise ValueError("Candidates differ from predeclared run seeds")
    ranges = protocol["candidate_selection"]["seed_ranges"]
    if {entry["profile"] for entry in ranges} != set(PROFILES) or len(ranges) != 3:
        raise ValueError("Protocol requires exactly one range per profile")
    ranges = {entry["profile"]: entry for entry in ranges}
    paths = {seed: Path(path) for seed, path in candidates.items()}
    policies = {seed: AdaptivePolicy.load(path, feature_names=FEATURE_NAMES)
                for seed, path in paths.items()}
    snapshots = {seed: checkpoint_files(path) for seed, path in paths.items()}
    info = {seed: {"weights_sha256": policy.weights_fingerprint(),
                   "metadata": deepcopy(policy.metadata),
                   "checkpoint_files_sha256": snapshots[seed],
                   "complete_run": _run_evidence(paths[seed], policy, seed, protocol)}
            for seed, policy in policies.items()}
    reference_path = BLUE_ROOT / "Checkpoints" / "adaptive-v1"
    reference = AdaptivePolicy.load(reference_path, feature_names=FEATURE_NAMES)
    if reference.weights_fingerprint() != V1_WEIGHTS:
        raise ValueError("The frozen v1 reference changed")
    old_selection_path = BLUE_ROOT / "Results" / "adaptive-v1" / "common-validation-selection.json"
    old_selection_bytes = old_selection_path.read_bytes()
    old_selection = json.loads(old_selection_bytes)
    if (old_selection.get("schema") != "triad.adaptive_model_selection.v1"
            or old_selection.get("selected", {}).get("weights_sha256") != V1_WEIGHTS):
        raise ValueError("Published v1 selection evidence does not match the frozen reference")
    reference_files = checkpoint_files(reference_path)
    lineage = {"candidate_checkpoints": {str(seed): row["metadata"] for seed, row in info.items()},
               "complete_training_runs": {str(seed): row["complete_run"]["last_checkpoint_metadata"]
                                          for seed, row in info.items()},
               "adaptive_v1_checkpoint": deepcopy(reference.metadata),
               "adaptive_v1_selection": old_selection["seed_provenance"]}
    for entry in ranges.values():
        validate_seed_range(entry["start"], entry["count"], "validation", lineage)
    for profile, entry in ranges.items():
        others = {p: row for p, row in ranges.items() if p != profile}
        validate_seed_range(entry["start"], entry["count"], "validation", others)
    sources_before = implementation_fingerprints()
    selection_source_before = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    factories = {f"candidate_{seed}": frozen_factory(paths[seed], info[seed]["weights_sha256"])
                 for seed in sorted(paths)}
    factories.update({"adaptive_v1": frozen_factory(reference_path, V1_WEIGHTS),
                      "greedy_public": GreedyPublicCoverage})
    reports = {}
    for profile in PROFILES:
        entry = ranges[profile]
        reports[profile] = evaluate_robust_methods(
            factories, profile=profile, stage="validation", seed=entry["start"],
            episodes=entry["count"], replay_count=0, bootstrap_samples=bootstrap_samples,
            seed_provenance=lineage)
        print(json.dumps({"profile": profile, "summaries": {
            name: method["summary"] for name, method in reports[profile]["methods"].items()}}), flush=True)
    if implementation_fingerprints() != sources_before:
        raise RuntimeError("Implementation changed during candidate selection")
    if {seed: checkpoint_files(path) for seed, path in paths.items()} != snapshots:
        raise RuntimeError("Checkpoint files changed during candidate selection")
    if (checkpoint_files(reference_path) != reference_files
            or protocol_path.read_bytes() != protocol_bytes
            or old_selection_path.read_bytes() != old_selection_bytes
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != selection_source_before
            or any(_run_evidence(paths[seed], policy, seed, protocol) != info[seed]["complete_run"]
                   for seed, policy in policies.items())):
        raise RuntimeError("Selection source, protocol, reference or completed run changed")
    selected, scores = select_from_reports(reports, info)
    summary = {
        "schema": SCHEMA, "stage": "validation", "final_test_accessed": False,
        "criterion": "maximum equally weighted normal/stress/capability mean return; ascending-seed ties",
        "selected": selected, "candidates": scores,
        "candidate_metadata": {str(seed): row for seed, row in info.items()},
        "seed_provenance": {"lineage": lineage, "selection_validation": list(ranges.values())},
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "implementation_sha256": sources_before,
        "selection_implementation_sha256": selection_source_before,
        "reference_summaries": {profile: {name: reports[profile]["methods"][name]["summary"]
                                          for name in ("adaptive_v1", "greedy_public")}
                                for profile in PROFILES},
        "reports": {},
    }
    encoded = {}
    for profile, report in reports.items():
        data = (json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        compressed = gzip.compress(data, mtime=0)
        filename = f"common-validation-{profile}.json.gz"
        encoded[filename] = compressed
        summary["reports"][profile] = {"path": filename, "sha256": hashlib.sha256(compressed).hexdigest(),
                                        "bytes": len(compressed)}
    summary_bytes = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode()
    output.mkdir(parents=True, exist_ok=True)
    for filename, data in encoded.items():
        (output / filename).write_bytes(data)
    (output / "selection.json").write_bytes(summary_bytes)
    print(json.dumps({"selected_seed": selected["seed"],
                      "weights_sha256": selected["weights_sha256"],
                      "balanced_mean_return": selected["balanced_mean_return"]}), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True, metavar="SEED=CHECKPOINT")
    parser.add_argument("--protocol", type=Path, default=BLUE_ROOT / "Results" / "robust-v2" / "protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(argv)
    candidates = {}
    for item in args.candidate:
        seed, separator, path = item.partition("=")
        if not separator or not seed.isdigit() or not path or int(seed) in candidates:
            parser.error("Each --candidate must be a unique SEED=CHECKPOINT pair")
        candidates[int(seed)] = Path(path)
    run_selection(candidates, protocol_path=args.protocol, output=args.output,
                  bootstrap_samples=args.bootstrap_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
