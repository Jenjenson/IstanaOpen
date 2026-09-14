"""Paired, frozen-policy validation/test of heterogeneous Blue sensor deployment.

Validation is explicitly reusable development evidence, never an independent
final test. Test seeds must be new, including relative to transferred models.
This additive evaluator leaves the published adaptive-v1 benchmark unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from triad_rl.adaptive_evaluation import (
    GreedyPublicCoverage, LegacyToy210, RandomLegal, UniformFixedRFAndRadar,
    evaluate_methods, json_safe, summarize,
)
from triad_rl.adaptive_inputs import FEATURE_NAMES
from triad_rl.adaptive_policy import AdaptivePolicy


REPORT_SCHEMA = "triad.robust_evaluation.v1"
VALIDATION_BASE = 10 ** 15
TEST_BASE = 2 * 10 ** 15
PROFILES = ("mixed", "normal", "stress", "capability")
V1_WEIGHTS = "2062905477954b810516fa14e2009d6d25afb391950354780c6161644eebae36"
BLUE_ROOT = Path(__file__).resolve().parents[1]
CONSUMED_TEST_RANGES = {
    "published_adaptive_v1_heldout": {"start": TEST_BASE, "count": 200},
    "published_adaptive_v1_stress": {"start": TEST_BASE + 10_000, "count": 200},
}


def nested_seed_ranges(value: Any, prefix: str = "provenance") -> dict[str, dict[str, int]]:
    """Find every declared range, including nested transfer/selection lineage.

    Accepting only a shallow trainer field would miss source-model validation
    and model selection. The path names survive in the report for an audit.
    Range-shaped entries fail closed on malformed/negative/noninteger values.
    """
    found: dict[str, dict[str, int]] = {}
    if isinstance(value, Mapping):
        if "start" in value or "count" in value:
            if (set(("start", "count")) - value.keys()
                    or any(isinstance(value[key], bool) or not isinstance(value[key], int)
                           or value[key] < 0 for key in ("start", "count"))):
                raise ValueError(f"Invalid nonnegative integer seed range at {prefix}")
            found[prefix] = {"start": value["start"], "count": value["count"]}
        for name, item in value.items():
            if isinstance(item, (Mapping, list, tuple)):
                found.update(nested_seed_ranges(item, f"{prefix}.{name}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.update(nested_seed_ranges(item, f"{prefix}[{index}]"))
    return found


def validate_seed_range(seed: int, episodes: int, stage: str,
                        provenance: Mapping[str, Any] | None = None) -> dict[str, dict[str, int]]:
    """Validate the *whole* requested interval before any scenario is sampled."""
    if (isinstance(seed, bool) or not isinstance(seed, int)
            or isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1):
        raise ValueError("Evaluation requires an integer seed and positive integer episodes")
    end = seed + episodes
    if stage == "validation":
        if seed < VALIDATION_BASE or end > TEST_BASE:
            raise ValueError("Validation seeds must stay in [10**15, 2*10**15)")
    elif stage == "test":
        if seed < TEST_BASE:
            raise ValueError("Test seeds must be at least 2*10**15")
    else:
        raise ValueError("Evaluation stage must be validation or test")
    ranges = nested_seed_ranges(provenance or {})
    reserved = {**ranges, **CONSUMED_TEST_RANGES}
    for name, entry in reserved.items():
        if max(seed, entry["start"]) < min(end, entry["start"] + entry["count"]):
            raise ValueError(f"Evaluation seeds overlap {name}")
    return ranges


def implementation_fingerprints() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    files = [Path("evaluate_robust.py"), *[Path("triad_rl") / name for name in (
        "adaptive_env.py", "adaptive_inputs.py", "adaptive_policy.py",
        "adaptive_evaluation.py", "train_adaptive.py", "robust_scenarios.py")]]
    # The evaluator also works before a new trainer is present (e.g. a separate
    # source checkout), but any trainer appearing/disappearing mid-run is drift.
    files.extend(path for path in (Path("triad_rl/train_robust.py"), Path("train_robust.py"))
                 if (directory / path).is_file())
    return {path.as_posix(): hashlib.sha256((directory / path).read_bytes()).hexdigest()
            for path in files}


class EvaluationCase:
    """Bind catalogue and curriculum to pairing without exposing truth to actors."""

    def __init__(self, env: Any):
        self.env = env

    @property
    def scenario(self) -> dict[str, Any]:
        scenario = deepcopy(self.env.scenario)
        if {"evaluation_catalogue", "curriculum_metadata"}.intersection(scenario):
            raise ValueError("Scenario uses reserved evaluation audit fields")
        scenario["evaluation_catalogue"] = deepcopy(self.env.catalogue)
        scenario["curriculum_metadata"] = deepcopy(self.env.case_metadata)
        return scenario

    def reset(self, *, seed: int):
        return self.env.reset(seed=seed)

    def step(self, action: int):
        return self.env.step(action)


def evaluate_robust_methods(method_factories: Mapping[str, Callable[[int], Any]], *,
                            seed: int, episodes: int = 200, profile: str = "mixed",
                            stage: str = "validation", replay_count: int = 1,
                            bootstrap_samples: int = 2000,
                            seed_provenance: Mapping[str, Any] | None = None,
                            env_factory: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Compare decisions using the same catalogue, public inputs and truth draw."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown robust profile: {profile}")
    ranges = validate_seed_range(seed, episodes, stage, seed_provenance)
    if env_factory is None:
        from triad_rl.robust_scenarios import RobustPlacementEnv
        env_factory = RobustPlacementEnv
    source_before = implementation_fingerprints()

    def paired_factory(*, seed: int, split: str):
        # split is an internal compatibility parameter, never a claim that
        # validation is independent test evidence or that profiles equal v1.
        return EvaluationCase(env_factory(seed=seed, profile=profile))

    report = evaluate_methods(method_factories, episodes=episodes, seed=seed,
                              split="heldout", replay_count=replay_count,
                              bootstrap_samples=bootstrap_samples,
                              seed_provenance=ranges, env_factory=paired_factory)
    report["schema"] = REPORT_SCHEMA
    report["profile"] = profile
    report["stage"] = stage
    report["split"] = stage
    report["environment"] = ("Robust randomized curriculum over frozen adaptive sensor physics; "
                             "not native Unreal or real-world validation")
    report["seed_provenance"] = {"declared_lineage": json_safe(deepcopy(seed_provenance or {})),
                                 "reserved_ranges": ranges,
                                 "consumed_prior_test_ranges": deepcopy(CONSUMED_TEST_RANGES),
                                 "evaluation": {"start": seed, "count": episodes}}
    report["protocol"].update({
        "stage": stage,
        "test_tuning_allowed": stage == "validation",
        "independent_final_test_evidence": stage == "test",
        "evidence_scope": ("Development/model-selection validation; may inform later training"
                           if stage == "validation" else
                           "Fresh declared-disjoint test scenarios; not a proof of optimality or real-world performance"),
        "declared_seed_disjointness_verified": bool(ranges),
        "prior_published_test_ranges_reserved": True,
        "scenario_pairing_includes": ["scenario_truth", "public_state", "sensor_catalogue", "curriculum_metadata"],
        "hidden_case_metadata_in_policy_observation": False,
    })
    for method in report["methods"].values():
        labels = sorted({str(row["scenario"]["curriculum_metadata"].get("profile", profile))
                         for row in method["episodes"]})
        method["subgroups"]["curriculum_profile"] = {
            label: summarize([row for row in method["episodes"]
                              if str(row["scenario"]["curriculum_metadata"].get("profile", profile)) == label])
            for label in labels}
    source_after = implementation_fingerprints()
    if source_before != source_after:
        raise RuntimeError("Robust implementation changed during evaluation; refusing mixed-version evidence")
    report["implementation_sha256"] = source_before
    report["implementation_sha256_after"] = source_after
    return json_safe(report)


def checkpoint_files(path: Path) -> dict[str, str]:
    return {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
            for name in ("checkpoint.json", "arrays.npz")}


def frozen_factory(path: Path, expected: str) -> Callable[[int], AdaptivePolicy]:
    def factory(seed: int) -> AdaptivePolicy:
        actor = AdaptivePolicy.load(path, feature_names=FEATURE_NAMES)
        if actor.weights_fingerprint() != expected:
            raise RuntimeError(f"Frozen checkpoint changed between episodes: {path.name}")
        return actor
    return factory


def read_selection_report(path: Path, weights_sha256: str) -> tuple[dict, dict, bytes]:
    """Bind candidate selection to weights and reserve every exposed lineage."""
    raw = path.read_bytes()
    report = json.loads(raw)
    json.dumps(report, allow_nan=False)
    if (report.get("schema") != "triad.robust_model_selection.v1"
            or report.get("stage") != "validation"
            or report.get("final_test_accessed") is not False):
        raise ValueError("Candidate selection must be robust validation evidence with no final test access")
    if report.get("selected", {}).get("weights_sha256") != weights_sha256:
        raise ValueError("Candidate weights do not match the model-selection report")
    provenance = report.get("seed_provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("Model selection requires seed provenance")
    validation = provenance.get("selection_validation", [])
    if (not isinstance(validation, list) or len(validation) != 3
            or any(not isinstance(entry, dict) for entry in validation)
            or {entry.get("profile") for entry in validation} != {"normal", "stress", "capability"}):
        raise ValueError("Model selection requires all three profile validation seed ranges")
    nested_seed_ranges(provenance)  # Validate all nested source ranges too.
    for entry in validation:
        validate_seed_range(entry.get("start"), entry.get("count"), "validation")
    evidence = {"name": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
                "schema": report["schema"], "stage": report["stage"],
                "final_test_accessed": False, "selected": deepcopy(report["selected"])}
    return deepcopy(provenance), evidence, raw


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initialized-checkpoint", type=Path, required=True,
                        help="Preserved candidate weights BEFORE robust training; may already be pretrained")
    parser.add_argument("--adaptive-v1-checkpoint", type=Path,
                        default=BLUE_ROOT / "Checkpoints" / "adaptive-v1")
    parser.add_argument("--legacy-checkpoint", type=Path,
                        default=BLUE_ROOT / "Checkpoints" / "toy-210")
    parser.add_argument("--skip-legacy", action="store_true", help="Omit the lossy toy210 comparison")
    parser.add_argument("--selection-report", type=Path,
                        help="Robust common-validation selection; binds candidate weights and reserves all exposed seeds")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, required=True, help="Explicit fresh validation/test seed range")
    parser.add_argument("--profile", choices=PROFILES, default="mixed")
    parser.add_argument("--stage", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(argv)
    # Reject unsafe stage/ranges even before reading checkpoint files.
    validate_seed_range(args.seed, args.episodes, args.stage)
    paths = {"adaptive": args.checkpoint, "adaptive_v1": args.adaptive_v1_checkpoint,
             "initialized": args.initialized_checkpoint}
    actors = {name: AdaptivePolicy.load(path, feature_names=FEATURE_NAMES) for name, path in paths.items()}
    if actors["adaptive_v1"].weights_fingerprint() != V1_WEIGHTS:
        raise ValueError("adaptive_v1 comparison requires the frozen published v1 weights")
    metadata = {name: deepcopy(actor.metadata) for name, actor in actors.items()}
    files_before = {name: checkpoint_files(path) for name, path in paths.items()}
    expected_weights = {name: actor.weights_fingerprint() for name, actor in actors.items()}
    candidate_state = metadata["adaptive"].get("training_state", {})
    initial_binding = candidate_state.get("initialization", {}).get("weights_sha256")
    if candidate_state.get("schema") == "triad.robust_training.v1" and not initial_binding:
        raise ValueError("Robust candidate is missing its preserved initialization fingerprint")
    if initial_binding is not None and initial_binding != expected_weights["initialized"]:
        raise ValueError("Initialized checkpoint does not match the candidate's pre-training weights")
    provenance: dict[str, Any] = {"checkpoint_lineage": metadata}
    # The published v1 selection tested three candidates on shared validation.
    # This exposure is not present in v1's original training checkpoint itself.
    selection_path = BLUE_ROOT / "Results" / "adaptive-v1" / "common-validation-selection.json"
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    if (selection.get("schema") != "triad.adaptive_model_selection.v1"
            or selection.get("selected", {}).get("weights_sha256") != V1_WEIGHTS):
        raise ValueError("Published v1 model-selection evidence does not match frozen weights")
    provenance["published_v1_selection"] = selection.get("seed_provenance", {})
    selection_evidence, candidate_selection_bytes = None, None
    if args.selection_report is not None:
        lineage, selection_evidence, candidate_selection_bytes = read_selection_report(
            args.selection_report, expected_weights["adaptive"])
        provenance["candidate_selection"] = lineage
    validate_seed_range(args.seed, args.episodes, args.stage, provenance)
    factories = {name: frozen_factory(paths[name], expected_weights[name]) for name in paths}
    factories.update({"greedy_public": GreedyPublicCoverage, "random_legal": RandomLegal,
                      "uniform_fixed_rf_radar": UniformFixedRFAndRadar})
    if not args.skip_legacy:
        # LegacyToy210 verifies the actual preserved toy-210 parameter hash.
        factories["legacy_toy210_projected"] = lambda seed: LegacyToy210(args.legacy_checkpoint, seed=seed)
        files_before["legacy_toy210_projected"] = {
            name: hashlib.sha256((args.legacy_checkpoint / name).read_bytes()).hexdigest()
            for name in ("checkpoint.json", "arrays.npz")}
    report = evaluate_robust_methods(factories, seed=args.seed, episodes=args.episodes,
                                      profile=args.profile, stage=args.stage,
                                      replay_count=args.replays, bootstrap_samples=args.bootstrap_samples,
                                      seed_provenance=provenance)
    files_after = {name: checkpoint_files(path) for name, path in paths.items()}
    if not args.skip_legacy:
        files_after["legacy_toy210_projected"] = {
            name: hashlib.sha256((args.legacy_checkpoint / name).read_bytes()).hexdigest()
            for name in ("checkpoint.json", "arrays.npz")}
    if files_before != files_after or selection_path.read_bytes() != selection_bytes:
        raise RuntimeError("Checkpoint/selection files changed during evaluation")
    if args.selection_report is not None and args.selection_report.read_bytes() != candidate_selection_bytes:
        raise RuntimeError("Candidate selection report changed during evaluation")
    report["checkpoint_files_sha256_before"] = files_before
    report["checkpoint_files_sha256_after"] = files_after
    report["checkpoint_weights_sha256"] = expected_weights
    report["published_v1_selection_sha256"] = hashlib.sha256(selection_bytes).hexdigest()
    if selection_evidence is not None:
        report["model_selection"] = selection_evidence
    report["checkpoints"] = {name: {"name": paths[name].name, "metadata": data}
                             for name, data in metadata.items()}
    report["methods"]["initialized"]["metadata"].update({
        "description": "Explicit preserved candidate checkpoint before robust training; may already be pretrained",
        "initialization_source": args.initialized_checkpoint.name,
        "is_claimed_untrained": False,
        "candidate_initialization_binding_verified": initial_binding is not None,
        "same_weights_as_published_v1": expected_weights["initialized"] == V1_WEIGHTS,
        "recorded_source_completed_episodes": metadata["initialized"].get("training_state", {}).get("completed_episodes"),
    })
    # Strict serialization precedes file creation so NaNs cannot leave a
    # partial report that looks like valid evidence to downstream consumers.
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    for name, result in report["methods"].items():
        print(name, json.dumps(result["summary"], sort_keys=True, allow_nan=False))
    print(f"Paired robust {args.stage} evaluation saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
