"""Evaluate a frozen adaptive checkpoint on paired unseen scenarios."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from triad_rl.adaptive_env import AdaptivePlacementEnv
from triad_rl.adaptive_policy import AdaptivePolicy
from triad_rl.adaptive_evaluation import (
    FINAL_TEST_SEED, GreedyPublicCoverage, LegacyToy210, RandomLegal,
    UniformFixedRFAndRadar, checkpoint_provenance, evaluate_methods,
)


def merge_selection_provenance(provenance, path: Path, weights_sha256: str):
    """Bind optional model-selection evidence to the exact checkpoint tested.

    All candidates' used seeds are reserved, not only the selected model's.
    The selection report is evidence of validation, never final-test evidence.
    """
    content = path.read_bytes()
    selection = json.loads(content)
    json.dumps(selection, allow_nan=False)
    if selection.get("schema") != "triad.adaptive_model_selection.v1":
        raise ValueError("Unsupported adaptive model-selection report schema")
    selected = selection.get("selected", {})
    if selected.get("weights_sha256") != weights_sha256:
        raise ValueError("Selected checkpoint weights do not match model-selection report")
    used = deepcopy(selection.get("seed_provenance", {}))
    if isinstance(used, dict) and not used.get("selection_validation"):
        if selection.get("selection_split") == "validation" and used.get("validation"):
            used["selection_validation"] = deepcopy(used["validation"])
    if not isinstance(used, dict) or not used.get("selection_validation"):
        raise ValueError("Model-selection report must declare selection_validation seeds")
    merged = deepcopy(dict(provenance))
    for name, block in used.items():
        blocks = block if isinstance(block, list) else [block]
        validated = []
        for entry in blocks:
            if (not isinstance(entry, dict)
                    or any(isinstance(entry.get(key), bool) or not isinstance(entry.get(key), int)
                           or entry[key] < 0 for key in ("start", "count"))):
                raise ValueError("Selection provenance requires nonnegative integer start/count")
            if entry["count"]:
                validated.append(deepcopy(entry))
        if name == "selection_validation" and not validated:
            raise ValueError("Model selection requires a nonempty validation range")
        existing = merged.get(name, [])
        existing = existing if isinstance(existing, list) else [existing]
        for entry in validated:
            if entry not in existing:
                existing.append(entry)
        if existing:
            merged[name] = existing[0] if len(existing) == 1 else existing
    evidence = {"filename": path.name, "sha256": hashlib.sha256(content).hexdigest(),
                "schema": selection["schema"], "selected": deepcopy(selected),
                "purpose": "model-selection validation, not final test"}
    return merged, evidence


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initialized-checkpoint", type=Path,
                        help="Preserved pretraining weights; defaults to sibling initialized/ when present")
    parser.add_argument("--selection-report", type=Path,
                        help="Paired model-selection validation report; binds selected weights and reserves used seeds")
    parser.add_argument("--legacy-checkpoint", type=Path,
                        default=Path(__file__).resolve().parents[1] / "Checkpoints" / "toy-210")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=FINAL_TEST_SEED)
    parser.add_argument("--split", choices=("heldout", "stress", "train"), default="heldout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(argv)
    observation = AdaptivePlacementEnv(seed=args.seed, split=args.split).reset(seed=args.seed)
    feature_names = observation["feature_names"]
    # Load once to validate schema and obtain training provenance. Factories
    # reload frozen weights per episode so no state can leak between scenarios.
    policy = AdaptivePolicy.load(args.checkpoint, feature_names=feature_names)
    metadata = json.loads((args.checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
    provenance = checkpoint_provenance(metadata)
    selection_evidence = None
    if args.selection_report is not None:
        provenance, selection_evidence = merge_selection_provenance(
            provenance, args.selection_report, policy.weights_fingerprint())
    hidden_size = getattr(policy, "hidden_size", 32)
    initialized_path = args.initialized_checkpoint
    if initialized_path is None and (args.checkpoint.parent / "initialized" / "checkpoint.json").is_file():
        initialized_path = args.checkpoint.parent / "initialized"
    if initialized_path is not None:
        AdaptivePolicy.load(initialized_path, feature_names=feature_names)
        initialized_factory = lambda seed: AdaptivePolicy.load(initialized_path, feature_names=feature_names)
    else:
        initialized_factory = lambda seed: AdaptivePolicy(feature_names, seed=0, hidden_size=hidden_size)
    factories = {
        "adaptive": lambda seed: AdaptivePolicy.load(args.checkpoint, feature_names=feature_names),
        # One architecture/initialization across the full evaluation; this is
        # not a different randomly initialized model selected every episode.
        "initialized": initialized_factory,
        "random_legal": RandomLegal,
        "greedy_public": GreedyPublicCoverage,
        "uniform_fixed_rf_radar": UniformFixedRFAndRadar,
        "legacy_toy210_projected": lambda seed: LegacyToy210(args.legacy_checkpoint, seed=seed),
    }
    report = evaluate_methods(factories, episodes=args.episodes, seed=args.seed,
                              split=args.split, replay_count=args.replays,
                              bootstrap_samples=args.bootstrap_samples,
                              seed_provenance=provenance)
    report["checkpoint"] = {"name": args.checkpoint.name, "metadata": metadata}
    if selection_evidence is not None:
        report["model_selection"] = selection_evidence
    report["methods"]["initialized"]["metadata"]["initialization_source"] = (
        "preserved pretraining checkpoint" if initialized_path else "new architecture with fixed seed0; original initialization unavailable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, result in report["methods"].items():
        print(name, json.dumps(result["summary"], sort_keys=True))
    print(f"Paired evaluation saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
