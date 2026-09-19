"""Verify completed warning artifacts without retraining or choosing a winner."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from triad_rl.warning_policy import WarningPolicy
from triad_rl.warning_scenario import digest, saturation_metrics
from train_warning_approach import paired_check


def audit(run):
    run = Path(run)
    root = Path(__file__).resolve().parents[3]
    read = lambda name: json.loads((run / name).read_text(encoding="utf-8"))
    protocol, summary = read("protocol.json"), read("summary.json")
    if summary["protocol"] != protocol:
        raise ValueError("Protocol changed between start and finish")
    archived_sources = False
    for relative, expected in protocol["source_sha256"].items():
        candidate = root / relative
        archive = run / "training-source-snapshot" / relative
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected and archive.is_file():
            candidate = archive
            archived_sources = True
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Training source changed: {relative}")
    approach = protocol.get("schema") == "istana.native_warning_approach.v2"
    physical = read("physical-contract.json") if approach else None
    training = [json.loads(line) for line in (run / "training.jsonl").read_text().splitlines()]
    if [r["seed"] for r in training] != protocol["training_seeds"]:
        raise ValueError("Missing, duplicate or reordered training episode")
    if set(protocol["training_seeds"]) & set(protocol["evaluation_seeds"]):
        raise ValueError("Training/evaluation seed overlap")
    initial = read("evaluation-0000.json")
    all_runs = list(training)
    for checkpoint, metadata in zip(protocol["checkpoints"], summary["checkpoints"], strict=True):
        path = run / f"policy-{checkpoint:04d}.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("Checkpoint changed")
        evaluation = read(f"evaluation-{checkpoint:04d}.json")
        if approach: paired_check(initial, evaluation)
        if [r["seed"] for r in evaluation] != protocol["evaluation_seeds"]:
            raise ValueError("Unpaired evaluation seeds")
        policy = WarningPolicy.load(path, evaluation[0]["context"])
        if policy.updates != checkpoint // protocol["batch_size"]:
            raise ValueError("Wrong update count")
        for i, (before, after) in enumerate(zip(initial, evaluation, strict=True)):
            if [(r["droneId"], r["zoneEntrySeconds"]) for r in before["warning_evidence"]] != [(r["droneId"], r["zoneEntrySeconds"]) for r in after["warning_evidence"]]:
                raise ValueError("Paired Red arrival times changed")
            placements, _ = policy.plan(evaluation[0]["context"], rng=np.random.default_rng(protocol["evaluation_action_seeds"][i]))
            if placements != after["placements"]:
                raise ValueError("Checkpoint does not reproduce its recorded layout")
        all_runs.extend(evaluation)
    greedy = read("evaluation-greedy.json")
    if approach: paired_check(initial, greedy)
    all_runs.extend(greedy)
    for episode in all_runs:
        rows = episode["warning_evidence"]
        if episode["metrics"]["unresolved"] or len(rows) != episode["metrics"]["targets"]:
            raise ValueError("Unresolved/missing targets")
        expected = [max(0, r["zoneEntrySeconds"] - r["firstDetectionSeconds"]) if r["firstDetectionSeconds"] is not None else 0 for r in rows]
        if not np.allclose(expected, [r["warningSeconds"] for r in rows], atol=1e-7, rtol=0):
            raise ValueError("Warning evidence mismatch")
        if not np.isclose(np.mean(expected), episode["metrics"]["mean_drone_warning_s"], atol=1e-7, rtol=0):
            raise ValueError("Warning mean mismatch")
        detections = [r["firstDetectionSeconds"] for r in rows if r["firstDetectionSeconds"] is not None]
        team = max(0., min(r["zoneEntrySeconds"] for r in rows)-min(detections)) if detections else 0.
        if not np.isclose(team, episode["metrics"]["team_warning_s"], atol=1e-7, rtol=0):
            raise ValueError("Team warning mismatch")
        if approach:
            if episode["physical_contract_sha256"] != digest(physical):
                raise ValueError("Physical contract drift")
            saturation = saturation_metrics(rows, physical["temporal"]["look_interval_s"])
            if episode["saturation"] != saturation or saturation["team_first_look_saturated"]:
                raise ValueError("Saturated or inconsistent first-look diagnostics")
            start = episode["start_diagnostics"]
            if (start["spawned_in_range_count"] or start["minimum_universal_clearance_m"] < 50
                    or len(start["drones"]) != len(rows)):
                raise ValueError("Invalid initial clearance diagnostics")
    return {"passed": True, "training_episodes": len(training),
            "evaluation_episodes_including_greedy": len(all_runs)-len(training),
            "fixed_checkpoints": len(protocol["checkpoints"]),
            "training_source_hashes_match": True, "all_checkpoint_layouts_reproduced": True,
            "archived_source_snapshot_used": archived_sources,
            "sampled_red_trajectories_and_contract_identical": approach,
            "paired_red_arrivals_identical": True, "all_targets_resolved": True,
            "warning_metrics_recomputed": True,
            "native_editor_dll_sha256_at_audit": hashlib.sha256((root / "Binaries/Win64/UnrealEditor-IstanaOpen.dll").read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    with (args.run / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
