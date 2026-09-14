"""Development-only temporal control probe on already-consumed ranking cases.

Exactly the first twenty archived cases per profile, never selected successes.
This is a non-RL diagnostic, not new held-out evidence or a promotion screen.
It scores new layouts against the unchanged simulator on those same cases.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from triad_rl.temporal_env import TemporalPlacementEnv
from triad_rl.temporal_inputs import TemporalConfig, TemporalPublicGreedy


BLUE = Path(__file__).resolve().parent.parent
ARCHIVE = BLUE / "Results/ranking-v5-pilot"
MANIFEST_SHA256 = "a9d975e38dbdf8a5095c52e0767e2142ac5616dcd09d338c85247747517dd058"
CASES_PER_PROFILE = 20
PROFILES = ("normal", "stress", "capability")
REFERENCE_METHODS = ("greedy_public", "balanced_v3", "ranker_403", "ranker_404", "ranker_405")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _metrics(info):
    return {"timely": 1. - info["breached_fraction"], "detected": info["detected_fraction"],
            "confirmed": info["confirmed_fraction"], "success": float(info["success"]),
            "cost": info["cost"], "return": info["return"], "invalid_actions": info["invalid_actions"]}


def run_probe(output, *, config=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh probe output; no existing evidence is overwritten")
    config = TemporalConfig() if config is None else config
    source_paths = [Path(__file__), *(BLUE / "Python/triad_rl" / name for name in
                    ("temporal_confirmation.py", "temporal_inputs.py", "temporal_env.py",
                     "adaptive_env.py", "adaptive_inputs.py", "robust_scenarios.py"))]
    sources = {path.relative_to(BLUE).as_posix(): sha(path.read_bytes()) for path in source_paths}
    manifest_path = ARCHIVE / "artifact-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    if sha(manifest_bytes) != MANIFEST_SHA256:
        raise ValueError("Published development archive manifest changed")
    manifest = json.loads(manifest_bytes)
    hashes = {row["path"]: row["sha256"] for row in manifest["files"]}
    reports, snapshots = {}, {manifest_path: manifest_bytes}
    for profile in PROFILES:
        path = ARCHIVE / f"evaluation/validation-{profile}.json.gz"
        raw = path.read_bytes()
        if sha(raw) != hashes[str(path.relative_to(BLUE)).replace("\\", "/")]:
            raise ValueError("Published development report changed")
        snapshots[path] = raw
        reports[profile] = json.loads(gzip.decompress(raw))
    actor, episodes, timings = TemporalPublicGreedy(), [], []
    for profile in PROFILES:
        methods = reports[profile]["methods"]
        for index in range(CASES_PER_PROFILE):
            recorded = methods["greedy_public"]["episodes"][index]
            for method in REFERENCE_METHODS:
                other = methods[method]["episodes"][index]
                if (other["seed"] != recorded["seed"]
                        or other["scenario_sha256"] != recorded["scenario_sha256"]):
                    raise ValueError("Archived comparator cases are not paired")
            scenario = deepcopy(recorded["scenario"])
            catalogue = scenario.pop("evaluation_catalogue")
            scenario.pop("curriculum_metadata", None)
            started = time.perf_counter()
            env = TemporalPlacementEnv(scenario=scenario, catalogue=catalogue, config=config)
            observation, actions = env.observe(), []
            while not env.done:
                action = actor.act(observation)
                actions.append(action)
                observation, _, _, info = env.step(action)
            timings.append(time.perf_counter() - started)
            episodes.append({"profile": profile, "seed": recorded["seed"],
                             "parent_scenario_sha256": recorded["scenario_sha256"],
                             "actions": actions, "placements": env.placements,
                             "temporal_public_control": _metrics(info),
                             "references": {method: _metrics(methods[method]["episodes"][index]["metrics"])
                                            for method in REFERENCE_METHODS}})
        print(f"Temporal reused-development probe: {profile} complete", flush=True)
    method_names = ("temporal_public_control", *REFERENCE_METHODS)
    summaries = {}
    for method in method_names:
        summaries[method] = {}
        for profile in PROFILES:
            metrics = [(row[method] if method == "temporal_public_control" else row["references"][method])
                       for row in episodes if row["profile"] == profile]
            summaries[method][profile] = {name: float(np.mean([row[name] for row in metrics])) for name in metrics[0]}
        summaries[method]["equal_profile"] = {name: float(np.mean([summaries[method][profile][name] for profile in PROFILES]))
                                                for name in summaries[method][PROFILES[0]]}
    result = {"schema": "triad.temporal_reused_development_probe.v1",
              "interpretation": "Non-RL diagnostic on already-consumed cases; not unseen generalization or policy promotion",
              "selection": "First twenty ranking-v5 validation cases per profile, fixed before this probe",
              "training_performed": False, "final_test_accessed": False, "policy_promoted": False,
              "source_sha256": sources, "archive_manifest_sha256": MANIFEST_SHA256,
              "input_sha256": {path.relative_to(BLUE).as_posix(): sha(data) for path, data in snapshots.items()},
              "temporal_config": observation["temporal_config"], "summaries": summaries,
              "episodes": episodes, "elapsed_seconds": float(sum(timings))}
    if sources != {path.relative_to(BLUE).as_posix(): sha(path.read_bytes()) for path in source_paths}:
        raise RuntimeError("Temporal probe sources changed; no output published")
    if any(path.read_bytes() != data for path, data in snapshots.items()):
        raise RuntimeError("Temporal probe inputs changed; no output published")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_probe(args.output)
    print(json.dumps({method: row["equal_profile"] for method, row in report["summaries"].items()}, sort_keys=True))
