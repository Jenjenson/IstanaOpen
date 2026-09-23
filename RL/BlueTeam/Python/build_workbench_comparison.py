"""Capture archived RL layouts versus the eight-camera workbench start.

This is a matched native replay, not training. Both layouts face the same five
synthetic approach lanes, paths, speeds, sensor model, site grid and episode
seed. The archived RL layout is replayed unchanged and may use fewer than the
eight sensors allowed by the workbench; it is not described as an eight-sensor RL
policy.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

from build_native_comparison import frame, target_results, trajectory_digest, write_json
from triad_rl.istana_live import IstanaLiveClient
from triad_rl.training_workbench import _red_centers, common_sense_start


ROOT = Path(__file__).resolve().parents[3]
BLUE = ROOT / "RL/BlueTeam"
ARCHIVED = BLUE / "Results/native-placement-comparison"
SCHEMA = "istana.workbench_layout_comparison_capture.v1"
POLICIES = {"406": "Temporal RL · policy A", "407": "Temporal RL · policy B",
            "408": "Temporal RL · policy C"}
SEEDS = (1700000, 1700001, 1700002)
SOURCES = (
    "RL/BlueTeam/Python/build_workbench_comparison.py",
    "RL/BlueTeam/Python/triad_rl/red_policy.py",
    "RL/BlueTeam/Python/triad_rl/training_workbench.py",
    "RL/BlueTeam/Python/triad_rl/istana_live.py",
    "RL/BlueTeam/Python/triad_rl/warning_policy.py",
    "Source/IstanaOpen/IstanaGameMode.cpp",
    "Source/IstanaOpen/Simulation/BlueTeam/BlueTeamCoordinator.cpp",
    "Source/IstanaOpen/Simulation/BlueTeam/BlueWarningTime.h",
    "Source/IstanaOpen/Simulation/RedTeam/RedTeamManager.cpp",
    "Source/IstanaOpen/Simulation/Swarm/IstanaSwarmSimulation.cpp",
    "Source/IstanaOpen/Simulation/Sensing/DirectionalSensorModel.cpp",
    "Source/IstanaOpen/Simulation/Sensing/DirectionalSensorModel.h",
)


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archived_layouts():
    manifest = json.loads((ARCHIVED / "manifest.json").read_text(encoding="utf-8"))
    blob = (ARCHIVED / manifest["bundle"]).read_bytes()
    if hashlib.sha256(blob).hexdigest() != manifest["sha256"]:
        raise ValueError("Archived native comparison checksum differs from its manifest")
    bundle = json.loads(gzip.decompress(blob))
    return {policy: next(row["rl"]["placements"] for row in bundle["episodes"]
                         if row["policy"] == policy) for policy in POLICIES}


def deployment_rows(saved, context):
    """Map saved public placements back to exact approved native site IDs."""
    sites = context["publicSnapshot"]["sites"]
    result = []
    for row in saved:
        distances = [sum((float(a) - float(b)) ** 2 for a, b in zip(site, row["position"][:2]))
                     for site in sites]
        site_id = min(range(len(sites)), key=distances.__getitem__)
        if distances[site_id] > 1e-8:
            raise ValueError("Archived RL placement does not map to the workbench site grid")
        result.append({"profileId": row["sensor_id"], "siteId": site_id,
                       "yawDeg": row.get("yaw_deg", 0.), "pitchDeg": row.get("pitch_deg", 0.)})
    return result


def capture(client, seed, label, placements_for_context, selection, *, step_batch=100):
    started = time.monotonic()
    red = client.reset(seed)
    context = client.get_blue_context()
    public = context["publicSnapshot"]
    if public["budget_total"] < 8 or public["max_sites"] < 8 or red["groupCount"] != 5:
        raise ValueError(
            "Restart Unreal with -IstanaTrainingWorkbench -IstanaDelayedDetectionDemo "
            "before capturing the eight-sensor comparison")
    placements = placements_for_context(context)
    client.deploy(placements)
    centers = _red_centers(red, seed)
    client.place_red(centers)
    blue = client.observe_blue()
    frames = [frame(client.request("observe")["observation"], blue, context["worldOriginCm"])]
    while not (blue["terminated"] or blue["truncated"]):
        if client.completed_steps >= 10000:
            client.cancel()
            raise RuntimeError("Native workbench comparison exceeded 10,000 steps")
        response = client.step(step_batch)
        blue = response["blueObservation"]
        frames.append(frame(response["observation"], blue, context["worldOriginCm"]))
    evidence = blue.get("warningEvidenceForEvaluationOnly")
    if not blue.get("metricsAvailable") or not evidence:
        raise RuntimeError("Native workbench episode ended without warning evidence")
    trajectory = trajectory_digest(frames)
    results = target_results(evidence, context["temporalConfig"]["lead_time_s"])
    indexed = {row["id"]: row for row in results}
    for sample in frames:
        for threat in sample["threats"]:
            target = indexed[threat["id"]]
            threat["detected"] = (target["first_detection"] is not None
                                  and target["first_detection"] <= sample["time"] + 1e-8)
            threat["confirmed"] = (target["first_confirmation"] is not None
                                   and target["first_confirmation"] <= sample["time"] + 1e-8)
    outcome = "unresolved" if any(row["unresolved"] for row in results) else "completed"
    metrics = {**blue["metrics"], "return": blue["reward"], "target_results": results,
               "outcome": outcome, "duration_seconds": blue["elapsedSeconds"]}
    view = {"mode": "comparison", "label": label,
        "coordinateLabel": "Native workbench · objective-relative metres",
        "catalogue": context["catalogue"], "placements": blue["publicSnapshot"]["placements"],
        "sites": public["sites"], "blockedSites": public["blocked_sites"],
        "budget": public["budget_total"], "maxSensors": public["max_sites"],
        "objectiveRadius": context["temporalConfig"]["objective_radius_m"],
        "surfaceMounted": True, "frames": frames, "metrics": metrics,
        "decisions": [], "selection": deepcopy(selection), "seed": seed,
        "weather": public["weather"], "policy": label, "outcome": outcome, "ended": True,
        "warningEvidence": evidence,
        "audit": {"recorded": True, "native_unreal": True, "live_unreal": False,
            "policy_truth_access": False, "training_performed": False,
            "trajectorySha256": trajectory, "nativeRunId": red["runId"],
            "completedSteps": client.completed_steps}}
    return {"view": view, "context": context, "redContext": red,
            "redCentersWorldCm": centers, "trajectorySha256": trajectory,
            "wallSeconds": time.monotonic() - started}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path,
                        default=BLUE / "Results/workbench-placement-comparison")
    parser.add_argument("--raw-output", type=Path,
                        default=ROOT / "Saved/WorkbenchPlacementComparison")
    parser.add_argument("--resume-partial", action="store_true",
                        help="Reuse incomplete output directories that contain no final bundle")
    parser.add_argument("--replace-existing", action="store_true",
                        help="Regenerate an existing bundle only when its schema matches this tool")
    args = parser.parse_args(argv)
    if args.resume_partial and args.replace_existing:
        raise ValueError("Choose either --resume-partial or --replace-existing")
    if args.replace_existing:
        manifest_path = args.output / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("There is no completed workbench comparison bundle to replace")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA:
            raise ValueError("Refusing to replace an output directory with a different schema")
        args.output.mkdir(parents=True, exist_ok=True)
        args.raw_output.mkdir(parents=True, exist_ok=True)
    elif args.resume_partial:
        if (args.output / "manifest.json").exists() or (args.output / "bundle.json.gz").exists():
            raise ValueError("Refusing to overwrite a completed workbench comparison bundle")
        args.output.mkdir(parents=True, exist_ok=True)
        args.raw_output.mkdir(parents=True, exist_ok=True)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        args.raw_output.mkdir(parents=True, exist_ok=False)
    layouts = archived_layouts()
    protocol = {"schema": SCHEMA, "createdUtc": datetime.now(timezone.utc).isoformat(),
        "baseCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                text=True).strip(),
        "sourceSha256": {path: file_digest(ROOT / path) for path in SOURCES},
        "episodeSeeds": list(SEEDS), "captureStepBatch": 100,
        "scenario": (
            "Five distinct seeded approaches selected from eight synthetic sectors, "
            "with small bearing jitter at one fixed spawn radius in the native Training Workbench"),
        "comparison": (
            "Archived temporal RL placement replayed unchanged versus the fixed eight-camera "
            "balanced workbench start. The archived policy was not retrained for eight sensors."),
        "warningDefinition": (
            "Per-drone max(0, 20 m zone arrival - first detection); undetected contributes zero."),
        "selection": "All three existing archived RL layouts and three predeclared seeds retained."}
    write_json(args.output / "protocol.json", protocol)
    episodes = []
    baseline_selection = {"label": "Eight directional sensors · workbench start",
        "kind": "fixed_directional_workbench_start",
        "explanation": (
            "Center eight surface-mounted limited-FOV thermal cameras on the benchmark "
            "sectors before the five seeded Red approaches are selected.")}
    with IstanaLiveClient(args.port, timeout=120.) as client:
        for case, seed in enumerate(SEEDS, 1):
            print(json.dumps({"case": case, "layout": "directional_balanced_8",
                              "status": "capturing"}), flush=True)
            baseline = capture(client, seed, baseline_selection["label"],
                lambda context: common_sense_start(
                    context, "directional_balanced_8")["placements"], baseline_selection)
            write_json(args.raw_output / f"capture-{case}-directional-balanced-8.json", baseline)
            for policy, label in POLICIES.items():
                print(json.dumps({"case": case, "layout": policy, "status": "capturing"}),
                      flush=True)
                selection = {"label": f"Archived {label} layout",
                    "kind": "archived_rl_layout_replay",
                    "explanation": (
                        "Replay the existing RL checkpoint placement unchanged in the sector-randomized "
                        "workbench. It was trained under the earlier three-sensor contract.")}
                rl = capture(client, seed, f"Archived {label} layout",
                    lambda context, rows=layouts[policy]: deployment_rows(rows, context), selection)
                write_json(args.raw_output / f"capture-{case}-{policy}.json", rl)
                if rl["trajectorySha256"] != baseline["trajectorySha256"]:
                    raise RuntimeError(f"Matched workbench trajectories differ: case {case}, {policy}")
                episodes.append({"id": f"workbench-{policy}-{case}", "policy": policy,
                    "policyLabel": label, "case": case,
                    "label": f"Eight-sector workbench episode {case}", "seed": seed,
                    "rl": rl["view"], "baseline": baseline["view"],
                    "audit": {"sameTrajectories": True, "sameBudget": True,
                        "sameCatalogue": True, "sameSensingDraws": True,
                        "archivedRlContractMaxSensors": 3, "workbenchMaxSensors": 8,
                        "eightSensorRlTrained": False,
                        "trajectorySha256": rl["trajectorySha256"],
                        "baselineTrajectorySha256": baseline["trajectorySha256"]}})
                print(json.dumps({"case": case, "layout": policy, "status": "completed",
                    "rlMeanWarning": rl["view"]["metrics"]["mean_drone_warning_seconds_lower_bound"],
                    "eightSensorMeanWarning": baseline["view"]["metrics"]["mean_drone_warning_seconds_lower_bound"]}),
                    flush=True)
    bundle = {"schema": SCHEMA, "protocol": protocol, "episodes": episodes}
    blob = gzip.compress(json.dumps(bundle, allow_nan=False,
                                    separators=(",", ":")).encode(), mtime=0)
    (args.output / "bundle.json.gz").write_bytes(blob)
    write_json(args.output / "manifest.json", {"schema": SCHEMA,
        "bundle": "bundle.json.gz", "sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": len(blob), "episodeCount": len(episodes),
        "pairedTrajectoriesVerified": True, "protocol": protocol})
    print(f"Saved {len(episodes)} matched workbench comparisons to {args.output}", flush=True)


if __name__ == "__main__":
    main()
