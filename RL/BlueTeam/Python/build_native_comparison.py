"""Record paired RL versus fixed common-sense placements in native Unreal.

Launch the ordinary ``-IstanaBlueLive`` scene (without warning/demo approach
overrides). The published temporal checkpoints require its exact temporal
configuration. This captures existing policies; it neither trains nor searches
for favorable examples. A separate loopback bridge avoids interrupting Live.
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

from train_warning_live import red_centers
from triad_rl.istana_live import IstanaLiveClient, make_plan


ROOT = Path(__file__).resolve().parents[3]
BLUE = ROOT / "RL/BlueTeam"
SCHEMA = "istana.native_comparison_capture.v1"
POLICIES = {"406": "Temporal RL · policy A", "407": "Temporal RL · policy B",
            "408": "Temporal RL · policy C"}
SEEDS = (2800001, 2800002, 2800003)
SOURCES = (
    "RL/BlueTeam/Python/build_native_comparison.py",
    "RL/BlueTeam/Python/train_warning_live.py",
    "RL/BlueTeam/Python/triad_rl/istana_live.py",
    "RL/BlueTeam/Python/triad_rl/common_sense.py",
    "RL/BlueTeam/Python/triad_rl/directional_inputs.py",
    "RL/BlueTeam/Python/triad_rl/temporal_policy.py",
    "Source/IstanaOpen/IstanaGameMode.cpp",
    "Source/IstanaOpen/Simulation/BlueTeam/BlueTeamCoordinator.cpp",
    "Source/IstanaOpen/Simulation/BlueTeam/BlueWarningTime.h",
    "Source/IstanaOpen/Simulation/RedTeam/RedTeamManager.cpp",
    "Source/IstanaOpen/Simulation/Swarm/IstanaSwarmSimulation.cpp",
    "Source/IstanaOpen/Simulation/Sensing/DirectionalSensorModel.cpp",
    "Source/IstanaOpen/Simulation/Sensing/DirectionalSensorModel.h",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame(red, blue, origin):
    return {"time": blue["elapsedSeconds"], "completedSteps": blue["completedSteps"],
            "threats": [{"id": f"drone-{drone['droneId']}",
                "position": [(drone["positionCm"][key] - origin[key]) / 100
                             for key in ("x", "y", "z")],
                "active": drone.get("bActive", True), "observer_truth": True}
                for drone in sorted(red["drones"], key=lambda row: row["droneId"])],
            "detections": [], "tracks": deepcopy(blue["publicSnapshot"]["tracks"])}


def trajectory_digest(frames):
    """Exclude all sensor outcomes so equality proves the same observed flight."""
    return digest([{key: row[key] for key in ("time", "completedSteps", "threats")}
                   for row in frames])


def target_results(evidence, lead_time):
    rows = []
    for row in evidence:
        entry, confirmation = row["zoneEntrySeconds"], row["firstConfirmationSeconds"]
        rows.append({"id": f"drone-{row['droneId']}",
            "first_detection": row["firstDetectionSeconds"],
            "first_confirmation": confirmation, "time_to_zone": entry,
            "warning_time": row["warningSeconds"],
            "timely_confirmed": entry is not None and confirmation is not None
                and confirmation <= entry - lead_time + 1e-8,
            "unresolved": entry is None})
    return rows


def capture(client, seed, policy, *, step_batch=20):
    started = time.monotonic()
    red = client.reset(seed)
    context = client.get_blue_context()
    checkpoint = BLUE / f"Results/temporal-v6-pilot/training/seed-{policy}/last"
    plan = (make_plan(context, common_sense=True) if policy == "common_sense"
            else make_plan(context, checkpoint=checkpoint))
    client.deploy(plan["placements"])
    centers = red_centers(red, seed)
    client.place_red(centers)
    blue = client.observe_blue()
    frames = [frame(client.request("observe")["observation"], blue, context["worldOriginCm"])]
    while not (blue["terminated"] or blue["truncated"]):
        if client.completed_steps >= 10000:
            client.cancel()
            raise RuntimeError("Native episode exceeded 10,000 steps")
        response = client.step(step_batch)
        blue = response["blueObservation"]
        frames.append(frame(response["observation"], blue, context["worldOriginCm"]))
    if not blue.get("metricsAvailable") or not blue.get("warningEvidenceForEvaluationOnly"):
        raise RuntimeError("Native episode ended without sensor evidence")
    trajectory = trajectory_digest(frames)
    evidence = blue["warningEvidenceForEvaluationOnly"]
    results = target_results(evidence, context["temporalConfig"]["lead_time_s"])
    indexed = {row["id"]: row for row in results}
    for row in frames:
        for threat in row["threats"]:
            result = indexed[threat["id"]]
            threat["detected"] = (result["first_detection"] is not None
                                  and result["first_detection"] <= row["time"] + 1e-8)
            threat["confirmed"] = (result["first_confirmation"] is not None
                                   and result["first_confirmation"] <= row["time"] + 1e-8)
    label = "Common-sense placement" if policy == "common_sense" else POLICIES[policy]
    selection = deepcopy(plan["recommendation"]["selection"])
    if "checkpoint" in selection:
        selection["checkpoint"] = str(checkpoint.relative_to(ROOT)).replace("\\", "/")
    outcome = "unresolved" if any(row["unresolved"] for row in results) else "completed"
    metrics = {**blue["metrics"], "return": blue["reward"], "target_results": results,
               "outcome": outcome, "duration_seconds": blue["elapsedSeconds"]}
    view = {"mode": "comparison", "label": label,
        "coordinateLabel": "Native Istana simulation · objective-relative metres",
        "catalogue": context["catalogue"], "placements": blue["publicSnapshot"]["placements"],
        "sites": context["publicSnapshot"]["sites"],
        "blockedSites": context["publicSnapshot"]["blocked_sites"],
        "budget": context["publicSnapshot"]["budget_total"],
        "maxSensors": context["publicSnapshot"]["max_sites"],
        "objectiveRadius": context["temporalConfig"]["objective_radius_m"],
        "surfaceMounted": True, "frames": frames, "metrics": metrics,
        "decisions": plan["recommendation"]["decisions"], "selection": selection,
        "seed": seed, "weather": context["publicSnapshot"]["weather"],
        "policy": label, "outcome": outcome, "ended": True,
        "warningEvidence": evidence,
        "audit": {"recorded": True, "native_unreal": True, "live_unreal": False,
            "policy_truth_access": False, "training_performed": False,
            "trajectorySha256": trajectory, "nativeRunId": red["runId"],
            "completedSteps": client.completed_steps}}
    return {"view": view, "context": context, "redContext": red,
            "redCentersWorldCm": centers, "trajectorySha256": trajectory,
            "wallSeconds": time.monotonic() - started}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--native-project", type=Path, required=True,
                        help="Checkout containing the actual compiled Unreal module")
    parser.add_argument("--native-runtime-note", default="Compiled module from --native-project; native source and binary hashes are recorded separately from the Python checkout.",
                        help="Optional factual provenance note for the compiled runtime")
    parser.add_argument("--output", type=Path,
                        default=BLUE / "Results/native-placement-comparison")
    parser.add_argument("--raw-output", type=Path, default=ROOT / "Saved/NativeComparisonCapture",
                        help="Local full captures retained separately from the portable bundle")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    args.raw_output.mkdir(parents=True, exist_ok=False)
    binary = args.native_project / "Binaries/Win64/UnrealEditor-IstanaOpen.dll"
    protocol = {"schema": SCHEMA, "createdUtc": datetime.now(timezone.utc).isoformat(),
        "baseCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sourceSha256": {path: file_digest(ROOT / path) for path in SOURCES},
        "nativeSourceSha256": {path: file_digest(args.native_project / path)
                               for path in SOURCES if path.startswith("Source/")},
        "nativeBinarySha256": file_digest(binary),
        "nativeRuntimeNote": args.native_runtime_note,
        "nativeLaunchFlags": ["/Game/Maps/Istana", "-game", "-IstanaBlueLive", "-nullrhi", "-unattended"],
        "episodeSeeds": list(SEEDS), "captureStepBatch": 20,
        "redPolicy": "Seeded rotation of equally spaced swarm centers; current native scripted objective-following movement. No learned Red placement or self-play is claimed.",
        "bluePolicy": "Published temporal RL policies with the merged public-prior directional orientation adapter; type/site choices are learned, orientations are not learned.",
        "baseline": "Fixed affordable coverage rule; same baseline layout for all policies and episodes sharing this public context.",
        "selection": "Three scenario seeds fixed before evaluation; all three published RL policies retained, irrespective of outcome.",
        "warningDefinition": "Per-drone max(0, zone entry - first detection), undetected contributes zero; unresolved entries are reported separately.",
        "sensingDrawRule": "Native indexed Uniform(episode seed, drone ID, site ID, sensing look, salt); independent of layout order and step batching.",
        "checkpointSha256": {policy: {name: file_digest(BLUE / f"Results/temporal-v6-pilot/training/seed-{policy}/last" / name)
                                      for name in ("checkpoint.json", "arrays.npz")}
                             for policy in POLICIES}}
    write_json(args.output / "protocol.json", protocol)
    episodes, fixed_baseline = [], None
    with IstanaLiveClient(args.port, timeout=120.) as client:
        for case, seed in enumerate(SEEDS, 1):
            captured = {}
            for policy in (*POLICIES, "common_sense"):
                print(json.dumps({"case": case, "policy": policy, "status": "capturing"}), flush=True)
                run = capture(client, seed, policy)
                captured[policy] = run
                write_json(args.raw_output / f"capture-{case}-{policy}.json", run)
                print(json.dumps({"case": case, "policy": policy, "status": "completed",
                    "seconds": run["wallSeconds"], "metrics": run["view"]["metrics"]
                        | {"target_results": "retained in capture"}}), flush=True)
            baseline = captured["common_sense"]
            layout = baseline["view"]["placements"]
            if fixed_baseline is None:
                fixed_baseline = layout
            elif fixed_baseline != layout:
                raise RuntimeError("Baseline changed between identical public scenarios")
            for policy in POLICIES:
                rl = captured[policy]
                if rl["trajectorySha256"] != baseline["trajectorySha256"]:
                    raise RuntimeError(f"Paired native trajectories differ: case {case}, policy {policy}")
                if rl["context"]["catalogue"] != baseline["context"]["catalogue"]:
                    raise RuntimeError("Paired native catalogues differ")
                episodes.append({"id": f"native-{policy}-{case}", "policy": policy,
                    "policyLabel": POLICIES[policy], "case": case,
                    "label": f"Perimeter approach {case}", "seed": seed,
                    "rl": rl["view"], "baseline": baseline["view"],
                    "context": rl["context"], "redContext": rl["redContext"],
                    "audit": {"sameTrajectories": True, "sameBudget": True,
                        "sameCatalogue": True, "sameSensingDraws": True,
                        "trajectorySha256": rl["trajectorySha256"],
                        "baselineTrajectorySha256": baseline["trajectorySha256"]}})
    # The native generator is intentionally deterministic; gzip headers must not
    # introduce timestamp noise when identical evidence is packaged again.
    bundle = {"schema": SCHEMA, "protocol": protocol, "episodes": episodes}
    blob = gzip.compress(json.dumps(bundle, allow_nan=False, separators=(",", ":")).encode(), mtime=0)
    (args.output / "bundle.json.gz").write_bytes(blob)
    write_json(args.output / "manifest.json", {"schema": SCHEMA,
        "bundle": "bundle.json.gz", "sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": len(blob), "episodeCount": len(episodes), "pairedTrajectoriesVerified": True,
        "protocol": protocol})
    print(f"Saved {len(episodes)} paired comparisons to {args.output}", flush=True)


if __name__ == "__main__":
    main()
