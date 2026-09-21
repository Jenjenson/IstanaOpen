"""Versioned, paired native long-approach experiment (separate from old pilot)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from train_warning_live import paired_interval, write_json
from triad_rl.istana_live import IstanaLiveClient, make_plan
from triad_rl.warning_policy import WarningPolicy, warning_metrics
from triad_rl.warning_scenario import (SCENARIO, approach_centers, digest,
                                      scenario_contract, start_diagnostics, saturation_metrics)


def episode(client, policy, seed, *, action_seed=None, placements=None, capture=False, contract=None):
    red = client.reset(seed)
    context = client.get_blue_context()
    physical = scenario_contract(red, context)
    if contract is not None and physical != contract:
        raise ValueError("Physical scenario changed; checkpoint comparison invalid")
    records = []
    if placements is None:
        placements, records = policy.plan(context, rng=None if action_seed is None else np.random.default_rng(action_seed))
    client.deploy(placements)
    centers = approach_centers(red, seed)
    client.place_red(centers)
    initial = client.request("observe")["observation"]["drones"]
    distances = start_diagnostics(context, placements, initial)
    if distances["minimum_universal_clearance_m"] < SCENARIO["minimum_start_clearance_m"]:
        raise ValueError("Red starts too close to a supported sensor's maximum range")
    path_hash = hashlib.sha256()
    path_hash.update(digest(initial).encode())
    frames = [{"t": 0., "drones": initial}] if capture else []
    # Same sampling cadence for EVERY episode/checkpoint, also hashing velocity
    # and positions of every target, independent of detection/Blue layout.
    for _ in range(400):
        response = client.step(100)
        blue, drones = response["blueObservation"], response["observation"]["drones"]
        frame = {"t": blue["elapsedSeconds"], "drones": drones}
        path_hash.update(digest(frame).encode())
        if capture: frames.append(frame)
        if capture and len(frames) % 4 == 0:
            print(json.dumps({"running_seed": seed, "simulated_s": blue["elapsedSeconds"]}), flush=True)
        if blue["terminated"] or blue["truncated"]: break
    else:
        client.cancel()
        raise RuntimeError("Episode exceeded bounded 400-second stepping guard")
    metrics = warning_metrics(blue)
    evidence = blue["warningEvidenceForEvaluationOnly"]
    saturation = saturation_metrics(evidence, context["temporalConfig"]["look_interval_s"])
    if saturation["team_first_look_saturated"]:
        raise ValueError("New benchmark unexpectedly saturated at its first sensing look")
    return {"seed": seed, "action_seed": action_seed, "placements": placements, "metrics": metrics,
            "warning_evidence": evidence, "start_diagnostics": distances, "saturation": saturation,
            "red_centers_world_cm": centers, "initial_states_sha256": digest(initial),
            "trajectory_sha256": path_hash.hexdigest(), "physical_contract_sha256": digest(physical),
            "frames": frames, "context": context if capture else None,
            "run_id": red["runId"], "steps": client.completed_steps}, records


def paired_check(reference, candidate):
    for before, after in zip(reference, candidate, strict=True):
        for key in ("seed", "initial_states_sha256", "trajectory_sha256", "physical_contract_sha256",
                    "red_centers_world_cm"):
            if before[key] != after[key]: raise ValueError(f"Unfair comparison: {key} changed")
        events = lambda row: [(r["droneId"], r["zoneEntrySeconds"]) for r in row["warning_evidence"]]
        if events(before) != events(after): raise ValueError("Red arrival times changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-cases", type=int, default=16)
    parser.add_argument("--seed", type=int, default=918)
    parser.add_argument("--workers", type=int, default=1, help="Independent native scenes on consecutive ports (1..4)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4: parser.error("workers must be 1..4")
    if args.episodes < 32 or args.episodes % (4 * args.batch_size) or args.eval_cases < 8:
        parser.error("At least 32 episodes in four complete batches, and at least 8 evaluation cases")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    sources = [Path(__file__), Path(__file__).parent / "triad_rl/warning_scenario.py",
               Path(__file__).parent / "triad_rl/warning_policy.py", Path(__file__).parent / "triad_rl/istana_live.py",
               root / "Source/IstanaOpen/IstanaGameMode.cpp",
               root / "Source/IstanaOpen/Simulation/BlueTeam/BlueTeamCoordinator.cpp",
               root / "Source/IstanaOpen/Simulation/BlueTeam/BlueWarningTime.h",
               root / "Source/IstanaOpen/Simulation/RedTeam/RedTeamManager.cpp",
               root / "Source/IstanaOpen/Simulation/Swarm/IstanaSwarmSimulation.cpp"]
    protocol = {"schema": "istana.native_warning_approach.v2", "scenario": SCENARIO,
                "episodes": args.episodes, "batch_size": args.batch_size, "learning_rate": .12,
                "training_seed": args.seed, "checkpoints": [0, 1] + [args.episodes*i//4 for i in range(1, 5)],
                "workers": args.workers, "worker_semantics": "All actions sampled serially before a batch; synchronous update in seed order",
                "training_seeds": list(range(1800000, 1800000+args.episodes)),
                "evaluation_seeds": list(range(2800000, 2800000+args.eval_cases)),
                "evaluation_action_seeds": list(range(3800000, 3800000+args.eval_cases)),
                "reward": "team_warning_s: max(0, first 20m zone entry - first detection of any drone), undetected=0",
                "secondary_metric": "mean per-drone warning across all targets, undetected=0",
                "endpoint": SCENARIO["endpoint"], "red_policy": "scripted synthetic sector approaches; native collision-aware motion",
                "selection": "fixed final endpoint; no held-out tuning or checkpoint selection",
                "evaluation": "paired spawn/position/velocity/arrival/physical-contract hashes; stochastic Blue with fixed action seeds",
                "trajectory_sample_interval_s": 5.,
                "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "native_dll_sha256": hashlib.sha256((root / "Binaries/Win64/UnrealEditor-IstanaOpen.dll").read_bytes()).hexdigest()}
    write_json(args.output / "protocol.json", protocol)
    for source in sources:
        archived = args.output / "training-source-snapshot" / source.relative_to(root)
        archived.parent.mkdir(parents=True, exist_ok=True)
        with archived.open("xb") as stream: stream.write(source.read_bytes())
    started = time.monotonic()
    with ExitStack() as stack:
        clients = [stack.enter_context(IstanaLiveClient(port=args.port+i)) for i in range(args.workers)]
        client = clients[0]
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=args.workers))
        red = client.reset(1799999)
        context = client.get_blue_context()
        contract = scenario_contract(red, context)
        write_json(args.output / "physical-contract.json", contract)
        write_json(args.output / "native_context.json", context)
        policy = WarningPolicy(context, seed=args.seed)
        def run_jobs(jobs, label):
            # Each worker exclusively owns one native clock/socket. Policy RNG
            # and optimizer are never shared with concurrent episode execution.
            def worker(index):
                completed = []
                for j in range(index, len(jobs), len(clients)):
                    try:
                        row, _ = episode(clients[index], policy, contract=contract, **jobs[j])
                    except Exception as error:
                        failure = {"phase": label, "case": j, "job": jobs[j], "error": str(error)}
                        try:
                            failure["native"] = clients[index].request("observe")
                            failure["blue"] = clients[index].observe_blue()
                        except Exception as diagnostic_error:
                            failure["diagnostic_error"] = str(diagnostic_error)
                        write_json(args.output / f"failure-{label}-{j}.json", failure)
                        raise
                    completed.append((j, row))
                    print(json.dumps({"phase": label, "case": j, **row["metrics"]}), flush=True)
                return completed
            futures = [pool.submit(worker, i) for i in range(len(clients))]
            return [row for _, row in sorted(pair for future in futures for pair in future.result())]
        if args.preflight:
            # Predeclared diagnostic controls: same type/count/cost, rotated
            # layouts. They are NOT the untrained RL baseline or optimized labels.
            layouts = {"north_arc": [16, 20, 24], "south_arc": [16, 28, 24]}
            runs = {}
            for label, sites in layouts.items():
                rows = []
                for seed in (4800000, 4800001, 4800002, 4800008):
                    row, _ = episode(client, policy, seed, placements=[{"profileId": "eo", "siteId": s,
                                                                        "yawDeg": 0., "pitchDeg": 0.} for s in sites],
                                     capture=True, contract=contract)
                    rows.append(row)
                    write_json(args.output / f"{label}-{seed}.json", row)
                    print(json.dumps({"preflight": label, "seed": seed, **row["metrics"],
                                      "start_clearance_m": row["start_diagnostics"]["minimum_universal_clearance_m"]}), flush=True)
                runs[label] = rows
            paired_check(runs["north_arc"], runs["south_arc"])
            deltas = [a["metrics"]["team_warning_s"]-b["metrics"]["team_warning_s"] for a,b in zip(*runs.values())]
            write_json(args.output / "preflight.json", {"paired_paths_identical": True, "team_warning_deltas_s": deltas,
                       "maximum_absolute_same_capability_placement_gap_s": max(map(abs, deltas)),
                       "wall_seconds": time.monotonic()-started})
            return
        summaries, checkpoints, reference = [], [], None
        def evaluate(number=None, greedy=False):
            nonlocal reference
            label = "greedy" if greedy else f"{number:04d}"
            jobs = []
            control = make_plan(context, temporal_public_control=True)["placements"] if greedy else None
            for i, (seed, action_seed) in enumerate(zip(protocol["evaluation_seeds"], protocol["evaluation_action_seeds"])):
                placements = control if greedy else policy.plan(context, rng=np.random.default_rng(action_seed))[0]
                jobs.append(dict(seed=seed, action_seed=action_seed, placements=placements, capture=i==0))
            rows = run_jobs(jobs, label)
            if reference is None: reference = rows
            else: paired_check(reference, rows)
            write_json(args.output / f"evaluation-{label}.json", rows)
            summaries.append({"checkpoint": label, **{metric: float(np.mean([r["metrics"][metric] for r in rows]))
                               for metric in ("team_warning_s", "mean_drone_warning_s")},
                              "mean_cost": float(np.mean([r["metrics"]["cost"] for r in rows])),
                              "first_look_saturated_cases": sum(r["saturation"]["team_first_look_saturated"] for r in rows)})
            return rows
        def checkpoint(number):
            sha = policy.save(args.output / f"policy-{number:04d}.json")
            checkpoints.append({"episode": number, "sha256": sha, "updates": policy.updates})
            return evaluate(number)
        initial = checkpoint(0)
        greedy = evaluate(greedy=True)
        batch = []
        with (args.output / "training.jsonl").open("x", encoding="utf-8") as log:
            for offset in range(0, args.episodes, args.batch_size):
                jobs, actions = [], []
                for seed in protocol["training_seeds"][offset:offset+args.batch_size]:
                    placements, records = policy.plan(context)
                    jobs.append(dict(seed=seed, placements=placements)); actions.append(records)
                rows = run_jobs(jobs, f"training-{offset+1}-{offset+args.batch_size}")
                for i, (row, records) in enumerate(zip(rows, actions), offset+1):
                    log.write(json.dumps({"episode": i, **row}, allow_nan=False)+"\n"); log.flush()
                    batch.append((records, row["metrics"]["team_warning_s"]))
                # No optimizer update has happened at Episode 1. Preserve that
                # exact policy, not a fabricated weak starting placement.
                if offset == 0: checkpoint(1)
                update = policy.update(batch); batch = []
                number = offset + args.batch_size
                print(json.dumps({"episode": number, **update, "wall_seconds": time.monotonic()-started}), flush=True)
                if number in protocol["checkpoints"]: final = checkpoint(number)
        comparison = {name: {metric: paired_interval([a["metrics"][metric]-b["metrics"][metric] for a,b in zip(final, control)])
                             for metric in ("team_warning_s", "mean_drone_warning_s")}
                      for name, control in (("untrained", initial), ("greedy", greedy))}
        write_json(args.output / "summary.json", {"protocol": protocol, "checkpoints": checkpoints,
                   "evaluations": summaries, "final_minus_control": comparison, "wall_seconds": time.monotonic()-started,
                   "limitations": "Single training seed; synthetic sector routes and radial sensing without occlusion; not real-world protection advice."})
    print(f"Completed: {args.output}", flush=True)


if __name__ == "__main__": main()
