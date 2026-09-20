"""Train a separate warning-reward Blue policy in native Istana; retain every run.

No checkpoint selection/resume, frozen artifacts untouched. Evaluation uses
paired Red and action seeds, independent of training. Frames are observer truth,
not policy inputs. Requires an already-running loopback native Blue bridge.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np

from triad_rl.istana_live import IstanaLiveClient, make_plan, public_planning_inputs
from triad_rl.warning_policy import WarningPolicy, warning_metrics


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def red_centers(context, seed):
    # Same accepted annulus and height as current scene. Rotated radial swarms
    # provide varied approaches without changing the simulator's public priors.
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * math.pi)
    radius = (context["minRadiusCm"] + context["maxRadiusCm"]) / 2
    origin, count = context["objectiveWorldCm"], context["groupCount"]
    return [[origin["x"] + radius * math.cos(phase + 2 * math.pi * i / count),
             origin["y"] + radius * math.sin(phase + 2 * math.pi * i / count),
             origin["z"] + context["heightOffsetCm"]] for i in range(count)]


def episode(client, policy, seed, *, action_seed=None, capture=False, greedy=False):
    red = client.reset(seed)
    context = client.get_blue_context()
    records = []
    if greedy:
        placements = make_plan(context, temporal_public_control=True)["placements"]
    else:
        rng = None if action_seed is None else np.random.default_rng(action_seed)
        placements, records = policy.plan(context, rng=rng)
    client.deploy(placements)
    result = client.place_red(red_centers(red, seed))
    frames = []
    if capture:
        # Observe the actual spawned positions, not requested group centers.
        frames.append({"t": 0., "drones": client.request("observe")["observation"]["drones"]})
    for _ in range(200):
        response = client.step(10 if capture else 1000)
        blue = response["blueObservation"]
        if capture:
            frames.append({"t": blue["elapsedSeconds"], "drones": response["observation"]["drones"]})
        if blue["terminated"] or blue["truncated"]:
            break
    else:
        client.cancel()
        raise RuntimeError("Episode exceeded step bound")
    metrics = warning_metrics(blue)
    return {"seed": seed, "action_seed": action_seed, "placements": placements,
            "metrics": metrics, "warning_evidence": blue["warningEvidenceForEvaluationOnly"],
            "frames": frames, "context": context if capture else None,
            "run_id": red["runId"], "steps": client.completed_steps}, records


def paired_interval(differences):
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(910091)
    samples = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "bootstrap_95_ci": np.quantile(samples, [.025, .975]).tolist(),
            "unit": "paired held-out scenario; single training seed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-cases", type=int, default=8)
    parser.add_argument("--seed", type=int, default=917)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 32 or args.episodes % (4 * args.batch_size) or args.eval_cases < 2:
        parser.error("episodes >=32 and divisible by four batches; at least two evaluation cases")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    source_paths = [Path(__file__), Path(__file__).parent / "triad_rl/warning_policy.py",
                    Path(__file__).parent / "triad_rl/istana_live.py",
                    root / "Source/IstanaOpen/Simulation/BlueTeam/BlueTeamCoordinator.cpp",
                    root / "Source/IstanaOpen/Simulation/BlueTeam/BlueWarningTime.h"]
    protocol = {"schema": "istana.native_warning_training.v1", "episodes": args.episodes,
                "batch_size": args.batch_size, "learning_rate": .12, "training_seed": args.seed,
                "checkpoints": [0] + [args.episodes * i // 4 for i in range(1, 5)],
                "training_seeds": [1700000 + i for i in range(args.episodes)],
                "evaluation_seeds": [2700000 + i for i in range(args.eval_cases)],
                "evaluation_action_seeds": [3700000 + i for i in range(args.eval_cases)],
                "reward": "mean per-drone max(0, zone entry - first detection), seconds; undetected=0; unresolved=error",
                "team_warning": "max(0, first zone entry among all Red - first detection among all Red)",
                "endpoint": "20m objective zone entry, not physical crash",
                "red_policy": "scripted radial swarms, seeded rotation; not learned Red or self-play",
                "blue_policy": "from-scratch masked type/site REINFORCE, stochastic evaluation",
                "baseline": "same untrained policy and non-RL temporal public greedy, same budget and scenarios",
                "selection": "fixed endpoints; all checkpoints retained; visual case index 0 chosen before outcomes",
                "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()}
    write_json(args.output / "protocol.json", protocol)
    started, evaluations, checkpoints = time.monotonic(), [], []
    with IstanaLiveClient(port=args.port) as client, (args.output / "training.jsonl").open("x", encoding="utf-8") as log:
        client.reset(1699999)
        initial_context = client.get_blue_context()
        write_json(args.output / "native_context.json", initial_context)
        policy = WarningPolicy(initial_context, seed=args.seed)
        def evaluate(label, greedy=False):
            rows = []
            for i, (seed, action_seed) in enumerate(zip(protocol["evaluation_seeds"], protocol["evaluation_action_seeds"])):
                run, _ = episode(client, policy, seed, action_seed=action_seed, capture=i == 0, greedy=greedy)
                rows.append(run)
                print(json.dumps({"evaluation": label, "case": i, **run["metrics"]}), flush=True)
            write_json(args.output / f"evaluation-{label}.json", rows)
            summary = {"checkpoint": label, "mean_drone_warning_s": float(np.mean([r["metrics"]["mean_drone_warning_s"] for r in rows])),
                       "team_warning_s": float(np.mean([r["metrics"]["team_warning_s"] for r in rows])),
                       "mean_cost": float(np.mean([r["metrics"]["cost"] for r in rows]))}
            evaluations.append(summary)
            return rows
        def checkpoint(number):
            sha = policy.save(args.output / f"policy-{number:04d}.json")
            checkpoints.append({"episode": number, "sha256": sha, "updates": policy.updates})
            return evaluate(f"{number:04d}")
        initial = checkpoint(0)
        greedy = evaluate("greedy", greedy=True)
        batch = []
        for i, seed in enumerate(protocol["training_seeds"], 1):
            run, records = episode(client, policy, seed)
            batch.append((records, run["metrics"]["mean_drone_warning_s"]))
            log.write(json.dumps({"episode": i, **run}, allow_nan=False) + "\n"); log.flush()
            if i % args.batch_size == 0:
                update = policy.update(batch); batch = []
                print(json.dumps({"episode": i, **update, "wall_seconds": time.monotonic() - started}), flush=True)
            if i in protocol["checkpoints"]:
                final = checkpoint(i)
        comparisons = {}
        for name, control in (("untrained", initial), ("greedy", greedy)):
            comparisons[name] = {metric: paired_interval([a["metrics"][metric] - b["metrics"][metric] for a, b in zip(final, control)])
                                 for metric in ("mean_drone_warning_s", "team_warning_s")}
        write_json(args.output / "summary.json", {"protocol": protocol, "checkpoints": checkpoints,
                   "evaluations": evaluations, "final_minus_control": comparisons,
                   "wall_seconds": time.monotonic() - started,
                   "limitations": "Single-seed pilot, eight default held-out scenarios; synthetic sensor probabilities, no sensing occlusion. Not a real-world response-time or collision validation."})
    print(f"Completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
