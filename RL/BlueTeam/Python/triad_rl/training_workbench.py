"""Background native-warning trainer used by the local simulation console.

This is a UI orchestration layer, not a second sensing implementation. Episodes
are executed by the same loopback Unreal bridge, warning evidence is validated
by :mod:`warning_policy`, and the optional warm start only changes initial
policy logits. No Red truth is used when constructing a starting layout.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Callable

import numpy as np

from .common_sense import plan_common_sense
from .directional_inputs import apply_placement, build_observation
from .istana_live import IstanaLiveClient, public_planning_inputs
from .trained_models import COMPARISON_SCHEMA, TrainedModelRegistry, validate_model_name
from .warning_algorithms import TRAINING_ALGORITHMS
from .warning_policy import WarningPolicy, warning_metrics


INITIALIZATIONS = {
    "untrained": "Untrained random policy",
    "directional_balanced_5": "Five directional sensors · balanced approaches",
    "directional_public_5": "Five directional sensors · public-prior weighted",
}
BALANCED_LANE_YAWS = (0., 180., 90., 270., 45.)


def _angle_distance(left, right):
    return abs((left - right + 180.) % 360. - 180.)


def _balanced_lane_plan(state, catalogue, directional_ids, count):
    """Choose outward perimeter cameras for five declared synthetic lanes."""
    current, rows = deepcopy(state), []
    sensor_ids = set(directional_ids)
    for desired_yaw in BALANCED_LANE_YAWS[:count]:
        observation = build_observation(current, catalogue)
        candidates = []
        for index, (option, legal) in enumerate(zip(observation["options"], observation["action_mask"])):
            if (not legal or option["stop"] or option["sensor_id"] not in sensor_ids
                    or _angle_distance(option["yaw_deg"], desired_yaw) > 1e-9):
                continue
            x, y = option["position"]
            site_yaw = math.degrees(math.atan2(y, x)) % 360.
            radius = math.hypot(x, y)
            sensor = catalogue[option["sensor_index"]]
            desired_pitch = math.degrees(math.atan2(
                max(0., current["forecast"]["altitude"] - sensor.get("height_m", 4.)), 300.))
            # Outer, same-bearing sites see an approaching drone before it
            # crosses the perimeter. Match pitch to the public flight altitude
            # near a useful pixels-on-target range, never to private Red truth.
            candidates.append(((-_angle_distance(site_yaw, desired_yaw), radius,
                                -abs(option["pitch_deg"] - desired_pitch), -index), index))
        if not candidates:
            raise ValueError(f"No legal directional site can face synthetic lane {desired_yaw:g} degrees")
        action = max(candidates)[1]
        option = observation["options"][action]
        rows.append({"sensor_id": option["sensor_id"], "site_index": option["site_index"],
                     "yaw_deg": option["yaw_deg"], "pitch_deg": option["pitch_deg"]})
        current = apply_placement(current, action, catalogue)
    return rows


def common_sense_start(context, preset: str, *, count: int = 5) -> dict:
    """Build a five-camera public-only start under the native legal contract."""
    if preset not in ("directional_balanced_5", "directional_public_5"):
        raise ValueError("Select an available common-sense starting placement")
    state, catalogue, _ = public_planning_inputs(context)
    if state["max_sites"] < count or state["budget_remaining"] < count:
        raise ValueError(
            "This Unreal scene does not allow five sensors. Restart it with "
            "Tools\\start_blue_live.ps1 -TrainingWorkbench -DelayedDetectionDemo."
        )
    directional_ids = [row["id"] for row in catalogue if row.get("directional") and row["cost"] <= 1.]
    if not directional_ids:
        raise ValueError("The native catalogue has no affordable directional sensor profile")
    planning = deepcopy(state)
    planning["available_sensor_ids"] = directional_ids
    if preset == "directional_balanced_5":
        planning["forecast"]["approach_weights"] = [1 / 8] * 8
        rows = _balanced_lane_plan(planning, catalogue, directional_ids, count)
        selection = {"kind": "common_sense", "rule": "outward perimeter camera per declared synthetic lane",
                     "sensor_model": "directional type/site/yaw/pitch choices",
                     "explanation": "Place one limited-FOV camera on each of five public synthetic approach lanes."}
    else:
        result = plan_common_sense(planning, catalogue)
        rows, selection = result["new_placements"], result["selection"]
    if len(rows) != count:
        raise ValueError(
            f"Only {len(rows)} legal useful directional placements were available; "
            f"five require supported, unblocked sites with enough separation and budget."
        )
    placements = [{"profileId": row["sensor_id"], "siteId": row["site_index"],
                   "yawDeg": row["yaw_deg"], "pitchDeg": row["pitch_deg"]}
                  for row in rows]
    return {
        "preset": preset, "label": INITIALIZATIONS[preset], "placements": placements,
        "selection": selection, "public_only": True,
        "synthetic_lane_yaws_deg": list(BALANCED_LANE_YAWS) if preset == "directional_balanced_5" else None,
        "coverage_intent": (
            "Five limited-FOV cameras face five declared synthetic approach lanes. "
            "This is a sensible benchmark start, not a guarantee of 360-degree or universal detection."
        ),
    }


def _red_centers(context, seed):
    rng = np.random.default_rng(seed)
    radius = (context["minRadiusCm"] + context["maxRadiusCm"]) / 2
    origin, count = context["objectiveWorldCm"], context["groupCount"]
    if count == len(BALANCED_LANE_YAWS):
        angles = [math.radians(yaw + rng.uniform(-4., 4.)) for yaw in BALANCED_LANE_YAWS]
    else:
        phase = rng.uniform(0, 2 * math.pi)
        angles = [phase + 2 * math.pi * i / count for i in range(count)]
    return [[origin["x"] + radius * math.cos(angle),
             origin["y"] + radius * math.sin(angle),
             origin["z"] + context["heightOffsetCm"]] for angle in angles]


def _capture_frame(red, blue, origin):
    return {"time": blue["elapsedSeconds"], "completedSteps": blue["completedSteps"],
            "threats": [{"id": f"drone-{drone['droneId']}",
                "position": [(drone["positionCm"][key] - origin[key]) / 100
                             for key in ("x", "y", "z")],
                "active": drone.get("bActive", True), "observer_truth": True}
                for drone in sorted(red["drones"], key=lambda row: row["droneId"])],
            "detections": [], "tracks": deepcopy(blue["publicSnapshot"]["tracks"])}


def _trajectory_digest(frames):
    # Detection/confirmation annotations depend on the Blue layout and are not
    # Red trajectory truth.  Excluding them makes equality mean what the fair
    # comparison contract claims: same times, positions and active states.
    value = [{"time": row["time"], "completedSteps": row["completedSteps"],
              "threats": [{key: threat[key] for key in
                           ("id", "position", "active", "observer_truth")}
                          for threat in row["threats"]]} for row in frames]
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _target_results(evidence, lead_time):
    return [{"id": f"drone-{row['droneId']}",
        "first_detection": row["firstDetectionSeconds"],
        "first_confirmation": row["firstConfirmationSeconds"],
        "time_to_zone": row["zoneEntrySeconds"], "warning_time": row["warningSeconds"],
        "timely_confirmed": row["firstConfirmationSeconds"] is not None
            and row["firstConfirmationSeconds"] <= row["zoneEntrySeconds"] - lead_time + 1e-8,
        "unresolved": row["zoneEntrySeconds"] is None} for row in evidence]


def run_episode(client, policy, seed, *, action_seed=None, deterministic=False,
                capture_frames=False, should_stop: Callable[[], bool] = lambda: False):
    """Run one complete native episode and return only validated evidence."""
    red = client.reset(seed)
    context = client.get_blue_context()
    rng = None if action_seed is None else np.random.default_rng(action_seed)
    placements, records = policy.plan(context, rng=rng, deterministic=deterministic)
    client.deploy(placements)
    client.place_red(_red_centers(red, seed))
    blue = client.observe_blue() if capture_frames else None
    frames = ([_capture_frame(client.request("observe")["observation"], blue,
                              context["worldOriginCm"])] if capture_frames else [])
    step_batch = 50 if capture_frames else 250
    for _ in range(2000 if capture_frames else 400):
        if should_stop():
            client.cancel()
            raise InterruptedError("Training stopped after the current native action")
        response = client.step(step_batch)
        blue = response["blueObservation"]
        if capture_frames:
            frames.append(_capture_frame(response["observation"], blue, context["worldOriginCm"]))
        if blue["terminated"] or blue["truncated"]:
            break
    else:
        client.cancel()
        raise RuntimeError("Native training episode exceeded its fixed step bound")
    metrics = warning_metrics(blue)
    evidence = blue["warningEvidenceForEvaluationOnly"]
    targets = _target_results(evidence, context["temporalConfig"]["lead_time_s"])
    if capture_frames:
        indexed = {row["id"]: row for row in targets}
        for frame in frames:
            for threat in frame["threats"]:
                target = indexed[threat["id"]]
                threat["detected"] = (target["first_detection"] is not None
                                      and target["first_detection"] <= frame["time"] + 1e-8)
                threat["confirmed"] = (target["first_confirmation"] is not None
                                       and target["first_confirmation"] <= frame["time"] + 1e-8)
    return {
        "seed": seed, "action_seed": action_seed, "placements": placements,
        "public_placements": blue["publicSnapshot"]["placements"],
        "metrics": metrics, "native_metrics": deepcopy(blue["metrics"]),
        "reward": blue["reward"], "elapsed_seconds": blue["elapsedSeconds"],
        "warning_evidence": evidence, "target_results": targets, "frames": frames,
        "trajectory_sha256": _trajectory_digest(frames) if capture_frames else None,
        "run_id": red["runId"], "steps": client.completed_steps,
        "context": context,
    }, records


class _FixedPlacementPolicy:
    def __init__(self, placements):
        self.placements = deepcopy(placements)

    def plan(self, _context, **_):
        return deepcopy(self.placements), []


def _comparison_view(run, label, selection):
    context, metrics = run["context"], run["metrics"]
    native = {**run["native_metrics"], "return": run["reward"],
              "target_results": run["target_results"], "outcome": "completed",
              "duration_seconds": run["elapsed_seconds"]}
    return {"mode": "comparison", "label": label,
        "coordinateLabel": "Native training workbench · objective-relative metres",
        "catalogue": context["catalogue"], "placements": run["public_placements"],
        "sites": context["publicSnapshot"]["sites"],
        "blockedSites": context["publicSnapshot"].get("blocked_sites", []),
        "budget": context["publicSnapshot"]["budget_total"],
        "maxSensors": context["publicSnapshot"]["max_sites"],
        "objectiveRadius": context["temporalConfig"]["objective_radius_m"],
        "surfaceMounted": True, "frames": run["frames"], "metrics": native,
        "decisions": [], "selection": deepcopy(selection), "seed": run["seed"],
        "weather": context["publicSnapshot"]["weather"], "policy": label,
        "outcome": "completed", "ended": True, "warningEvidence": run["warning_evidence"],
        "audit": {"recorded": True, "native_unreal": True, "live_unreal": False,
            "policy_truth_access": False, "training_performed": True,
            "trajectorySha256": run["trajectory_sha256"], "nativeRunId": run["run_id"],
            "completedSteps": run["steps"]}}


def _viewer(run, episode, total):
    context, metrics = run["context"], run["metrics"]
    native_metrics = {
        "detected_fraction": metrics["detected_fraction"],
        "confirmed_fraction": None,
        "mean_drone_warning_seconds_lower_bound": metrics["mean_drone_warning_s"],
        "team_warning_seconds_lower_bound": metrics["team_warning_s"],
        "cost": metrics["cost"], "targets": metrics["targets"],
    }
    return {
        "mode": "training", "label": f"RL training · episode {episode} / {total}",
        "coordinateLabel": "Native Unreal training · objective-relative XY metres",
        "catalogue": context["catalogue"], "placements": run["public_placements"],
        "sites": context["publicSnapshot"]["sites"],
        "blockedSites": context["publicSnapshot"].get("blocked_sites", []),
        "budget": context["publicSnapshot"]["budget_total"],
        "objectiveRadius": context["temporalConfig"]["objective_radius_m"],
        "frames": [{"time": 0., "threats": [], "detections": [], "tracks": [], "completedSteps": 0}],
        "metrics": native_metrics, "outcome": "training episode complete", "ended": True,
    }


def _starting_view(context, layout, total):
    catalogue = context["catalogue"]
    indices = {row["id"]: index for index, row in enumerate(catalogue)}
    sites = context["publicSnapshot"]["sites"]
    placements = []
    for row in layout["placements"]:
        sensor_index = indices[row["profileId"]]
        sensor = catalogue[sensor_index]
        placements.append({"sensor_id": row["profileId"], "sensor_index": sensor_index,
                           "position": sites[row["siteId"]], "cost": sensor["cost"],
                           "yaw_deg": row["yawDeg"], "pitch_deg": row["pitchDeg"]})
    return {"mode": "training", "label": "Selected common-sense starting placement",
            "coordinateLabel": "Native Unreal training · public-only initialization",
            "catalogue": catalogue, "placements": placements, "sites": sites,
            "blockedSites": context["publicSnapshot"].get("blocked_sites", []),
            "budget": context["publicSnapshot"]["budget_total"],
            "objectiveRadius": context["temporalConfig"]["objective_radius_m"],
            "frames": [{"time": 0., "threats": [], "detections": [], "tracks": [], "completedSteps": 0}],
            "metrics": None, "outcome": "initialization", "ended": True, "totalEpisodes": total}


class TrainingManager:
    """Own one cancellable background training run and its serial bridge."""
    def __init__(self, bridge_port=8765, output_root=None, *, client_factory=IstanaLiveClient,
                 policy_factory=None, policy_factories=None, episode_runner=run_episode, registry=None):
        self.bridge_port = bridge_port
        self.output_root = Path(output_root or Path.cwd() / "Saved/WarningTraining")
        self.client_factory = client_factory
        if policy_factory is not None and policy_factories is not None:
            raise ValueError("Choose either one injected policy factory or an algorithm factory map")
        self.policy_factories = ({key: row["factory"] for key, row in TRAINING_ALGORITHMS.items()}
                                 if policy_factories is None else dict(policy_factories))
        if policy_factory is not None:
            self.policy_factories["reinforce"] = policy_factory
        self.episode_runner = episode_runner
        self.registry = registry or TrainedModelRegistry(self.output_root / "models")
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self._status = self._blank()

    def _blank(self):
        return {"running": False, "phase": "idle", "episode": 0, "totalEpisodes": 0,
                "history": [], "checkpoints": [], "initialization": "untrained",
                "initializationLabel": INITIALIZATIONS["untrained"], "initialLayout": None,
                "algorithm": "reinforce", "algorithmLabel": TRAINING_ALGORITHMS["reinforce"]["label"],
                "modelName": "", "bestEpisode": None, "bestWarningSeconds": None,
                "registeredModel": None, "viewer": None, "outputDirectory": None,
                "error": "", "stopRequested": False}

    def status(self):
        with self.lock:
            return deepcopy(self._status)

    def _set(self, **values):
        with self.lock:
            self._status.update(values)

    @staticmethod
    def validate(config):
        required = {"name", "algorithm", "episodes", "batchSize", "seed", "initialization"}
        if not isinstance(config, dict) or set(config) != required:
            raise ValueError(
                "Training requires a model name, algorithm, episodes, batchSize, seed and initialization")
        episodes, batch = config["episodes"], config["batchSize"]
        seed, initialization = config["seed"], config["initialization"]
        if type(episodes) is not int or not 4 <= episodes <= 512:
            raise ValueError("Episodes must be an integer from 4 to 512")
        if type(batch) is not int or not 2 <= batch <= 32 or episodes % batch:
            raise ValueError("Batch size must be 2..32 and divide the episode count")
        if type(seed) is not int or not -(2**31) <= seed < 2**31:
            raise ValueError("Training seed must be a signed 32-bit integer")
        if initialization not in INITIALIZATIONS:
            raise ValueError("Select an available training initialization")
        if config["algorithm"] not in TRAINING_ALGORITHMS:
            raise ValueError("Select an available training algorithm")
        result = deepcopy(config)
        result["name"] = validate_model_name(config["name"])
        return result

    def start(self, config):
        config = self.validate(config)
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                raise ValueError("A training run is already active")
            if config["algorithm"] not in self.policy_factories:
                raise ValueError("The selected training algorithm is unavailable")
            self.registry.ensure_available(config["name"])
            self.stop_event.clear()
            self._status = self._blank()
            self._status.update(running=True, phase="connecting", totalEpisodes=config["episodes"],
                                modelName=config["name"],
                                algorithm=config["algorithm"],
                                algorithmLabel=TRAINING_ALGORITHMS[config["algorithm"]]["label"],
                                initialization=config["initialization"],
                                initializationLabel=INITIALIZATIONS[config["initialization"]])
            self.thread = threading.Thread(target=self._run, args=(config,), daemon=True,
                                           name="istana-training-workbench")
            self.thread.start()
            return deepcopy(self._status)

    def stop(self):
        with self.lock:
            if not self._status["running"]:
                return deepcopy(self._status)
            self.stop_event.set()
            self._status["stopRequested"] = True
            self._status["phase"] = "stopping"
            return deepcopy(self._status)

    def close(self):
        self.stop_event.set()

    def _run(self, config):
        started = time.monotonic()
        stamp = datetime.now(timezone.utc).strftime("console-%Y%m%d-%H%M%S-%f")
        output = self.output_root / stamp
        client = None
        try:
            client = self.client_factory(port=self.bridge_port, timeout=120.)
            client.reset(1600000)
            context = client.get_blue_context()
            policy = self.policy_factories[config["algorithm"]](context, seed=config["seed"])
            start_layout = None
            if config["initialization"] != "untrained":
                start_layout = common_sense_start(context, config["initialization"])
                start_layout["warm_start"] = policy.initialize_from_placements(start_layout["placements"])
            output.mkdir(parents=True, exist_ok=False)
            self._set(phase="training", initialLayout=start_layout, outputDirectory=str(output),
                      viewer=_starting_view(context, start_layout, config["episodes"]) if start_layout else None)
            (output / "configuration.json").write_text(json.dumps({
                "schema": "istana.console_warning_training.v1", **config,
                "bridgePort": self.bridge_port, "initialLayout": start_layout,
                "warningMetric": "mean per-drone max(0, 20m zone arrival - first detection); undetected=0",
                "limitations": "Synthetic native simulation. Limited-FOV layouts do not guarantee universal detection."
            }, indent=2, allow_nan=False), encoding="utf-8")
            checkpoints = [0]
            policy.save(output / "policy-0000.json")
            self._set(checkpoints=checkpoints)
            batch = []
            best_run = best_policy = best_path = None
            best_episode = None
            milestones = {config["episodes"] * i // 4 for i in range(1, 5)}
            with (output / "training.jsonl").open("x", encoding="utf-8") as log:
                for number in range(1, config["episodes"] + 1):
                    if self.stop_event.is_set():
                        raise InterruptedError("Training stopped before the next episode")
                    run, records = self.episode_runner(
                        client, policy, 1700000 + number - 1,
                        action_seed=(config["seed"] & 0xffffffff) * 100000 + number,
                        should_stop=self.stop_event.is_set)
                    reward = run["metrics"]["mean_drone_warning_s"]
                    batch.append((records, reward))
                    entry = {"episode": number, "reward": reward,
                             "teamWarningSeconds": run["metrics"]["team_warning_s"],
                             "detectedFraction": run["metrics"]["detected_fraction"],
                             "cost": run["metrics"]["cost"]}
                    log.write(json.dumps({**entry, "runId": run["run_id"],
                                          "placements": run["placements"]}, allow_nan=False) + "\n")
                    log.flush()
                    if number % config["batchSize"] == 0:
                        update = policy.update(batch)
                        batch = []
                        entry["update"] = update
                        validation, _ = self.episode_runner(
                            client, policy, 2700000, action_seed=3700000,
                            deterministic=True, should_stop=self.stop_event.is_set)
                        score = validation["metrics"]["mean_drone_warning_s"]
                        entry["validationWarningSeconds"] = score
                        if best_run is None or score > best_run["metrics"]["mean_drone_warning_s"]:
                            candidate_path = output / f"best-policy-{number:04d}.json"
                            policy.save(candidate_path)
                            best_run, best_policy, best_path, best_episode = (
                                validation, deepcopy(policy), candidate_path, number)
                            self._set(bestEpisode=number, bestWarningSeconds=score)
                    history = self.status()["history"]
                    history.append(entry)
                    self._set(episode=number, history=history, viewer=_viewer(run, number, config["episodes"]))
                    if number in milestones:
                        policy.save(output / f"policy-{number:04d}.json")
                        checkpoints = self.status()["checkpoints"] + [number]
                        self._set(checkpoints=checkpoints)
            if best_policy is None:
                raise RuntimeError("Training completed without a held-out best checkpoint")
            self._set(phase="evaluating")
            final, _ = self.episode_runner(
                client, best_policy, 2700000, action_seed=3700000, deterministic=True,
                capture_frames=True, should_stop=self.stop_event.is_set)
            baseline_layout = common_sense_start(context, "directional_balanced_5")
            baseline, _ = self.episode_runner(
                client, _FixedPlacementPolicy(baseline_layout["placements"]), 2700000,
                action_seed=3700000, deterministic=True, capture_frames=True,
                should_stop=self.stop_event.is_set)
            if final["trajectory_sha256"] != baseline["trajectory_sha256"]:
                raise RuntimeError("Best-model and baseline comparison trajectories differ")
            model_id = self.registry.identifier_for(config["name"])
            algorithm_label = TRAINING_ALGORITHMS[config["algorithm"]]["label"]
            policy_selection = {"label": f"{config['name']} · {algorithm_label}",
                "kind": "trained_warning_policy", "algorithm": config["algorithm"],
                "explanation": (
                    f"{algorithm_label} best checkpoint selected by mean per-drone warning time on the fixed "
                    "held-out native episode after each policy update.")}
            baseline_selection = {"label": INITIALIZATIONS["directional_balanced_5"],
                "kind": "fixed_directional_workbench_start",
                "explanation": baseline_layout["selection"]["explanation"]}
            comparison_episode = {"schema": COMPARISON_SCHEMA, "id": model_id,
                "policy": model_id, "policyLabel": f"{config['name']} · {algorithm_label}", "case": 1,
                "label": f"{config['name']} · held-out native episode", "seed": 2700000,
                "rl": _comparison_view(final, f"{config['name']} · {algorithm_label}", policy_selection),
                "baseline": _comparison_view(
                    baseline, INITIALIZATIONS["directional_balanced_5"], baseline_selection),
                "audit": {"sameTrajectories": True, "sameBudget": True,
                    "sameCatalogue": True, "sameSensingDraws": True,
                    "trajectorySha256": final["trajectory_sha256"],
                    "baselineTrajectorySha256": baseline["trajectory_sha256"]}}
            registered = self.registry.register(name=config["name"], policy_path=best_path,
                comparison_episode=comparison_episode, metadata={
                    "createdUtc": datetime.now(timezone.utc).isoformat(),
                    "trainingOutput": str(output), "trainingEpisodes": config["episodes"],
                    "bestEpisode": best_episode, "bestWarningSeconds":
                        final["metrics"]["mean_drone_warning_s"],
                    "algorithm": config["algorithm"], "algorithmLabel": algorithm_label,
                    "evaluationSeed": 2700000, "initialization": config["initialization"],
                    "deploymentPlacements": final["placements"]})
            summary = {"schema": "istana.console_warning_training_summary.v1",
                       "modelName": config["name"], "modelId": registered["id"],
                       "algorithm": config["algorithm"], "algorithmLabel": algorithm_label,
                       "episodes": config["episodes"], "initialization": config["initialization"],
                       "bestEpisode": best_episode, "bestEvaluation": final["metrics"],
                       "comparisonBaselineEvaluation": baseline["metrics"],
                       "elapsedWallSeconds": time.monotonic() - started}
            (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
            self._set(running=False, phase="complete", registeredModel={
                "id": registered["id"], "name": registered["name"],
                "algorithm": config["algorithm"], "algorithmLabel": algorithm_label,
                "bestEpisode": best_episode,
                "bestWarningSeconds": final["metrics"]["mean_drone_warning_s"]},
                viewer=_viewer(final, best_episode, config["episodes"]))
        except InterruptedError:
            self._set(running=False, phase="stopped", stopRequested=True)
        except Exception as error:
            self._set(running=False, phase="failed", error=str(error))
        finally:
            if client is not None:
                client.close()
