"""Background native-warning trainer used by the local simulation console.

This is a UI orchestration layer, not a second sensing implementation. Episodes
are executed by the same loopback Unreal bridge, warning evidence is validated
by :mod:`warning_policy`. Legacy policies can warm-start their logits; local
PPO refines the chosen layout using paired contractor-relative rewards.
No Red truth is used when constructing a starting layout or planning an edit.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import secrets
from pathlib import Path
import threading
import time
from typing import Callable

import numpy as np

from .common_sense import plan_common_sense
from .directional_inputs import apply_placement, build_observation
from .istana_live import IstanaLiveClient, public_planning_inputs
from .red_policy import FixedRadiusSectorRedPolicy
from .trained_models import COMPARISON_SCHEMA, TrainedModelRegistry, validate_model_name
from .training_evidence import (evaluation_summary, layout_changes, layout_key,
                               legal_layout_probes, paired_training_reward, rollout_entropy)
from .warning_algorithms import TRAINING_ALGORITHMS
from .warning_policy import WarningPolicy, warning_metrics


INITIALIZATIONS = {
    "untrained": "Untrained random policy",
    "directional_balanced_8": "Directional sensors · balanced sectors",
    "directional_public_8": "Directional sensors · public-prior weighted",
}
BALANCED_LANE_YAWS = (0., 180., 90., 270., 45., 225., 135., 315.)
MIN_TRAINING_EPISODES = 4
MAX_TRAINING_EPISODES = 10_000
MIN_TRAINING_SENSORS = 1
MAX_TRAINING_SENSORS = len(BALANCED_LANE_YAWS)


def scenario_panels(config):
    """Persist reproducible, disjoint scenario panels; old runs keep their seeds."""
    base = config.get("scenarioSeed")
    training, validation, test = ((1700000, 2700000, 3800000) if base is None
                                  else (base, base + 10000, base + 20000))
    return (training, list(range(validation, validation + config["validationCases"])),
            list(range(test, test + config["validationCases"])))


def _angle_distance(left, right):
    return abs((left - right + 180.) % 360. - 180.)


def _balanced_lane_plan(state, catalogue, directional_ids, count):
    """Choose outward perimeter cameras on representative coverage bearings."""
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
            raise ValueError(f"No legal directional site can face coverage bearing {desired_yaw:g} degrees")
        action = max(candidates)[1]
        option = observation["options"][action]
        rows.append({"sensor_id": option["sensor_id"], "site_index": option["site_index"],
                     "yaw_deg": option["yaw_deg"], "pitch_deg": option["pitch_deg"]})
        current = apply_placement(current, action, catalogue)
    return rows


def common_sense_start(context, preset: str, *, count: int = 8, allowed_sensor_ids=None) -> dict:
    """Build a selected-count public-only start under the native legal contract."""
    if preset not in ("directional_balanced_8", "directional_public_8"):
        raise ValueError("Select an available common-sense starting placement")
    if type(count) is not int or not MIN_TRAINING_SENSORS <= count <= MAX_TRAINING_SENSORS:
        raise ValueError(
            f"Sensor count must be an integer from {MIN_TRAINING_SENSORS} "
            f"to {MAX_TRAINING_SENSORS}")
    state, catalogue, _ = public_planning_inputs(context)
    if state["max_sites"] < count or state["budget_remaining"] < count:
        raise ValueError(
            f"This Unreal scene cannot fund {count} sensors. Restart it with "
            "Tools\\start_blue_live.ps1 -TrainingWorkbench -DelayedDetectionDemo."
        )
    directional_ids = [row["id"] for row in catalogue if row.get("directional") and row["cost"] <= 1.
                       and row["id"] in state["available_sensor_ids"]
                       and (allowed_sensor_ids is None or row["id"] in allowed_sensor_ids)]
    if not directional_ids:
        raise ValueError("The native catalogue has no affordable directional sensor profile")
    planning = deepcopy(state)
    planning["max_sites"] = count
    planning["available_sensor_ids"] = directional_ids
    if preset == "directional_balanced_8":
        planning["forecast"]["approach_weights"] = [1 / 8] * 8
        rows = _balanced_lane_plan(planning, catalogue, directional_ids, count)
        selection = {"kind": "common_sense", "rule": "outward perimeter cameras centered on eight benchmark sectors",
                     "sensor_model": "directional type/site/yaw/pitch choices",
                     "explanation": (
                         f"Spread {count} limited-FOV camera{'s' if count != 1 else ''} across "
                         "the eight benchmark sectors before randomized approaches are sampled.")}
    else:
        result = plan_common_sense(planning, catalogue)
        rows, selection = result["new_placements"], result["selection"]
    if len(rows) != count:
        raise ValueError(
            f"Only {len(rows)} legal useful directional placements were available; "
            f"{count} require supported, unblocked sites with enough separation and budget."
        )
    placements = [{"profileId": row["sensor_id"], "siteId": row["site_index"],
                   "yawDeg": row["yaw_deg"], "pitchDeg": row["pitch_deg"]}
                  for row in rows]
    return {
        "preset": preset, "label": f"{count} · {INITIALIZATIONS[preset]}",
        "sensor_count": count, "placements": placements,
        "selection": selection, "public_only": True,
        "synthetic_lane_yaws_deg": (list(BALANCED_LANE_YAWS[:count])
                                    if preset == "directional_balanced_8" else None),
        "coverage_yaws_deg": (list(BALANCED_LANE_YAWS[:count])
                              if preset == "directional_balanced_8" else None),
        "coverage_intent": (
            f"{count} limited-FOV camera{'s' if count != 1 else ''} cover representative "
            "bearings before five distinct Red approach sectors are sampled. "
            "This is a sensible benchmark start, not a guarantee of 360-degree or universal detection."
        ),
    }


def _red_scenario(context, seed):
    """Return one reproducible fixed-radius scenario from distinct sectors."""
    return FixedRadiusSectorRedPolicy(seed).select(context)


def _red_centers(context, seed):
    """Compatibility helper retained for callers that only need centers."""
    return _red_scenario(context, seed)["centers"]


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


def _reward_breakdown(run):
    """Explain the native terminal reward without changing the RL objective."""
    native, metrics = run["native_metrics"], run["metrics"]
    targets = metrics["targets"]
    budget = run["context"]["publicSnapshot"]["budget_total"]
    detected = native["detected_fraction"]
    confirmed = native["confirmed_fraction"]
    timely = native["timely_fraction"]
    breached = native["breached_fraction"]
    cost = native["cost"]
    components = [
        {"key": "detected", "value": 2 * detected,
         "label": f"{round(detected * targets)} / {targets} drones detected"},
        {"key": "confirmed", "value": 3 * confirmed,
         "label": f"{round(confirmed * targets)} / {targets} tracks confirmed"},
        {"key": "timely", "value": 5 * timely,
         "label": f"{round(timely * targets)} / {targets} confirmations were timely"},
        {"key": "breached", "value": -5 * breached,
         "label": f"{round(breached * targets)} / {targets} arrivals lacked timely confirmation"},
        {"key": "cost", "value": -cost / budget,
         "label": f"{cost:g} / {budget:g} deployment cost"},
    ]
    total = sum(row["value"] for row in components)
    if not np.isclose(total, run["reward"], atol=1e-7, rtol=0):
        raise RuntimeError("Native reward does not match its detection/timeliness/cost breakdown")
    return {"total": float(total), "components": components,
            "formula": "2·detected + 3·confirmed + 5·timely − 5·late/unconfirmed − cost/budget"}


def run_episode(client, policy, seed, *, action_seed=None, deterministic=False,
                capture_frames=False, should_stop: Callable[[], bool] = lambda: False):
    """Run one complete native episode and return only validated evidence."""
    red = client.reset(seed)
    context = client.get_blue_context()
    rng = None if action_seed is None else np.random.default_rng(action_seed)
    placements, records = policy.plan(context, rng=rng, deterministic=deterministic)
    client.deploy(placements)
    red_decision = _red_scenario(red, seed)
    client.place_red(red_decision["centers"])
    blue = client.observe_blue()
    red_observation = client.request("observe")["observation"]
    # Retain a light 5-second-cadence replay for the Training tab.  Formal
    # comparison capture still uses the denser one-second cadence below.
    frames = [_capture_frame(red_observation, blue, context["worldOriginCm"])]
    step_batch = 50 if capture_frames else 250
    for _ in range(2000 if capture_frames else 400):
        if should_stop():
            client.cancel()
            raise InterruptedError("Training stopped after the current native action")
        response = client.step(step_batch)
        blue = response["blueObservation"]
        red_observation = response["observation"]
        frames.append(_capture_frame(red_observation, blue, context["worldOriginCm"]))
        if blue["terminated"] or blue["truncated"]:
            break
    else:
        client.cancel()
        raise RuntimeError("Native training episode exceeded its fixed step bound")
    metrics = warning_metrics(blue)
    evidence = blue["warningEvidenceForEvaluationOnly"]
    targets = _target_results(evidence, context["temporalConfig"]["lead_time_s"])
    indexed = {row["id"]: row for row in targets}
    for frame in frames:
        for threat in frame["threats"]:
            target = indexed[threat["id"]]
            threat["detected"] = (target["first_detection"] is not None
                                  and target["first_detection"] <= frame["time"] + 1e-8)
            threat["confirmed"] = (target["first_confirmation"] is not None
                                   and target["first_confirmation"] <= frame["time"] + 1e-8)
    red_reward = (red_observation.get("reward") if red_observation.get("bHasReward")
                  else -blue["reward"])
    return {
        "seed": seed, "action_seed": action_seed, "placements": placements,
        "public_placements": blue["publicSnapshot"]["placements"],
        "metrics": metrics, "native_metrics": deepcopy(blue["metrics"]),
        "reward": blue["reward"], "elapsed_seconds": blue["elapsedSeconds"],
        "warning_evidence": evidence, "target_results": targets, "frames": frames,
        "trajectory_sha256": _trajectory_digest(frames),
        "run_id": red["runId"], "steps": client.completed_steps,
        "red_policy": red_decision["policy"], "red_decision": red_decision,
        "red_native_reward": red_reward,
        "sensor_count": getattr(policy, "sensor_count", None) or len(placements),
        "context": context, "red_context": deepcopy(red),
    }, records


class _FixedPlacementPolicy:
    def __init__(self, placements):
        self.placements = deepcopy(placements)
        self.sensor_count = len(placements)

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
        "maxSensors": run.get("sensor_count", context["publicSnapshot"]["max_sites"]),
        "objectiveRadius": context["temporalConfig"]["objective_radius_m"],
        "surfaceMounted": True, "frames": run["frames"], "metrics": native,
        "decisions": [], "selection": deepcopy(selection), "seed": run["seed"],
        "weather": context["publicSnapshot"]["weather"], "policy": label,
        "outcome": "completed", "ended": True, "warningEvidence": run["warning_evidence"],
        "audit": {"recorded": True, "native_unreal": True, "live_unreal": False,
            "policy_truth_access": False, "training_performed": True,
            "trajectorySha256": run["trajectory_sha256"], "nativeRunId": run["run_id"],
            "completedSteps": run["steps"]}}


def _write_json_atomic(path, value):
    """Replace a JSON artifact only after the complete document is on disk."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def _viewer(run, episode, total):
    context, metrics = run["context"], run["metrics"]
    native_metrics = {
        **run.get("native_metrics", {}),
        "detected_fraction": metrics["detected_fraction"],
        "confirmed_fraction": run.get("native_metrics", {}).get("confirmed_fraction"),
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
        "maxSensors": run.get("sensor_count", context["publicSnapshot"]["max_sites"]),
        "objectiveRadius": context["temporalConfig"]["objective_radius_m"],
        "frames": run.get("frames") or [{"time": 0., "threats": [], "detections": [], "tracks": [], "completedSteps": 0}],
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
                "sensorCount": MAX_TRAINING_SENSORS,
                "modelName": "", "bestEpisode": None, "bestWarningSeconds": None,
                "bestObservedEpisode": None, "bestObservedWarningSeconds": None,
                "registeredModel": None, "viewer": None, "outputDirectory": None,
                "initialEvaluation": None, "baselineEvaluation": None,
                "bestValidationWarningSeconds": None, "testEvaluation": None,
                "validationCases": 5, "exploration": {}, "baselineProbe": None,
                "viewerSelection": "latest", "error": "", "stopRequested": False}

    def status(self):
        with self.lock:
            result = {key: value for key, value in self._status.items() if key != "history"}
            history = self._status["history"]
            result["historyCount"] = len(history)
            # Bound polling payloads while retaining complete logs on disk.
            if len(history) > 1000:
                indices = np.linspace(0, len(history) - 201, 800, dtype=int)
                result["history"] = [history[int(index)] for index in indices] + history[-200:]
            else:
                result["history"] = history
            return deepcopy(result)

    def _set(self, **values):
        with self.lock:
            self._status.update(values)

    @staticmethod
    def validate(config):
        required = {"name", "algorithm", "episodes", "batchSize", "seed", "initialization",
                    "sensorCount"}
        optional = {"explorationProbability", "entropyCoefficient", "validationCases",
                    "checkpointInterval", "headroomProbes", "sensorIds", "scenarioSeed", "validationInterval"}
        if (not isinstance(config, dict) or not required <= set(config)
                or set(config) - required - optional):
            raise ValueError(
                "Training requires a model name, algorithm, episodes, batchSize, seed, "
                "sensorCount and initialization")
        episodes, batch = config["episodes"], config["batchSize"]
        seed, initialization = config["seed"], config["initialization"]
        sensor_count = config["sensorCount"]
        if (type(episodes) is not int
                or not MIN_TRAINING_EPISODES <= episodes <= MAX_TRAINING_EPISODES):
            raise ValueError(
                f"Episodes must be an integer from {MIN_TRAINING_EPISODES} "
                f"to {MAX_TRAINING_EPISODES}")
        if type(batch) is not int or not 2 <= batch <= 32 or episodes % batch:
            raise ValueError("Batch size must be 2..32 and divide the episode count")
        if (type(sensor_count) is not int
                or not MIN_TRAINING_SENSORS <= sensor_count <= MAX_TRAINING_SENSORS):
            raise ValueError(
                f"Sensor count must be an integer from {MIN_TRAINING_SENSORS} "
                f"to {MAX_TRAINING_SENSORS}")
        if type(seed) is not int or not -(2**31) <= seed < 2**31:
            raise ValueError("Training seed must be a signed 32-bit integer")
        if initialization not in INITIALIZATIONS:
            raise ValueError("Select an available training initialization")
        if config["algorithm"] not in TRAINING_ALGORITHMS:
            raise ValueError("Select an available training algorithm")
        result = deepcopy(config)
        if config["algorithm"] in ("local_ppo", "paired_bandit"):
            if initialization == "untrained":
                raise ValueError("Layout refinement requires a common-sense starting layout")
            # Freeze fresh scenario panels before the first native outcome.
            # The action seed remains separately controlled by the user.
            result.setdefault("scenarioSeed", 10000000 + secrets.randbelow(2**31 - 10030000))
        if "scenarioSeed" in result and (type(result["scenarioSeed"]) is not int
                or not 0 <= result["scenarioSeed"] <= 2**31 - 30001):
            raise ValueError("scenarioSeed must leave room for disjoint signed-32-bit scenario panels")
        defaults = {"explorationProbability": .2, "entropyCoefficient": .01,
                    "validationCases": 5, "checkpointInterval": 500, "headroomProbes": 4,
                    "validationInterval": config["batchSize"]}
        for key, default in defaults.items():
            result.setdefault(key, default)
        for key, lower, upper in (("explorationProbability", .05, .8),
                                  ("entropyCoefficient", 0., 1.)):
            value = result[key]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not lower <= value <= upper):
                raise ValueError(f"{key} must be a finite number from {lower} to {upper}")
        for key, lower, upper in (("validationCases", 2, 200), ("checkpointInterval", 1, 10000),
                                  ("validationInterval", 1, 10000),
                                  ("headroomProbes", 0, 8)):
            if type(result[key]) is not int or not lower <= result[key] <= upper:
                raise ValueError(f"{key} must be an integer from {lower} to {upper}")
        result["name"] = validate_model_name(config["name"])
        if "sensorIds" in result and (not isinstance(result["sensorIds"], list)
                or not result["sensorIds"] or any(not isinstance(value, str) for value in result["sensorIds"])
                or len(set(result["sensorIds"])) != len(result["sensorIds"])):
            raise ValueError("sensorIds must be a nonempty list of distinct selected sensor IDs")
        return result

    def replay(self, selection):
        status = self.status()
        if not status["outputDirectory"]:
            raise ValueError("Start a training run before selecting a recorded episode")
        output = Path(status["outputDirectory"])
        special = {"initial": "initial-episode.json", "baseline": "baseline-episode.json",
                   "best": "best-episode.json"}
        if selection == "latest":
            selection = str(status["episode"])
        if selection in special:
            path = output / special[selection]
        elif isinstance(selection, str) and selection.startswith("checkpoint-") and selection[11:].isascii() and selection[11:].isdigit():
            number = int(selection[11:])
            if not 1 <= number <= MAX_TRAINING_EPISODES:
                raise ValueError("Choose a saved policy checkpoint")
            path = output / f"checkpoint-{number:06d}-episode.json"
        elif isinstance(selection, str) and selection.isascii() and selection.isdigit():
            number = int(selection)
            if not 1 <= number <= MAX_TRAINING_EPISODES:
                raise ValueError("Choose a completed training episode")
            path = output / "episodes" / f"episode-{number:06d}.json"
        else:
            raise ValueError("Choose initial, baseline, latest, best or an episode number")
        if not path.is_file():
            raise ValueError("That replay has not been recorded yet")
        document = json.loads(path.read_text(encoding="utf-8"))
        return {"viewer": document["viewer"], "episode": document["episode"],
                "label": document["viewer"]["label"]}

    def export(self):
        status = self.status()
        if not status["outputDirectory"]:
            raise ValueError("No recorded training run is available")
        output = Path(status["outputDirectory"])
        result = {"schema": "istana.training_evidence_export.v1", "history": []}
        for name, filename in (("configuration", "configuration.json"),
                               ("summary", "summary.json"),
                               ("recording", "recording-manifest.json")):
            path = output / filename
            result[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        log = output / "training.jsonl"
        if log.exists():
            # Writers flush one line at a time; an unfinished final line is not evidence.
            for line in log.read_text(encoding="utf-8").splitlines(keepends=True):
                if line.endswith("\n"):
                    result["history"].append(json.loads(line))
        return result

    def _evaluate(self, client, policy, seeds, *, baseline=None, should_stop=None):
        runs = []
        for seed in seeds:
            if should_stop is not None and should_stop():
                raise InterruptedError("Stopped between evaluation scenarios")
            run, _ = self.episode_runner(client, policy, seed, action_seed=seed + 1000000,
                deterministic=True, should_stop=should_stop or (lambda: False))
            runs.append(run)
        return evaluation_summary(runs, baseline), runs

    @staticmethod
    def _record(output, filename, run, episode, total, label=None):
        viewer = _viewer(run, episode, total)
        if label:
            viewer["label"] = label
        document = {"schema": "istana.console_episode_recording.v1", "episode": episode,
                    "run": run, "viewer": viewer}
        _write_json_atomic(output / filename, document)
        return viewer

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
                                sensorCount=config["sensorCount"],
                                initialization=config["initialization"],
                                initializationLabel=(INITIALIZATIONS[config["initialization"]]
                                    if config["initialization"] == "untrained" else
                                    f"{config['sensorCount']} · {INITIALIZATIONS[config['initialization']]}"))
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

    def _publish(self, *, client, context, config, output, best_policy, best_path,
                 best_episode, best_observed, started, stopped_early=False):
        """Atomically publish the general policy and its best exact episode replay."""
        if best_policy is None or best_path is None or best_episode is None:
            raise RuntimeError("Training has no held-out checkpoint to publish")
        if best_observed is None:
            raise RuntimeError("Training has no completed episode to retain")
        self._set(phase="evaluating")
        should_stop = self.stop_event.is_set
        selection_path = output / "evaluation-summary.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else None
        test_seeds = (selection["testSeeds"] if selection else
                      scenario_panels(config)[2])
        baseline_layout = common_sense_start(
            context, config["initialization"] if config["initialization"] != "untrained"
            else "directional_balanced_8", count=config["sensorCount"],
            allowed_sensor_ids=config.get("sensorIds"))
        baseline_policy = _FixedPlacementPolicy(baseline_layout["placements"])
        test_baseline, baseline_runs = self._evaluate(client, baseline_policy, test_seeds, should_stop=should_stop)
        test_evaluation, test_runs = self._evaluate(client, best_policy, test_seeds, baseline=test_baseline,
                                                   should_stop=should_stop)
        final, baseline = test_runs[0], baseline_runs[0]
        _write_json_atomic(output / "test-evaluation.json", {
            "schema": "istana.training_test_evaluation.v1", "policy": test_evaluation,
            "contractor": test_baseline, "usedForSelection": False})
        best_warning = selection["best"]["meanWarningSeconds"] if selection else self.status()["bestWarningSeconds"]
        self._set(testEvaluation=test_evaluation)

        observed_source = best_observed["run"]
        observed_replay_exact = all(key in observed_source for key in (
            "context", "native_metrics", "frames", "warning_evidence", "target_results"))
        if observed_replay_exact:
            observed = deepcopy(observed_source)
            observed["trajectory_sha256"] = (
                observed.get("trajectory_sha256") or _trajectory_digest(observed["frames"]))
        else:
            # Legacy stopped runs did not persist full per-episode frames. Keep
            # their exact logged score and placement, while explicitly marking
            # this regenerated seed/layout replay as reconstructed.
            observed, _ = self.episode_runner(
                client, _FixedPlacementPolicy(observed_source["placements"]),
                observed_source["seed"], action_seed=observed_source.get("action_seed"),
                deterministic=True, capture_frames=True, should_stop=should_stop)
        observed_baseline, _ = self.episode_runner(
            client, baseline_policy, observed_source["seed"],
            action_seed=observed_source.get("action_seed"), deterministic=True,
            capture_frames=not observed_replay_exact, should_stop=should_stop)
        if observed["trajectory_sha256"] != observed_baseline["trajectory_sha256"]:
            raise RuntimeError("Best-observed and baseline comparison trajectories differ")

        model_id = self.registry.identifier_for(config["name"])
        observed_id = f"observed-{model_id}"
        algorithm_label = TRAINING_ALGORITHMS[config["algorithm"]]["label"]
        policy_selection = {"label": f"{config['name']} · {algorithm_label}",
            "kind": "trained_warning_policy", "algorithm": config["algorithm"],
            "explanation": (
                f"{algorithm_label} checkpoint selected by mean per-drone warning time on "
                f"{'a fixed validation panel' if selection else 'the legacy single validation case'}. "
                "This replay uses a separate test scenario; all test cases are retained in the run report.")}
        baseline_label = baseline_layout["label"]
        baseline_selection = {"label": baseline_label,
            "kind": "fixed_directional_workbench_start",
            "explanation": baseline_layout["selection"]["explanation"]}
        comparison_episode = {"schema": COMPARISON_SCHEMA, "id": model_id,
            "policy": model_id, "policyLabel": f"{config['name']} · {algorithm_label}", "case": 1,
            "label": f"{config['name']} · unseen test scenario 1", "seed": test_seeds[0],
            "rl": _comparison_view(final, f"{config['name']} · {algorithm_label}", policy_selection),
            "baseline": _comparison_view(baseline, baseline_label, baseline_selection),
            "audit": {"sameTrajectories": True, "sameBudget": True,
                "sameCatalogue": True, "sameSensingDraws": True,
                "trajectorySha256": final["trajectory_sha256"],
                "baselineTrajectorySha256": baseline["trajectory_sha256"],
                "usedForSelection": False, "testEvaluation": test_evaluation}}
        observed_episode = best_observed["episode"]
        # The retained value is always the value logged during training. New
        # runs retain their captured frames; legacy runs regenerate a viewer
        # from the same seed/layout without substituting its new replay score.
        observed_warning = observed_source["metrics"]["mean_drone_warning_s"]
        observed_selection = {
            "label": f"{config['name']} · best observed episode {observed_episode}",
            "kind": "retained_training_episode_layout", "algorithm": config["algorithm"],
            "explanation": (
                f"{'Exact captured replay' if observed_replay_exact else 'Reconstructed seed/layout replay'} "
                f"from training episode {observed_episode}, retained because that single sampled "
                "episode had the highest observed warning time. This is illustrative evidence, "
                "not the held-out-selected deployable policy.")}
        observed_comparison = {"schema": COMPARISON_SCHEMA, "id": observed_id,
            "policy": observed_id,
            "policyLabel": f"{config['name']} · best observed episode {observed_episode}", "case": 1,
            "label": (f"{config['name']} · "
                      f"{'exact training episode' if observed_replay_exact else 'retained episode layout'} "
                      f"{observed_episode}"),
            "seed": observed_source["seed"],
            "rl": _comparison_view(observed,
                f"{config['name']} · observed episode {observed_episode}", observed_selection),
            "baseline": _comparison_view(
                observed_baseline, baseline_label, baseline_selection),
            "audit": {"sameTrajectories": True, "sameBudget": True,
                "sameCatalogue": True, "sameSensingDraws": True,
                "exactTrainingEpisode": observed_episode,
                "exactTrainingReplay": observed_replay_exact,
                "generalPolicyClaim": False,
                "trajectorySha256": observed["trajectory_sha256"],
                "baselineTrajectorySha256": observed_baseline["trajectory_sha256"]}}
        _write_json_atomic(output / "best-observed-episode.json", {
            "schema": "istana.console_best_observed_episode.v1",
            "episode": observed_episode, "seed": observed_source["seed"],
            "actionSeed": observed_source.get("action_seed"),
            "warningSeconds": observed_warning,
            "replayWarningSeconds": observed["metrics"]["mean_drone_warning_s"],
            "exactTrainingReplay": observed_replay_exact,
            "placements": observed["placements"],
            "comparison": observed_comparison,
            "selectionRule": "highest mean warning time among completed sampled training episodes",
            "generalPolicyClaim": False})
        registered = self.registry.register(
            name=config["name"], policy_path=best_path,
            comparison_episode=comparison_episode, observed_episode=observed_comparison,
            metadata={
                "createdUtc": datetime.now(timezone.utc).isoformat(),
                "trainingOutput": str(output), "trainingEpisodes": config["episodes"],
                "completedEpisodes": self.status()["episode"],
                "stoppedEarly": stopped_early,
                "bestEpisode": best_episode,
                "bestWarningSeconds": best_warning,
                "validationEvaluation": selection["best"] if selection else None,
                "testEvaluation": test_evaluation,
                "bestObservedEpisode": observed_episode,
                "bestObservedWarningSeconds": observed_warning,
                "bestObservedSeed": observed_source["seed"],
                "bestObservedPlacements": observed["placements"],
                "bestObservedReplayExact": observed_replay_exact,
                "algorithm": config["algorithm"], "algorithmLabel": algorithm_label,
                "rewardMode": getattr(best_policy, "reward_mode", "absolute_warning"),
                "scenarioSeed": config.get("scenarioSeed"),
                "actionSpace": getattr(best_policy, "action_space", None),
                "sensorCount": config["sensorCount"],
                "sensorIds": config.get("sensorIds"),
                "bestNativeReward": final["reward"],
                "bestBlueNativeReward": final["reward"],
                "bestRedNativeReward": final["red_native_reward"],
                "redPolicy": final["red_policy"],
                "evaluationSeed": test_seeds[0], "initialization": config["initialization"],
                "deploymentPlacements": final["placements"]})
        summary = {"schema": "istana.console_warning_training_summary.v1",
                   "modelName": config["name"], "modelId": registered["id"],
                   "algorithm": config["algorithm"], "algorithmLabel": algorithm_label,
                   "rewardMode": getattr(best_policy, "reward_mode", "absolute_warning"),
                   "scenarioSeed": config.get("scenarioSeed"),
                   "actionSpace": getattr(best_policy, "action_space", None),
                   "episodes": config["episodes"],
                   "requestedEpisodes": config["episodes"],
                   "completedEpisodes": self.status()["episode"],
                   "stoppedEarly": stopped_early, "sensorCount": config["sensorCount"],
                   "sensorIds": config.get("sensorIds"),
                   "initialization": config["initialization"],
                   "bestEpisode": best_episode, "bestNativeReward": final["reward"],
                   "bestBlueNativeReward": final["reward"],
                   "bestRedNativeReward": final["red_native_reward"],
                   "bestObservedEpisode": observed_episode,
                   "bestObservedWarningSeconds": observed_warning,
                   "redPolicy": final["red_policy"],
                   "bestEvaluation": final["metrics"],
                   "comparisonBaselineEvaluation": baseline["metrics"],
                   "validationEvaluation": selection["best"] if selection else None,
                   "initialEvaluation": selection["initial"] if selection else None,
                   "contractorValidationEvaluation": selection["contractor"] if selection else None,
                   "testEvaluation": test_evaluation,
                   "selectionRule": selection["selectionRule"] if selection else "legacy single validation case",
                   "elapsedWallSeconds": time.monotonic() - started}
        _write_json_atomic(output / "summary.json", summary)
        self._record(output, "best-test-episode.json", final, best_episode, config["episodes"],
                     f"Best policy · episode {best_episode} · unseen test scenario 1")
        manifest_path = output / "recording-manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["entries"].append({"checkpoint": best_episode, "kind": "best_test",
                "label": "Best policy · independent test", "run": "best-test-episode.json",
                "evaluation": test_evaluation, "evaluation_seeds": test_seeds})
            manifest["testEvaluation"] = test_evaluation
            _write_json_atomic(manifest_path, manifest)
        self._set(running=False, phase="stopped" if stopped_early else "complete",
            bestObservedEpisode=observed_episode,
            bestObservedWarningSeconds=observed_warning,
            registeredModel={
                "id": registered["id"], "name": registered["name"],
                "algorithm": config["algorithm"], "algorithmLabel": algorithm_label,
                "bestEpisode": best_episode,
                "bestNativeReward": final["reward"],
                "bestBlueNativeReward": final["reward"],
                "bestRedNativeReward": final["red_native_reward"],
                "bestWarningSeconds": best_warning,
                "bestObservedEpisode": observed_episode,
                "bestObservedWarningSeconds": observed_warning})
        return registered

    def publish_saved_run(self, directory):
        """Explicitly finish evaluation of a stopped current or legacy run."""
        output = Path(directory).resolve()
        config_document = json.loads((output / "configuration.json").read_text(encoding="utf-8"))
        fields = {"name", "algorithm", "episodes", "batchSize", "seed", "initialization",
                  "sensorCount", "explorationProbability", "entropyCoefficient", "validationCases",
                  "checkpointInterval", "headroomProbes", "sensorIds", "scenarioSeed", "validationInterval"}
        config = self.validate({key: value for key, value in config_document.items() if key in fields})
        self.registry.ensure_available(config["name"])
        rows = [json.loads(line) for line in
                (output / "training.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if not rows:
            raise ValueError("Saved training log has no completed episodes")
        selection_path = output / "evaluation-summary.json"
        if selection_path.exists():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            best_episode = selection["bestEpisode"]
            if type(best_episode) is not int or not 0 <= best_episode <= config["episodes"]:
                raise ValueError("Saved selected checkpoint episode is invalid")
            best_path = output / ("policy-0000.json" if best_episode == 0
                                  else f"best-policy-{best_episode:04d}.json")
            best_warning = selection["best"]["meanWarningSeconds"]
        else:
            try:
                checkpoints = sorted(output.glob("best-policy-*.json"),
                    key=lambda path: int(path.stem.rsplit("-", 1)[1]))
            except (IndexError, ValueError) as error:
                raise ValueError("Saved held-out checkpoint name is invalid") from error
            if not checkpoints:
                raise ValueError("Saved training run has no held-out best checkpoint")
            best_path = checkpoints[-1]
            try:
                best_episode = int(best_path.stem.rsplit("-", 1)[1])
            except (IndexError, ValueError) as error:
                raise ValueError("Saved held-out checkpoint name is invalid") from error
            validation_row = next((row for row in rows if row.get("episode") == best_episode), None)
            if validation_row is None or "validationWarningSeconds" not in validation_row:
                raise ValueError("Saved held-out checkpoint has no matching validation evidence")
            best_warning = validation_row["validationWarningSeconds"]
        if not best_path.is_file():
            raise ValueError("Saved selected checkpoint file is missing")
        observed_row = max(rows, key=lambda row: row["meanWarningSeconds"])
        observed_run = {
            "seed": scenario_panels(config)[0] + observed_row["episode"] - 1,
            "action_seed": (config["seed"] & 0xffffffff) * 100000 + observed_row["episode"],
            "placements": observed_row["placements"],
            "metrics": {"mean_drone_warning_s": observed_row["meanWarningSeconds"]},
        }
        observed_path = output / "episodes" / f"episode-{observed_row['episode']:06d}.json"
        if observed_path.exists():
            observed_run = json.loads(observed_path.read_text(encoding="utf-8"))["run"]
        factory = self.policy_factories.get(config["algorithm"])
        if factory is None or not hasattr(factory, "load"):
            raise ValueError("Saved training policy type cannot be loaded")
        client = None
        started = time.monotonic()
        try:
            client = self.client_factory(port=self.bridge_port, timeout=120.)
            client.reset(1600000)
            context = client.get_blue_context()
            policy = factory.load(best_path, context)
            self.stop_event.clear()
            self._status = self._blank()
            self._status.update(
                running=True, phase="evaluating", episode=max(row["episode"] for row in rows),
                totalEpisodes=config["episodes"], modelName=config["name"],
                algorithm=config["algorithm"],
                algorithmLabel=TRAINING_ALGORITHMS[config["algorithm"]]["label"],
                sensorCount=config["sensorCount"], initialization=config["initialization"],
                initializationLabel=(INITIALIZATIONS[config["initialization"]]
                    if config["initialization"] == "untrained" else
                    f"{config['sensorCount']} · {INITIALIZATIONS[config['initialization']]}"),
                outputDirectory=str(output), bestEpisode=best_episode,
                bestWarningSeconds=best_warning, bestValidationWarningSeconds=best_warning,
                bestObservedEpisode=observed_row["episode"],
                bestObservedWarningSeconds=observed_row["meanWarningSeconds"],
                stopRequested=False)
            return self._publish(
                client=client, context=context, config=config, output=output,
                best_policy=policy, best_path=best_path, best_episode=best_episode,
                best_observed={"episode": observed_row["episode"], "run": observed_run},
                started=started, stopped_early=True)
        except Exception as error:
            self._set(running=False, phase="failed", error=str(error))
            raise
        finally:
            if client is not None:
                client.close()

    def _run(self, config):
        started = time.monotonic()
        output = self.output_root / datetime.now(timezone.utc).strftime("console-%Y%m%d-%H%M%S-%f")
        client = best_policy = best_path = best_episode = best_observed = None
        policy = None
        try:
            client = self.client_factory(port=self.bridge_port, timeout=120.)
            client.reset(1600000)
            context = client.get_blue_context()
            if context["publicSnapshot"]["max_sites"] < config["sensorCount"]:
                raise ValueError("The native scene has too few sensor slots; restart with -TrainingWorkbench")
            available = context["publicSnapshot"]["available_sensor_ids"]
            directional = [row["id"] for row in context["catalogue"]
                           if row.get("directional") and row["id"] in available]
            config["sensorIds"] = config.get("sensorIds", directional)
            if not config["sensorIds"] or not set(config["sensorIds"]) <= set(directional):
                raise ValueError("Training sensors must be selected, available limited-FOV profiles")
            policy = self.policy_factories[config["algorithm"]](
                context, seed=config["seed"], sensor_count=config["sensorCount"],
                entropy_coefficient=config["entropyCoefficient"], allowed_sensor_ids=config["sensorIds"])
            preset = config["initialization"] if config["initialization"] != "untrained" else "directional_balanced_8"
            contractor = common_sense_start(context, preset, count=config["sensorCount"],
                                            allowed_sensor_ids=config["sensorIds"])
            start_layout = None
            if config["initialization"] != "untrained":
                start_layout = deepcopy(contractor)
                start_layout["warm_start"] = policy.initialize_from_placements(
                    start_layout["placements"],
                    baseline_action_probability=1. - config["explorationProbability"])
            output.mkdir(parents=True, exist_ok=False)
            (output / "episodes").mkdir()
            training_seed, validation_seeds, test_seeds = scenario_panels(config)
            paired_reward = getattr(policy, "reward_mode", None) == "paired_contractor_delta"
            if paired_reward:
                (output / "contractor-episodes").mkdir()
            configuration = {"schema": "istana.console_warning_training.v2", **config,
                "bridgePort": self.bridge_port, "initialLayout": start_layout,
                "contractorLayout": contractor,
                "redScenario": {"policy": FixedRadiusSectorRedPolicy.name, "trained": False,
                    "episodeSeedRule": f"{training_seed} + episode - 1", "validationSeeds": validation_seeds,
                    "testSeeds": test_seeds},
                "rewardMode": "paired_contractor_delta" if paired_reward else "absolute_warning",
                "warningMetric": "mean per-drone max(0, 20m zone arrival - first detection); undetected=0",
                "limitations": "Synthetic native simulation; ties and regressions are retained."}
            _write_json_atomic(output / "configuration.json", configuration)
            policy.save(output / "policy-0000.json")
            manifest = {"schema": "istana.console_training_recording.v1",
                "configuration": configuration, "context": context, "entries": [],
                "episodeDirectory": "episodes", "episodePattern": "episode-{episode:06d}.json",
                "completedEpisodes": 0, "checkpointInterval": config["checkpointInterval"]}
            def recording_entry(number, kind, run, evaluation, filename, label):
                viewer = self._record(output, filename, run, number, config["episodes"], label)
                entry = {"checkpoint": number, "kind": kind, "label": label,
                         "run": filename, "evaluation": evaluation,
                         "evaluation_seeds": evaluation["seeds"]}
                manifest["entries"] = [row for row in manifest["entries"]
                    if not (row["kind"] == kind and (kind == "best" or row["checkpoint"] == number))]
                manifest["entries"].append(entry)
                _write_json_atomic(output / "recording-manifest.json", manifest)
                return viewer
            self._set(phase="evaluating", initialLayout=start_layout, outputDirectory=str(output),
                checkpoints=[0], validationCases=config["validationCases"],
                rewardMode=configuration["rewardMode"], scenarioSeed=config.get("scenarioSeed"),
                exploration={"explorationProbability": config["explorationProbability"], "uniqueLayouts": 0})
            fixed = _FixedPlacementPolicy(contractor["placements"])
            baseline_summary, baseline_runs = self._evaluate(
                client, fixed, validation_seeds, should_stop=self.stop_event.is_set)
            baseline_view = recording_entry(0, "baseline", baseline_runs[0], baseline_summary,
                "baseline-episode.json", "Contractor layout · validation scenario 1")
            initial_summary, initial_runs = self._evaluate(
                client, policy, validation_seeds, baseline=baseline_summary, should_stop=self.stop_event.is_set)
            initial_placements = initial_runs[0]["placements"]
            initial_view = recording_entry(0, "initial", initial_runs[0], initial_summary,
                "initial-episode.json", "Initial policy · validation scenario 1")
            best_policy, best_path, best_episode = deepcopy(policy), output / "policy-0000.json", 0
            best_score = initial_summary["meanWarningSeconds"]
            recording_entry(0, "best", initial_runs[0], initial_summary, "best-episode.json",
                            "Best validated policy · initial policy retained")
            selection = {"schema": "istana.training_selection.v1", "validationSeeds": validation_seeds,
                "testSeeds": test_seeds, "initial": initial_summary, "contractor": baseline_summary,
                "best": initial_summary, "bestEpisode": 0,
                "selectionRule": "highest mean warning across fixed validation panel; ties retain earlier checkpoint"}
            _write_json_atomic(output / "evaluation-summary.json", selection)
            self._set(initialEvaluation=initial_summary, baselineEvaluation=baseline_summary,
                bestWarningSeconds=best_score, bestValidationWarningSeconds=best_score,
                bestEpisode=0, viewer=initial_view, viewerSelection="initial")
            # Probe alternatives before learning. These diagnostics never replace
            # the contractor baseline or contribute gradients to the actor.
            probe_rows = []
            for number, placements in enumerate(legal_layout_probes(
                    context, contractor["placements"], limit=config["headroomProbes"])
                    if config["headroomProbes"] else [], 1):
                summary, _ = self._evaluate(client, _FixedPlacementPolicy(placements), validation_seeds,
                    baseline=baseline_summary, should_stop=self.stop_event.is_set)
                probe_rows.append({"candidate": number, "placements": placements, "evaluation": summary})
            probe_report = {"candidates": len(probe_rows),
                "bestDeltaSeconds": max((row["evaluation"]["deltaSeconds"] for row in probe_rows), default=None),
                "improvesBaseline": any(row["evaluation"]["deltaSeconds"] > 1e-9 for row in probe_rows),
                "exhaustive": False, "results": probe_rows}
            _write_json_atomic(output / "baseline-probes.json", probe_report)
            self._set(baselineProbe={key: value for key, value in probe_report.items() if key != "results"},
                      phase="training")
            previous_validation = best_score
            batch, unique_layouts = [], set()
            with (output / "training.jsonl").open("x", encoding="utf-8") as log, \
                    (output / "evaluations.jsonl").open("x", encoding="utf-8") as evaluations:
                for number in range(1, config["episodes"] + 1):
                    if self.stop_event.is_set():
                        raise InterruptedError("Training stopped before the next episode")
                    run, records = self.episode_runner(client, policy, training_seed + number - 1,
                        action_seed=(config["seed"] & 0xffffffff) * 100000 + number,
                        should_stop=self.stop_event.is_set)
                    reward = run["metrics"]["mean_drone_warning_s"]
                    training_reward, paired = reward, None
                    if paired_reward:
                        reference_run, _ = self.episode_runner(client, fixed, run["seed"],
                            deterministic=True, should_stop=self.stop_event.is_set)
                        paired = paired_training_reward(run, reference_run)
                        training_reward = paired["rewardSeconds"]
                        self._record(output, f"contractor-episodes/episode-{number:06d}.json",
                            reference_run, number, config["episodes"], "Matched contractor control")
                    unique_layouts.add(layout_key(run["placements"]))
                    changed = layout_changes(run["placements"], initial_placements)
                    entropy = rollout_entropy(records)
                    if best_observed is None or reward > best_observed["run"]["metrics"]["mean_drone_warning_s"]:
                        best_observed = {"episode": number, "run": deepcopy(run)}
                        _write_json_atomic(output / "best-observed-episode.json", {
                            "schema": "istana.console_best_observed_episode.v1", "episode": number,
                            "seed": run["seed"], "actionSeed": run.get("action_seed"),
                            "warningSeconds": reward, "placements": run["placements"], "run": run,
                            "generalPolicyClaim": False})
                        self._set(bestObservedEpisode=number, bestObservedWarningSeconds=reward)
                    viewer = self._record(output, f"episodes/episode-{number:06d}.json", run,
                                          number, config["episodes"])
                    breakdown = _reward_breakdown(run)
                    entry = {"episode": number, "seed": run["seed"], "actionSeed": run.get("action_seed"),
                        "reward": reward, "trainingReward": training_reward, "meanWarningSeconds": reward,
                        "rewardMode": configuration["rewardMode"], "pairedTraining": paired,
                        "blueNativeReward": breakdown["total"], "nativeReward": breakdown["total"],
                        "redNativeReward": run["red_native_reward"], "redPolicy": run["red_policy"],
                        "redDecision": run["red_decision"], "metrics": run["metrics"],
                        "redScenario": {"formation": run["red_decision"]["formation"],
                            "spawnRadiusM": run["red_decision"]["radius_cm"] / 100.,
                            "spawnBearingsDeg": run["red_decision"]["angles_degrees"],
                            "sectorCentersDeg": run["red_decision"].get("sector_centers_degrees", [])},
                        "rewardBreakdown": breakdown, "teamWarningSeconds": run["metrics"]["team_warning_s"],
                        "firstDetectionSeconds": run["metrics"]["first_detection_s"],
                        "detectedFraction": run["metrics"]["detected_fraction"], "cost": run["metrics"]["cost"],
                        "sensorsPlaced": len(run["placements"]), "elapsedSeconds": run["elapsed_seconds"],
                        "firstConfirmationSeconds": min((row["firstConfirmationSeconds"]
                            for row in run["warning_evidence"] if row["firstConfirmationSeconds"] is not None), default=None),
                        "detectedCount": sum(row["firstDetectionSeconds"] is not None for row in run["warning_evidence"]),
                        "missedCount": sum(row["firstDetectionSeconds"] is None for row in run["warning_evidence"]),
                        "nativeMetrics": run["native_metrics"],
                        "sensorsChangedFromInitial": changed, "uniqueLayouts": len(unique_layouts), **entropy}
                    diagnostics = getattr(policy, "episode_diagnostics", None)
                    if callable(diagnostics):
                        entry["learningDecision"] = diagnostics(records)
                    batch.append((records, training_reward))
                    milestone = number % config["checkpointInterval"] == 0 or number == config["episodes"]
                    try:
                        if number % config["batchSize"] == 0:
                            entry["update"] = policy.update(batch)
                            batch = []
                        if number % config["validationInterval"] == 0 or milestone:
                            validation, validation_runs = self._evaluate(client, policy, validation_seeds,
                                baseline=baseline_summary, should_stop=self.stop_event.is_set)
                            score = validation["meanWarningSeconds"]
                            entry.update(validationWarningSeconds=score, validationReward=score,
                                validationDeltaSeconds=validation["deltaSeconds"],
                                validationCases=config["validationCases"], validationChanges=[])
                            difference = score - previous_validation
                            if abs(difference) > 1e-9:
                                entry["validationChanges"].append({"value": difference, "unit": "seconds",
                                    "label": f"Validation mean warning changed by {difference:+.2f} s"})
                            previous_validation = score
                            evaluations.write(json.dumps({"episode": number, "evaluation": validation}, allow_nan=False) + "\n")
                            evaluations.flush()
                            if score > best_score + 1e-9:
                                best_score, best_episode = score, number
                                best_path = output / f"best-policy-{number:04d}.json"
                                policy.save(best_path)
                                best_policy = deepcopy(policy)
                                selection.update(best=validation, bestEpisode=number)
                                _write_json_atomic(output / "evaluation-summary.json", selection)
                                recording_entry(number, "best", validation_runs[0], validation, "best-episode.json",
                                                f"Best validated policy · episode {number}")
                                self._set(bestEpisode=number, bestWarningSeconds=score, bestValidationWarningSeconds=score)
                            if milestone:
                                policy.save(output / f"policy-{number:04d}.json")
                                recording_entry(number, "checkpoint", validation_runs[0], validation,
                                    f"checkpoint-{number:06d}-episode.json", f"Policy checkpoint · episode {number}")
                                self._set(checkpoints=self.status()["checkpoints"] + [number])
                    finally:
                        log.write(json.dumps({**entry, "runId": run["run_id"], "placements": run["placements"]}, allow_nan=False) + "\n")
                        log.flush()
                        manifest["completedEpisodes"] = number
                        _write_json_atomic(output / "recording-manifest.json", manifest)
                        with self.lock:
                            self._status["history"].append(entry)
                        self._set(episode=number, viewer=viewer, viewerSelection="latest",
                            exploration={"uniqueLayouts": len(unique_layouts), "changedFromInitial": changed,
                                "sensorsChangedFromInitial": changed, "meanNormalizedEntropy": entropy["normalizedEntropy"],
                                "explorationProbability": None if paired_reward else config["explorationProbability"],
                                "localEdits": config["algorithm"] == "local_ppo",
                                "actionValueBandit": config["algorithm"] == "paired_bandit"})
            policy.save(output / "final-policy.json")
            self._publish(client=client, context=context, config=config, output=output,
                best_policy=best_policy, best_path=best_path, best_episode=best_episode,
                best_observed=best_observed, started=started)
        except InterruptedError:
            if policy is not None and output.exists():
                policy.save(output / "stopped-policy.json")
                _write_json_atomic(output / "stopped-summary.json", {
                    "schema": "istana.console_stopped_training.v1", "completedEpisodes": self.status()["episode"],
                    "bestEpisode": best_episode, "bestPolicy": best_path.name if best_path else None,
                    "pendingBatchEpisodes": len(batch) if 'batch' in locals() else 0,
                    "published": False, "reason": "Stopped; native evaluation was not continued",
                    "elapsedWallSeconds": time.monotonic() - started})
            self._set(running=False, phase="stopped")
        except Exception as error:
            self._set(running=False, phase="failed", error=str(error))
        finally:
            if client is not None:
                client.close()
