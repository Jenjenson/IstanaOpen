"""Asynchronous native experiments, durable metrics, history and checkpoints.

The UI never runs an optimizer. Native bridge ownership is supplied by the console;
algorithms receive public Blue context, and evaluator truth stays in episode logs.
"""
from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import secrets
import subprocess
import threading
import time

from .istana_live import IstanaLiveClient
from .training_environment import (SUCCESS_DEFINITION, WARNING_DEFINITION, TrainingStopped,
                                   blue_configuration, require_training_runtime, run_episode)

CHECKPOINT_CRITERION = "Highest mean per-threat first-detection warning across the same 3 held-out seeds, using deterministic placement. Ties retain the earlier checkpoint. Selection panel scores are not an unbiased final test."
ACTIVE = {"starting", "running", "evaluating", "stopping"}
CSV_FIELDS = ["episode", "seed", "reward", "warningTime", "teamWarningTime", "firstDetectionTime",
              "confirmationTime", "confirmationWarningTime", "successRate", "threatsDetected",
              "threatsMissed", "threatCount", "sensorsPlaced", "budgetUsed", "episodeDuration",
              "wallDuration", "policyLoss", "valueLoss", "entropy", "optimizerUpdate"]


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}")
    return value


class TrainingManager:
    def __init__(self, root, *, bridge_port=8765, client_factory=IstanaLiveClient,
                 acquire=None, release=None, probe=None):
        self.root = Path(root).resolve()
        self.bridge_port, self.client_factory = bridge_port, client_factory
        self.acquire = acquire or (lambda owner: None)
        self.release = release or (lambda owner: None)
        self.probe = probe
        self.lock = threading.RLock()
        self.worker = None
        self.stop_event = threading.Event()
        self.current = None
        self.started_monotonic = None
        self.context = None
        self.probe_error = "Start the updated Unreal application with -IstanaBlueLive."
        self._recover()

    def _recover(self):
        if not self.root.exists():
            return
        for path in self.root.glob("*/run.json"):
            try:
                run = json.loads(path.read_text(encoding="utf-8"))
                if run["status"] in ACTIVE:
                    run.update(status="interrupted", error="Console stopped before this run finished; saved checkpoints and complete episodes are retained.")
                    atomic_json(path, run)
            except (OSError, ValueError, KeyError):
                continue

    def options(self):
        from .training_algorithms import ALGORITHMS
        with self.lock:
            active = self.worker is not None and self.worker.is_alive()
        if not active:
            owner = "training-options"
            acquired = False
            try:
                if self.probe is not None:
                    context = self.probe()
                else:
                    self.acquire(owner)
                    acquired = True
                    with self.client_factory(self.bridge_port, timeout=3.) as client:
                        context = client.request("blue_context")["context"]
                require_training_runtime(context)
                self.context, self.probe_error = context, ""
            except (ValueError, RuntimeError, OSError) as exc:
                self.probe_error = str(exc)
            finally:
                if acquired:
                    self.release(owner)
        context = self.context or {}
        catalogue = [{**row, "label": row.get("label", row.get("name", row["id"]))} for row in context.get("catalogue", [])]
        registry = list(ALGORITHMS.values()) if isinstance(ALGORITHMS, dict) else ALGORITHMS
        algorithms = [{"id": row["id"], "label": row["label"], "description": row["description"],
                       "metrics": [{"policy_loss": "policyLoss", "value_loss": "valueLoss"}.get(metric, metric)
                                   for metric in row["supported_metrics"]]} for row in registry]
        return {"algorithms": algorithms, "catalogue": catalogue,
                "defaults": {"episodes": 100, "budget": 3, "checkpointFrequency": 10, "batchSize": 8},
                "environment": {"available": bool(context) and not self.probe_error,
                    "detail": self.probe_error or f"Native Unreal directional sensors · bridge {self.bridge_port}",
                    "warningDefinition": WARNING_DEFINITION, "successDefinition": SUCCESS_DEFINITION,
                    "sensorModel": context.get("sensorModel"),
                    "maxSensors": context.get("publicSnapshot", {}).get("max_sites")},
                "checkpointCriterion": CHECKPOINT_CRITERION,
                "unavailableAlgorithms": "Historical temporal/adaptive checkpoints use different action contracts and cannot train native directional orientations."}

    def _validate(self, payload):
        from .training_algorithms import ALGORITHMS
        allowed = {"runName", "algorithm", "episodes", "budget", "seed", "enabledSensorIds", "checkpointFrequency"}
        if not isinstance(payload, dict) or set(payload) - allowed:
            raise ValueError("Unexpected training configuration field")
        name = payload.get("runName", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
            raise ValueError("Run name: use 1–64 letters, digits, hyphens or underscores, beginning with a letter or digit")
        if name.upper() in {"CON", "PRN", "AUX", "NUL", *[f"{p}{n}" for p in ("COM", "LPT") for n in range(1, 10)]}:
            raise ValueError("Choose a non-reserved run name")
        algorithm = payload.get("algorithm", "reinforce")
        ids = set(ALGORITHMS) if isinstance(ALGORITHMS, dict) else {row["id"] for row in ALGORITHMS}
        if not isinstance(algorithm, str) or algorithm not in ids:
            raise ValueError("Select a supported native directional algorithm")
        budget = payload.get("budget", 3)
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or not .001 <= budget <= 100:
            raise ValueError("Sensor budget must be a finite number between 0.001 and 100")
        sensor_ids = payload.get("enabledSensorIds")
        if not isinstance(sensor_ids, list) or len(sensor_ids) > 32 or any(not isinstance(x, str) for x in sensor_ids) or len(set(sensor_ids)) != len(sensor_ids):
            raise ValueError("Select sensor types from the native catalogue")
        seed = payload.get("seed")
        seed = secrets.randbelow(2**31) if seed is None else integer(seed, "Seed", 0, 2**31 - 1)
        episodes = integer(payload.get("episodes", 100), "Episodes", 1, 100000)
        rng = random.Random(seed)
        seeds = rng.sample(range(2**31), episodes + 3)
        return {"runName": name, "algorithm": algorithm, "episodes": episodes, "budget": float(budget),
                "seed": seed, "enabledSensorIds": sensor_ids,
                "checkpointFrequency": integer(payload.get("checkpointFrequency", 10), "Checkpoint frequency", 1, 100000),
                "batchSize": 8, "trainingSeeds": seeds[:episodes], "evaluationSeeds": seeds[episodes:],
                "objective": "mean_drone_warning_s", "warningDefinition": WARNING_DEFINITION,
                "successDefinition": SUCCESS_DEFINITION, "checkpointCriterion": CHECKPOINT_CRITERION,
                "redScenario": "Existing seeded radial native trainer; uses scene's actual group count and swarm size."}

    def start(self, payload):
        config = self._validate(payload)
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                raise ValueError("A training run is already active; stop it before starting another")
            self.root.mkdir(parents=True, exist_ok=True)
            base = config["runName"]
            for number in range(1, 10000):
                name = base if number == 1 else f"{base}-{number}"
                directory = self.root / name
                try:
                    directory.mkdir()
                    break
                except FileExistsError:
                    continue
            else:
                raise ValueError("Too many runs with this name")
            config["runName"] = name
            try:
                self.acquire(name)
            except Exception:
                directory.rmdir()  # only our newly-created, empty directory
                raise
            run = {"id": name, "runName": name, "algorithm": config["algorithm"], "config": config,
                   "status": "starting", "startedAt": utc_now(), "elapsedSeconds": 0., "episode": 0,
                   "totalEpisodes": config["episodes"], "metrics": [], "summary": {}, "checkpoints": {},
                   "evaluations": [], "error": None, "phase": "Connecting to native Unreal"}
            try:
                self.current = run
                self.stop_event = threading.Event()
                self.started_monotonic = time.monotonic()
                atomic_json(directory / "config.json", config)
                self._persist(run)
                self.worker = threading.Thread(target=self._train, args=(run,), name=f"training-{name}", daemon=True)
                self.worker.start()
            except Exception:
                self.release(name)
                self.current = None
                raise
            return {"run": self.detail(name)}

    def stop(self, run_id):
        with self.lock:
            if not self.current or self.current["id"] != run_id:
                raise ValueError("This run is not active")
            if self.worker and self.worker.is_alive():
                self.stop_event.set()
                self.current["status"] = "stopping"
                self.current["phase"] = "Finishing the current native request and saving completed episodes"
            return {"run": self.detail(run_id)}

    def close(self):
        self.stop_event.set()
        if self.worker:
            self.worker.join(timeout=15.)

    def _path(self, run_id):
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
            raise ValueError("Invalid run id")
        path = (self.root / run_id).resolve()
        if path.parent != self.root or not (path / "run.json").is_file():
            raise ValueError("Training run not found")
        return path

    def _read(self, run_id):
        path = self._path(run_id)
        if self.current and self.current["id"] == run_id:
            return deepcopy(self.current)
        run = json.loads((path / "run.json").read_text(encoding="utf-8"))
        run["metrics"] = []
        if (path / "metrics.jsonl").exists():
            for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
                try:
                    run["metrics"].append(json.loads(line))
                except ValueError:
                    break  # retain all complete rows after an interrupted disk write
        run["episode"] = len(run["metrics"])
        run["summary"] = self._summary(run["metrics"])
        return run

    def runs(self):
        rows = []
        with self.lock:
            for path in self.root.glob("*/run.json"):
                try:
                    run = json.loads(path.read_text(encoding="utf-8"))
                    if self.current and run["id"] == self.current["id"]:
                        run = self.current
                    rows.append({key: deepcopy(run.get(key)) for key in
                        ("id", "runName", "algorithm", "status", "episode", "totalEpisodes", "startedAt", "elapsedSeconds", "summary")}
                        | {"budget": run["config"]["budget"]})
                except (OSError, ValueError, KeyError):
                    continue
        return {"runs": sorted(rows, key=lambda row: row["startedAt"], reverse=True)}

    def detail(self, run_id):
        with self.lock:
            if self.current and self.current["id"] == run_id:
                source = self.current
            else:
                source = self._read(run_id)
            run = deepcopy({key: value for key, value in source.items() if key not in {"metrics", "config"}})
            run["config"] = deepcopy({key: value for key, value in source["config"].items() if key != "trainingSeeds"})
            rows = source["metrics"]
            run["latestMetric"] = deepcopy(rows[-1]) if rows else None
            indices = (sorted({round(i * (len(rows) - 1) / 999) for i in range(1000)})
                       if len(rows) > 1000 else range(len(rows)))
            run["metrics"] = deepcopy([rows[i] for i in indices])
            if source is self.current and source["status"] in ACTIVE and self.started_monotonic is not None:
                run["elapsedSeconds"] = time.monotonic() - self.started_monotonic
        run.update(metricsSampled=len(rows) > 1000, movingAverageWindow=20,
                   metricScope="Loss and entropy are recorded only on optimizer-update episodes and describe that completed batch.",
                   artifactDirectory=str(self.root / run_id))
        # Seeds are saved in config.json; avoid resending thousands during polling.
        run["config"].pop("trainingSeeds", None)
        return run

    @staticmethod
    def _summary(rows):
        if not rows:
            return {}
        mean = lambda key: sum(row[key] for row in rows) / len(rows)
        return {"averageReward": mean("reward"), "bestReward": max(row["reward"] for row in rows),
                "averageWarningTime": mean("warningTime"), "bestWarningTime": max(row["warningTime"] for row in rows),
                "latestWarningTime": rows[-1]["warningTime"],
                "movingAverageWarningTime": sum(row["warningTime"] for row in rows[-20:]) / len(rows[-20:]),
                "successRate": mean("successRate"), "averageSensorsUsed": mean("sensorsPlaced"),
                "averageBudgetUsed": mean("budgetUsed"), "episodes": len(rows)}

    def _persist(self, run):
        directory = self.root / run["id"]
        snapshot = {key: value for key, value in run.items() if key != "metrics"}
        snapshot["config"] = {key: value for key, value in run["config"].items() if key != "trainingSeeds"}
        atomic_json(directory / "run.json", snapshot)

    def _record(self, run, row, evidence):
        directory = self.root / run["id"]
        with self.lock:
            row["episode"] = len(run["metrics"]) + 1
            recent = run["metrics"][-19:] + [row]
            for field in ("reward", "warningTime", "successRate", "budgetUsed"):
                row["movingAverage" + field[0].upper() + field[1:]] = sum(x[field] for x in recent) / len(recent)
            with (directory / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            with (directory / "metrics.csv").open("a", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
                if stream.tell() == 0:
                    writer.writeheader()
                csv_row = dict(row)
                csv_row["optimizerUpdate"] = json.dumps(row["optimizerUpdate"], allow_nan=False) if row.get("optimizerUpdate") else ""
                writer.writerow(csv_row)
            with (directory / "evidence.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"episode": row["episode"], "nativeRunId": row["nativeRunId"],
                                         "warningEvidenceForEvaluationOnly": evidence}, allow_nan=False) + "\n")
            run["metrics"].append(row)
            run["episode"] = row["episode"]
            previous, count = run["summary"], row["episode"]
            summary = {}
            for name, field in (("averageReward", "reward"), ("averageWarningTime", "warningTime"),
                                ("successRate", "successRate"), ("averageSensorsUsed", "sensorsPlaced"),
                                ("averageBudgetUsed", "budgetUsed")):
                summary[name] = (previous.get(name, 0.) * (count - 1) + row[field]) / count
            summary.update(bestReward=max(previous.get("bestReward", -math.inf), row["reward"]),
                           bestWarningTime=max(previous.get("bestWarningTime", -math.inf), row["warningTime"]),
                           latestWarningTime=row["warningTime"], movingAverageWarningTime=row["movingAverageWarningTime"], episodes=count)
            run["summary"] = summary
            self._persist(run)

    def _save_model(self, policy, run, label):
        directory = self.root / run["id"] / "checkpoints"
        directory.mkdir(exist_ok=True)
        temporary = directory / f"{label}-{secrets.token_hex(4)}.json"
        policy.save(temporary)
        target = directory / f"{label}.json"
        temporary.replace(target)
        entry = {"episode": run["episode"], "path": str(target.relative_to(self.root / run["id"])),
                 "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        with self.lock:
            run["checkpoints"][label] = entry
            self._persist(run)
        return entry

    def _checkpoint(self, client, policy, run):
        self._save_model(policy, run, "latest")
        self._save_model(policy, run, f"episode-{run['episode']:06d}")
        with self.lock:
            run.update(status="evaluating", phase="Selecting checkpoints on 3 fixed held-out scenarios")
        values = []
        for seed in run["config"]["evaluationSeeds"]:
            row, _, evidence = run_episode(client, policy, run["config"], seed, self.stop_event, deterministic=True)
            values.append({**row, "evidence": evidence})
        score = sum(row["warningTime"] for row in values) / len(values)
        directory = self.root / run["id"]
        atomic_json(directory / f"evaluation-{run['episode']:06d}.json", values)
        if score > run["checkpoints"].get("best", {}).get("score", -math.inf):
            entry = self._save_model(policy, run, "best")
            with self.lock:
                entry["score"] = score
        with self.lock:
            run["evaluations"].append({"episode": run["episode"], "meanWarningTime": score,
                                       "seeds": run["config"]["evaluationSeeds"]})
            run.update(status="running", phase="Training native episodes")
            self._persist(run)

    def _train(self, run):
        from .training_algorithms import create_algorithm
        started = time.monotonic()
        policy, batch = None, []
        directory = self.root / run["id"]
        try:
            # Use the existing native trainer's transport timeout. Dense collision
            # batches can exceed the short interactive console timeout; do not
            # retry a timed-out mutation, which may already have advanced physics.
            with self.client_factory(self.bridge_port, timeout=120.) as client:
                # Check capability before any reset, so an outdated runtime fails explicitly.
                require_training_runtime(client.request("blue_context")["context"])
                red_context = client.reset(run["config"]["seed"], blue_configuration=blue_configuration(run["config"]))
                context = client.get_blue_context()
                require_training_runtime(context)
                self.context, self.probe_error = context, ""
                atomic_json(directory / "native_context.json", context)
                atomic_json(directory / "red_context.json", red_context)
                atomic_json(directory / "sensor_config.json", {"catalogue": context["catalogue"],
                    "blueConfiguration": blue_configuration(run["config"]), "sensorModel": context.get("sensorModel"),
                    "temporalConfig": context["temporalConfig"]})
                source = Path(__file__).resolve().parents[4]
                try:
                    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
                except (OSError, subprocess.CalledProcessError):
                    commit = None
                source_files = [Path(__file__), Path(__file__).with_name("training_algorithms.py"),
                                Path(__file__).with_name("training_environment.py"),
                                *[Path(__file__).with_name(name) for name in ("warning_policy.py", "adaptive_policy.py", "directional_inputs.py", "istana_live.py")],
                                source / "Source/IstanaOpen/Simulation/BlueTeam/BlueTeamCoordinator.cpp",
                                source / "Source/IstanaOpen/Simulation/BlueTeam/BlueWarningTime.h",
                                source / "Source/IstanaOpen/Simulation/Sensing/DirectionalSensorModel.cpp"]
                atomic_json(directory / "reproducibility.json", {"commit": commit, "seed": run["config"]["seed"],
                    "sourceSha256": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                    "nativeRuntime": context.get("runtimeBuild"), "pythonVersion": __import__("sys").version})
                policy = create_algorithm(run["algorithm"], context, run["config"]["seed"], {})
                atomic_json(directory / "algorithm_config.json", policy.metadata)
                run["config"]["algorithmConfig"] = policy.config
                atomic_json(directory / "config.json", run["config"])
                self._checkpoint(client, policy, run)
                # Even a stop before episode one retains a reconstructible initialized policy.
                with self.lock:
                    run.update(status="running", phase="Training native episodes")
                for seed in run["config"]["trainingSeeds"]:
                    row, records, evidence = run_episode(client, policy, run["config"], seed, self.stop_event)
                    batch.append((records, row["warningTime"]))
                    number = run["episode"] + 1
                    boundary = number % run["config"]["checkpointFrequency"] == 0 or number == run["totalEpisodes"]
                    if len(batch) >= run["config"]["batchSize"] or boundary:
                        update = policy.update(batch)
                        batch = []
                        row.update(policyLoss=update.get("policy_loss"), valueLoss=update.get("value_loss"),
                                   entropy=update.get("entropy"), optimizerUpdate=update)
                    with self.lock:
                        run["elapsedSeconds"] = time.monotonic() - started
                    self._record(run, row, evidence)
                    if boundary:
                        self._checkpoint(client, policy, run)
                with self.lock:
                    run["status"] = "completed"
        except TrainingStopped:
            with self.lock:
                run["status"] = "stopped"
        except Exception as exc:
            with self.lock:
                run.update(status="failed", error=str(exc))
        finally:
            try:
                if policy is not None:
                    if batch:
                        # A stop preserves learning from completed episodes in the partial batch.
                        update = policy.update(batch)
                        atomic_json(directory / "final_partial_update.json", update)
                    self._save_model(policy, run, "latest")
                    self._save_model(policy, run, "final")
                with self.lock:
                    run.update(elapsedSeconds=time.monotonic() - started, finishedAt=utc_now(), phase="Finished")
                    self._persist(run)
                    atomic_json(directory / "summary.json", {**run["summary"], "status": run["status"],
                        "elapsedSeconds": run["elapsedSeconds"], "error": run["error"], "checkpointCriterion": CHECKPOINT_CRITERION})
                    atomic_json(directory / "metrics.json", run["metrics"])
                    atomic_json(directory / "graph_data.json", {"movingAverageWindow": 20, "metrics": run["metrics"]})
            except Exception as exc:
                with self.lock:
                    run.update(status="failed", error=f"Result save failed: {exc}; original error: {run.get('error')}")
                    self._persist(run)
            finally:
                self.release(run["id"])

    def model_spec(self, run_id, checkpoint="best"):
        if checkpoint not in {"best", "latest", "final"}:
            raise ValueError("Select best, latest or final checkpoint")
        with self.lock:
            run = self._read(run_id)
            if run["status"] in ACTIVE:
                raise ValueError("Stop training before loading its model into Live simulation")
            entry = run["checkpoints"].get(checkpoint)
            if not entry:
                raise ValueError(f"This run has no {checkpoint} checkpoint; select an available saved checkpoint")
            path = self._path(run_id) / "checkpoints" / f"{checkpoint}.json"
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("Checkpoint integrity mismatch; restore the original saved model")
            return {"id": f"trained:{run_id}:{checkpoint}", "label": f"{run_id} · {checkpoint} · {run['algorithm'].upper()}",
                    "path": path, "config": run["config"]}

    def load(self, payload):
        if set(payload) - {"runId", "checkpoint"}:
            raise ValueError("Unexpected model load input")
        model = self.model_spec(payload.get("runId"), payload.get("checkpoint", "best"))
        return {"model": {"id": model["id"], "label": model["label"]}}
