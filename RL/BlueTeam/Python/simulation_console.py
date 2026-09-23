"""Local visual console for published replays and the optional Unreal bridge.

Run from any directory with the project's Python environment. No cloud service.
The HTTP controller owns one serial bridge session. Blue plans receive only the
public snapshot; the observer display separately shows simulator drone truth.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import threading
import time

from triad_rl.istana_live import IstanaLiveClient, make_plan, scripted_red_centers
from triad_rl.trained_models import TrainedModelRegistry
from triad_rl.training_workbench import INITIALIZATIONS, TrainingManager
from triad_rl.warning_algorithms import TRAINING_ALGORITHMS
from native_comparison import NativeComparisons, policy_label

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "console"
BLUE_ROOT = ROOT.parent


def load_replays(path=None):
    page = (Path(path) if path else BLUE_ROOT / "Results/temporal-v6-demo.html").read_text(encoding="utf-8")
    match = re.search(r'<script id="replay-data" type="application/json">(.*?)</script>', page, re.S)
    if not match:
        raise ValueError("Published replay data is missing")
    result = json.loads(match[1])
    if not result.get("replays") or any(not r.get("frames") for r in result["replays"]):
        raise ValueError("Published replay has no frames")
    return result["replays"]


def replay_view(replay):
    """Read archived evidence; never resimulate, filter cases or infer new results."""
    return {"mode": "recorded", "label": f"{policy_label(replay['temporal_seed'])} · {replay['profile']} · case {replay['case_index']}",
            "coordinateLabel": "Synthetic evaluation arena · objective-relative metres",
            "catalogue": replay["catalogue"], "placements": replay["placements"],
            "sites": replay["scenario"]["public"]["sites"],
            "budget": replay["scenario"]["public"]["budget_total"],
            "objectiveRadius": replay["scenario"].get("objective_radius", 20),
            "frames": replay["frames"], "metrics": replay["metrics"],
            "decisions": replay["decisions"], "seed": replay["seed"],
            "weather": replay["scenario"]["public"]["weather"],
            "policy": policy_label(replay['temporal_seed']),
            "outcome": replay["metrics"]["outcome"],
            "audit": {"recorded": True, "live_unreal": False, "policy_truth_access": False}}


class ConsoleState:
    def __init__(self, bridge_port=8765, client_factory=IstanaLiveClient, planner=make_plan,
                 trained_models=None):
        self.bridge_port, self.client_factory, self.planner = bridge_port, client_factory, planner
        self.trained_models = trained_models
        self.lock = threading.RLock()
        self.client = None
        self.view = None
        self.context = None
        self.error = ""

    def close(self):
        if self.client is not None:
            self.client.close()
        self.client = None

    def status(self):
        with self.lock:
            return {"connected": self.client is not None and not self.client.closed,
                    "episode": self.view is not None, "error": self.error,
                    "bridgePort": self.bridge_port}

    def frame(self, red, blue):
        origin = self.context["worldOriginCm"]
        threats = []
        for drone in red.get("drones", []):
            p = drone["positionCm"]
            threats.append({"id": f"drone-{drone['droneId']}",
                "position": [(p[k] - origin[k]) / 100 for k in ("x", "y", "z")],
                "active": drone.get("bActive", True), "observer_truth": True})
        return {"time": blue["elapsedSeconds"], "threats": threats, "detections": [],
                "tracks": blue["publicSnapshot"]["tracks"], "completedSteps": blue["completedSteps"],
                "directionalDiagnostics": blue.get("directionalDiagnosticsForPresentationOnly", [])}

    def action(self, operation, payload):
        with self.lock:
            try:
                if operation == "connect":
                    if self.client is None or self.client.closed:
                        self.close()
                        self.client = self.client_factory(self.bridge_port, timeout=10.)
                    # A transport connection alone does not imply Blue is present.
                    reply = self.client.request("blue_context")
                    if reply["context"].get("coordinateSystem") != "unreal_xy_relative_m_z_up":
                        raise ValueError("Unreal does not expose the Blue integration contract")
                    self.error = ""
                    return self.status()
                if operation == "disconnect":
                    self.close()
                    self.view = self.context = None
                    self.error = ""
                    return self.status()
                if self.client is None or self.client.closed:
                    raise ConnectionError("Start the compiled Unreal project with -IstanaBlueLive, then connect.")
                if operation == "preview":
                    # Display saved output only; never invoke a planner or deploy Red.
                    from model_switch_demo import load_layouts, LAYOUT_SOURCE, ROOT as PROJECT_ROOT
                    from capture_warning_3d import capture
                    selection = payload.get("policy", "")
                    if not isinstance(selection, str) or not selection.startswith(("saved-", "trained-")):
                        raise ValueError("Select an available saved layout")
                    if selection.startswith("trained-"):
                        if self.trained_models is None:
                            raise ValueError("Named trained models are unavailable")
                        model = self.trained_models.get(selection)
                        key = selection
                        row = {"label": f"{model['name']} · {model.get('algorithmLabel', 'REINFORCE')}",
                               "seed": model["evaluationSeed"],
                               "placements": model["deploymentPlacements"]}
                    else:
                        layouts = load_layouts(LAYOUT_SOURCE)
                        key = selection.removeprefix("saved-")
                        if key not in layouts:
                            raise ValueError("Select an available saved layout")
                        row = layouts[key]
                    self.client.reset(row["seed"])
                    self.context = self.client.get_blue_context()
                    # Archived layouts predate orientation. Keep their positions
                    # unchanged and use the first advertised neutral bins only
                    # for this explicitly labelled saved-output preview.
                    preview_placements = [{**placement,
                        "yawDeg": placement.get("yawDeg", 0.), "pitchDeg": placement.get("pitchDeg", 0.)}
                        for placement in row["placements"]]
                    self.client.deploy(preview_placements)
                    capture(self.client, PROJECT_ROOT/"Saved", "layout")
                    time.sleep(.3)
                    shot = capture(self.client, PROJECT_ROOT/"Saved", "layout")
                    blue = self.client.observe_blue()
                    audit_dir = PROJECT_ROOT/"Saved/ConsoleModelDemo"
                    audit_dir.mkdir(parents=True, exist_ok=True)
                    (audit_dir/f"{self.context['runId']}.json").write_text(json.dumps(
                        {**row, "model": key, "placements": preview_placements, "shot": shot,
                         "completed_steps": self.client.completed_steps}, indent=2))
                    self.view = {"mode": "live", "label": f"{row['label']} · saved layout",
                        "coordinateLabel": "Native Unreal · archived output preview",
                        "catalogue": self.context["catalogue"], "placements": blue["publicSnapshot"]["placements"],
                        "sites": self.context["publicSnapshot"]["sites"], "surfaceMounted": True,
                        "budget": self.context["publicSnapshot"]["budget_total"], "objectiveRadius": 20,
                        "policy": f"{row['label']} (saved output)", "metrics": None, "outcome": "preview",
                        "ended": True, "frames": [self.frame({"drones": []}, blue)],
                        "nativePreview": {"image": "data:image/png;base64," + base64.b64encode(Path(shot["path"]).read_bytes()).decode(),
                                          "anchors": shot["sensor_screen_anchors"]}}
                elif operation == "reset":
                    seed = payload.get("seed", 12345)
                    if type(seed) is not int or not -(2**31) <= seed < 2**31:
                        raise ValueError("Episode seed must be a signed 32-bit integer")
                    policy = payload.get("policy", "406")
                    if policy not in ("406", "407", "408", "greedy", "control", "common_sense"):
                        raise ValueError("Select an available RL policy, greedy, or common-sense placement")
                    greedy = policy in ("greedy", "control")  # retain the old API alias
                    common_sense = policy == "common_sense"
                    self.view = self.context = None
                    red_context = self.client.reset(seed)
                    self.context = self.client.get_blue_context()
                    checkpoint = None if greedy or common_sense else BLUE_ROOT / f"Results/temporal-v6-pilot/training/seed-{policy}/last"
                    plan = self.planner(self.context, checkpoint=checkpoint, temporal_public_control=greedy,
                                        **({"common_sense": True} if common_sense else {}))
                    self.client.deploy(plan["placements"])
                    placed = self.client.place_red(scripted_red_centers(red_context))
                    blue = self.client.observe_blue()
                    self.view = {"mode": "live", "label": "Istana · live episode", "seed": seed,
                        "coordinateLabel": "Istana top-down telemetry · objective-relative XY metres",
                        "catalogue": self.context["catalogue"],
                        "placements": blue["publicSnapshot"]["placements"],
                        "sites": self.context["publicSnapshot"]["sites"],
                        "blockedSites": self.context["publicSnapshot"].get("blocked_sites", []),
                        "surfaceMounted": self.context.get("placementRule") in
                            ("static_surface_mast_v1", "static_surface_directional_mast_v2"),
                        "budget": self.context["publicSnapshot"]["budget_total"],
                        "objectiveRadius": self.context["temporalConfig"]["objective_radius_m"],
                        "fixedStepSeconds": self.context.get("fixedStepSeconds", .05),
                        "timeLimitSeconds": self.context.get("timeLimitSeconds", 96.),
                        "stepDurationSeconds": 10 * self.context.get("fixedStepSeconds", .05),
                        "weather": self.context["publicSnapshot"]["weather"],
                        "policy": ("Common-sense placement (non-RL heuristic)" if common_sense else
                                   "Greedy placement (non-RL)" if greedy else policy_label(policy)),
                        "decisions": plan["recommendation"]["decisions"],
                        "frames": [self.frame({"drones": placed["initialStates"]}, blue)],
                        "metrics": None, "outcome": "running", "ended": False,
                        "audit": {"live_unreal": True, "red_control": "scripted centers", "policy_truth_access": False}}
                elif operation == "step":
                    if self.view is None:
                        raise ValueError("Plan an episode before stepping")
                    if not self.view["ended"]:
                        # Each request advances a bounded half-second (10 x 50ms by default).
                        response = self.client.step(10)
                        blue = response["blueObservation"]
                        self.view["frames"].append(self.frame(response["observation"], blue))
                        self.view["ended"] = bool(blue["terminated"] or blue["truncated"])
                        self.view["outcome"] = blue["reason"] if self.view["ended"] else "running"
                        self.view["metrics"] = ({**blue["metrics"], "return": blue["reward"]}
                                                if blue["metricsAvailable"] else None)
                else:
                    raise ValueError("Unknown console action")
                self.error = ""
                return deepcopy(self.view)
            except Exception as exc:
                self.error = ("Unreal is not listening on the local bridge. Build and launch the project with -IstanaBlueLive, then reconnect."
                              if isinstance(exc, ConnectionRefusedError) else str(exc))
                # Ambiguous mutations or a failed reset cannot be resumed with stale UI state.
                self.close()
                self.view = self.context = None
                if isinstance(exc, ConnectionRefusedError):
                    raise ConnectionError(self.error) from exc
                raise


def make_server(port=9048, bridge_port=8765, *, state=None, replays=None, comparisons=None, trainer=None):
    model_root = BLUE_ROOT.parents[1] / "Saved/WarningTraining/models"
    registry = (getattr(trainer, "registry", None) or getattr(comparisons, "registry", None)
                or TrainedModelRegistry(model_root))
    state = state or ConsoleState(bridge_port, trained_models=registry)
    if getattr(state, "trained_models", None) is None:
        state.trained_models = registry
    replays = replays if replays is not None else load_replays()
    comparisons = comparisons if comparisons is not None else NativeComparisons(registry=registry)
    trainer = trainer or TrainingManager(
        bridge_port, BLUE_ROOT.parents[1] / "Saved/WarningTraining", registry=registry)
    token = secrets.token_urlsafe(32)
    # Saved native evidence is read independently of the live bridge session.
    comparison_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def valid_host(self):
            expected = self.server.server_address[1]
            return self.headers.get("Host") in (f"127.0.0.1:{expected}", f"localhost:{expected}")

        def reply(self, value, code=200, content_type="application/json"):
            raw = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if not self.valid_host():
                return self.reply({"error": "Invalid local host"}, 403)
            if self.path == "/api/session":
                named_models = trainer.registry.list()
                archived_models = ([{"id": "saved-rl", "label": "RL Policy · saved output",
                                     "kind": "archived"},
                                    {"id": "saved-greedy", "label": "Greedy · saved output",
                                     "kind": "archived"},
                                    {"id": "saved-initial", "label": "Initial Policy · saved output",
                                     "kind": "archived"}]
                                   if all((BLUE_ROOT/f"Results/model-switch-demo/evaluation-{key}.json").exists()
                                          for key in ("0000", "0128", "greedy")) else [])
                return self.reply({"token": token, "status": state.status(), "replays": [
                    {"id": i, "profile": r["profile"], "policy": r["temporal_seed"], "case": r["case_index"],
                     "outcome": r["metrics"]["outcome"]} for i, r in enumerate(replays)],
                    "comparisonEpisodes": comparisons.list(),
                    "comparisonLayouts": comparisons.layouts(), "training": trainer.status(),
                    "trainingInitializations": [{"id": key, "label": label}
                                                for key, label in INITIALIZATIONS.items()],
                    "trainingAlgorithms": [{"id": key, "label": row["label"],
                                             "description": row["description"]}
                                            for key, row in TRAINING_ALGORITHMS.items()],
                    "trainedModels": [{"id": row["id"],
                                       "label": f"{row['name']} · {row.get('algorithmLabel', 'REINFORCE')}",
                                       "bestEpisode": row["bestEpisode"],
                                       "bestWarningSeconds": row["bestWarningSeconds"]}
                                      for row in named_models],
                    "savedModels": archived_models + [
                        {"id": row["id"],
                         "label": f"{row['name']} · {row.get('algorithmLabel', 'REINFORCE')} best",
                         "kind": "trained"} for row in named_models]})
            if self.path == "/api/status":
                return self.reply(state.status())
            if self.path == "/api/training/status":
                return self.reply(trainer.status())
            match = re.fullmatch(r"/api/replay/(\d+)", self.path)
            if match and int(match[1]) < len(replays):
                return self.reply(replay_view(replays[int(match[1])]))
            static = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/presentation.css": ("presentation.css", "text/css; charset=utf-8"),
                      "/style.css": ("style.css", "text/css; charset=utf-8")}
            if self.path in static:
                name, mime = static[self.path]
                return self.reply((ASSETS / name).read_bytes(), content_type=mime)
            self.reply({"error": "Not found"}, 404)

        def do_POST(self):
            allowed_origin = {f"http://127.0.0.1:{self.server.server_address[1]}", f"http://localhost:{self.server.server_address[1]}"}
            if (not self.valid_host() or self.headers.get("Origin") not in allowed_origin
                    or self.headers.get("X-Console-Token") != token):
                return self.reply({"error": "Local session required"}, 403)
            match = re.fullmatch(r"/api/action/(connect|disconnect|reset|step|preview)", self.path)
            comparison = re.fullmatch(r"/api/comparison/(scenario|run)", self.path)
            training = re.fullmatch(r"/api/training/(start|stop)", self.path)
            if not match and not comparison and not training:
                return self.reply({"error": "Unknown action"}, 404)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096 or self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected a small JSON object")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object")
                if training:
                    if training[1] == "start":
                        with state.lock:
                            state.close()
                            state.view = state.context = None
                            state.error = ""
                        result = trainer.start(payload)
                    else:
                        if payload:
                            raise ValueError("Stop does not accept training parameters")
                        result = trainer.stop()
                elif comparison:
                    if set(payload) not in ({"episodeId"}, {"episodeId", "layoutId"}):
                        raise ValueError("Choose a native episode; comparison layouts are fixed and cannot be edited")
                    with comparison_lock:
                        result = comparisons.get(payload["episodeId"],
                                                 scenario=comparison[1] == "scenario",
                                                 layout_id=payload.get("layoutId", "matched_common_sense"))
                else:
                    if trainer.status()["running"]:
                        raise ValueError("Live controls are reserved by the active Training run")
                    result = state.action(match[1], payload)
                self.reply(result)
            except (ValueError, KeyError, RuntimeError, OSError) as error:
                self.reply({"error": str(error)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.console_state = state
    server.training_manager = trainer
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9048)
    parser.add_argument("--bridge-port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = make_server(args.port, args.bridge_port)
    print(f"Istana simulation console: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.training_manager.close()
        server.console_state.close()
        server.server_close()


if __name__ == "__main__":
    main()
