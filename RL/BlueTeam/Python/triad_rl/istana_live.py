"""Loopback-only Istana simulation integration; no hardware control.

The wire client uses the standard library. Planning imports are lazy so native
Automation can exercise transport without a NumPy installation. Only the
validated public Blue snapshot reaches the temporal policy; Red observations
are retained solely as explicitly labelled simulation replay evidence.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import math
import socket
import time


COORDINATE_SYSTEM = "unreal_xy_relative_m_z_up"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_REQUEST_BYTES = 65536


class BridgeProtocolError(ConnectionError):
    """The session cannot safely continue; reconnect and reset the episode."""


class BridgeRejected(RuntimeError):
    """The bridge rejected a request without accepting its action."""


def _integer(value, name, low=0, high=9007199254740991):
    # UE JSON serializes integer-valued numbers as either 1 or 1.0.
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or int(value) != value or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(value)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


class IstanaLiveClient:
    """One synchronous JSON-line session with exact IDs and guarded step clocks.

    Socket/protocol failures close the transport. We never retry an ambiguous
    mutation automatically: the server may already have advanced the episode.
    Server-declared rejections leave the session usable for inspection/reset.
    """

    def __init__(self, port=8765, timeout=120., *, connection=None, record=None):
        port = _integer(port, "port", 1, 65535)
        timeout = _finite(timeout, "timeout")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.socket = connection if connection is not None else socket.create_connection(("127.0.0.1", port), timeout)
        self.socket.settimeout(timeout)
        self.stream = self.socket.makefile("rb")
        self.next_id = 0
        self.red_request_id = self.blue_request_id = 0
        self.red_context = self.blue_context = None
        self.completed_steps = 0
        self.elapsed_seconds = 0.
        self.closed = False
        self.record = record
        self.last_wire = None

    def request(self, op, **fields):
        if self.closed:
            raise BridgeProtocolError("Bridge session is closed; reconnect and reset")
        if not isinstance(op, str) or not op or "id" in fields or "op" in fields:
            raise ValueError("Supply an operation without reserved id/op fields")
        request_id = _integer(self.next_id, "request id")
        payload = {"id": request_id, "op": op, **fields}
        wire = (json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        if len(wire) > MAX_REQUEST_BYTES:
            raise ValueError("Request exceeds the bridge's 64 KiB limit")
        self.next_id += 1
        self.last_wire = wire
        try:
            self.socket.sendall(wire)
            line = self.stream.readline(MAX_RESPONSE_BYTES + 1)
            if not line or len(line) > MAX_RESPONSE_BYTES or not line.endswith(b"\n"):
                raise BridgeProtocolError("Bridge disconnected or response exceeded 4 MiB")
            result = json.loads(line, object_pairs_hook=_object_pairs, parse_constant=_invalid_constant)
            if (not isinstance(result, dict) or isinstance(result.get("id"), bool)
                    or result.get("id") != request_id or type(result.get("ok")) is not bool):
                raise BridgeProtocolError("Bridge returned an invalid response or mismatched request id")
        except (OSError, ValueError, ConnectionError) as error:
            self.close()
            raise BridgeProtocolError(f"{error}; reconnect and reset before continuing") from error
        if self.record:
            self.record({"request": deepcopy(payload), "response": deepcopy(result)})
        if not result["ok"]:
            raise BridgeRejected(str(result.get("error", "Bridge rejected request")))
        return result

    def reset(self, seed=12345):
        seed = _integer(seed, "seed", -(2**31), 2**31 - 1)
        context = self.request("reset", seed=seed)["context"]
        if not isinstance(context.get("runId"), str) or not context["runId"]:
            raise BridgeProtocolError("Reset returned no run ID")
        _integer(context.get("revision"), "revision")
        self.red_context = deepcopy(context)
        self.blue_context = None
        self.completed_steps = 0
        self.elapsed_seconds = 0.
        self.red_request_id = self.blue_request_id = 0
        return deepcopy(context)

    def _require_reset(self):
        if self.red_context is None:
            raise RuntimeError("Call reset before episode operations")

    def get_blue_context(self):
        self._require_reset()
        context = self.request("blue_context")["context"]
        if context.get("runId") != self.red_context["runId"]:
            raise BridgeProtocolError("Blue context belongs to a different run")
        if _integer(context.get("completedSteps"), "completedSteps") != self.completed_steps:
            raise BridgeProtocolError("Stale Blue context clock")
        _integer(context.get("revision"), "Blue revision")
        if context.get("coordinateSystem") != COORDINATE_SYSTEM:
            raise BridgeProtocolError("Unsupported Blue coordinate system")
        self.blue_context = deepcopy(context)
        return deepcopy(context)

    def deploy(self, placements):
        """Atomically commit profile/site identifiers, including an empty STOP layout."""
        self._require_reset()
        if self.blue_context is None:
            raise RuntimeError("Read Blue context before deploying")
        if self.completed_steps != 0 or self.blue_context.get("committed"):
            raise ValueError("The initial Blue layout can only be committed once at step zero")
        rows = []
        for row in placements:
            if set(row) != {"profileId", "siteId"} or not isinstance(row["profileId"], str):
                raise ValueError("Deployment requires only profileId and siteId")
            rows.append({"profileId": row["profileId"], "siteId": _integer(row["siteId"], "siteId", 0, 511)})
        action = {"schemaVersion": 1, "runId": self.blue_context["runId"],
                  "revision": self.blue_context["revision"], "requestId": self.blue_request_id,
                  "expectedStep": self.completed_steps, "placements": rows, "commit": True}
        self.blue_request_id += 1
        response = self.request("blue_deploy", action=action)
        result = response["result"]
        if result.get("accepted") is not True or result.get("committed") is not True:
            raise BridgeRejected(result.get("error", "Blue layout was not committed"))
        self.blue_context["committed"] = True
        if "blueObservation" in response:
            self._check_blue(response["blueObservation"], self.completed_steps)
        return response

    def place_red(self, centers):
        """Red-only placement uses absolute Unreal world centimeters."""
        self._require_reset()
        if len(centers) != _integer(self.red_context["groupCount"], "groupCount", 1):
            raise ValueError("Provide one center per advertised Red group")
        rows = []
        for i, center in enumerate(centers):
            if len(center) != 3:
                raise ValueError("Red center requires x,y,z world centimeters")
            rows.append({"groupId": i, "centerWorldCm": dict(zip(("x", "y", "z"),
                [_finite(value, "Red center") for value in center]))})
        action = {"schemaVersion": 1, "runId": self.red_context["runId"],
                  "revision": self.red_context["revision"], "requestId": self.red_request_id,
                  "centers": rows}
        self.red_request_id += 1
        result = self.request("place", action=action)["result"]
        if result.get("bAccepted") is not True:
            raise BridgeRejected(result.get("error", "Red placement rejected"))
        return result

    def _check_blue(self, observation, completed_steps):
        if (observation.get("runId") != self.red_context["runId"]
                or _integer(observation.get("completedSteps"), "Blue completedSteps") != completed_steps):
            raise BridgeProtocolError("Blue observation has a stale run or clock")
        elapsed = _finite(observation.get("elapsedSeconds"), "elapsedSeconds")
        if elapsed < self.elapsed_seconds:
            raise BridgeProtocolError("Blue simulation clock moved backwards")
        self.elapsed_seconds = elapsed

    def step(self, steps=1):
        self._require_reset()
        steps = _integer(steps, "steps", 1, 1000)
        response = self.request("step", runId=self.red_context["runId"],
                                expectedStep=self.completed_steps, steps=steps)
        red = response["observation"]
        completed = _integer(red.get("completedSteps"), "completedSteps")
        if red.get("runId") != self.red_context["runId"] or not self.completed_steps <= completed <= self.completed_steps + steps:
            raise BridgeProtocolError("Step returned a stale run or invalid clock")
        blue = response["blueObservation"]
        self._check_blue(blue, completed)
        ended = any(red.get(key) is True for key in ("bTerminated", "bTruncated")) or any(
            blue.get(key) is True for key in ("terminated", "truncated"))
        if completed == self.completed_steps and not ended:
            raise BridgeProtocolError("Step made no progress in an active episode")
        self.completed_steps = completed
        return response

    def observe_blue(self):
        self._require_reset()
        observation = self.request("blue_observe")["blueObservation"]
        self._check_blue(observation, self.completed_steps)
        return observation

    def cancel(self):
        self._require_reset()
        return self.request("cancel")

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.stream.close()
            finally:
                self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def scripted_red_centers(context):
    """A fixed smoke/demo control, explicitly not a learned Red policy."""
    origin = context["objectiveWorldCm"]
    radius = (_finite(context["minRadiusCm"], "minRadiusCm") +
              _finite(context["maxRadiusCm"], "maxRadiusCm")) / 2
    count = _integer(context["groupCount"], "groupCount", 1)
    return [[origin["x"] + radius * math.cos(2 * math.pi * i / count),
             origin["y"] + radius * math.sin(2 * math.pi * i / count),
             origin["z"] + context["heightOffsetCm"]] for i in range(count)]


def public_planning_inputs(context):
    """Extract and validate an allowlisted snapshot; never forward the envelope."""
    from .adaptive_inputs import validate_catalogue, validate_public_state
    from .temporal_inputs import TemporalConfig

    if context.get("coordinateSystem") != COORDINATE_SYSTEM:
        raise ValueError("Expected objective-relative Unreal XY meters with Z up")
    if _integer(context["completedSteps"], "completedSteps") != 0 or context.get("committed"):
        raise ValueError("This runner plans one initial layout before the episode advances")
    catalogue = validate_catalogue(context["catalogue"])
    state = validate_public_state(context["publicSnapshot"], catalogue)
    if state["timestamp"] != 0 or state["done"] or state["placements"]:
        raise ValueError("Initial Blue snapshot must be uncommitted at simulation time zero")
    # UE serializes whole-valued doubles as JSON integers (20 rather than 20.0).
    # Restore the declared numeric types at this wire boundary so the frozen
    # checkpoint contract remains strict without rejecting equivalent values.
    config_values = dict(context["temporalConfig"])
    for name, default in asdict(TemporalConfig()).items():
        if name in config_values:
            config_values[name] = (_finite(config_values[name], name) if type(default) is float
                                   else _integer(config_values[name], name))
    config = TemporalConfig(**config_values)
    return state, catalogue, config


def make_plan(context, *, checkpoint=None, temporal_public_control=False, common_sense=False):
    """Use an explicit checkpoint or an explicitly selected non-RL control."""
    from .adaptive_inputs import LiveObservationAdapter, apply_placement
    from .temporal_inputs import TemporalObservationBuilder, TemporalPublicGreedy
    from .temporal_policy import TemporalPolicy

    if sum((checkpoint is not None, bool(temporal_public_control), bool(common_sense))) != 1:
        raise ValueError("Choose exactly one explicit checkpoint, temporal public control, or common-sense baseline")
    state, catalogue, config = public_planning_inputs(context)
    if common_sense:
        from .common_sense import plan_common_sense
        plan = plan_common_sense(state, catalogue)
    elif checkpoint is not None:
        from recommend_temporal import recommend_layout
        policy = TemporalPolicy.load(checkpoint, config=config)
        plan = recommend_layout(policy, state, config=config, catalogue=catalogue)
        plan["selection"] = {"kind": "explicit_checkpoint", "checkpoint": str(checkpoint),
                             "actor_updates": policy.actor_update_count,
                             "claim": "User-selected experimental checkpoint; no winner or transfer-performance claim"}
    else:
        builder, policy = TemporalObservationBuilder(config), TemporalPublicGreedy()
        decisions, final = [], deepcopy(state)
        for _ in range(state["max_sites"] + 1):
            observation = builder.observe(final, catalogue)
            action = policy.act(observation)
            row = LiveObservationAdapter.recommendation(observation, action)
            decisions.append(row)
            final = apply_placement(final, action, catalogue)
            if row["stop"]:
                break
        else:
            raise RuntimeError("Temporal control did not commit a bounded layout")
        plan = {"schema": "istana.temporal_public_control_plan.v1", "decisions": decisions,
                "new_placements": [row for row in decisions if not row["stop"]],
                "final_public_state": final, "temporal_config": asdict(config),
                "selection": {"kind": "temporal_public_control", "claim": "Non-RL public-model control"}}
    # Do not rewrite the existing offline recommendation's no-device-command
    # status; this separate wrapper records when Unreal accepts the layout.
    return {"schema": "istana.blue_live_plan.v1", "runId": context["runId"],
            "revision": context["revision"], "public_only": True,
            "coordinateSystem": COORDINATE_SYSTEM, "recommendation": plan,
            "placements": [{"profileId": row["sensor_id"], "siteId": row["site_index"]}
                           for row in plan["new_placements"]]}


def placement_world_cm(context, placement):
    """For display/audit only. Native deploy resolves IDs, never accepts these coordinates."""
    site_id = _integer(placement["siteId"], "siteId", 0, len(context["publicSnapshot"]["sites"]) - 1)
    profile = next(row for row in context["catalogue"] if row["id"] == placement["profileId"])
    east, north = context["publicSnapshot"]["sites"][site_id]
    origin = context["worldOriginCm"]
    if "siteSurfacesWorldCm" in context:
        surfaces = context["siteSurfacesWorldCm"]
        if not isinstance(surfaces, list) or len(surfaces) != len(context["publicSnapshot"]["sites"]):
            raise ValueError("Invalid surface catalogue")
        base = surfaces[site_id]
        if not isinstance(base, list) or len(base) != 3:
            raise ValueError("Sensor site has no supporting surface")
        x, y, z = [_finite(v, "surface coordinate") for v in base]
        if not (math.isclose(x, origin["x"] + east * 100., abs_tol=.1)
                and math.isclose(y, origin["y"] + north * 100., abs_tol=.1)):
            raise ValueError("Surface does not match approved site")
        return {"x": x, "y": y, "z": z + profile["height_m"] * 100.}
    if context.get("placementRule") == "static_surface_mast_v1":
        raise ValueError("Surface-mounted context is missing surface coordinates")
    # Backward-compatible audit of older recorded wire contexts only.
    return {"x": origin["x"] + east * 100., "y": origin["y"] + north * 100.,
            "z": origin["z"] + profile["height_m"] * 100.}


def run_episode(client, *, seed=12345, checkpoint=None, temporal_public_control=False, common_sense=False,
                red_policy=None, red_deterministic=True,
                max_steps=5000, step_batch=10, paced=False, frame=None):
    """One Blue plan, one Red placement decision, then real native fixed steps.

    Native terminal metrics are measured synthetic simulation outcomes. They
    are never substituted with the planner's forecast or offline benchmark.
    """
    max_steps = _integer(max_steps, "max_steps", 1, 1_000_000)
    step_batch = _integer(step_batch, "step_batch", 1, 1000)
    red_context = client.reset(seed)
    blue_context = client.get_blue_context()
    plan = make_plan(blue_context, checkpoint=checkpoint, temporal_public_control=temporal_public_control,
                     **({"common_sense": True} if common_sense else {}))
    deployed = client.deploy(plan["placements"])
    if red_policy is None:
        from .red_policy import ScriptedRadialRedPolicy
        red_policy = ScriptedRadialRedPolicy()
    red_decision = red_policy.select(red_context, blue_context=blue_context,
                                     blue_placements=plan["placements"],
                                     deterministic=red_deterministic)
    centers = red_decision["centers"]
    red_result = client.place_red(centers)
    blue = client.observe_blue()
    final_red = None
    start_wall = time.monotonic()
    while not blue["terminated"] and not blue["truncated"] and client.completed_steps < max_steps:
        response = client.step(min(step_batch, max_steps - client.completed_steps))
        blue = response["blueObservation"]
        if frame:
            frame({"type": "simulation_frame", "blue": blue, "red_truth_for_replay_only": response["observation"]})
        if paced:
            remaining = blue["elapsedSeconds"] - (time.monotonic() - start_wall)
            if remaining > 0:
                time.sleep(min(remaining, 60.))
        red = response["observation"]
        final_red = red
        if (red.get("bTerminated") or red.get("bTruncated")) and not (blue["terminated"] or blue["truncated"]):
            # Ask the coordinator to finalize if the Red clock stopped first.
            client.cancel()
            blue = client.observe_blue()
    limit_reached = not blue["terminated"] and not blue["truncated"]
    if limit_reached:
        client.cancel()
        blue = client.observe_blue()
    return {"schema": "istana.blue_live_run.v1", "runId": red_context["runId"], "seed": seed,
            "environment": "Unreal Istana live synthetic simulation", "physical_commands_sent": False,
            "blue_policy": plan["recommendation"]["selection"], "red_policy": red_decision["policy"],
            "red_decision": red_decision,
            "red_context": red_context, "public_blue_context": blue_context,
            "plan": plan, "deployment_result": deployed["result"],
            "accepted_sensor_world_cm": [placement_world_cm(blue_context, row) for row in plan["placements"]],
            "red_placement_accepted": red_result["bAccepted"], "runner_step_limit_reached": limit_reached,
            "completed_steps": client.completed_steps, "final_blue_observation": blue,
            "measured_red_reward": (final_red.get("reward") if final_red and final_red.get("bHasReward") else
                                    -blue["reward"] if blue.get("metricsAvailable") else None),
            "measured_synthetic_metrics": deepcopy(blue["metrics"]) if blue.get("metricsAvailable") else None,
            "limitations": "Experimental simulator integration. Sensor models and rewards are synthetic. A live run does not establish real-world calibration or improvement over frozen offline baselines."}
