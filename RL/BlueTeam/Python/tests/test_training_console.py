"""Training HTTP protection, exclusive native ownership and live model integration."""
from copy import deepcopy
import http.client
import json
import threading

import pytest

from simulation_console import ConsoleState, make_server
from test_training_manager import ProtocolFixture, configuration, finish


class ConsoleProtocolFixture(ProtocolFixture):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.closed = False
        self.closes = 0
        self.completed_steps = 0
        self.context["worldOriginCm"] = {"x": 0., "y": 0., "z": 0.}

    def __call__(self, port, timeout):
        self.closed = False
        return super().__call__(port, timeout)

    def close(self):
        self.closed = True
        self.closes += 1

    def reset(self, seed, *, blue_configuration):
        result = super().reset(seed, blue_configuration=blue_configuration)
        self.completed_steps = 0
        self.context["runId"] = result["runId"]
        return result

    def place_red(self, centers):
        super().place_red(centers)
        return {"initialStates": []}

    def observe_blue(self):
        state = deepcopy(self.context["publicSnapshot"])
        if self.layouts:
            state["placements"] = [{"sensor_id": row["profileId"], "position": state["sites"][row["siteId"]],
                                    "cost": 1., "yaw_deg": row["yawDeg"], "pitch_deg": row["pitchDeg"]}
                                   for row in self.layouts[-1]]
        return {"elapsedSeconds": 0., "publicSnapshot": state, "completedSteps": self.completed_steps}

    def step(self, count):
        result = super().step(100)
        blue = result["blueObservation"]
        if blue["terminated"]:
            self.completed_steps += count
            blue.update(publicSnapshot=self.observe_blue()["publicSnapshot"], completedSteps=self.completed_steps,
                        reason="terminated")
            result["observation"] = {"drones": []}
        return result


@pytest.fixture
def server(tmp_path):
    bridge = ConsoleProtocolFixture()
    state = ConsoleState(client_factory=bridge)
    httpd = make_server(0, state=state, replays=[], training_root=tmp_path)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    httpd.fixture_bridge = bridge
    yield httpd
    bridge.release.set()
    httpd.training_manager.close()
    httpd.shutdown()
    httpd.server_close()
    state.close()
    thread.join(timeout=2.)


def request(server, path, body=None, *, authorize=True, origin=None, extra_headers=None):
    headers = dict(extra_headers or {})
    if body is not None:
        headers["Content-Type"] = "application/json"
        if authorize:
            _, session = request(server, "/api/session")
            headers.update(Origin=origin or f"http://127.0.0.1:{server.server_port}",
                           **{"X-Console-Token": session["token"]})
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=20.)
    try:
        connection.request("GET" if body is None else "POST", path,
                           body=None if body is None else json.dumps(body), headers=headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


@pytest.mark.parametrize("operation", ["start", "stop", "load"])
def test_training_mutations_require_same_origin_and_session_token(server, operation):
    path = f"/api/training/{operation}"
    assert request(server, path, {}, authorize=False)[0] == 403
    assert request(server, path, {}, origin="https://attacker.invalid")[0] == 403
    assert not server.fixture_bridge.resets


def test_foreign_host_cannot_read_training_history(server):
    assert request(server, "/api/training/runs", extra_headers={"Host": "attacker.invalid"})[0] == 403


def test_options_and_history_are_readonly_and_preserve_live_view(server):
    state, bridge = server.console_state, server.fixture_bridge
    state.client = bridge
    state.view, state.context = {"existing": "episode"}, {"existing": "context"}
    status, options = request(server, "/api/training/options")
    assert status == 200 and options["environment"]["available"]
    assert {row["id"] for row in options["algorithms"]} == {"reinforce", "ppo"}
    assert state.view == {"existing": "episode"} and state.context == {"existing": "context"}
    assert not bridge.resets and bridge.closes == 0
    assert request(server, "/api/training/runs") == (200, {"runs": []})


def test_http_start_history_load_and_trained_live_reset_use_the_same_context(server):
    status, result = request(server, "/api/training/start", configuration(episodes=1))
    assert status == 200 and result["run"]["runName"] == "trial"
    run = finish(server.training_manager)
    assert run["status"] == "completed", run.get("error")
    assert server.console_state.training_owner is None
    assert request(server, "/api/training/runs")[1]["runs"][0]["runName"] == "trial"
    assert request(server, "/api/training/run/trial")[1]["episode"] == 1
    status, result = request(server, "/api/training/load", {"runId": "trial", "checkpoint": "final"})
    assert status == 200, result
    assert result["model"]["id"] == "trained:trial:final"
    assert result["status"]["connected"]
    status, view = request(server, "/api/action/reset", {"policy": result["model"]["id"], "seed": 19})
    assert status == 200, view
    assert view["mode"] == "live" and view["budget"] == 2.
    assert view["audit"]["live_unreal"]
    assert "trial" in view["policy"]
    assert server.fixture_bridge.resets[-1] == {"budget": 2., "availableSensorIds": ["thermal"]}
    status, view = request(server, "/api/action/step", {})
    assert status == 200 and view["ended"]
    assert view["metrics"]["mean_drone_warning_seconds_lower_bound"] == 9.


def test_training_owns_native_until_stopped_but_http_polling_remains_available(server):
    bridge = server.fixture_bridge
    bridge.block_after = 3  # finish initial selection, block on first training request
    status, _ = request(server, "/api/training/start", configuration(episodes=10))
    assert status == 200 and bridge.blocked.wait(10.)
    assert request(server, "/api/status")[1]["trainingRun"] == "trial"
    for action in ("connect", "disconnect", "reset", "step"):
        status, result = request(server, f"/api/action/{action}", {})
        assert status == 400 and "Training owns" in result["error"]
    assert request(server, "/api/training/start", configuration("other"))[0] == 400
    assert request(server, "/api/training/run/trial")[0] == 200
    assert request(server, "/api/training/options")[0] == 200
    status, result = request(server, "/api/training/stop", {"runId": "trial"})
    assert status == 200 and result["run"]["status"] == "stopping"
    bridge.release.set()
    assert finish(server.training_manager)["status"] == "stopped"
    assert server.console_state.training_owner is None


@pytest.mark.parametrize("payload", [[], {"runName": "../outside"}, {"unexpected": True},
                                      {**configuration(), "budget": float("nan")},
                                      {**configuration(), "episodes": True},
                                      {**configuration(), "algorithm": "fake_ppo"}])
def test_invalid_training_input_returns_http_error_without_native_mutation(server, payload):
    status, result = request(server, "/api/training/start", payload)
    assert status == 400 and result["error"]
    assert not server.fixture_bridge.resets


def test_incompatible_model_load_clears_old_live_state_after_native_reset(server):
    request(server, "/api/training/start", configuration(episodes=1))
    assert finish(server.training_manager)["status"] == "completed"
    state = server.console_state
    state.view, state.context = {"stale": "live view"}, {"stale": "live context"}
    server.fixture_bridge.context["sensorModel"] = "changed-native-simulation"
    status, result = request(server, "/api/training/load", {"runId": "trial", "checkpoint": "final"})
    assert status == 400 and "incompatible" in result["error"]
    assert state.view is None and state.context is None
    assert not state.status()["connected"]
