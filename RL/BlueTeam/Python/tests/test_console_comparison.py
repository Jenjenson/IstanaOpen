"""The comparison API works offline and preserves the live/recorded sessions."""
import http.client
import json
import threading

import pytest

from simulation_console import ConsoleState, load_replays, make_server


@pytest.fixture(scope="module")
def server():
    state = ConsoleState()
    state.view = {"existing_live_episode": True}
    httpd = make_server(0, state=state, replays=load_replays())
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=2)


def request(server, path, body=None, *, authorize=True):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=20)
    headers = {}
    if body is not None:
        headers = {"Content-Type": "application/json"}
        if authorize:
            _, session = request(server, "/api/session")
            headers.update({"Origin": f"http://127.0.0.1:{server.server_port}",
                            "X-Console-Token": session["token"]})
    try:
        connection.request("GET" if body is None else "POST", path,
                           body=None if body is None else json.dumps(body), headers=headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


@pytest.mark.parametrize("action", ["scenario", "run"])
def test_comparison_requires_same_origin_session(server, action):
    code, _ = request(server, f"/api/comparison/{action}", {"replayId": 0}, authorize=False)
    assert code == 403


@pytest.mark.parametrize("replay_id", [None, -1, 18, True, "0", 0.0, []])
def test_comparison_rejects_invalid_case_without_touching_live_state(server, replay_id):
    code, result = request(server, "/api/comparison/run", {"replayId": replay_id})
    assert code == 400 and "error" in result
    assert server.console_state.view == {"existing_live_episode": True}


def test_public_scenario_and_paired_comparison_work_without_unreal(server):
    code, scenario = request(server, "/api/comparison/scenario", {"replayId": 0})
    assert code == 200 and scenario["replayId"] == 0
    assert scenario["sites"] and scenario["catalogue"]
    assert not {"targets", "frames", "metrics", "rl"} & scenario.keys()
    code, result = request(server, "/api/comparison/run", {"replayId": 0})
    assert code == 200 and result["method"] == "common_sense"
    assert result["rl"]["frames"] and result["baseline"]["frames"]
    assert result["rl"]["budget"] == result["baseline"]["budget"] == scenario["budget"]
    assert not server.console_state.status()["connected"]
    assert server.console_state.view == {"existing_live_episode": True}


def test_manual_empty_layout_is_a_real_zero_detection_comparison(server):
    code, result = request(server, "/api/comparison/run", {
        "replayId": 0, "baseline": "manual", "placements": []})
    assert code == 200
    assert result["baseline"]["placements"] == []
    assert result["baseline"]["metrics"]["detected_fraction"] == 0


@pytest.mark.parametrize("body", [
    {"replayId": 0, "baseline": "unknown"},
    {"replayId": 0, "baseline": "manual", "placements": [{"sensor_id": "unknown", "site_index": 0}]},
    {"replayId": 0, "baseline": "manual", "placements": "invalid"},
    {"replayId": 0, "budget": 1000},
])
def test_comparison_invalid_input_is_an_actionable_http_error(server, body):
    code, result = request(server, "/api/comparison/run", body)
    assert code == 400 and result["error"]
    assert server.console_state.view == {"existing_live_episode": True}
