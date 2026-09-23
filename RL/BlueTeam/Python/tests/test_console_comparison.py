"""Fixed native comparison selection is read-only and cannot reset Live."""
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
    code, _ = request(server, f"/api/comparison/{action}", {"episodeId": "native-406-1"}, authorize=False)
    assert code == 403


@pytest.mark.parametrize("identifier", [None, -1, True, "unknown", 0.0, []])
def test_invalid_selection_does_not_touch_live_state(server, identifier):
    code, result = request(server, "/api/comparison/run", {"episodeId": identifier})
    assert code == 400 and "error" in result
    assert server.console_state.view == {"existing_live_episode": True}


def test_session_offers_native_cases_separately_from_historical_replays(server):
    code, session = request(server, "/api/session")
    assert code == 200 and len(session["replays"]) == 18
    archived = [row for row in session["comparisonEpisodes"] if not row.get("trainedModel")]
    assert len(archived) == 9
    assert {row["policyLabel"] for row in archived} == {
        "Temporal RL · policy A", "Temporal RL · policy B", "Temporal RL · policy C"}
    assert [row["id"] for row in session["comparisonLayouts"]] == [
        "matched_common_sense", "directional_balanced_8"]


def test_native_recordings_work_without_unreal_and_preserve_live(server):
    code, scenario = request(server, "/api/comparison/scenario", {"episodeId": "native-406-1"})
    assert code == 200 and scenario["sites"] and scenario["catalogue"]
    code, result = request(server, "/api/comparison/run", {"episodeId": "native-406-1"})
    assert code == 200 and result["method"] == "common_sense"
    assert result["rl"]["frames"] and result["baseline"]["frames"]
    assert result["audit"]["native_unreal_capture"]
    assert result["metrics"]["rl"]["target_count"] == result["metrics"]["baseline"]["target_count"] == 60
    assert not server.console_state.status()["connected"]
    assert server.console_state.view == {"existing_live_episode": True}


def test_eight_directional_placement_has_matched_native_warning_metrics(server):
    body = {"episodeId": "native-406-1", "layoutId": "directional_balanced_8"}
    code, scenario = request(server, "/api/comparison/scenario", body)
    assert code == 200 and not scenario["layoutOnly"] and scenario["budget"] == 8
    code, result = request(server, "/api/comparison/run", body)
    assert code == 200 and not result["layoutOnly"]
    assert result["metrics"]["rl"]["target_count"] == 5
    assert len(result["baseline"]["placements"]) == 8
    assert result["fairness"]["matched"] is True
    assert server.console_state.view == {"existing_live_episode": True}


def test_unknown_fixed_comparison_placement_is_rejected(server):
    code, result = request(server, "/api/comparison/run", {
        "episodeId": "native-406-1", "layoutId": "manual"})
    assert code == 400 and "available fixed" in result["error"]


@pytest.mark.parametrize("extra", [{"baseline": "manual"}, {"placements": []}, {"budget": 1000}, {"replayId": 0}])
def test_manual_configuration_is_no_longer_an_api_option(server, extra):
    code, result = request(server, "/api/comparison/run", {"episodeId": "native-406-1", **extra})
    assert code == 400 and "fixed" in result["error"]
    assert server.console_state.view == {"existing_live_episode": True}
