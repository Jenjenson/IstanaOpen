"""Invoked by native Automation with a live loopback bridge in a collision-enabled UE world."""
from red_team_client import RedTeamClient, scripted_centers
import json
import sys

with RedTeamClient(int(sys.argv[1])) as client:
    context = client.reset(41)
    assert context["groupCount"] == 2
    assert context["membersPerGroup"] == 3
    centers = scripted_centers(context)
    try:
        client.request("place", action={"schemaVersion": 1.5, "runId": context["runId"],
                                       "revision": context["revision"], "requestId": 0, "centers": []})
        raise AssertionError("fractional schema version was accepted")
    except RuntimeError:
        pass
    result = client.place(centers)
    assert result["bAccepted"] and len(result["initialStates"]) == 6
    for group, center in zip(result["acceptedGroups"], centers):
        assert list(group["spawnOriginCm"].values()) == list(center)
    a = client.step(3)
    assert a["completedSteps"] == 3 and not a["bHasReward"]
    # Exact wire retry must return the prior step result without advancing again.
    client.socket.sendall(client.last_wire)
    assert client._response()["observation"]["completedSteps"] == 3
    assert client.request("observe")["observation"]["completedSteps"] == 3
    try:
        client.request("step", runId=context["runId"], expectedStep=0, steps=1)
        raise AssertionError("stale step was accepted")
    except RuntimeError:
        pass
    client.request("cancel")
    assert client.request("observe")["observation"]["bTruncated"]

# A new transport session can start IDs at zero, but needs a new episode.
import time
with RedTeamClient(int(sys.argv[1])) as client:
    next_context = client.reset(41)
    assert next_context["runId"] != context["runId"]
    time.sleep(1.5)
    try:
        client.request("observe")
        raise AssertionError("idle timeout did not disconnect")
    except (ConnectionError, OSError):
        pass
