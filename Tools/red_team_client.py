"""External runtime client for RedTeamAgentBridge; standard library only.

Place a bridge actor, assign an AgentPlacement RedTeamManager, enable Start On Begin Play,
then run this script during PIE or in a packaged development game. No editor Python required.
"""
import argparse
import json
import math
import socket


class RedTeamClient:
    def __init__(self, port=8765, timeout=120):
        self.socket = socket.create_connection(("127.0.0.1", port), timeout)
        self.stream = self.socket.makefile("rb")
        self.request_id = 0
        self.context = None
        self.completed_steps = 0
        self.placement_request_id = 0
        self.last_wire = None

    def request(self, op, **fields):
        payload = {"id": self.request_id, "op": op, **fields}
        self.request_id += 1
        self.last_wire = (json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n").encode()
        self.socket.sendall(self.last_wire)
        return self._response()

    def _response(self):
        line = self.stream.readline(4 * 1024 * 1024)
        if not line or not line.endswith(b"\n"):
            raise ConnectionError("Bridge disconnected or response exceeded 4 MiB; episode must be reset")
        result = json.loads(line)
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Bridge rejected request"))
        return result

    def reset(self, seed=12345):
        self.context = self.request("reset", seed=seed)["context"]
        self.completed_steps = 0
        self.placement_request_id = 0
        return self.context

    def place(self, centers):
        """centers: sequence of (x,y,z) world centimeters, one per advertised group ID."""
        if self.context is None:
            raise RuntimeError("Call reset first")
        action = {"schemaVersion": 1, "runId": self.context["runId"],
                  "revision": self.context["revision"], "requestId": self.placement_request_id,
                  "centers": [{"groupId": i, "centerWorldCm": {"x": x, "y": y, "z": z}}
                              for i, (x, y, z) in enumerate(centers)]}
        self.placement_request_id += 1
        return self.request("place", action=action)["result"]

    def step(self, steps=1):
        observation = self.request("step", runId=self.context["runId"],
                                   expectedStep=self.completed_steps, steps=steps)["observation"]
        self.completed_steps = observation["completedSteps"]
        return observation

    def close(self):
        self.stream.close()
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def scripted_centers(context):
    """A smoke-test policy, not a trained agent. Replace this function with inference."""
    objective = context["objectiveWorldCm"]
    radius = (context["minRadiusCm"] + context["maxRadiusCm"]) / 2
    count = context["groupCount"]
    return [(objective["x"] + radius * math.cos(2 * math.pi * i / count),
             objective["y"] + radius * math.sin(2 * math.pi * i / count),
             objective["z"] + context["heightOffsetCm"]) for i in range(count)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    with RedTeamClient(args.port) as client:
        context = client.reset()
        print("Placement:", client.place(scripted_centers(context)))
        for _ in range(args.steps):
            observation = client.step()
            if observation["bTerminated"] or observation["bTruncated"]:
                break
        print(json.dumps(observation, indent=2))
