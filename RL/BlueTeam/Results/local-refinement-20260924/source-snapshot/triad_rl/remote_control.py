from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
import urllib.parse
from dataclasses import dataclass
from typing import Any


def normalize_enum(value, names):
    """UE Remote Control emits display names ('Blue Placement'), not C++ names."""
    if type(value) is int and 0 <= value < len(names):
        return names[value]
    if isinstance(value, str):
        name = value.rsplit('::', 1)[-1].replace(' ', '')
        if name in names:
            return name
    raise RuntimeError(f'Unknown native RL enum: {value!r}')


@dataclass(frozen=True)
class TRIADRemoteControlClient:
    """Narrow client for a locally running, explicitly enabled Unreal RL manager."""

    object_path: str
    endpoint: str = "http://127.0.0.1:30010"
    timeout_seconds: float = 10.0

    def _call(self, function_name: str, parameters: dict[str, Any]) -> dict[str, Any]:
        address = urllib.parse.urlsplit(self.endpoint)
        if (address.scheme != "http" or address.hostname != "127.0.0.1" or address.port is None
                or address.username or address.password or address.query or address.fragment
                or address.path not in ("", "/")):
            raise ValueError("TRIAD RL Remote Control must use the IPv4 loopback endpoint")
        body = json.dumps(
            {
                "objectPath": self.object_path,
                "functionName": function_name,
                "generateTransaction": False,
                "parameters": parameters,
            },
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.endpoint}/remote/object/call",
            data=body,
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                if response.status != 200:
                    raise RuntimeError(f"Unreal Remote Control returned HTTP {response.status}")
                payload = response.read(1_048_577)
        except urllib.error.URLError as error:
            raise RuntimeError(f"Could not reach the local TRIAD RL manager: {error}") from error
        if len(payload) > 1_048_576:
            raise RuntimeError("Unreal RL response exceeded the 1 MiB client limit")
        decoded = json.loads(payload.decode("utf-8"))
        if not decoded.get("ReturnValue", False):
            raise RuntimeError(decoded.get("OutError") or f"{function_name} failed")
        state = decoded['OutResult']
        state['Phase'] = normalize_enum(state['Phase'], (
            'Inactive', 'BluePlacement', 'RedDeployment', 'RedMovement', 'Terminal'))
        state['TerminationReason'] = normalize_enum(state['TerminationReason'], (
            'None', 'ProtectedZoneReached', 'AllTargetsConfirmed', 'HorizonReached',
            'ConstraintViolation', 'InvalidAction', 'SustainedTrackDefence'))
        return state

    def reset(self, seed: int) -> dict[str, Any]:
        return self._call("ResetEpisode", {"Seed": int(seed)})

    def snapshot(self) -> dict[str, Any]:
        return self._call("GetEpisodeSnapshot", {})

    def evaluation_labels(self, blue: str, red: str) -> dict[str, Any]:
        return self._call("SetEvaluationLabels", {"BlueCheckpoint": blue, "RedCheckpoint": red})

    def blue_action(
        self,
        catalogue_index: int,
        normalized_position: list[float] | tuple[float, float] = (0.0, 0.0),
        stop: bool = False,
    ) -> dict[str, Any]:
        if isinstance(catalogue_index, bool) or not isinstance(catalogue_index, int) or catalogue_index < 0:
            raise ValueError("Blue placement requires a non-negative catalogue index")
        try:
            if len(normalized_position) != 2:
                raise ValueError
            position = tuple(float(value) for value in normalized_position)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "Blue placement position must be a finite normalized point in the unit disk"
            ) from error
        if (not all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in position)
                or sum(value * value for value in position) > 1.0 + 1e-9):
            raise ValueError("Blue placement position must be a finite normalized point in the unit disk")
        return self._call(
            "ApplyBlueAction",
            {
                "Action": {
                    "CatalogueIndex": catalogue_index,
                    "NormalizedPosition": {
                        "X": position[0],
                        "Y": position[1],
                    },
                    "bStopPlacement": bool(stop),
                }
            },
        )

    def red_deployment(self, normalized_deployment: list[float]) -> dict[str, Any]:
        if len(normalized_deployment) != 5 or any(
            not -1.0 <= float(value) <= 1.0 for value in normalized_deployment
        ):
            raise ValueError("Red deployment must contain five finite normalized values")
        names = (
            "NormalizedBearing",
            "NormalizedRadius",
            "NormalizedAltitude",
            "NormalizedSwarmSize",
            "NormalizedFormationSpacing",
        )
        return self._call(
            "ApplyRedDeploymentAction",
            {"Action": {name: float(value) for name, value in zip(names, normalized_deployment)}},
        )

    def red_actions(self, normalized_velocities: list[list[float]]) -> dict[str, Any]:
        if len(normalized_velocities) > 64:
            raise ValueError("The simulation-only target cap is 64")
        actions = []
        for vector in normalized_velocities:
            if len(vector) != 3 or any(not -1.0 <= float(value) <= 1.0 for value in vector):
                raise ValueError("Every Red action must contain three finite normalized values")
            actions.append(
                {
                    "NormalizedVelocityEnu": {
                        "X": float(vector[0]),
                        "Y": float(vector[1]),
                        "Z": float(vector[2]),
                    }
                }
            )
        return self._call("ApplyRedActions", {"Actions": actions})
