"""Synthetic long-approach benchmark and observer-only saturation diagnostics.

No real-world route model. Private geometry stays outside the Blue actor input.
The route distribution is fixed before training; held-out seeds never tune it.
"""
import hashlib
import json
import math

import numpy as np


SCENARIO = {
    "id": "synthetic-warning-approach-v2",
    "spawn_annulus_m": [260., 300.],
    "center_radius_m": 280.,
    "spawn_height_above_objective_m": 120.,
    "sector_angles_rad": [0., math.pi / 2, math.pi],
    "sector_probabilities": [.5, .3, .2],
    "sector_jitter_rad": .08,
    "group_fan_radians": .4,
    "minimum_start_clearance_m": 50.,
    "endpoint": "20m XY zone entry; not impact",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def approach_centers(context, seed):
    if not (context["minRadiusCm"] == 26000 and context["maxRadiusCm"] == 30000
            and context["heightOffsetCm"] == 12000):
        raise ValueError("Start native scene with -IstanaWarningApproachV2")
    rng = np.random.default_rng(seed)
    sector = rng.choice(SCENARIO["sector_angles_rad"], p=SCENARIO["sector_probabilities"])
    phase = sector + rng.uniform(-.08, .08)
    count = context["groupCount"]
    origin = np.array([context["objectiveWorldCm"][k] for k in ("x", "y", "z")])
    centers = []
    for offset in np.linspace(-.2, .2, count):
        angle = phase + offset
        centers.append((origin + [28000 * math.cos(angle), 28000 * math.sin(angle),
                                  context["heightOffsetCm"]]).tolist())
    minimum = 2 * context["spreadRadiusCm"] + context["movement"]["spacingCm"]
    if any(np.linalg.norm(np.array(a)-b) < minimum for i, a in enumerate(centers) for b in centers[i+1:]):
        raise ValueError("Scenario fan cannot accommodate current swarm spread/count")
    return centers


def scenario_contract(red, blue):
    """Freeze all advertised physical settings, excluding only episode state."""
    red = {k: v for k, v in red.items() if k not in ("runId", "revision", "memberSeed")}
    snapshot = {k: v for k, v in blue["publicSnapshot"].items() if k not in
                ("episode_id", "timestamp", "placements", "budget_remaining", "tracks",
                 "done", "fresh_track_fraction")}
    return {"red": red, "blue": snapshot, "catalogue": blue["catalogue"],
            "surfaces": blue["siteSurfacesWorldCm"], "temporal": blue["temporalConfig"],
            "placement_rule": blue["placementRule"], "sensor_model": blue["sensorModel"]}


def start_diagnostics(context, placements, drones):
    catalogue = {p["id"]: p for p in context["catalogue"]}
    surfaces = context["siteSurfacesWorldCm"]
    blocked = context["publicSnapshot"]["blocked_sites"]
    available = context["publicSnapshot"]["available_sensor_ids"]
    def sensor(p):
        profile = catalogue[p["profileId"]]
        base = surfaces[p["siteId"]]
        if base is None:
            raise ValueError("Unsupported sensor in diagnostic")
        return np.array(base) / 100 + [0, 0, profile["height_m"]], max(profile["ranges"].values())
    deployed = [sensor(p) for p in placements]
    # Conservative universal check: every supported site and available profile,
    # even combinations the budget would not allow. Range is 3D, as in native.
    possible = [sensor({"profileId": name, "siteId": i}) for i, base in enumerate(surfaces)
                if base is not None and i not in blocked for name in available]
    rows = []
    for drone in drones:
        pos = np.array([drone["positionCm"][k] for k in ("x", "y", "z")]) / 100
        distances = [float(np.linalg.norm(pos-point)) for point, _ in deployed]
        clearance = min(float(np.linalg.norm(pos-point))-radius for point, radius in possible)
        rows.append({"drone_id": drone["droneId"],
                     "nearest_sensor_distance_m": min(distances) if distances else None,
                     "within_deployed_max_range": any(d <= radius for d, (_, radius) in zip(distances, deployed)),
                     "clearance_from_any_supported_sensor_range_m": clearance})
    return {"drones": rows, "spawned_in_range_count": sum(r["within_deployed_max_range"] for r in rows),
            "minimum_universal_clearance_m": min(r["clearance_from_any_supported_sensor_range_m"] for r in rows),
            "empty_layout": not placements}


def saturation_metrics(evidence, look_interval):
    detected = [r["firstDetectionSeconds"] for r in evidence if r["firstDetectionSeconds"] is not None]
    count = sum(t <= look_interval + 1e-8 for t in detected)
    return {"first_look_detected_count": count, "first_look_detected_fraction": count / len(evidence),
            "team_first_look_saturated": bool(count), "sensing_interval_s": look_interval}
