"""Versioned, backend-neutral inputs for adaptive sensor placement.

``build_observation(public_state, catalogue)`` is the *only* feature builder
used by the simulator and ``LiveObservationAdapter``.  A provider supplies
local east/north/up metres, UTC/monotonic seconds on one consistent clock,
current tracks and explicit forecast priors.  Neither entry point accepts
future trajectories, hidden emitter schedules or simulator ground truth.
The live adapter only returns recommendations; it has no network, actuator
or sensor-control implementation.  Capability numbers are synthetic, not a
calibration for any real sensor.
"""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping, Sequence

import numpy as np


INPUT_SCHEMA = "triad.sensor_input.v1"
FEATURE_SCHEMA = "triad.adaptive_placement_features.v1"
MODALITIES = ("rf", "radar", "eo", "thermal")
DEFAULT_CATALOGUE = [
    {"id": "rf", "label": "Passive RF", "cost": .8,
     "ranges": {"rf": 130.}, "strengths": {"rf": .88}, "height_m": 4.},
    {"id": "radar", "label": "Search radar", "cost": 1.2,
     "ranges": {"radar": 100.}, "strengths": {"radar": .86}, "height_m": 4.},
    {"id": "eo", "label": "Electro-optical", "cost": .7,
     "ranges": {"eo": 100.}, "strengths": {"eo": .94}, "height_m": 4.},
    {"id": "thermal", "label": "Thermal", "cost": 1.,
     "ranges": {"thermal": 115.}, "strengths": {"thermal": .86}, "height_m": 4.},
    {"id": "fused", "label": "Radar + thermal", "cost": 2.,
     "ranges": {"radar": 100., "thermal": 125.},
     "strengths": {"radar": .86, "thermal": .86}, "height_m": 4.},
]

FEATURE_NAMES = (
    "bias", "stop", "east", "north", "radius", "cost",
    *[f"modality_{m}" for m in MODALITIES],
    *[f"range_{m}" for m in MODALITIES],
    *[f"effectiveness_{m}" for m in MODALITIES],
    "budget_remaining", "budget_total", "remaining_sites", "placed_count",
    "budget_fraction", "visibility", "rain", "illumination", "humidity", "rf_noise",
    "threat_altitude", "threat_speed", "emitter_probability", "swarm_size",
    "angular_uncertainty", "approach_alignment", "candidate_coverage",
    "marginal_coverage", "overlap", "candidate_early_coverage", "marginal_early_coverage",
    "existing_coverage", "existing_early_coverage", "nearest_placement_distance",
    "nearest_track_distance", "track_confidence", "fresh_track_fraction",
    *[f"placed_{m}" for m in MODALITIES],
    "available_sensor_fraction",
    *[f"approach_sector_{i}" for i in range(8)],
    *[f"covered_sector_{i}" for i in range(8)],
)


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not np.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} must be finite in [{low}, {high}]")
    return result


def _vector(value: Any, size: int, name: str, bound: float) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must have {size} values")
    return [_number(v, name, -bound, bound) for v in value]


def validate_catalogue(catalogue: Sequence[Mapping[str, Any]] | None = None) -> list[dict]:
    """Validate 1..32 interchangeable capability rows, independent of policy size."""
    rows = deepcopy(DEFAULT_CATALOGUE if catalogue is None else list(catalogue))
    if not 1 <= len(rows) <= 32:
        raise ValueError("Catalogue must contain 1..32 sensors")
    result, seen = [], set()
    for row in rows:
        sensor_id = str(row["id"])
        if not sensor_id or len(sensor_id) > 128 or sensor_id in seen:
            raise ValueError("Sensor IDs must be unique nonempty strings")
        seen.add(sensor_id)
        ranges = {m: _number(row.get("ranges", {}).get(m, 0), f"{m} range", 0, 2000)
                  for m in MODALITIES}
        if not any(ranges.values()):
            raise ValueError("Every sensor requires a positive modality range")
        result.append({"id": sensor_id, "label": str(row.get("label", sensor_id)),
                       "cost": _number(row["cost"], "cost", .001, 100), "ranges": ranges,
                       "strengths": {m: _number(row.get("strengths", {}).get(m, .85 if ranges[m] else 0),
                                                 f"{m} strength", 0, 1) for m in MODALITIES},
                       "height_m": _number(row.get("height_m", 4), "sensor height", 0, 200)})
    return result


def validate_public_state(state: Mapping[str, Any], catalogue: Sequence[Mapping[str, Any]],
                          *, now: float | None = None) -> dict:
    """Copy and validate input; stale/missing tracks degrade to explicit priors.

    Unknown top-level fields are rejected to catch accidental ground-truth
    exports. Invalid tracks raise rather than silently fabricating observations.
    Optional tracks default to empty; stale tracks are removed, with their
    fraction retained in the features.  Forecasts remain explicitly *priors*.
    """
    allowed = {"schema", "timestamp", "source", "episode_id", "max_track_age", "sites", "placements",
               "budget_total", "budget_remaining", "max_sites", "min_separation",
               "deployment_min_radius", "deployment_max_radius", "weather", "forecast", "tracks",
               "available_sensor_ids", "blocked_sites", "fresh_track_fraction", "done"}
    if not isinstance(state, Mapping) or state.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"Provider requires schema {INPUT_SCHEMA}")
    extra = set(state) - allowed
    if extra:
        raise ValueError(f"Unknown public input fields: {sorted(extra)}")
    result = deepcopy(dict(state))
    # JSON validation prevents NaN and objects leaking into reports/providers.
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Public inputs must be finite JSON values") from error
    timestamp = _number(result.get("timestamp", 0), "timestamp", 0, 1e15)
    clock = timestamp if now is None else _number(now, "now", timestamp, 1e15)
    max_age = _number(result.get("max_track_age", 10), "max_track_age", .001, 3600)
    result.update(timestamp=timestamp, max_track_age=max_age)
    result["budget_total"] = _number(result["budget_total"], "budget_total", .001, 100)
    result["budget_remaining"] = _number(result["budget_remaining"], "budget_remaining", 0, result["budget_total"])
    max_sites = _number(result["max_sites"], "max_sites", 1, 32)
    if int(max_sites) != max_sites:
        raise ValueError("max_sites must be an integer")
    result["max_sites"] = int(max_sites)
    result["min_separation"] = _number(result.get("min_separation", 20), "min_separation", 0, 1000)
    result["deployment_min_radius"] = _number(result.get("deployment_min_radius", 30), "deployment_min_radius", 0, 1000)
    result["deployment_max_radius"] = _number(result.get("deployment_max_radius", 150), "deployment_max_radius", result["deployment_min_radius"], 2000)
    sites = result.get("sites")
    if not isinstance(sites, list) or not 1 <= len(sites) <= 512:
        raise ValueError("sites must contain 1..512 east/north coordinate pairs")
    result["sites"] = [_vector(s, 2, "site", 10000) for s in sites]
    ids = {sensor["id"]: index for index, sensor in enumerate(catalogue)}
    available = result.get("available_sensor_ids", list(ids))
    if not isinstance(available, list) or any(sensor not in ids for sensor in available):
        raise ValueError("available_sensor_ids must refer to the provided catalogue")
    result["available_sensor_ids"] = list(dict.fromkeys(available))
    blocked = result.get("blocked_sites", [])
    if not isinstance(blocked, list) or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= len(sites) for i in blocked):
        raise ValueError("blocked_sites must contain valid integer site indices")
    result["blocked_sites"] = blocked
    placements = []
    for p in result.get("placements", []):
        if p.get("sensor_id") not in ids:
            raise ValueError("Placement sensor_id is absent from catalogue")
        idx = ids[p["sensor_id"]]
        placements.append({"sensor_id": p["sensor_id"], "sensor_index": idx,
                           "position": _vector(p["position"], 2, "placement", 10000),
                           "cost": catalogue[idx]["cost"]})
    if len(placements) > result["max_sites"]:
        raise ValueError("placements exceeds max_sites")
    if sum(p["cost"] for p in placements) + result["budget_remaining"] > result["budget_total"] + 1e-8:
        raise ValueError("Placed sensor costs plus remaining budget exceed total budget")
    result["placements"] = placements
    weather = result.get("weather", {})
    result["weather"] = {key: _number(weather.get(key, default), key, 0, 1)
                         for key, default in {"visibility": 1., "rain": 0., "illumination": 1.,
                                              "humidity": .4, "rf_noise": .1}.items()}
    forecast = result.get("forecast", {})
    weights = forecast.get("approach_weights", [.125] * 8)
    if len(weights) != 8:
        raise ValueError("forecast.approach_weights requires eight sector probabilities")
    weights = np.asarray([_number(w, "approach weight", 0, 1) for w in weights])
    if weights.sum() <= 0:
        raise ValueError("approach_weights must have positive mass")
    result["forecast"] = {
        "approach_weights": (weights / weights.sum()).tolist(),
        "altitude": _number(forecast.get("altitude", 45), "forecast altitude", 0, 1000),
        "speed": _number(forecast.get("speed", 12), "forecast speed", .1, 200),
        "emitter_probability": _number(forecast.get("emitter_probability", .5), "emitter_probability", 0, 1),
        "swarm_size": _number(forecast.get("swarm_size", 1), "forecast swarm size", 1, 64),
        "angular_uncertainty": _number(forecast.get("angular_uncertainty", .4), "angular_uncertainty", 0, np.pi),
    }
    tracks = result.get("tracks", [])
    if not isinstance(tracks, list) or len(tracks) > 256:
        raise ValueError("tracks must contain at most 256 tracks")
    fresh = []
    for track in tracks:
        record = {"id": str(track["id"]), "position": _vector(track["position"], 3, "track position", 100000),
                  "velocity": _vector(track.get("velocity", [0, 0, 0]), 3, "track velocity", 500),
                  "confidence": _number(track.get("confidence", .5), "track confidence", 0, 1),
                  "emitter_probability": _number(track.get("emitter_probability", result["forecast"]["emitter_probability"]),
                                                  "track emitter_probability", 0, 1),
                  "timestamp": _number(track.get("timestamp", timestamp), "track timestamp", 0, clock),
                  "confirmed": bool(track.get("confirmed", False))}
        if clock - record["timestamp"] <= max_age:
            fresh.append(record)
    result["tracks"] = fresh
    fresh_fraction = (len(fresh) / len(tracks)) if tracks else 0.
    if "fresh_track_fraction" in result:
        fresh_fraction *= _number(result["fresh_track_fraction"], "fresh_track_fraction", 0, 1)
    result["fresh_track_fraction"] = fresh_fraction
    result["source"] = str(result.get("source", "unspecified"))
    result["done"] = bool(result.get("done", False))
    return result


def modality_effectiveness(weather: Mapping[str, float], emitter_probability: float | np.ndarray) -> np.ndarray:
    """Synthetic conditional per-look reliability, before slant-range attenuation."""
    emitter = np.asarray(emitter_probability)
    return np.stack(np.broadcast_arrays(
        emitter * (1. - .65 * weather["rf_noise"]),
        1. - .50 * weather["rain"],
        weather["visibility"] ** 1.65 * (.08 + .92 * weather["illumination"]) * (1. - .45 * weather["rain"]),
        (1. - .4 * weather["rain"]) * (1. - .3 * weather["humidity"]) * (.65 + .35 * weather["visibility"])), axis=-1)


def sensing_probabilities(positions: np.ndarray, sensors: Sequence[Mapping[str, Any]],
                          points: np.ndarray, weather: Mapping[str, float],
                          emitter_probability: float | np.ndarray) -> np.ndarray:
    """[sensor, point, modality] probabilities; no random draws or hidden inputs.

    A provider can replace the capability calibration while keeping the same
    feature schema. Range is hard slant range in metres, reduced by weather
    for optical/thermal modalities.  The within-range probability rolls off
    towards the edge; absent modalities have probability exactly zero.
    """
    if len(sensors) == 0:
        return np.zeros((0, len(points), 4), dtype=np.float64)
    ranges = np.asarray([[s["ranges"].get(m, 0) for m in MODALITIES] for s in sensors])
    strengths = np.asarray([[s["strengths"].get(m, 0) for m in MODALITIES] for s in sensors])
    xyz = np.column_stack((np.asarray(positions), [s.get("height_m", 4) for s in sensors]))
    distance = np.linalg.norm(xyz[:, None, :] - np.asarray(points)[None, :, :], axis=2)
    weather_range = np.array([1., 1. - .1 * weather["rain"],
                              .5 + .5 * weather["visibility"], 1. - .12 * weather["rain"]])
    effective_ranges = ranges * weather_range
    ratio = distance[:, :, None] / np.maximum(effective_ranges[:, None, :], 1e-9)
    radial = np.where(ratio <= 1., np.maximum(0., 1. - .7 * ratio ** 2), 0.)
    return radial * strengths[:, None, :] * modality_effectiveness(weather, emitter_probability)[None, ...]


def forecast_points(state: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quadrature over public bearing priors + noisy current track bearings.

    Forecasts deliberately do not know the true future path, emitter phase,
    target altitude oscillations or exact drone state.  Samples represent
    plausible ingress corridors, not ground-truth trajectory samples.
    """
    forecast = state["forecast"]
    sector_angles = np.arange(8) * (2 * np.pi / 8)
    bearings, masses = [], []
    uncertainty = max(.10, forecast["angular_uncertainty"])
    for angle, weight in zip(sector_angles, forecast["approach_weights"]):
        for offset in (-.5, 0., .5):
            bearings.append(angle + offset * min(uncertainty, .8))
            masses.append(weight / 3.)
    tracks = state["tracks"]
    if tracks:
        masses = [mass * .45 for mass in masses]
        confidence = np.array([max(.05, t["confidence"]) for t in tracks])
        for track, weight in zip(tracks, confidence / confidence.sum() * .55):
            angle = np.arctan2(track["position"][1], track["position"][0])
            for offset in (-.5, 0., .5):
                bearings.append(angle + offset * uncertainty)
                masses.append(weight / 3.)
    bearings = np.asarray(bearings)
    radii = np.array([55., 105., 155., 205.])
    angles = np.repeat(bearings, 4)
    radius = np.tile(radii, len(bearings))
    points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles),
                              np.full(len(radius), forecast["altitude"])))
    weights = np.repeat(np.asarray(masses), 4) * np.tile([.12, .25, .34, .29], len(bearings))
    early_weights = weights * (radius / radii[-1])
    early_weights /= early_weights.sum()
    sectors = np.floor((angles + np.pi / 8) % (2 * np.pi) / (np.pi / 4)).astype(int)
    return points, weights / weights.sum(), early_weights, sectors


def coverage_statistics(state: Mapping[str, Any], catalogue: Sequence[Mapping[str, Any]]) -> dict:
    """Coverage of forecast corridors by existing placements, not hidden tracks."""
    points, weights, early_weights, sectors = forecast_points(state)
    placements = state["placements"]
    probabilities = sensing_probabilities(np.asarray([p["position"] for p in placements]),
                                          [catalogue[p["sensor_index"]] for p in placements], points,
                                          state["weather"], state["forecast"]["emitter_probability"])
    covered = 1. - np.prod(1. - probabilities, axis=(0, 2))
    sector_coverage = [float(np.average(covered[sectors == s], weights=weights[sectors == s]))
                       if weights[sectors == s].sum() > 1e-12 else 0. for s in range(8)]
    return {"coverage": float(covered @ weights), "early_coverage": float(covered @ early_weights),
            "sector_coverage": sector_coverage, "points": points, "weights": weights,
            "early_weights": early_weights, "covered": covered}


def build_observation(state: Mapping[str, Any], catalogue: Sequence[Mapping[str, Any]] | None = None,
                      *, now: float | None = None) -> dict:
    """Build variable joint type/site options and fixed-width public features.

    Sensor-major ordering, then site order; last row is always STOP. An
    operational provider may supply arbitrary approved site coordinates and
    unavailable/blocked options. Masking enforces budget, separation, annulus
    and remaining site count. It never authorizes physical deployment.
    """
    catalogue = validate_catalogue(catalogue)
    state = validate_public_state(state, catalogue, now=now)
    sites = np.asarray(state["sites"], dtype=np.float64)
    n_sites, n_sensors = len(sites), len(catalogue)
    positions = np.tile(sites, (n_sensors, 1))
    sensors = [s for s in catalogue for _ in sites]
    n = len(sensors)
    options = [{"sensor_id": s["id"], "sensor_index": si, "site_index": j, "position": pos.tolist(), "stop": False}
               for si, s in enumerate(catalogue) for j, pos in enumerate(sites)]
    options.append({"sensor_id": None, "sensor_index": -1, "site_index": -1, "position": [0., 0.], "stop": True})
    stats = coverage_statistics(state, catalogue)
    probabilities = sensing_probabilities(positions, sensors, stats["points"], state["weather"],
                                          state["forecast"]["emitter_probability"])
    covered = 1. - np.prod(1. - probabilities, axis=2)
    gain = covered * (1. - stats["covered"])
    placement_positions = np.asarray([p["position"] for p in state["placements"]])
    nearest_placement = (np.linalg.norm(positions[:, None, :] - placement_positions[None, :, :], axis=2).min(axis=1)
                         if len(placement_positions) else np.full(n, 300.))
    track_positions = np.asarray([t["position"][:2] for t in state["tracks"]])
    nearest_track = (np.linalg.norm(positions[:, None, :] - track_positions[None, :, :], axis=2).min(axis=1)
                     if len(track_positions) else np.full(n, 500.))
    forecast = state["forecast"]
    cost = np.array([s["cost"] for s in sensors])
    radius = np.linalg.norm(positions, axis=1)
    modality = np.array([[s["ranges"][m] > 0 for m in MODALITIES] for s in sensors])
    ranges = np.array([[s["ranges"][m] for m in MODALITIES] for s in sensors])
    effectiveness = modality * modality_effectiveness(state["weather"], forecast["emitter_probability"])
    weights = np.array(forecast["approach_weights"])
    alignment = np.cos(np.arctan2(positions[:, 1], positions[:, 0])[:, None] - np.arange(8)[None, :] * np.pi / 4) @ weights
    placed_modalities = [sum(catalogue[p["sensor_index"]]["ranges"][m] > 0 for p in state["placements"]) / 4.
                         for m in MODALITIES]
    track_confidence = float(np.mean([t["confidence"] for t in state["tracks"]])) if state["tracks"] else 0.
    # Named assignments keep the external feature layout inspectable and stable.
    fields: dict[str, Any] = {"bias": 1., "stop": 0., "east": positions[:, 0] / 150.,
                             "north": positions[:, 1] / 150., "radius": radius / 150., "cost": cost / 4.,
                             "budget_remaining": state["budget_remaining"] / 4., "budget_total": state["budget_total"] / 4.,
                             "remaining_sites": (state["max_sites"] - len(state["placements"])) / 4.,
                             "placed_count": len(state["placements"]) / 4.,
                             "budget_fraction": state["budget_remaining"] / state["budget_total"],
                             **state["weather"], "threat_altitude": forecast["altitude"] / 160.,
                             "threat_speed": forecast["speed"] / 30., "emitter_probability": forecast["emitter_probability"],
                             "swarm_size": forecast["swarm_size"] / 8., "angular_uncertainty": forecast["angular_uncertainty"] / np.pi,
                             "approach_alignment": alignment, "candidate_coverage": covered @ stats["weights"],
                             "marginal_coverage": gain @ stats["weights"],
                             "overlap": (covered * stats["covered"]) @ stats["weights"],
                             "candidate_early_coverage": covered @ stats["early_weights"],
                             "marginal_early_coverage": gain @ stats["early_weights"],
                             "existing_coverage": stats["coverage"], "existing_early_coverage": stats["early_coverage"],
                             "nearest_placement_distance": nearest_placement / 300., "nearest_track_distance": nearest_track / 500.,
                             "track_confidence": track_confidence, "fresh_track_fraction": state["fresh_track_fraction"],
                             "available_sensor_fraction": len(state["available_sensor_ids"]) / 8.}
    for i, m in enumerate(MODALITIES):
        fields.update({f"modality_{m}": modality[:, i], f"range_{m}": ranges[:, i] / 200.,
                       f"effectiveness_{m}": effectiveness[:, i], f"placed_{m}": placed_modalities[i]})
    for i in range(8):
        fields[f"approach_sector_{i}"] = weights[i]
        fields[f"covered_sector_{i}"] = stats["sector_coverage"][i]
    features = np.zeros((n + 1, len(FEATURE_NAMES)), dtype=np.float32)
    for i, name in enumerate(FEATURE_NAMES):
        features[:n, i] = fields[name]
        if np.ndim(fields[name]) == 0:
            features[n, i] = fields[name]
    features[n, FEATURE_NAMES.index("stop")] = 1.
    mask = np.array([s["id"] in state["available_sensor_ids"] for s in sensors])
    mask &= cost <= state["budget_remaining"] + 1e-9
    mask &= (radius >= state["deployment_min_radius"] - 1e-9) & (radius <= state["deployment_max_radius"] + 1e-9)
    mask &= nearest_placement >= state["min_separation"] - 1e-9
    # Always prohibit reusing exactly the same site, even with zero separation.
    if len(placement_positions):
        mask &= nearest_placement > 1e-7
    mask &= ~np.tile(np.isin(np.arange(n_sites), state["blocked_sites"]), n_sensors)
    if len(state["placements"]) >= state["max_sites"] or state["done"]:
        mask[:] = False
    return {"schema": FEATURE_SCHEMA, "feature_schema": FEATURE_SCHEMA, "option_features": features, "action_mask": np.append(mask, True),
            "feature_names": FEATURE_NAMES, "options": options, "state": state,
            "catalogue": catalogue}


def _apply_legal_option(state: Mapping[str, Any], option: Mapping[str, Any], catalogue: Sequence[Mapping[str, Any]]) -> dict:
    """Internal transition for an option already checked against current mask."""
    result = deepcopy(dict(state))
    if option["stop"]:
        result["done"] = True
        return result
    sensor = catalogue[option["sensor_index"]]
    result["placements"].append({"sensor_id": sensor["id"], "sensor_index": option["sensor_index"],
                                  "position": list(option["position"]), "cost": sensor["cost"]})
    result["budget_remaining"] = max(0., result["budget_remaining"] - sensor["cost"])
    return result


def apply_placement(state: Mapping[str, Any], action: int,
                    catalogue: Sequence[Mapping[str, Any]] | None = None,
                    *, now: float | None = None) -> dict:
    """Apply one legal recommendation to a *local planning snapshot* only.

    Rebuilds the current mask before resolving ``action``; returns a new public
    state with placement and budget updated, or ``done=True`` for STOP. Use
    build_observation on the returned state for the next sequential decision.
    It does not mark an actual sensor deployed or communicate with hardware.
    A stopped state cannot be advanced until replaced by a new provider state.
    """
    observation = build_observation(state, catalogue, now=now)
    if observation["state"]["done"]:
        raise ValueError("Planning snapshot is done")
    option = LiveObservationAdapter.recommendation(observation, action)
    return _apply_legal_option(observation["state"], option, observation["catalogue"])


class LiveObservationAdapter:
    """Validate a versioned sensor/provider snapshot and produce policy input.

    ``observe(payload, now=...)`` accepts a dict or JSON string. Stale tracks
    are discarded against the caller's clock. No tracks means forecast-only
    recommendations; the output exposes track confidence/freshness. A policy
    trained against the simulation consumes exactly this observation schema.
    ``recommendation`` resolves a legal selected row to a JSON recommendation,
    never sends commands. Actual deployments require separate authorization,
    sensor calibration, safety constraints and an audited integration layer.
    """
    def __init__(self, catalogue: Sequence[Mapping[str, Any]] | None = None):
        self.catalogue = validate_catalogue(catalogue)

    def observe(self, payload: str | Mapping[str, Any], *, now: float | None = None) -> dict:
        if isinstance(payload, str):
            payload = json.loads(payload)
        return build_observation(payload, self.catalogue, now=now)

    @staticmethod
    def recommendation(observation: Mapping[str, Any], action: int) -> dict:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= action < len(observation["options"]):
            raise ValueError("Recommendation action must index an offered option")
        if not observation["action_mask"][action]:
            raise ValueError("Cannot recommend an invalid placement")
        return deepcopy(observation["options"][action])
