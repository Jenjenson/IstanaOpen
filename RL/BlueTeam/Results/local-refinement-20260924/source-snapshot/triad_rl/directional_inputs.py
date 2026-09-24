"""Versioned public inputs for directional sensor placement.

This module deliberately extends, rather than edits, the checksum-pinned
``adaptive_inputs`` contract used by historical experiments. Manufacturer
facts, calculated geometry and simulator assumptions remain separate in every
directional profile. Unreal line traces are authoritative for live LOS; this
public planner can only use an explicitly supplied LOS mask.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np

from . import adaptive_inputs as legacy


FEATURE_SCHEMA = "triad.directional_placement_features.v1"
FEATURE_NAMES = legacy.FEATURE_NAMES
MODALITIES = legacy.MODALITIES


def vertical_fov_from_horizontal(horizontal_fov_deg: float, width: int, height: int) -> float:
    """Calculate rectilinear VFOV from HFOV and detector aspect ratio."""
    return float(np.degrees(2 * np.arctan(np.tan(np.radians(horizontal_fov_deg) / 2) * height / width)))


BOSON_PLUS_640_18MM = {
    "id": "thermal",
    "label": "Teledyne FLIR Boson+ 640 18 mm",
    "cost": 1.,
    "ranges": {"thermal": 500.},
    "strengths": {"thermal": .98},
    "height_m": 4.,
    "directional": True,
    "manufacturer_specifications": {
        "manufacturer": "Teledyne FLIR",
        "model": "Boson+ 640, 24deg HFOV, 18 mm",
        "modality": "uncooled LWIR thermal",
        "source": "https://oem.flir.com/products/boson-plus/?model=22640A024",
        "resolution": [640, 512],
        "pixel_pitch_um": 12.,
        "horizontal_fov_deg": 24.,
        "ifov_mrad": .667,
        "frame_rate_hz": 60.,
        "selectable_frame_rate_hz": 30.,
        "industrial_nedt_mk_max": 20.,
    },
    "calculated_geometry": {
        "vertical_fov_deg": vertical_fov_from_horizontal(24., 640, 512),
        "method": "rectilinear HFOV and 640:512 detector aspect ratio",
    },
    "simulation_assumptions": {
        "detection_model": "pixels_on_target_v1",
        "max_evaluation_distance_m": 500.,
        "nominal_thermal_contrast_k": 8.,
        "contrast_noise_multiplier": 8.,
        "pixels_for_63_percent": 3.,
        "atmospheric_attenuation_distance_m": 900.,
        "rain_loss_at_maximum": .45,
        "humidity_loss_at_maximum": .35,
        "edge_falloff_exponent": 1.5,
        "requires_line_of_sight": True,
    },
    "yaw_bins_deg": [0., 45., 90., 135., 180., 225., 270., 315.],
    "pitch_bins_deg": [0., 10., 20.],
}


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


def _numbers(values: Any, name: str, low: float, high: float, maximum: int) -> list[float]:
    if not isinstance(values, (list, tuple)) or not values or len(values) > maximum:
        raise ValueError(f"{name} must contain 1..{maximum} values")
    result = [_number(value, name, low, high) for value in values]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} values must be unique")
    return result


def validate_catalogue(catalogue: Sequence[Mapping[str, Any]] | None = None) -> list[dict]:
    """Validate legacy profiles plus generic directional profile metadata."""
    raw = deepcopy([BOSON_PLUS_640_18MM] if catalogue is None else list(catalogue))
    base = legacy.validate_catalogue(raw)
    for source, sensor in zip(raw, base):
        directional = source.get("directional", False)
        if type(directional) is not bool:
            raise ValueError("directional must be boolean")
        sensor["directional"] = directional
        if not directional:
            continue
        hardware = source.get("manufacturer_specifications")
        calculated = source.get("calculated_geometry")
        assumptions = source.get("simulation_assumptions")
        if not all(isinstance(row, Mapping) for row in (hardware, calculated, assumptions)):
            raise ValueError("Directional profiles require manufacturer, calculated and simulation sections")
        resolution = hardware.get("resolution")
        if (not isinstance(resolution, (list, tuple)) or len(resolution) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or int(value) != value or not 1 <= value <= 16384 for value in resolution)):
            raise ValueError("resolution must contain two positive integer dimensions")
        width, height = map(int, resolution)
        manufacturer = {
            "manufacturer": str(hardware.get("manufacturer", "")),
            "model": str(hardware.get("model", "")),
            "modality": str(hardware.get("modality", "")),
            "source": str(hardware.get("source", "")),
            "resolution": [width, height],
            "pixel_pitch_um": _number(hardware.get("pixel_pitch_um"), "pixel pitch", .1, 100.),
            "horizontal_fov_deg": _number(hardware.get("horizontal_fov_deg"), "horizontal FOV", .1, 179.),
            "ifov_mrad": _number(hardware.get("ifov_mrad"), "IFOV", .001, 100.),
            "frame_rate_hz": _number(hardware.get("frame_rate_hz"), "frame rate", .1, 1000.),
            "selectable_frame_rate_hz": _number(hardware.get("selectable_frame_rate_hz"), "selectable frame rate", .1, 1000.),
            "industrial_nedt_mk_max": _number(hardware.get("industrial_nedt_mk_max"), "industrial NEdT", .1, 1000.),
        }
        if any(not manufacturer[key] for key in ("manufacturer", "model", "modality", "source")):
            raise ValueError("Directional manufacturer text fields must be nonempty")
        expected_vfov = vertical_fov_from_horizontal(manufacturer["horizontal_fov_deg"], width, height)
        vertical = _number(calculated.get("vertical_fov_deg"), "vertical FOV", .1, 179.)
        if not np.isclose(vertical, expected_vfov, atol=.01, rtol=0):
            raise ValueError("Calculated vertical FOV does not match detector geometry")
        simulation = {
            "detection_model": str(assumptions.get("detection_model", "")),
            "max_evaluation_distance_m": _number(assumptions.get("max_evaluation_distance_m"), "evaluation distance", 1., 2000.),
            "nominal_thermal_contrast_k": _number(assumptions.get("nominal_thermal_contrast_k"), "thermal contrast", .001, 1000.),
            "contrast_noise_multiplier": _number(assumptions.get("contrast_noise_multiplier"), "contrast noise multiplier", .001, 1000.),
            "pixels_for_63_percent": _number(assumptions.get("pixels_for_63_percent"), "pixels for 63 percent", .01, 1000.),
            "atmospheric_attenuation_distance_m": _number(assumptions.get("atmospheric_attenuation_distance_m"), "atmospheric attenuation distance", 1., 100000.),
            "rain_loss_at_maximum": _number(assumptions.get("rain_loss_at_maximum"), "rain loss", 0., 1.),
            "humidity_loss_at_maximum": _number(assumptions.get("humidity_loss_at_maximum"), "humidity loss", 0., 1.),
            "edge_falloff_exponent": _number(assumptions.get("edge_falloff_exponent"), "edge falloff exponent", .01, 20.),
            "requires_line_of_sight": assumptions.get("requires_line_of_sight"),
        }
        if simulation["detection_model"] != "pixels_on_target_v1" or type(simulation["requires_line_of_sight"]) is not bool:
            raise ValueError("Unsupported directional detection model or LOS setting")
        sensor.update(
            manufacturer_specifications=manufacturer,
            calculated_geometry={"vertical_fov_deg": vertical,
                                 "method": str(calculated.get("method", "rectilinear detector geometry"))},
            simulation_assumptions=simulation,
            yaw_bins_deg=_numbers(source.get("yaw_bins_deg"), "yaw bins", 0., 360., 72),
            pitch_bins_deg=_numbers(source.get("pitch_bins_deg"), "pitch bins", -89., 89., 19),
        )
    return base


def _legacy_state(state: Mapping[str, Any]) -> dict:
    result = deepcopy(dict(state))
    result["forecast"] = deepcopy(result.get("forecast", {}))
    result["forecast"].pop("target_size_m", None)
    for placement in result.get("placements", []):
        placement.pop("yaw_deg", None)
        placement.pop("pitch_deg", None)
    return result


def validate_public_state(state: Mapping[str, Any], catalogue: Sequence[Mapping[str, Any]],
                          *, now: float | None = None) -> dict:
    """Validate the frozen public state plus target size and placement angles."""
    raw = deepcopy(dict(state))
    base_catalogue = legacy.validate_catalogue(catalogue)
    result = legacy.validate_public_state(_legacy_state(raw), base_catalogue, now=now)
    target_size = _number(raw.get("forecast", {}).get("target_size_m", .4), "target size", .01, 100.)
    result["forecast"]["target_size_m"] = target_size
    ids = {sensor["id"]: index for index, sensor in enumerate(catalogue)}
    for index, (source, placement) in enumerate(zip(raw.get("placements", []), result["placements"])):
        sensor = catalogue[ids[placement["sensor_id"]]]
        yaw = _number(source.get("yaw_deg", 0.), "placement yaw", 0., 360.)
        pitch = _number(source.get("pitch_deg", 0.), "placement pitch", -89., 89.)
        if sensor.get("directional"):
            if yaw not in sensor["yaw_bins_deg"] or pitch not in sensor["pitch_bins_deg"]:
                raise ValueError("Placement orientation is outside the profile's discrete bins")
        elif yaw != 0 or pitch != 0:
            raise ValueError("Nondirectional placement orientation must be zero")
        result["placements"][index].update(yaw_deg=yaw, pitch_deg=pitch)
    return result


def to_legacy_inputs(state: Mapping[str, Any], catalogue: Sequence[Mapping[str, Any]]) -> tuple[dict, list[dict]]:
    """Return an explicit v1 view for frozen policies that never learned angles."""
    base_catalogue = legacy.validate_catalogue(catalogue)
    return legacy.validate_public_state(_legacy_state(state), base_catalogue), base_catalogue


def pixels_on_target(target_size_m: float | np.ndarray, distance_m: float | np.ndarray,
                     ifov_mrad: float) -> np.ndarray:
    """Exact angular diameter divided by IFOV, in radians per pixel."""
    size, distance = np.asarray(target_size_m, dtype=float), np.asarray(distance_m, dtype=float)
    if (not np.isfinite(size).all() or not np.isfinite(distance).all() or np.any(size <= 0)
            or np.any(distance <= 0) or not np.isfinite(ifov_mrad) or ifov_mrad <= 0):
        raise ValueError("Target size, distance and IFOV must be finite and positive")
    return 2 * np.arctan(size / (2 * distance)) / (ifov_mrad / 1000.)


def sensing_probabilities(positions: np.ndarray, sensors: Sequence[Mapping[str, Any]],
                          points: np.ndarray, weather: Mapping[str, float],
                          emitter_probability: float | np.ndarray, *, orientations: np.ndarray | None = None,
                          target_size_m: float | np.ndarray = .4,
                          line_of_sight: np.ndarray | None = None) -> np.ndarray:
    """Return per-look probabilities; Unreal supplies the authoritative LOS."""
    if len(sensors) == 0:
        return np.zeros((0, len(points), len(MODALITIES)), dtype=np.float64)
    positions = np.asarray(positions, dtype=float)
    points = np.asarray(points, dtype=float)
    if positions.shape != (len(sensors), 2) or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("positions and points require [sensor,2] and [point,3] shapes")
    # ``sensors`` contains one row per offered action and can legitimately be
    # much larger than the 32-profile catalogue validation limit.
    result = legacy.sensing_probabilities(positions, sensors, points, weather, emitter_probability)
    if orientations is None:
        orientations = np.zeros((len(sensors), 2), dtype=float)
    orientations = np.asarray(orientations, dtype=float)
    if orientations.shape != (len(sensors), 2) or not np.isfinite(orientations).all():
        raise ValueError("orientations must provide finite yaw/pitch degrees per sensor")
    if line_of_sight is None:
        line_of_sight = np.ones((len(sensors), len(points)), dtype=bool)
    line_of_sight = np.asarray(line_of_sight)
    if line_of_sight.shape != (len(sensors), len(points)) or line_of_sight.dtype.kind != "b":
        raise ValueError("line_of_sight must be a boolean [sensor, point] matrix")
    sizes = np.broadcast_to(np.asarray(target_size_m, dtype=float), (len(points),))
    if not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError("target_size_m must be finite and positive")
    thermal = MODALITIES.index("thermal")
    xyz = np.column_stack((positions, [sensor.get("height_m", 4.) for sensor in sensors]))
    vectors = points[None, :, :] - xyz[:, None, :]
    distance = np.linalg.norm(vectors, axis=2)
    unit = vectors / np.maximum(distance[:, :, None], 1e-12)
    for index, sensor in enumerate(sensors):
        if not sensor.get("directional"):
            continue
        hardware = sensor["manufacturer_specifications"]
        calculated = sensor["calculated_geometry"]
        model = sensor["simulation_assumptions"]
        yaw, pitch = np.radians(orientations[index])
        forward = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        right = np.array([-np.sin(yaw), np.cos(yaw), 0.])
        up = np.array([-np.sin(pitch) * np.cos(yaw), -np.sin(pitch) * np.sin(yaw), np.cos(pitch)])
        local_forward = unit[index] @ forward
        horizontal = np.arctan2(unit[index] @ right, local_forward)
        vertical = np.arctan2(unit[index] @ up,
                              np.sqrt(np.maximum(0., local_forward ** 2 + (unit[index] @ right) ** 2)))
        half_h = np.radians(hardware["horizontal_fov_deg"] / 2)
        half_v = np.radians(calculated["vertical_fov_deg"] / 2)
        inside = ((local_forward > 0) & (np.abs(horizontal) <= half_h) & (np.abs(vertical) <= half_v)
                  & (distance[index] <= model["max_evaluation_distance_m"]) & line_of_sight[index])
        pixels = pixels_on_target(sizes, np.maximum(distance[index], 1e-12), hardware["ifov_mrad"])
        pixel_response = 1 - np.exp(-pixels / model["pixels_for_63_percent"])
        noise_k = hardware["industrial_nedt_mk_max"] / 1000.
        contrast = model["nominal_thermal_contrast_k"] / (
            model["nominal_thermal_contrast_k"] + model["contrast_noise_multiplier"] * noise_k)
        normalized_edge = np.maximum(np.abs(horizontal) / half_h, np.abs(vertical) / half_v)
        edge = np.maximum(0., np.cos(np.minimum(1., normalized_edge) * np.pi / 2)) ** model["edge_falloff_exponent"]
        taper = np.maximum(0., np.cos(np.minimum(1., distance[index] / model["max_evaluation_distance_m"])
                                      * np.pi / 2)) ** 2
        atmosphere = np.exp(-distance[index] / model["atmospheric_attenuation_distance_m"])
        degradation = (weather["visibility"] * (1 - model["rain_loss_at_maximum"] * weather["rain"])
                       * (1 - model["humidity_loss_at_maximum"] * weather["humidity"]))
        probability = (sensor["strengths"]["thermal"] * pixel_response * contrast * edge
                       * taper * atmosphere * degradation)
        result[index, :, thermal] = np.where(inside, np.clip(probability, 0., 1.), 0.)
    return result


def _coverage_statistics(state: Mapping[str, Any], catalogue: Sequence[Mapping[str, Any]]) -> dict:
    points, weights, early_weights, sectors = legacy.forecast_points(state)
    placements = state["placements"]
    probabilities = sensing_probabilities(
        np.asarray([row["position"] for row in placements]),
        [catalogue[row["sensor_index"]] for row in placements], points, state["weather"],
        state["forecast"]["emitter_probability"],
        orientations=np.asarray([[row.get("yaw_deg", 0.), row.get("pitch_deg", 0.)] for row in placements]),
        target_size_m=state["forecast"]["target_size_m"])
    covered = 1. - np.prod(1. - probabilities, axis=(0, 2))
    sector_coverage = [float(np.average(covered[sectors == sector], weights=weights[sectors == sector]))
                       if weights[sectors == sector].sum() > 1e-12 else 0. for sector in range(8)]
    return {"coverage": float(covered @ weights), "early_coverage": float(covered @ early_weights),
            "sector_coverage": sector_coverage, "points": points, "weights": weights,
            "early_weights": early_weights, "covered": covered}


def build_observation(state: Mapping[str, Any], catalogue: Sequence[Mapping[str, Any]] | None = None,
                      *, now: float | None = None) -> dict:
    """Build joint profile/site/yaw/pitch actions with one final STOP row."""
    catalogue = validate_catalogue(catalogue)
    state = validate_public_state(state, catalogue, now=now)
    sites = np.asarray(state["sites"], dtype=np.float64)
    options, sensors, position_rows, orientation_rows, site_indices = [], [], [], [], []
    for sensor_index, sensor in enumerate(catalogue):
        angles = ([(yaw, pitch) for yaw in sensor["yaw_bins_deg"] for pitch in sensor["pitch_bins_deg"]]
                  if sensor.get("directional") else [(0., 0.)])
        for site_index, position in enumerate(sites):
            for yaw, pitch in angles:
                row = {"sensor_id": sensor["id"], "sensor_index": sensor_index, "site_index": site_index,
                       "position": position.tolist(), "yaw_deg": yaw, "pitch_deg": pitch, "stop": False}
                options.append(row); sensors.append(sensor); position_rows.append(position)
                orientation_rows.append((yaw, pitch)); site_indices.append(site_index)
    positions = np.asarray(position_rows, dtype=float)
    orientations = np.asarray(orientation_rows, dtype=float)
    count = len(sensors)
    options.append({"sensor_id": None, "sensor_index": -1, "site_index": -1, "position": [0., 0.],
                    "yaw_deg": 0., "pitch_deg": 0., "stop": True})
    stats = _coverage_statistics(state, catalogue)
    probabilities = sensing_probabilities(positions, sensors, stats["points"], state["weather"],
                                          state["forecast"]["emitter_probability"], orientations=orientations,
                                          target_size_m=state["forecast"]["target_size_m"])
    covered = 1. - np.prod(1. - probabilities, axis=2)
    gain = covered * (1. - stats["covered"])
    placement_positions = np.asarray([row["position"] for row in state["placements"]])
    nearest_placement = (np.linalg.norm(positions[:, None, :] - placement_positions[None, :, :], axis=2).min(axis=1)
                         if len(placement_positions) else np.full(count, 300.))
    track_positions = np.asarray([row["position"][:2] for row in state["tracks"]])
    nearest_track = (np.linalg.norm(positions[:, None, :] - track_positions[None, :, :], axis=2).min(axis=1)
                     if len(track_positions) else np.full(count, 500.))
    forecast = state["forecast"]
    cost = np.array([sensor["cost"] for sensor in sensors])
    radius = np.linalg.norm(positions, axis=1)
    modality = np.array([[sensor["ranges"][name] > 0 for name in MODALITIES] for sensor in sensors])
    ranges = np.array([[sensor["ranges"][name] for name in MODALITIES] for sensor in sensors])
    effectiveness = modality * legacy.modality_effectiveness(state["weather"], forecast["emitter_probability"])
    weights = np.array(forecast["approach_weights"])
    alignment_angles = np.array([np.radians(orientations[i, 0]) if sensor.get("directional")
                                 else np.arctan2(positions[i, 1], positions[i, 0])
                                 for i, sensor in enumerate(sensors)])
    alignment = np.cos(alignment_angles[:, None] - np.arange(8)[None, :] * np.pi / 4) @ weights
    placed_modalities = [sum(catalogue[row["sensor_index"]]["ranges"][name] > 0
                             for row in state["placements"]) / 4. for name in MODALITIES]
    track_confidence = float(np.mean([row["confidence"] for row in state["tracks"]])) if state["tracks"] else 0.
    fields: dict[str, Any] = {
        "bias": 1., "stop": 0., "east": positions[:, 0] / 150., "north": positions[:, 1] / 150.,
        "radius": radius / 150., "cost": cost / 4., "budget_remaining": state["budget_remaining"] / 4.,
        "budget_total": state["budget_total"] / 4.,
        "remaining_sites": (state["max_sites"] - len(state["placements"])) / 4.,
        "placed_count": len(state["placements"]) / 4.,
        "budget_fraction": state["budget_remaining"] / state["budget_total"], **state["weather"],
        "threat_altitude": forecast["altitude"] / 160., "threat_speed": forecast["speed"] / 30.,
        "emitter_probability": forecast["emitter_probability"], "swarm_size": forecast["swarm_size"] / 8.,
        "angular_uncertainty": forecast["angular_uncertainty"] / np.pi,
        "approach_alignment": alignment, "candidate_coverage": covered @ stats["weights"],
        "marginal_coverage": gain @ stats["weights"], "overlap": (covered * stats["covered"]) @ stats["weights"],
        "candidate_early_coverage": covered @ stats["early_weights"],
        "marginal_early_coverage": gain @ stats["early_weights"], "existing_coverage": stats["coverage"],
        "existing_early_coverage": stats["early_coverage"],
        "nearest_placement_distance": nearest_placement / 300., "nearest_track_distance": nearest_track / 500.,
        "track_confidence": track_confidence, "fresh_track_fraction": state["fresh_track_fraction"],
        "available_sensor_fraction": len(state["available_sensor_ids"]) / 8.,
    }
    for index, name in enumerate(MODALITIES):
        fields.update({f"modality_{name}": modality[:, index], f"range_{name}": ranges[:, index] / 200.,
                       f"effectiveness_{name}": effectiveness[:, index], f"placed_{name}": placed_modalities[index]})
    for index in range(8):
        fields[f"approach_sector_{index}"] = weights[index]
        fields[f"covered_sector_{index}"] = stats["sector_coverage"][index]
    features = np.zeros((count + 1, len(FEATURE_NAMES)), dtype=np.float32)
    for index, name in enumerate(FEATURE_NAMES):
        features[:count, index] = fields[name]
        if np.ndim(fields[name]) == 0:
            features[count, index] = fields[name]
    features[count, FEATURE_NAMES.index("stop")] = 1.
    mask = np.array([sensor["id"] in state["available_sensor_ids"] for sensor in sensors])
    mask &= cost <= state["budget_remaining"] + 1e-9
    mask &= (radius >= state["deployment_min_radius"] - 1e-9) & (radius <= state["deployment_max_radius"] + 1e-9)
    mask &= nearest_placement >= state["min_separation"] - 1e-9
    if len(placement_positions):
        mask &= nearest_placement > 1e-7
    mask &= ~np.isin(np.asarray(site_indices), state["blocked_sites"])
    if len(state["placements"]) >= state["max_sites"] or state["done"]:
        mask[:] = False
    return {"schema": FEATURE_SCHEMA, "feature_schema": FEATURE_SCHEMA, "option_features": features,
            "action_mask": np.append(mask, True), "feature_names": FEATURE_NAMES, "options": options,
            "state": state, "catalogue": catalogue}


def apply_placement(state: Mapping[str, Any], action: int,
                    catalogue: Sequence[Mapping[str, Any]] | None = None,
                    *, now: float | None = None) -> dict:
    observation = build_observation(state, catalogue, now=now)
    if observation["state"]["done"]:
        raise ValueError("Planning snapshot is done")
    if (isinstance(action, bool) or not isinstance(action, (int, np.integer))
            or not 0 <= action < len(observation["options"]) or not observation["action_mask"][action]):
        raise ValueError("Cannot apply an invalid directional placement")
    row = observation["options"][action]
    result = deepcopy(observation["state"])
    if row["stop"]:
        result["done"] = True
        return result
    sensor = observation["catalogue"][row["sensor_index"]]
    result["placements"].append({"sensor_id": sensor["id"], "sensor_index": row["sensor_index"],
                                  "position": list(row["position"]), "cost": sensor["cost"],
                                  "yaw_deg": row["yaw_deg"], "pitch_deg": row["pitch_deg"]})
    result["budget_remaining"] = max(0., result["budget_remaining"] - sensor["cost"])
    return result


def best_orientation(state: Mapping[str, Any], sensor: Mapping[str, Any],
                     site_position: Sequence[float]) -> tuple[float, float]:
    """Public-prior adapter for frozen policies; this is not learned orientation."""
    if not sensor.get("directional"):
        return 0., 0.
    catalogue = validate_catalogue([sensor])
    validated = deepcopy(dict(state))
    if "target_size_m" not in validated.get("forecast", {}):
        validated.setdefault("forecast", {})["target_size_m"] = .4
    points, weights, early_weights, _ = legacy.forecast_points(validated)
    angles = [(yaw, pitch) for yaw in catalogue[0]["yaw_bins_deg"] for pitch in catalogue[0]["pitch_bins_deg"]]
    probabilities = sensing_probabilities(
        np.repeat(np.asarray([site_position], dtype=float), len(angles), axis=0),
        [catalogue[0]] * len(angles), points, validated["weather"],
        validated["forecast"]["emitter_probability"], orientations=np.asarray(angles),
        target_size_m=validated["forecast"]["target_size_m"])
    covered = 1. - np.prod(1. - probabilities, axis=2)
    score = covered @ (weights + early_weights)
    return angles[int(np.argmax(score))]
