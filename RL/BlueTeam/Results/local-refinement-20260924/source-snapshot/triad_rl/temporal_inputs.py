"""Public-only, deadline-aware sensor/site features, additive to frozen v1.

These are uncertain forecasts, NOT simulated future truth. Public tracks and
ingress priors define deterministic trajectory quadrature. Exact rolling-window
confirmation probabilities are conditional on those hypotheses and independent
per-tick sensing draws. RF uses a declared mixture of independent emission and
pass-persistent emission, shared across all RF sensors. Neither component is a
claim to know the real emitter phase, period, future path or sensing outcome.

Mission timing/calibration is explicit provider configuration, not read from a
private scenario. Existing input validation, legal actions and v1 features are
preserved. The new feature schema is intentionally rejected by old checkpoints.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping

import numpy as np

from . import adaptive_inputs as legacy
from .temporal_confirmation import confirmation_statistics


FEATURE_SCHEMA = "triad.temporal_placement_features.v2"
FORECAST_SCHEMA = "triad.public_temporal_forecast.v1"
EXTRA_FEATURE_NAMES = (
    "temporal_detection", "temporal_confirmation", "temporal_timely", "temporal_early",
    "temporal_marginal_detection", "temporal_marginal_confirmation",
    "temporal_marginal_timely", "temporal_marginal_early",
    "temporal_existing_detection", "temporal_existing_confirmation",
    "temporal_existing_timely", "temporal_existing_early",
    "temporal_unconfirmed_detection", "temporal_confirmation_without_timely",
    "temporal_window_support", "temporal_predeadline_hits",
    "temporal_iid_timely", "temporal_persistent_timely",
    "temporal_marginal_return", "temporal_best_legal_timely_gain",
    "temporal_track_mass", "temporal_horizon_truncated_mass",
    "temporal_no_predicted_entry_mass", "temporal_quadrature_reduced", "sensor_height",
    *[f"calibrated_strength_{modality}" for modality in legacy.MODALITIES],
)
FEATURE_NAMES = (*legacy.FEATURE_NAMES, *EXTRA_FEATURE_NAMES)
_METRICS = ("detected", "confirmed", "timely", "early")
_EXPOSED = ("detection", "confirmation", "timely", "early")


@dataclass(frozen=True)
class TemporalConfig:
    """Public mission rules and explicit, imperfect forecast assumptions.

    Defaults match the synthetic benchmark's public operating rules. A real
    provider must supply and calibrate its own settings. Forecast horizon and
    quadrature caps bound computation; truncation is exposed, never concealed.
    """
    objective_radius_m: float = 20.
    lead_time_s: float = 4.
    look_interval_s: float = 1.
    required_confirmations: int = 2
    confirmation_window: int = 3
    horizon_s: float = 96.
    prior_spawn_radius_m: float = 320.
    altitude_uncertainty_m: float = 10.
    max_hypotheses: int = 64
    rf_persistent_weight: float = .5

    def __post_init__(self):
        bounds = {"objective_radius_m": (1, 100), "lead_time_s": (0, 30),
                  "look_interval_s": (.1, 3), "horizon_s": (.1, 300),
                  "prior_spawn_radius_m": (180, 2000), "altitude_uncertainty_m": (0, 100),
                  "rf_persistent_weight": (0, 1)}
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or not low <= value <= high):
                raise ValueError(f"Invalid temporal configuration: {name}")
        for name, low, high in (("required_confirmations", 1, 8),
                                ("confirmation_window", 1, 8), ("max_hypotheses", 8, 128)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid temporal configuration: {name}")
        if (self.required_confirmations > self.confirmation_window
                or self.prior_spawn_radius_m <= self.objective_radius_m
                or int(np.floor(self.horizon_s / self.look_interval_s)) + 1 > 512):
            raise ValueError("Temporal window, radius or horizon is inconsistent")


def _config(value):
    if value is None:
        return TemporalConfig()
    if isinstance(value, TemporalConfig):
        return value
    if isinstance(value, Mapping):
        return TemporalConfig(**dict(value))
    raise ValueError("Temporal configuration must be TemporalConfig or a mapping")


def _hypotheses(state, config, clock):
    """Use only validated public values; stable under public-track reordering."""
    forecast, tracks = state["forecast"], state["tracks"]
    confidence_sum = sum(track["confidence"] for track in tracks)
    track_mass = min(.75, state["fresh_track_fraction"] * confidence_sum / forecast["swarm_size"])
    # Explicit modes distinguish motion extrapolation from uncertain ingress.
    # Only closing tracks split between both; outward/stationary/vertical
    # tracks never gain an invented inward velocity. IDs are not features.
    sources = [("ingress", angle, config.prior_spawn_radius_m, forecast["altitude"], forecast["speed"],
                0., 0., forecast["emitter_probability"], False, 0., 0., weight * (1. - track_mass))
               for angle, weight in zip(np.arange(8) * np.pi / 4, forecast["approach_weights"])]
    for track in tracks:
        position = np.asarray(track["position"], dtype=float)
        velocity = np.asarray(track["velocity"], dtype=float)
        position = position + velocity * (clock - track["timestamp"])
        bearing = float(np.arctan2(position[1], position[0]))
        radius = float(np.linalg.norm(position[:2]))
        speed = float(np.linalg.norm(velocity))
        vertical = float(velocity[2])
        tangent = np.array([-np.sin(bearing), np.cos(bearing)])
        curve = float(np.clip(velocity[:2] @ tangent / max(speed, .1), -.7, .7))
        closing = -float(velocity[:2] @ np.array([np.cos(bearing), np.sin(bearing)]))
        mass = track_mass * track["confidence"] / confidence_sum if confidence_sum else 0.
        values = (bearing, radius, float(np.clip(position[2], 0, 1000)), max(speed, .1), curve,
                  vertical, track["emitter_probability"], track["confirmed"], float(velocity[0]), float(velocity[1]))
        sources.append(("kinematic", *values, mass * (.5 if closing > .1 else 1.)))
        if closing > .1:
            sources.append(("ingress", *values, mass * .5))
    hypotheses = []
    angle_error = max(.10, forecast["angular_uncertainty"]) * .5
    sigma = ((0., 0., .4), (-angle_error, 0., .15), (angle_error, 0., .15),
             (0., -config.altitude_uncertainty_m, .15), (0., config.altitude_uncertainty_m, .15))
    for mode, bearing, radius, altitude, speed, curve, vertical, emitter, confirmed, vx, vy, mass in sources:
        for angular, height, weight in sigma:
            if mass * weight > 0:
                rotated_x = vx * np.cos(angular) - vy * np.sin(angular)
                rotated_y = vx * np.sin(angular) + vy * np.cos(angular)
                hypotheses.append((mode, bearing + angular, radius, max(0., altitude + height),
                                   speed, curve, vertical, emitter, confirmed, rotated_x, rotated_y, mass * weight))
    # Merge exact duplicates and sort by physical values, never provider order.
    merged = {}
    for *values, mass in hypotheses:
        key = tuple(values)
        merged[key] = merged.get(key, 0.) + mass
    ordered = sorted(merged)
    weights = np.array([merged[key] for key in ordered], dtype=float)
    weights /= weights.sum()
    reduced = len(ordered) > config.max_hypotheses
    if reduced:
        # Deterministic weighted midpoint quadrature; small masses may merge or
        # disappear. This approximation is recorded as a feature/audit field.
        cdf = np.cumsum(weights)
        cdf[-1] = 1.
        indices = np.searchsorted(cdf, (np.arange(config.max_hypotheses) + .5) / config.max_hypotheses)
        selected, counts = np.unique(indices, return_counts=True)
        ordered, weights = [ordered[index] for index in selected], counts / config.max_hypotheses
    return ordered, weights, track_mass, reduced


def _forecast(state, config, clock):
    hypotheses, weights, track_mass, reduced = _hypotheses(state, config, clock)
    times = np.arange(int(np.floor(config.horizon_s / config.look_interval_s)) + 1) * config.look_interval_s
    u = np.linspace(0., 1., 65)
    paths, durations, emitters, confirmed, entries = [], [], [], [], []
    for mode, bearing, radius, altitude, speed, curve, vertical, emitter, known, vx, vy in hypotheses:
        distance = max(0., radius - config.objective_radius_m)
        enters = True
        if distance == 0:
            duration = config.look_interval_s * 1e-6
            path = np.tile([radius * np.cos(bearing), radius * np.sin(bearing), altitude], (len(times), 1))
        elif mode == "kinematic":
            east, north = radius * np.cos(bearing), radius * np.sin(bearing)
            a, b, c = vx * vx + vy * vy, 2. * (east * vx + north * vy), radius * radius - config.objective_radius_m ** 2
            discriminant = b * b - 4. * a * c
            enters = a > 1e-12 and b < 0 and discriminant >= 0
            duration = ((-b - np.sqrt(max(0., discriminant))) / (2. * a) if enters else
                        config.horizon_s + config.look_interval_s)
            elapsed = np.minimum(times, duration) if enters else times
            path = np.column_stack((east + vx * elapsed, north + vy * elapsed,
                                    np.clip(altitude + vertical * elapsed, 0, 1000)))
        else:
            radii = radius - distance * u
            angles = bearing + curve * np.sin(np.pi * u)
            heights = np.clip(altitude + vertical * distance / speed * u, 0, 1000)
            dense = np.column_stack((radii * np.cos(angles), radii * np.sin(angles), heights))
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
            duration = arc[-1] / speed
            travelled = np.minimum(times * speed, arc[-1])
            path = np.column_stack([np.interp(travelled, arc, dense[:, axis]) for axis in range(3)])
        paths.append(path)
        durations.append(duration)
        emitters.append(emitter)
        confirmed.append(known)
        entries.append(enters)
    return {"points": np.asarray(paths), "times": times, "zone_times": np.asarray(durations),
            "emitters": np.asarray(emitters), "confirmed": np.asarray(confirmed), "weights": weights,
            "enters": np.asarray(entries, dtype=bool), "track_mass": track_mass, "reduced": reduced}


def _statistics(no_hit_by_modality, forecast, config):
    """RF emission is shared across sensors before mixing temporal hypotheses."""
    non_rf_no_hit = np.prod(no_hit_by_modality[..., 1:], axis=-1)
    rf_hit_given_on = 1. - no_hit_by_modality[..., 0]
    duty = forecast["emitters"][:, None]
    iid_hits = 1. - non_rf_no_hit * (1. - duty * rf_hit_given_on)
    on_hits = 1. - non_rf_no_hit * (1. - rf_hit_given_on)
    off_hits = 1. - non_rf_no_hit
    args = dict(required_confirmations=config.required_confirmations,
                confirmation_window=config.confirmation_window, lead_time=config.lead_time_s)
    # Ticks after the final possible active hit cannot change these four
    # expectations. Pruning zero tails is exact, including a wholly empty pass.
    active = forecast["times"] <= forecast["zone_times"][:, None]
    used = np.flatnonzero(np.any((on_hits > 0) & active, axis=tuple(range(on_hits.ndim - 1))))
    ticks = int(used[-1]) + 1 if len(used) else 0
    calculate = lambda hits: confirmation_statistics(np.clip(hits[..., :ticks], 0, 1),
                                                     forecast["times"][:ticks], forecast["zone_times"], **args)
    iid = calculate(iid_hits)
    if np.any((rf_hit_given_on > 0) & (duty > 0) & (duty < 1)):
        on, off = calculate(on_hits), calculate(off_hits)
        persistent = {name: on[name] * forecast["emitters"] + off[name] * (1. - forecast["emitters"])
                      for name in _METRICS}
    else:
        persistent = {name: iid[name].copy() for name in _METRICS}
    mix = config.rf_persistent_weight
    combined = {name: (1. - mix) * iid[name] + mix * persistent[name] for name in _METRICS}
    known = forecast["confirmed"]
    if known.any():
        for name in _METRICS:
            initial = (forecast["zone_times"] >= config.lead_time_s).astype(float) if name == "timely" else 1.
            combined[name] = np.where(known, initial, combined[name])
        initial_timely = (forecast["zone_times"] >= config.lead_time_s).astype(float)
        iid["timely"] = np.where(known, initial_timely, iid["timely"])
        persistent["timely"] = np.where(known, initial_timely, persistent["timely"])
    # No-entry hypotheses have finite-horizon detection/confirmation only, not
    # an invented arrival deadline. Already-confirmed tracks are assumed known
    # at the current planning time; past confirmation timestamps are unavailable.
    for name in ("timely", "early"):
        combined[name] = np.where(forecast["enters"], combined[name], 0.)
    iid["timely"] = np.where(forecast["enters"], iid["timely"], 0.)
    persistent["timely"] = np.where(forecast["enters"], persistent["timely"], 0.)
    predeadline = (forecast["times"] <= (forecast["zone_times"] - config.lead_time_s)[:, None]) & forecast["enters"][:, None]
    possible = ((on_hits > 0) & predeadline & (duty > 0)) | ((off_hits > 0) & predeadline)
    cumulative = np.cumsum(possible, axis=-1)
    window_hits = cumulative.copy()
    window_hits[..., config.confirmation_window:] -= cumulative[..., :-config.confirmation_window]
    support = np.max(window_hits, axis=-1) >= config.required_confirmations
    support = np.where(known, forecast["zone_times"] >= config.lead_time_s, support)
    support &= forecast["enters"]
    combined.update(iid_timely=iid["timely"], persistent_timely=persistent["timely"],
                    support=support, predeadline_hits=np.sum(iid_hits * predeadline, axis=-1))
    return {key: np.asarray(value) @ forecast["weights"] for key, value in combined.items()}


class TemporalObservationBuilder:
    """Same public builder for simulated and external snapshots; no actuators.

    A one-snapshot cache reuses trajectory/capability calculations as placements
    change. Cache contents are private implementation details, not actor state.
    All returned features/public records are newly allocated by the v1 builder.
    """
    def __init__(self, config=None):
        self.config = _config(config)
        self._key = None
        self._forecast_cache = None
        self._probability_cache = None

    def observe(self, payload, catalogue=None, *, now=None):
        if isinstance(payload, str):
            payload = json.loads(payload)
        observation = legacy.build_observation(payload, catalogue, now=now)
        state, catalogue = observation["state"], observation["catalogue"]
        clock = state["timestamp"] if now is None else float(now)
        inputs = {key: state[key] for key in ("forecast", "tracks", "weather", "sites", "timestamp", "fresh_track_fraction")}
        inputs.update(catalogue=catalogue, now=clock, config=asdict(self.config))
        key = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).digest()
        if key != self._key:
            self._key, self._forecast_cache, self._probability_cache = key, _forecast(state, self.config, clock), None
        forecast = self._forecast_cache
        n = len(observation["options"]) - 1
        positions = np.asarray([option["position"] for option in observation["options"][:-1]])
        sensors = [catalogue[option["sensor_index"]] for option in observation["options"][:-1]]
        points = forecast["points"].reshape(-1, 3)
        shape = forecast["points"].shape[:2]
        if self._probability_cache is None and n * len(points) * 4 <= 4_000_000:
            self._probability_cache = legacy.sensing_probabilities(positions, sensors, points, state["weather"], 1.).reshape(n, *shape, 4)
        placements = state["placements"]
        existing_probabilities = legacy.sensing_probabilities(
            np.asarray([p["position"] for p in placements]), [catalogue[p["sensor_index"]] for p in placements],
            points, state["weather"], 1.)
        existing_no_hit = np.prod(1. - existing_probabilities, axis=0).reshape(*shape, 4)
        existing = _statistics(existing_no_hit, forecast, self.config)
        joined = {name: np.empty(n + 1) for name in existing}
        # Bounded row chunks also bound the confirmation DP's leading dimensions.
        for start in range(0, n, 32):
            end = min(n, start + 32)
            probability = (self._probability_cache[start:end] if self._probability_cache is not None else
                           legacy.sensing_probabilities(positions[start:end], sensors[start:end], points,
                                                        state["weather"], 1.).reshape(end - start, *shape, 4))
            values = _statistics((1. - probability) * existing_no_hit, forecast, self.config)
            for name in joined:
                joined[name][start:end] = values[name]
        for name in joined:
            joined[name][-1] = existing[name]
        fields = {}
        for internal, exposed in zip(_METRICS, _EXPOSED):
            fields["temporal_" + exposed] = joined[internal]
            fields["temporal_marginal_" + exposed] = np.maximum(0., joined[internal] - existing[internal])
            fields["temporal_existing_" + exposed] = existing[internal]
        marginal = lambda name: fields["temporal_marginal_" + name]
        old = observation["option_features"]
        cost = np.r_[[sensor["cost"] for sensor in sensors], 0.]
        coverage_gain = old[:, legacy.FEATURE_NAMES.index("marginal_coverage")].astype(float)
        redundancy = .3 * np.maximum(0., .05 - coverage_gain) / .05
        redundancy[-1] = 0.
        # Original terminal objective in expectation, not a new simulator reward.
        gain = (16. * marginal("timely") + 2. * marginal("detection") + marginal("confirmation")
                + 3. * marginal("early") + 4.5 * coverage_gain - .55 * cost - redundancy)
        gain[-1] = 0.
        fields.update(temporal_unconfirmed_detection=np.maximum(0., joined["detected"] - joined["confirmed"]),
                      temporal_confirmation_without_timely=np.maximum(0., joined["confirmed"] - joined["timely"]),
                      temporal_window_support=joined["support"], temporal_predeadline_hits=joined["predeadline_hits"] / 16.,
                      temporal_iid_timely=joined["iid_timely"], temporal_persistent_timely=joined["persistent_timely"],
                      temporal_marginal_return=gain / 20.,
                      temporal_best_legal_timely_gain=float(marginal("timely")[observation["action_mask"]].max()),
                      temporal_track_mass=forecast["track_mass"],
                      temporal_horizon_truncated_mass=float(((forecast["zone_times"] > self.config.horizon_s) & forecast["enters"]) @ forecast["weights"]),
                      temporal_no_predicted_entry_mass=float((~forecast["enters"]) @ forecast["weights"]),
                      temporal_quadrature_reduced=float(forecast["reduced"]),
                      sensor_height=np.r_[[sensor["height_m"] / 100. for sensor in sensors], 0.])
        for modality in legacy.MODALITIES:
            fields["calibrated_strength_" + modality] = np.r_[[sensor["strengths"][modality] for sensor in sensors], 0.]
        features = np.empty((n + 1, len(FEATURE_NAMES)), dtype=np.float32)
        features[:, :len(legacy.FEATURE_NAMES)] = old
        for column, name in enumerate(EXTRA_FEATURE_NAMES, start=len(legacy.FEATURE_NAMES)):
            features[:, column] = fields[name]
        if not np.isfinite(features).all():
            raise ValueError("Temporal forecast produced nonfinite features")
        observation.update(schema=FEATURE_SCHEMA, feature_schema=FEATURE_SCHEMA, feature_names=FEATURE_NAMES,
                           option_features=features, temporal_config=asdict(self.config),
                           temporal_forecast={"schema": FORECAST_SCHEMA, "hypotheses": len(forecast["weights"]),
                                              "quadrature_reduced": forecast["reduced"],
                                              "public_only": True, "physical_commands": False,
                                              "track_motion": "Closing: equal kinematic/ingress mixture; nonclosing: kinematic only",
                                              "no_entry": "Finite-horizon detection/confirmation; zero timely/early utility",
                                              "confirmed_track": "Assumed already confirmed at current planning time, not a known past timestamp"})
        return observation


def build_observation(public_state, catalogue=None, *, config=None, now=None):
    """Stateless convenience API; reuse a builder for sequential planning."""
    return TemporalObservationBuilder(config).observe(public_state, catalogue, now=now)


class TemporalPublicGreedy:
    """Non-RL initialization/control, not evidence that an RL policy has learned."""
    def act(self, observation, deterministic=True):
        if (observation.get("feature_schema") != FEATURE_SCHEMA
                or tuple(observation.get("feature_names", ())) != FEATURE_NAMES):
            raise ValueError("Temporal greedy requires the exact temporal feature contract")
        features = np.asarray(observation["option_features"])
        mask = np.asarray(observation["action_mask"])
        if mask.dtype != bool or features.shape != (len(mask), len(FEATURE_NAMES)) or not mask.any() or not np.isfinite(features).all():
            raise ValueError("Invalid temporal observation")
        legal = np.flatnonzero(mask)
        utility = features[:, FEATURE_NAMES.index("temporal_marginal_return")]
        stops = np.flatnonzero((features[:, FEATURE_NAMES.index("stop")] == 1.) & mask)
        if len(stops) != 1:
            raise ValueError("Temporal greedy requires exactly one legal STOP")
        if utility[legal].max() <= 0.:
            return int(stops[0])
        return int(legal[np.argmax(utility[legal])])
