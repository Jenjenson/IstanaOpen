"""Randomized, sensing-only Blue Team deployment simulator.

This is a deliberately synthetic benchmark, not a validated RF/radar model
or an engagement simulator. 'Defended' means every approaching target gains
two confirmations with at least four seconds remaining before the protected
zone; no weapon engagement or physical control is modeled. RF/radar/EO/
thermal capability costs and nominal ranges follow the legacy toy catalogue.

Training and heldout use the same randomized family with disjoint seeds;
stress shifts ranges and introduces larger/more divergent swarms. Sensors
are selected jointly with arbitrary offered site coordinates. Future truth
exists only in ``scenario`` and terminal replay, never in public observation.
"""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .adaptive_inputs import (
    DEFAULT_CATALOGUE, FEATURE_NAMES, INPUT_SCHEMA, MODALITIES, build_observation,
    coverage_statistics, sensing_probabilities, validate_catalogue, validate_public_state, _apply_legal_option,
)


SCENARIO_SCHEMA = "triad.adaptive_scenario.v1"


def generate_scenario(seed: int, split: str = "train") -> dict:
    """Generate independent truth and imperfect public scenario priors.

    Splits do not secretly change the observation builder or sensor physics.
    A seed+split completely defines truth, forecast noise and sensing draws;
    callers must reserve heldout seeds from training and model selection.
    """
    if split not in ("train", "heldout", "validation", "stress"):
        raise ValueError("split must be train, validation, heldout or stress")
    rng = np.random.default_rng(seed)
    stress = split == "stress"
    count = int(rng.integers(3, 9) if stress else rng.integers(1, 6))
    approach = float(rng.uniform(-np.pi, np.pi))
    multi_direction = bool(rng.random() < (.65 if stress else .3))
    spread = float(rng.uniform(.2, 1.2 if stress else .65))
    base_altitude = float(rng.uniform(65, 160) if stress else rng.uniform(8, 92))
    base_speed = float(rng.uniform(17, 32) if stress else rng.uniform(6, 22))
    emitter_duty = float(rng.choice([.0, .15, .45, .8, 1.]))
    weather = {"visibility": float(rng.uniform(.08, .55) if stress else rng.uniform(.2, 1.)),
               "rain": float(rng.uniform(.4, 1.) if stress else rng.beta(1.2, 2.2)),
               "illumination": float(rng.choice([.08, .35, 1.])),
               "humidity": float(rng.uniform(.15, .95)), "rf_noise": float(rng.uniform(0, .6 if stress else .4))}
    targets, tracks = [], []
    public_bearings = []
    paths = ("direct", "curved", "weaving")
    for i in range(count):
        bearing = float(approach + rng.normal(0, spread))
        if multi_direction and i % 2:
            bearing += float(rng.uniform(1.4, 3.2))
        target = {"id": f"drone-{i + 1}", "bearing": bearing, "spawn_radius": float(rng.uniform(270, 370)),
                  "altitude": float(np.clip(base_altitude + rng.normal(0, 9), 5, 185)),
                  "speed": float(np.clip(base_speed + rng.normal(0, 2), 3, 38)),
                  "path": str(rng.choice(paths)), "curvature": float(rng.uniform(-.8, .8)),
                  "weave_amplitude": float(rng.uniform(.10, .5 if stress else .32)),
                  "weave_phase": float(rng.uniform(0, 2 * np.pi)),
                  "altitude_amplitude": float(rng.uniform(0, 15 if stress else 6)),
                  "emitter_duty": float(np.clip(emitter_duty + rng.normal(0, .05), 0, 1)),
                  "emitter_period": int(rng.integers(5, 14)), "emitter_phase": float(rng.random())}
        targets.append(target)
        public_bearing = bearing + float(rng.normal(0, .23 if not stress else .38))
        public_bearings.append(public_bearing)
        # Initial reports are noisy and may be missing; not exact target state.
        if rng.random() > (.3 if stress else .16):
            radius = target["spawn_radius"] + float(rng.normal(0, 20))
            tracks.append({"id": target["id"],
                           "position": [float(radius * np.cos(public_bearing)), float(radius * np.sin(public_bearing)),
                                        float(max(0., target["altitude"] + rng.normal(0, 10)))],
                           "velocity": [float(-base_speed * np.cos(public_bearing) + rng.normal(0, 2)),
                                        float(-base_speed * np.sin(public_bearing) + rng.normal(0, 2)), 0.],
                           "confidence": float(rng.uniform(.45, .9)), "timestamp": 0., "confirmed": False,
                           "emitter_probability": float(np.clip(target["emitter_duty"] + rng.normal(0, .18), 0, 1))})
    sector_angles = np.arange(8) * np.pi / 4
    weights = .04 + np.sum(np.exp(3.8 * (np.cos(sector_angles[:, None] - np.asarray(public_bearings)[None, :]) - 1.)), axis=1)
    weights /= weights.sum()
    rotation = float(rng.uniform(0, 2 * np.pi))
    site_angles = rotation + np.arange(16) * np.pi / 8
    sites = [[float(radius * np.cos(angle)), float(radius * np.sin(angle))]
             for radius in (60., 120.) for angle in site_angles]
    available = [s["id"] for s in DEFAULT_CATALOGUE]
    if rng.random() < .4:
        available.remove(str(rng.choice(available)))
    if rng.random() < .12 and len(available) > 3:
        available.remove(str(rng.choice(available)))
    budget = float(rng.choice([2., 2.4, 2.8, 3.2, 3.6, 4.]))
    public = {"schema": INPUT_SCHEMA, "timestamp": 0., "source": "simulation", "episode_id": str(seed),
              "max_track_age": 10., "sites": sites, "placements": [], "budget_total": budget,
              "budget_remaining": budget, "max_sites": 3, "min_separation": 20.,
              "deployment_min_radius": 30., "deployment_max_radius": 150.,
              "weather": weather, "forecast": {"approach_weights": weights.tolist(),
                "altitude": float(max(1., base_altitude + rng.normal(0, 9))),
                "speed": float(max(1., base_speed + rng.normal(0, 2))),
                "emitter_probability": float(np.clip(emitter_duty + rng.normal(0, .13), 0, 1)),
                "swarm_size": float(count), "angular_uncertainty": .65 if stress else .4},
              "tracks": tracks, "available_sensor_ids": available, "blocked_sites": [], "done": False}
    return {"schema": SCENARIO_SCHEMA, "seed": int(seed), "split": split, "public": public,
            "targets": targets, "objective_radius": 20., "dt": 1., "required_confirmations": 2,
            "confirmation_window": 3, "defence_lead_time": 4., "max_invalid_actions": 3}


def _validate_scenario(scenario: Mapping[str, Any], catalogue: Sequence[Mapping[str, Any]]) -> dict:
    scenario = deepcopy(dict(scenario))
    if scenario.get("schema") != SCENARIO_SCHEMA:
        raise ValueError(f"Scenario requires schema {SCENARIO_SCHEMA}")
    try:
        json.dumps(scenario, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Scenario must be finite JSON") from error
    scenario["public"] = validate_public_state(scenario["public"], catalogue)
    if scenario["public"]["placements"] or scenario["public"]["done"]:
        raise ValueError("A new simulated scenario must start with no placements and done=false")
    if abs(scenario["public"]["budget_remaining"] - scenario["public"]["budget_total"]) > 1e-9:
        raise ValueError("A new simulated scenario must start with its full budget")
    if not isinstance(scenario.get("seed"), int) or scenario["seed"] < 0:
        raise ValueError("Scenario seed must be a nonnegative integer")
    targets = scenario.get("targets")
    if not isinstance(targets, list) or not 1 <= len(targets) <= 64:
        raise ValueError("Scenario requires 1..64 targets")
    bounds = {"bearing": (-100, 100), "spawn_radius": (180, 2000), "altitude": (0, 1000),
              "speed": (1, 200), "curvature": (-3, 3), "weave_amplitude": (0, 2),
              "weave_phase": (-100, 100), "altitude_amplitude": (0, 300), "emitter_duty": (0, 1),
              "emitter_period": (1, 1000), "emitter_phase": (0, 1)}
    ids = set()
    for target in targets:
        if not isinstance(target.get("id"), str) or target["id"] in ids:
            raise ValueError("Target IDs must be unique strings")
        ids.add(target["id"])
        if target.get("path") not in ("direct", "curved", "weaving"):
            raise ValueError("Target path must be direct, curved or weaving")
        for name, (low, high) in bounds.items():
            value = target.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
                raise ValueError(f"Invalid target {name}: expected [{low}, {high}]")
    for name, low, high in (("objective_radius", 1, 100), ("dt", .1, 3), ("required_confirmations", 1, 10),
                            ("confirmation_window", 1, 10), ("defence_lead_time", 0, 30), ("max_invalid_actions", 1, 20)):
        value = scenario.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not low <= value <= high:
            raise ValueError(f"Invalid scenario {name}")
    for name in ("required_confirmations", "confirmation_window", "max_invalid_actions"):
        if int(scenario[name]) != scenario[name]:
            raise ValueError(f"Scenario {name} must be an integer")
        scenario[name] = int(scenario[name])
    if scenario["required_confirmations"] > scenario["confirmation_window"]:
        raise ValueError("required_confirmations cannot exceed confirmation_window")
    return scenario


class AdaptivePlacementEnv:
    """Sequential joint sensor/site placement with randomized sensing rollout.

    ``reset(seed=..., scenario=...) -> observation``;
    ``step(option_index) -> (observation, reward, done, info)``.
    The last action is STOP. A mask excludes illegal sensor/site combinations.
    An invalid attempt costs 1 and ends after three attempts, avoiding loops.
    Calling step after termination raises. Future detections are evaluated
    once placement stops/exhausts. Reward shaping telescopes: summed step
    rewards equal terminal ``info['return']`` and reward component sum.
    """
    feature_names = FEATURE_NAMES

    def __init__(self, seed: int = 0, split: str = "train", catalogue: Sequence[Mapping[str, Any]] | None = None):
        self.catalogue = validate_catalogue(catalogue)
        self._custom_catalogue = catalogue is not None
        self.split = split
        self._rng = np.random.default_rng(seed)
        self.scenario: dict = {}
        self.public_state: dict = {}
        self.placements: list[dict] = []
        self.total_reward = 0.
        self.done = False
        self.info: dict = {}
        self.reset(seed=seed)

    def reset(self, seed: int | None = None, scenario: Mapping[str, Any] | None = None) -> dict:
        if scenario is None:
            if seed is None:
                seed = int(self._rng.integers(0, 2 ** 62))
            scenario = generate_scenario(seed, self.split)
            if self._custom_catalogue:
                # Capability-compatible external catalogues need not use the five default IDs.
                scenario["public"]["available_sensor_ids"] = [s["id"] for s in self.catalogue]
        self.scenario = _validate_scenario(scenario, self.catalogue)
        self.public_state = deepcopy(self.scenario["public"])
        self.placements = self.public_state["placements"]
        self.total_reward = 0.
        self.done = False
        self.invalid_actions = 0
        self._redundancy_penalty = 0.
        self._potential = 0.
        self.info = {}
        self._observation = build_observation(self.public_state, self.catalogue)
        return self.observe()

    def observe(self) -> dict:
        # Features/options are copied so a consumer cannot mutate internal legality.
        return deepcopy(self._observation)

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode is done; reset before stepping again")
        valid_index = isinstance(action, (int, np.integer)) and not isinstance(action, bool) and 0 <= action < len(self._observation["options"])
        valid = valid_index and bool(self._observation["action_mask"][action])
        reward = 0.
        if not valid:
            self.invalid_actions += 1
            reward -= 1.
            if self.invalid_actions >= self.scenario["max_invalid_actions"]:
                reward += self._finish()
        elif self._observation["options"][action]["stop"]:
            reward += self._finish()
        else:
            option = self._observation["options"][action]
            sensor = self.catalogue[option["sensor_index"]]
            marginal = float(self._observation["option_features"][action, FEATURE_NAMES.index("marginal_coverage")])
            # Penalize near-zero forecast contribution, not merely having >1 sensor.
            redundancy = .3 * max(0., .05 - marginal) / .05
            self._redundancy_penalty += redundancy
            self.public_state = _apply_legal_option(self.public_state, option, self.catalogue)
            self.placements = self.public_state["placements"]
            self._observation = build_observation(self.public_state, self.catalogue)
            f = self._observation["option_features"][-1]
            potential = 2.5 * float(f[FEATURE_NAMES.index("existing_coverage")]) + float(f[FEATURE_NAMES.index("existing_early_coverage")])
            reward += potential - self._potential - redundancy
            self._potential = potential
            if not self._observation["action_mask"][:-1].any():
                reward += self._finish()
        self.total_reward += reward
        if self.done:
            self.info["return"] = self.total_reward
            self.info["reward"] = self.total_reward
            self._observation = build_observation(self.public_state, self.catalogue)
        return self.observe(), float(reward), self.done, deepcopy(self.info)

    def _finish(self) -> float:
        result = self._simulate()
        stats = coverage_statistics(self.public_state, self.catalogue)
        cost = sum(p["cost"] for p in self.placements)
        components = {
            "successful_defence": 8. * (1. - result["breached_fraction"]),
            "breaches": -8. * result["breached_fraction"],
            "detections": 2. * result["detected_fraction"],
            "track_confirmation": result["confirmed_fraction"],
            "early_detection": 3. * result["early_detection"],
            "coverage": 2. * stats["coverage"],
            "blind_spots": -2.5 * (1. - stats["coverage"]),
            "sensor_cost": -.55 * cost,
            "unnecessary_deployment": -self._redundancy_penalty,
            "invalid_actions": -float(self.invalid_actions),
        }
        self.done = True
        self.public_state["done"] = True
        self.info = {**result, "coverage": stats["coverage"], "sector_coverage": stats["sector_coverage"],
                     "cost": cost, "total_cost": cost, "invalid_actions": self.invalid_actions,
                     "reward_components": components, "placements": deepcopy(self.placements),
                     "catalogue": deepcopy(self.catalogue), "scenario_seed": self.scenario["seed"],
                     "split": self.scenario.get("split", "supplied"),
                     "success": result["breached_fraction"] == 0.,
                     "detection_rate": result["detected_fraction"], "breach_rate": result["breached_fraction"],
                     "objective_radius": self.scenario["objective_radius"],
                     "defence_definition": "timely confirmed detection; no engagement model"}
        # Invalid/redundant penalties were already paid incrementally.
        objective = sum(v for k, v in components.items() if k not in ("invalid_actions", "unnecessary_deployment"))
        return float(objective - self._potential)

    def _simulate(self) -> dict:
        scenario = self.scenario
        targets = scenario["targets"]
        dt = scenario["dt"]
        n = len(targets)
        # Reparameterize curved/weaving 3-D paths by arc length. A target's
        # speed is actual metres/second, not a radial closure approximation.
        dense_u = np.linspace(0., 1., 1001)
        dense_paths, arc_lengths = [], []
        for target in targets:
            radius = target["spawn_radius"] * (1. - dense_u) + scenario["objective_radius"] * dense_u
            angle = np.full(len(dense_u), target["bearing"])
            if target["path"] == "curved":
                angle += target["curvature"] * np.sin(np.pi * dense_u)
            elif target["path"] == "weaving":
                angle += target["weave_amplitude"] * np.sin(4 * np.pi * dense_u + target["weave_phase"]) * np.sin(np.pi * dense_u)
            path = np.column_stack((radius * np.cos(angle), radius * np.sin(angle),
                                    np.maximum(1., target["altitude"] + target["altitude_amplitude"] * np.sin(2 * np.pi * dense_u))))
            dense_paths.append(path)
            arc_lengths.append(np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        duration = np.array([distance[-1] / target["speed"] for distance, target in zip(arc_lengths, targets)])
        times = np.arange(0, float(duration.max()) + dt, dt)
        points = np.empty((len(times), n, 3))
        emissions = np.empty((len(times), n))
        for i, target in enumerate(targets):
            distance = np.minimum(times * target["speed"], arc_lengths[i][-1])
            for axis in range(3):
                points[:, i, axis] = np.interp(distance, arc_lengths[i], dense_paths[i][:, axis])
            phase = (times / target["emitter_period"] + target["emitter_phase"]) % 1.
            emissions[:, i] = phase < target["emitter_duty"]
        positions = np.asarray([p["position"] for p in self.placements])
        sensors = [self.catalogue[p["sensor_index"]] for p in self.placements]
        probabilities = sensing_probabilities(positions, sensors, points.reshape(-1, 3), scenario["public"]["weather"], emissions.ravel())
        probabilities = probabilities.reshape(len(sensors), len(times), n, 4)
        # One draw per target/time/modality, independent of sensor number/order.
        # Union aggregation gives redundant sensors legitimate reliability benefits.
        rng = np.random.default_rng(np.random.SeedSequence([scenario["seed"], 0xB10E]))
        draws = rng.random((len(times), n, 4))
        union = 1. - np.prod(1. - probabilities, axis=0)
        modality_hits = draws < union
        active = times[:, None] <= duration[None, :] + 1e-9
        hits = modality_hits.any(axis=2) & active
        first_detection = np.full(n, np.inf)
        first_confirmation = np.full(n, np.inf)
        frames = []
        window = scenario["confirmation_window"]
        for tick, time in enumerate(times):
            first_detection = np.where(hits[tick] & np.isinf(first_detection), time, first_detection)
            confirmed_now = hits[max(0, tick - window + 1):tick + 1].sum(axis=0) >= scenario["required_confirmations"]
            first_confirmation = np.where(confirmed_now & active[tick] & np.isinf(first_confirmation), time, first_confirmation)
            detections = []
            for target_index in np.flatnonzero(hits[tick]):
                # Attribution is explanatory: pick strongest sensor for each successful modality.
                successful = np.flatnonzero(modality_hits[tick, target_index])
                by_sensor: dict[int, list[str]] = {}
                for m in successful:
                    s = int(np.argmax(probabilities[:, tick, target_index, m]))
                    by_sensor.setdefault(s, []).append(MODALITIES[m])
                detections.extend({"sensor_index": s, "target_id": targets[target_index]["id"], "modalities": ms}
                                  for s, ms in by_sensor.items())
            frame_targets = []
            for i, target in enumerate(targets):
                breached = bool(time >= duration[i] and first_confirmation[i] > duration[i] - scenario["defence_lead_time"])
                frame_targets.append({"id": target["id"], "position": points[tick, i].tolist(),
                                      "detected": bool(hits[tick, i]), "ever_detected": bool(first_detection[i] <= time),
                                      "tracked": bool(first_confirmation[i] <= time), "breached": breached,
                                      "emitting": bool(emissions[tick, i]), "active": bool(active[tick, i])})
            frames.append({"time": float(time), "threats": frame_targets, "detections": detections})
        detected = np.isfinite(first_detection)
        confirmed = np.isfinite(first_confirmation)
        defended = first_confirmation <= duration - scenario["defence_lead_time"]
        early = np.where(confirmed, np.clip(1. - first_confirmation / duration, 0, 1), 0.)
        breached_fraction = float(1. - defended.mean())
        return {"outcome": "defended" if defended.all() else "breached" if not defended.any() else "partial",
                "detected_fraction": float(detected.mean()), "confirmed_fraction": float(confirmed.mean()), "early_detection": float(early.mean()),
                "breached_fraction": breached_fraction, "frames": frames,
                "target_results": [{"id": target["id"], "first_detection": float(first_detection[i]) if np.isfinite(first_detection[i]) else None,
                                    "first_confirmation": float(first_confirmation[i]) if np.isfinite(first_confirmation[i]) else None,
                                    "time_to_zone": float(duration[i]), "timely_confirmed": bool(defended[i])}
                                   for i, target in enumerate(targets)]}
