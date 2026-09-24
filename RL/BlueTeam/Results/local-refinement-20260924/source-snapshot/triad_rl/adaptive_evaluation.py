"""Paired, leakage-free comparisons for the adaptive deployment simulator.

Methods receive only the public observation. Scenario truth is recorded after
reset for audit/replay, but is never passed to a method. The legacy baseline is
the preserved, hash-checked toy-210 model through a deliberately lossy v4
projection, NOT a claim about its performance in the original Unreal world.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .placement_policy import DynamicPlacementPolicy, PlacementFeatures


REPORT_SCHEMA = "triad.adaptive_evaluation.v1"
FINAL_TEST_SEED = 2_000_000_000_000_000
LEGACY_PARAMETER_SHA256 = "1a7d10dbab564cbea654d46f0dbea684a68d62f76453ccade94210545fa17de7"
LEGACY_IDS = ("rf", "radar", "eo", "thermal", "fused")
LEGACY_MODALITIES = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0),
                     (0, 0, 0, 1), (0, 1, 0, 1))


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def legal_indices(observation: Mapping[str, Any]) -> np.ndarray:
    mask = np.asarray(observation["action_mask"], dtype=bool)
    if mask.ndim != 1 or not mask.any():
        raise ValueError("Evaluation observation has no legal actions")
    return np.flatnonzero(mask)


def nearest_same_sensor(observation: Mapping[str, Any], sensor_id: str,
                        position: Any) -> tuple[int, float | None]:
    """Project to a legal site without changing the requested sensor type.

    If that sensor has no legal option, stop. Never silently substitute a more
    useful sensor. Ties are stable in catalogue/site order.
    """
    choices = [int(i) for i in legal_indices(observation)
               if observation["options"][i].get("sensor_id") == sensor_id]
    if not choices:
        stop = len(observation["action_mask"]) - 1
        if not observation["action_mask"][stop]:
            raise ValueError("Missing sensor projection requires a legal stop action")
        return stop, None
    requested = np.asarray(position, dtype=float)
    if requested.shape != (2,) or not np.isfinite(requested).all():
        raise ValueError("Projection position must be finite East/North metres")
    distances = [float(np.linalg.norm(np.asarray(observation["options"][i]["position"])
                                      - requested)) for i in choices]
    nearest = int(np.argmin(distances))
    return choices[nearest], distances[nearest]


class RandomLegal:
    """Uniform over all legal sensor/site pairs and the legal stop action."""
    description = "Seeded uniform legal sensor/site-or-stop baseline; no truth access."

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, observation: Mapping[str, Any], deterministic: bool = True) -> int:
        return int(self.rng.choice(legal_indices(observation)))


class GreedyPublicCoverage:
    """One-step public-feature utility; the same input available to the learner."""
    description = "Greedy public expected marginal coverage/early benefit minus cost and overlap."
    metadata = {"utility": "3*marginal_coverage + marginal_early_coverage - .25*cost - .15*overlap",
                "cost_normalization": "public feature cost=actual_cost/4",
                "stop_utility": 0., "selection": "largest legal utility; deterministic stable ties",
                "coefficients": "fixed before heldout evaluation; not selected on test scenarios"}

    def __init__(self, seed: int = 0):
        pass

    def act(self, observation: Mapping[str, Any], deterministic: bool = True) -> int:
        names = list(observation["feature_names"])
        rows = np.asarray(observation["option_features"], dtype=float)
        # Names are a versioned public contract: fail rather than silently
        # degrading into argmax(zeros) if the environment changes its schema.
        weights = {"marginal_coverage": 3.0, "marginal_early_coverage": 1.0,
                   "cost": -0.25, "overlap": -0.15}
        missing = set(weights) - set(names)
        if missing:
            raise ValueError(f"Greedy public features missing: {sorted(missing)}")
        utility = sum(weight * rows[:, names.index(name)] for name, weight in weights.items())
        stop = len(observation["action_mask"]) - 1
        utility[stop] = 0.0
        legal = legal_indices(observation)
        return int(legal[np.argmax(utility[legal])])


class UniformFixedRFAndRadar:
    """Two opposite, fixed sites; no scenario-dependent sensor/position choice."""
    description = ("Fixed RF at (0,100m), radar at (0,-100m), then stop; nearest legal "
                   "same-sensor sites, skipping unavailable/unaffordable planned sensors. "
                   "This is a nonadaptive layout, not the toy policy.")

    def __init__(self, seed: int = 0):
        self.index = 0
        self.projections: list[dict[str, Any]] = []

    def act(self, observation: Mapping[str, Any], deterministic: bool = True) -> int:
        plan = (("rf", [0.0, 100.0]), ("radar", [0.0, -100.0]))
        while self.index < len(plan):
            sensor_id, position = plan[self.index]
            self.index += 1
            action, distance = nearest_same_sensor(observation, sensor_id, position)
            self.projections.append({"sensor_id": sensor_id, "requested_position": position,
                                     "distance_m": distance, "stopped_unavailable": False,
                                     "skipped_unavailable": distance is None})
            if distance is not None:
                return action
        return len(observation["action_mask"]) - 1


class LegacyToy210:
    """Execute the real old weights, with an explicit v4-compatible projection.

    The old Blue schema had no weather, track rows, site coverage or sensor
    emitter-effectiveness fields. This adapter intentionally never reads those
    fields or the new option feature matrix. Missing old native context is
    zero/constant-filled and documented in ``projection_contract``.
    """
    description = "Preserved toy-210 weights via lossy v4 context and nearest same-sensor legal sites."
    projection_contract = {
        "schema": "triad.legacy_to_adaptive_projection.v1",
        "lossy": True,
        "feature_width": 114,
        "catalogue_order": list(LEGACY_IDS),
        "old_distance_normalizer_m": 300.0,
        "budget_normalizer": "current total budget (minimum1), matching v4",
        "placement_normalizer": "current maximum deployment radius",
        "context_mapping": "Old25-vector using public deployment geometry, dominant prior bearing sector +/- angular uncertainty, duplicated public point forecasts for altitude/swarm/speed; spawn radius unavailable and zero-filled. Forecasts are noisy priors, not truth.",
        "excluded_public_fields": ["weather", "tracks", "coverage", "option_features", "emitter_behaviour"],
        "constant_context": {"phase": 0.25, "time": 0, "difficulty": 0,
                             "fixed_step_over2": 0.25, "formation_min_over300": 10 / 300,
                             "formation_max_over300": 40 / 300,
                             "confirmation_over60": 3 / 60, "hold_over60": 4 / 60,
                             "all_tracked": 0, "detected": 0, "tracked": 0,
                             "movement_axes": [1, 1, 1]},
        "position_projection": "tanh controls to current legal annulus, then nearest legal site of SAME sensor type; never substitute types",
        "interpretation": "Same surrogate scenarios, but old model has less information and projected actions; not native Unreal metrics or an architecture-only comparison.",
    }

    def __init__(self, checkpoint: str | Path, *, seed: int = 0,
                 require_preserved_fingerprint: bool = True):
        self.policy, metadata = DynamicPlacementPolicy.load_checkpoint(checkpoint)
        if self.policy.feature_size != 114:
            raise ValueError("Legacy projection requires the original114-feature model")
        if require_preserved_fingerprint and self.policy.parameter_hash() != LEGACY_PARAMETER_SHA256:
            raise ValueError("Legacy checkpoint is not the preserved toy-210 policy")
        self.policy.rng = np.random.default_rng(seed)
        self.metadata = {"parameter_sha256": metadata["parameter_sha256"],
                         "arrays_sha256": metadata["arrays_sha256"],
                         "projection": self.projection_contract}
        self.projections: list[dict[str, Any]] = []

    def features(self, observation: Mapping[str, Any]) -> PlacementFeatures:
        state = observation["state"]
        # Deliberately allowlist public fields. Do not inspect scenario truth,
        # weather, tracks, or new option features, even for convenient summaries.
        catalogue = {row["id"]: row for row in observation["catalogue"]}
        total_budget = max(float(state["budget_total"]), 1.0)
        radius = float(state["deployment_max_radius"])
        max_sites = int(state["max_sites"])
        mask = np.zeros(6, dtype=bool)
        mounts = np.zeros((5, 22), dtype=float)
        legal = legal_indices(observation)
        for i, (sensor_id, bits) in enumerate(zip(LEGACY_IDS, LEGACY_MODALITIES)):
            row = catalogue.get(sensor_id)
            if row is None:
                continue
            mask[i] = any(observation["options"][j].get("sensor_id") == sensor_id for j in legal)
            fov = ((360, 180), (360, 180), (60, 60), (90, 90), (90, 90))[i]
            mounts[i] = [0, 0, float(row.get("height_m", 4)) / 300, 0, 0, 1,
                         1, 0, 0, fov[0] / 360, fov[1] / 180,
                         max(row["ranges"].values(), default=0) / 300,
                         float(row["cost"]) / total_budget, *bits, 1, 1, 0, 1, 0]
        mask[-1] = bool(observation["action_mask"][-1])
        placed = np.zeros((max_sites, 9), dtype=float)
        for i, row in enumerate(state["placements"]):
            if i >= max_sites:
                raise ValueError("Public placement state exceeds maximum sites")
            if row["sensor_id"] not in LEGACY_IDS:
                # A legacy actor cannot have selected an unsupported type.
                raise ValueError("Legacy trajectory contains an unsupported sensor type")
            sensor_index = LEGACY_IDS.index(row["sensor_id"])
            placed[i] = [*(np.asarray(row["position"]) / radius), sensor_index / 4,
                         *LEGACY_MODALITIES[sensor_index], float(row["cost"]) / total_budget, 1]
        forecast = state.get("forecast", {})
        # Scalar forecasts populate the corresponding old config min/max
        # fields identically. They remain forecasts, not exact target values.
        spawn = (0., 0.)
        altitude = (float(forecast.get("altitude", 0.)),) * 2
        swarm = (float(forecast.get("swarm_size", 0.)),) * 2
        speed = (float(forecast.get("speed", 0.)),) * 2
        weights = np.asarray(forecast.get("approach_weights", []), dtype=float)
        bearing = (0., 0.)
        if weights.size:
            # New azimuth is0=east; old v4 bearing is0=north.
            angle = 90. - float(np.argmax(weights)) * 360. / len(weights)
            angle = (angle + 180.) % 360. - 180.
            spread = float(np.rad2deg(forecast.get("angular_uncertainty", 0.)))
            bearing = (max(-180., angle - spread), min(180., angle + spread))
        swarm_norm = max(swarm[1], 1.0)
        context = np.array([0.25, 0, float(state.get("protected_radius_m", 20)) / 300,
                            radius / 300, spawn[0] / 300, spawn[1] / 300,
                            bearing[0] / 180, bearing[1] / 180, 0, .25,
                            speed[1] / 50, altitude[0] / 300, altitude[1] / 300,
                            swarm[0] / swarm_norm, swarm[1] / swarm_norm,
                            10 / 300, 40 / 300, 3 / 60, 4 / 60, 0, 0, 0, 1, 1, 1])
        budget = [float(state["budget_remaining"]) / total_budget,
                  (max_sites - len(state["placements"])) / max_sites]
        global_features = np.concatenate([mounts.mean(0), mounts.max(0),
                                          placed.mean(0), placed.max(0), budget, context,
                                          [np.log1p(5), mask[:-1].mean()]])
        local = np.concatenate([mounts, np.zeros((5, 1))], axis=1)
        local = np.vstack([local, np.r_[np.zeros(22), 1]])
        rows = np.concatenate([local, np.tile(global_features, (6, 1))], axis=1)
        return PlacementFeatures(rows, mask, np.r_[mask[:-1], False], self.policy.feature_contract)

    def act(self, observation: Mapping[str, Any], deterministic: bool = True) -> int:
        sample = self.policy.act(self.features(observation), deterministic=deterministic)
        if sample.stop:
            return len(observation["action_mask"]) - 1
        state = observation["state"]
        radius = float(state["deployment_max_radius"])
        request = np.asarray(sample.position, dtype=float) * radius
        length = float(np.linalg.norm(request))
        inner = float(state["deployment_min_radius"]) + .5
        outer = radius - .5
        if length < 1e-12:
            request = np.array([0, inner])
        else:
            request *= np.clip(length, inner, outer) / length
        sensor_id = LEGACY_IDS[sample.catalogue_index]
        action, distance = nearest_same_sensor(observation, sensor_id, request)
        self.projections.append({"sensor_id": sensor_id, "requested_position": request.tolist(),
                                 "distance_m": distance, "stopped_unavailable": distance is None})
        return action


def paired_bootstrap(values: Any, reference: Any, *, seed: int = 0,
                     samples: int = 2000) -> dict[str, float | int]:
    a, b = np.asarray(values, dtype=float), np.asarray(reference, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError("Paired bootstrap requires nonempty matched vectors")
    if samples < 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid bootstrap settings/data")
    delta = a - b
    rng = np.random.default_rng(seed)
    # Bound peak memory for large benchmark suites.
    means = np.empty(samples)
    for start in range(0, samples, 128):
        count = min(128, samples - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        means[start:start + count] = delta[indices].mean(axis=1)
    return {"difference": float(delta.mean()), "lower95": float(np.quantile(means, .025)),
            "upper95": float(np.quantile(means, .975)), "pairs": int(a.size),
            "bootstrap_samples": samples}


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        return {"episodes": 0}
    numeric = set.intersection(*[{k for k, v in row["metrics"].items()
                                  if isinstance(v, (int, float, bool)) and v is not None}
                                 for row in episodes])
    result: dict[str, Any] = {"episodes": len(episodes)}
    for key in sorted(numeric):
        values = np.array([row["metrics"][key] for row in episodes], dtype=float)
        if np.isfinite(values).all():
            result[f"mean_{key}"] = float(values.mean())
    result["sensor_counts"] = dict(Counter(p["sensor_id"] for row in episodes
                                           for p in row.get("placements", [])))
    return result


def scenario_groups(scenario: Mapping[str, Any]) -> dict[str, str]:
    weather = scenario.get("public", {}).get("weather", scenario.get("weather", {}))
    threats = scenario.get("targets", scenario.get("threats", []))
    if isinstance(weather, Mapping) and "visibility" in weather:
        visibility = float(weather["visibility"])
        weather_name = "visibility_low" if visibility < .35 else "visibility_medium" if visibility < .65 else "visibility_high"
        weather_name += "/night" if weather.get("illumination", 1) < .2 else "/lit"
    else:
        weather_name = weather.get("name", weather.get("label", "unknown")) if isinstance(weather, Mapping) else weather
    emitters = set()
    for target in threats:
        if "emitter_duty" in target:
            duty = float(target["emitter_duty"])
            emitters.add("silent" if duty <= .05 else "low_duty" if duty <= .25 else "intermittent" if duty <= .7 else "mostly_on")
        else:
            emitters.add(str(target.get("emitter", "unknown")))
    altitudes = [float(t["altitude"]) for t in threats if "altitude" in t]
    speeds = [float(t["speed"]) for t in threats if "speed" in t]
    altitude = float(np.mean(altitudes)) if altitudes else None
    speed = float(np.mean(speeds)) if speeds else None
    rain = float(weather.get("rain", 0)) if isinstance(weather, Mapping) else 0.
    light = float(weather.get("illumination", 1)) if isinstance(weather, Mapping) else 1.
    return {"weather": str(weather_name), "swarm_size": str(len(threats)),
            "rain": "heavy" if rain >= .65 else "moderate" if rain >= .3 else "light",
            "illumination": "night" if light < .2 else "dim" if light < .65 else "day",
            "altitude": "unknown" if altitude is None else "low_below35m" if altitude < 35 else "medium35to75m" if altitude < 75 else "high75mplus",
            "speed": "unknown" if speed is None else "slow_below12mps" if speed < 12 else "medium12to20mps" if speed < 20 else "fast20mpsplus",
            "emitter": "+".join(sorted(emitters)),
            "path": "+".join(sorted({str(t.get("path_type", t.get("path", "unknown"))) for t in threats}))}


def assert_disjoint_seeds(seed: int, episodes: int, provenance: Mapping[str, Any]) -> None:
    """Reject held-out ranges overlapping declared training/validation ranges."""
    end = seed + episodes
    for name, block in provenance.items():
        blocks = block if isinstance(block, list) else [block]
        for entry in blocks:
            if isinstance(entry, Mapping) and "start" in entry and "count" in entry:
                a, b = int(entry["start"]), int(entry["start"]) + int(entry["count"])
                if max(a, seed) < min(b, end):
                    raise ValueError(f"Evaluation seeds overlap {name} seed range")


def checkpoint_provenance(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Read the actual trainer checkpoint schema, including older nested copies."""
    provenance = metadata.get("seed_provenance")
    if not provenance:
        provenance = metadata.get("training_state", {}).get("seed_provenance", {})
    return deepcopy(dict(provenance))


def actor_fingerprint(actor: Any) -> str | None:
    policy = getattr(actor, "policy", actor)
    if hasattr(policy, "weights_fingerprint"):
        return str(policy.weights_fingerprint())
    if hasattr(policy, "parameter_hash"):
        return str(policy.parameter_hash())
    return None


def implementation_fingerprints() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("adaptive_env.py", "adaptive_inputs.py", "adaptive_policy.py", "adaptive_evaluation.py")
            if (directory / name).is_file()}


def evaluate_methods(method_factories: Mapping[str, Callable[[int], Any]], *,
                     episodes: int = 200, seed: int = FINAL_TEST_SEED,
                     split: str = "heldout", replay_count: int = 1,
                     bootstrap_samples: int = 2000,
                     seed_provenance: Mapping[str, Any] | None = None,
                     env_factory: Callable[..., Any] | None = None) -> dict[str, Any]:
    """All methods face identical reset seeds; method RNG cannot consume env RNG.

    Each factory receives a deterministic, independent per-episode policy seed.
    The method is reset by re-creation every episode (important for fixed plans).
    Environment failures are surfaced, never silently dropped from comparison.
    """
    if episodes < 1 or seed < 0 or replay_count < 0 or bootstrap_samples < 1:
        raise ValueError("Invalid evaluation count/seed")
    if split not in {"train", "heldout", "stress"} or not method_factories:
        raise ValueError("Invalid evaluation split/methods")
    provenance = deepcopy(dict(seed_provenance or {}))
    if split != "train":
        assert_disjoint_seeds(seed, episodes, provenance)
    if env_factory is None:
        from .adaptive_env import AdaptivePlacementEnv
        env_factory = AdaptivePlacementEnv
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA, "training_performed": False,
        "environment": "adaptive approximate sensor simulator; not native Unreal/real-world validation",
        "split": split, "seed_provenance": {**provenance, "evaluation": {"start": seed, "count": episodes}},
        "protocol": {"paired_scenarios": True, "test_tuning_allowed": False,
                     "policy_rng_independent": True, "bootstrap": "paired percentile95% by episode",
                     "declared_seed_disjointness_verified": bool(provenance) and split != "train",
                     "uncertainty_scope": "scenario sampling only; not training-run variance or simulator fidelity",
                     "replay_count_per_method": min(replay_count, episodes)},
        "methods": {}, "paired_differences": {},
        "implementation_sha256": implementation_fingerprints(),
    }
    scenario_hashes: list[str] = []
    for method_index, (name, factory) in enumerate(method_factories.items()):
        records = []
        method_metadata: dict[str, Any] = {}
        for episode in range(episodes):
            episode_seed = seed + episode
            method_seed = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")
            policy_seed = int(np.random.SeedSequence([episode_seed, method_seed, 7183]).generate_state(1)[0])
            actor = factory(policy_seed)
            before_fingerprint = actor_fingerprint(actor)
            env = env_factory(seed=episode_seed, split=split)
            obs = env.reset(seed=episode_seed)
            scenario = json_safe(deepcopy(env.scenario))
            fingerprint = canonical_hash(scenario)
            if method_index == 0:
                scenario_hashes.append(fingerprint)
            elif scenario_hashes[episode] != fingerprint:
                raise RuntimeError("Paired scenario mismatch; refusing an unpaired comparison")
            total_return, done, actions = 0.0, False, []
            # Hard bound catches an invalid actor/environment loop, not losses.
            for _ in range(256):
                action = int(actor.act(deepcopy(obs), deterministic=True))
                actions.append(action)
                obs, reward, done, info = env.step(action)
                total_return += float(reward)
                if done:
                    break
            if not done:
                raise RuntimeError(f"Method{name} did not finish episode{episode_seed}")
            after_fingerprint = actor_fingerprint(actor)
            if before_fingerprint != after_fingerprint:
                raise RuntimeError(f"Method{name} changed weights during evaluation")
            result = json_safe(deepcopy(info))
            metrics = {k: v for k, v in result.items()
                       if (isinstance(v, (int, float, bool)) or v is None)
                       and k not in {"scenario_seed", "objective_radius"}}
            if isinstance(result.get("metrics"), Mapping):
                metrics.update(result["metrics"])
            metrics["return"] = total_return
            state = obs.get("state", {})
            placements = result.get("placements", state.get("placements", []))
            metrics["sensors_placed"] = len(placements)
            if "coverage" in metrics:
                metrics["blind_spot_fraction"] = 1. - metrics["coverage"]
            if "cost" in metrics and "budget_total" in state:
                metrics["budget_fraction_spent"] = metrics["cost"] / state["budget_total"]
            targets = result.get("target_results", [])
            if targets:
                # Undetected targets are censored at zone-arrival time rather
                # than silently omitted from latency comparisons.
                metrics["mean_confirmation_time_censored"] = float(np.mean([
                    t["time_to_zone"] if t["first_confirmation"] is None else min(t["first_confirmation"], t["time_to_zone"])
                    for t in targets]))
            projections = json_safe(getattr(actor, "projections", []))
            record = {"seed": episode_seed, "policy_seed": policy_seed,
                      "scenario_sha256": fingerprint, "scenario": scenario,
                      "weights_sha256_before": before_fingerprint,
                      "weights_sha256_after": after_fingerprint,
                      "groups": scenario_groups(scenario), "metrics": metrics,
                      "outcome": result.get("outcome"),
                      "reward_components": result.get("reward_components", {}),
                      "target_results": targets,
                      "actions": actions, "placements": placements, "projections": projections}
            if episode < replay_count:
                record["replay"] = result.get("replay", {
                    "frames": result.get("frames", []), "placements": placements,
                    "scenario": scenario, "metrics": metrics, "outcome": result.get("outcome"),
                    "reward_components": result.get("reward_components", {}),
                    "catalogue": result.get("catalogue", obs.get("catalogue", [])),
                    "sector_coverage": result.get("sector_coverage", []),
                    "objective_radius": result.get("objective_radius", 20),
                    "target_results": result.get("target_results", []),
                })
            records.append(record)
            if episode == 0:
                method_metadata = json_safe(deepcopy(getattr(actor, "metadata", {})))
                method_metadata["description"] = getattr(actor, "description", type(actor).__name__)
        subgroups: dict[str, Any] = {}
        for category in ("weather", "rain", "illumination", "swarm_size", "emitter", "path", "altitude", "speed"):
            labels = sorted({r["groups"][category] for r in records})
            subgroups[category] = {label: summarize([r for r in records if r["groups"][category] == label]) for label in labels}
        distances = [p["distance_m"] for r in records for p in r["projections"] if p["distance_m"] is not None]
        method_metadata["projection_summary"] = {
            "requests": sum(len(r["projections"]) for r in records),
            "mean_distance_m": float(np.mean(distances)) if distances else None,
            "max_distance_m": max(distances) if distances else None,
            "stopped_unavailable": sum(p["stopped_unavailable"] for r in records for p in r["projections"]),
            "skipped_unavailable": sum(p.get("skipped_unavailable", False) for r in records for p in r["projections"]),
        }
        report["methods"][name] = {"summary": summarize(records), "episodes": records,
                                   "subgroups": subgroups, "metadata": method_metadata}
    reference_name = "adaptive" if "adaptive" in report["methods"] else next(iter(report["methods"]))
    reference = report["methods"][reference_name]["episodes"]
    for name, method in report["methods"].items():
        if name == reference_name:
            continue
        comparison = {}
        for key in reference[0]["metrics"]:
            a = [r["metrics"].get(key) for r in reference]
            b = [r["metrics"].get(key) for r in method["episodes"]]
            if all(isinstance(v, (float, int, bool)) for v in a + b):
                if np.isfinite(a + b).all():
                    comparison[key] = paired_bootstrap(a, b, seed=seed % (2**32), samples=bootstrap_samples)
        report["paired_differences"][f"{reference_name}_minus_{name}"] = comparison
    report["scenario_sequence_sha256"] = canonical_hash(scenario_hashes)
    if implementation_fingerprints() != report["implementation_sha256"]:
        raise RuntimeError("Implementation changed during evaluation; refusing mixed-version evidence")
    return json_safe(report)
