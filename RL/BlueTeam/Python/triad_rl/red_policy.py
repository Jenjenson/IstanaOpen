"""Small initial-placement policies for the live Istana Red interface.

The learned policy is intentionally a one-step contextual-bandit policy.  It
chooses a legal, symmetric layout template; Unreal remains authoritative for
collision validation, movement, sensing, reward and termination.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np


POLICY_SCHEMA = "istana.red_template_softmax.v1"
CHECKPOINT_SCHEMA = "istana.red_template_checkpoint.v1"


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _integer(value, name, low=0, high=2**31 - 1):
    value = _number(value, name)
    if int(value) != value or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(value)


def _json(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class RedLayoutSpec:
    """Fixed action catalogue: global rotation crossed with common radius."""

    rotation_bins: int = 16
    radius_fractions: tuple[float, ...] = (.5,)
    formations: tuple[str, ...] = ("wedge",)

    def validate(self):
        if not 4 <= self.rotation_bins <= 128:
            raise ValueError("rotation_bins must be in [4, 128]")
        if not self.radius_fractions or len(self.radius_fractions) > 16:
            raise ValueError("radius_fractions must contain 1..16 values")
        previous = -1.0
        for value in self.radius_fractions:
            value = _number(value, "radius fraction")
            if not 0 <= value <= 1 or value <= previous:
                raise ValueError("radius fractions must be unique, increasing and in [0,1]")
            previous = value
        if not self.formations or any(value not in ("ring", "wedge") for value in self.formations):
            raise ValueError("formations must contain ring and/or wedge")
        if len(set(self.formations)) != len(self.formations):
            raise ValueError("formations must be unique")
        return self

    @property
    def action_count(self):
        return self.rotation_bins * len(self.radius_fractions) * len(self.formations)


def _context(context):
    if not isinstance(context, dict):
        raise ValueError("Red context must be an object")
    origin = context.get("objectiveWorldCm")
    if not isinstance(origin, dict) or set(("x", "y", "z")) - set(origin):
        raise ValueError("Red context requires objectiveWorldCm")
    result = {
        "origin": tuple(_number(origin[k], f"objectiveWorldCm.{k}") for k in ("x", "y", "z")),
        "groups": _integer(context.get("groupCount"), "groupCount", 1, 256),
        "minimum": _number(context.get("minRadiusCm"), "minRadiusCm"),
        "maximum": _number(context.get("maxRadiusCm"), "maxRadiusCm"),
        "spread": _number(context.get("spreadRadiusCm"), "spreadRadiusCm"),
        "height": _number(context.get("heightOffsetCm"), "heightOffsetCm"),
    }
    if result["minimum"] < 0 or result["maximum"] < result["minimum"] or result["spread"] < 0:
        raise ValueError("Invalid Red radius/spread constraints")
    movement = context.get("movement", {})
    result["spacing"] = _number(movement.get("spacingCm", 0), "movement.spacingCm")
    if result["spacing"] < 0:
        raise ValueError("movement.spacingCm cannot be negative")
    return result


def layout_catalogue(context, spec=RedLayoutSpec()):
    """Return deterministic template metadata, centers and a conservative mask."""
    spec.validate()
    values = _context(context)
    layouts, mask = [], []
    required = 2 * values["spread"] + values["spacing"]
    for radius_index, fraction in enumerate(spec.radius_fractions):
        radius = values["minimum"] + fraction * (values["maximum"] - values["minimum"])
        neighbor_distance = 2 * radius * math.sin(math.pi / values["groups"]) if values["groups"] > 1 else math.inf
        wedge_step = (2 * math.asin(min(1., required / (2 * radius))) + .02
                      if values["groups"] > 1 and radius > 0 else 0.)
        for formation in spec.formations:
            formation_fits = (neighbor_distance + 1e-9 >= required if formation == "ring" else
                              wedge_step * max(0, values["groups"] - 1) < 2 * math.pi)
            for rotation_index in range(spec.rotation_bins):
                rotation = 2 * math.pi * rotation_index / spec.rotation_bins
                centers = []
                for group in range(values["groups"]):
                    angle = (rotation + 2 * math.pi * group / values["groups"] if formation == "ring" else
                             rotation + (group - (values["groups"] - 1) / 2) * wedge_step)
                    centers.append([values["origin"][0] + radius * math.cos(angle),
                                    values["origin"][1] + radius * math.sin(angle),
                                    values["origin"][2] + values["height"]])
                layouts.append({"action": len(layouts), "formation": formation,
                                "rotation_index": rotation_index,
                                "rotation_degrees": 360 * rotation_index / spec.rotation_bins,
                                "radius_index": radius_index, "radius_cm": radius,
                                "centers": centers})
                mask.append(formation_fits)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("No template can satisfy the advertised group separation")
    return layouts, mask


def public_approach_exposure(red_context, blue_context, placements, centers, samples=32):
    """Approximate path exposure from public Blue geometry/capabilities only.

    This bounded train-only shaping signal deliberately does not inspect native
    truth, detections, RNG state or future observations. Native terminal reward
    remains separately reported and is the only evaluation outcome.
    """
    values = _context(red_context)
    samples = _integer(samples, "samples", 4, 512)
    if not isinstance(blue_context, dict) or not isinstance(placements, list):
        raise ValueError("Blue context and placements are required")
    sites = blue_context.get("publicSnapshot", {}).get("sites")
    catalogue = blue_context.get("catalogue")
    weather = blue_context.get("publicSnapshot", {}).get("weather", {})
    if not isinstance(sites, list) or not isinstance(catalogue, list):
        raise ValueError("Blue context requires public sites and catalogue")
    by_id = {row.get("id"): row for row in catalogue if isinstance(row, dict)}
    sensors = []
    for placement in placements:
        profile = by_id.get(placement.get("profileId"))
        site_id = _integer(placement.get("siteId"), "siteId", 0, len(sites) - 1)
        if profile is None or len(sites[site_id]) != 2:
            raise ValueError("Placement references an unknown public sensor/site")
        east, north = (_number(value, "site coordinate") for value in sites[site_id])
        sensors.append((np.asarray([values["origin"][0] / 100 + east,
                                    values["origin"][1] / 100 + north,
                                    values["origin"][2] / 100 + _number(profile.get("height_m", 0), "height_m")]),
                        profile))
    if not sensors:
        return 0.0
    factors = {"rf": .5 * (1 - .7 * _number(weather.get("rf_noise", 0), "rf_noise")),
               "radar": 1 - .4 * _number(weather.get("rain", 0), "rain"),
               "eo": _number(weather.get("visibility", 1), "visibility")
                     * (.15 + .85 * _number(weather.get("illumination", 1), "illumination"))
                     * (1 - .5 * _number(weather.get("rain", 0), "rain")),
               "thermal": (1 - .45 * _number(weather.get("humidity", 0), "humidity"))
                          * (1 - .3 * _number(weather.get("rain", 0), "rain"))}
    origin_m = np.asarray(values["origin"], dtype=float) / 100
    exposures = []
    for center in centers:
        if len(center) != 3:
            raise ValueError("Each Red center requires XYZ")
        start = np.asarray([_number(value, "Red center") for value in center]) / 100
        for fraction in np.linspace(0., .95, samples):
            point = start + fraction * (origin_m - start)
            all_miss = 1.0
            for sensor, profile in sensors:
                distance = float(np.linalg.norm(point - sensor))
                sensor_miss = 1.0
                ranges, strengths = profile.get("ranges", {}), profile.get("strengths", {})
                for modality in ("rf", "radar", "eo", "thermal"):
                    radius = _number(ranges.get(modality, 0), f"{modality} range")
                    strength = _number(strengths.get(modality, 0), f"{modality} strength")
                    probability = (0. if radius <= 0 or distance > radius else
                                   min(1., max(0., strength * factors[modality]
                                               * (1 - .5 * (distance / radius) ** 2))))
                    sensor_miss *= 1 - probability
                all_miss *= sensor_miss
            exposures.append(1 - all_miss)
    return float(np.mean(exposures))


class ScriptedRadialRedPolicy:
    name = "scripted_radial"

    def select(self, context, **_):
        values = _context(context)
        radius = (values["minimum"] + values["maximum"]) / 2
        centers = [[values["origin"][0] + radius * math.cos(2 * math.pi * i / values["groups"]),
                    values["origin"][1] + radius * math.sin(2 * math.pi * i / values["groups"]),
                    values["origin"][2] + values["height"]] for i in range(values["groups"])]
        return {"policy": self.name, "learned": False, "centers": centers,
                "radius_cm": radius, "rotation_degrees": 0.0}


class RandomLegalRedPolicy:
    name = "random_legal"

    def __init__(self, seed=0, spec=RedLayoutSpec()):
        self.rng, self.spec = np.random.default_rng(seed), spec.validate()

    def select(self, context, **_):
        layouts, mask = layout_catalogue(context, self.spec)
        action = int(self.rng.choice(np.flatnonzero(mask)))
        return {"policy": self.name, "learned": False, **layouts[action]}


class DispersedRandomRedPolicy:
    """Demo policy with independently randomized, legally separated approaches."""

    name = "random_dispersed"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(_integer(seed, "seed", 0, 2**63 - 1))

    def select(self, context, **_):
        values = _context(context)
        required = 2 * values["spread"] + values["spacing"]
        centers, angles, radii = [], [], []
        # Bounded sequential rejection stays deterministic for a seed. Unreal
        # remains authoritative for terrain and member-level placement.
        for _group in range(values["groups"]):
            for _attempt in range(4096):
                radius = float(self.rng.uniform(values["minimum"], values["maximum"]))
                angle = float(self.rng.uniform(0., 2 * math.pi))
                candidate = [values["origin"][0] + radius * math.cos(angle),
                             values["origin"][1] + radius * math.sin(angle),
                             values["origin"][2] + values["height"]]
                if all(math.dist(candidate[:2], existing[:2]) + 1e-9 >= required
                       for existing in centers):
                    centers.append(candidate)
                    angles.append(math.degrees(angle))
                    radii.append(radius)
                    break
            else:
                raise ValueError("Could not sample legally separated dispersed Red centers")
        return {"policy": self.name, "learned": False, "formation": "dispersed",
                "angles_degrees": angles, "radii_cm": radii, "centers": centers}


class LearnedRedPlacementPolicy:
    """Masked tabular softmax trained from one terminal reward per episode."""

    name = "learned_template_softmax"

    def __init__(self, seed=0, spec=RedLayoutSpec()):
        self.spec = spec.validate()
        self.rng = np.random.default_rng(_integer(seed, "seed", 0, 2**63 - 1))
        self.logits = np.zeros(self.spec.action_count, dtype=np.float64)
        self.adam_m = np.zeros_like(self.logits)
        self.adam_v = np.zeros_like(self.logits)
        self.baseline = 0.0
        self.episodes = self.updates = 0

    def _probabilities(self, mask, temperature=1.0):
        temperature = _number(temperature, "temperature")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        scores = np.where(mask, self.logits / temperature, -np.inf)
        scores -= np.max(scores[mask])
        weights = np.where(mask, np.exp(scores), 0.0)
        return weights / weights.sum()

    def select(self, context, *, deterministic=True, temperature=1.0, **_):
        layouts, mask = layout_catalogue(context, self.spec)
        probabilities = self._probabilities(mask, temperature)
        if deterministic:
            action = int(np.flatnonzero(mask)[np.argmax(self.logits[mask])])
        else:
            action = int(self.rng.choice(len(probabilities), p=probabilities))
        record = {"action": action, "probabilities": probabilities.tolist(),
                  "mask": mask.tolist(), "baseline": self.baseline}
        self.last_record = record
        return {"policy": self.name, "learned": True, **layouts[action],
                "training_record": record}

    def update(self, record, reward, *, learning_rate=.03, baseline_rate=.1,
               entropy_coefficient=.01, max_gradient_norm=5.0):
        reward = _number(reward, "reward")
        learning_rate = _number(learning_rate, "learning_rate")
        baseline_rate = _number(baseline_rate, "baseline_rate")
        entropy_coefficient = _number(entropy_coefficient, "entropy_coefficient")
        limit = _number(max_gradient_norm, "max_gradient_norm")
        if learning_rate <= 0 or not 0 < baseline_rate <= 1 or entropy_coefficient < 0 or limit <= 0:
            raise ValueError("Invalid Red optimizer settings")
        probabilities = np.asarray(record.get("probabilities"), dtype=np.float64)
        mask = np.asarray(record.get("mask"), dtype=bool)
        action = _integer(record.get("action"), "action", 0, self.spec.action_count - 1)
        before = _number(record.get("baseline"), "sample baseline")
        if (probabilities.shape != self.logits.shape or mask.shape != self.logits.shape
                or not mask[action] or not np.isfinite(probabilities).all()
                or not np.isclose(probabilities.sum(), 1.0)):
            raise ValueError("Invalid Red training record")
        # A zero baseline is badly biased when every return has the same large
        # sign (the live Red reward is about -28 after shaping). Bootstrap from
        # the first return, then learn only relative improvements.
        first_update = self.episodes == 0
        advantage = 0.0 if first_update else reward - before
        gradient = probabilities.copy()
        gradient[action] -= 1.0
        gradient *= advantage
        # Gradient of sum(p log p); adding it to the loss encourages entropy.
        safe = np.where(mask, np.maximum(probabilities, 1e-300), 1.0)
        entropy_gradient = probabilities * (np.log(safe) + 1.0)
        entropy_gradient -= probabilities * entropy_gradient.sum()
        gradient += entropy_coefficient * entropy_gradient
        norm = float(np.linalg.norm(gradient))
        gradient *= min(1.0, limit / max(norm, 1e-12))
        self.updates += 1
        self.adam_m = .9 * self.adam_m + .1 * gradient
        self.adam_v = .999 * self.adam_v + .001 * gradient * gradient
        mean = self.adam_m / (1 - .9 ** self.updates)
        variance = self.adam_v / (1 - .999 ** self.updates)
        self.logits -= learning_rate * mean / (np.sqrt(variance) + 1e-8)
        if first_update:
            self.baseline = reward
        else:
            self.baseline += baseline_rate * (reward - self.baseline)
        self.episodes += 1
        return {"reward": reward, "advantage": advantage, "baseline": self.baseline,
                "gradient_norm": norm, "action": action, "episodes": self.episodes}

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        archive = io.BytesIO()
        np.savez_compressed(archive, logits=self.logits, adam_m=self.adam_m, adam_v=self.adam_v)
        blob = archive.getvalue()
        metadata = {"schema": CHECKPOINT_SCHEMA, "policy_schema": POLICY_SCHEMA,
                    "layout_spec": asdict(self.spec), "episodes": self.episodes,
                    "updates": self.updates, "baseline": self.baseline,
                    "rng_state": self.rng.bit_generator.state,
                    "arrays_sha256": hashlib.sha256(blob).hexdigest()}
        (path / "arrays.npz").write_bytes(blob)
        (path / "checkpoint.json").write_text(_json(metadata) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        path = Path(path)
        metadata = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
        blob = (path / "arrays.npz").read_bytes()
        if metadata.get("schema") != CHECKPOINT_SCHEMA or metadata.get("policy_schema") != POLICY_SCHEMA:
            raise ValueError("Unsupported Red checkpoint schema")
        if hashlib.sha256(blob).hexdigest() != metadata.get("arrays_sha256"):
            raise ValueError("Red checkpoint array hash differs")
        spec_data = metadata.get("layout_spec", {})
        spec = RedLayoutSpec(rotation_bins=spec_data.get("rotation_bins"),
                             radius_fractions=tuple(spec_data.get("radius_fractions", ())),
                             formations=tuple(spec_data.get("formations", ())))
        policy = cls(spec=spec)
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as zipped:
                if {row.filename for row in zipped.infolist()} != {"logits.npy", "adam_m.npy", "adam_v.npy"}:
                    raise ValueError("Unexpected Red checkpoint arrays")
            with np.load(io.BytesIO(blob), allow_pickle=False) as arrays:
                loaded = [arrays[name].copy() for name in ("logits", "adam_m", "adam_v")]
        except (OSError, KeyError, zipfile.BadZipFile) as error:
            raise ValueError("Invalid Red checkpoint archive") from error
        if any(value.dtype != np.float64 or value.shape != (spec.action_count,) or not np.isfinite(value).all()
               for value in loaded):
            raise ValueError("Invalid Red checkpoint parameter shape or value")
        policy.logits, policy.adam_m, policy.adam_v = loaded
        policy.episodes = _integer(metadata.get("episodes"), "episodes", 0, 2**63 - 1)
        policy.updates = _integer(metadata.get("updates"), "updates", 0, 2**63 - 1)
        policy.baseline = _number(metadata.get("baseline"), "baseline")
        policy.rng.bit_generator.state = metadata.get("rng_state")
        return policy
