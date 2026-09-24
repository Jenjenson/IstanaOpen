"""Additive, reproducible robustness curriculum for the frozen v1 simulator.

This module changes the *distribution of inputs*, not the v1 sensing physics,
reward, feature schema or success definition. Capability samples are synthetic
calibration experiments, not specifications for manufactured sensor products.
No scenario is filtered or regenerated based on hidden truth, feasibility,
reward or the presence of an affordable sensor. In particular, the unchanged
stress family retains physically unobservable cases.

The five sensor profiles keep their four existing modalities. Capability
samples vary public calibration, availability, budget and offered positions;
the simulator scores those exact same calibrations. Private truth and these
public operational constraints use separate, named PCG64/SeedSequence streams.
Metadata remains outside the policy observation and input-provider contract.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np

from .adaptive_env import AdaptivePlacementEnv, generate_scenario
from .adaptive_inputs import DEFAULT_CATALOGUE, FEATURE_NAMES, validate_catalogue


ROBUST_SCENARIO_VERSION = "triad.robust_scenarios.v1"
PROFILES = ("mixed", "normal", "stress", "capability")
# Fixed integer namespaces, not Python's process-randomized hash(). Adding a
# new stream never changes the draw sequence of an existing named component.
_STREAMS = {
    "profile": 0xB10E2001,
    "capability_threat_family": 0xB10E2002,
    "catalogue": 0xB10E2003,
    "sites": 0xB10E2004,
    "resources": 0xB10E2005,
    "availability": 0xB10E2006,
    "blocked_sites": 0xB10E2007,
    "reset_sequence": 0xB10E2008,
}


def _seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("Scenario seed must be a nonnegative integer")
    return int(value)


def _profile(value: str) -> str:
    if value not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    return value


def _rng(seed: int, stream: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, _STREAMS[stream]])))


def curriculum_manifest() -> dict:
    """Return a fresh JSON-safe, auditable recipe; never an episode input."""
    return {
        "schema": ROBUST_SCENARIO_VERSION,
        "base_scenario_schema": "triad.adaptive_scenario.v1",
        "rng": "numpy.random.PCG64(SeedSequence([scenario_seed, named_stream]))",
        "stream_namespaces": deepcopy(_STREAMS),
        "mixed_probabilities": {"normal": .4, "stress": .3, "capability": .3},
        "normal": {"base_split": "train", "changes": "none"},
        "stress": {"base_split": "stress", "changes": "none", "retain_impossible_cases": True},
        "capability": {
            "base_split_probabilities": {"train": .5, "stress": .5},
            "profiles": [row["id"] for row in DEFAULT_CATALOGUE],
            "range_scale_uniform": [.65, 1.75],
            "range_scale_shared_within_sensor": True,
            "active_modality_strength_uniform": [.55, .98],
            "cost_scale_uniform": [.6, 1.5],
            "height_m_uniform": [2., 18.],
            "geometry_probabilities": {"jittered_rings": 1 / 3, "annular_cloud": 1 / 3, "partial_arc": 1 / 3},
            "jittered_rings": {"sites_per_ring_inclusive": [8, 20], "radii_uniform_m": [[42., 76.], [102., 138.]],
                                "angle_jitter_fraction_of_spacing": .3},
            "annular_cloud": {"site_count_inclusive": [16, 40], "radius_uniform_m": [35., 145.]},
            "partial_arc": {"site_count_inclusive": [16, 40], "radius_uniform_m": [35., 145.],
                            "arc_width_pi_uniform": [.75, 1.6]},
            "budget_uniform": [1.6, 4.8],
            "max_sites_inclusive": [2, 4],
            "min_separation_m_uniform": [12., 32.],
            "deployment_annulus_m": [30., 150.],
            "sensor_available_independent_probability": .8,
            "site_blocked_independent_probability": .1,
            "rejection_sampling": False,
            "calibration_status": "synthetic variation only; not validated real hardware specifications",
        },
        "base_truth_seed": "scenario_seed unchanged; original v1 generator owns truth and noisy public priors",
        "leakage_boundary": "capability streams never inspect targets, paths, sensing draws or outcomes",
        "unchanged": ["feature schema", "input-provider schema", "sensing physics", "reward", "defence definition"],
    }


def _capability_catalogue(seed: int) -> list[dict]:
    rng = _rng(seed, "catalogue")
    catalogue = deepcopy(DEFAULT_CATALOGUE)
    for sensor in catalogue:
        scale = float(rng.uniform(.65, 1.75))
        sensor["ranges"] = {modality: float(value * scale) for modality, value in sensor["ranges"].items()}
        sensor["strengths"] = {modality: float(rng.uniform(.55, .98)) for modality in sensor["ranges"]}
        sensor["cost"] *= float(rng.uniform(.6, 1.5))
        sensor["height_m"] = float(rng.uniform(2., 18.))
    return validate_catalogue(catalogue)


def _capability_sites(seed: int) -> tuple[list[list[float]], str]:
    rng = _rng(seed, "sites")
    geometry = str(rng.choice(("jittered_rings", "annular_cloud", "partial_arc")))
    rotation = float(rng.uniform(-np.pi, np.pi))
    if geometry == "jittered_rings":
        count = int(rng.integers(8, 21))
        spacing = 2 * np.pi / count
        # Independently jitter both rings; neither follows forecast bearings.
        angles = np.tile(rotation + np.arange(count) * spacing, 2)
        angles += rng.uniform(-.3, .3, 2 * count) * spacing
        radii = np.repeat([rng.uniform(42., 76.), rng.uniform(102., 138.)], count)
    else:
        count = int(rng.integers(16, 41))
        radii = rng.uniform(35., 145., count)
        width = 2 * np.pi if geometry == "annular_cloud" else float(rng.uniform(.75, 1.6) * np.pi)
        angles = rotation + rng.uniform(-width / 2, width / 2, count)
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles))).tolist(), geometry


def make_case(seed: int, profile: str = "mixed") -> tuple[dict, list[dict], dict]:
    """Resolve one deterministic case, matching the input seed without retries.

    Returns ``(v1_scenario, catalogue, audit_metadata)``. Normal and stress
    scenarios are byte-for-byte JSON-equivalent to ``generate_scenario`` for
    that seed and corresponding split. Capability modifies only operational
    fields in ``scenario['public']`` and the separately supplied catalogue;
    the original private targets, weather and noisy threat reports are kept.
    """
    seed, requested = _seed(seed), _profile(profile)
    resolved = requested
    if requested == "mixed":
        resolved = str(_rng(seed, "profile").choice(("normal", "stress", "capability"), p=(.4, .3, .3)))
    split = "stress" if resolved == "stress" else "train"
    if resolved == "capability":
        split = "train" if _rng(seed, "capability_threat_family").random() < .5 else "stress"
    scenario = generate_scenario(seed, split)
    catalogue = validate_catalogue()
    geometry = "v1_two_rotated_rings"
    if resolved == "capability":
        catalogue = _capability_catalogue(seed)
        public = scenario["public"]
        public["sites"], geometry = _capability_sites(seed)
        resources = _rng(seed, "resources")
        budget = float(resources.uniform(1.6, 4.8))
        public.update(budget_total=budget, budget_remaining=budget,
                      max_sites=int(resources.integers(2, 5)),
                      min_separation=float(resources.uniform(12., 32.)))
        available = _rng(seed, "availability").random(len(catalogue)) < .8
        public["available_sensor_ids"] = [sensor["id"] for sensor, keep in zip(catalogue, available) if keep]
        blocked = _rng(seed, "blocked_sites").random(len(public["sites"])) < .1
        public["blocked_sites"] = np.flatnonzero(blocked).tolist()
    metadata = {"schema": ROBUST_SCENARIO_VERSION, "scenario_seed": seed,
                "requested_profile": requested, "profile": resolved,
                "threat_split": split, "geometry": geometry,
                "synthetic_capability_variation": resolved == "capability",
                "scenario_filtered": False}
    return scenario, catalogue, metadata


class RobustPlacementEnv:
    """Versioned curriculum around the unmodified v1 placement environment.

    ``reset(seed=...)`` chooses solely from that seed/profile, independently
    of constructor calls or previous episodes. Unseeded resets use a separate
    reproducible sequence rooted in the constructor seed. As in v1, explicit
    seeded resets do not advance or reseed that sequence. ``case_metadata`` is
    available to training/reporting code, never inserted into observations.

    Supplied scenarios use the current catalogue unless ``catalogue`` is
    explicitly supplied. That path is intended for exact replay and tests and
    is labeled ``supplied``; it never silently modifies the supplied case.
    All v1 read attributes, including scenario, public_state, placements, info,
    catalogue, done and total_reward, delegate to the scoring environment.
    """
    feature_names = FEATURE_NAMES

    def __init__(self, seed: int = 0, profile: str = "mixed"):
        seed = _seed(seed)
        self.profile = _profile(profile)
        self._sequence_rng = _rng(seed, "reset_sequence")
        self._env = AdaptivePlacementEnv(seed=seed)
        self.case_metadata: dict = {}
        self.reset(seed=seed)

    def __getattr__(self, name: str) -> Any:
        # Do not recurse while construction/copying has not installed _env.
        env = self.__dict__.get("_env")
        if env is None:
            raise AttributeError(name)
        return getattr(env, name)

    def reset(self, seed: int | None = None, scenario: Mapping[str, Any] | None = None,
              catalogue: Sequence[Mapping[str, Any]] | None = None) -> dict:
        if scenario is None:
            if catalogue is not None:
                raise ValueError("A catalogue override requires an explicit scenario")
            seed = int(self._sequence_rng.integers(0, 2 ** 62)) if seed is None else _seed(seed)
            scenario, catalogue, metadata = make_case(seed, self.profile)
        else:
            if seed is not None and _seed(seed) != scenario.get("seed"):
                raise ValueError("Explicit seed must match the supplied scenario seed")
            catalogue = self._env.catalogue if catalogue is None else catalogue
            metadata = {"schema": ROBUST_SCENARIO_VERSION, "scenario_seed": scenario.get("seed"),
                        "requested_profile": self.profile, "profile": "supplied",
                        "threat_split": scenario.get("split", "supplied"), "geometry": "supplied",
                        "synthetic_capability_variation": None, "scenario_filtered": False}
        self._env.catalogue = validate_catalogue(catalogue)
        self._env.split = scenario.get("split", "supplied")
        observation = self._env.reset(scenario=scenario)
        self.case_metadata = metadata
        return observation

    def observe(self) -> dict:
        return self._env.observe()

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        return self._env.step(action)
