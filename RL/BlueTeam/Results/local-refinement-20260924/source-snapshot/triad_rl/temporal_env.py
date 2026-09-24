"""Temporal public observations around the unchanged adaptive scoring core.

Controlled initialization bypasses AdaptivePlacementEnv.__init__ because that
constructor generates an initial case before reset. We initialize its four
constructor dependencies explicitly, then call the frozen reset with exactly
the requested scenario. No sensing, reward, termination or action code changes.
Private scenario/report attributes are available to reporting callers only;
the observation builder receives public_state and catalogue, never the case.
"""
from collections.abc import Mapping

import numpy as np

from .adaptive_env import AdaptivePlacementEnv
from .adaptive_inputs import validate_catalogue
from . import robust_scenarios as robust
from .temporal_inputs import FEATURE_NAMES, TemporalObservationBuilder, _config


_MISSION_RULES = {"objective_radius": "objective_radius_m", "dt": "look_interval_s",
                  "required_confirmations": "required_confirmations",
                  "confirmation_window": "confirmation_window", "defence_lead_time": "lead_time_s"}


class TemporalPlacementEnv:
    """Explicit case or seed/profile initialization; public configuration only.

    Supplied cases are never used to derive mission configuration. All five
    scoring rules must match it exactly. Unseeded resets use the frozen robust
    reset-sequence stream; explicit resets neither reseed nor advance it.
    Read attributes delegate to the core for reporting, not policy inputs.
    """
    feature_names = FEATURE_NAMES

    def __init__(self, seed=None, profile="mixed", *, scenario=None, catalogue=None, config=None):
        if seed is None and scenario is None:
            raise ValueError("Temporal environment requires an explicit seed or scenario")
        if scenario is not None and not isinstance(scenario, Mapping):
            raise ValueError("Supplied scenario must be a mapping")
        self.profile, self.config = robust._profile(profile), _config(config)
        initial_seed = robust._seed(seed if seed is not None else scenario.get("seed"))
        self._sequence_rng = robust._rng(initial_seed, "reset_sequence")
        self.reset(seed=seed, scenario=scenario, catalogue=catalogue)

    def __getattr__(self, name):
        core = self.__dict__.get("_env")
        if core is None or name.startswith("_"):
            raise AttributeError(name)
        return getattr(core, name)

    def reset(self, seed=None, scenario=None, catalogue=None, *, profile=None):
        selected_profile = self.profile if profile is None else robust._profile(profile)
        if scenario is None:
            if catalogue is not None:
                raise ValueError("A catalogue override requires an explicit scenario")
            seed = int(self._sequence_rng.integers(0, 2 ** 62)) if seed is None else robust._seed(seed)
            scenario, catalogue, metadata = robust.make_case(seed, selected_profile)
        else:
            if not isinstance(scenario, Mapping):
                raise ValueError("Supplied scenario must be a mapping")
            scenario_seed = robust._seed(scenario.get("seed"))
            if seed is not None and robust._seed(seed) != scenario_seed:
                raise ValueError("Explicit seed must match the supplied scenario")
            if catalogue is None:
                core = self.__dict__.get("_env")
                catalogue = None if core is None else core.catalogue
            metadata = {"schema": robust.ROBUST_SCENARIO_VERSION, "scenario_seed": scenario_seed,
                        "requested_profile": selected_profile, "profile": "supplied",
                        "threat_split": scenario.get("split", "supplied"), "geometry": "supplied",
                        "synthetic_capability_variation": None, "scenario_filtered": False}
        for name, configured in _MISSION_RULES.items():
            value = scenario.get(name)
            if isinstance(value, (bool, np.bool_)) or value != getattr(self.config, configured):
                raise ValueError(f"Scenario {name} differs from explicit public temporal configuration")
        core = AdaptivePlacementEnv.__new__(AdaptivePlacementEnv)
        core.catalogue = validate_catalogue(catalogue)
        core.split, core._custom_catalogue = scenario.get("split", "supplied"), catalogue is not None
        core._rng = np.random.default_rng(robust._seed(scenario.get("seed")))
        core.reset(scenario=scenario)
        builder = TemporalObservationBuilder(self.config)
        observation = builder.observe(core.public_state, core.catalogue)
        self._env, self._builder = core, builder
        self.profile, self.case_metadata = selected_profile, metadata
        return observation

    def observe(self):
        return self._builder.observe(self._env.public_state, self._env.catalogue)

    def step(self, action):
        _, reward, done, info = self._env.step(action)
        return self.observe(), reward, done, info


TemporalRobustPlacementEnv = TemporalPlacementEnv
