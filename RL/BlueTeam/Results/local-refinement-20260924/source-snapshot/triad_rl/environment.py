"""AEC bridge. Unreal owns transitions and cumulative team rewards.

Red has separate deployment/movement decision heads. Both receive Red's team
return for credit assignment; never sum their copies in episode/team metrics.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from gymnasium import spaces
import numpy as np
from pettingzoo import AECEnv


class _HybridPlacementSpace(spaces.Dict):
    """Gym Dict space that also accepts PettingZoo's flat discrete mask.

    PettingZoo's generic AEC test passes ``observation["action_mask"]``
    directly to ``action_space.sample(mask)``.  For a hybrid action that mask
    describes only the categorical head, while Gym's normal Dict API expects a
    mapping for every head.  Keep the public action contract a real Dict and
    translate that one conventional shorthand locally.
    """

    def sample(self, mask=None, probability=None):
        if mask is not None and not isinstance(mask, dict):
            catalogue = self.spaces["catalogue_index"].sample(mask=mask)
            position = self.spaces["position"].sample()
            return {"catalogue_index": catalogue, "position": position}
        return super().sample(mask=mask, probability=probability)


class TRIADRedBlueEnv(AECEnv):
    metadata = {"name": "triad_red_blue_v4", "is_parallelizable": False, "render_modes": ["human"]}
    observation_schema = "triad.policy_observation.v4"
    possible_agents = ["blue_placement", "red_deployment", "red_movement"]
    team_by_agent = {"blue_placement": "blue", "red_deployment": "red", "red_movement": "red"}
    mount_feature_names = (
        "offset_east", "offset_north", "offset_up",
        "normal_east", "normal_north", "normal_up",
        "yaw_cos", "yaw_sin", "pitch", "horizontal_fov", "vertical_fov",
        "range", "cost", "passive_rf", "search_radar", "electro_optical",
        "thermal", "allow_dynamic_position", "surface_resolved", "occupied",
        "enabled", "currently_detecting",
    )
    placement_feature_names = (
        "position_east", "position_north", "catalogue_index", "passive_rf",
        "search_radar", "electro_optical", "thermal", "cost", "active",
    )

    def __init__(self, client, config_path, render_mode=None):
        super().__init__()
        self.client = client
        config_bytes = Path(config_path).read_bytes()
        self.config = json.loads(config_bytes.decode("utf-8"))
        try:
            fingerprint = hashlib.md5(config_bytes, usedforsecurity=False).hexdigest()
        except TypeError:  # Python builds without the optional OpenSSL keyword.
            fingerprint = hashlib.md5(config_bytes).hexdigest()
        self.config_fingerprint = f"md5:{fingerprint}"
        if self.config.get("SchemaVersion") != "triad.rl_training.v4":
            raise ValueError("The RL bridge requires a v4 dynamic-sensor catalogue configuration")
        self.catalogue = self.config["SensorCandidates"]
        self.option_count = len(self.catalogue)
        if not 1 <= self.option_count <= 256:
            raise ValueError("The approved catalogue must contain 1..256 options")
        self.max_targets = int(self.config["RedMaximumSwarmSize"])
        if not 1 <= self.max_targets <= 64:
            raise ValueError("RedMaximumSwarmSize must be in 1..64")
        if not 1 <= int(self.config["MaximumSensorSites"]) <= 256:
            raise ValueError("MaximumSensorSites must be in 1..256")
        self.render_mode = render_mode
        self._rng = np.random.default_rng(int(self.config["RandomSeed"]))
        self._action_spaces = {
            "blue_placement": _HybridPlacementSpace({
                "catalogue_index": spaces.Discrete(self.option_count + 1),
                "position": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            }),
            "red_deployment": spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32),
            "red_movement": spaces.Box(-1.0, 1.0, shape=(self.max_targets, 3), dtype=np.float32),
        }
        # Fixed, documented array layout for PettingZoo and policy libraries.
        # Flattening retains every field; it does not remove placement/terrain state.
        layout_shapes = {"mounts": (self.option_count, len(self.mount_feature_names)),
                         "placements": (int(self.config["MaximumSensorSites"]), len(self.placement_feature_names)),
                         "budget": (2,), "context": (25,)}
        self.observation_fields = {}
        self._observation_spaces = {}
        for agent in self.possible_agents:
            shapes = dict(layout_shapes)
            if agent != "blue_placement":
                shapes.update(targets=(self.max_targets, 13), active_mask=(self.max_targets,))
            offset, fields = 0, {}
            for name, shape in shapes.items():
                size = int(np.prod(shape))
                fields[name] = {"start": offset, "stop": offset + size, "shape": list(shape)}
                if name == "mounts":
                    fields[name]["columns"] = list(self.mount_feature_names)
                elif name == "placements":
                    fields[name]["columns"] = list(self.placement_feature_names)
                offset += size
            self.observation_fields[agent] = fields
            # Native bounds and finiteness are validated. Do not silently clip
            # large sensor ranges, costs or terrain information out of the policy.
            space = {"observation": spaces.Box(-np.inf, np.inf, shape=(offset,), dtype=np.float32)}
            if agent == "blue_placement":
                space["action_mask"] = spaces.MultiBinary(self.option_count + 1)
            self._observation_spaces[agent] = spaces.Dict(space)
        self.snapshot = {}
        self._last_team_totals = {"blue": 0.0, "red": 0.0}
        self._requires_reset = True
        self.invalid_actions = 0

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def action_dimension_mask(self, agent):
        """Active continuous dimensions; also used to mask policy log-probabilities.

        Toy deployment parameters are fixed and its movement is a one-axis
        corridor. Later curriculum configs may enable full planar/3D movement.
        This is not a Discrete action_mask: Gym's Box.sample does not accept one.
        """
        cfg = self.config
        if agent == "red_deployment":
            pairs = [("RedMinimumBearingDegrees", "RedMaximumBearingDegrees"),
                     ("RedMinimumSpawnRadiusMeters", "RedMaximumSpawnRadiusMeters"),
                     ("RedMinimumAltitudeMeters", "RedMaximumAltitudeMeters"),
                     ("RedMinimumSwarmSize", "RedMaximumSwarmSize"),
                     ("RedMinimumFormationSpacingMeters", "RedMaximumFormationSpacingMeters")]
            mask = np.asarray([cfg[a] != cfg[b] for a, b in pairs], dtype=np.float32)
            if self.max_targets == 1:
                mask[4] = 0  # formation spacing has no effect for a single target
            return mask
        if agent == "red_movement":
            mask = np.tile(self._vector(cfg["RedMovementAxisMask"]), (self.max_targets, 1)).astype(np.float32)
            mask[int(self.snapshot.get("ActiveTargetCount", 0)):] = 0
            return mask
        raise ValueError("Blue uses its hybrid catalogue action mask")

    @staticmethod
    def phase(state):
        value = state.get("Phase", "Inactive")
        if isinstance(value, int):
            if not 0 <= value <= 4:
                raise ValueError("Unknown Unreal phase")
            return value
        return {"Inactive": 0, "BluePlacement": 1, "RedDeployment": 2,
                "RedMovement": 3, "Terminal": 4}[str(value).rsplit("::", 1)[-1]]

    @staticmethod
    def _vector(value):
        return [float(value.get(axis, 0.0)) for axis in ("X", "Y", "Z")]

    @staticmethod
    def _vector2(value):
        return [float(value.get(axis, 0.0)) for axis in ("X", "Y")]

    @staticmethod
    def _sensor_modality_mask(candidate):
        sensor = candidate["Sensor"]
        mask = 0
        if float(sensor.get("DetectionRangeMeters", 0.0)) > 0.0 and sensor.get("SupportedFrequenciesGHz", []):
            mask |= 1
        if sensor.get("bEnableSearchRadar", False):
            mask |= 2
        if sensor.get("bEnableEOPTZ", False):
            mask |= 4
        if sensor.get("bEnableThermalPTZ", False):
            mask |= 8
        return mask

    def _blue_annulus_bounds(self):
        """Normalized annulus bounds with native/geodetic precision reserve."""
        placement_radius = float(self.config["BluePlacementRadiusMeters"])
        if not np.isfinite(placement_radius) or placement_radius <= 0.0:
            raise ValueError("Blue placement radius must be positive and finite")
        minimum = float(self.config["BlueMinimumObjectiveStandoffMeters"]) / placement_radius
        # Keep both boundaries half a metre away from native rejection planes.
        # This is intentionally much larger than float32 epsilon because Unreal
        # round-trips the normalized point through geodetic coordinates.
        margin = max(0.5 / placement_radius, 4.0 * np.finfo(np.float32).eps)
        lower, upper = minimum + margin, 1.0 - margin
        if minimum < 0.0 or lower > upper:
            raise ValueError("Blue placement annulus is too narrow for precision-safe actions")
        return lower, upper, placement_radius

    def _project_blue_position(self, position):
        """Map normalized East/North controls into the configured legal annulus."""
        value = np.asarray(position, dtype=np.float64)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError("Blue position must contain two finite normalized controls")
        radius = float(np.linalg.norm(value))
        minimum, maximum, _ = self._blue_annulus_bounds()
        if radius <= np.finfo(np.float64).eps:
            return np.asarray([0.0, minimum], dtype=np.float32)
        legal_radius = min(max(radius, minimum), maximum)
        return (value * (legal_radius / radius)).astype(np.float32)

    def _legalize_blue_position(self, position):
        """Also move a request away from committed sites, deterministically.

        The native mask cannot express which continuous coordinates violate
        separation.  Preserve an already-valid request exactly; otherwise pick
        the closest point from a stable annular lattice.  The small margin
        absorbs geodetic round-trip precision before Unreal checks the point.
        """
        requested = self._project_blue_position(position).astype(np.float64)
        existing = np.asarray([
            self._vector2(value) for value in self.snapshot.get("SensorObservations", [])
        ], dtype=np.float64).reshape(-1, 2)
        if existing.size == 0:
            return requested.astype(np.float32)
        minimum_radius, maximum_radius, scale = self._blue_annulus_bounds()
        separation = float(self.config["MinimumSensorSeparationMeters"]) / scale
        # Native converts normalized ENU through geodetic coordinates before
        # checking metre separation.  Reserve at least half a metre so a point
        # that is legal in Python remains legal after that round trip.
        margin = max(1e-4, 0.5 / scale)
        required = separation + margin
        if np.all(np.linalg.norm(existing - requested, axis=1) >= required):
            return requested.astype(np.float32)

        # Search is only needed for a conflicting request.  Include the request
        # bearing first, then a dense fixed polar lattice; selection is nearest
        # Euclidean point with stable array-order tie breaking.
        bearing = float(np.arctan2(requested[1], requested[0]))
        angles = bearing + np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
        radii = np.linspace(minimum_radius, maximum_radius, 65)
        candidates = np.stack([
            (radii[:, None] * np.cos(angles)[None, :]).ravel(),
            (radii[:, None] * np.sin(angles)[None, :]).ravel(),
        ], axis=1)
        # Stream across placed sites instead of allocating a
        # [lattice, MaximumSensorSites, 2] tensor at the 256-site limit.
        valid = np.ones(candidates.shape[0], dtype=np.bool_)
        required_squared = required * required
        for placed in existing:
            valid &= np.sum((candidates - placed) ** 2, axis=1) >= required_squared
        if not valid.any():
            return None
        valid_candidates = candidates[valid]
        nearest = int(np.argmin(np.sum((valid_candidates - requested) ** 2, axis=1)))
        return valid_candidates[nearest].astype(np.float32)

    def _effective_blue_action_mask(self):
        mask = np.asarray(self.snapshot["BlueActionMask"], dtype=np.int8).copy()
        if self.phase(self.snapshot) == 1 and any(
                self.catalogue[index].get("bAllowDynamicPosition", False) and mask[index]
                for index in range(self.option_count)):
            if self._legalize_blue_position([0.0, 0.0]) is None:
                for index, candidate in enumerate(self.catalogue):
                    if candidate.get("bAllowDynamicPosition", False):
                        mask[index] = 0
        return mask

    def _layout(self):
        cfg, state = self.config, self.snapshot
        scale = max(float(cfg["MaximumDistanceFromZoneMeters"]), 1.0)
        budget = max(float(cfg["SensorBudgetUnits"]), 1.0)
        mounts = np.zeros((self.option_count, len(self.mount_feature_names)), dtype=np.float32)
        for row, data in enumerate(state["MountObservations"]):
            yaw = np.deg2rad(float(data["YawDegrees"]))
            modality = int(data["SensorModalityMask"])
            mounts[row] = [
                *(np.asarray(self._vector(data["OffsetEnuMeters"])) / scale),
                *self._vector(data["SurfaceNormal"]), np.cos(yaw), np.sin(yaw),
                float(data["PitchDegrees"]) / 90.0, float(data["HorizontalFovDegrees"]) / 360.0,
                float(data["VerticalFovDegrees"]) / 180.0, float(data["RangeMeters"]) / scale,
                float(data["CostUnits"]) / budget,
                float(bool(modality & 1)), float(bool(modality & 2)),
                float(bool(modality & 4)), float(bool(modality & 8)),
                float(data["bAllowDynamicPosition"]),
                float(data["bSurfaceResolved"]), float(data["bOccupied"]), float(data["bEnabled"]),
                float(data["CurrentlyDetectingTargetCount"]) / self.max_targets,
            ]
        placed = np.zeros((int(cfg["MaximumSensorSites"]), len(self.placement_feature_names)), dtype=np.float32)
        for row, (index, position) in enumerate(zip(
                state["PlacedCatalogueIndices"], state["SensorObservations"])):
            index = int(index)
            modality = int(state["MountObservations"][index]["SensorModalityMask"])
            placed[row] = [
                *self._vector2(position), index / max(self.option_count - 1, 1),
                float(bool(modality & 1)), float(bool(modality & 2)),
                float(bool(modality & 4)), float(bool(modality & 8)),
                float(state["MountObservations"][index]["CostUnits"]) / budget, 1.0,
            ]
        context = np.asarray([
            self.phase(state) / 4.0, float(state["StepIndex"]) / int(cfg["EpisodeHorizonSteps"]),
            float(cfg["ProtectedZone"]["RadiusMeters"]) / scale,
            float(cfg["BluePlacementRadiusMeters"]) / scale,
            float(cfg["RedMinimumSpawnRadiusMeters"]) / scale,
            float(cfg["RedMaximumSpawnRadiusMeters"]) / scale,
            float(cfg["RedMinimumBearingDegrees"]) / 180.0,
            float(cfg["RedMaximumBearingDegrees"]) / 180.0,
            float(cfg.get("Difficulty", 0)) / 10.0,
            float(cfg["FixedStepSeconds"]) / 2.0,
            float(cfg["MaximumTargetSpeedMetersPerSecond"]) / 50.0,
            float(cfg["RedMinimumAltitudeMeters"]) / scale,
            float(cfg["RedMaximumAltitudeMeters"]) / scale,
            float(cfg["RedMinimumSwarmSize"]) / self.max_targets,
            float(cfg["RedMaximumSwarmSize"]) / self.max_targets,
            float(cfg["RedMinimumFormationSpacingMeters"]) / scale,
            float(cfg["RedMaximumFormationSpacingMeters"]) / scale,
            float(cfg["TrackConfirmationSteps"]) / int(cfg["EpisodeHorizonSteps"]),
            float(cfg["DefenceTrackHoldSteps"]) / int(cfg["EpisodeHorizonSteps"]),
            float(state["AllTargetsTrackedSteps"]) / int(cfg["DefenceTrackHoldSteps"]),
            float(state["CurrentlyDetectedTargetCount"]) / self.max_targets,
            float(state["TrackedTargetCount"]) / self.max_targets,
            *self._vector(cfg["RedMovementAxisMask"]),
        ], dtype=np.float32)
        return {"mounts": mounts, "placements": placed,
                "budget": np.asarray([float(state["RemainingBudgetUnits"]) / budget,
                    float(state["RemainingSiteCount"]) / int(cfg["MaximumSensorSites"])], dtype=np.float32),
                "context": context}

    def observe(self, agent):
        if self._requires_reset:
            raise RuntimeError("A successful reset is required before observing or stepping")
        layout = self._layout()
        if agent == "blue_placement":
            return {"observation": self._flatten(agent, layout),
                    "action_mask": self._effective_blue_action_mask()}
        targets = np.zeros((self.max_targets, 13), dtype=np.float32)
        active = np.zeros(self.max_targets, dtype=np.int8)
        for row, data in enumerate(self.snapshot.get("TargetObservations", [])):
            targets[row, :3] = self._vector(data["NormalizedZoneRelativeEnu"])
            targets[row, 3:6] = self._vector(data["NormalizedVelocityEnu"])
            targets[row, 6:9] = [float(data["bCurrentlyDetected"]),
                                float(data["bTracked"]), float(data["bConfirmedDetected"])]
            targets[row, 9:] = [float(data["ConsecutiveDetectionSteps"]) / self.config["TrackConfirmationSteps"],
                float(data["FirstDetectionStep"]) / self.config["EpisodeHorizonSteps"] if data["FirstDetectionStep"] >= 0 else -1.0,
                (self.snapshot["StepIndex"] - data["LastDetectionStep"]) / self.config["EpisodeHorizonSteps"] if data["LastDetectionStep"] >= 0 else -1.0,
                float(data["TrackedStepCount"]) / max(self.snapshot["StepIndex"], 1)]
            active[row] = 1
        return {"observation": self._flatten(agent, {**layout, "targets": targets, "active_mask": active})}

    def _flatten(self, agent, values):
        return np.concatenate([np.asarray(values[key], dtype=np.float32).ravel()
                               for key in self.observation_fields[agent]])

    def decode_observation(self, agent, observation):
        """Named copies for inspection; checkpoint schemas store the same offsets."""
        vector = observation["observation"]
        return {name: vector[field["start"]:field["stop"]].reshape(field["shape"]).copy()
                for name, field in self.observation_fields[agent].items()}

    def _validate_snapshot(self, state):
        if (state.get("SchemaVersion") != "triad.rl_step.v4"
                or state.get("ScenarioId") != self.config["ScenarioId"]
                or state.get("ConfigFingerprint") != self.config_fingerprint):
            raise RuntimeError("Unreal scenario/schema/fingerprint does not match the requested experiment")
        if len(state.get("BlueActionMask", [])) != self.option_count + 1 or len(state.get("MountObservations", [])) != self.option_count:
            raise RuntimeError("Unreal catalogue/mask does not match the policy schema")
        if any(type(value) is not bool for value in state["BlueActionMask"]):
            raise RuntimeError("Unreal returned a non-boolean Blue action mask")
        for index, mount in enumerate(state["MountObservations"]):
            candidate = self.catalogue[index]
            if (mount["CandidateId"] != candidate["CandidateId"]
                    or mount["SensorProfileId"] != candidate["SensorProfileId"]
                    or mount["MountId"] != candidate["MountId"]
                    or mount["CatalogueIndex"] != index
                    or bool(mount["bAllowDynamicPosition"]) != bool(candidate["bAllowDynamicPosition"])
                    or int(mount["SensorModalityMask"]) != self._sensor_modality_mask(candidate)):
                raise RuntimeError("Unreal catalogue order/identity does not match the policy")
            numeric = [*self._vector(mount["OffsetEnuMeters"]), *self._vector(mount["SurfaceNormal"]),
                       *[mount[k] for k in ["YawDegrees", "PitchDegrees", "HorizontalFovDegrees",
                                          "VerticalFovDegrees", "RangeMeters", "CostUnits",
                                          "SensorModalityMask", "CurrentlyDetectingTargetCount"]]]
            if (not np.isfinite(numeric).all() or float(mount["CostUnits"]) <= 0.0
                    or not 0 <= int(mount["CurrentlyDetectingTargetCount"]) <= self.max_targets):
                raise RuntimeError("Unreal returned non-finite mount information")
        if not np.isfinite([state["BlueReward"], state["RedReward"]]).all():
            raise RuntimeError("Unreal returned a non-finite reward")
        if not np.isfinite(state["RemainingBudgetUnits"]) or not -1e-6 <= state["RemainingBudgetUnits"] <= self.config["SensorBudgetUnits"] + 1e-6:
            raise RuntimeError("Unreal returned an invalid remaining budget")
        placed = state["PlacedCatalogueIndices"]
        if (len(placed) > int(self.config["MaximumSensorSites"])
                or any(type(i) is not int or not 0 <= i < self.option_count for i in placed)):
            raise RuntimeError("Unreal returned invalid placed catalogue indices")
        for index in set(placed):
            if placed.count(index) > 1 and not self.catalogue[index]["bAllowDynamicPosition"]:
                raise RuntimeError("Unreal duplicated a fixed-mount catalogue option")
        sensors = state.get("SensorObservations", [])
        if len(sensors) != len(placed):
            raise RuntimeError("Unreal sensor observations disagree with placed catalogue indices")
        minimum_radius = float(self.config["BlueMinimumObjectiveStandoffMeters"]) / max(
            float(self.config["BluePlacementRadiusMeters"]), 1.0
        )
        for sensor in sensors:
            position = np.asarray(self._vector2(sensor), dtype=np.float64)
            radius = float(np.linalg.norm(position))
            if not np.isfinite(position).all() or radius > 1.0 + 1e-3 or radius < minimum_radius - 1e-3:
                raise RuntimeError("Unreal returned an invalid normalized sensor position")
        spent = sum(float(self.catalogue[index]["CostUnits"]) for index in placed)
        expected_budget = float(self.config["SensorBudgetUnits"]) - spent
        if not np.isclose(float(state["RemainingBudgetUnits"]), expected_budget, atol=1e-5, rtol=1e-6):
            raise RuntimeError("Unreal remaining budget disagrees with placed sensor costs")
        if state["RemainingSiteCount"] != self.config["MaximumSensorSites"] - len(placed):
            raise RuntimeError("Unreal remaining-site count disagrees with placements")
        targets = state.get("TargetObservations", [])
        if state["ActiveTargetCount"] != len(targets) or not 0 <= len(targets) <= self.max_targets:
            raise RuntimeError("Unreal target count disagrees with the observation schema")
        for target in targets:
            if not np.isfinite(self._vector(target["NormalizedZoneRelativeEnu"]) + self._vector(target["NormalizedVelocityEnu"])).all():
                raise RuntimeError("Unreal returned a non-finite target state")
        if bool(state["bTerminal"]) != (self.phase(state) == 4):
            raise RuntimeError("Unreal phase and terminal status disagree")

    def reset(self, seed=None, options=None):
        self._requires_reset = True
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        episode_seed = int(seed) if seed is not None else int(self._rng.integers(0, 2**31 - 1))
        for index, space in enumerate(self._action_spaces.values()):
            space.seed((episode_seed + index) % (2**32))
        snapshot = self.client.reset(episode_seed)
        self._validate_snapshot(snapshot)
        if self.phase(snapshot) != 1 or int(snapshot["TransitionIndex"]) != 0:
            raise RuntimeError("Reset did not produce a fresh Blue placement episode")
        if snapshot["BlueReward"] != 0 or snapshot["RedReward"] != 0:
            raise RuntimeError("Reset leaked previous episode rewards")
        if snapshot["EpisodeSeed"] != episode_seed:
            raise RuntimeError("Unreal reset used a different episode seed")
        self.snapshot = snapshot
        self.agents = self.possible_agents[:]
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self._last_team_totals = {"blue": 0.0, "red": 0.0}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {"episode_seed": episode_seed} for a in self.agents}
        self.agent_selection = "blue_placement"
        self._skip_agent_selection = None
        self.invalid_actions = 0
        self._requires_reset = False

    def step(self, action):
        if self._requires_reset:
            raise RuntimeError("A successful reset is required before observing or stepping")
        acting = self.agent_selection
        if self.terminations[acting] or self.truncations[acting]:
            self._was_dead_step(action)
            return
        if not self.action_space(acting).contains(action):
            self.invalid_actions += 1
            raise ValueError(f"Action outside the {acting} space; no command sent to Unreal")
        if acting == "blue_placement":
            index = int(action["catalogue_index"])
            if not self._effective_blue_action_mask()[index]:
                self.invalid_actions += 1
                raise ValueError("Masked Blue action; no command sent to Unreal")
            stop = index == self.option_count
            if stop or not self.catalogue[index]["bAllowDynamicPosition"]:
                position = np.zeros(2, dtype=np.float32)
            else:
                position = self._legalize_blue_position(action["position"])
                if position is None:
                    self.invalid_actions += 1
                    raise ValueError("No separation-valid dynamic Blue position remains; no command sent to Unreal")
        # A timeout or malformed response may follow an accepted native action.
        # Fail closed until reset; never retry against an uncertain episode.
        self._requires_reset = True
        if acting == "blue_placement":
            state = self.client.blue_action(index, position.tolist(), stop=stop)
        elif acting == "red_deployment":
            state = self.client.red_deployment((np.asarray(action) * self.action_dimension_mask(acting)).tolist())
        else:
            velocities = np.asarray(action) * self.action_dimension_mask(acting)
            state = self.client.red_actions(velocities[:int(self.snapshot["ActiveTargetCount"])].tolist())
        self._validate_snapshot(state)
        if int(state["TransitionIndex"]) != int(self.snapshot["TransitionIndex"]) + 1:
            raise RuntimeError("Stale, repeated or skipped Unreal transition; reset instead of replaying an action")
        if state["EpisodeSeed"] != self.snapshot["EpisodeSeed"]:
            raise RuntimeError("Unreal episode changed during an action")
        old_phase, new_phase = self.phase(self.snapshot), self.phase(state)
        if new_phase not in {1: {1, 2}, 2: {3}, 3: {3, 4}}[old_phase]:
            raise RuntimeError("Unreal returned an illegal phase transition")
        if state["StepIndex"] != self.snapshot["StepIndex"] + int(old_phase == 3):
            raise RuntimeError("Unreal movement-step counter disagrees with the action phase")
        # last() is read-only. Clear only this head's delivered return when it acts.
        # Non-acting heads keep delayed credit until their next action or terminal.
        self._cumulative_rewards[acting] = 0.0
        self._clear_rewards()
        current = {"blue": float(state["BlueReward"]), "red": float(state["RedReward"])}
        for agent in self.agents:
            team = self.team_by_agent[agent]
            self.rewards[agent] = current[team] - self._last_team_totals[team]
        self._last_team_totals = current
        self.snapshot = state
        terminal = bool(state["bTerminal"])
        for agent in self.agents:
            self.terminations[agent] = terminal
            self.infos[agent] = {"termination_reason": state["TerminationReason"],
                "episode_seed": state["EpisodeSeed"], "transition_index": state["TransitionIndex"],
                "team": self.team_by_agent[agent], "team_returns": current.copy()}
        self.agent_selection = {1: "blue_placement", 2: "red_deployment",
                                3: "red_movement", 4: "red_movement"}[self.phase(state)]
        self._accumulate_rewards()
        if terminal:
            self._deads_step_first()
        self._requires_reset = False

    def render(self):
        if self.render_mode == "human":
            print(f"phase={self.snapshot.get('Phase')} step={self.snapshot.get('StepIndex')} "
                  f"budget={self.snapshot.get('RemainingBudgetUnits')} returns={self._last_team_totals}")

    def close(self):
        pass  # Unreal is externally owned; an env wrapper must not kill it.
