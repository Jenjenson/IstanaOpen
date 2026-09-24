"""Focused Blue-placement rollouts with reproducible scripted Red actions."""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .placement_policy import (
    DynamicPlacementPolicy,
    PlacementFeatureAdapter,
    PlacementSample,
)


def _finite_controls(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all() or np.any(np.abs(result) > 1.0):
        raise ValueError(f"{label} must have shape {shape} with finite values in [-1, 1]")
    return result


@dataclass(frozen=True)
class RedActionScript:
    """Deterministic or seeded-uniform Red deployment and movement controls.

    JSON form::

        {
          "mode": "fixed",
          "deployment": [0, 0, 0, 0, 0],
          "movement": [0, -1, 0],
          "movement_steps": [[0, -1, 0], [0.2, -1, 0]]
        }

    ``movement_steps`` is optional and overrides ``movement`` by simulation
    step, repeating its final row after the script ends. A three-vector is
    broadcast to every active target. ``mode: uniform`` samples all active
    dimensions from the rollout's seeded RNG.
    """

    mode: str = "fixed"
    deployment: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    movement: tuple[float, ...] = (0.0, -1.0, 0.0)
    movement_steps: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"fixed", "uniform"}:
            raise ValueError("Red script mode must be 'fixed' or 'uniform'")
        _finite_controls(self.deployment, (5,), "Red deployment")
        _finite_controls(self.movement, (3,), "Red movement")
        for index, row in enumerate(self.movement_steps):
            _finite_controls(row, (3,), f"Red movement_steps[{index}]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RedActionScript":
        allowed = {"mode", "deployment", "movement", "movement_steps"}
        extra = set(value) - allowed
        if extra:
            raise ValueError(f"Unknown Red script fields: {sorted(extra)}")
        return cls(
            mode=str(value.get("mode", "fixed")),
            deployment=tuple(float(item) for item in value.get("deployment", (0,) * 5)),
            movement=tuple(float(item) for item in value.get("movement", (0, -1, 0))),
            movement_steps=tuple(
                tuple(float(item) for item in row)
                for row in value.get("movement_steps", ())
            ),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "RedActionScript":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("Red script JSON must contain an object")
        return cls.from_mapping(value)

    @staticmethod
    def _dimension_mask(env: Any, agent: str, shape: tuple[int, ...]) -> np.ndarray:
        if not hasattr(env, "action_dimension_mask"):
            return np.ones(shape, dtype=np.float32)
        mask = np.asarray(env.action_dimension_mask(agent), dtype=np.float32)
        if mask.shape != shape:
            raise ValueError(f"{agent} action-dimension mask has the wrong shape")
        return mask

    def deployment_action(self, env: Any, rng: np.random.Generator) -> np.ndarray:
        shape = tuple(int(value) for value in env.action_space("red_deployment").shape)
        if shape != (5,):
            raise ValueError("TRIAD Red deployment action must have shape (5,)")
        if self.mode == "uniform":
            action = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        else:
            action = _finite_controls(self.deployment, shape, "Red deployment").copy()
        return action * self._dimension_mask(env, "red_deployment", shape)

    def movement_action(
        self, env: Any, step_index: int, rng: np.random.Generator
    ) -> np.ndarray:
        shape = tuple(int(value) for value in env.action_space("red_movement").shape)
        if len(shape) != 2 or shape[1] != 3:
            raise ValueError("TRIAD Red movement action must have shape (targets, 3)")
        if self.mode == "uniform":
            action = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        else:
            if self.movement_steps:
                vector = self.movement_steps[min(int(step_index), len(self.movement_steps) - 1)]
            else:
                vector = self.movement
            action = np.repeat(
                _finite_controls(vector, (3,), "Red movement")[None, :], shape[0], axis=0
            )
        return action * self._dimension_mask(env, "red_movement", shape)


def blue_environment_action(env: Any, sample: PlacementSample) -> tuple[Any, PlacementSample]:
    """Adapt a sample to v4 hybrid Dict actions or the legacy scalar bridge."""
    action_space = env.action_space("blue_placement")
    subspaces = getattr(action_space, "spaces", None)
    if isinstance(subspaces, Mapping) and set(subspaces) == {"catalogue_index", "position"}:
        return sample.environment_action(), sample
    # Compatibility with the v3 bridge while native v4 is being built. The
    # position was not dispatched, so it must not contribute a policy gradient.
    if sample.position_active:
        sample = replace(
            sample,
            position=np.zeros(2, dtype=np.float32),
            position_latent=np.zeros(2, dtype=np.float64),
            position_active=False,
        )
    return int(sample.catalogue_index), sample


def _phase(env: Any) -> int:
    state = env.snapshot
    if hasattr(env, "phase"):
        return int(env.phase(state))
    value = str(state.get("Phase", "Inactive")).rsplit("::", 1)[-1].replace(" ", "")
    return {"Inactive": 0, "BluePlacement": 1, "RedDeployment": 2,
            "RedMovement": 3, "Terminal": 4}[value]


def _profile_id(env: Any, catalogue_index: int) -> str:
    catalogue = getattr(env, "catalogue", None)
    if catalogue is None:
        catalogue = getattr(env, "config", {}).get("SensorCandidates", [])
    if not 0 <= catalogue_index < len(catalogue):
        return str(catalogue_index)
    candidate = catalogue[catalogue_index]
    return str(candidate.get("SensorProfileId", candidate.get("CandidateId", catalogue_index)))


def episode_metrics(
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    samples: Sequence[PlacementSample] = (),
    profile_ids: Sequence[str] = (),
    modality_counts: Mapping[str, int] | None = None,
    local_invalid: int = 0,
) -> dict[str, Any]:
    """Metrics use native cumulative team returns exactly once per team."""
    if not bool(state.get("bTerminal")):
        raise ValueError("Metrics require a completed native episode")
    targets = list(state.get("TargetObservations", ()))
    confirmed = [target for target in targets if bool(target.get("bConfirmedDetected"))]
    steps = int(state.get("StepIndex", 0))
    first_steps = [int(target.get("FirstDetectionStep", -1)) for target in confirmed]
    tracked_steps = sum(int(target.get("TrackedStepCount", 0)) for target in targets)
    reason = str(state.get("TerminationReason", "Unknown")).rsplit("::", 1)[-1]
    breach = reason.replace(" ", "") == "ProtectedZoneReached"
    selected = [sample for sample in samples if not sample.stop]
    observed_positions: list[np.ndarray] = []
    for value in state.get("SensorObservations", ()):
        if isinstance(value, Mapping):
            position = np.asarray([value.get("X", 0.0), value.get("Y", 0.0)], dtype=np.float64)
        else:
            position = np.asarray(value, dtype=np.float64)
        if position.shape != (2,) or not np.isfinite(position).all():
            observed_positions = []
            break
        observed_positions.append(position)
    if len(observed_positions) == len(selected):
        position_radii = [float(np.linalg.norm(position)) for position in observed_positions]
        position_source = "native_sensor_observations"
    else:
        position_radii = [float(np.linalg.norm(sample.position)) for sample in selected]
        position_source = "sampled_controls_fallback"
    fixed_step = float(config.get("FixedStepSeconds", 1.0))
    return {
        "seed": int(state.get("EpisodeSeed", 0)),
        "reason": reason,
        "steps": steps,
        "blue_return": float(state.get("BlueReward", 0.0)),
        "red_return": float(state.get("RedReward", 0.0)),
        "placement_decisions": len(samples),
        "sensors_placed": len(selected),
        "selected_catalogue_indices": [sample.catalogue_index for sample in selected],
        "selected_sensor_profiles": list(profile_ids),
        "selected_modalities": dict(modality_counts or {}),
        "mean_position_radius": float(np.mean(position_radii)) if position_radii else 0.0,
        "position_metric_source": position_source,
        "mean_policy_entropy": (
            float(np.mean([sample.entropy for sample in samples])) if samples else 0.0
        ),
        "breach_rate": float(breach),
        "blue_win_rate": float(not breach),
        "red_win_rate": float(breach),
        "detection_rate": len(confirmed) / max(len(targets), 1),
        "mean_detection_latency_seconds_detected_targets": (
            float(np.mean(first_steps) * fixed_step) if first_steps else None
        ),
        "tracking_coverage": tracked_steps / max(len(targets) * steps, 1),
        "invalid_actions": int(local_invalid) + int(state.get("InvalidActionCount", 0)),
        "defence_semantics": (
            "Sustained observed track or survived horizon; not physical interception"
        ),
    }


@dataclass(frozen=True)
class BlueEpisodeRollout:
    samples: tuple[PlacementSample, ...]
    returns_to_go: np.ndarray
    metrics: Mapping[str, Any]
    terminal_state: Mapping[str, Any]


def collect_blue_episode(
    env: Any,
    policy: DynamicPlacementPolicy,
    adapter: PlacementFeatureAdapter,
    red_script: RedActionScript,
    seed: int,
    *,
    deterministic: bool = False,
    pace_seconds: float = 0.0,
) -> BlueEpisodeRollout:
    """Run all TRIAD phases while collecting only trainable Blue decisions.

    Unreal's ``BlueReward`` and ``RedReward`` are cumulative team totals. Each
    Blue decision receives ``final_blue_total - total_before_that_decision``;
    this captures placement costs and terminal delayed credit without summing
    the two Red AEC heads' duplicate team-credit views.
    """
    if pace_seconds < 0.0:
        raise ValueError("pace_seconds cannot be negative")
    env.reset(seed=int(seed))
    if _phase(env) != 1 or getattr(env, "agent_selection", None) != "blue_placement":
        raise RuntimeError("Reset did not enter the Blue placement phase")
    red_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x524544]))
    samples: list[PlacementSample] = []
    totals_before: list[float] = []
    profile_ids: list[str] = []
    modality_counts: dict[str, int] = {}
    invalid_before = int(getattr(env, "invalid_actions", 0))
    max_turns = (
        int(getattr(env, "config", {}).get("EpisodeHorizonSteps", 10_000))
        + int(getattr(env, "config", {}).get("MaximumSensorSites", 256))
        + 16
    )
    movement_step = 0

    for _ in range(max_turns):
        phase = _phase(env)
        if phase == 4 or bool(env.snapshot.get("bTerminal")):
            break
        expected_agent = {1: "blue_placement", 2: "red_deployment", 3: "red_movement"}.get(phase)
        if expected_agent is None or getattr(env, "agent_selection", None) != expected_agent:
            raise RuntimeError("TRIAD phase and AEC agent_selection disagree")

        if phase == 1:
            feature_batch = adapter.extract(env.observe("blue_placement"))
            sample = policy.act(feature_batch, deterministic=deterministic)
            action, sample = blue_environment_action(env, sample)
            totals_before.append(float(env.snapshot.get("BlueReward", 0.0)))
            samples.append(sample)
            if not sample.stop:
                profile_ids.append(_profile_id(env, sample.catalogue_index))
                if adapter.mount_columns:
                    row = sample.option_features[sample.catalogue_index, :adapter.mount_width]
                    for name in ("passive_rf", "search_radar", "electro_optical", "thermal"):
                        if name in adapter.mount_columns:
                            index = adapter.mount_columns.index(name)
                            modality_counts[name] = modality_counts.get(name, 0) + int(row[index] > 0.5)
            env.step(action)
        elif phase == 2:
            env.step(red_script.deployment_action(env, red_rng))
        else:
            env.step(red_script.movement_action(env, movement_step, red_rng))
            movement_step += 1
        if pace_seconds:
            time.sleep(pace_seconds)
    else:
        raise RuntimeError("TRIAD episode exceeded its phase/step safety bound")

    if not bool(env.snapshot.get("bTerminal")) or _phase(env) != 4:
        raise RuntimeError("TRIAD episode ended without a terminal snapshot")
    terminal = copy.deepcopy(env.snapshot)
    final_blue_total = float(terminal.get("BlueReward", 0.0))
    returns_to_go = np.asarray(
        [final_blue_total - value for value in totals_before], dtype=np.float64
    )
    if not samples or not np.isfinite(returns_to_go).all():
        raise RuntimeError("Blue episode did not produce finite placement samples")
    metrics = episode_metrics(
        terminal,
        getattr(env, "config", {}),
        samples=samples,
        profile_ids=profile_ids,
        modality_counts=modality_counts,
        local_invalid=max(0, int(getattr(env, "invalid_actions", 0)) - invalid_before),
    )
    return BlueEpisodeRollout(
        samples=tuple(samples),
        returns_to_go=returns_to_go,
        metrics=metrics,
        terminal_state=terminal,
    )


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"episodes": 0}
    numeric = (
        "blue_return", "red_return", "steps", "sensors_placed", "detection_rate",
        "tracking_coverage", "breach_rate", "blue_win_rate", "invalid_actions",
        "mean_policy_entropy", "mean_position_radius",
    )
    result: dict[str, Any] = {"episodes": len(rows)}
    for name in numeric:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        if values:
            result[f"mean_{name}"] = float(np.mean(values))
    reasons: dict[str, int] = {}
    profiles: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason", "Unknown"))
        reasons[reason] = reasons.get(reason, 0) + 1
        for profile in row.get("selected_sensor_profiles", ()):
            profiles[str(profile)] = profiles.get(str(profile), 0) + 1
    result["termination_reasons"] = reasons
    result["sensor_profile_counts"] = profiles
    return result
