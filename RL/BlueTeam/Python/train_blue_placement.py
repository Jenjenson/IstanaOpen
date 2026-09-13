"""Train the dynamic Blue sensor-placement policy against scripted Red actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from gymnasium import spaces

from triad_rl import TRIADRedBlueEnv, TRIADRemoteControlClient
from triad_rl.placement_policy import DynamicPlacementPolicy, PlacementFeatureAdapter
from triad_rl.rollout import RedActionScript, aggregate_metrics, collect_blue_episode


MOUNT_FEATURE_NAMES = (
    "offset_east", "offset_north", "offset_up",
    "normal_east", "normal_north", "normal_up",
    "yaw_cos", "yaw_sin", "pitch", "horizontal_fov", "vertical_fov",
    "range", "cost", "passive_rf", "search_radar", "electro_optical",
    "thermal", "allow_dynamic_position", "surface_resolved", "occupied",
    "enabled", "currently_detecting",
)


def _vector_argument(text: str, size: int, label: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value.strip()) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated numbers") from error
    if len(values) != size or not np.isfinite(values).all() or any(abs(value) > 1 for value in values):
        raise argparse.ArgumentTypeError(f"{label} requires {size} finite values in [-1, 1]")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_config = Path(__file__).resolve().parents[1] / "DefaultTrainingConfig.json"
    parser = argparse.ArgumentParser(
        description=(
            "Train only TRIAD's dynamic Blue sensor-placement policy; Red follows a "
            "fixed JSON script or a reproducible seeded-uniform baseline"
        )
    )
    parser.add_argument("--object-path", help="PIE object path of TRIAD_AdversarialTraining_Manager")
    parser.add_argument("--endpoint", default="http://127.0.0.1:30010")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--episodes", type=int, default=100, help="Episodes to add in this run")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("blue_checkpoints"))
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--metrics-path", type=Path)
    parser.add_argument("--red-script", type=Path, help="JSON RedActionScript configuration")
    parser.add_argument("--red-mode", choices=("fixed", "uniform"), default="fixed")
    parser.add_argument(
        "--red-deployment", default="0,0,0,0,0",
        help="Five comma-separated normalized deployment controls",
    )
    parser.add_argument(
        "--red-velocity", default="0,-1,0",
        help="Three comma-separated normalized ENU movement controls",
    )
    parser.add_argument("--deterministic", action="store_true", help="Evaluate greedy actions")
    parser.add_argument("--no-update", action="store_true", help="Collect metrics without learning")
    parser.add_argument("--pace-seconds", type=float, default=0.0)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run the complete trainer against a local deterministic toy, without Unreal",
    )
    parser.add_argument("--dry-run-options", type=int, default=8)
    args = parser.parse_args(argv)
    if not args.dry_run and not args.object_path:
        parser.error("--object-path is required unless --dry-run is used")
    if args.episodes < 1 or args.batch_size < 1:
        parser.error("--episodes and --batch-size must be positive")
    if not 0 <= args.seed < 2**31:
        parser.error("--seed must be in [0, 2^31)")
    if not 1 <= args.hidden_size <= 1024:
        parser.error("--hidden-size must be in [1, 1024]")
    if not 0.0 < args.learning_rate <= 1.0 or args.entropy_coefficient < 0.0:
        parser.error("invalid optimizer settings")
    if args.checkpoint_every < 0 or args.pace_seconds < 0.0:
        parser.error("checkpoint interval and pacing cannot be negative")
    if args.deterministic and not args.no_update:
        parser.error("--deterministic is evaluation-only and requires --no-update")
    if not 2 <= args.dry_run_options <= 256:
        parser.error("--dry-run-options must be in [2, 256]")
    try:
        args.red_deployment = _vector_argument(args.red_deployment, 5, "--red-deployment")
        args.red_velocity = _vector_argument(args.red_velocity, 3, "--red-velocity")
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    return args


def episode_seed(master_seed: int, episode_index: int) -> int:
    """Stateless derivation makes episode seeds identical after checkpoint resume."""
    if episode_index < 0:
        raise ValueError("episode_index cannot be negative")
    sequence = np.random.SeedSequence([int(master_seed), int(episode_index), 0x54524941])
    return int(sequence.generate_state(1, dtype=np.uint32)[0] & np.uint32(0x7FFFFFFF))


class DryRunTRIADEnv:
    """Tiny phase-accurate environment for CLI and CI smoke runs.

    It exercises changing modality rows and hybrid positions; it is explicitly
    not an Unreal-fidelity or learning-quality claim.
    """

    observation_schema = "triad.policy_observation.v4"
    possible_agents = ["blue_placement", "red_deployment", "red_movement"]
    mount_feature_names = MOUNT_FEATURE_NAMES

    def __init__(self, option_count: int = 8, maximum_sites: int = 2) -> None:
        if not 2 <= option_count <= 256:
            raise ValueError("Dry-run catalogue size must be in [2, 256]")
        self.option_count = int(option_count)
        self.max_targets = 1
        self.maximum_sites = min(int(maximum_sites), self.option_count)
        self.config = {
            "EpisodeHorizonSteps": 3,
            "MaximumSensorSites": self.maximum_sites,
            "FixedStepSeconds": 0.5,
        }
        modality_names = ("passive-rf", "search-radar", "electro-optical", "thermal")
        self.catalogue = [
            {"CandidateId": f"dry-{index}", "SensorProfileId": modality_names[index % 4]}
            for index in range(self.option_count)
        ]
        self._action_spaces = {
            "blue_placement": spaces.Dict({
                "catalogue_index": spaces.Discrete(self.option_count + 1),
                "position": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            }),
            "red_deployment": spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32),
            "red_movement": spaces.Box(-1.0, 1.0, shape=(1, 3), dtype=np.float32),
        }
        shapes = {
            "mounts": (self.option_count, len(MOUNT_FEATURE_NAMES)),
            "placements": (self.maximum_sites, 9),
            "budget": (2,),
            "context": (7,),
        }
        self.observation_fields = {"blue_placement": {}}
        offset = 0
        for name, shape in shapes.items():
            size = int(np.prod(shape))
            field: dict[str, Any] = {"start": offset, "stop": offset + size, "shape": list(shape)}
            if name == "mounts":
                field["columns"] = list(MOUNT_FEATURE_NAMES)
            if name == "placements":
                field["columns"] = [
                    "east", "north", "catalogue_id", "passive_rf", "search_radar",
                    "electro_optical", "thermal", "cost", "active",
                ]
            self.observation_fields["blue_placement"][name] = field
            offset += size
        self.invalid_actions = 0
        self.snapshot: dict[str, Any] = {}
        self.agent_selection = "blue_placement"
        self._rng = np.random.default_rng(0)
        self._mounts = np.zeros(shapes["mounts"], dtype=np.float32)
        for index, angle in enumerate(np.linspace(0.0, 2.0 * np.pi, self.option_count, endpoint=False)):
            modality = np.zeros(4, dtype=np.float32)
            modality[index % 4] = 1.0
            self._mounts[index] = [
                np.cos(angle), np.sin(angle), 0.05,
                0.0, 0.0, 1.0, np.cos(angle), np.sin(angle), 0.0,
                0.25, 0.5, 0.8, 0.25 + 0.1 * (index % 3),
                *modality, float(index % 3 != 0), 1.0, 0.0, 1.0, 0.0,
            ]
        self._placements = np.zeros(shapes["placements"], dtype=np.float32)
        self._used = np.zeros(self.option_count, dtype=np.bool_)
        self._target = np.zeros(2, dtype=np.float32)
        self._required_modality = 0

    @staticmethod
    def phase(state: Mapping[str, Any]) -> int:
        value = str(state.get("Phase", "Inactive")).replace(" ", "")
        return {"Inactive": 0, "BluePlacement": 1, "RedDeployment": 2,
                "RedMovement": 3, "Terminal": 4}[value]

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def action_dimension_mask(self, agent: str) -> np.ndarray:
        if agent == "red_deployment":
            return np.ones(5, dtype=np.float32)
        if agent == "red_movement":
            return np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32)
        raise ValueError("Blue uses a discrete catalogue mask")

    def reset(self, seed: int | None = None, options: Any = None) -> None:
        del options
        used_seed = int(seed or 0)
        self._rng = np.random.default_rng(used_seed)
        angle = float(self._rng.uniform(-np.pi, np.pi))
        self._target[:] = [np.cos(angle), np.sin(angle)]
        self._required_modality = int(self._rng.integers(0, 4))
        self._used[:] = False
        self._placements[:] = 0.0
        self._mounts[:, MOUNT_FEATURE_NAMES.index("occupied")] = 0.0
        self.snapshot = {
            "Phase": "BluePlacement", "EpisodeSeed": used_seed, "StepIndex": 0,
            "BlueReward": 0.0, "RedReward": 0.0, "bTerminal": False,
            "TerminationReason": "None", "TargetObservations": [],
            "PlacedCatalogueIndices": [], "SensorObservations": [], "InvalidActionCount": 0,
            "ActiveTargetCount": 0,
        }
        self.agent_selection = "blue_placement"
        self.invalid_actions = 0

    def _mask(self) -> np.ndarray:
        blue = self.snapshot["Phase"] == "BluePlacement"
        dynamic = self._mounts[:, MOUNT_FEATURE_NAMES.index("allow_dynamic_position")] > 0.5
        candidate = ((~self._used) | dynamic) & blue
        return np.concatenate([candidate, np.asarray([blue], dtype=np.bool_)])

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        if agent != "blue_placement":
            raise ValueError("Dry-run trainer observes only Blue")
        placed = len(self.snapshot["PlacedCatalogueIndices"])
        budget = np.asarray([
            (self.maximum_sites - placed) / self.maximum_sites,
            (self.maximum_sites - placed) / self.maximum_sites,
        ], dtype=np.float32)
        context = np.asarray([
            self.phase(self.snapshot) / 4.0,
            self.snapshot["StepIndex"] / self.config["EpisodeHorizonSteps"],
            self._target[0], self._target[1],
            self._required_modality / 3.0,
            placed / self.maximum_sites,
            1.0,
        ], dtype=np.float32)
        values = {
            "mounts": self._mounts,
            "placements": self._placements,
            "budget": budget,
            "context": context,
        }
        vector = np.concatenate([
            np.asarray(values[name], dtype=np.float32).ravel()
            for name in self.observation_fields[agent]
        ])
        return {"observation": vector, "action_mask": self._mask().astype(np.int8)}

    def _finish(self) -> None:
        required_column = 13 + self._required_modality
        scores = []
        for row, index in zip(self._placements, self.snapshot["PlacedCatalogueIndices"]):
            if not row[-1]:
                continue
            modality_match = float(self._mounts[index, required_column] > 0.5)
            distance = min(float(np.linalg.norm(row[:2] - self._target)), 2.0)
            scores.append(2.0 * modality_match + 1.0 - distance / 2.0)
        defended = bool(scores and max(scores) >= 2.25)
        terminal_reward = 10.0 if defended else -10.0
        self.snapshot["BlueReward"] += terminal_reward
        self.snapshot["RedReward"] = -self.snapshot["BlueReward"]
        self.snapshot["Phase"] = "Terminal"
        self.snapshot["bTerminal"] = True
        self.snapshot["TerminationReason"] = (
            "SustainedTrackDefence" if defended else "ProtectedZoneReached"
        )
        detected = defended
        self.snapshot["TargetObservations"] = [{
            "bConfirmedDetected": detected,
            "FirstDetectionStep": 1 if detected else -1,
            "TrackedStepCount": self.snapshot["StepIndex"] if detected else 0,
        }]

    def step(self, action: Any) -> None:
        phase = self.phase(self.snapshot)
        if phase == 1:
            if not self._action_spaces["blue_placement"].contains(action):
                self.invalid_actions += 1
                raise ValueError("Invalid dry-run Blue action")
            index = int(action["catalogue_index"])
            if not self._mask()[index]:
                self.invalid_actions += 1
                raise ValueError("Masked dry-run Blue action")
            if index == self.option_count:
                self.snapshot["Phase"] = "RedDeployment"
                self.agent_selection = "red_deployment"
                return
            site = len(self.snapshot["PlacedCatalogueIndices"])
            dynamic = self._mounts[
                index, MOUNT_FEATURE_NAMES.index("allow_dynamic_position")
            ] > 0.5
            position = (
                np.asarray(action["position"], dtype=np.float32)
                if dynamic else self._mounts[index, :2].copy()
            )
            modality = self._mounts[index, 13:17]
            self._placements[site] = [
                position[0], position[1], index / max(self.option_count - 1, 1),
                *modality, self._mounts[index, 12], 1.0,
            ]
            self.snapshot["PlacedCatalogueIndices"].append(index)
            self.snapshot["SensorObservations"].append(
                {"X": float(position[0]), "Y": float(position[1])}
            )
            self.snapshot["BlueReward"] -= 0.05
            self._used[index] = True
            self._mounts[index, MOUNT_FEATURE_NAMES.index("occupied")] = 1.0
            if site + 1 >= self.maximum_sites:
                self.snapshot["Phase"] = "RedDeployment"
                self.agent_selection = "red_deployment"
        elif phase == 2:
            if not self._action_spaces["red_deployment"].contains(action):
                raise ValueError("Invalid dry-run Red deployment action")
            self.snapshot["Phase"] = "RedMovement"
            self.snapshot["ActiveTargetCount"] = 1
            self.agent_selection = "red_movement"
        elif phase == 3:
            if not self._action_spaces["red_movement"].contains(action):
                raise ValueError("Invalid dry-run Red movement action")
            self.snapshot["StepIndex"] += 1
            if self.snapshot["StepIndex"] >= self.config["EpisodeHorizonSteps"]:
                self._finish()
        else:
            raise RuntimeError("Cannot step a terminal dry-run episode")


def _red_script(args: argparse.Namespace) -> RedActionScript:
    if args.red_script:
        return RedActionScript.from_json(args.red_script)
    return RedActionScript(
        mode=args.red_mode,
        deployment=tuple(args.red_deployment),
        movement=tuple(args.red_velocity),
    )


def _red_script_record(red_script: RedActionScript) -> dict[str, Any]:
    return {
        "mode": red_script.mode,
        "deployment": list(red_script.deployment),
        "movement": list(red_script.movement),
        "movement_steps": [list(row) for row in red_script.movement_steps],
    }


def _environment_record(args: argparse.Namespace, env: Any) -> dict[str, Any]:
    return {
        "observation_schema": str(getattr(env, "observation_schema", "unknown")),
        "config_fingerprint": getattr(
            env, "config_fingerprint", f"dry-run-options:{getattr(env, 'option_count', 0)}"
        ),
        "config_path": None if args.dry_run else str(args.config.resolve()),
    }


def _checkpoint_details(
    args: argparse.Namespace,
    env: Any,
    red_script: RedActionScript,
    completed: int,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "trainer": {
            "name": "train_blue_placement",
            "master_seed": args.seed,
            "episodes_completed": completed,
            "dry_run": bool(args.dry_run),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "entropy_coefficient": args.entropy_coefficient,
            "deterministic_collection": bool(args.deterministic),
            "updates_enabled": not bool(args.no_update),
        },
        "environment": _environment_record(args, env),
        "red_script": _red_script_record(red_script),
        "metrics": dict(summary),
    }


def run_training(args: argparse.Namespace) -> tuple[DynamicPlacementPolicy, dict[str, Any]]:
    if args.dry_run:
        env: Any = DryRunTRIADEnv(args.dry_run_options)
    else:
        client = TRIADRemoteControlClient(object_path=args.object_path, endpoint=args.endpoint)
        env = TRIADRedBlueEnv(client, args.config)

    # A reset is required to inspect the feature layout. The first rollout uses
    # its own derived seed and performs a fresh reset before any action.
    env.reset(seed=episode_seed(args.seed, 0))
    adapter = PlacementFeatureAdapter(env)
    adapter.extract(env.observe("blue_placement"))
    red_script = _red_script(args)
    script_record = _red_script_record(red_script)
    environment_record = _environment_record(args, env)
    completed_before = 0
    if args.resume:
        policy, metadata = DynamicPlacementPolicy.load_checkpoint(
            args.resume, expected_feature_contract=adapter.feature_contract
        )
        trainer = metadata.get("details", {}).get("trainer", {})
        completed_before = int(trainer.get("episodes_completed", 0))
        saved_seed = trainer.get("master_seed")
        if saved_seed is not None and int(saved_seed) != args.seed:
            raise ValueError("--seed differs from the resumed trainer checkpoint")
        saved_script = metadata.get("details", {}).get("red_script")
        if saved_script is not None and saved_script != script_record:
            raise ValueError("Red script differs from the resumed trainer checkpoint")
        saved_environment = metadata.get("details", {}).get("environment", {})
        saved_fingerprint = saved_environment.get("config_fingerprint")
        if (saved_fingerprint is not None
                and saved_fingerprint != environment_record["config_fingerprint"]):
            raise ValueError("Environment config differs from the resumed trainer checkpoint")
        expected_settings = {
            "dry_run": bool(args.dry_run),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "entropy_coefficient": args.entropy_coefficient,
            "deterministic_collection": bool(args.deterministic),
            "updates_enabled": not bool(args.no_update),
        }
        for name, current in expected_settings.items():
            if name in trainer and trainer[name] != current:
                raise ValueError(
                    f"{name} differs from the resumed trainer checkpoint"
                )
    else:
        policy = DynamicPlacementPolicy.from_adapter(
            adapter, hidden_size=args.hidden_size, seed=args.seed
        )

    metrics_path = args.metrics_path or args.checkpoint_dir / "training_metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[Mapping[str, Any]] = []
    added = 0
    while added < args.episodes:
        batch_rollouts = []
        batch_size = min(args.batch_size, args.episodes - added)
        for _ in range(batch_size):
            absolute_index = completed_before + added
            seed = episode_seed(args.seed, absolute_index)
            rollout = collect_blue_episode(
                env, policy, adapter, red_script, seed,
                deterministic=args.deterministic, pace_seconds=args.pace_seconds,
            )
            row = {"episode": absolute_index + 1, **dict(rollout.metrics)}
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            print(json.dumps({"kind": "episode", **row}, allow_nan=False))
            all_metrics.append(row)
            batch_rollouts.append(rollout)
            added += 1

        update: Mapping[str, Any] = {"skipped": "--no-update"}
        if not args.no_update:
            samples = [sample for rollout in batch_rollouts for sample in rollout.samples]
            returns = np.concatenate([rollout.returns_to_go for rollout in batch_rollouts])
            update = policy.update(
                samples,
                policy.advantages(returns),
                learning_rate=args.learning_rate,
                entropy_coefficient=args.entropy_coefficient,
            )
            policy.observe_episode_returns(returns)
        completed = completed_before + added
        batch_summary = aggregate_metrics([rollout.metrics for rollout in batch_rollouts])
        print(json.dumps({"kind": "update", "completed": completed,
                          "optimizer": dict(update), "metrics": batch_summary}, allow_nan=False))
        if args.checkpoint_every and completed % args.checkpoint_every == 0:
            path = args.checkpoint_dir / f"episode_{completed:06d}"
            policy.save_checkpoint(
                path,
                details=_checkpoint_details(
                    args, env, red_script, completed, aggregate_metrics(all_metrics)
                ),
            )
            print(json.dumps({"kind": "checkpoint", "path": str(path.resolve())}))

    completed = completed_before + added
    summary = aggregate_metrics(all_metrics)
    final_path = args.checkpoint_dir / f"episode_{completed:06d}_final"
    policy.save_checkpoint(
        final_path, details=_checkpoint_details(args, env, red_script, completed, summary)
    )
    result = {
        "episodes_added": added,
        "episodes_completed": completed,
        "checkpoint": str(final_path.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "summary": summary,
        "parameter_sha256": policy.parameter_hash(),
        "config_fingerprint": environment_record["config_fingerprint"],
    }
    print(json.dumps({"kind": "complete", **result}, allow_nan=False))
    return policy, result


def main(argv: Sequence[str] | None = None) -> None:
    run_training(parse_args(argv))


if __name__ == "__main__":
    main()
