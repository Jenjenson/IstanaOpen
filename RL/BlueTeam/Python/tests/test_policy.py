"""NumPy-only tests for dynamic Blue placement and focused rollouts."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_blue_placement import DryRunTRIADEnv, parse_args, run_training
from evaluate_checkpoint import parse_args as parse_evaluation_args, run_evaluation
from triad_rl.placement_policy import DynamicPlacementPolicy, PlacementFeatureAdapter
from triad_rl.rollout import RedActionScript, collect_blue_episode


def features_for(option_count: int = 7):
    env = DryRunTRIADEnv(option_count=option_count)
    env.reset(seed=11)
    adapter = PlacementFeatureAdapter(env)
    return env, adapter, adapter.extract(env.observe("blue_placement"))


def test_shared_features_support_different_catalogue_lengths_and_modalities():
    _, small_adapter, small = features_for(5)
    _, large_adapter, large = features_for(17)
    assert small_adapter.feature_contract == large_adapter.feature_contract
    assert small.option_features.shape == (6, small_adapter.feature_size)
    assert large.option_features.shape == (18, large_adapter.feature_size)
    assert small.dynamic_position_mask.tolist() == [False, True, True, False, True, False]
    modality_start = small_adapter.mount_columns.index("passive_rf")
    assert small.option_features[:4, modality_start:modality_start + 4].tolist() == np.eye(4).tolist()
    assert small.option_features[-1, :small_adapter.mount_width].tolist() == [0.0] * 22


def test_masked_policy_never_samples_invalid_and_rejects_empty_mask():
    env, adapter, features = features_for()
    observation = env.observe("blue_placement")
    observation["action_mask"][:] = 0
    observation["action_mask"][[2, env.option_count]] = 1
    restricted = adapter.extract(observation)
    policy = DynamicPlacementPolicy.from_adapter(adapter, seed=9)
    assert {policy.act(restricted).catalogue_index for _ in range(200)} <= {2, env.option_count}
    observation["action_mask"][:] = 0
    with pytest.raises(ValueError, match="empty"):
        adapter.extract(observation)


def test_position_head_is_gated_by_named_dynamic_feature():
    env, adapter, _ = features_for()
    policy = DynamicPlacementPolicy.from_adapter(adapter, seed=3)
    observation = env.observe("blue_placement")
    observation["action_mask"][:] = 0
    observation["action_mask"][0] = 1  # fixed option in the dry-run catalogue
    fixed = policy.act(adapter.extract(observation))
    assert not fixed.position_active
    assert fixed.position.tolist() == [0.0, 0.0]
    observation["action_mask"][:] = 0
    observation["action_mask"][1] = 1
    dynamic = policy.act(adapter.extract(observation))
    assert dynamic.position_active
    assert dynamic.position.dtype == np.float32
    assert np.all(np.abs(dynamic.position) <= 1.0)
    assert set(dynamic.environment_action()) == {"catalogue_index", "position"}


def test_reinforce_updates_discrete_and_only_active_position_head():
    env, adapter, _ = features_for()
    policy = DynamicPlacementPolicy.from_adapter(adapter, seed=12)
    observation = env.observe("blue_placement")
    observation["action_mask"][:] = 0
    observation["action_mask"][0] = 1
    fixed = policy.act(adapter.extract(observation))
    position_before = policy.position_weights.copy()
    policy.update([fixed], [2.0], learning_rate=1e-3, entropy_coefficient=0.0)
    assert np.array_equal(position_before, policy.position_weights)

    observation["action_mask"][:] = 0
    observation["action_mask"][1] = 1
    dynamic = policy.act(adapter.extract(observation))
    policy.update([dynamic], [2.0], learning_rate=1e-3, entropy_coefficient=0.0)
    assert not np.array_equal(position_before, policy.position_weights)


def test_checkpoint_roundtrip_is_versioned_integrity_checked_and_catalogue_agnostic(tmp_path):
    _, small_adapter, small = features_for(4)
    _, large_adapter, large = features_for(13)
    policy = DynamicPlacementPolicy.from_adapter(small_adapter, seed=44)
    sample = policy.act(small)
    policy.update([sample], [1.5], learning_rate=1e-3)
    policy.observe_episode_returns([1.5])
    before = policy.deterministic_action(large)
    path = tmp_path / "checkpoint"
    metadata = policy.save_checkpoint(path, details={"episodes": 1})
    restored, loaded = DynamicPlacementPolicy.load_checkpoint(
        path, expected_feature_contract=large_adapter.feature_contract
    )
    after = restored.deterministic_action(large)
    assert metadata["schema"] == "triad.dynamic_placement_checkpoint.v1"
    assert loaded["details"] == {"episodes": 1}
    assert restored.parameter_hash() == policy.parameter_hash()
    assert before.catalogue_index == after.catalogue_index
    assert before.position.tolist() == after.position.tolist()

    arrays = path / "arrays.npz"
    original = arrays.read_bytes()
    arrays.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="hash"):
        DynamicPlacementPolicy.load_checkpoint(path)
    arrays.write_bytes(original)
    assert hashlib.sha256(original).hexdigest() == metadata["arrays_sha256"]

    with np.load(arrays, allow_pickle=False) as archive:
        contents = {name: archive[name].copy() for name in archive.files}
    contents["adam_variance__input_bias"][0] = -1.0
    np.savez_compressed(arrays, **contents)
    checkpoint_json = path / "checkpoint.json"
    tampered_metadata = json.loads(checkpoint_json.read_text())
    tampered_metadata["arrays_sha256"] = hashlib.sha256(arrays.read_bytes()).hexdigest()
    checkpoint_json.write_text(json.dumps(tampered_metadata))
    with pytest.raises(ValueError, match="negative Adam variance"):
        DynamicPlacementPolicy.load_checkpoint(path)


def test_phase_rollout_uses_hybrid_actions_and_delayed_native_team_total():
    env, adapter, features = features_for(6)
    base = DynamicPlacementPolicy.from_adapter(adapter, seed=7)

    class ScriptedBlue:
        def __init__(self):
            self.calls = 0

        def act(self, current, deterministic=False):
            del deterministic
            sample = base.act(current, deterministic=True)
            index = 1 if self.calls == 0 else current.option_count
            self.calls += 1
            return replace(
                sample,
                catalogue_index=index,
                position=np.asarray([0.25, -0.5], dtype=np.float32) if index == 1
                else np.zeros(2, dtype=np.float32),
                position_active=index == 1,
                option_features=current.option_features.copy(),
                action_mask=current.action_mask.copy(),
            )

    rollout = collect_blue_episode(
        env, ScriptedBlue(), adapter, RedActionScript(), seed=123
    )
    assert env.snapshot["bTerminal"] and env.phase(env.snapshot) == 4
    assert [sample.catalogue_index for sample in rollout.samples] == [1, env.option_count]
    assert rollout.samples[0].position.tolist() == pytest.approx([0.25, -0.5])
    assert rollout.returns_to_go[0] == pytest.approx(rollout.metrics["blue_return"])
    assert rollout.returns_to_go[1] == pytest.approx(rollout.metrics["blue_return"] + 0.05)
    assert rollout.metrics["selected_sensor_profiles"] == ["search-radar"]
    assert rollout.metrics["selected_modalities"]["search_radar"] == 1
    assert rollout.metrics["position_metric_source"] == "native_sensor_observations"
    assert rollout.metrics["mean_position_radius"] == pytest.approx(np.hypot(0.25, 0.5))


def test_red_script_is_configurable_seeded_and_dimension_masked():
    env = DryRunTRIADEnv()
    env.reset(seed=1)
    fixed = RedActionScript.from_mapping({
        "deployment": [1, -1, 0.5, 0, 0],
        "movement": [0.5, -0.75, 1],
        "movement_steps": [[-0.5, -1, 1]],
    })
    rng = np.random.default_rng(3)
    assert fixed.deployment_action(env, rng).tolist() == pytest.approx([1, -1, 0.5, 0, 0])
    assert fixed.movement_action(env, 4, rng) == pytest.approx(np.asarray([[-0.5, -1, 0]]))
    uniform = RedActionScript(mode="uniform")
    a = uniform.deployment_action(env, np.random.default_rng(99))
    b = uniform.deployment_action(env, np.random.default_rng(99))
    assert a.tolist() == b.tolist()


def test_cli_dry_run_trains_writes_metrics_and_checkpoint_without_unreal(tmp_path):
    args = parse_args([
        "--dry-run", "--episodes", "2", "--batch-size", "2", "--seed", "22",
        "--checkpoint-dir", str(tmp_path / "checkpoints"),
        "--metrics-path", str(tmp_path / "metrics.jsonl"),
        "--checkpoint-every", "0",
    ])
    policy, result = run_training(args)
    assert result["episodes_added"] == 2
    assert policy.update_count == 1
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 2
    restored, metadata = DynamicPlacementPolicy.load_checkpoint(result["checkpoint"])
    assert restored.parameter_hash() == policy.parameter_hash()
    assert metadata["details"]["trainer"]["dry_run"] is True
    assert metadata["details"]["red_script"]["movement"] == [0.0, -1.0, 0.0]
    assert metadata["details"]["environment"]["config_fingerprint"] == "dry-run-options:8"
    assert policy.return_count >= 2  # one baseline observation per placement decision


def test_trainer_rejects_biased_deterministic_updates_and_changed_resume_script(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(["--dry-run", "--deterministic"])

    first = parse_args([
        "--dry-run", "--episodes", "1", "--batch-size", "1",
        "--checkpoint-dir", str(tmp_path / "first"), "--checkpoint-every", "0",
    ])
    _, result = run_training(first)
    resumed = parse_args([
        "--dry-run", "--episodes", "1", "--batch-size", "1",
        "--resume", result["checkpoint"], "--red-velocity", "1,0,0",
        "--checkpoint-dir", str(tmp_path / "resume"), "--checkpoint-every", "0",
    ])
    with pytest.raises(ValueError, match="Red script differs"):
        run_training(resumed)


def test_evaluation_cli_compares_initial_and_checkpoint_without_updates(tmp_path):
    training = parse_args([
        "--dry-run", "--episodes", "1", "--batch-size", "1",
        "--checkpoint-dir", str(tmp_path / "training"), "--checkpoint-every", "0",
    ])
    _, trained = run_training(training)
    checkpoint_args = parse_evaluation_args([
        "--dry-run", "--checkpoint", trained["checkpoint"], "--episodes", "2",
        "--output", str(tmp_path / "trained_eval"), "--seed", "2000000000",
    ])
    checkpoint_report = run_evaluation(checkpoint_args)
    assert checkpoint_report["schema"] == "triad.dynamic_placement_evaluation.v1"
    assert checkpoint_report["training_performed"] is False
    assert checkpoint_report["parameter_sha256_before"] == checkpoint_report["parameter_sha256_after"]
    assert checkpoint_report["environment"]["config_fingerprint"] == "dry-run-options:8"
    assert checkpoint_report["summary"]["episodes"] == 2
    assert (tmp_path / "trained_eval" / "evaluation.json").is_file()

    initial_args = parse_evaluation_args([
        "--dry-run", "--initial", "--stochastic", "--episodes", "1",
        "--output", str(tmp_path / "initial_eval"), "--seed", "2000000000",
    ])
    initial_report = run_evaluation(initial_args)
    assert initial_report["source"]["kind"] == "initial"
    assert initial_report["deterministic"] is False
