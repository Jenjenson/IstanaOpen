"""Short real credit-pilot arms, exact updates and fail-before-write guards."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from triad_rl import train_credit_pilot as trainer
from triad_rl.balanced_policy import BalancedPolicy


SOURCE = trainer.BLUE_ROOT / "Checkpoints" / "balanced-v3-candidate"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tree(path):
    return {str(file.relative_to(path)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in path.rglob("*") if file.is_file()}


def protocol(path, **training):
    value = {"schema": trainer.PROTOCOL_SCHEMA,
             "arms": deepcopy(trainer.ARMS),
             "training": {"policy_seed": 499, "scenario_seed_start": 499_000_000,
                          "episodes": 4, "batch_size": 2, "learning_rate": .002,
                          "entropy_coef": .015, "gamma": 1., "max_grad_norm": 1.,
                          "profile": "mixed", **training},
             "frozen_reference": {"checkpoint": "Checkpoints/balanced-v3-candidate",
                                  "selection_report": "Results/balanced-v3/selection.json",
                                  "publication_manifest": "Results/balanced-v3/artifact-manifest.json",
                                  "selection_report_sha256": trainer._file_hash(trainer.BLUE_ROOT / "Results/balanced-v3/selection.json"),
                                  "publication_manifest_sha256": trainer._file_hash(trainer.BLUE_ROOT / "Results/balanced-v3/artifact-manifest.json"),
                                  "weights_sha256": BalancedPolicy.load(SOURCE).weights_fingerprint()},
             "training_implementation_sha256": trainer.source_provenance(),
             "test_only": True}
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("credit-arms")
    declaration = protocol(root / "protocol.json")
    original = tree(SOURCE)
    summaries = {arm: trainer.run_training(declaration, arm, root / arm) for arm in trainer.ARMS}
    assert tree(SOURCE) == original
    return root, declaration, summaries


def events(path):
    return [json.loads(line) for line in (path / "training.jsonl").read_text().splitlines()]


def same_policy(a, b):
    assert a.weights_fingerprint() == b.weights_fingerprint()
    assert a.update_count == b.update_count
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
    for group in ("parameters", "adam_m", "adam_v"):
        for key, array in getattr(a, group).items():
            np.testing.assert_array_equal(array, getattr(b, group)[key])


def test_all_arms_copy_exact_frozen_v3_parameters_zero_adam_and_same_fresh_rng(runs):
    root, _, summaries = runs
    base = BalancedPolicy.load(SOURCE)
    initials = [BalancedPolicy.load(root / arm / "initialized") for arm in trainer.ARMS]
    for initial in initials:
        assert initial.weights_fingerprint() == base.weights_fingerprint()
        assert initial.update_count == 0
        assert initial.rng.bit_generator.state == np.random.default_rng(499).bit_generator.state
        for key in base.parameters:
            np.testing.assert_array_equal(initial.parameters[key], base.parameters[key])
            assert not initial.adam_m[key].any() and not initial.adam_v[key].any()
    for other in initials[1:]:
        same_policy(initials[0], other)
    assert all(summary["last_weights_sha256"] != base.weights_fingerprint() for summary in summaries.values())
    assert len({summary["first_batch_trajectory_sha256"] for summary in summaries.values()}) == 1
    assert len({events(root / arm)[1]["trajectory_sha256"] for arm in trainer.ARMS}) == 1


@pytest.mark.parametrize("arm", list(trainer.ARMS))
def test_real_run_matches_manual_unchanged_policy_updates_exactly(runs, arm):
    root, declaration, _ = runs
    config = read(declaration)
    train = config["training"]
    policy = BalancedPolicy.load(root / arm / "initialized")
    env = trainer.RobustPlacementEnv(seed=train["scenario_seed_start"], profile="mixed")
    for begin in range(0, 4, 2):
        batch = []
        for offset in range(begin, begin + 2):
            if arm == "paired_stop":
                transitions, _ = trainer.counterfactual_rollout(env, policy, 499_000_000 + offset)
            else:
                transitions, _ = trainer.rollout(env, policy, 499_000_000 + offset, training=True)
            batch.append(transitions)
        policy.update(batch, learning_rate=.002, entropy_coef=.015, gamma=1.,
                      value_coef=trainer.ARMS[arm]["value_coef"], max_grad_norm=1.)
    same_policy(policy, BalancedPolicy.load(root / arm / "last"))


def test_zero_value_gradient_arms_freeze_critic_head_but_not_actor_encoder(runs):
    root, _, _ = runs
    for arm in ("no_critic_gradient", "paired_stop"):
        initial, last = (BalancedPolicy.load(root / arm / name) for name in ("initialized", "last"))
        for name in ("wv", "bv"):
            np.testing.assert_array_equal(initial.parameters[name], last.parameters[name])
            assert not last.adam_m[name].any() and not last.adam_v[name].any()
        assert not np.array_equal(initial.parameters["w1"], last.parameters["w1"])
        assert all(row["critic_head_delta_l2"] == 0 for row in events(root / arm)[1:])


def test_real_logs_bind_fixed_endpoint_sources_seeds_and_honest_extra_compute(runs):
    root, declaration, summaries = runs
    for arm, summary in summaries.items():
        directory = root / arm
        assert (directory / "protocol.json").read_bytes() == declaration.read_bytes()
        config, state = read(directory / "config.json"), BalancedPolicy.load(directory / "last").training_state
        assert config["source_sha256"] == trainer.source_provenance()
        assert config["protocol_sha256"] == hashlib.sha256(declaration.read_bytes()).hexdigest()
        assert summary["config_sha256"] == state["config_sha256"] == config["config_sha256"]
        assert summary["completed_episodes"] == 4 and summary["optimizer_updates"] == 2
        assert state["completed_episodes"] == state["target_episodes"] == 4
        assert state["seed_provenance"]["training"] == {"start": 499_000_000, "count": 4}
        assert summary["last_files_sha256"] == trainer._checkpoint_files(directory / "last")
        assert summary["initialized_files_sha256"] == trainer._checkpoint_files(directory / "initialized")
        assert summary["log_sha256"] == state["log_sha256"] == trainer._file_hash(directory / "training.jsonl")
        assert not (directory / "best").exists()
        rows = events(directory)
        assert [row["event"] for row in rows] == ["initialization", "training_batch", "training_batch"]
        assert [(row["episode_start"], row["episode"]) for row in rows[1:]] == [(0, 2), (2, 4)]
        assert summary["compute"]["live_episode_rollouts"] == 4
        decisions = sum(row["transitions"] for row in rows[1:])
        assert summary["compute"]["extra_branch_rollouts"] == (decisions if arm == "paired_stop" else 0)
        assert summary["compute"]["live_decisions"] == decisions
        assert summary["credit"]["first"]["return_to_go"]["count"] == 4
        assert summary["credit"]["all"]["return_to_go"]["count"] == decisions
        assert summary["mean_invalid_actions"] == 0
        for key in ("mean_return", "success_rate", "detection_rate", "coverage", "breach_rate", "mean_cost", "mean_steps"):
            assert summary[key] == pytest.approx(np.mean([row[key] for row in rows[1:]]))
        assert summary["credit"]["first_deploy"]["baseline"]["count"] + summary["credit"]["first_stop"]["baseline"]["count"] == 4
        if arm == "paired_stop":
            assert summary["credit"]["first"]["baseline"]["mean"] == -10.5
            assert summary["credit"]["first"]["baseline"]["variance"] == 0
        for row in rows[1:]:
            assert row["diagnostic_total_gradient_norm"] == row["gradient_norm"]
            assert row["clipped_gradient_norm"] <= 1. + 1e-12
            assert sum(row["case_profile_counts"].values()) == 2
            assert row["compute"]["rollout_wall_seconds"] > 0
            assert row["compute"]["gradient_audit_wall_seconds"] > 0
            assert row["weighted_critic_gradient_norm"] == 0 if arm != "shared_critic" else row["weighted_critic_gradient_norm"] > 0
            if arm != "shared_critic":
                assert row["actor_critic_gradient_dot"] == 0 and row["actor_critic_gradient_cosine"] is None
            else:
                assert abs(row["actor_critic_gradient_cosine"]) <= 1


def record(reward, baseline, stop=False):
    features = np.zeros((1, len(trainer.FEATURE_NAMES)))
    features[0, trainer.FEATURE_NAMES.index("stop")] = float(stop)
    return {"features": features, "action": 0, "reward": reward, "value": baseline}


def test_credit_diagnostics_match_global_normalization_and_stop_caveat():
    batch = [[record(2., 1.), record(3., 3., True)], [record(4., 2., True)]]
    _, returns, normalized, groups = trainer._credit_arrays(batch)
    np.testing.assert_array_equal(returns, [5., 3., 4.])
    expected = np.array([4., 0., 2.])
    np.testing.assert_array_equal(normalized, (expected - expected.mean()) / (expected.std() + 1e-8))
    assert groups["later_stop"][0, 2] == 0 and groups["later_stop"][0, 3] != 0
    stats = trainer._credit_summary(groups)
    assert stats["first"]["baseline"]["count"] == 2
    assert stats["later"]["return_to_go"]["mean"] == 3.
    assert stats["all"]["raw_advantage"]["variance"] == np.var(expected)


@pytest.mark.parametrize("batch", [[[record(2., 1.)]], [[record(2., 1.)], [record(2., 1.)]]])
def test_degenerate_advantages_remain_unnormalized_exactly_like_frozen_update(batch):
    _, _, advantages, groups = trainer._credit_arrays(batch)
    np.testing.assert_array_equal(advantages, groups["all"][:, 2])


@pytest.mark.parametrize("change", ["schema", "source", "reference", "weights", "arms", "episodes", "batch_size",
                                   "gamma", "learning_rate", "entropy_coef", "max_grad_norm", "profile", "seed", "range"])
def test_invalid_predeclaration_fails_before_output(tmp_path, change):
    declaration = protocol(tmp_path / "protocol.json")
    value = read(declaration)
    if change == "schema": value["schema"] = "other"
    elif change == "source": value["training_implementation_sha256"]["triad_rl/train_credit_pilot.py"] = "0" * 64
    elif change == "reference": value["frozen_reference"]["checkpoint"] = "../escape"
    elif change == "weights": value["frozen_reference"]["weights_sha256"] = "0" * 64
    elif change == "arms": value["arms"].pop("paired_stop")
    elif change == "episodes": value["training"]["episodes"] = 1025
    elif change == "batch_size": value["training"]["batch_size"] = 3
    elif change == "seed": value["training"]["policy_seed"] = True
    elif change == "range": value["training"]["scenario_seed_start"] = 2 * 10**15
    elif change == "profile": value["training"]["profile"] = "normal"
    else: value["training"][change] = .123
    declaration.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        trainer.run_training(declaration, "shared_critic", tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()


@pytest.mark.parametrize("name", ["unknown", "", None])
def test_unknown_arm_fails_without_output(tmp_path, name):
    declaration = protocol(tmp_path / "protocol.json")
    with pytest.raises(ValueError):
        trainer.run_training(declaration, name, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_reused_output_is_never_overwritten(runs):
    root, declaration, _ = runs
    before = tree(root / "shared_critic")
    with pytest.raises(ValueError, match="new or empty"):
        trainer.run_training(declaration, "shared_critic", root / "shared_critic")
    assert tree(root / "shared_critic") == before


def test_source_drift_during_preflight_leaves_no_output(tmp_path, monkeypatch):
    declaration = protocol(tmp_path / "protocol.json")
    original, calls = trainer.source_provenance, 0
    def changed():
        nonlocal calls
        calls += 1
        result = original()
        if calls >= 3:
            result["triad_rl/train_credit_pilot.py"] = "changed"
        return result
    monkeypatch.setattr(trainer, "source_provenance", changed)
    with pytest.raises(ValueError, match="changed"):
        trainer.run_training(declaration, "shared_critic", tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_protocol_drift_during_preflight_leaves_no_output(tmp_path, monkeypatch):
    declaration = protocol(tmp_path / "protocol.json")
    original = trainer.load_published_lineage
    def changed():
        result = original()
        declaration.write_bytes(declaration.read_bytes() + b" ")
        return result
    monkeypatch.setattr(trainer, "load_published_lineage", changed)
    with pytest.raises(ValueError, match="changed"):
        trainer.run_training(declaration, "shared_critic", tmp_path / "run")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("token", ['"schema":"duplicate",', '"overflow":1e999,'])
def test_strict_protocol_json_rejects_duplicates_and_overflow(tmp_path, token):
    declaration = protocol(tmp_path / "protocol.json")
    declaration.write_text("{" + token + declaration.read_text()[1:], encoding="utf-8")
    with pytest.raises(ValueError):
        trainer.run_training(declaration, "shared_critic", tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_baseline_independent_trajectory_digest_changes_for_actions_not_values():
    first = record(2., 1.)
    first["mask"] = np.array([True])
    batch = [[first]]
    original = trainer._trajectory_digest(batch, 499_000_000)
    first["value"] = -10.5
    assert trainer._trajectory_digest(batch, 499_000_000) == original
    first["reward"] += .01
    assert trainer._trajectory_digest(batch, 499_000_000) != original


def test_boolean_value_coefficient_is_not_a_valid_zero_arm(tmp_path):
    declaration = protocol(tmp_path / "protocol.json")
    value = read(declaration)
    value["arms"]["paired_stop"]["value_coef"] = False
    declaration.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="three fixed"):
        trainer.run_training(declaration, "shared_critic", tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_source_selection_hash_mismatch_fails_before_writes(tmp_path):
    declaration = protocol(tmp_path / "protocol.json")
    value = read(declaration)
    value["frozen_reference"]["selection_report_sha256"] = "0" * 64
    declaration.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="Initial weights"):
        trainer.run_training(declaration, "shared_critic", tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_all_inherited_manifest_bindings_checked_before_writes(tmp_path, monkeypatch):
    declaration = protocol(tmp_path / "protocol.json")
    original = trainer.load_published_lineage
    def changed():
        result = original()
        result["publication_manifests"][0]["sha256"] = "0" * 64
        return result
    monkeypatch.setattr(trainer, "load_published_lineage", changed)
    with pytest.raises(ValueError, match="changed"):
        trainer.run_training(declaration, "shared_critic", tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_initialized_checkpoint_tamper_prevents_endpoint_write(tmp_path, monkeypatch):
    declaration = protocol(tmp_path / "protocol.json", episodes=2)
    original, calls = trainer.load_published_lineage, 0
    def changed():
        nonlocal calls
        calls += 1
        result = original()
        if calls == 2:
            metadata = tmp_path / "run" / "initialized" / "checkpoint.json"
            metadata.write_bytes(metadata.read_bytes() + b" ")
        return result
    monkeypatch.setattr(trainer, "load_published_lineage", changed)
    with pytest.raises(ValueError, match="Immutable initialized"):
        trainer.run_training(declaration, "shared_critic", tmp_path / "run")
    assert not (tmp_path / "run" / "last").exists()
    assert not (tmp_path / "run" / "summary.json").exists()
