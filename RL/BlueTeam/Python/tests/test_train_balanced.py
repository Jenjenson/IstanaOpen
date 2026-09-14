"""Real balanced-policy training, complete transfer lineage and exact resume."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from triad_rl import train_balanced as trainer
from triad_rl.adaptive_inputs import LiveObservationAdapter
from triad_rl.adaptive_policy import AdaptivePolicy
from triad_rl.balanced_policy import BalancedPolicy, POLICY_SCHEMA
from triad_rl.robust_scenarios import RobustPlacementEnv


PACKAGE = Path(__file__).resolve().parents[2]
PUBLISHED = PACKAGE / "Checkpoints" / "robust-v2-candidate"
SELECTION = PACKAGE / "Results" / "robust-v2" / "selection.json"
OPTIONS = {"batch_size": 2, "seed": 230, "validation_every": 4,
           "validation_episodes": 2, "validation_run_seed": 992300}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def tree_hashes(path):
    return {str(file.relative_to(path)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in path.rglob("*") if file.is_file()}


def events(path):
    return [json.loads(line) for line in (path / "training.jsonl").read_text().splitlines()]


def run(output, episodes=2, **kwargs):
    options = {**OPTIONS, **kwargs}
    if options.get("resume") is None:
        options.setdefault("initial_checkpoint", PUBLISHED)
        options.setdefault("initial_selection_report", SELECTION)
    return trainer.run_training(output=output, episodes=episodes, **options)


def assert_same_policy(left, right):
    assert left.weights_fingerprint() == right.weights_fingerprint()
    assert left.update_count == right.update_count
    assert left.rng.bit_generator.state == right.rng.bit_generator.state
    for key in left.parameters:
        np.testing.assert_array_equal(left.parameters[key], right.parameters[key])
        np.testing.assert_array_equal(left.adam_m[key], right.adam_m[key])
        np.testing.assert_array_equal(left.adam_v[key], right.adam_v[key])


@pytest.mark.parametrize("pause", [2, 4, 6])
def test_resume_exact_on_and_off_validation_cadence(tmp_path, pause):
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    run(full, 8)
    run(resumed, pause)
    original = tree_hashes(resumed / "initialized")
    run(resumed, 8, resume=resumed / "last")
    assert tree_hashes(resumed / "initialized") == original
    for name in ("last", "best"):
        left, right = BalancedPolicy.load(full / name), BalancedPolicy.load(resumed / name)
        assert_same_policy(left, right)
        for key in ("config", "config_sha256", "completed_episodes", "best_episode", "best_validation",
                    "initialization", "seed_provenance"):
            assert left.training_state[key] == right.training_state[key]
    def comparable(path):
        return [{k: v for k, v in event.items() if k != "elapsed_seconds"}
                for event in events(path) if event["event"] != "resume"]
    assert comparable(full) == comparable(resumed)
    assert [row["episode"] for row in events(resumed) if row["event"] == "validation"] == [0, 4, 8]


def test_transfer_copies_real_v2_weights_resets_adam_and_learns(tmp_path):
    before = tree_hashes(PUBLISHED)
    original = AdaptivePolicy.load(PUBLISHED)
    summary = run(tmp_path / "run", 4)
    initial = BalancedPolicy.load(tmp_path / "run" / "initialized")
    trained = BalancedPolicy.load(tmp_path / "run" / "last")
    for key in original.parameters:
        np.testing.assert_array_equal(original.parameters[key], initial.parameters[key])
    assert initial.weights_fingerprint() != original.weights_fingerprint()  # Versioned decision semantics.
    assert initial.update_count == 0 and original.update_count > 0
    assert all(not np.any(value) for value in initial.adam_m.values())
    assert all(not np.any(value) for value in initial.adam_v.values())
    assert trained.update_count == 2
    assert summary["last_weights_sha256"] != summary["initialization"]["weights_sha256"]
    assert any(np.any(value) for value in trained.adam_m.values())
    assert summary["initialization"]["kind"] == "transferred_adaptive_weights"
    assert summary["initialization"]["source_weights_sha256"] == original.weights_fingerprint()
    assert summary["initialization"]["source_files_sha256"] == before
    assert summary["seed_provenance"]["inherited_training"] == original.training_state["seed_provenance"]
    assert summary["seed_provenance"]["inherited_selection"] == read(SELECTION)["seed_provenance"]
    assert tree_hashes(PUBLISHED) == before
    rows = [event for event in events(tmp_path / "run") if event["event"] == "training_batch"]
    assert all(row["mean_invalid_actions"] == 0 for row in rows)


def test_versioned_config_captures_all_dependencies_and_curriculum(tmp_path):
    run(tmp_path / "run")
    config = read(tmp_path / "run" / "config.json")
    assert config["schema"] == "triad.balanced_training.v1"
    assert config["policy_schema"] == POLICY_SCHEMA
    assert config["deterministic_decision"].startswith("gate first")
    assert config["source_sha256"] == trainer.source_provenance()
    assert set(config["source_sha256"]) == set(trainer.SOURCE_FILES)
    assert {"balanced_policy.py", "train_balanced.py", "train_robust.py"} <= set(config["source_sha256"])
    assert config["curriculum"]["mixed_probabilities"] == {"normal": .4, "stress": .3, "capability": .3}


def test_validation_is_gate_first_equal_profile_return_without_rng_drift():
    env = RobustPlacementEnv(profile="mixed")
    base = AdaptivePolicy.load(PUBLISHED)
    policy = BalancedPolicy.transfer_from_adaptive(base, seed=234)
    before = (policy.weights_fingerprint(), policy.rng.bit_generator.state)
    validation = trainer.validate(policy, 2, 992310)
    assert set(validation["profiles"]) == {"normal", "stress", "capability"}
    assert validation["balanced_mean_return"] == pytest.approx(
        np.mean([row["mean_return"] for row in validation["profiles"].values()]))
    for profile, row in validation["profiles"].items():
        assert row["episodes"] == 2 and row["mean_invalid_actions"] == 0
        direct = [trainer.rollout(RobustPlacementEnv(profile=profile), policy,
                                 validation["seed_ranges"][profile]["start"] + i)[1] for i in range(2)]
        assert row["mean_return"] == pytest.approx(np.mean([record["episode_return"] for record in direct]))
    assert (policy.weights_fingerprint(), policy.rng.bit_generator.state) == before
    observation = env.observe()
    assert observation["options"][policy.act(observation)]["stop"] == (
        policy.diagnostics(observation)["stop_probability"] >= .5)


def test_same_public_observation_interface():
    env = RobustPlacementEnv(seed=992340, profile="capability")
    simulated = env.observe()
    external = LiveObservationAdapter(env.catalogue).observe(simulated["state"])
    policy, metadata = trainer._initial_policy(simulated, 235, None, PUBLISHED, SELECTION)
    assert policy.act(simulated) == policy.act(external)
    np.testing.assert_array_equal(policy.probabilities(simulated), policy.probabilities(external))
    assert metadata["decision_semantics_changed"] is True


def test_selection_is_strict_maximum_return_with_profile_counts(tmp_path):
    output = tmp_path / "run"
    summary = run(output, 8, validation_every=2)
    rows = [event for event in events(output) if event["event"] == "validation"]
    expected = max(rows, key=lambda row: row["balanced_mean_return"])
    assert summary["best_episode"] == expected["episode"]
    assert summary["best_validation_balanced_mean_return"] == expected["balanced_mean_return"]
    best = -float("inf")
    for row in rows:
        assert row["selected"] == (row["balanced_mean_return"] > best)
        best = max(best, row["balanced_mean_return"])
    batches = [event for event in events(output) if event["event"] == "training_batch"]
    assert sum(sum(row["case_profile_counts"].values()) for row in batches) == 8


@pytest.mark.parametrize("corruption", ["config", "checkpoint_config", "initialized", "best", "log", "source"])
def test_resume_integrity_failure_never_writes(tmp_path, monkeypatch, corruption):
    output = tmp_path / "run"
    run(output)
    if corruption == "config":
        path = output / "config.json"
        data = read(path)
        data["learning_rate"] *= 2
        path.write_text(json.dumps(data))
    elif corruption == "checkpoint_config":
        path = output / "last" / "checkpoint.json"
        data = read(path)
        data["training_state"]["config"]["gamma"] = .5
        path.write_text(json.dumps(data))
    elif corruption in ("initialized", "best"):
        path = output / corruption / "checkpoint.json"
        path.write_text(path.read_text() + " ")
    elif corruption == "log":
        path = output / "training.jsonl"
        path.write_text(path.read_text() + "{}\n")
    else:
        changed = {**trainer.source_provenance(), "balanced_policy.py": "0" * 64}
        monkeypatch.setattr(trainer, "source_provenance", lambda: changed)
    before = tree_hashes(output)
    with pytest.raises(ValueError):
        run(output, 4, resume=output / "last")
    assert tree_hashes(output) == before


def test_wrong_resume_options_and_target_reject(tmp_path):
    output = tmp_path / "run"
    run(output)
    before = tree_hashes(output)
    with pytest.raises(ValueError, match="configuration"):
        run(output, 4, resume=output / "last", learning_rate=.1)
    with pytest.raises(ValueError, match="latest last"):
        run(output, 4, resume=output / "best")
    with pytest.raises(ValueError, match="combined"):
        run(output, 4, resume=output / "last", initial_checkpoint=PUBLISHED)
    assert tree_hashes(output) == before


def test_nonempty_output_and_partial_batch_never_reused(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    sentinel = output / "user-file.txt"
    sentinel.write_text("keep me")
    with pytest.raises(ValueError, match="already contains"):
        run(output)
    assert sentinel.read_text() == "keep me"
    with pytest.raises(ValueError, match="multiple"):
        run(tmp_path / "unaligned", 3)
    assert not (tmp_path / "unaligned").exists()


@pytest.mark.parametrize("options", [{"validation_run_seed": 999998}, {"gamma": float("nan")},
                                     {"entropy_coef": -.1}, {"seed": True}, {"episodes": True}])
def test_invalid_seed_or_hyperparameters_fail_before_output(tmp_path, options):
    with pytest.raises(ValueError):
        run(tmp_path / "run", **options)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("changes", [{"seed": 42}, {"seed": 101}, {"seed": 102}, {"seed": 103},
                                    {"validation_run_seed": 43}, {"validation_run_seed": 990100},
                                    {"validation_run_seed": 990200}])
def test_inherits_every_pretraining_and_selection_seed_reservation(tmp_path, changes):
    with pytest.raises(ValueError, match="overlap inherited"):
        run(tmp_path / "run", **changes)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("corruption", ["schema", "stage", "weights", "files", "candidate_lineage", "full_runs", "selection_ranges"])
def test_tampered_transfer_selection_rejected(tmp_path, corruption):
    data = read(SELECTION)
    if corruption == "schema":
        data["schema"] = "unversioned"
    elif corruption == "stage":
        data["stage"] = "test"
    elif corruption == "weights":
        data["selected"]["weights_sha256"] = "0" * 64
    elif corruption == "files":
        data["candidate_metadata"]["103"]["checkpoint_files_sha256"]["arrays.npz"] = "0" * 64
    elif corruption == "candidate_lineage":
        data["seed_provenance"]["lineage"]["candidate_checkpoints"].pop("101")
    elif corruption == "full_runs":
        data["seed_provenance"]["lineage"]["complete_training_runs"].pop("102")
    else:
        data["seed_provenance"]["selection_validation"] = []
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="selection report"):
        run(tmp_path / "run", initial_selection_report=path)
    assert not (tmp_path / "run").exists()


def test_transfer_requires_explicit_checkpoint_and_selection(tmp_path):
    with pytest.raises(ValueError, match="requires initial_checkpoint"):
        trainer.run_training(output=tmp_path / "run", episodes=2, **OPTIONS)
    with pytest.raises(ValueError, match="requires initial_checkpoint"):
        run(tmp_path / "run", initial_selection_report=None)
    with pytest.raises(ValueError, match="hidden_size"):
        run(tmp_path / "run", hidden_size=4)
    assert not (tmp_path / "run").exists()


def test_source_drift_during_initial_validation_rejects_before_first_write(tmp_path, monkeypatch):
    real_validate = trainer.validate
    def drift_after_validate(*args, **kwargs):
        result = real_validate(*args, **kwargs)
        changed = {**trainer.source_provenance(), "balanced_policy.py": "0" * 64}
        monkeypatch.setattr(trainer, "source_provenance", lambda: changed)
        return result
    monkeypatch.setattr(trainer, "validate", drift_after_validate)
    with pytest.raises(RuntimeError, match="preflight"):
        run(tmp_path / "run")
    assert not (tmp_path / "run").exists()
