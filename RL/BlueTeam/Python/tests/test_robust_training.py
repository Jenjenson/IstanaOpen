"""Real-environment robust trainer provenance, transfer and exact-resume checks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from triad_rl.adaptive_inputs import LiveObservationAdapter
from triad_rl.adaptive_policy import AdaptivePolicy
from triad_rl.robust_scenarios import RobustPlacementEnv
from triad_rl import train_robust as trainer


PACKAGE = Path(__file__).resolve().parents[2]
PUBLISHED = PACKAGE / "Checkpoints" / "adaptive-v1"
SELECTION = PACKAGE / "Results" / "adaptive-v1" / "common-validation-selection.json"
OPTIONS = {"batch_size": 2, "seed": 120, "hidden_size": 4,
           "validation_every": 4, "validation_episodes": 2, "validation_run_seed": 990300}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def tree_hashes(path):
    return {str(file.relative_to(path)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in path.rglob("*") if file.is_file()}


def events(path):
    return [json.loads(line) for line in (path / "training.jsonl").read_text().splitlines()]


def assert_same_policy(left, right):
    assert left.weights_fingerprint() == right.weights_fingerprint()
    assert left.update_count == right.update_count
    assert left.rng.bit_generator.state == right.rng.bit_generator.state
    for key in left.parameters:
        np.testing.assert_array_equal(left.parameters[key], right.parameters[key])
        np.testing.assert_array_equal(left.adam_m[key], right.adam_m[key])
        np.testing.assert_array_equal(left.adam_v[key], right.adam_v[key])


@pytest.mark.parametrize("pause", [2, 4, 6])
def test_full_trainer_resume_is_exact_even_off_validation_cadence(tmp_path, pause):
    full_dir, resumed_dir = tmp_path / "full", tmp_path / "resumed"
    trainer.run_training(output=full_dir, episodes=8, **OPTIONS)
    trainer.run_training(output=resumed_dir, episodes=pause, **OPTIONS)
    initialized_hashes = tree_hashes(resumed_dir / "initialized")
    trainer.run_training(output=resumed_dir, episodes=8, resume=resumed_dir / "last", **OPTIONS)
    assert tree_hashes(resumed_dir / "initialized") == initialized_hashes
    left = AdaptivePolicy.load(full_dir / "last")
    right = AdaptivePolicy.load(resumed_dir / "last")
    assert_same_policy(left, right)
    assert_same_policy(AdaptivePolicy.load(full_dir / "best"), AdaptivePolicy.load(resumed_dir / "best"))
    for key in ("config", "config_sha256", "completed_episodes", "best_episode", "best_validation",
                "initialization", "seed_provenance"):
        assert left.training_state[key] == right.training_state[key]
    # Audit timestamps and the explicit resume event differ; numerical batches
    # and scheduled validation/model-selection decisions are exactly equal.
    def comparable(path):
        return [{k: v for k, v in event.items() if k != "elapsed_seconds"}
                for event in events(path) if event["event"] != "resume"]
    assert comparable(full_dir) == comparable(resumed_dir)
    assert [event["episode"] for event in events(resumed_dir) if event["event"] == "validation"] == [0, 4, 8]


def test_validation_is_equal_profile_return_and_does_not_consume_policy_rng():
    env = RobustPlacementEnv(profile="mixed")
    policy = AdaptivePolicy(env.observe()["feature_names"], seed=124, hidden_size=4)
    before = (policy.weights_fingerprint(), policy.rng.bit_generator.state)
    validation = trainer.validate(policy, 3, 990310)
    assert set(validation["profiles"]) == {"normal", "stress", "capability"}
    assert validation["balanced_mean_return"] == pytest.approx(
        np.mean([row["mean_return"] for row in validation["profiles"].values()]))
    for profile, row in validation["profiles"].items():
        assert row["episodes"] == 3
        assert row["mean_invalid_actions"] == 0
        assert all(key in row for key in ("success_rate", "detection_rate", "mean_cost", "coverage", "breach_rate"))
        direct = [trainer.rollout(RobustPlacementEnv(profile=profile), policy,
                                  validation["seed_ranges"][profile]["start"] + i)[1] for i in range(3)]
        assert row["mean_return"] == pytest.approx(np.mean([record["episode_return"] for record in direct]))
    assert (policy.weights_fingerprint(), policy.rng.bit_generator.state) == before


def test_actual_v1_transfer_preserves_weights_but_resets_optimizer_and_reserves_exposure(tmp_path):
    source_hashes = tree_hashes(PUBLISHED)
    initial = AdaptivePolicy.load(PUBLISHED)
    options = {**OPTIONS, "hidden_size": None, "initial_checkpoint": PUBLISHED,
               "initial_selection_report": SELECTION}
    summary = trainer.run_training(output=tmp_path / "run", episodes=4, **options)
    initialized = AdaptivePolicy.load(tmp_path / "run" / "initialized")
    trained = AdaptivePolicy.load(tmp_path / "run" / "last")
    assert initialized.weights_fingerprint() == initial.weights_fingerprint()
    assert initialized.update_count == 0 and initial.update_count > 0
    assert all(not np.any(value) for value in initialized.adam_m.values())
    assert all(not np.any(value) for value in initialized.adam_v.values())
    assert trained.update_count == 2
    assert summary["initialization"]["kind"] == "transferred_weights"
    assert summary["initialization"]["source_files_sha256"] == source_hashes
    assert summary["seed_provenance"]["inherited_training"] == initial.training_state["seed_provenance"]
    assert summary["seed_provenance"]["inherited_selection"] == read(SELECTION)["seed_provenance"]
    assert tree_hashes(PUBLISHED) == source_hashes
    assert summary["last_weights_sha256"] != summary["initialization"]["weights_sha256"]


def test_transferred_checkpoint_works_on_same_live_observation_contract(tmp_path):
    env = RobustPlacementEnv(seed=701, profile="capability")
    simulated = env.observe()
    external = LiveObservationAdapter(env.catalogue).observe(simulated["state"])
    original = AdaptivePolicy.load(PUBLISHED, feature_names=simulated["feature_names"])
    transferred, metadata = trainer._initial_policy(simulated, 124, None, PUBLISHED, SELECTION)
    assert original.act(simulated) == transferred.act(simulated) == transferred.act(external)
    np.testing.assert_array_equal(transferred.probabilities(simulated), transferred.probabilities(external))
    assert metadata["kind"] == "transferred_weights"


def test_transferred_full_run_and_resume_preserve_identical_optimizer_rng_and_selection(tmp_path):
    options = {**OPTIONS, "hidden_size": None}
    transfer = {"initial_checkpoint": PUBLISHED, "initial_selection_report": SELECTION}
    trainer.run_training(output=tmp_path / "full", episodes=4, **options, **transfer)
    trainer.run_training(output=tmp_path / "resumed", episodes=2, **options, **transfer)
    initialized_hashes = tree_hashes(tmp_path / "resumed" / "initialized")
    # Resume inherits recorded transfer provenance, without rereading or
    # requiring the original pretrained checkpoint or its selection report.
    trainer.run_training(output=tmp_path / "resumed", episodes=4,
                         resume=tmp_path / "resumed" / "last", **options)
    assert_same_policy(AdaptivePolicy.load(tmp_path / "full" / "last"),
                       AdaptivePolicy.load(tmp_path / "resumed" / "last"))
    assert_same_policy(AdaptivePolicy.load(tmp_path / "full" / "best"),
                       AdaptivePolicy.load(tmp_path / "resumed" / "best"))
    assert tree_hashes(tmp_path / "resumed" / "initialized") == initialized_hashes


def test_logged_selection_is_strict_maximum_balanced_return(tmp_path):
    options = {**OPTIONS, "validation_every": 2}
    summary = trainer.run_training(output=tmp_path / "run", episodes=8, **options)
    rows = [event for event in events(tmp_path / "run") if event["event"] == "validation"]
    expected = max(rows, key=lambda event: event["balanced_mean_return"])
    assert summary["best_episode"] == expected["episode"]
    assert summary["best_validation_balanced_mean_return"] == expected["balanced_mean_return"]
    best_score = -float("inf")
    for row in rows:
        assert row["selected"] == (row["balanced_mean_return"] > best_score)
        best_score = max(best_score, row["balanced_mean_return"])
    batches = [event for event in events(tmp_path / "run") if event["event"] == "training_batch"]
    assert sum(sum(event["case_profile_counts"].values()) for event in batches) == 8
    assert all(set(event["case_profile_counts"]) <= {"normal", "stress", "capability"} for event in batches)


@pytest.mark.parametrize("corruption", ["config", "checkpoint_config", "initialized", "best", "log", "source"])
def test_resume_integrity_failures_reject_before_writing(tmp_path, monkeypatch, corruption):
    output = tmp_path / "run"
    trainer.run_training(output=output, episodes=2, **OPTIONS)
    if corruption == "config":
        path = output / "config.json"
        config = read(path)
        config["learning_rate"] *= 2
        path.write_text(json.dumps(config))
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
        changed = {**trainer.source_provenance(), "robust_scenarios.py": "0" * 64}
        monkeypatch.setattr(trainer, "source_provenance", lambda: changed)
    before = tree_hashes(output)
    with pytest.raises(ValueError):
        trainer.run_training(output=output, episodes=4, resume=output / "last", **OPTIONS)
    assert tree_hashes(output) == before


def test_changed_hyperparameters_and_wrong_resume_target_reject(tmp_path):
    output = tmp_path / "run"
    trainer.run_training(output=output, episodes=2, **OPTIONS)
    before = tree_hashes(output)
    with pytest.raises(ValueError, match="configuration"):
        trainer.run_training(output=output, episodes=4, resume=output / "last", learning_rate=.1, **OPTIONS)
    with pytest.raises(ValueError, match="latest last"):
        trainer.run_training(output=output, episodes=4, resume=output / "best", **OPTIONS)
    assert tree_hashes(output) == before


def test_nonempty_output_and_non_batch_endpoint_are_never_reused(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    sentinel = output / "user-file.txt"
    sentinel.write_text("keep me")
    with pytest.raises(ValueError, match="already contains"):
        trainer.run_training(output=output, episodes=2, **OPTIONS)
    assert sentinel.read_text() == "keep me"
    with pytest.raises(ValueError, match="multiple"):
        trainer.run_training(output=tmp_path / "unaligned", episodes=3, **OPTIONS)
    assert not (tmp_path / "unaligned").exists()


def test_fresh_checkpoint_identified_separately_from_transfer(tmp_path):
    summary = trainer.run_training(output=tmp_path / "run", episodes=2, **OPTIONS)
    assert summary["initialization"]["kind"] == "fresh"
    assert "source_files_sha256" not in summary["initialization"]
    assert not summary["seed_provenance"]["inherited_training"]
    config = read(tmp_path / "run" / "config.json")
    assert config["source_sha256"] == trainer.source_provenance()
    assert set(config["source_sha256"]) == set(trainer.SOURCE_FILES)
    assert config["curriculum"]["mixed_probabilities"] == {"normal": .4, "stress": .3, "capability": .3}


@pytest.mark.parametrize("options", [{"validation_run_seed": 999998}, {"gamma": float("nan")},
                                     {"entropy_coef": -.1}, {"seed": True}, {"episodes": True}])
def test_bad_seed_slots_and_hyperparameters_fail_before_output(tmp_path, options):
    with pytest.raises(ValueError):
        trainer.run_training(output=tmp_path / "run", **{**OPTIONS, "episodes": 2, **options})
    assert not (tmp_path / "run").exists()


def test_transfer_rejects_overlapping_pretrained_train_and_validation_exposure(tmp_path):
    options = {**OPTIONS, "hidden_size": None, "initial_checkpoint": PUBLISHED,
               "initial_selection_report": SELECTION}
    for changes in ({"seed": 43}, {"validation_run_seed": 43}, {"validation_run_seed": 999}):
        with pytest.raises(ValueError, match="overlap inherited"):
            trainer.run_training(output=tmp_path / "run", episodes=2, **{**options, **changes})
    assert not (tmp_path / "run").exists()


def test_bad_transfer_selection_and_hidden_size_are_rejected(tmp_path):
    corrupted = read(SELECTION)
    corrupted["selected_weights_sha256"] = "0" * 64
    report = tmp_path / "selection.json"
    report.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="selection report"):
        trainer.run_training(output=tmp_path / "run", episodes=2,
                             **{**OPTIONS, "hidden_size": None, "initial_checkpoint": PUBLISHED,
                                "initial_selection_report": report})
    with pytest.raises(ValueError, match="hidden_size"):
        trainer.run_training(output=tmp_path / "run", episodes=2, initial_checkpoint=PUBLISHED, **OPTIONS)
    assert not (tmp_path / "run").exists()
