"""Real short anchored arms and read-only, source-bound endpoint validation."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from triad_rl import train_anchored_pilot as trainer
from triad_rl.balanced_policy import BalancedPolicy


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tree(path):
    return {file.relative_to(path).as_posix(): trainer.shared._file_hash(file)
            for file in path.rglob("*") if file.is_file()}


def declaration(path, **training):
    value = read(trainer.shared.BLUE_ROOT / "Results/anchored-v4-pilot/protocol.json")
    value["training"].update(policy_seed=499, scenario_seed_start=499_000_000, episodes=4, batch_size=2)
    value["training"].update(training)
    value["training_implementation_sha256"] = trainer.source_provenance()
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def events(directory):
    return [json.loads(line) for line in (directory / "training.jsonl").read_text().splitlines()]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("anchored-arms")
    protocol = declaration(root / "protocol.json")
    lineage = trainer.load_published_lineage()
    summaries = {arm: trainer.run_training(protocol, arm, root / arm) for arm in trainer.ARMS}
    return root, protocol, lineage, summaries


def same_policy(first, second):
    assert first.weights_fingerprint() == second.weights_fingerprint()
    assert first.update_count == second.update_count
    assert first.rng.bit_generator.state == second.rng.bit_generator.state
    for group in ("parameters", "adam_m", "adam_v"):
        for key, value in getattr(first, group).items():
            assert value.tobytes() == getattr(second, group)[key].tobytes()


def test_common_initialization_trajectory_and_first_coefficients(runs):
    root, _, _, summaries = runs
    initials = [BalancedPolicy.load(root / arm / "initialized") for arm in trainer.ARMS]
    for other in initials[1:]:
        same_policy(initials[0], other)
    assert initials[0].rng.bit_generator.state == np.random.default_rng(499).bit_generator.state
    assert initials[0].update_count == 0
    assert all(not array.any() for values in (initials[0].adam_m, initials[0].adam_v) for array in values.values())
    assert len({value["first_batch_trajectory_sha256"] for value in summaries.values()}) == 1
    assert len({value["first_batch_coefficient_identity"]["reference_first_coefficients_sha256"] for value in summaries.values()}) == 1
    assert len({value["first_batch_coefficient_identity"]["applied_first_coefficients_sha256"] for value in summaries.values()}) == 1
    for summary in summaries.values():
        assert summary["last_weights_sha256"] != initials[0].weights_fingerprint()


@pytest.mark.parametrize("arm", list(trainer.ARMS))
def test_real_runs_match_exact_direct_updates_and_endpoint_validator(runs, arm):
    root, protocol, lineage, summaries = runs
    actor = BalancedPolicy.load(root / arm / "initialized")
    env = trainer.RobustPlacementEnv(seed=499_000_000, profile="mixed")
    for begin in (0, 2):
        batch = []
        for offset in range(begin, begin + 2):
            if arm == "anchored_later_stop":
                episode, _ = trainer.anchored_credit.rollout(env, actor, 499_000_000 + offset)
            else:
                episode, _ = trainer.rollout(env, actor, 499_000_000 + offset, training=True)
            batch.append(episode)
        if arm == "anchored_later_stop":
            trainer.anchored_credit.update(actor, batch)
        else:
            actor.update(batch, learning_rate=.002, entropy_coef=.015, gamma=1., value_coef=trainer.ARMS[arm]["value_coef"], max_grad_norm=1.)
    endpoint, evidence = trainer.validate_run(root / arm, protocol, arm, lineage)
    same_policy(actor, endpoint)
    assert evidence["summary"] == summaries[arm]
    assert evidence["run_files_sha256"] == tree(root / arm)
    assert "metadata" not in evidence


def test_later_only_branch_accounting_and_reference_applied_credit(runs):
    root, _, _, summaries = runs
    for arm, summary in summaries.items():
        rows = events(root / arm)[1:]
        assert summary["completed_episodes"] == 4 and summary["optimizer_updates"] == 2
        assert set(tree(root / arm)) == set(trainer.RUN_FILES)
        assert summary["all_first_coefficients_identical"] is True
        assert summary["compute"]["extra_branch_rollouts"] == (summary["compute"]["live_decisions"] - 4 if arm == "anchored_later_stop" else 0)
        for row in rows:
            proof = row["coefficient_identity"]
            assert proof["first_coefficients_count"] == 2 and proof["first_coefficients_identical"]
            assert proof["reference_first_coefficients_sha256"] == proof["applied_first_coefficients_sha256"]
            assert row["credit"]["first"]["reference_normalized_advantage"] == row["credit"]["first"]["normalized_advantage"]
            assert row["credit"]["first"]["baseline"] == row["credit"]["first"]["credit_baseline"]
            assert row["diagnostic_total_gradient_norm"] == row["gradient_norm"]
            assert row["mean_invalid_actions"] == 0
            assert row["compute"]["extra_branch_rollouts"] == (row["transitions"] - 2 if arm == "anchored_later_stop" else 0)
            if arm != "shared_critic":
                assert row["weighted_critic_gradient_norm"] == row["critic_head_delta_l2"] == 0
        if arm != "shared_critic":
            trainer._head_check(BalancedPolicy.load(root / arm / "last"), BalancedPolicy.load(root / arm / "initialized"))


def test_validation_never_samples_infers_or_writes(runs, monkeypatch):
    root, protocol, lineage, _ = runs
    def forbidden(*args, **kwargs):
        pytest.fail("Endpoint validation must be read-only and perform no inference/sampling")
    monkeypatch.setattr(trainer, "RobustPlacementEnv", forbidden)
    monkeypatch.setattr(BalancedPolicy, "sample", forbidden)
    monkeypatch.setattr(BalancedPolicy, "act", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    for arm in trainer.ARMS:
        trainer.validate_run(root / arm, protocol, arm, lineage)


@pytest.mark.parametrize("change", ["schema", "arm", "source", "seed", "count", "batch", "gamma", "boolean_coefficient", "reference", "unknown_training_field"])
def test_protocol_rejection_before_outputs(tmp_path, change):
    protocol = declaration(tmp_path / "protocol.json")
    value = read(protocol)
    if change == "schema": value["schema"] = "bad"
    elif change == "arm": value["arms"].pop("anchored_later_stop")
    elif change == "source": value["training_implementation_sha256"]["triad_rl/train_anchored_pilot.py"] = "0" * 64
    elif change == "seed": value["training"]["scenario_seed_start"] = 2 * 10**15
    elif change == "count": value["training"]["episodes"] = 4097
    elif change == "batch": value["training"]["batch_size"] = 3
    elif change == "gamma": value["training"]["gamma"] = .99
    elif change == "boolean_coefficient": value["arms"]["anchored_later_stop"]["value_coef"] = False
    elif change == "reference": value["frozen_reference"]["weights_sha256"] = "0" * 64
    else: value["training"]["unrecognized"] = True
    protocol.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        trainer.run_training(protocol, "shared_critic", tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_preflight_drift_and_reused_output_do_not_write(tmp_path, monkeypatch, runs):
    root, protocol, _, _ = runs
    before = tree(root / "shared_critic")
    with pytest.raises(ValueError, match="new or empty"):
        trainer.run_training(protocol, "shared_critic", root / "shared_critic")
    assert tree(root / "shared_critic") == before
    protocol = declaration(tmp_path / "protocol.json")
    original = trainer.load_published_lineage
    def drift():
        result = original()
        protocol.write_bytes(protocol.read_bytes() + b" ")
        return result
    monkeypatch.setattr(trainer, "load_published_lineage", drift)
    with pytest.raises(ValueError, match="changed"):
        trainer.run_training(protocol, "shared_critic", tmp_path / "new")
    assert not (tmp_path / "new").exists()


def rebind_log(directory, mutate):
    rows = events(directory)
    mutate(rows)
    (directory / "training.jsonl").write_text("".join(trainer.shared._json(row) + "\n" for row in rows), encoding="utf-8")
    digest = trainer.shared._file_hash(directory / "training.jsonl")
    metadata = read(directory / "last/checkpoint.json")
    metadata["training_state"]["log_sha256"] = digest
    (directory / "last/checkpoint.json").write_text(trainer.shared._json(metadata) + "\n", encoding="utf-8")
    summary = read(directory / "summary.json")
    summary["log_sha256"] = digest
    summary["last_files_sha256"] = trainer.shared._checkpoint_files(directory / "last")
    (directory / "summary.json").write_text(trainer.shared._json(summary) + "\n", encoding="utf-8")


@pytest.mark.parametrize("change", ["cadence", "first_identity", "first_hash", "branches", "moments", "groups", "sensor_count", "profile_count", "clip", "updates"])
def test_rehashed_log_tampering_is_rejected(runs, tmp_path, change):
    root, protocol, lineage, _ = runs
    directory = tmp_path / "copy"
    shutil.copytree(root / "anchored_later_stop", directory)
    def mutate(rows):
        row = rows[1]
        if change == "cadence": row["episode_start"] = 1
        elif change == "first_identity": row["coefficient_identity"]["first_coefficients_identical"] = 1
        elif change == "first_hash": row["coefficient_identity"]["applied_first_coefficients_sha256"] = "0" * 64
        elif change == "branches": row["compute"]["extra_branch_rollouts"] += 1
        elif change == "moments": row["credit"]["first"]["normalized_advantage"]["mean"] += .1
        elif change == "groups": row["credit"].pop("later_stop")
        elif change == "sensor_count": row["sensor_deployment_counts"]["rf"] = 999
        elif change == "profile_count": row["case_profile_counts"]["normal"] = 999
        elif change == "clip": row["gradient_scale"] = .5
        else: row["updates"] = 999
    rebind_log(directory, mutate)
    with pytest.raises(ValueError):
        trainer.validate_run(directory, protocol, "anchored_later_stop", lineage)


@pytest.mark.parametrize("change", ["credit", "cost", "count", "branches", "config", "protocol", "extra_file"])
def test_endpoint_summary_and_artifact_tampering_rejected(runs, tmp_path, change):
    root, protocol, lineage, _ = runs
    directory = tmp_path / "copy"
    shutil.copytree(root / "shared_critic", directory)
    if change == "config":
        value = read(directory / "config.json")
        value["training"]["episodes"] = 2
        (directory / "config.json").write_text(trainer.shared._json(value) + "\n", encoding="utf-8")
    elif change == "protocol":
        (directory / "protocol.json").write_bytes((directory / "protocol.json").read_bytes() + b" ")
    elif change == "extra_file":
        (directory / "not-a-run-artifact.txt").write_text("extra")
    else:
        value = read(directory / "summary.json")
        if change == "credit": value["credit"]["later"]["raw_advantage"]["mean"] += .1
        elif change == "cost": value["mean_cost"] += .1
        elif change == "count": value["completed_episodes"] = 4.0
        else: value["compute"]["extra_branch_rollouts"] = 1
        (directory / "summary.json").write_text(trainer.shared._json(value) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        trainer.validate_run(directory, protocol, "shared_critic", lineage)


def test_declared_prior_publication_mismatch_rejected_before_writes(tmp_path):
    protocol = declaration(tmp_path / "protocol.json")
    value = read(protocol)
    value["prior_pilot_publication"]["sha256"] = "0" * 64
    protocol.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="Declared prior"):
        trainer.run_training(protocol, "shared_critic", tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_protocol_accepts_predeclared_4096_endpoint_without_sampling(tmp_path):
    protocol = declaration(tmp_path / "protocol.json", episodes=4096, batch_size=16)
    parsed, _ = trainer.load_protocol(protocol)
    assert parsed["training"]["episodes"] == 4096
