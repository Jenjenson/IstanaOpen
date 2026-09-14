"""Bounded driver checks on handmade cases only; no proposed pilot slots."""
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

import train_temporal as trainer
from triad_rl import adaptive_env, robust_scenarios
from triad_rl.temporal_env import TemporalPlacementEnv
from triad_rl.temporal_policy import TemporalPolicy
from test_temporal_env import scenario, catalogue, config


def forbidden(*args, **kwargs):
    raise AssertionError("No new scenario generation, learning, inference or output writes allowed")


def bind(monkeypatch):
    fixed = {**trainer.FIXED_TRAINING, "policy_seeds": [499], "episodes": 4, "batch_size": 2}
    monkeypatch.setattr(trainer, "FIXED_TRAINING", fixed)
    lineage = {"schema": "triad.temporal_published_lineage.v1", "consumed_seed_ranges": [],
        "reserved_final_seed_ranges": [{"start": 2 * 10**15, "count": 100}], "publication_manifests": [],
        "prior_pilot": {"publication_manifest": {"path": "fixture-manifest.json", "sha256": "a" * 64}}}
    monkeypatch.setattr(trainer, "load_published_lineage", lambda: deepcopy(lineage))
    monkeypatch.setattr(adaptive_env, "generate_scenario", forbidden)
    monkeypatch.setattr(robust_scenarios, "make_case", forbidden)
    calls = []
    def hand_environment(*, seed, profile, config):
        assert 499_000_000 <= seed < 499_000_004 and profile == "mixed"
        calls.append(seed)
        case = scenario()
        case["seed"] = seed
        env = TemporalPlacementEnv(scenario=case, catalogue=catalogue(), config=config)
        env.case_metadata["profile"] = ("normal", "stress", "capability")[seed % 3]
        return env
    monkeypatch.setattr(trainer, "TemporalPlacementEnv", hand_environment)
    return lineage, calls


def make_protocol(path, lineage):
    protocol = {"schema": trainer.PROTOCOL_SCHEMA, "experiment": "handmade-test-only",
        "temporal_config": asdict(config()), "prior_publication": lineage["prior_pilot"]["publication_manifest"],
        "training": {**trainer.FIXED_TRAINING, "scenario_seed_ranges": {"499": {"start": 499_000_000, "count": 4}}},
        "training_implementation_sha256": trainer.source_provenance()}
    path.write_bytes(trainer._bytes(protocol))
    return path


@pytest.fixture(scope="module")
def archived(tmp_path_factory):
    root = tmp_path_factory.mktemp("temporal-handmade")
    # Never retain global monkeypatches across unrelated test modules.
    with pytest.MonkeyPatch.context() as patch:
        lineage, calls = bind(patch)
        protocol = make_protocol(root / "protocol.json", lineage)
        summary = trainer.run_training(protocol, 499, root / "run")
        assert calls == list(range(499_000_000, 499_000_004))
    return {"root": root, "protocol": protocol, "summary": summary, "lineage": lineage}


@pytest.fixture
def run(archived, monkeypatch):
    bind(monkeypatch)
    return archived


def test_complete_handmade_run_has_one_update_per_full_batch_and_real_parameter_changes(run):
    path = run["root"] / "run"
    actor, evidence = trainer.validate_run(path, run["protocol"], lineage=run["lineage"])
    initial = TemporalPolicy.load(path / "initialized", config=config())
    assert {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()} == set(trainer.RUN_FILES)
    assert run["summary"]["completed_episodes"] == 4 and actor.update_count == actor.critic_update_count == 2
    assert actor.weights_fingerprint() != initial.weights_fingerprint()
    assert np.any(actor.parameters["wa"] != initial.parameters["wa"])
    assert np.any(actor.parameters["wv"] != initial.parameters["wv"])
    assert run["summary"]["mean_invalid_actions"] == 0.
    assert evidence["run_files_sha256"]["summary.json"] == trainer._sha(path / "summary.json")
    assert evidence["verification_scope"]["optimizer_replays"] == 0
    assert set(run["summary"]["profile_counts"]) == {"normal", "stress", "capability"}


def test_validator_performs_no_learning_sampling_inference_or_writes(run, monkeypatch):
    path = run["root"] / "run"
    before = {name: trainer._sha(path / name) for name in trainer.RUN_FILES}
    monkeypatch.setattr(trainer, "collect_episode", forbidden)
    monkeypatch.setattr(trainer, "TemporalPlacementEnv", forbidden)
    for name in ("sample", "act", "value", "scores", "probabilities", "update", "save"):
        monkeypatch.setattr(TemporalPolicy, name, forbidden)
    for name in ("write_bytes", "write_text", "mkdir"):
        monkeypatch.setattr(Path, name, forbidden)
    _, evidence = trainer.validate_run(path, run["protocol"], lineage=run["lineage"])
    assert evidence["verification_scope"]["filesystem_writes"] == 0
    assert before == {name: trainer._sha(path / name) for name in trainer.RUN_FILES}


def test_recorded_sampling_rng_values_and_rewards_exactly_reconstruct_small_run(run):
    path = run["root"] / "run"
    actor = TemporalPolicy.load(path / "initialized", config=config())
    rows = [json.loads(line) for line in gzip.open(path / "episodes.jsonl.gz", "rt")]
    logs = [json.loads(line) for line in (path / "training.jsonl").read_text().splitlines()]
    # Test-only optimizer replay proves the trainer recorded the exact actual inputs.
    for index in range(0, 4, 2):
        batch = rows[index:index + 2]
        for row in batch:
            assert actor.rng_fingerprint() == row["rng_before_sha256"]
            for record in row["transitions"]:
                observation = {"feature_schema": record["feature_schema"], "feature_names": actor.feature_names,
                    "temporal_config": record["temporal_config"], "option_features": record["features"],
                    "action_mask": np.asarray(record["mask"], dtype=bool)}
                action, sampled = actor.sample(observation)
                assert action == record["action"] and sampled["value"] == record["value"]
            assert actor.rng_fingerprint() == row["rng_after_sha256"]
        metrics = actor.update([row["transitions"] for row in batch])
        assert metrics == logs[index // 2]["update_metrics"]
    saved = TemporalPolicy.load(path / "last", config=config())
    assert actor.weights_fingerprint() == saved.weights_fingerprint()
    assert actor.rng_fingerprint() == saved.rng_fingerprint()
    for key in actor.parameters:
        for group in ("parameters", "adam_m", "adam_v"):
            np.testing.assert_array_equal(getattr(actor, group)[key], getattr(saved, group)[key])


@pytest.mark.parametrize("mutation", ["schema", "seed", "episodes", "batch_size", "learning_rate", "critic_learning_rate",
    "entropy_coef", "normalize_advantages", "range", "source", "mission"])
def test_protocol_drift_fails_before_output_or_case_construction(run, tmp_path, monkeypatch, mutation):
    protocol = json.loads(run["protocol"].read_text())
    if mutation == "schema": protocol["schema"] = "wrong"
    elif mutation == "seed": protocol["training"]["policy_seeds"] = [498]
    elif mutation == "range": protocol["training"]["scenario_seed_ranges"]["499"]["start"] += 1
    elif mutation == "source": protocol["training_implementation_sha256"]["train_temporal.py"] = "0" * 64
    elif mutation == "mission": protocol["temporal_config"].pop("lead_time_s")
    elif mutation == "normalize_advantages": protocol["training"][mutation] = 0
    else: protocol["training"][mutation] += 1
    altered = tmp_path / "protocol.json"
    altered.write_bytes(trainer._bytes(protocol))
    monkeypatch.setattr(trainer, "TemporalPlacementEnv", forbidden)
    with pytest.raises(ValueError): trainer.run_training(altered, 499, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("kind", ["consumed_seed_ranges", "reserved_final_seed_ranges", "prior"])
def test_inherited_exposure_or_prior_mismatch_fails_before_writes(run, tmp_path, monkeypatch, kind):
    lineage = deepcopy(run["lineage"])
    if kind == "prior": lineage["prior_pilot"]["publication_manifest"]["sha256"] = "b" * 64
    else: lineage[kind] = [{"start": 499_000_002, "count": 1}]
    monkeypatch.setattr(trainer, "load_published_lineage", lambda: lineage)
    monkeypatch.setattr(trainer, "TemporalPlacementEnv", forbidden)
    with pytest.raises(ValueError): trainer.run_training(run["protocol"], 499, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_reused_output_even_empty_directory_and_unlisted_seed_rejected(run, tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(trainer, "TemporalPlacementEnv", forbidden)
    with pytest.raises(ValueError): trainer.run_training(run["protocol"], 499, output)
    assert not list(output.iterdir())
    with pytest.raises(ValueError): trainer.run_training(run["protocol"], 498, tmp_path / "new")
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("name", ["config.json", "training.jsonl", "episodes.jsonl.gz", "last/arrays.npz", "initialized/arrays.npz"])
def test_altered_bound_artifact_rejected(run, tmp_path, name):
    path = tmp_path / "copy"
    shutil.copytree(run["root"] / "run", path)
    target = path / name
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises((ValueError, EOFError, OSError)):
        trainer.validate_run(path, run["protocol"], lineage=run["lineage"])


def test_summary_counts_and_unknown_artifacts_are_not_silently_accepted(run, tmp_path):
    path = tmp_path / "copy"
    shutil.copytree(run["root"] / "run", path)
    summary = json.loads((path / "summary.json").read_text())
    summary["completed_episodes"] = 3
    (path / "summary.json").write_bytes(trainer._bytes(summary))
    with pytest.raises(ValueError): trainer.validate_run(path, run["protocol"], lineage=run["lineage"])
    (path / "extra.json").write_text("{}")
    with pytest.raises(ValueError): trainer.validate_run(path, run["protocol"], lineage=run["lineage"])


def rebind_log(path):
    """Rehash a copied fixture so tests reach derived checks, not just byte guards."""
    evidence = {name: trainer._sha(path / name) for name in ("config.json", "protocol.json", "training.jsonl", "episodes.jsonl.gz")}
    checkpoint = json.loads((path / "last/checkpoint.json").read_text())
    checkpoint["training_state"]["evidence_files_sha256"] = evidence
    (path / "last/checkpoint.json").write_bytes(trainer._bytes(checkpoint))
    summary = json.loads((path / "summary.json").read_text())
    summary["evidence_files_sha256"] = evidence
    summary["last_files_sha256"] = trainer._checkpoint_files(path / "last")
    (path / "summary.json").write_bytes(trainer._bytes(summary))


@pytest.mark.parametrize("mutation", ["bool_counter", "missing_metric", "wrong_rng", "negative_time", "wrong_cadence", "false_normalization"])
def test_rehashed_batch_tampering_rejected_without_optimizer_replay(run, tmp_path, monkeypatch, mutation):
    path = tmp_path / "copy"
    shutil.copytree(run["root"] / "run", path)
    rows = [json.loads(line) for line in (path / "training.jsonl").read_text().splitlines()]
    if mutation == "bool_counter": rows[0]["update_metrics"]["updates"] = True
    elif mutation == "missing_metric": rows[0]["update_metrics"].pop("actor_loss")
    elif mutation == "wrong_rng": rows[0]["rng_sha256"] = "0" * 64
    elif mutation == "negative_time": rows[0]["collection_wall_seconds"] = -1.
    elif mutation == "wrong_cadence": rows[0]["episode_start"] = 1
    else: rows[0]["update_metrics"]["advantages_normalized"] = 0
    (path / "training.jsonl").write_bytes(b"".join(trainer._bytes(row) for row in rows))
    rebind_log(path)
    monkeypatch.setattr(TemporalPolicy, "update", forbidden)
    with pytest.raises(ValueError): trainer.validate_run(path, run["protocol"], lineage=run["lineage"])
