"""Bounded real collection, dataset integrity and read-only endpoint replay."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from triad_rl import train_ranking as trainer
from triad_rl.ranking_policy import RankPolicy


def _fixture_bindings(monkeypatch):
    lineage = {"schema": "triad.ranking_published_lineage.v1", "consumed_seed_ranges": [],
        "reserved_final_seed_ranges": [], "publication_manifests": [],
        "prior_pilot": {"publication_manifest": {"path": "fixture/manifest.json", "sha256": "a" * 64}}}
    monkeypatch.setattr(trainer, "load_published_lineage", lambda: deepcopy(lineage))
    monkeypatch.setattr(trainer, "source_provenance", lambda: {"triad_rl/fixture.py": "b" * 64})


def make_protocol(tmp_path, monkeypatch, episodes=4, batch_size=2, passes_per_batch=2):
    """Fast isolated protocol fixture; only designated training seed499 is used."""
    _fixture_bindings(monkeypatch)
    train = {"policy_seeds": [499], "episodes": episodes, "batch_size": batch_size,
        "passes_per_batch": passes_per_batch, "scenario_seed_ranges": {"499": {"start": 499000000, "count": episodes}},
        "profile": "mixed", "hidden_size": 64, "max_actions": 6, "learning_rate": .002,
        "temperature": .1, "regularization": .01, "max_grad_norm": 1., "continuation": "frozen_public_greedy",
        "behavior": "current_ranker_deterministic_before_branch_labels",
        "initialization": "exact_public_greedy_prior_plus_zero_residual", "endpoint": "fixed_no_selection_or_resume"}
    protocol = {"schema": trainer.PROTOCOL_SCHEMA, "training": train,
        "training_implementation_sha256": trainer.source_provenance(),
        "proposal_v3_weights_sha256": trainer.V3_WEIGHTS,
        "prior_publication": {"path": "fixture/manifest.json", "sha256": "a" * 64}}
    path = tmp_path / "protocol.json"
    path.write_bytes(trainer._bytes(protocol))
    return path


@pytest.fixture(scope="module")
def _cached_small_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ranking-run")
    with pytest.MonkeyPatch.context() as patch:
        protocol = make_protocol(tmp, patch)
        run = tmp / "run"
        summary = trainer.run_training(protocol, 499, run)
    return run, protocol, summary


@pytest.fixture
def small_run(_cached_small_run, monkeypatch):
    # Cached computation must never retain global patches between test modules.
    _fixture_bindings(monkeypatch)
    return _cached_small_run


def load_dataset(run):
    return [json.loads(line) for line in gzip.decompress((run / "comparisons.jsonl.gz").read_bytes()).splitlines()]


def load_log(run):
    return [json.loads(line) for line in (run / "training.jsonl").read_bytes().splitlines()]


def copy_run(small_run, tmp_path):
    original, protocol, _ = small_run
    run = tmp_path / "copy"
    shutil.copytree(original, run)
    return run, protocol


def rebind(run):
    """Rebind attacker-controlled hashes; semantic replay must still reject edits."""
    actor = RankPolicy.load(run / "last")
    evidence = {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                for name in ("protocol.json", "config.json", "training.jsonl", "comparisons.jsonl.gz")}
    state = deepcopy(actor.training_state)
    state["evidence_files_sha256"] = evidence
    actor.save(run / "last", state)
    summary = json.loads((run / "summary.json").read_text())
    summary["evidence_files_sha256"] = evidence
    summary["last_files_sha256"] = trainer.shared._checkpoint_files(run / "last")
    summary["last_weights_sha256"] = actor.weights_fingerprint()
    (run / "summary.json").write_bytes(trainer._bytes(summary))


def test_real_run_complete_data_and_exact_endpoint_contract(small_run):
    run, protocol, summary = small_run
    actor, evidence = trainer.validate_run(run, protocol)
    assert set(path.relative_to(run).as_posix() for path in run.rglob("*") if path.is_file()) == set(trainer.RUN_FILES)
    assert summary["completed_episodes"] == 4
    assert summary["optimizer_updates"] == actor.update_count == 4
    assert evidence["summary"] == summary
    assert evidence["weights_sha256"] == actor.weights_fingerprint()
    assert evidence["verification_scope"]["scenario_rollouts"] == 0
    assert evidence["verification_scope"]["filesystem_writes"] == 0
    assert evidence["verification_scope"]["full_public_observations_reconstructed"] is False
    data, log = load_dataset(run), load_log(run)
    assert len(data) == summary["decisions"]
    assert sum(len(row["outcomes"]) for row in data) == summary["branches"]
    assert sum(len(row["comparisons"]) for row in data) == summary["comparisons"]
    assert [r["seed"] for row in log for r in row["episodes"]] == list(range(499000000, 499000004))
    initial = RankPolicy.load(run / "initialized")
    assert initial.update_count == 0 and not np.any(initial.parameters["wa"])
    assert initial.weights_fingerprint() != actor.weights_fingerprint()
    assert initial.rng.bit_generator.state == actor.rng.bit_generator.state


def test_validator_calls_no_environment_action_sampler_or_filesystem_writer(small_run, monkeypatch):
    run, protocol, _ = small_run
    lineage = trainer.load_published_lineage()
    before = {name: (run / name).read_bytes() for name in trainer.RUN_FILES}
    def fail(*args, **kwargs): raise AssertionError("Forbidden scenario, inference action or filesystem write")
    monkeypatch.setattr(trainer, "RobustPlacementEnv", fail)
    monkeypatch.setattr(trainer.rollout, "evaluate_slate", fail)
    monkeypatch.setattr(trainer.rollout, "propose_slate", fail)
    monkeypatch.setattr(trainer, "collect_episode", fail)
    monkeypatch.setattr(trainer, "load_published_lineage", fail)
    monkeypatch.setattr(RankPolicy, "act", fail)
    monkeypatch.setattr(RankPolicy, "probabilities", fail)
    monkeypatch.setattr(RankPolicy, "save", fail)
    monkeypatch.setattr(Path, "write_bytes", fail)
    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(Path, "mkdir", fail)
    trainer.validate_run(run, protocol, lineage=lineage)
    assert before == {name: (run / name).read_bytes() for name in trainer.RUN_FILES}


@pytest.mark.parametrize("field,value", [("episodes", 4097), ("batch_size", 3), ("hidden_size", True),
    ("temperature", .2), ("max_actions", 7), ("passes_per_batch", 0), ("policy_seeds", [True])])
def test_invalid_protocol_rejected_before_output(tmp_path, monkeypatch, field, value):
    protocol = make_protocol(tmp_path, monkeypatch)
    document = json.loads(protocol.read_text())
    document["training"][field] = value
    protocol.write_bytes(trainer._bytes(document))
    output = tmp_path / "output"
    with pytest.raises(ValueError): trainer.run_training(protocol, 499, output)
    assert not output.exists()


def test_source_and_prior_binding_fail_before_output(tmp_path, monkeypatch):
    protocol = make_protocol(tmp_path, monkeypatch)
    document = json.loads(protocol.read_text())
    document["prior_publication"]["sha256"] = "c" * 64
    protocol.write_bytes(trainer._bytes(document))
    with pytest.raises(ValueError, match="prior publication"):
        trainer.run_training(protocol, 499, tmp_path / "bad-prior")
    assert not (tmp_path / "bad-prior").exists()
    document["training_implementation_sha256"] = {}
    protocol.write_bytes(trainer._bytes(document))
    with pytest.raises(ValueError, match="source"):
        trainer.run_training(protocol, 499, tmp_path / "bad-source")
    assert not (tmp_path / "bad-source").exists()


def test_reused_output_is_never_overwritten(small_run):
    run, protocol, _ = small_run
    before = {name: (run / name).read_bytes() for name in trainer.RUN_FILES}
    with pytest.raises(ValueError): trainer.run_training(protocol, 499, run)
    assert before == {name: (run / name).read_bytes() for name in trainer.RUN_FILES}


@pytest.mark.parametrize("mutation", ["seed", "prefix", "slate_repeat", "live_action", "feature_shape", "feature_bool",
    "feature_value", "stop_flag", "pair_weight", "pair_reverse", "outcome_action", "outcome_return", "branch_steps",
    "placement_cost", "extra_field", "missing_state", "extra_state", "crc"])
def test_rehashed_dataset_tampering_rejected(small_run, tmp_path, mutation):
    run, protocol = copy_run(small_run, tmp_path)
    data = load_dataset(run)
    item = next(row for row in data if row["comparisons"])
    if mutation == "seed": item["seed"] += 1
    elif mutation == "prefix": item["prefix"] = [0]
    elif mutation == "slate_repeat": item["slate_actions"][1] = item["slate_actions"][0]
    elif mutation == "live_action": item["live_action"] = 16384
    elif mutation == "feature_shape": item["features"][0].pop()
    elif mutation == "feature_bool": item["features"][0][0] = True
    elif mutation == "feature_value":
        for row in item["features"]: row[trainer.FEATURE_NAMES.index("visibility")] += .125
    elif mutation == "stop_flag": item["features"][0][trainer.FEATURE_NAMES.index("stop")] = 0.
    elif mutation == "pair_weight": item["comparisons"][0]["weight"] *= .5
    elif mutation == "pair_reverse":
        pair = item["comparisons"][0]; pair["preferred"], pair["rejected"] = pair["rejected"], pair["preferred"]
    elif mutation == "outcome_action": item["outcomes"][0]["action"] += 1
    elif mutation == "outcome_return":
        terminal = data[-1]
        terminal["outcomes"][terminal["slate_actions"].index(terminal["live_action"])]["episode_return"] += .5
    elif mutation == "branch_steps": item["outcomes"][0]["steps"] += 1
    elif mutation == "placement_cost": item["outcomes"][1]["placements"][0]["cost"] += .1
    elif mutation == "extra_field": item["private"] = 1
    elif mutation == "missing_state": data.pop()
    elif mutation == "extra_state": data.append(deepcopy(data[-1]))
    blob = gzip.compress(b"".join(trainer._bytes(row) for row in data), mtime=0)
    if mutation == "crc": blob = blob[:-8] + bytes([blob[-8] ^ 1]) + blob[-7:]
    (run / "comparisons.jsonl.gz").write_bytes(blob)
    rebind(run)
    with pytest.raises(ValueError): trainer.validate_run(run, protocol)


@pytest.mark.parametrize("mutation", ["cadence", "counts", "update_metric", "flat_metric", "profile_counts", "sensor_counts",
    "live_return", "live_invalid", "live_bool", "mean_cost", "negative_time", "extra_field"])
def test_rehashed_log_tampering_rejected(small_run, tmp_path, mutation):
    run, protocol = copy_run(small_run, tmp_path)
    rows = load_log(run); row = rows[0]
    if mutation == "cadence": row["episode_start"] += 1
    elif mutation == "counts": row["branch_rollouts"] += 1
    elif mutation == "update_metric": row["updates"][0]["pairwise_loss"] += .1
    elif mutation == "flat_metric": row["mean_update_metrics"]["pairwise_loss"] += .1
    elif mutation == "profile_counts": row["case_profile_counts"] = {"normal": 2}
    elif mutation == "sensor_counts": row["sensor_deployment_counts"] = {"rf": 999}
    elif mutation == "live_return": row["episodes"][0]["episode_return"] += .25
    elif mutation == "live_invalid": row["episodes"][0]["invalid_actions"] = 1
    elif mutation == "live_bool": row["episodes"][0]["seed"] = True
    elif mutation == "mean_cost": row["mean_cost"] += .2
    elif mutation == "negative_time": row["collection_wall_seconds"] = -1.
    else: row["private"] = 1
    (run / "training.jsonl").write_bytes(b"".join(trainer._bytes(value) for value in rows))
    rebind(run)
    with pytest.raises(ValueError): trainer.validate_run(run, protocol)


@pytest.mark.parametrize("mutation", ["summary_mean", "summary_count", "summary_hash", "summary_extra", "summary_time",
    "config_source", "protocol_bytes", "extra_file", "missing_file", "endpoint_parameter", "endpoint_adam", "endpoint_rng",
    "initializer"])
def test_run_and_endpoint_tampering_rejected(small_run, tmp_path, mutation):
    run, protocol = copy_run(small_run, tmp_path)
    summary = json.loads((run / "summary.json").read_text())
    if mutation.startswith("summary"):
        if mutation == "summary_mean": summary["mean_timely_fraction"] += .01
        elif mutation == "summary_count": summary["branches"] += 1
        elif mutation == "summary_hash": summary["evidence_files_sha256"]["training.jsonl"] = "0" * 64
        elif mutation == "summary_extra": summary["private"] = 1
        else: summary["total_trainer_wall_seconds"] = 0.
        (run / "summary.json").write_bytes(trainer._bytes(summary))
    elif mutation == "config_source":
        config = json.loads((run / "config.json").read_text()); config["source_sha256"] = {}
        (run / "config.json").write_bytes(trainer._bytes(config))
    elif mutation == "protocol_bytes":
        (run / "protocol.json").write_bytes((run / "protocol.json").read_bytes() + b" ")
    elif mutation == "extra_file": (run / "unexpected").write_bytes(b"x")
    elif mutation == "missing_file": (run / "initialized/arrays.npz").unlink()
    elif mutation == "initializer":
        actor = RankPolicy.load(run / "initialized"); actor.parameters["wa"][0] += .1; actor.save(run / "initialized")
    else:
        actor = RankPolicy.load(run / "last")
        if mutation == "endpoint_parameter": actor.parameters["w1"][0, 0] += .01
        elif mutation == "endpoint_adam": actor.adam_m["w1"][0, 0] += .01
        else: actor.rng.random()
        actor.save(run / "last")
        rows = load_log(run); rows[-1]["weights_sha256"] = actor.weights_fingerprint()
        (run / "training.jsonl").write_bytes(b"".join(trainer._bytes(row) for row in rows))
        rebind(run)
    with pytest.raises(ValueError): trainer.validate_run(run, protocol)


def test_validator_rejects_artifact_symlink_alias(small_run, tmp_path):
    run, protocol = copy_run(small_run, tmp_path)
    original = run / "initialized/arrays.npz"
    original.unlink()
    try: original.symlink_to(run / "last/arrays.npz")
    except OSError: pytest.skip("Symlink creation unavailable")
    with pytest.raises(ValueError, match="independent"):
        trainer.validate_run(run, protocol)


def test_all_tie_states_are_archived_but_skip_optimizer(tmp_path, monkeypatch):
    protocol = make_protocol(tmp_path, monkeypatch, episodes=2, batch_size=2)
    def only_stop(env, actor, v3, seed):
        features = np.zeros((1, len(trainer.FEATURE_NAMES)))
        features[0, trainer.FEATURE_NAMES.index("bias")] = 1.
        features[0, trainer.FEATURE_NAMES.index("stop")] = 1.
        outcome = {"action": 0, "timely_fraction": 0., "detection_fraction": 0., "episode_return": -10.5,
                   "cost": 0., "placements": [], "steps": 1, "continuation_steps": 0}
        state = {"seed": seed, "prefix": [], "slate_actions": [0], "live_action": 0,
                 "public_observation_sha256": "a" * 64, "features": features.tolist(),
                 "comparisons": [], "outcomes": [outcome]}
        record = {"seed": seed, "profile": "normal", "decisions": 1, "branches": 1, "comparisons": 0,
                  "training_states": 0, "timely_fraction": 0., "detection_fraction": 0., "episode_return": -10.5,
                  "cost": 0, "success": False, "invalid_actions": 0, "placements": []}
        return [state], record
    monkeypatch.setattr(trainer, "collect_episode", only_stop)
    run = tmp_path / "all-ties"
    summary = trainer.run_training(protocol, 499, run)
    actor, evidence = trainer.validate_run(run, protocol)
    assert actor.update_count == 0
    assert summary["decisions"] == 2 and summary["training_states"] == summary["comparisons"] == 0
    assert len(load_dataset(run)) == 2
    assert load_log(run)[0]["updates"] == []
    assert load_log(run)[0]["mean_update_metrics"] == {}
    assert summary["initial_weights_sha256"] == summary["last_weights_sha256"]
    assert evidence["verification_scope"]["optimizer_updates_replayed"] == 0
