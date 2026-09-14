"""STOP exploration probes inspect public snapshots and preserve frozen actors."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe_stop_exploration as probe
from triad_rl.adaptive_inputs import FEATURE_NAMES, build_observation
from triad_rl.adaptive_policy import AdaptivePolicy
from triad_rl.robust_scenarios import RobustPlacementEnv


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "checkpoint"
    AdaptivePolicy(FEATURE_NAMES, seed=12).save(path)
    return path


@pytest.mark.parametrize("seed,episodes", [
    (0, 2), (10 ** 15 - 1, 1), (2 * 10 ** 15, 1), (2 * 10 ** 15 - 1, 2),
    (True, 2), (1.1, 2), (None, 2), (probe.DEFAULT_PROBE_SEED, 0),
    (probe.DEFAULT_PROBE_SEED, -1), (probe.DEFAULT_PROBE_SEED, True),
    (probe.DEFAULT_PROBE_SEED, 1.5),
])
def test_reject_training_final_and_invalid_seed_ranges_before_loading(tmp_path, seed, episodes):
    with pytest.raises(ValueError, match="validation-only"):
        probe.run_probe(tmp_path / "must-not-load", seed=seed, episodes=episodes)


def test_profile_rejected_before_checkpoint_loading(tmp_path):
    with pytest.raises(ValueError, match="profile"):
        probe.run_probe(tmp_path / "must-not-load", profile="test")


def test_reproducible_public_only_wrapper_blocks_truth_and_rollouts(checkpoint, monkeypatch):
    original = RobustPlacementEnv
    observed_seeds = []

    class PublicOnlyEnvironment:
        def __init__(self, *, seed, profile):
            assert probe.VALIDATION_BASE <= seed < probe.FINAL_TEST_BASE
            self.inner = original(seed=seed, profile=profile)

        def reset(self, *, seed):
            observed_seeds.append(seed)
            return self.inner.reset(seed=seed)

        def __getattr__(self, name):
            raise AssertionError(f"Probe attempted forbidden environment access: {name}")

        def step(self, *args):
            raise AssertionError("Probe must not step a simulation")

        def _simulate(self):
            raise AssertionError("Probe must not simulate outcomes")

        @property
        def scenario(self):
            raise AssertionError("Probe must not inspect truth")

    monkeypatch.setattr(probe, "RobustPlacementEnv", PublicOnlyEnvironment)
    before = probe.checkpoint_fingerprints(checkpoint)
    first = probe.run_probe(checkpoint, episodes=3)
    second = probe.run_probe(checkpoint, episodes=3)
    assert first == second
    assert observed_seeds == list(range(probe.DEFAULT_PROBE_SEED, probe.DEFAULT_PROBE_SEED + 3)) * 2
    assert len(first["records"]) == 3 and first["groups"]["all"]["count"] == 3
    assert first["checkpoint_files_sha256_before"] == first["checkpoint_files_sha256_after"] == before
    assert first["source_sha256_before"] == first["source_sha256_after"]
    assert first["policy_rng_sha256_before"] == first["policy_rng_sha256_after"]
    assert not first["simulation_rollouts_performed"] and not first["private_truth_accessed"]
    assert not first["training_performed"] and not first["final_test_accessed"]
    assert not first["independent_unseen_evidence"]
    assert "probe_stop_exploration.py" in first["source_sha256_before"]
    assert "triad_rl/robust_scenarios.py" in first["source_sha256_before"]
    assert "triad_rl/adaptive_policy.py" in first["source_sha256_before"]
    assert str(checkpoint) not in json.dumps(first)
    json.dumps(first, allow_nan=False)


def test_first_action_measures_existing_public_distribution_only(checkpoint):
    policy = AdaptivePolicy.load(checkpoint)
    observation = RobustPlacementEnv(seed=probe.DEFAULT_PROBE_SEED, profile="stress").observe()
    original = deepcopy(observation)
    rng = deepcopy(policy.rng.bit_generator.state)
    probability = policy.probabilities(observation)
    row = probe.first_action_record(policy, observation, seed=probe.DEFAULT_PROBE_SEED)
    action = int(np.argmax(probability))
    assert row["first_action"]["index"] == action
    assert row["deterministic_stop"] == observation["options"][action]["stop"]
    assert row["stop_probability"] == probability[-1]
    assert row["legal_deployment_count"] == observation["action_mask"][:-1].sum()
    assert row["categorical_entropy_nats"] == pytest.approx(-sum(p * np.log(p) for p in probability if p > 0))
    sensor_id = row["first_action"]["sensor_id"]
    expected_cost = next((sensor["cost"] for sensor in observation["catalogue"] if sensor["id"] == sensor_id), 0.)
    assert row["first_action"]["cost"] == expected_cost
    assert "final_cost" not in row and "final_placements" not in row and "success" not in row
    assert policy.rng.bit_generator.state == rng
    np.testing.assert_array_equal(observation["option_features"], original["option_features"])
    assert observation["state"] == original["state"]


def test_stop_only_counted_separately_and_empty_groups_json_safe(checkpoint):
    policy = AdaptivePolicy.load(checkpoint)
    observation = RobustPlacementEnv(seed=probe.DEFAULT_PROBE_SEED, profile="stress").observe()
    state = deepcopy(observation["state"])
    state["available_sensor_ids"] = []
    row = probe.first_action_record(policy, build_observation(state, observation["catalogue"]),
                                   seed=probe.DEFAULT_PROBE_SEED)
    assert row["forced_stop"] and row["deterministic_stop"]
    assert row["stop_probability"] == 1 and row["categorical_entropy_nats"] == 0
    assert row["max_legal_marginal_coverage"] == 0 and row["legal_deployment_count"] == 0
    assert row["first_action"]["cost"] == 0
    assert row["stop_versus_best_deployment_probability"] is None
    summary = probe.summarize_records([row])
    assert summary["forced_stop_count"] == summary["deterministic_stop_count"] == 1
    assert summary["deterministic_stop_with_deployment_available_count"] == 0
    empty = probe.summarize_records([])
    assert empty["count"] == 0 and empty["stop_probability"] is None
    json.dumps([summary, empty], allow_nan=False)


def test_groups_recomputed_from_real_records_and_boundary_seeds_valid(checkpoint):
    report = probe.run_probe(checkpoint, episodes=4, profile="capability")
    for name, threshold in (("coverage_le_0.01", .01), ("coverage_le_0.001", .001), ("coverage_zero", 0.)):
        expected = [row for row in report["records"] if row["max_legal_marginal_coverage"] <= threshold]
        assert report["groups"][name] == probe.summarize_records(expected)
    for seed in (probe.VALIDATION_BASE, probe.FINAL_TEST_BASE - 1):
        one = probe.run_probe(checkpoint, seed=seed, episodes=1)
        assert one["records"][0]["scenario_seed"] == seed


@pytest.mark.parametrize("mutation", ["rng", "weights"])
def test_unexpected_policy_mutation_rejected(checkpoint, monkeypatch, mutation):
    policy = AdaptivePolicy.load(checkpoint)
    original = policy.probabilities

    def probabilities(observation):
        result = original(observation)
        if mutation == "rng":
            policy.rng.random()
        else:
            policy.parameters["wa"][0] += .01
        return result

    monkeypatch.setattr(policy, "probabilities", probabilities)
    monkeypatch.setattr(probe.AdaptivePolicy, "load", lambda *args, **kwargs: policy)
    with pytest.raises(RuntimeError, match="weights or RNG"):
        probe.run_probe(checkpoint, episodes=1)


def test_source_drift_rejected(checkpoint, monkeypatch):
    calls = []

    def fingerprints():
        calls.append(True)
        return {"probe": str(len(calls))}

    monkeypatch.setattr(probe, "source_fingerprints", fingerprints)
    with pytest.raises(RuntimeError, match="Implementation changed"):
        probe.run_probe(checkpoint, episodes=1)


def test_checkpoint_metadata_drift_rejected(checkpoint, monkeypatch):
    original = probe.AdaptivePolicy.probabilities

    def probabilities(self, observation):
        result = original(self, observation)
        path = checkpoint / "checkpoint.json"
        path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(probe.AdaptivePolicy, "probabilities", probabilities)
    with pytest.raises(RuntimeError, match="Checkpoint files changed"):
        probe.run_probe(checkpoint, episodes=1)


def test_cli_writes_full_report_and_refuses_overwrite(checkpoint, tmp_path):
    output = tmp_path / "report.json"
    args = ["--checkpoint", str(checkpoint), "--output", str(output), "--episodes", "2"]
    assert probe.main(args) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == probe.PROBE_SCHEMA and len(report["records"]) == 2
    before = output.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        probe.main(args)
    assert output.read_bytes() == before
