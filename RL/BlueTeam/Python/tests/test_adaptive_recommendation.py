"""Same frozen policy, same decisions from simulated and external public data."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

from triad_rl.adaptive_env import AdaptivePlacementEnv
from triad_rl.adaptive_inputs import LiveObservationAdapter
from triad_rl.adaptive_policy import AdaptivePolicy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recommend_adaptive import recommend_layout


def deployment_policy(observation):
    policy = AdaptivePolicy(observation["feature_names"], seed=7, hidden_size=4)
    # A nontrivial, deterministic fixture encourages deployment; it is not
    # published as a trained model or used by the learning benchmark.
    policy.parameters["w1"][:] = 0
    policy.parameters["w1"][list(policy.feature_names).index("stop"), :] = -2
    policy.parameters["wa"][:] = 1
    return policy


def test_external_snapshot_and_simulator_produce_identical_layouts():
    env = AdaptivePlacementEnv(seed=88)
    obs = env.reset(seed=88)
    policy = deployment_policy(obs)
    public = deepcopy(env.public_state)
    original = deepcopy(public)
    public["source"] = "external-test-provider"
    report = recommend_layout(policy, public, catalogue=env.catalogue, now=0)
    assert public == {**original, "source": "external-test-provider"}
    assert report["physical_commands_sent"] is False
    assert len(report["new_placements"]) >= 1
    while not env.done:
        obs, _, _, _ = env.step(policy.act(obs))
    assert report["final_public_state"]["placements"] == env.placements
    assert report["final_public_state"]["budget_remaining"] == pytest.approx(env.public_state["budget_remaining"])
    assert report["checkpoint_weights_sha256"] == policy.weights_fingerprint()


def test_external_clock_removes_stale_tracks_without_changing_contract():
    env = AdaptivePlacementEnv(seed=22)
    payload = deepcopy(env.public_state)
    payload["tracks"] = [{"id": "stale", "position": [100, 0, 20], "timestamp": 0}]
    adapter = LiveObservationAdapter(env.catalogue)
    fresh = adapter.observe(payload, now=0)
    old = adapter.observe(payload, now=100)
    assert len(fresh["state"]["tracks"]) == 1
    assert old["state"]["tracks"] == []
    assert old["feature_names"] == fresh["feature_names"]
    report = recommend_layout(deployment_policy(fresh), payload, catalogue=env.catalogue, now=100)
    assert report["final_public_state"]["tracks"] == []


def test_new_sensor_and_new_approved_sites_need_no_network_resize():
    env = AdaptivePlacementEnv(seed=22)
    original = env.observe()
    policy = deployment_policy(original)
    catalogue = deepcopy(env.catalogue)
    catalogue.append({**deepcopy(catalogue[0]), "id": "rf-new", "cost": .9})
    state = deepcopy(env.public_state)
    state["available_sensor_ids"] = ["rf-new"]
    state["sites"] = [[80, 10], [-75, 20], [45, -50]]
    state["blocked_sites"] = [1]
    observation = LiveObservationAdapter(catalogue).observe(state)
    assert observation["option_features"].shape[1] == len(policy.feature_names)
    assert len(observation["options"]) != len(original["options"])
    report = recommend_layout(policy, state, catalogue=catalogue)
    assert report["new_placements"]
    assert all(p["sensor_id"] == "rf-new" for p in report["new_placements"])
    assert all(p["site_index"] != 1 for p in report["new_placements"])
    assert np.isfinite(observation["option_features"]).all()
