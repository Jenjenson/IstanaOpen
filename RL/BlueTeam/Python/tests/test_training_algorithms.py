"""Algorithm mathematics and native public-contract/checkpoint integration."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from triad_rl.directional_inputs import BOSON_PLUS_640_18MM, FEATURE_NAMES, build_observation
from triad_rl.istana_live import public_planning_inputs
from triad_rl.training_algorithms import (ALGORITHMS, algorithm_catalogue, create_algorithm,
                                          generalized_advantages, load_algorithm,
                                          validate_algorithm_config)


@pytest.fixture
def context():
    examples = Path(__file__).resolve().parents[2] / "Examples"
    state = json.loads((examples / "public-snapshot.json").read_text())
    state.update(timestamp=0, placements=[], done=False, budget_total=2., budget_remaining=2.,
                 max_sites=3, sites=[[60., 0.], [0., 60.], [-60., 0.]], blocked_sites=[],
                 available_sensor_ids=["thermal"])
    return {"completedSteps": 0, "committed": False, "coordinateSystem": "unreal_xy_relative_m_z_up",
            "catalogue": [deepcopy(BOSON_PLUS_640_18MM)], "publicSnapshot": state,
            "temporalConfig": json.loads((examples / "temporal-config.json").read_text()),
            "sensorModel": "native_directional_test_contract", "trainingConfigurationVersion": 1}


def test_registry_only_offers_real_native_algorithms():
    assert set(ALGORITHMS) == {"reinforce", "ppo"}
    rows = algorithm_catalogue()
    rows[0]["default_config"]["learning_rate"] = 99
    assert ALGORITHMS["reinforce"]["default_config"]["learning_rate"] == .12
    assert "value_loss" not in ALGORITHMS["reinforce"]["supported_metrics"]
    with pytest.raises(ValueError, match="Unsupported"):
        validate_algorithm_config("a2c")


@pytest.mark.parametrize("algorithm,config", [
    ("reinforce", {"learning_rate": 0}), ("reinforce", {"unknown": 3}),
    ("ppo", {"hidden_size": True}), ("ppo", {"epochs": 2.5}),
    ("ppo", {"gae_lambda": 1.1}), ("ppo", {"clip_ratio": 0}),
    ("ppo", {"entropy_coef": float("nan")}), ("ppo", {"normalize_advantages": 1}),
])
def test_invalid_hyperparameters_fail_before_training(algorithm, config):
    with pytest.raises(ValueError):
        validate_algorithm_config(algorithm, config)


@pytest.mark.parametrize("algorithm", ["reinforce", "ppo"])
def test_planning_public_only_directional_budget_and_masks(algorithm, context):
    adapter = create_algorithm(algorithm, context, seed=18)
    original = deepcopy(context)
    poisoned = deepcopy(context)
    poisoned.update(private_red_truth={"position": [99, 400, 900]}, reward=123456,
                    warningEvidenceForEvaluationOnly=[{"firstDetectionSeconds": 0}])
    for seed in range(8):
        placements, records = adapter.plan(context, rng=np.random.default_rng(seed))
        other, _ = adapter.plan(poisoned, rng=np.random.default_rng(seed))
        assert placements == other
        assert len(placements) <= 2
        assert len({row["siteId"] for row in placements}) == len(placements)
        assert all(row["yawDeg"] in BOSON_PLUS_640_18MM["yaw_bins_deg"]
                   and row["pitchDeg"] in BOSON_PLUS_640_18MM["pitch_bins_deg"] for row in placements)
        for row in records:
            if algorithm == "ppo":
                assert row["mask"][row["action"]]
                _, p, _ = adapter.network._forward(row["features"], row["mask"])
                assert np.all(p[~row["mask"]] == 0)
                assert np.isclose(np.exp(row["old_log_probability"]), p[row["action"]])
            else:
                assert row["probabilities"][row["action"]] > 0
    assert context == original


def test_gae_complete_terminal_returns_and_lambda_zero():
    advantages, returns = generalized_advantages([0, 0, 10], [1, 2, 3], 1., 1.)
    np.testing.assert_allclose(advantages, [9, 8, 7])
    np.testing.assert_allclose(returns, [10, 10, 10])
    advantages, returns = generalized_advantages([0, 0, 10], [1, 2, 3], .9, 0.)
    np.testing.assert_allclose(advantages, [.8, .7, 7])
    np.testing.assert_allclose(returns, [1.8, 2.7, 10])
    with pytest.raises(ValueError):
        generalized_advantages([0], [1, 2])


@pytest.mark.parametrize("advantage,ratio", [(2., 1.), (2., 1.5), (-2., .5), (-2., 1.5)])
def test_ppo_exact_gradients_match_finite_differences_including_clipping(context, advantage, ratio):
    adapter = create_algorithm("ppo", context, seed=7, config={"hidden_size": 2})
    rng = np.random.default_rng(222)
    x = rng.normal(0, .2, (3, len(FEATURE_NAMES)))
    mask = np.array([True, False, True])
    adapter.network.parameters["wv"][:] = [.17, -.12]
    _, p, _ = adapter.network._forward(x, mask)
    record = {"features": x, "mask": mask, "action": 0,
              "old_log_probability": np.log(p[0]) - np.log(ratio)}
    args = ([record], np.array([1.7]), np.array([advantage]))
    _, gradients, _ = adapter._loss_and_gradients(*args)
    epsilon = 1e-6
    for key, parameter in adapter.network.parameters.items():
        numerical = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            before = parameter[index]
            parameter[index] = before + epsilon
            plus = adapter._loss_and_gradients(*args)[0]
            parameter[index] = before - epsilon
            minus = adapter._loss_and_gradients(*args)[0]
            parameter[index] = before
            numerical[index] = (plus - minus) / (2 * epsilon)
        np.testing.assert_allclose(gradients[key], numerical, rtol=2e-5, atol=2e-8, err_msg=key)


@pytest.mark.parametrize("advantage,ratio", [(3., 1.5), (-3., .5)])
def test_ppo_clipping_removes_actor_gradient_when_improvement_exceeds_limit(context, advantage, ratio):
    adapter = create_algorithm("ppo", context, config={"hidden_size": 2, "entropy_coef": 0., "value_coef": 0.})
    x = np.random.default_rng(9).normal(size=(3, len(FEATURE_NAMES)))
    mask = np.array([True, True, False])
    _, p, _ = adapter.network._forward(x, mask)
    rows = [{"features": x, "mask": mask, "action": 0, "old_log_probability": np.log(p[0]) - np.log(ratio)}]
    _, gradients, metrics = adapter._loss_and_gradients(rows, [0.], [advantage])
    assert all(np.count_nonzero(value) == 0 for value in gradients.values())
    assert metrics["clip_fraction"] == 1.
    assert metrics["policy_loss"] == pytest.approx(-np.clip(ratio, .8, 1.2) * advantage)


def test_ppo_update_improves_rewarded_action_and_learns_a_real_critic(context):
    adapter = create_algorithm("ppo", context, seed=12, config={
        "epochs": 1, "learning_rate": .01, "entropy_coef": 0., "normalize_advantages": False})
    state, catalogue, _ = public_planning_inputs(context)
    obs = build_observation(state, catalogue)
    x, mask = obs["option_features"].astype(float), obs["action_mask"]
    action = int(np.flatnonzero(mask)[0])
    _, before_p, before_v = adapter.network._forward(x, mask)
    row = {"features": x, "mask": mask, "action": action, "value": before_v,
           "old_log_probability": np.log(before_p[action]), "generation": 0}
    metrics = adapter.update([([row], 10.)])
    _, after_p, after_v = adapter.network._forward(x, mask)
    assert after_p[action] > before_p[action]
    assert after_v > before_v
    assert metrics["value_loss"] == pytest.approx(50.)
    assert metrics["entropy"] > 0
    assert metrics["updates"] == 1
    with pytest.raises(ValueError, match="stale"):
        adapter.update([([row], 10.)])


@pytest.mark.parametrize("algorithm", ["reinforce", "ppo"])
def test_checkpoint_restores_optimizer_rng_and_exact_next_update(algorithm, context, tmp_path):
    adapter = create_algorithm(algorithm, context, seed=12)
    episodes = [(adapter.plan(context)[1], float(i)) for i in range(4)]
    metrics = adapter.update(episodes)
    assert metrics["value_loss"] is None if algorithm == "reinforce" else metrics["value_loss"] >= 0
    path = tmp_path / "model.json"
    checksum = adapter.save(path)
    assert checksum == hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_algorithm(path, context)
    assert loaded.metadata == adapter.metadata
    left, right = [], []
    for i in range(3):
        a, a_records = adapter.plan(context)
        b, b_records = loaded.plan(context)
        assert a == b
        left.append((a_records, float(i + 4)))
        right.append((b_records, float(i + 4)))
    assert adapter.update(left) == loaded.update(right)
    adapter.save(path)  # latest is atomically replaceable
    other = tmp_path / "replica.json"
    loaded.save(other)
    assert path.read_bytes() == other.read_bytes()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("change", ["budget", "catalogue", "site", "temporal", "model", "availability", "architecture", "mount_geometry"])
@pytest.mark.parametrize("algorithm", ["reinforce", "ppo"])
def test_incompatible_models_fail_with_a_useful_error(algorithm, context, tmp_path, change):
    adapter = create_algorithm(algorithm, context)
    path = tmp_path / "model.json"
    adapter.save(path)
    other = deepcopy(context)
    if change == "budget":
        other["publicSnapshot"].update(budget_total=3., budget_remaining=3.)
    elif change == "catalogue":
        other["catalogue"][0]["simulation_assumptions"]["pixels_for_63_percent"] = 4.
    elif change == "site":
        other["publicSnapshot"]["sites"][0][0] += 1
    elif change == "temporal":
        other["temporalConfig"]["required_confirmations"] += 1
    elif change == "model":
        other["sensorModel"] = "different_environment"
    elif change == "availability":
        other["publicSnapshot"]["available_sensor_ids"] = []
    elif change == "architecture":
        data = json.loads(path.read_text())
        data["architecture"]["kind"] = "some_other_model"
        path.write_text(json.dumps(data))
    elif change == "mount_geometry":
        other["siteSurfacesWorldCm"] = [[6000., 0., 200.], [0., 6000., 0.], [-6000., 0., 0.]]
    with pytest.raises(ValueError, match="[Ii]ncompatible"):
        load_algorithm(path, other)


def test_corrupt_checkpoint_is_rejected(context, tmp_path):
    path = tmp_path / "model.json"
    create_algorithm("ppo", context).save(path)
    data = json.loads(path.read_text())
    data["model"]["parameters"]["bv"][0] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        load_algorithm(path, context)
