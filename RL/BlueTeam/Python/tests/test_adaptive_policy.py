"""Gradient, schema, safety, reproducibility and learning checks for the policy."""
import hashlib
import json

import numpy as np
import pytest

from triad_rl.adaptive_policy import AdaptivePolicy, FEATURE_SCHEMA


def observation(rows=5):
    return {"feature_names": ("a", "b", "stop"),
            "feature_schema": FEATURE_SCHEMA,
            "option_features": np.random.default_rng(11).normal(size=(rows, 3)),
            "action_mask": np.ones(rows, dtype=bool)}


def batch(policy, count=4):
    result = []
    for index in range(count):
        episode = []
        for step in range(2):
            obs = observation(3 + index)
            action, record = policy.sample(obs)
            record["reward"] = float(obs["option_features"][action, 0] + step)
            episode.append(record)
        result.append(episode)
    return result


def test_mask_and_variable_option_permutation():
    obs = observation(9)
    obs["action_mask"][[0, 3, 6]] = False
    policy = AdaptivePolicy(obs["feature_names"], seed=1)
    probability = policy.probabilities(obs)
    assert probability.sum() == pytest.approx(1)
    assert np.all(probability[~obs["action_mask"]] == 0)
    assert all(obs["action_mask"][policy.act(obs, deterministic=False)] for _ in range(100))
    permutation = np.array([4, 0, 2, 8, 3, 1, 7, 6, 5])
    permuted = {**obs, "option_features": obs["option_features"][permutation],
                "action_mask": obs["action_mask"][permutation]}
    np.testing.assert_allclose(policy.probabilities(permuted), probability[permutation], atol=1e-15)
    assert policy.probabilities(observation(17)).shape == (17,)


@pytest.mark.parametrize("mutation", ["names", "nonfinite", "shape", "mask", "empty_mask", "semantics"])
def test_observation_contract_rejects_mismatches(mutation):
    obs = observation()
    policy = AdaptivePolicy(obs["feature_names"])
    if mutation == "names":
        obs["feature_names"] = ("b", "a", "stop")
    elif mutation == "nonfinite":
        obs["option_features"][0, 0] = np.nan
    elif mutation == "shape":
        obs["option_features"] = np.zeros((5, 4))
    elif mutation == "mask":
        obs["action_mask"] = np.ones(5, dtype=int)
    elif mutation == "empty_mask":
        obs["action_mask"][:] = False
    else:
        obs["feature_schema"] = "new-meaning-same-columns"
    with pytest.raises(ValueError):
        policy.act(obs)


def test_analytic_gradients_match_finite_difference_including_entropy_and_critic():
    obs = observation(4)
    obs["action_mask"][1] = False
    policy = AdaptivePolicy(obs["feature_names"], seed=7, hidden_size=3)
    policy.parameters["wv"][:] = [0.2, -0.1, 0.3]
    _, record = policy.sample(obs)
    returns, advantages = np.array([0.75]), np.array([-0.65])
    _, gradients, _ = policy._loss_and_gradients([record], returns, advantages, 0.04, 0.6)
    epsilon = 1e-6
    for key, parameter in policy.parameters.items():
        numerical = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + epsilon
            positive = policy._loss_and_gradients([record], returns, advantages, 0.04, 0.6)[0]
            parameter[index] = original - epsilon
            negative = policy._loss_and_gradients([record], returns, advantages, 0.04, 0.6)[0]
            parameter[index] = original
            numerical[index] = (positive - negative) / (2 * epsilon)
        np.testing.assert_allclose(gradients[key], numerical, rtol=1e-5, atol=1e-8)


def test_real_policy_gradient_improves_synthetic_joint_action_reward():
    # Randomized row order prevents learning an action-index shortcut.
    policy = AdaptivePolicy(("quality", "cost", "stop"), seed=9, hidden_size=8)
    rng = np.random.default_rng(90)
    probe = {"feature_names": policy.feature_names,
             "option_features": np.array([[-1, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=float),
             "action_mask": np.ones(3, dtype=bool)}
    initial = policy.probabilities(probe)[1]
    for _ in range(80):
        episodes = []
        for _ in range(12):
            permutation = rng.permutation(3)
            obs = {**probe, "option_features": probe["option_features"][permutation]}
            action, record = policy.sample(obs)
            record["reward"] = float(obs["option_features"][action, 0])
            episodes.append([record])
        policy.update(episodes, learning_rate=0.02, entropy_coef=0.01)
    assert policy.probabilities(probe)[1] > 0.97
    assert policy.probabilities(probe)[1] > initial + 0.5


def test_checkpoint_roundtrip_resumes_adam_rng_and_weights_exactly(tmp_path):
    policy = AdaptivePolicy(observation()["feature_names"], seed=5, hidden_size=8)
    policy.update(batch(policy))
    policy.save(tmp_path / "checkpoint", {"completed_episodes": 4})
    resumed = AdaptivePolicy.load(tmp_path / "checkpoint", policy.feature_names)
    assert resumed.weights_fingerprint() == policy.weights_fingerprint()
    assert resumed.training_state["completed_episodes"] == 4
    original_batch, resumed_batch = batch(policy), batch(resumed)
    assert [[r["action"] for r in ep] for ep in original_batch] == [[r["action"] for r in ep] for ep in resumed_batch]
    assert policy.update(original_batch) == resumed.update(resumed_batch)
    assert resumed.weights_fingerprint() == policy.weights_fingerprint()
    for key in policy.parameters:
        np.testing.assert_array_equal(policy.adam_m[key], resumed.adam_m[key])
        np.testing.assert_array_equal(policy.adam_v[key], resumed.adam_v[key])


@pytest.mark.parametrize("corruption", ["bytes", "nonfinite", "features", "schema", "weights"])
def test_checkpoint_rejects_corruption(tmp_path, corruption):
    policy = AdaptivePolicy(observation()["feature_names"])
    policy.save(tmp_path)
    path = tmp_path / "checkpoint.json"
    metadata = json.loads(path.read_text())
    if corruption == "bytes":
        with (tmp_path / "arrays.npz").open("ab") as handle:
            handle.write(b"corrupted")
    elif corruption == "nonfinite":
        with np.load(tmp_path / "arrays.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["param_w1"][0, 0] = np.nan
        np.savez_compressed(tmp_path / "arrays.npz", **arrays)
        metadata["arrays_sha256"] = hashlib.sha256((tmp_path / "arrays.npz").read_bytes()).hexdigest()
    elif corruption == "features":
        metadata["feature_names"] = list(reversed(metadata["feature_names"]))
    elif corruption == "schema":
        metadata["feature_schema"] = "different-semantics"
    else:
        metadata["weights_sha256"] = "0" * 64
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        AdaptivePolicy.load(tmp_path)


def test_invalid_training_data_does_not_mutate_policy():
    policy = AdaptivePolicy(observation()["feature_names"])
    records = batch(policy)
    before = policy.weights_fingerprint()
    records[0][0]["reward"] = float("inf")
    with pytest.raises(ValueError):
        policy.update(records)
    assert policy.weights_fingerprint() == before
    assert policy.update_count == 0


def test_reward_to_go_and_gradient_clipping():
    policy = AdaptivePolicy(observation()["feature_names"], seed=1)
    _, first = policy.sample(observation())
    _, second = policy.sample(observation())
    first["reward"], second["reward"] = 1.0, 3.0
    metrics = policy.update([[first, second]], gamma=0.5, max_grad_norm=0.01)
    assert metrics["mean_return_to_go"] == pytest.approx((2.5 + 3) / 2)
    assert metrics["gradient_norm"] * metrics["gradient_scale"] <= 0.01 + 1e-12


def test_deterministic_validation_does_not_consume_training_rng():
    policy = AdaptivePolicy(observation()["feature_names"], seed=3)
    state = json.dumps(policy.rng.bit_generator.state, sort_keys=True)
    for _ in range(5):
        policy.act(observation(), deterministic=True)
    assert json.dumps(policy.rng.bit_generator.state, sort_keys=True) == state


def test_training_seed_bands_do_not_overlap():
    from triad_rl.train_adaptive import scenario_seed
    assert scenario_seed("train", 999999, 999999) < scenario_seed("validation", 0, 0)
    assert scenario_seed("validation", 999999, 999999) < scenario_seed("test", 0, 0)
    with pytest.raises(ValueError):
        scenario_seed("train", 0, 1_000_000)


def test_trainer_exact_resume_on_full_batch_boundary(tmp_path):
    from triad_rl.train_adaptive import run_training
    options = {"batch_size": 2, "seed": 3, "hidden_size": 4,
               "validation_every": 2, "validation_episodes": 2}
    run_training(output=tmp_path / "full", episodes=4, **options)
    run_training(output=tmp_path / "resumed", episodes=2, **options)
    run_training(output=tmp_path / "resumed", episodes=4,
                 resume=tmp_path / "resumed" / "last", **options)
    full = AdaptivePolicy.load(tmp_path / "full" / "last")
    resumed = AdaptivePolicy.load(tmp_path / "resumed" / "last")
    assert full.weights_fingerprint() == resumed.weights_fingerprint()
    assert full.training_state == resumed.training_state
