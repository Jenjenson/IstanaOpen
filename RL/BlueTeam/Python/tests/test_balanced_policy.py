"""Learning, count invariance, exact gradients and safe versioned persistence."""
import hashlib
import json

import numpy as np
import pytest

from triad_rl.adaptive_policy import AdaptivePolicy, FEATURE_SCHEMA
from triad_rl.balanced_policy import BalancedPolicy, CHECKPOINT_SCHEMA, POLICY_CONTRACT, POLICY_SCHEMA


def observation(rows=6):
    features = np.random.default_rng(11).normal(size=(rows, 3))
    features[:, 2] = 0
    features[-1, 2] = 1
    return {"feature_names": ("a", "b", "stop"), "feature_schema": FEATURE_SCHEMA,
            "option_features": features, "action_mask": np.ones(rows, dtype=bool)}


def batch(policy, count=4):
    episodes = []
    for index in range(count):
        episode = []
        for step in range(2):
            obs = observation(3 + index)
            action, record = policy.sample(obs)
            record["reward"] = float(obs["option_features"][action, 0] + step)
            episode.append(record)
        episodes.append(episode)
    return episodes


def test_mask_permutation_and_variable_count():
    obs = observation(9)
    obs["action_mask"][[0, 3, 6]] = False
    policy = BalancedPolicy(obs["feature_names"], seed=1)
    probability = policy.probabilities(obs)
    assert probability.sum() == pytest.approx(1)
    assert np.all(probability[~obs["action_mask"]] == 0)
    assert all(obs["action_mask"][policy.act(obs, deterministic=False)] for _ in range(100))
    permutation = np.array([4, 0, 2, 8, 3, 1, 7, 6, 5])
    permuted = {**obs, "option_features": obs["option_features"][permutation],
                "action_mask": obs["action_mask"][permutation]}
    np.testing.assert_allclose(policy.probabilities(permuted), probability[permutation], atol=1e-15)
    assert permutation[policy.act(permuted)] == policy.act(obs)
    assert policy.probabilities(observation(17)).shape == (17,)


@pytest.mark.parametrize("mutation", ["names", "nonfinite", "shape", "mask", "empty_mask", "semantics",
                                     "missing_stop", "two_stops", "fractional_stop", "masked_stop"])
def test_observation_contract_rejects_mismatches(mutation):
    obs = observation()
    policy = BalancedPolicy(obs["feature_names"])
    if mutation == "names":
        obs["feature_names"] = ("b", "a", "stop")
    elif mutation == "nonfinite":
        obs["option_features"][0, 0] = np.nan
    elif mutation == "shape":
        obs["option_features"] = np.zeros((6, 4))
    elif mutation == "mask":
        obs["action_mask"] = np.ones(6, dtype=int)
    elif mutation == "empty_mask":
        obs["action_mask"][:] = False
    elif mutation == "semantics":
        obs["feature_schema"] = "new-meaning-same-columns"
    elif mutation == "missing_stop":
        obs["option_features"][-1, 2] = 0
    elif mutation == "two_stops":
        obs["option_features"][0, 2] = 1
    elif mutation == "fractional_stop":
        obs["option_features"][0, 2] = 0.2
    else:
        obs["action_mask"][-1] = False
    with pytest.raises(ValueError):
        policy.act(obs)


def test_missing_named_stop_feature_rejected():
    with pytest.raises(ValueError, match="named 'stop'"):
        BalancedPolicy(("a", "b"))


@pytest.mark.parametrize("stop_only", [False, True])
def test_all_parameter_gradients_match_finite_differences(stop_only):
    policy = BalancedPolicy(observation()["feature_names"], seed=7, hidden_size=3)
    policy.parameters["wv"][:] = [0.2, -0.1, 0.3]
    records = []
    for rows in (4, 7):
        obs = observation(rows)
        obs["action_mask"][1] = False
        if stop_only:
            obs["action_mask"][:-1] = False
        _, record = policy.sample(obs)
        records.append(record)
    returns, advantages = np.array([0.75, -0.3]), np.array([-0.65, 0.8])
    _, gradients, _ = policy._loss_and_gradients(records, returns, advantages, 0.4, 0.6)
    epsilon = 1e-6
    for key, parameter in policy.parameters.items():
        numerical = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + epsilon
            positive = policy._loss_and_gradients(records, returns, advantages, 0.4, 0.6)[0]
            parameter[index] = original - epsilon
            negative = policy._loss_and_gradients(records, returns, advantages, 0.4, 0.6)[0]
            parameter[index] = original
            numerical[index] = (positive - negative) / (2 * epsilon)
        np.testing.assert_allclose(gradients[key], numerical, rtol=1e-5, atol=1e-8)


def test_stop_only_has_probability_one_and_no_actor_or_entropy_gradients():
    obs = observation()
    obs["action_mask"][:-1] = False
    policy = BalancedPolicy(obs["feature_names"])
    probability = policy.probabilities(obs)
    np.testing.assert_array_equal(probability, [0, 0, 0, 0, 0, 1])
    action, record = policy.sample(obs)
    assert action == policy.act(obs) == 5
    diagnostics = policy.diagnostics(obs)
    assert diagnostics["stop_probability"] == 1
    for key in ("deployment_probability", "joint_entropy", "gate_entropy", "relative_entropy",
                "deployment_entropy", "legal_deployments"):
        assert diagnostics[key] == 0
    _, gradients, metrics = policy._loss_and_gradients([record], np.array([3.]), np.array([2.]), 0.1, 0)
    assert metrics["actor_loss"] == metrics["entropy"] == 0
    for gradient in gradients.values():
        assert not np.any(gradient)


@pytest.mark.parametrize("copies", [2, 9, 40])
def test_duplicate_deployments_preserve_gate_and_relative_entropy_and_gradients(copies):
    obs = observation(8)
    obs["action_mask"][1] = False
    policy = BalancedPolicy(obs["feature_names"], seed=3, hidden_size=4)
    policy.parameters["wa"] *= 10  # Non-uniform conditional deployment scores.
    duplicated = {**obs,
                  "option_features": np.concatenate([np.repeat(obs["option_features"][:-1], copies, axis=0),
                                                       obs["option_features"][-1:]]),
                  "action_mask": np.concatenate([np.repeat(obs["action_mask"][:-1], copies), [True]])}
    original, repeated = policy.diagnostics(obs), policy.diagnostics(duplicated)
    for key in ("stop_probability", "deployment_probability", "gate_entropy", "relative_entropy"):
        assert original[key] == pytest.approx(repeated[key], abs=1e-14)
    assert repeated["joint_entropy"] == pytest.approx(
        original["joint_entropy"] + original["deployment_probability"] * np.log(copies))
    assert original["relative_entropy"] == pytest.approx(
        original["gate_entropy"] - original["deployment_probability"] *
        (np.log(original["legal_deployments"]) - original["deployment_entropy"]))
    outputs = []
    for current in (obs, duplicated):
        _, record = policy.sample(current)
        record["action"] = len(current["action_mask"]) - 1
        outputs.append(policy._loss_and_gradients([record], np.array([0.]), np.array([0.]), 0.7, 0))
    assert outputs[0][0] == pytest.approx(outputs[1][0])
    for key in policy.parameters:
        np.testing.assert_allclose(outputs[0][1][key], outputs[1][1][key], atol=1e-14)


def test_equal_logits_have_half_gate_and_gate_first_differs_from_joint_argmax():
    obs = {"feature_names": ("deployment", "stop"),
           "option_features": np.array([[1, 0]] * 6 + [[0, 1]], dtype=float),
           "action_mask": np.ones(7, dtype=bool)}
    policy = BalancedPolicy(obs["feature_names"], hidden_size=2)
    policy.parameters["w1"][:] = np.eye(2)
    policy.parameters["wa"][:] = 0
    assert policy.probabilities(obs)[-1] == pytest.approx(0.5)
    assert policy.act(obs) == 6  # Gate ties stop.
    policy.parameters["wa"][:] = [np.log(2) / np.tanh(1), 0]
    probability = policy.probabilities(obs)
    assert probability[-1] == pytest.approx(1 / 3)
    assert np.argmax(probability) == 6
    assert policy.act(obs) == 0  # Deploy mass is 2/3, though every deployment is 1/9.


def test_relative_entropy_can_be_negative_and_does_not_reward_catalogue_size():
    policy = BalancedPolicy(("quality", "stop"), seed=1, hidden_size=2)
    policy.parameters["w1"][:] = np.eye(2)
    policy.parameters["wa"][:] = [30, -20]
    obs = {"feature_names": policy.feature_names,
           "option_features": np.array([[1, 0]] + [[-1, 0]] * 15 + [[0, 1]], dtype=float),
           "action_mask": np.ones(17, dtype=bool)}
    diagnostic = policy.diagnostics(obs)
    assert diagnostic["relative_entropy"] < -2.7
    assert diagnostic["joint_entropy"] >= 0


def test_real_policy_gradient_learns_contextual_stopping_and_useful_deployment():
    policy = BalancedPolicy(("signal", "quality", "stop", "bias"), seed=9, hidden_size=16)
    rng = np.random.default_rng(90)

    def make_obs(signal, count=12):
        features = np.array([[signal, -1, 0, 1]] * count + [[signal, 0, 1, 1]], dtype=float)
        features[0, 1] = 1
        return {"feature_names": policy.feature_names, "option_features": features,
                "action_mask": np.ones(count + 1, dtype=bool)}

    initial_good = policy.probabilities(make_obs(1))[0]
    initial_stop = policy.probabilities(make_obs(-1))[-1]
    for _ in range(180):
        episodes = []
        for index in range(24):
            signal = 1 if index % 2 else -1
            obs = make_obs(signal, int(rng.integers(4, 17)))
            obs["option_features"] = obs["option_features"][rng.permutation(len(obs["action_mask"]))]
            action, record = policy.sample(obs)
            row = obs["option_features"][action]
            record["reward"] = 0.0 if row[2] else (1.0 if signal > 0 and row[1] > 0 else -1.0)
            episodes.append([record])
        policy.update(episodes, learning_rate=0.015, entropy_coef=0.01)
    assert policy.probabilities(make_obs(-1))[-1] > 0.95
    assert policy.probabilities(make_obs(1))[0] > 0.95
    assert policy.probabilities(make_obs(-1))[-1] > initial_stop + 0.4
    assert policy.probabilities(make_obs(1))[0] > initial_good + 0.8
    assert policy.act(make_obs(-1)) == 12
    assert policy.act(make_obs(1)) == 0


def test_transfer_copies_weights_without_aliasing_and_resets_adam_and_rng():
    base = AdaptivePolicy(observation()["feature_names"], seed=2, hidden_size=4)
    base.update(batch(base))
    policy = BalancedPolicy.transfer_from_adaptive(base, seed=81)
    assert policy.weights_fingerprint() != base.weights_fingerprint()
    for key in policy.parameters:
        np.testing.assert_array_equal(policy.parameters[key], base.parameters[key])
        assert not np.shares_memory(policy.parameters[key], base.parameters[key])
        assert not np.any(policy.adam_m[key])
        assert not np.any(policy.adam_v[key])
    assert policy.update_count == 0
    assert policy.rng.bit_generator.state == np.random.default_rng(81).bit_generator.state
    assert policy.training_state["transfer"]["source_weights_sha256"] == base.weights_fingerprint()
    with pytest.raises(ValueError):
        BalancedPolicy.transfer_from_adaptive(policy)


def test_checkpoint_roundtrip_resumes_next_update_exactly(tmp_path):
    policy = BalancedPolicy(observation()["feature_names"], seed=5, hidden_size=8)
    policy.update(batch(policy))
    policy.save(tmp_path, {"completed_episodes": 4, "seed_provenance": {"train": [1, 4]}})
    resumed = BalancedPolicy.load(tmp_path, policy.feature_names)
    assert resumed.weights_fingerprint() == policy.weights_fingerprint()
    assert resumed.metadata["schema"] == CHECKPOINT_SCHEMA
    assert resumed.metadata["policy_schema"] == POLICY_SCHEMA
    assert resumed.metadata["policy_contract"] == POLICY_CONTRACT
    assert resumed.training_state == policy.training_state
    assert resumed.metadata["seed_provenance"] == {"train": [1, 4]}
    original_batch, resumed_batch = batch(policy), batch(resumed)
    assert [[r["action"] for r in ep] for ep in original_batch] == [[r["action"] for r in ep] for ep in resumed_batch]
    assert policy.update(original_batch) == resumed.update(resumed_batch)
    assert resumed.weights_fingerprint() == policy.weights_fingerprint()
    for key in policy.parameters:
        np.testing.assert_array_equal(policy.adam_m[key], resumed.adam_m[key])
        np.testing.assert_array_equal(policy.adam_v[key], resumed.adam_v[key])


def test_loaders_reject_cross_policy_schema(tmp_path):
    balanced = BalancedPolicy(observation()["feature_names"])
    balanced.save(tmp_path / "balanced")
    with pytest.raises(ValueError, match="schema"):
        AdaptivePolicy.load(tmp_path / "balanced")
    base = AdaptivePolicy(observation()["feature_names"])
    base.save(tmp_path / "adaptive")
    with pytest.raises(ValueError, match="schema"):
        BalancedPolicy.load(tmp_path / "adaptive")


@pytest.mark.parametrize("corruption", ["bytes", "nonfinite", "features", "schema", "weights", "contract",
                                       "count", "rng", "negative_moment", "shape", "dtype", "member",
                                       "seed_provenance", "unknown_field", "not_object", "duplicate_key",
                                       "overflow_float"])
def test_checkpoint_rejects_corruption(tmp_path, corruption):
    policy = BalancedPolicy(observation()["feature_names"])
    policy.save(tmp_path)
    path = tmp_path / "checkpoint.json"
    metadata = json.loads(path.read_text())
    if corruption == "bytes":
        with (tmp_path / "arrays.npz").open("ab") as handle:
            handle.write(b"corrupted")
    elif corruption in {"nonfinite", "negative_moment", "shape", "dtype", "member"}:
        with np.load(tmp_path / "arrays.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if corruption == "nonfinite":
            arrays["param_w1"][0, 0] = np.nan
        elif corruption == "negative_moment":
            arrays["adam_v_w1"][0, 0] = -1
        elif corruption == "shape":
            arrays["param_w1"] = arrays["param_w1"].reshape(-1)
        elif corruption == "dtype":
            arrays["param_w1"] = arrays["param_w1"].astype(np.float32)
        else:
            arrays["unexpected"] = np.zeros(1)
        np.savez_compressed(tmp_path / "arrays.npz", **arrays)
        metadata["arrays_sha256"] = hashlib.sha256((tmp_path / "arrays.npz").read_bytes()).hexdigest()
    elif corruption == "features":
        metadata["feature_names"] = list(reversed(metadata["feature_names"]))
    elif corruption == "schema":
        metadata["feature_schema"] = "different-semantics"
    elif corruption == "weights":
        metadata["weights_sha256"] = "0" * 64
    elif corruption == "contract":
        metadata["policy_contract"]["deterministic_rule"] = "joint-argmax"
    elif corruption == "count":
        metadata["update_count"] = True
    elif corruption == "rng":
        metadata["rng_state"] = {"bit_generator": "MT19937"}
    elif corruption == "seed_provenance":
        metadata["seed_provenance"] = {"train": "false claim"}
    elif corruption == "unknown_field":
        metadata["unknown"] = 1
    elif corruption == "not_object":
        metadata = []
    if corruption == "duplicate_key":
        path.write_text('{"schema":"duplicate",' + json.dumps(metadata)[1:], encoding="utf-8")
    elif corruption == "overflow_float":
        metadata["training_state"] = {"numeric": "OVERFLOW"}
        path.write_text(json.dumps(metadata).replace('"OVERFLOW"', "1e999"), encoding="utf-8")
    else:
        path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        BalancedPolicy.load(tmp_path)


def test_stochastic_and_sample_use_joint_probabilities_not_deterministic_gate():
    obs = {"feature_names": ("deployment", "stop"),
           "option_features": np.array([[1, 0]] * 6 + [[0, 1]], dtype=float),
           "action_mask": np.ones(7, dtype=bool)}
    policy = BalancedPolicy(obs["feature_names"], seed=32, hidden_size=2)
    policy.parameters["w1"][:] = np.eye(2)
    policy.parameters["wa"][:] = [np.log(2) / np.tanh(1), 0]
    assert policy.act(obs) == 0
    for sampling in (lambda: policy.act(obs, deterministic=False), lambda: policy.sample(obs)[0]):
        frequencies = np.bincount([sampling() for _ in range(3000)], minlength=7) / 3000
        np.testing.assert_allclose(frequencies, policy.probabilities(obs), atol=0.04)


def test_invalid_training_data_does_not_mutate_policy():
    policy = BalancedPolicy(observation()["feature_names"])
    records = batch(policy)
    before = policy.weights_fingerprint()
    records[0][0]["features"][-1, 2] = 0
    with pytest.raises(ValueError):
        policy.update(records)
    assert policy.weights_fingerprint() == before
    assert policy.update_count == 0


def test_reward_to_go_gradient_clipping_and_deterministic_rng_preservation():
    policy = BalancedPolicy(observation()["feature_names"], seed=1)
    _, first = policy.sample(observation())
    _, second = policy.sample(observation())
    first["reward"], second["reward"] = 1.0, 3.0
    metrics = policy.update([[first, second]], gamma=0.5, max_grad_norm=0.01)
    assert metrics["mean_return_to_go"] == pytest.approx((2.5 + 3) / 2)
    assert metrics["gradient_norm"] * metrics["gradient_scale"] <= 0.01 + 1e-12
    assert metrics["entropy"] == metrics["relative_entropy"]
    state = json.dumps(policy.rng.bit_generator.state, sort_keys=True)
    for _ in range(5):
        policy.act(observation(), deterministic=True)
        policy.diagnostics(observation())
    assert json.dumps(policy.rng.bit_generator.state, sort_keys=True) == state


def test_actual_environment_observation_is_accepted():
    from triad_rl.robust_scenarios import RobustPlacementEnv
    env = RobustPlacementEnv(seed=320, profile="capability")
    obs = env.observe()
    policy = BalancedPolicy(obs["feature_names"])
    while not env.done:
        action = policy.act(obs)
        assert obs["action_mask"][action]
        obs, _, _, _ = env.step(action)


@pytest.mark.parametrize("corruption", ["nonfinite", "negative_moment", "shape", "count", "metadata"])
def test_invalid_save_does_not_replace_existing_checkpoint(tmp_path, corruption):
    policy = BalancedPolicy(observation()["feature_names"])
    policy.save(tmp_path)
    previous = [(tmp_path / name).read_bytes() for name in ("checkpoint.json", "arrays.npz")]
    if corruption == "nonfinite":
        policy.parameters["w1"][0, 0] = np.nan
    elif corruption == "negative_moment":
        policy.adam_v["wa"][0] = -1
    elif corruption == "shape":
        policy.parameters["wa"] = np.zeros(1)
    elif corruption == "count":
        policy.update_count = -1
    else:
        policy.training_state = {"nonfinite": float("inf")}
    with pytest.raises(ValueError):
        policy.save(tmp_path)
    assert previous == [(tmp_path / name).read_bytes() for name in ("checkpoint.json", "arrays.npz")]
