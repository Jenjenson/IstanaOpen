"""Numerical/synthetic-array policy tests; no simulator or scenario access."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from triad_rl.temporal_inputs import FEATURE_NAMES, FEATURE_SCHEMA, TemporalConfig, TemporalPublicGreedy
from triad_rl.temporal_policy import ACTOR, CRITIC, TemporalPolicy

CONFIG = TemporalConfig()


def observation(priors=(.03, .08, -.04), *, stop_first=False):
    width = len(FEATURE_NAMES)
    x = .2 * np.sin(np.arange((len(priors) + 1) * width).reshape(-1, width) * .1)
    x[:, FEATURE_NAMES.index("stop")] = 0.
    x[-1, FEATURE_NAMES.index("stop")] = 1.
    x[:, FEATURE_NAMES.index("temporal_marginal_return")] = [*priors, 0.]
    for name, value in {"budget_total": 1.5, "budget_remaining": 1., "existing_coverage": .4,
                        "existing_early_coverage": .2, "temporal_existing_detection": .7,
                        "temporal_existing_confirmation": .5, "temporal_existing_timely": .3,
                        "temporal_existing_early": .2}.items():
        x[:, FEATURE_NAMES.index(name)] = value
    if stop_first:
        x = np.roll(x, 1, axis=0)
    return {"feature_schema": FEATURE_SCHEMA, "feature_names": FEATURE_NAMES,
            "temporal_config": asdict(CONFIG), "option_features": x, "action_mask": np.ones(len(x), dtype=bool)}


def policy():
    return TemporalPolicy(CONFIG, seed=499, hidden_size=2)


def nonzero_policy():
    result = policy()
    result.parameters["wa"][:] = [.13, -.21]
    result.parameters["b1"][:] = [.03, -.02]
    result.parameters["wv"][:] = np.linspace(-.02, .03, len(result.parameters["wv"]))
    result.parameters["bv"][0] = .1
    return result


def records(agent, count=2):
    result = []
    for index in range(count):
        obs = observation()
        obs["action_mask"][index % 2] = False
        _, record = agent.sample(obs)
        record["reward"] = float(index + 1)
        result.append(record)
    return result


def all_state(agent):
    return ({key: value.copy() for key, value in agent._arrays().items()},
            (agent.update_count, agent.actor_update_count, agent.critic_update_count), agent.rng_fingerprint())


def assert_state_equal(left, right):
    assert left[1:] == right[1:]
    for key in left[0]:
        np.testing.assert_array_equal(left[0][key], right[0][key])


@pytest.mark.parametrize("priors", [(.03, .08, -.04), (-.1, -.2), (0., 0.), (.1, .1), ()])
@pytest.mark.parametrize("stop_first", [False, True])
def test_exact_temporal_greedy_initialization(priors, stop_first):
    agent, obs = policy(), observation(priors, stop_first=stop_first)
    expected = obs["option_features"][:, FEATURE_NAMES.index("temporal_marginal_return")]
    np.testing.assert_array_equal(agent.scores(obs), expected)
    assert agent.act(obs) == TemporalPublicGreedy().act(obs)
    np.testing.assert_array_equal(agent.parameters["wa"], [0., 0.])


def test_learned_stop_wins_positive_score_ties_and_residual_is_bounded():
    agent, obs = policy(), observation((0., 0.), stop_first=False)
    agent.parameters["w1"][:] = 0.
    agent.parameters["b1"][:] = 1.
    agent.parameters["wa"][:] = 1000.
    np.testing.assert_array_equal(agent.scores(obs), np.full(3, .25))
    assert agent.act(obs) == 2
    obs = observation()
    for sign in (-1., 1.):
        agent.parameters["wa"][:] = sign * 1000.
        residual = agent.scores(obs) - obs["option_features"][:, FEATURE_NAMES.index("temporal_marginal_return")]
        assert np.max(np.abs(residual)) <= .25 + 1e-16


def test_deterministic_raw_score_selection_is_not_joint_probability_mode():
    agent, obs = policy(), observation((.01,) * 10)
    probability = agent.probabilities(obs)
    assert agent.act(obs) == 0 and np.argmax(probability) == 10
    expected_stop = 1. / (1. + np.exp(.01 / .05))
    assert probability[-1] == pytest.approx(expected_stop, abs=1e-15)
    np.testing.assert_allclose(probability[:-1], np.full(10, (1. - expected_stop) / 10), atol=1e-16)


def test_masked_rows_zero_probability_and_stop_only_distribution():
    agent, obs = nonzero_policy(), observation()
    obs["action_mask"][:] = [False, False, False, True]
    np.testing.assert_array_equal(agent.probabilities(obs), [0., 0., 0., 1.])
    assert agent.act(obs) == agent.act(obs, deterministic=False) == 3
    _, record = agent.sample(obs)
    loss, gradients, metrics = agent._loss_and_gradients([record], [1.], [2.], .3, 0.)
    assert loss == 0. and metrics["entropy"] == metrics["gate_entropy"] == 0.
    for key in ACTOR:
        np.testing.assert_array_equal(gradients[key], np.zeros_like(gradients[key]))


def test_permutation_equivariance_and_deployment_duplication_invariance():
    agent, obs = nonzero_policy(), observation()
    obs["action_mask"][1] = False
    permutation = np.array([3, 2, 0, 1])
    permuted = {**obs, "option_features": obs["option_features"][permutation], "action_mask": obs["action_mask"][permutation]}
    np.testing.assert_allclose(agent.scores(permuted), agent.scores(obs)[permutation], atol=2e-16)
    np.testing.assert_allclose(agent.probabilities(permuted), agent.probabilities(obs)[permutation], atol=2e-16)
    assert permutation[agent.act(permuted)] == agent.act(obs)
    assert agent.value(permuted) == pytest.approx(agent.value(obs), abs=1e-14)
    repeated = np.array([0, 0, 0, 2, 2, 2, 1, 3])
    duplicated = {**obs, "option_features": obs["option_features"][repeated], "action_mask": obs["action_mask"][repeated]}
    before, after = agent.probabilities(obs), agent.probabilities(duplicated)
    assert before[-1] == pytest.approx(after[-1], abs=1e-15)
    assert before[0] == pytest.approx(after[:3].sum(), abs=1e-15)
    assert before[2] == pytest.approx(after[3:6].sum(), abs=1e-15)
    # Critic averages deployments only; duplicating all legal deployments does
    # not silently change STOP's weight in its separate context either.
    assert agent.value(obs) == pytest.approx(agent.value(duplicated), abs=1e-14)
    diagnostics = []
    for item in (obs, duplicated):
        x, mask, stop = agent._input(item)
        diagnostics.append(agent._distribution(agent._forward(x)[2], mask, stop, .05)[3])
    for key in ("entropy", "gate_entropy", "stop_probability", "deployment_probability"):
        assert diagnostics[0][key] == pytest.approx(diagnostics[1][key], abs=2e-15)


def test_independent_public_stop_rtg_prior_and_deployment_only_context():
    agent, obs = policy(), observation()
    # Original terminal expectation less current shaping potential. Budget
    # features are divided by four; already-paid shaping/redundancy is not RTG.
    expected = 16 * .3 + 2 * .7 + .5 + 3 * .2 + 2 * .4 - .2 - 10.5 - .55 * 4 * (1.5 - 1.)
    assert agent.value(obs) == pytest.approx(expected)
    x, mask, stop = agent._input(obs)
    context, _ = agent._value(x, mask, stop)
    np.testing.assert_array_equal(context, np.r_[x[:-1].mean(axis=0), x[-1]])
    obs["action_mask"][:-1] = False
    context, _ = agent._value(*agent._input(obs))
    np.testing.assert_array_equal(context[:len(FEATURE_NAMES)], np.zeros(len(FEATURE_NAMES)))
    np.testing.assert_array_equal(context[len(FEATURE_NAMES):], x[-1])


@pytest.mark.parametrize("entropy_coef,value_coef", [(0., 0.), (.17, 0.), (.17, .6)])
def test_every_actor_and_critic_parameter_matches_finite_differences(entropy_coef, value_coef):
    agent = nonzero_policy()
    transitions = records(agent)
    targets, advantages = np.array([1.7, -.8]), np.array([.6, -.4])
    _, analytic, _ = agent._loss_and_gradients(transitions, targets, advantages, entropy_coef, value_coef)
    epsilon = 1e-6
    for key, parameter in agent.parameters.items():
        numeric = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + epsilon
            plus = agent._loss_and_gradients(transitions, targets, advantages, entropy_coef, value_coef)[0]
            parameter[index] = original - epsilon
            minus = agent._loss_and_gradients(transitions, targets, advantages, entropy_coef, value_coef)[0]
            parameter[index] = original
            numeric[index] = (plus - minus) / (2 * epsilon)
        np.testing.assert_allclose(analytic[key], numeric, rtol=2e-5, atol=2e-8, err_msg=key)


def test_critic_only_loss_does_not_update_actor_parameters_or_moments():
    agent = nonzero_policy()
    transition = records(agent, 1)[0]
    transition["value"] = transition["reward"] = 30.
    before = all_state(agent)
    metrics = agent.update([[transition]], max_grad_norm=.01)
    assert metrics["actor_gradient_norm"] == 0. and metrics["critic_gradient_norm"] > .01
    for prefix in ("param", "adam_m", "adam_v"):
        for key in ACTOR:
            np.testing.assert_array_equal(agent._arrays()[f"{prefix}_{key}"], before[0][f"{prefix}_{key}"])
    assert not np.array_equal(agent.parameters["wv"], before[0]["param_wv"])


def test_detached_sample_values_and_separate_clipping_isolate_actor_from_critic():
    agent = nonzero_policy()
    transitions = records(agent)
    no_critic, huge_critic = deepcopy(agent), deepcopy(agent)
    huge_critic.parameters["wv"] += 10.
    for _ in range(3):
        first = no_critic.update([transitions], value_coef=0., entropy_coef=.02, max_grad_norm=.001)
        second = huge_critic.update([transitions], value_coef=100., entropy_coef=.02, max_grad_norm=.001)
        assert first["actor_gradient_norm"] == second["actor_gradient_norm"]
        for key in ACTOR:
            for group in ("parameters", "adam_m", "adam_v"):
                np.testing.assert_array_equal(getattr(no_critic, group)[key], getattr(huge_critic, group)[key])
    assert no_critic.critic_update_count == 0 and huge_critic.critic_update_count == 3
    assert no_critic.rng_fingerprint() == huge_critic.rng_fingerprint()


def test_update_matches_independent_per_block_clipped_adam_arithmetic():
    agent = nonzero_policy()
    transitions = records(agent)
    returns = np.array([3., 2.])
    advantages = returns - np.array([row["value"] for row in transitions])
    _, gradients, _ = agent._loss_and_gradients(transitions, returns, advantages, .03, .7)
    before = deepcopy(agent.parameters)
    metrics = agent.update([transitions], learning_rate=.003, critic_learning_rate=.007,
                           entropy_coef=.03, value_coef=.7, max_grad_norm=.1)
    for keys, rate in ((ACTOR, .003), (CRITIC, .007)):
        norm = np.sqrt(sum(np.sum(gradients[key] ** 2) for key in keys))
        scale = min(1., .1 / max(norm, 1e-12))
        for key in keys:
            gradient = gradients[key] * scale
            mean, variance = .1 * gradient, .001 * gradient * gradient
            expected = before[key] - rate * (mean / (1 - .9)) / (np.sqrt(variance / (1 - .999)) + 1e-8)
            np.testing.assert_array_equal(agent.parameters[key], expected)
            np.testing.assert_array_equal(agent.adam_m[key], mean)
            np.testing.assert_array_equal(agent.adam_v[key], variance)
    assert metrics["advantages_normalized"] is False
    assert metrics["mean_return_to_go"] == 2.5


def test_optional_advantage_normalization_is_explicit_and_uses_stable_rule():
    agent = policy()
    transitions = records(agent)
    normalized, manual = deepcopy(agent), deepcopy(agent)
    targets = np.array([3., 2.])
    raw = targets - np.array([row["value"] for row in transitions])
    applied = (raw - raw.mean()) / (raw.std() + 1e-8)
    # Modified detached values construct exactly the same normalized actor
    # coefficients without altering critic targets.
    normalized_records = deepcopy(transitions)
    for row, target, advantage in zip(normalized_records, targets, applied):
        row["value"] = target - advantage
    result = normalized.update([transitions], normalize_advantages=True)
    manual.update([normalized_records])
    assert result["advantages_normalized"] is True
    for key in ACTOR:
        np.testing.assert_allclose(normalized.parameters[key], manual.parameters[key], atol=1e-16)
    equal = records(agent)
    equal[0]["value"], equal[1]["value"] = 1., 0.
    assert agent.update([equal], normalize_advantages=True)["advantages_normalized"] is False


def test_sample_is_detached_and_deterministic_calls_do_not_consume_rng():
    agent, obs = policy(), observation()
    before = all_state(agent)
    for _ in range(3):
        agent.act(obs)
        agent.scores(obs)
        agent.probabilities(obs)
        agent.value(obs)
        agent.weights_fingerprint()
    assert_state_equal(before, all_state(agent))
    action, record = agent.sample(obs)
    assert action == record["action"] and record["value"] == agent.value(obs)
    assert agent.rng_fingerprint() != before[2]
    record["features"][:] = 0.
    record["mask"][:] = False
    record["temporal_config"]["lead_time_s"] = 0.
    assert obs["action_mask"].all() and np.count_nonzero(obs["option_features"]) > 0
    assert obs["temporal_config"] == asdict(CONFIG)


def test_checkpoint_exact_resume_next_action_and_update(tmp_path):
    agent = nonzero_policy()
    agent.update([records(agent)], value_coef=0.)
    agent.update([records(agent)])
    checkpoint = tmp_path / "checkpoint"
    agent.save(checkpoint, {"seed_provenance": {"training": {"start": 499, "count": 0}}, "note": "synthetic arrays only"})
    restored = TemporalPolicy.load(checkpoint, config=CONFIG)
    assert_state_equal(all_state(agent), all_state(restored))
    assert agent.weights_fingerprint() == restored.weights_fingerprint()
    assert agent.training_state == restored.training_state
    left, right = records(agent), records(restored)
    assert [row["action"] for row in left] == [row["action"] for row in right]
    assert agent.update([left], entropy_coef=.03) == restored.update([right], entropy_coef=.03)
    assert_state_equal(all_state(agent), all_state(restored))
    before = {path.name: path.read_bytes() for path in checkpoint.iterdir()}
    with pytest.raises(FileExistsError):
        agent.save(checkpoint)
    assert before == {path.name: path.read_bytes() for path in checkpoint.iterdir()}


@pytest.mark.parametrize("field,value", [("schema", "triad.adaptive_checkpoint.v1"),
    ("policy_schema", "other"), ("feature_schema", "triad.adaptive_placement_features.v1"),
    ("update_count", True), ("actor_update_count", 1), ("critic_update_count", 1),
    ("weights_sha256", "0" * 64), ("hidden_size", 2.), ("temperature", False),
    ("seed_provenance", {"unbound": 1}), ("feature_names", list(reversed(FEATURE_NAMES)))])
def test_checkpoint_metadata_tamper_rejected(tmp_path, field, value):
    checkpoint = tmp_path / "checkpoint"
    policy().save(checkpoint)
    metadata = json.loads((checkpoint / "checkpoint.json").read_bytes())
    metadata[field] = value
    (checkpoint / "checkpoint.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        TemporalPolicy.load(checkpoint, config=CONFIG)


def test_wrong_config_and_old_loader_reject_temporal_checkpoint(tmp_path):
    from triad_rl.ranking_policy import RankPolicy
    checkpoint = tmp_path / "checkpoint"
    policy().save(checkpoint)
    with pytest.raises(ValueError):
        TemporalPolicy.load(checkpoint, config=TemporalConfig(lead_time_s=5.))
    with pytest.raises(ValueError):
        RankPolicy.load(checkpoint)


@pytest.mark.parametrize("mutation", ["nonfinite", "float32", "negative_variance", "shape", "extra_member", "hash"])
def test_checkpoint_npz_tamper_rejected_even_with_rehashed_archive(tmp_path, mutation):
    checkpoint = tmp_path / "checkpoint"
    policy().save(checkpoint)
    path = checkpoint / "arrays.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if mutation == "nonfinite":
        arrays["param_w1"][0, 0] = np.nan
    elif mutation == "float32":
        arrays["param_w1"] = arrays["param_w1"].astype(np.float32)
    elif mutation == "negative_variance":
        arrays["adam_v_wa"][0] = -1.
    elif mutation == "shape":
        arrays["param_w1"] = arrays["param_w1"][:-1]
    elif mutation == "extra_member":
        arrays["extra"] = np.zeros(1)
    else:
        arrays["param_wa"][0] += 1.
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    path.write_bytes(buffer.getvalue())
    metadata = json.loads((checkpoint / "checkpoint.json").read_bytes())
    metadata["arrays_sha256"] = hashlib.sha256(buffer.getvalue()).hexdigest()
    (checkpoint / "checkpoint.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        TemporalPolicy.load(checkpoint, config=CONFIG)


@pytest.mark.parametrize("content", ['{"schema":1,"schema":2}', '{"overflow":1e999}', '{"nan":NaN}'])
def test_strict_json_rejects_duplicates_and_nonfinite_literals(tmp_path, content):
    checkpoint = tmp_path / "checkpoint"
    policy().save(checkpoint)
    (checkpoint / "checkpoint.json").write_text(content)
    with pytest.raises(ValueError):
        TemporalPolicy.load(checkpoint, config=CONFIG)


@pytest.mark.parametrize("mutation", ["schema", "names", "config", "nan", "infinity", "wrong_shape", "numeric_mask",
    "empty_mask", "masked_stop", "multiple_stop", "fractional_stop", "missing_stop", "nonzero_stop_prior"])
def test_invalid_observation_rejected_without_rng_mutation(mutation):
    agent, obs = policy(), observation()
    x, mask = obs["option_features"], obs["action_mask"]
    if mutation == "schema": obs["feature_schema"] = "old"
    elif mutation == "names": obs["feature_names"] = tuple(reversed(FEATURE_NAMES))
    elif mutation == "config": obs["temporal_config"]["lead_time_s"] = 5.
    elif mutation == "nan": x[0, 0] = np.nan
    elif mutation == "infinity": x[0, 0] = np.inf
    elif mutation == "wrong_shape": obs["option_features"] = x[:, :-1]
    elif mutation == "numeric_mask": obs["action_mask"] = mask.astype(int)
    elif mutation == "empty_mask": mask[:] = False
    elif mutation == "masked_stop": mask[-1] = False
    elif mutation == "multiple_stop": x[0, FEATURE_NAMES.index("stop")] = 1.
    elif mutation == "fractional_stop": x[0, FEATURE_NAMES.index("stop")] = .5
    elif mutation == "missing_stop": x[-1, FEATURE_NAMES.index("stop")] = 0.
    else: x[-1, FEATURE_NAMES.index("temporal_marginal_return")] = .1
    before = agent.rng_fingerprint()
    for method in (agent.act, agent.sample, agent.value, agent.probabilities):
        with pytest.raises(ValueError): method(obs)
    assert agent.rng_fingerprint() == before


@pytest.mark.parametrize("kwargs", [{"gamma": value} for value in (True, -1., 1.1, np.nan)]
    + [{"normalize_advantages": 1}, {"learning_rate": 0.}, {"critic_learning_rate": True},
       {"max_grad_norm": 0.}, {"entropy_coef": -1.}, {"value_coef": True}])
def test_invalid_update_controls_are_transactional(kwargs):
    agent = policy()
    transitions = records(agent)
    before = all_state(agent)
    with pytest.raises(ValueError): agent.update([transitions], **kwargs)
    assert_state_equal(before, all_state(agent))


@pytest.mark.parametrize("field,value", [("action", True), ("action", 0), ("action", 100), ("reward", np.inf),
                                         ("reward", True), ("value", np.nan), ("value", False), ("temperature", 0.)])
def test_invalid_records_are_transactional(field, value):
    agent = policy()
    transition = records(agent, 1)[0]  # Action 0 is masked in this fixture.
    transition[field] = value
    before = all_state(agent)
    with pytest.raises(ValueError): agent.update([[transition]])
    assert_state_equal(before, all_state(agent))


def test_nonfinite_parameters_cannot_infer_update_or_save(tmp_path):
    agent, obs = policy(), observation()
    transitions = records(agent)
    agent.parameters["wv"][0] = np.nan
    for method in (lambda: agent.probabilities(obs), lambda: agent.update([transitions]),
                   lambda: agent.save(tmp_path / "invalid")):
        with pytest.raises(ValueError): method()
    assert not (tmp_path / "invalid").exists()


def test_oversized_npy_shape_rejected_before_numpy_array_allocation(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    policy().save(checkpoint)
    with zipfile.ZipFile(checkpoint / "arrays.npz") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {"descr": "<f8", "fortran_order": False, "shape": (2**40,)})
    members["param_w1.npy"] = header.getvalue()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    (checkpoint / "arrays.npz").write_bytes(buffer.getvalue())
    metadata = json.loads((checkpoint / "checkpoint.json").read_bytes())
    metadata["arrays_sha256"] = hashlib.sha256(buffer.getvalue()).hexdigest()
    (checkpoint / "checkpoint.json").write_text(json.dumps(metadata))

    def forbidden(*args, **kwargs):
        raise AssertionError("Header must be checked before allocating/loading arrays")

    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(ValueError, match="header"):
        TemporalPolicy.load(checkpoint, config=CONFIG)


def test_advantage_moment_overflow_cannot_be_hidden_by_normalization():
    agent = policy()
    transitions = records(agent)
    for row, value in zip(transitions, (-1e308, 1e308)):
        row["value"], row["reward"] = value, 0.
    before = all_state(agent)
    with pytest.raises(ValueError, match="advantage moments"):
        agent.update([transitions], normalize_advantages=True, value_coef=0.)
    assert_state_equal(before, all_state(agent))


def test_array_and_transition_bounds_include_full_public_catalogue():
    from triad_rl.temporal_policy import MAX_ROWS, MAX_TRANSITIONS
    agent, obs = policy(), observation((0.,) * (MAX_ROWS - 1))
    probability = agent.probabilities(obs)
    assert len(probability) == MAX_ROWS and probability[-1] == pytest.approx(.5)
    invalid = {**obs, "option_features": np.concatenate((obs["option_features"][:-1], obs["option_features"])),
               "action_mask": np.ones(2 * MAX_ROWS - 1, dtype=bool)}
    with pytest.raises(ValueError, match="bounded"):
        agent.probabilities(invalid)
    transition = records(agent, 1)[0]
    before = all_state(agent)
    with pytest.raises(ValueError, match="Too many"):
        agent.update([[transition] * (MAX_TRANSITIONS + 1)])
    assert_state_equal(before, all_state(agent))


@pytest.mark.parametrize("kwargs", [{"seed": True}, {"seed": -1}, {"hidden_size": 2.}, {"hidden_size": 0},
                                     {"hidden_size": 513}, {"temperature": 0.}, {"temperature": True}])
def test_invalid_constructor_controls(kwargs):
    with pytest.raises(ValueError):
        TemporalPolicy(CONFIG, **kwargs)


def test_explicit_config_and_invalid_save_metadata_fail_before_write(tmp_path):
    with pytest.raises(ValueError):
        TemporalPolicy(None)
    with pytest.raises(ValueError):
        policy().save(tmp_path / "invalid", {"value": np.nan})
    assert not (tmp_path / "invalid").exists()
