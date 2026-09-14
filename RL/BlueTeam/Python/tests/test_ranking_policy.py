"""Greedy identity, exact pairwise gradients, isolated updates and safe persistence."""
from copy import deepcopy
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from triad_rl.adaptive_env import generate_scenario
from triad_rl.adaptive_evaluation import GreedyPublicCoverage
from triad_rl.adaptive_inputs import DEFAULT_CATALOGUE, FEATURE_NAMES, build_observation
from triad_rl.adaptive_policy import AdaptivePolicy, FEATURE_SCHEMA
from triad_rl.balanced_policy import BalancedPolicy
from triad_rl.ranking_policy import (CHECKPOINT_SCHEMA, PARAMETERS, POLICY_CONTRACT,
    POLICY_SCHEMA, RankPolicy, RankingPolicy)


NAMES = ("signal", "stop", "marginal_coverage", "marginal_early_coverage", "cost", "overlap")


def observation(rows=5):
    x = np.random.default_rng(499).normal(size=(rows, len(NAMES))) * .2
    x[:, 1] = 0
    x[-1, 1] = 1
    return {"feature_names": NAMES, "feature_schema": FEATURE_SCHEMA,
            "option_features": x, "action_mask": np.ones(rows, dtype=bool)}


def records():
    x = observation()["option_features"]
    y = observation(4)["option_features"]
    return [{"features": x.copy(), "comparisons": [{"preferred": 1, "rejected": 0, "weight": 2.},
                                                   {"preferred": 4, "rejected": 2, "weight": .25}]},
            {"features": y.copy(), "comparisons": [{"preferred": 2, "rejected": 3, "weight": .5}]}]


def snapshot(policy):
    return {"arrays": {f"{group}_{key}": value.tobytes() for group in ("parameters", "adam_m", "adam_v")
                       for key, value in getattr(policy, group).items()},
            "count": policy.update_count, "rng": deepcopy(policy.rng.bit_generator.state)}


def test_default_shape_alias_and_exact_zero_residual():
    policy = RankPolicy(seed=499)
    assert RankingPolicy is RankPolicy
    assert policy.feature_names == FEATURE_NAMES
    assert policy.parameters["w1"].shape == (len(FEATURE_NAMES), 64)
    assert set(policy.parameters) == set(PARAMETERS) == {"w1", "b1", "wa"}
    assert not np.any(policy.parameters["wa"])
    assert all(not np.any(v) for group in (policy.adam_m, policy.adam_v) for v in group.values())
    assert snapshot(policy) == snapshot(RankPolicy(seed=499))


@pytest.mark.parametrize("variant", ["default", "changed", "extended", "permuted", "depleted"])
def test_exact_greedy_initialization_with_real_public_custom_catalogues(variant):
    state = generate_scenario(499)["public"]  # Only the designated test scenario.
    catalogue = deepcopy(DEFAULT_CATALOGUE)
    if variant == "changed":
        catalogue[0]["ranges"]["rf"] = 200.
        catalogue[0]["cost"] = .9
        catalogue[1]["strengths"]["radar"] = .4
    if variant == "extended":
        catalogue.append({**deepcopy(catalogue[0]), "id": "custom_rf", "cost": .7})
    if variant == "permuted":
        catalogue.reverse()
        state["sites"].reverse()
    state["available_sensor_ids"] = [row["id"] for row in catalogue]
    if variant == "depleted":
        state["budget_remaining"] = 0.
    obs = build_observation(state, catalogue)
    policy = RankPolicy(seed=499)
    greedy = GreedyPublicCoverage()
    expected = sum(w * np.asarray(obs["option_features"], float)[:, FEATURE_NAMES.index(k)]
                   for k, w in (("marginal_coverage", 3.), ("marginal_early_coverage", 1.), ("cost", -.25), ("overlap", -.15)))
    expected[-1] = 0.
    np.testing.assert_array_equal(policy.scores(obs), expected)
    assert policy.act(obs) == greedy.act(obs)


@pytest.mark.parametrize("masked_stop", [False, True])
def test_exact_ties_stable_first_legal_row_and_masked_stop(masked_stop):
    obs = observation(5)
    obs["option_features"][:] = 0
    obs["option_features"][-1, 1] = 1
    obs["action_mask"][0] = False
    obs["action_mask"][-1] = not masked_stop
    policy = RankPolicy(NAMES, seed=499)
    assert policy.act(obs) == GreedyPublicCoverage().act(obs) == 1
    obs["action_mask"][:-1] = False
    obs["action_mask"][-1] = True
    assert policy.act(obs) == 4
    np.testing.assert_array_equal(policy.probabilities(obs), [0, 0, 0, 0, 1])


def test_stop_prior_zero_despite_nonzero_stop_features_and_masked_high_score():
    obs = observation()
    obs["option_features"][-1, 2:] = 1e8
    obs["option_features"][0, 2] = 1e7
    obs["action_mask"][0] = False
    policy = RankPolicy(NAMES, seed=499)
    assert policy.scores(obs)[-1] == 0
    assert policy.act(obs) == GreedyPublicCoverage().act(obs)
    assert policy.probabilities(obs)[0] == 0


def test_permutation_variable_count_and_private_field_isolation():
    policy = RankPolicy(NAMES, seed=499)
    policy.update_comparisons(records())
    obs = observation()
    before = snapshot(policy)
    perm = np.array([4, 2, 0, 3, 1])
    permuted = {**obs, "option_features": obs["option_features"][perm], "action_mask": obs["action_mask"][perm]}
    np.testing.assert_allclose(policy.scores(permuted), policy.scores(obs)[perm], atol=1e-15)
    np.testing.assert_allclose(policy.probabilities(permuted), policy.probabilities(obs)[perm], atol=1e-15)
    assert perm[policy.act(permuted)] == policy.act(obs)
    private = {**obs, "state": object(), "truth": object(), "catalogue": object(), "targets": object()}
    np.testing.assert_array_equal(policy.scores(private), policy.scores(obs))
    assert policy.act(private) == policy.act(obs)
    assert policy.scores(observation(18)).shape == (18,)
    assert snapshot(policy) == before


@pytest.mark.parametrize("mutation", ["schema", "names", "nonfinite", "shape", "empty", "mask_dtype",
    "mask_shape", "no_legal", "missing_stop", "two_stops", "fractional_stop"])
def test_observation_guards(mutation):
    obs = observation()
    if mutation == "schema": obs["feature_schema"] = "changed"
    elif mutation == "names": obs["feature_names"] = tuple(reversed(NAMES))
    elif mutation == "nonfinite": obs["option_features"][0, 0] = np.nan
    elif mutation == "shape": obs["option_features"] = np.zeros((5, 3))
    elif mutation == "empty": obs["option_features"] = np.zeros((0, len(NAMES)))
    elif mutation == "mask_dtype": obs["action_mask"] = np.ones(5, dtype=int)
    elif mutation == "mask_shape": obs["action_mask"] = np.ones(4, dtype=bool)
    elif mutation == "no_legal": obs["action_mask"][:] = False
    elif mutation == "missing_stop": obs["option_features"][-1, 1] = 0
    elif mutation == "two_stops": obs["option_features"][0, 1] = 1
    else: obs["option_features"][0, 1] = .5
    with pytest.raises(ValueError): RankPolicy(NAMES).act(obs)


@pytest.mark.parametrize("temperature,regularization", [(.1, 0.), (.7, .2)])
def test_all_analytic_gradients_match_finite_differences(temperature, regularization):
    policy = RankPolicy(NAMES, seed=499, hidden_size=3)
    policy.parameters["wa"][:] = [.3, -.2, .1]
    batch = records()
    before = snapshot(policy)
    loss, gradients, metrics = policy._loss_and_gradients(batch, temperature, regularization)
    assert loss == pytest.approx(metrics["pairwise_loss"] + metrics["regularization_loss"])
    assert metrics["comparison_weight"] == 2.75 and metrics["slate_rows"] == 9
    epsilon = 1e-6
    for key, parameter in policy.parameters.items():
        numerical = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + epsilon
            plus = policy._loss_and_gradients(batch, temperature, regularization)[0]
            parameter[index] = original - epsilon
            minus = policy._loss_and_gradients(batch, temperature, regularization)[0]
            parameter[index] = original
            numerical[index] = (plus - minus) / (2 * epsilon)
        np.testing.assert_allclose(gradients[key], numerical, rtol=1e-5, atol=2e-9)
    assert snapshot(policy) == before


def test_weight_and_state_duplication_normalization_and_first_encoder_gradient():
    policy = RankPolicy(NAMES, seed=499, hidden_size=3)
    batch = records()
    original = policy._loss_and_gradients(batch)
    assert not np.any(original[1]["w1"]) and not np.any(original[1]["b1"])
    assert np.any(original[1]["wa"])
    larger = deepcopy(batch)
    for state in larger:
        for pair in state["comparisons"]: pair["weight"] *= 3
    for candidate in (larger, batch + deepcopy(batch)):
        loss, gradients, _ = policy._loss_and_gradients(candidate)
        assert loss == pytest.approx(original[0], abs=1e-15)
        for key in PARAMETERS: np.testing.assert_allclose(gradients[key], original[1][key], atol=1e-15)
    policy.update_comparisons(batch)
    assert np.any(policy._loss_and_gradients(batch)[1]["w1"])


@pytest.mark.parametrize("preferred", [1, 2])
def test_preference_learning_can_change_deployment_and_stop_choice(preferred):
    obs = observation(3)
    obs["option_features"][:] = 0
    obs["option_features"][:, 0] = [-1., 1., 0.]
    obs["option_features"][-1, 1] = 1.
    obs["option_features"][0, 2] = .02  # Initially greedy selects row 0.
    policy = RankPolicy(NAMES, seed=499)
    batch = [{"features": obs["option_features"], "comparisons": [
        {"preferred": preferred, "rejected": other, "weight": 1.} for other in range(3) if other != preferred]}]
    initial_loss = policy._loss_and_gradients(batch)[0]
    assert policy.act(obs) == 0
    for _ in range(100): policy.update_comparisons(batch)
    assert policy.act(obs) == preferred
    assert policy._loss_and_gradients(batch)[0] < initial_loss * .2
    assert policy.update_count == 100


def test_gradient_clipping_and_adam_first_step_are_exact():
    policy = RankPolicy(NAMES, seed=499, hidden_size=3)
    _, gradients, _ = policy._loss_and_gradients(records())
    before = {k: v.copy() for k, v in policy.parameters.items()}
    norm = float(np.sqrt(sum(np.sum(g * g) for g in gradients.values())))
    scale = min(1., .01 / max(norm, 1e-12))
    metrics = policy.update_comparisons(records(), max_grad_norm=.01)
    assert metrics["gradient_norm"] == norm
    assert metrics["gradient_scale"] == scale < 1
    for key in PARAMETERS:
        g = gradients[key] * scale
        m, v = .1 * g, .001 * g * g
        expected = before[key] - .002 * (m / (1 - .9)) / (np.sqrt(v / (1 - .999)) + 1e-8)
        np.testing.assert_array_equal(policy.adam_m[key], m)
        np.testing.assert_array_equal(policy.adam_v[key], v)
        np.testing.assert_array_equal(policy.parameters[key], expected)


@pytest.mark.parametrize("mutation", ["empty_batch", "extra_state", "missing_features", "nan_feature", "two_stops",
    "empty_pairs", "extra_pair", "negative_index", "large_index", "bool_index", "self", "duplicate", "reverse",
    "zero_weight", "negative_weight", "nan_weight", "bool_weight", "overflow_weights"])
def test_invalid_comparisons_fail_without_mutation(mutation):
    policy = RankPolicy(NAMES, seed=499)
    batch = records()
    pair = batch[0]["comparisons"][0]
    if mutation == "empty_batch": batch = []
    elif mutation == "extra_state": batch[0]["private"] = 4
    elif mutation == "missing_features": del batch[0]["features"]
    elif mutation == "nan_feature": batch[0]["features"][0, 0] = np.nan
    elif mutation == "two_stops": batch[0]["features"][0, 1] = 1
    elif mutation == "empty_pairs": batch[0]["comparisons"] = []
    elif mutation == "extra_pair": pair["unknown"] = 1
    elif mutation == "negative_index": pair["preferred"] = -1
    elif mutation == "large_index": pair["preferred"] = 9
    elif mutation == "bool_index": pair["preferred"] = True
    elif mutation == "self": pair["preferred"] = pair["rejected"]
    elif mutation == "duplicate": batch[0]["comparisons"].append(deepcopy(pair))
    elif mutation == "reverse": batch[0]["comparisons"].append({"preferred": pair["rejected"], "rejected": pair["preferred"], "weight": 1.})
    elif mutation == "zero_weight": pair["weight"] = 0
    elif mutation == "negative_weight": pair["weight"] = -1
    elif mutation == "nan_weight": pair["weight"] = float("nan")
    elif mutation == "bool_weight": pair["weight"] = True
    else:
        for state in batch:
            for value in state["comparisons"]: value["weight"] = 1e308
    before = snapshot(policy)
    with pytest.raises(ValueError): policy.update_comparisons(batch)
    assert snapshot(policy) == before


@pytest.mark.parametrize("field,value", [("learning_rate", 0), ("learning_rate", True), ("temperature", 0),
    ("temperature", float("nan")), ("regularization", -1), ("max_grad_norm", 0), ("learning_rate", 2**10000)])
def test_hyperparameters_rejected_before_update(field, value):
    policy = RankPolicy(NAMES, seed=499)
    before = snapshot(policy)
    with pytest.raises(ValueError): policy.update_comparisons(records(), **{field: value})
    assert snapshot(policy) == before


def test_stop_free_slate_and_input_records_not_mutated():
    policy = RankPolicy(NAMES, seed=499)
    x = observation()["option_features"][:-1].copy()
    batch = [{"features": x, "comparisons": [{"preferred": 1, "rejected": 0, "weight": 1.}]}]
    before = x.copy()
    policy.update_comparisons(batch)
    np.testing.assert_array_equal(x, before)


def test_checkpoint_exact_reload_and_continued_adam_rng_training(tmp_path):
    policy = RankPolicy(NAMES, seed=499)
    for _ in range(3): policy.update_comparisons(records())
    for _ in range(7): policy.act(observation(), deterministic=False)
    state = {"seed_provenance": {"training": {"start": 499000000, "count": 0}}, "note": "synthetic rows only"}
    path = tmp_path / "checkpoint"
    policy.save(path, state)
    restored = RankPolicy.load(path, feature_names=NAMES)
    assert snapshot(policy) == snapshot(restored)
    assert policy.training_state == restored.training_state == state
    assert policy.metadata == restored.metadata
    assert policy.weights_fingerprint() == restored.weights_fingerprint()
    for _ in range(9):
        assert policy.act(observation(), deterministic=False) == restored.act(observation(), deterministic=False)
        assert policy.update_comparisons(records()) == restored.update_comparisons(records())
    assert snapshot(policy) == snapshot(restored)
    assert set(path.iterdir()) == {path / "checkpoint.json", path / "arrays.npz"}
    for loader in (AdaptivePolicy.load, BalancedPolicy.load):
        with pytest.raises(ValueError): loader(path)


def rewrite_metadata(path, edit):
    metadata = json.loads((path / "checkpoint.json").read_text())
    edit(metadata)
    (path / "checkpoint.json").write_text(json.dumps(metadata))


@pytest.mark.parametrize("mutation", ["schema", "policy_schema", "feature_schema", "contract", "contract_bool",
    "extra", "names", "hidden_bool", "count_bool", "negative_count", "rng_bool", "rng_extra", "rng_overflow",
    "rng_even_increment", "state_type", "provenance", "provenance_bool", "weight_hash", "array_hash", "duplicate_json",
    "nan_json", "overflow_json"])
def test_checkpoint_metadata_tampering_rejected(tmp_path, mutation):
    policy = RankPolicy(NAMES, seed=499)
    policy.save(tmp_path, {"seed_provenance": {"count": 1}})
    def edit(m):
        if mutation in ("schema", "policy_schema", "feature_schema"): m[mutation] = "old"
        elif mutation == "contract": m["policy_contract"]["scores"] = "changed"
        elif mutation == "contract_bool": m["policy_contract"]["prior"]["stop_prior"] = False
        elif mutation == "extra": m["unknown"] = 1
        elif mutation == "names": m["feature_names"] = list(reversed(NAMES))
        elif mutation == "hidden_bool": m["hidden_size"] = True
        elif mutation == "count_bool": m["update_count"] = False
        elif mutation == "negative_count": m["update_count"] = -1
        elif mutation == "rng_bool": m["rng_state"]["has_uint32"] = False
        elif mutation == "rng_extra": m["rng_state"]["unknown"] = 0
        elif mutation == "rng_overflow": m["rng_state"]["state"]["state"] = 2**128
        elif mutation == "rng_even_increment": m["rng_state"]["state"]["inc"] = 2
        elif mutation == "state_type": m["training_state"] = []
        elif mutation == "provenance": m["seed_provenance"]["count"] = 2
        elif mutation == "provenance_bool": m["seed_provenance"]["count"] = True
        elif mutation == "weight_hash": m["weights_sha256"] = "0" * 64
        elif mutation == "array_hash": m["arrays_sha256"] = "0" * 64
    rewrite_metadata(tmp_path, edit)
    meta = tmp_path / "checkpoint.json"
    if mutation == "duplicate_json": meta.write_text(meta.read_text().replace('{', '{"schema":"duplicate",', 1))
    if mutation == "nan_json": meta.write_text(meta.read_text().replace('"update_count": 0', '"update_count": NaN'))
    if mutation == "overflow_json": meta.write_text(meta.read_text().replace('"update_count": 0', '"update_count": 1e999'))
    with pytest.raises(ValueError): RankPolicy.load(tmp_path)


@pytest.mark.parametrize("mutation", ["extra", "shape", "dtype", "nan", "negative_variance", "pickle", "duplicate_member", "oversized_member"])
def test_rehashed_bad_npz_rejected(tmp_path, mutation):
    policy = RankPolicy(NAMES, seed=499, hidden_size=3)
    policy.save(tmp_path)
    with np.load(tmp_path / "arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if mutation == "extra": arrays["private"] = np.zeros(1)
    elif mutation == "shape": arrays["param_wa"] = np.zeros(4)
    elif mutation == "dtype": arrays["param_wa"] = arrays["param_wa"].astype(np.float32)
    elif mutation == "nan": arrays["param_wa"][0] = np.nan
    elif mutation == "negative_variance": arrays["adam_v_wa"][0] = -1
    elif mutation == "pickle": arrays["param_wa"] = np.array([{}, {}, {}], dtype=object)
    np.savez_compressed(tmp_path / "arrays.npz", **arrays)
    if mutation in ("duplicate_member", "oversized_member"):
        raw = (tmp_path / "arrays.npz").read_bytes()
        with zipfile.ZipFile(io.BytesIO(raw)) as old:
            members = [(x.filename, old.read(x)) for x in old.infolist()]
        with zipfile.ZipFile(tmp_path / "arrays.npz", "w") as archive:
            for name, content in members:
                archive.writestr(name, b"x" * 5000 if mutation == "oversized_member" and name == "param_wa.npy" else content)
            if mutation == "duplicate_member":
                with pytest.warns(UserWarning): archive.writestr(members[0][0], members[0][1])
    rewrite_metadata(tmp_path, lambda m: m.update(arrays_sha256=hashlib.sha256((tmp_path / "arrays.npz").read_bytes()).hexdigest()))
    with pytest.raises(ValueError): RankPolicy.load(tmp_path)


@pytest.mark.parametrize("mutation", ["bad_state", "nonfinite", "negative_variance", "count_bool", "stale_temporary"])
def test_save_rejects_bad_state_before_replacing_existing_checkpoint(tmp_path, mutation):
    policy = RankPolicy(NAMES, seed=499)
    policy.save(tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    kwargs = {}
    if mutation == "bad_state": kwargs["training_state"] = {"bad": float("nan")}
    elif mutation == "nonfinite": policy.parameters["wa"][0] = np.nan
    elif mutation == "negative_variance": policy.adam_v["wa"][0] = -1
    elif mutation == "count_bool": policy.update_count = False
    else: (tmp_path / "arrays.tmp.npz").write_bytes(b"not ours")
    with pytest.raises(ValueError): policy.save(tmp_path, **kwargs)
    for name, blob in before.items(): assert (tmp_path / name).read_bytes() == blob
    if mutation == "stale_temporary": assert (tmp_path / "arrays.tmp.npz").read_bytes() == b"not ours"


def test_feature_name_and_temperature_constructor_guards():
    for names in (None, (), ("stop",), NAMES + ("stop",)):
        with pytest.raises(ValueError): RankPolicy(names)
    for kwargs in ({"hidden_size": True}, {"hidden_size": 0}, {"seed": True}, {"seed": -1}):
        with pytest.raises(ValueError): RankPolicy(NAMES, **kwargs)
    policy = RankPolicy(NAMES)
    for value in (True, 0, -1, np.nan):
        with pytest.raises(ValueError): policy.probabilities(observation(), temperature=value)
    assert CHECKPOINT_SCHEMA != POLICY_SCHEMA
    assert POLICY_CONTRACT["prior"]["stop_prior"] == 0.


def test_save_contract_metadata_is_detached_and_directory_guard_precedes_writes(tmp_path):
    policy = RankPolicy(NAMES, seed=499)
    policy.save(tmp_path)
    policy.metadata["policy_contract"]["prior"]["stop_prior"] = 7.
    assert POLICY_CONTRACT["prior"]["stop_prior"] == 0.
    initial_arrays = (tmp_path / "arrays.npz").read_bytes()
    (tmp_path / "checkpoint.json").unlink()
    (tmp_path / "checkpoint.json").mkdir()
    with pytest.raises(ValueError, match="must be files"):
        policy.save(tmp_path)
    assert (tmp_path / "arrays.npz").read_bytes() == initial_arrays
    assert not (tmp_path / "arrays.tmp.npz").exists()
    assert not (tmp_path / "checkpoint.tmp.json").exists()


def test_checkpoint_symlink_target_rejected_without_writes(tmp_path):
    target = tmp_path / "outside.bin"
    target.write_bytes(b"preserve")
    output = tmp_path / "checkpoint"
    output.mkdir()
    try:
        (output / "arrays.npz").symlink_to(target)
    except OSError:
        pytest.skip("Symlink creation unavailable")
    with pytest.raises(ValueError, match="symlinks"):
        RankPolicy(NAMES, seed=499).save(output)
    assert target.read_bytes() == b"preserve"
    assert not (output / "checkpoint.json").exists()


def test_largest_public_option_matrix_includes_the_extra_stop_row():
    policy = RankPolicy(NAMES, seed=499, hidden_size=1)
    obs = observation(32 * 512 + 1)
    assert policy.scores(obs).shape == (16385,)
    assert obs["action_mask"][policy.act(obs)]
    with pytest.raises(ValueError, match="bounded"):
        policy.scores(observation(16386))
