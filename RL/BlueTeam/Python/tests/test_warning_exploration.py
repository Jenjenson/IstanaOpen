"""Learning diagnostics on a small action contract, without claiming native gains."""
from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest

from triad_rl.adaptive_inputs import INPUT_SCHEMA, DEFAULT_CATALOGUE
from triad_rl.directional_inputs import BOSON_PLUS_640_18MM
from triad_rl.temporal_inputs import TemporalConfig
from triad_rl.warning_algorithms import MaskedA2CPolicy, MaskedPPOPolicy
from triad_rl.warning_policy import WarningPolicy, _entropy_and_gradient, categorical_entropy


POLICIES = [WarningPolicy, MaskedA2CPolicy, MaskedPPOPolicy]


def context(extra_profile=False):
    # Two legal orientations make a reproducible bandit for optimizer tests.
    sensor = deepcopy(BOSON_PLUS_640_18MM)
    sensor.update(yaw_bins_deg=[0., 45.], pitch_bins_deg=[0.])
    catalogue = [sensor]
    if extra_profile:
        other = deepcopy(sensor)
        other["id"] = "not-selected"
        catalogue.append(other)
    return {"completedSteps": 0, "committed": False,
        "coordinateSystem": "unreal_xy_relative_m_z_up", "catalogue": catalogue,
        "temporalConfig": asdict(TemporalConfig()),
        "publicSnapshot": {
            "schema": INPUT_SCHEMA, "timestamp": 0., "sites": [[45., 0.]],
            "placements": [], "budget_total": 1., "budget_remaining": 1.,
            "max_sites": 1, "min_separation": 20., "weather": {}, "tracks": [],
            "blocked_sites": [], "available_sensor_ids": ["thermal"],
            "forecast": {"approach_weights": [1 / 8] * 8, "altitude": 45.,
                         "target_size_m": .4, "angular_uncertainty": .1},
        }}


START = [{"profileId": "thermal", "siteId": 0, "yawDeg": 0., "pitchDeg": 0.}]


def probabilities(logits, mask):
    result = np.zeros(len(logits))
    result[mask] = np.exp(logits[mask] - max(logits[mask]))
    return result / result.sum()


def records(policy, action):
    """Exact current-policy records for the two-action, then forced STOP contract."""
    result = []
    for step, (selected, mask) in enumerate([
            (action, np.array([True, True, False])),
            (2, np.array([False, False, True]))]):
        p = probabilities(policy.logits(), mask)
        if isinstance(policy, (MaskedA2CPolicy, MaskedPPOPolicy)):
            result.append({"action": selected, "old_probability": float(p[selected]),
                           "probabilities": p, "mask": mask, "step": step,
                           "value": float(policy.values[step])})
        else:
            result.append((selected, p))
    return result


@pytest.mark.parametrize("policy_type", POLICIES)
def test_calibrated_warm_start_explores_legal_selected_sensor_actions(policy_type):
    ctx = context(extra_profile=True)
    policy = policy_type(ctx, seed=11, sensor_count=1)
    report = policy.initialize_from_placements(START, baseline_action_probability=.8)
    assert report["initial_baseline_action_probability"] == pytest.approx(.8)
    assert report["initial_exploration_probability"] == pytest.approx(.2)
    assert report["legal_baseline_actions"] == report["legal_alternative_actions"] == 1
    assert report["strength"] == pytest.approx(np.log(4))
    assert policy.plan(ctx, deterministic=True)[0] == START
    changed = 0
    for _ in range(48):
        placements, _ = policy.plan(ctx)
        assert len(placements) == 1 and placements[0]["profileId"] == "thermal"
        changed += placements != START
    # This would be essentially impossible under the previous +10 logit gap.
    assert 4 <= changed <= 20


def test_warm_start_rejects_sensor_not_selected_for_training():
    policy = WarningPolicy(context(extra_profile=True), sensor_count=1)
    excluded = [{**START[0], "profileId": "not-selected"}]
    with pytest.raises(ValueError, match="not legal"):
        policy.initialize_from_placements(excluded)
    assert not np.any(policy.logits())


@pytest.mark.parametrize("policy_type", POLICIES)
def test_saved_sensor_restriction_masks_radial_and_unselected_profiles(policy_type, tmp_path):
    ctx = context(extra_profile=True)
    ctx["catalogue"].insert(0, deepcopy(DEFAULT_CATALOGUE[0]))  # Omnidirectional RF.
    ctx["publicSnapshot"]["available_sensor_ids"] = ["rf", "thermal", "not-selected"]
    before = deepcopy(ctx)
    policy = policy_type(ctx, seed=9, sensor_count=1, allowed_sensor_ids=["thermal"])
    report = policy.initialize_from_placements(START)
    assert report["legal_alternative_actions"] == 1
    # Even very large learned/logged logits may never override a selected-type mask.
    for index, option in enumerate(policy.contract["options"]):
        if option["sensor_id"] != "thermal":
            policy.option_logits[index] = 20.
    checkpoint = tmp_path / "restricted.json"
    policy.save(checkpoint)
    restored = policy_type.load(checkpoint, ctx)
    assert restored.allowed_sensor_ids == ("thermal",)
    for candidate in (policy, restored):
        assert candidate.plan(ctx, deterministic=True)[0][0]["profileId"] == "thermal"
        for _ in range(12):
            placements, _ = candidate.plan(ctx)
            assert placements[0]["profileId"] == "thermal"
    assert ctx == before
    unavailable = deepcopy(ctx)
    unavailable["publicSnapshot"]["available_sensor_ids"] = ["rf"]
    with pytest.raises(ValueError, match="unavailable"):
        restored.plan(unavailable)
    with pytest.raises(ValueError, match="unavailable"):
        policy_type.load(checkpoint, unavailable)


@pytest.mark.parametrize("allowed", [[], ["thermal", "thermal"], ["missing"], ["not-selected"], "thermal"])
def test_invalid_or_unavailable_sensor_restrictions_rejected(allowed):
    with pytest.raises(ValueError, match="sensor|sensor profile|Allowed"):
        WarningPolicy(context(extra_profile=True), sensor_count=1, allowed_sensor_ids=allowed)


def test_explicit_historical_warm_start_strength_is_supported():
    policy = WarningPolicy(context(), sensor_count=1)
    report = policy.initialize_from_placements(START, strength=10.)
    assert report["strength"] == 10.
    assert report["initial_baseline_action_probability"] > .9999


def test_warm_start_reports_when_native_mask_offers_no_alternative():
    ctx = context()
    ctx["catalogue"][0]["yaw_bins_deg"] = [0.]
    policy = WarningPolicy(ctx, sensor_count=1)
    report = policy.initialize_from_placements(START)
    assert report["initial_exploration_probability"] == 0.
    assert report["legal_alternative_actions"] == 0
    assert policy.plan(ctx)[0] == START
    with pytest.raises(ValueError, match="fresh"):
        policy.initialize_from_placements(START)


@pytest.mark.parametrize("target", [0., 1., float("nan"), True, "0.8"])
def test_invalid_warm_start_probability_rejected(target):
    with pytest.raises(ValueError, match="action probability"):
        WarningPolicy(context(), sensor_count=1).initialize_from_placements(
            START, baseline_action_probability=target)


@pytest.mark.parametrize("policy_type", POLICIES)
@pytest.mark.parametrize("coefficient", [-.1, 1.1, float("nan"), True])
def test_invalid_entropy_configuration_rejected(policy_type, coefficient):
    with pytest.raises(ValueError, match="Entropy coefficient"):
        policy_type(context(), entropy_coefficient=coefficient)


def test_masked_entropy_gradient_matches_finite_differences():
    logits, mask = np.array([1.3, -.4, 20., .2]), np.array([True, True, False, True])
    _, gradient = _entropy_and_gradient(probabilities(logits, mask))
    numerical = np.zeros_like(logits)
    for index in range(len(logits)):
        plus, minus = logits.copy(), logits.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical[index] = (categorical_entropy(probabilities(plus, mask))
                            - categorical_entropy(probabilities(minus, mask))) / 2e-6
    np.testing.assert_allclose(gradient, numerical, atol=1e-9)
    assert gradient[2] == 0 and abs(gradient.sum()) < 1e-12
    assert categorical_entropy([0., 1., 0.]) == 0.


@pytest.mark.parametrize("policy_type", [WarningPolicy, MaskedA2CPolicy])
def test_single_update_reward_plus_entropy_gradient_is_correct(policy_type):
    policy = policy_type(context(), sensor_count=1, entropy_coefficient=.07)
    policy.initialize_from_placements(START)
    rewards = np.array([1., .2])
    batch = [(records(policy, action), float(reward))
             for action, reward in enumerate(rewards)]
    advantages = np.array([.8, -.8]) if policy_type is WarningPolicy else rewards
    initial = policy.logits()
    mask = np.array([True, True, False])

    def objective(logits):
        p = probabilities(logits, mask)
        reward_term = float(np.dot(advantages, np.log(p[:2]))) / 2
        # Each complete episode includes one placement and one forced STOP.
        return reward_term + .07 * categorical_entropy(p) / 2

    numerical = np.zeros_like(initial)
    for index in range(len(initial)):
        plus, minus = initial.copy(), initial.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical[index] = (objective(plus) - objective(minus)) / 2e-6
    report = policy.update(batch)
    np.testing.assert_allclose(policy.m / .1,
                               numerical / max(1., np.linalg.norm(numerical)), atol=1e-9)
    assert report["objective_loss"] == pytest.approx(-objective(initial))
    assert report["parameter_delta_norm"] == pytest.approx(np.linalg.norm(policy.logits() - initial))


@pytest.mark.parametrize("actor_logits", [[.05, -.03, 9.], [2., -2., 9.], [-2., 2., 9.]])
@pytest.mark.parametrize("entropy_coefficient", [0., .07])
def test_ppo_clipped_surrogate_and_entropy_have_exact_gradient(actor_logits, entropy_coefficient):
    policy = MaskedPPOPolicy(context(), sensor_count=1, entropy_coefficient=entropy_coefficient)
    # Record a uniform behavior policy, then evaluate all PPO clipping regimes.
    batch = [(records(policy, 0), 2.), (records(policy, 1), -1.)]
    _, samples = policy._samples(batch)
    advantages = np.array([2., 2., -1., -1.])
    policy.option_logits[:] = actor_logits
    analytic, _ = policy._objective_gradient(samples, advantages, .2)

    def objective():
        _, report = policy._objective_gradient(samples, advantages, .2)
        return -report["policy_loss"] + entropy_coefficient * report["entropy"]

    numerical = np.zeros_like(policy.option_logits)
    for index in range(len(numerical)):
        policy.option_logits[index] += 1e-6
        plus = objective()
        policy.option_logits[index] -= 2e-6
        minus = objective()
        policy.option_logits[index] += 1e-6
        numerical[index] = (plus - minus) / 2e-6
    np.testing.assert_allclose(analytic, numerical, atol=1e-9)
    assert analytic[-1] == 0


@pytest.mark.parametrize("policy_type", POLICIES)
def test_entropy_regularization_moves_away_from_saturation_with_zero_reward(policy_type):
    policy = policy_type(context(), sensor_count=1, entropy_coefficient=.1)
    policy.initialize_from_placements(START)
    before = policy.logits()
    report = policy.update([(records(policy, 0), 0.), (records(policy, 1), 0.)])
    assert policy.logits()[0] - policy.logits()[1] < before[0] - before[1]
    assert report["entropy"] > 0 and report["parameter_delta_norm"] > 0
    assert all(np.isfinite(value) for value in report.values() if isinstance(value, (int, float)))


@pytest.mark.parametrize("policy_type", POLICIES)
def test_rewarded_alternative_can_overtake_common_sense_initialization(policy_type):
    policy = policy_type(context(), seed=11, sensor_count=1)
    policy.initialize_from_placements(START)
    generator = np.random.default_rng(31)
    mask = np.array([True, True, False])
    for _ in range(32):
        p = probabilities(policy.logits(), mask)
        batch = []
        for _ in range(32):
            action = int(generator.choice(len(p), p=p))
            batch.append((records(policy, action), float(action == 1)))
        policy.update(batch)
    result, _ = policy.plan(context(), deterministic=True)
    assert result[0]["yawDeg"] == 45.
    assert probabilities(policy.logits(), mask)[1] > .8


@pytest.mark.parametrize("policy_type", POLICIES)
def test_entropy_configuration_roundtrip_and_legacy_checkpoint_load(policy_type, tmp_path):
    policy = policy_type(context(), sensor_count=1, entropy_coefficient=.05)
    policy.initialize_from_placements(START)
    checkpoint = tmp_path / "policy.json"
    policy.save(checkpoint)
    restored = policy_type.load(checkpoint, context())
    assert restored.entropy_coefficient == .05
    assert restored.plan(context())[0] == policy.plan(context())[0]
    data = json.loads(checkpoint.read_text())
    del data["entropy_coefficient"]
    del data["allowed_sensor_ids"]
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(data))
    loaded = policy_type.load(legacy, context())
    assert loaded.entropy_coefficient == 0.
    assert loaded.allowed_sensor_ids is None
    np.testing.assert_array_equal(loaded.logits(), policy.logits())
