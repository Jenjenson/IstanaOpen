"""Finite-action learning from paired feedback, independent of native performance claims."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest

from test_local_refinement_policy import assert_native_mask_accepts, context_and_base
from triad_rl.local_refinement_policy import LocalRefinementPPOPolicy
from triad_rl.paired_layout_bandit import PairedLayoutBanditPolicy
from triad_rl.warning_algorithms import TRAINING_ALGORITHMS


def initialized(seed=917, **kwargs):
    context, base = context_and_base()
    for row in base:
        row["pitchDeg"] = 20.
    policy = PairedLayoutBanditPolicy(context, seed=seed, sensor_count=4, **kwargs)
    policy.initialize_from_placements(base, baseline_action_probability=.8)
    return context, base, policy


def one_alternative(**kwargs):
    context, _ = context_and_base()
    context["publicSnapshot"].update(sites=[[45., 0.]], max_sites=1, budget_total=1.,
        budget_remaining=1., blocked_sites=[])
    context["catalogue"][0].update(yaw_bins_deg=[0., 45.], pitch_bins_deg=[20.])
    base = [{"profileId": "thermal", "siteId": 0, "yawDeg": 0., "pitchDeg": 20.}]
    policy = PairedLayoutBanditPolicy(context, sensor_count=1, **kwargs)
    policy.initialize_from_placements(base)
    return context, base, policy


def test_arms_are_exact_public_single_edits_and_initial_deployment_is_contractor():
    context, base, policy = initialized()
    assert len(policy.arms) == 21
    assert policy.plan(context, deterministic=True) == (base, [])
    assert policy.arms[0]["placements"] == base
    for arm in policy.arms[1:]:
        changed = [index for index, (a, b) in enumerate(zip(arm["placements"], base)) if a != b]
        assert changed == [arm["slot"]]
        assert len(arm["placements"]) == 4
        assert {row["profileId"] for row in arm["placements"]} == {"thermal"}
        assert any(row["placement"] == arm["placements"][arm["slot"]]
                   for row in policy._local.slot_candidates[arm["slot"]])
    for kind in ("site", "yaw", "pitch"):
        arm = next(row for row in policy.arms if row["kind"] == kind)
        assert_native_mask_accepts(context, arm["placements"])
    assert policy._warm_start_report["initialization"] == "balanced_action_value"
    assert policy._warm_start_report["legacy_entropy_coefficient_ignored"] == .01
    assert "no automatic combinations" in policy.action_space["limitation"]
    assert "not a confidence-interval" in policy.action_space["uncertaintyNote"]


def test_batched_warmup_reserves_every_arm_without_inventing_observations():
    context, _, policy = initialized()
    for round_number in range(8):
        episodes = []
        for _ in range(20):
            _, records = policy.plan(context)
            assert records[0]["selection"] == "balanced"
            episodes.append((records, 0.))
        assert {records[0]["action"] for records, _ in episodes} == set(range(1, 21))
        assert policy.counts.tolist() == [0] + [round_number] * 20
        assert policy._reservations().tolist() == [0] + [1] * 20
        update = policy.update(episodes)
        assert update["observation_count"] == 20 * (round_number + 1)
        assert update["pending_reservations"] == 0
    _, records = policy.plan(context)
    assert records[0]["selection"] == "ucb"
    assert records[0]["ucb_score_s"] == pytest.approx(2 * math.sqrt(2 * math.log(160) / 8))
    assert not any(key in update for key in ("gradient_norm", "policy_loss", "value_loss", "epochs"))


@pytest.mark.parametrize("better_arm", [3, 17])
def test_reward_values_drive_learning_and_ucb_allocation_without_pose_hardcoding(better_arm):
    context, base, policy = initialized(warmup_samples_per_arm=2)
    for _ in range(2):
        episodes = []
        for _ in range(20):
            _, records = policy.plan(context)
            reward = 4. if records[0]["action"] == better_arm else -.5
            episodes.append((records, reward))
        report = policy.update(episodes)
    learned, _ = policy.plan(context, deterministic=True)
    assert learned == policy.arms[better_arm]["placements"] and learned != base
    assert policy.means[better_arm] == 4.
    assert report["arm_mean_advantage_s"][better_arm] == 4.
    chosen = []
    for _ in range(20):
        _, records = policy.plan(context)
        chosen.append(records[0]["action"])
        policy.update([(records, 4. if records[0]["action"] == better_arm else -.5)])
    assert chosen.count(better_arm) > 10  # Uniform exploration would allocate about one.
    assert_native_mask_accepts(context, learned)


def test_exact_online_mean_variance_and_standard_error_use_observed_samples_only():
    context, _, policy = one_alternative(warmup_samples_per_arm=3)
    episodes = [(policy.plan(context)[1], reward) for reward in (-2., 0., 4.)]
    assert policy.counts.tolist() == [0, 0]
    assert policy._reservations().tolist() == [0, 3]
    report = policy.update(episodes)
    assert policy.means[1] == pytest.approx(2 / 3)
    assert policy.m2[1] == pytest.approx(sum((x - 2 / 3) ** 2 for x in (-2., 0., 4.)))
    assert report["arm_standard_error_s"][1] == pytest.approx(np.std([-2., 0., 4.], ddof=1) / math.sqrt(3))
    assert report["parameter_delta_norm"] == pytest.approx(2 / 3)
    assert policy.counts.tolist() == [0, 3] and not policy._pending
    assert "no gradient optimizer" in report["parameter_delta_definition"]


def test_negative_observed_arms_never_displace_known_zero_contractor():
    context, base, policy = one_alternative()
    policy.update([(policy.plan(context)[1], -1.)])
    assert policy.plan(context, deterministic=True)[0] == base
    assert policy.means[0] == 0. and policy.counts[0] == 0


def test_diagnostics_are_json_safe_and_describe_decision_time_not_future_statistics():
    context, _, policy = initialized()
    _, records = policy.plan(context)
    diagnostics = policy.episode_diagnostics(records)
    assert json.loads(json.dumps(diagnostics, allow_nan=False)) == diagnostics
    assert diagnostics["selectionMode"] == "balanced" and diagnostics["observedCount"] == 0
    assert diagnostics["estimatedAdvantageSeconds"] is None
    policy.update([(records, 3.)])
    assert policy.episode_diagnostics(records) == diagnostics


def test_invalid_batch_is_atomic_and_reservations_cannot_be_reused_or_changed():
    context, _, policy = initialized()
    _, first = policy.plan(context)
    _, second = policy.plan(context)
    before = deepcopy(policy._pending)
    with pytest.raises(ValueError, match="duplicated"):
        policy.update([(first, 1.), (first, 2.)])
    altered = deepcopy(second)
    altered[0]["observed_count"] = 20
    with pytest.raises(ValueError, match="modified"):
        policy.update([(first, 1.), (altered, 2.)])
    with pytest.raises(ValueError, match="finite"):
        policy.update([(first, 1.), (second, float("nan"))])
    assert policy._pending == before and policy.counts.sum() == 0
    policy.update([(first, 1.), (second, 2.)])
    with pytest.raises(ValueError, match="not pending"):
        policy.update([(first, 1.)])


def test_checkpoint_restores_observations_pending_reservations_and_random_sequence(tmp_path):
    context, _, policy = initialized()
    policy.update([(policy.plan(context)[1], 1.) for _ in range(8)])
    outstanding = [policy.plan(context)[1] for _ in range(6)]
    checkpoint = tmp_path / "bandit.json"
    assert len(policy.save(checkpoint)) == 64
    restored = TRAINING_ALGORITHMS["paired_bandit"]["factory"].load(checkpoint, context)
    assert isinstance(restored, PairedLayoutBanditPolicy)
    assert restored.arms == policy.arms and restored._pending == policy._pending
    assert restored.action_space == policy.action_space
    assert restored._next_reservation_id == policy._next_reservation_id
    assert restored.updates == policy.updates
    for name in ("counts", "means", "m2"):
        np.testing.assert_array_equal(getattr(restored, name), getattr(policy, name))
    for _ in range(20):
        a, records_a = policy.plan(context)
        b, records_b = restored.plan(context)
        assert a == b and policy._record_data(records_a[0]) == restored._record_data(records_b[0])
    batch = [(records, .5) for records in outstanding]
    assert policy.update(batch) == restored.update(batch)
    with pytest.raises(FileExistsError):
        restored.save(checkpoint)


@pytest.mark.parametrize("field", ["arms", "means", "counts", "m2", "next_reservation_id", "pending", "contract"])
def test_checkpoint_rejects_tampered_candidates_or_inconsistent_statistics(tmp_path, field):
    context, _, policy = initialized()
    policy.plan(context)
    checkpoint = tmp_path / "bandit.json"
    policy.save(checkpoint)
    data = json.loads(checkpoint.read_text())
    if field == "arms": data[field][1]["placements"][0]["siteId"] = 28
    if field == "means": data[field][1] = 1.  # No observed reward supports this value.
    if field == "counts": data[field][1] = 1.5
    if field == "m2": data[field][1] = -1.
    if field == "next_reservation_id": data[field] += 1
    if field == "pending": data[field][0]["reservationId"] = 1000
    if field == "contract": data[field]["sites"][0][0] += 1
    checkpoint.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        PairedLayoutBanditPolicy.load(checkpoint, context)


def test_private_red_changes_do_not_affect_arms_or_planning():
    context, _, policy = initialized()
    private = deepcopy(context)
    private.update(red_context={"hidden_routes": [[999, 888, 777]]},
                   warningEvidenceForEvaluationOnly={"return": 1e9})
    other_context, _, other = initialized()
    assert context == other_context
    for _ in range(8):
        assert policy.plan(context)[0] == other.plan(private)[0]


def test_selected_directional_inventory_and_current_legal_mask_survive_reload(tmp_path):
    context, base, _ = initialized()
    camera = deepcopy(context["catalogue"][0]); camera["id"] = "camera"
    radial = {"id": "radial", "label": "Radial RF", "cost": .1,
              "ranges": {"rf": 500.}, "strengths": {"rf": .9}, "height_m": 4.}
    context["catalogue"].extend([camera, radial])
    context["publicSnapshot"]["available_sensor_ids"].extend(["camera", "radial"])
    with pytest.raises(ValueError, match="limited-FOV"):
        PairedLayoutBanditPolicy(context, sensor_count=4, allowed_sensor_ids=["radial"])
    policy = PairedLayoutBanditPolicy(context, sensor_count=4, allowed_sensor_ids=["thermal"])
    policy.initialize_from_placements(base)
    checkpoint = tmp_path / "selected.json"
    policy.save(checkpoint)
    policy = PairedLayoutBanditPolicy.load(checkpoint, context)
    moved = next(arm for arm in policy.arms if arm["kind"] == "site")
    site = moved["placements"][moved["slot"]]["siteId"]
    changed = deepcopy(context)
    changed["publicSnapshot"]["blocked_sites"].append(site)
    for _ in range(12):
        layout, records = policy.plan(changed)
        assert len(layout) == 4 and {row["profileId"] for row in layout} == {"thermal"}
        assert site not in {row["siteId"] for row in layout}
        assert not records[0]["mask"][moved["id"]]
    changed["publicSnapshot"]["available_sensor_ids"].remove("thermal")
    with pytest.raises(ValueError, match="unavailable"):
        policy.plan(changed)


def test_forced_keep_records_zero_advantage_and_no_alternative_invention():
    context, base, _ = one_alternative()
    context["catalogue"][0]["yaw_bins_deg"] = [0.]
    policy = PairedLayoutBanditPolicy(context, sensor_count=1)
    policy.initialize_from_placements(base)
    layout, records = policy.plan(context)
    assert layout == base and records[0]["selection"] == "forced_keep"
    assert records[0]["probabilities"].tolist() == [1.]
    with pytest.raises(ValueError, match="zero paired"):
        policy.update([(records, 1.)])
    report = policy.update([(records, 0.)])
    assert report["arm_mean_advantage_s"] == [0.]
    assert report["arm_standard_error_s"] == [0.]


def test_uninitialized_and_legacy_schema_calls_fail_without_changing_local_ppo(tmp_path):
    context, base = context_and_base()
    policy = TRAINING_ALGORITHMS["paired_bandit"]["factory"](context, sensor_count=4)
    for call in (lambda: policy.plan(context), lambda: policy.update([]), lambda: policy.save(tmp_path / "bad.json")):
        with pytest.raises(ValueError, match="Initialize"):
            call()
    old = LocalRefinementPPOPolicy(context, sensor_count=4)
    old.initialize_from_placements(base)
    checkpoint = tmp_path / "local.json"
    old.save(checkpoint)
    assert LocalRefinementPPOPolicy.load(checkpoint, context).plan(context, deterministic=True)[0] == base
    with pytest.raises(ValueError, match="schema"):
        PairedLayoutBanditPolicy.load(checkpoint, context)
