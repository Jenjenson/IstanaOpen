"""Local PPO correctness, legal deployment and controlled learning (not native wins)."""
from copy import deepcopy
from dataclasses import asdict
import json
import math

import numpy as np
import pytest

from triad_rl.adaptive_inputs import INPUT_SCHEMA
from triad_rl.directional_inputs import BOSON_PLUS_640_18MM, apply_placement, build_observation
from triad_rl.istana_live import public_planning_inputs
from triad_rl.local_refinement_policy import LocalRefinementPPOPolicy
from triad_rl.temporal_inputs import TemporalConfig
from triad_rl.warning_algorithms import MaskedPPOPolicy, TRAINING_ALGORITHMS


def context_and_base():
    sites = [[radius * math.cos(i * math.pi / 8), radius * math.sin(i * math.pi / 8)]
             for radius in (30., 45.) for i in range(16)]
    state = {
        "schema": INPUT_SCHEMA, "timestamp": 0., "sites": sites, "placements": [],
        "budget_total": 4., "budget_remaining": 4., "max_sites": 4,
        "min_separation": 20., "weather": {}, "tracks": [], "blocked_sites": list(range(6)),
        "available_sensor_ids": ["thermal"],
        "forecast": {"approach_weights": [1 / 8] * 8, "altitude": 45.,
                     "target_size_m": .4, "angular_uncertainty": .1},
    }
    context = {"runId": "local-test", "revision": 1, "completedSteps": 0,
        "committed": False, "coordinateSystem": "unreal_xy_relative_m_z_up",
        "worldOriginCm": {"x": 0., "y": 0., "z": 0.},
        "catalogue": [deepcopy(BOSON_PLUS_640_18MM)], "publicSnapshot": state,
        "temporalConfig": asdict(TemporalConfig())}
    base = [{"profileId": "thermal", "siteId": site, "yawDeg": float(i * 90), "pitchDeg": 10.}
            for i, site in enumerate((16, 20, 24, 28))]
    return context, base


def initialized(seed=917, **kwargs):
    context, base = context_and_base()
    policy = LocalRefinementPPOPolicy(context, seed, sensor_count=4, **kwargs)
    policy.initialize_from_placements(base, baseline_action_probability=.8)
    return context, base, policy


def assert_native_mask_accepts(context, placements):
    """Independent check through the existing public/native deployment adapter."""
    state, catalogue, _ = public_planning_inputs(context)
    for placement in placements:
        observation = build_observation(state, catalogue)
        action = next(i for i, row in enumerate(observation["options"])
                      if not row["stop"] and row["sensor_id"] == placement["profileId"]
                      and row["site_index"] == placement["siteId"]
                      and row["yaw_deg"] == placement["yawDeg"]
                      and row["pitch_deg"] == placement["pitchDeg"])
        assert observation["action_mask"][action]
        state = apply_placement(state, action, catalogue)
    assert len(state["placements"]) == len(placements)
    assert state["budget_remaining"] >= 0


def synthetic_delta(placements, base):
    # A known separable local improvement, deliberately unrelated to native
    # warning results. Moving or turning is costly; raising pitch is beneficial.
    return sum((2. if row["pitchDeg"] == 20. else -1. if row["pitchDeg"] == 0. else 0.)
               - .5 * (row["siteId"] != original["siteId"])
               - (row["yawDeg"] != original["yawDeg"])
               for row, original in zip(placements, base))


def test_uniform_local_actions_begin_at_keep_and_use_no_private_red_information():
    context, base, policy = initialized()
    before = deepcopy(context)
    state, catalogue, _ = public_planning_inputs(context)
    assert int(build_observation(state, catalogue)["action_mask"][:-1].sum()) == 624
    greedy, records = policy.plan(context, deterministic=True)
    assert greedy == base and len(records) == 4
    assert len(policy.option_logits) <= 28
    for slot, record in enumerate(records):
        assert record["step"] == slot
        assert record["action"] == policy._slot_offsets[slot]
        assert record["old_probability"] == pytest.approx(1 / record["mask"].sum())
        np.testing.assert_allclose(record["probabilities"][record["mask"]], record["old_probability"])
        assert not record["probabilities"][~record["mask"]].any()
    metadata = policy._warm_start_report
    assert metadata["initialization"] == "uniform_local_edits"
    assert metadata["legacy_baseline_action_probability_ignored"] == .8
    assert metadata["initial_full_layout_probability"] == pytest.approx(
        np.prod(metadata["initial_keep_probabilities"]))
    assert metadata["actionSpace"]["deploymentRadiusMeters"] == [30., 150.]
    assert "Not unrestricted" in metadata["actionSpace"]["limitation"]
    changed = deepcopy(context)
    changed.update(red_context={"private_routes": [[999, 888, 777]]},
                   warningEvidenceForEvaluationOnly={"score": 1e9},
                   private_truth={"future_drone": "hidden"})
    assert policy.plan(changed, deterministic=True)[0] == base
    assert policy.plan(changed, rng=np.random.default_rng(23))[0] == policy.plan(
        context, rng=np.random.default_rng(23))[0]
    assert context == before


@pytest.mark.parametrize("seed", [917, 918, 919])
def test_controlled_reward_changes_multiple_greedy_sensors_within_25_ppo_updates(seed):
    context, base, policy = initialized(seed)
    assert synthetic_delta(policy.plan(context, deterministic=True)[0], base) == 0.
    reports = []
    for _ in range(25):
        episodes = []
        for _ in range(20):
            layout, records = policy.plan(context)
            episodes.append((records, synthetic_delta(layout, base)))
        reports.append(policy.update(episodes))
    greedy, _ = policy.plan(context, deterministic=True)
    assert sum(row != original for row, original in zip(greedy, base)) >= 2
    assert synthetic_delta(greedy, base) >= 6.
    assert policy.optimizer_steps == 100 and policy.updates == 25
    assert all(row["parameter_delta_norm"] > 0 for row in reports)
    assert all(row["reward_mode"] == "paired_contractor_delta" for row in reports)
    assert "mean_training_warning_s" not in reports[-1]
    assert "mean_training_advantage_s" in reports[-1]
    for name in ("gradient_norm", "policy_loss", "value_loss", "entropy", "objective_loss"):
        assert all(np.isfinite(row[name]) for row in reports)
    assert_native_mask_accepts(context, greedy)


def test_interacting_site_edits_keep_the_whole_layout_legal():
    context, _ = context_and_base()
    context["publicSnapshot"].update(sites=[[-40., 0.], [40., 0.], [0., 0.], [-80., 0.], [80., 0.]],
        max_sites=2, budget_total=2., budget_remaining=2., blocked_sites=[], deployment_min_radius=0.)
    base = [{"profileId": "thermal", "siteId": site, "yawDeg": 0., "pitchDeg": 10.} for site in (0, 1)]
    policy = LocalRefinementPPOPolicy(context, sensor_count=2)
    policy.initialize_from_placements(base)
    center_actions = []
    for slot, candidates in enumerate(policy.slot_candidates):
        local = next(i for i, row in enumerate(candidates) if row["placement"]["siteId"] == 2)
        action = policy._slot_offsets[slot] + local
        center_actions.append(action)
        policy.option_logits[action] = 10.
    greedy, records = policy.plan(context, deterministic=True)
    assert greedy[0]["siteId"] == 2 and greedy[1]["siteId"] == 1
    assert records[0]["mask"][center_actions[0]]
    assert not records[1]["mask"][center_actions[1]]
    assert records[1]["probabilities"][center_actions[1]] == 0.
    assert_native_mask_accepts(context, greedy)
    for _ in range(6):
        layout, _ = policy.plan(context)
        assert_native_mask_accepts(context, layout)


@pytest.mark.parametrize("displacement", [.06, 3.])
def test_conditional_local_ppo_gradient_matches_finite_difference(displacement):
    context, _, policy = initialized(entropy_coefficient=.07)
    episodes = [(policy.plan(context)[1], reward) for reward in (-2., .5, 3.)]
    _, samples = policy._samples(episodes)
    advantages = np.linspace(-1.1, 1.3, len(samples))
    policy.option_logits[:] = np.linspace(-displacement, displacement, len(policy.option_logits))
    gradient, report = policy._objective_gradient(samples, advantages, .2)
    if displacement > 1:
        assert report["clip_fraction"] > 0
    numerical = np.zeros_like(gradient)
    for action in range(len(gradient)):
        original = policy.option_logits[action]
        values = []
        for delta in (1e-6, -1e-6):
            policy.option_logits[action] = original + delta
            _, result = policy._objective_gradient(samples, advantages, .2)
            values.append(-result["policy_loss"] + policy.entropy_coefficient * result["entropy"])
        numerical[action] = (values[0] - values[1]) / 2e-6
        policy.option_logits[action] = original
    np.testing.assert_allclose(gradient, numerical, atol=2e-9, rtol=2e-6)


def test_saved_local_policy_recomputes_candidates_and_preserves_rng_and_optimizer(tmp_path):
    context, base, policy = initialized()
    episodes = []
    for _ in range(12):
        layout, records = policy.plan(context)
        episodes.append((records, synthetic_delta(layout, base)))
    policy.update(episodes)
    checkpoint = tmp_path / "local.json"
    digest = policy.save(checkpoint)
    assert len(digest) == 64
    restored = TRAINING_ALGORITHMS["local_ppo"]["factory"].load(checkpoint, context)
    assert isinstance(restored, LocalRefinementPPOPolicy)
    assert restored.base_placements == base
    assert restored.slot_candidates == policy.slot_candidates
    assert restored.action_space == policy.action_space
    assert restored.allowed_sensor_ids == ("thermal",)
    assert restored.optimizer_steps == policy.optimizer_steps
    assert restored.updates == policy.updates
    for name in ("option_logits", "m", "v", "values", "value_updates"):
        np.testing.assert_array_equal(getattr(restored, name), getattr(policy, name))
    for _ in range(8):
        original_layout, original_records = policy.plan(context)
        loaded_layout, loaded_records = restored.plan(context)
        assert loaded_layout == original_layout
        for original, loaded in zip(original_records, loaded_records):
            assert loaded["action"] == original["action"]
            np.testing.assert_array_equal(loaded["probabilities"], original["probabilities"])
    assert policy.update(episodes) == restored.update(episodes)
    with pytest.raises(FileExistsError):
        policy.save(checkpoint)


@pytest.mark.parametrize("field", ["slot_candidates", "base_placements", "allowed_sensor_ids", "option_logits",
                                  "adam_v", "optimizer_steps", "action_space"])
def test_checkpoint_rejects_corrupt_or_incompatible_local_contract(tmp_path, field):
    context, _, policy = initialized()
    checkpoint = tmp_path / "local.json"
    policy.save(checkpoint)
    data = json.loads(checkpoint.read_text())
    if field == "slot_candidates":
        data[field][0][1]["placement"]["siteId"] = 24
    elif field == "base_placements":
        data[field][0]["siteId"] = data[field][1]["siteId"]
    elif field == "allowed_sensor_ids":
        data[field] = ["unknown"]
    elif field == "option_logits":
        data[field].append(0.)
    elif field == "adam_v":
        data[field][0] = -1.
    elif field == "optimizer_steps":
        data[field] = -1
    else:
        data[field]["nearbySitesPerSlot"] = 8
    checkpoint.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        LocalRefinementPPOPolicy.load(checkpoint, context)


def test_selection_stays_directional_and_exact_across_checkpoint(tmp_path):
    context, base = context_and_base()
    camera = deepcopy(BOSON_PLUS_640_18MM)
    camera["id"] = "camera"
    radial = {"id": "radial", "label": "Radial RF", "cost": .1,
              "ranges": {"rf": 500.}, "strengths": {"rf": .9}, "height_m": 4.}
    context["catalogue"].extend([camera, radial])
    context["publicSnapshot"]["available_sensor_ids"].extend(["camera", "radial"])
    default = LocalRefinementPPOPolicy(context, sensor_count=4)
    assert default.allowed_sensor_ids == ("camera", "thermal")
    with pytest.raises(ValueError, match="limited-FOV"):
        LocalRefinementPPOPolicy(context, sensor_count=4, allowed_sensor_ids=["radial"])
    policy = LocalRefinementPPOPolicy(context, sensor_count=4, allowed_sensor_ids=["thermal"])
    policy.initialize_from_placements(base)
    checkpoint = tmp_path / "selected.json"
    policy.save(checkpoint)
    loaded = LocalRefinementPPOPolicy.load(checkpoint, context)
    for _ in range(10):
        layout, _ = loaded.plan(context)
        assert len(layout) == 4 and {row["profileId"] for row in layout} == {"thermal"}
    unavailable = deepcopy(context)
    unavailable["publicSnapshot"]["available_sensor_ids"].remove("thermal")
    with pytest.raises(ValueError, match="unavailable"):
        loaded.plan(unavailable)
    with pytest.raises(ValueError, match="unavailable"):
        LocalRefinementPPOPolicy.load(checkpoint, unavailable)


@pytest.mark.parametrize("change", ["count", "duplicate", "blocked", "angle", "budget", "radius"])
def test_illegal_starting_layout_fails_before_training(change):
    context, base = context_and_base()
    if change == "count":
        base.pop()
    elif change == "duplicate":
        base[0] = deepcopy(base[1])
    elif change == "blocked":
        base[0]["siteId"] = 0
    elif change == "angle":
        base[0]["yawDeg"] = 17.
    elif change == "budget":
        context["publicSnapshot"].update(budget_total=3., budget_remaining=3.)
    elif change == "radius":
        context["publicSnapshot"]["deployment_max_radius"] = 40.
    with pytest.raises(ValueError):
        policy = LocalRefinementPPOPolicy(context, sensor_count=4)
        policy.initialize_from_placements(base)


def test_uninitialized_calls_and_incomplete_or_cross_slot_records_fail(tmp_path):
    context, _ = context_and_base()
    policy = TRAINING_ALGORITHMS["local_ppo"]["factory"](context, sensor_count=4)
    assert isinstance(policy, LocalRefinementPPOPolicy)
    with pytest.raises(ValueError, match="Initialize"):
        policy.plan(context)
    with pytest.raises(ValueError, match="Initialize"):
        policy.update([])
    with pytest.raises(ValueError, match="Initialize"):
        policy.save(tmp_path / "uninitialized.json")
    context, _, policy = initialized()
    _, records = policy.plan(context)
    with pytest.raises(ValueError, match="one recorded decision"):
        policy.update([(records[:-1], 1.)])
    records[0]["mask"][policy._slot_offsets[1]] = True
    with pytest.raises(ValueError, match="only to its sensor slot"):
        policy.update([(records, 1.)])


def test_local_registration_does_not_change_legacy_ppo_checkpoint(tmp_path):
    context, base = context_and_base()
    legacy = MaskedPPOPolicy(context, sensor_count=4, allowed_sensor_ids=["thermal"])
    report = legacy.initialize_from_placements(base, baseline_action_probability=.8)
    assert report["initial_baseline_action_probability"] == pytest.approx(.8)
    assert legacy.plan(context, deterministic=True)[0] == base
    checkpoint = tmp_path / "legacy.json"
    legacy.save(checkpoint)
    loaded = MaskedPPOPolicy.load(checkpoint, context)
    assert loaded.schema == "istana.warning_directional_masked_ppo.v1"
    assert loaded.plan(context, deterministic=True)[0] == base
    np.testing.assert_array_equal(loaded.option_logits, legacy.option_logits)
    with pytest.raises(ValueError, match="schema"):
        LocalRefinementPPOPolicy.load(checkpoint, context)


def test_local_legality_respects_canonical_empty_layout_distance_limit():
    context, _ = context_and_base()
    context["publicSnapshot"].update(sites=[[-400., 0.], [400., 0.]], max_sites=2,
        budget_total=2., budget_remaining=2., blocked_sites=[], deployment_max_radius=500.,
        min_separation=400.)
    base = [{"profileId": "thermal", "siteId": site, "yawDeg": 0., "pitchDeg": 10.}
            for site in (0, 1)]
    state, catalogue, _ = public_planning_inputs(context)
    assert not build_observation(state, catalogue)["action_mask"][:-1].any()
    policy = LocalRefinementPPOPolicy(context, sensor_count=2)
    with pytest.raises(ValueError, match="not legal"):
        policy.initialize_from_placements(base)


def test_forced_keep_has_finite_zero_entropy_and_never_removes_a_sensor():
    context, _ = context_and_base()
    context["publicSnapshot"].update(sites=[[45., 0.]], max_sites=1,
        budget_total=1., budget_remaining=1., blocked_sites=[])
    context["catalogue"][0].update(yaw_bins_deg=[0.], pitch_bins_deg=[10.])
    base = [{"profileId": "thermal", "siteId": 0, "yawDeg": 0., "pitchDeg": 10.}]
    policy = LocalRefinementPPOPolicy(context, sensor_count=1)
    report = policy.initialize_from_placements(base)
    assert report["initial_full_layout_probability"] == 1.
    episodes = []
    for reward in (-1., 0., 2.):
        layout, records = policy.plan(context)
        assert layout == base and len(records) == 1
        assert records[0]["old_probability"] == 1.
        episodes.append((records, reward))
    update = policy.update(episodes)
    assert update["entropy"] == 0. and update["parameter_delta_norm"] == 0.
    assert np.isfinite(update["objective_loss"])
    assert policy.plan(context, deterministic=True)[0] == base
